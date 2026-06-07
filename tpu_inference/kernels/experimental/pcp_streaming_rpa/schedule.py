# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side schedules for PCP streaming prefill RPA."""

import dataclasses

import numpy as np

from tpu_inference.layers.common.pcp_layout import pcp_query_chunk_ranges


class ScheduleField:
    REQ_ID = 0
    KV_PAGE_RANK = 1
    KV_PAGE_IDX = 2
    IS_FIRST_KV = 3
    IS_LAST_KV = 4
    LOAD_Q = 5
    Q_GLOBAL_START = 6
    KV_GLOBAL_START = 7
    KV_VALID_LEN = 8
    Q_HBM_OFFSET = 9
    Q_TILE_SIZE = 10
    O_HBM_OFFSET = 11
    NUM_FIELDS = 12
    PACKED_NUM_FIELDS = 128
    KV_PAGE_INDICES_START = NUM_FIELDS
    MAX_KV_PAGES_PER_BLOCK = PACKED_NUM_FIELDS - KV_PAGE_INDICES_START


_PACKED_FIELD_NAMES = (
    "req_id",
    "kv_page_rank",
    "kv_page_idx",
    "is_first_kv",
    "is_last_kv",
    "load_q",
    "q_global_start",
    "kv_global_start",
    "kv_valid_len",
    "q_hbm_offset",
    "q_tile_size",
    "o_hbm_offset",
)


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _q_global_last_for_strided_tile(q_global_start: int, q_tile_len: int, *,
                                    pcp_size: int,
                                    interleave_size: int) -> int:
    last_row = q_tile_len - 1
    return (q_global_start +
            (last_row // interleave_size) * pcp_size * interleave_size +
            (last_row % interleave_size))


def _iter_pcp_q_tiles(q_len: int, q_global_base: int, consumer_rank: int, *,
                      pcp_size: int, interleave_size: int, bq_sz: int):
    """Yield local-contiguous Q tiles for one rank.

    The original schedule emitted one tile per PCP interleave chunk. For small
    interleave sizes that forces tiny Q tiles and repeats the same streamed KV
    traffic. When adjacent rank-owned chunks are full interleave chunks, they
    are contiguous in local HBM and can be consumed as one larger tile. The
    kernel reconstructs each row's strided global Q position from the first
    chunk's global start.
    """
    ranges = list(
        pcp_query_chunk_ranges(q_len, q_global_base, consumer_rank, pcp_size,
                               interleave_size))
    if bq_sz <= interleave_size:
        for chunk_start, chunk_end in ranges:
            chunk_len = chunk_end - chunk_start
            for tile_start in range(0, chunk_len, bq_sz):
                tile_len = min(bq_sz, chunk_len - tile_start)
                yield chunk_start + tile_start, tile_len
        return

    range_idx = 0
    while range_idx < len(ranges):
        chunk_start, chunk_end = ranges[range_idx]
        chunk_len = chunk_end - chunk_start
        if chunk_len != interleave_size:
            for tile_start in range(0, chunk_len, bq_sz):
                tile_len = min(bq_sz, chunk_len - tile_start)
                yield chunk_start + tile_start, tile_len
            range_idx += 1
            continue

        tile_global_start = chunk_start
        tile_len = 0
        expected_chunk_start = chunk_start
        while range_idx < len(ranges):
            next_start, next_end = ranges[range_idx]
            next_len = next_end - next_start
            if (next_len != interleave_size
                    or next_start != expected_chunk_start
                    or tile_len + next_len > bq_sz):
                break
            tile_len += next_len
            range_idx += 1
            expected_chunk_start += pcp_size * interleave_size

        yield tile_global_start, tile_len


@dataclasses.dataclass(frozen=True)
class PcpStreamingSchedule:
    """PCP streaming schedule arrays.

    All per-step fields have shape [pcp_size, max_steps, num_lanes]. The first
    dimension is the consumer rank. All PCP ranks receive the same replicated
    schedule so source ranks can push their local KV pages to any consumer.
    """

    req_id: np.ndarray
    kv_page_rank: np.ndarray
    kv_page_idx: np.ndarray
    is_first_kv: np.ndarray
    is_last_kv: np.ndarray
    load_q: np.ndarray
    q_global_start: np.ndarray
    kv_global_start: np.ndarray
    kv_valid_len: np.ndarray
    q_hbm_offset: np.ndarray
    q_tile_size: np.ndarray
    o_hbm_offset: np.ndarray
    packed_schedule: np.ndarray
    actual_steps: np.ndarray
    global_actual_steps: np.ndarray
    kv_page_indices: np.ndarray | None = None

    @property
    def pcp_size(self) -> int:
        return self.req_id.shape[0]

    @property
    def max_steps(self) -> int:
        return self.req_id.shape[1]

    @property
    def num_lanes(self) -> int:
        return self.req_id.shape[2]


@dataclasses.dataclass(frozen=True)
class _Entry:
    req_id: int
    kv_page_rank: int
    kv_page_idx: int
    is_first_kv: int
    is_last_kv: int
    load_q: int
    q_global_start: int
    kv_global_start: int
    kv_valid_len: int
    q_hbm_offset: int
    q_tile_size: int
    o_hbm_offset: int
    kv_page_indices: tuple[int, ...] | None = None


def _validate_inputs(
    kv_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    q_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    kv_pages_per_block: int,
) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")
    if num_lanes <= 0:
        raise ValueError("num_lanes must be positive.")
    if bq_sz <= 0:
        raise ValueError("bq_sz must be positive.")
    if kv_pages_per_block <= 0:
        raise ValueError("kv_pages_per_block must be positive.")
    if kv_pages_per_block > ScheduleField.MAX_KV_PAGES_PER_BLOCK:
        raise ValueError(
            "kv_pages_per_block exceeds packed schedule capacity: "
            f"{kv_pages_per_block} > "
            f"{ScheduleField.MAX_KV_PAGES_PER_BLOCK}.")
    if page_size != interleave_size:
        raise NotImplementedError(
            "PCP streaming schedule currently requires "
            f"page_size == interleave_size, got {page_size} != "
            f"{interleave_size}.")
    if cu_q_lens.ndim != 1 or cu_q_lens.size == 0:
        raise ValueError("cu_q_lens must be a non-empty 1D array.")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D array.")

    num_reqs = int(cu_q_lens.size - 1)
    if kv_lens.size < num_reqs:
        raise ValueError("kv_lens must cover every request in cu_q_lens.")
    if q_start_offsets.size < num_reqs:
        raise ValueError("q_start_offsets must cover every request.")
    if block_tables.shape[0] < num_reqs:
        raise ValueError("block_tables must cover every request.")
    return num_reqs


def pack_pcp_streaming_schedule_fields(
    *,
    req_id: np.ndarray,
    kv_page_rank: np.ndarray,
    kv_page_idx: np.ndarray,
    is_first_kv: np.ndarray,
    is_last_kv: np.ndarray,
    load_q: np.ndarray,
    q_global_start: np.ndarray,
    kv_global_start: np.ndarray,
    kv_valid_len: np.ndarray,
    q_hbm_offset: np.ndarray,
    q_tile_size: np.ndarray,
    o_hbm_offset: np.ndarray,
) -> np.ndarray:
    """Pack schedule fields into [max_steps, pcp_size, lanes, padded_fields]."""
    field_arrays = (
        req_id,
        kv_page_rank,
        kv_page_idx,
        is_first_kv,
        is_last_kv,
        load_q,
        q_global_start,
        kv_global_start,
        kv_valid_len,
        q_hbm_offset,
        q_tile_size,
        o_hbm_offset,
    )
    if len({array.shape for array in field_arrays}) != 1:
        raise ValueError("all schedule fields must have identical shapes.")
    logical = np.stack(field_arrays, axis=-1).astype(np.int32, copy=False)
    padded_shape = logical.shape[:-1] + (ScheduleField.PACKED_NUM_FIELDS, )
    packed = np.zeros(padded_shape, dtype=np.int32)
    packed[..., :ScheduleField.NUM_FIELDS] = logical
    return np.transpose(packed, (1, 0, 2, 3)).copy()


def unpack_pcp_streaming_schedule_field(
    packed_schedule: np.ndarray,
    field: int,
) -> np.ndarray:
    """Unpack one field to [pcp_size, max_steps, num_lanes]."""
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must be rank 4.")
    if field < 0 or field >= ScheduleField.NUM_FIELDS:
        raise ValueError(f"invalid schedule field index: {field}")
    return np.transpose(packed_schedule[..., field], (1, 0, 2)).copy()


def generate_pcp_streaming_schedule(
    kv_lens: list[int] | np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    q_start_offsets: list[int] | np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool = False,
    pad_steps_to: int | None = None,
    kv_pages_per_block: int = 1,
) -> PcpStreamingSchedule:
    """Generate a replicated PCP streaming schedule for page-aligned PCP.

    Q ownership follows PCP interleave chunks. KV page ownership follows the
    fast path where page_size == interleave_size, so each global KV page belongs
    to exactly one PCP rank. When pad_kv_pages_to_pcp_group is true, each Q
    tile's KV pages are padded with no-op entries to a multiple of pcp_size so
    the schedule can drive ring-grouped kernels.

    kv_pages_per_block groups consecutive local pages from the same source rank
    into one ring step. The grouped pages are strided in global token order by
    pcp_size * page_size, and their physical page ids are stored in the padded
    schedule fields starting at ScheduleField.KV_PAGE_INDICES_START.
    """
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    cu_q_lens = np.asarray(cu_q_lens, dtype=np.int64)
    q_start_offsets = np.asarray(q_start_offsets, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    num_reqs = _validate_inputs(
        kv_lens,
        cu_q_lens,
        q_start_offsets,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        num_lanes,
        bq_sz,
        kv_pages_per_block,
    )
    kv_pages_per_block = int(kv_pages_per_block)

    schedules: list[list[list[_Entry]]] = []
    actual_steps = np.zeros(pcp_size, dtype=np.int32)
    for consumer_rank in range(pcp_size):
        lane_entries: list[list[_Entry]] = [[] for _ in range(num_lanes)]
        lane_lengths = np.zeros(num_lanes, dtype=np.int64)
        rank_q_offset = 0

        for req_idx in range(num_reqs):
            q_len = int(cu_q_lens[req_idx + 1] - cu_q_lens[req_idx])
            if q_len <= 0:
                continue
            q_global_base = int(q_start_offsets[req_idx])
            kv_len = int(kv_lens[req_idx])
            if kv_len < q_global_base + q_len:
                raise ValueError("kv_lens must include all scheduled Q tokens.")
            num_kv_pages = _cdiv(kv_len, page_size)

            for q_global, tile_len in _iter_pcp_q_tiles(
                    q_len,
                    q_global_base,
                    consumer_rank,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                    bq_sz=bq_sz):
                q_global_last = _q_global_last_for_strided_tile(
                    q_global,
                    tile_len,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size)
                q_hbm_offset = rank_q_offset
                target_lane = int(np.argmin(lane_lengths))
                effective_kv_pages = min(num_kv_pages,
                                         q_global_last // page_size + 1)

                if kv_pages_per_block == 1:
                    scheduled_kv_pages = effective_kv_pages
                    if pad_kv_pages_to_pcp_group:
                        scheduled_kv_pages = _cdiv(effective_kv_pages,
                                                   pcp_size) * pcp_size
                    last_block = -1
                    last_store_src_rank = -1
                else:
                    block_span_pages = pcp_size * kv_pages_per_block
                    scheduled_kv_blocks = _cdiv(effective_kv_pages,
                                                 block_span_pages)
                    scheduled_kv_pages = scheduled_kv_blocks * pcp_size
                    last_global_page = effective_kv_pages - 1
                    last_local_page = last_global_page // pcp_size
                    last_block = last_local_page // kv_pages_per_block
                    last_store_src_rank = 0
                    last_local_page_start = last_block * kv_pages_per_block
                    for candidate_src_rank in range(pcp_size):
                        for page_offset in range(kv_pages_per_block):
                            candidate_global_page = (
                                (last_local_page_start + page_offset) *
                                pcp_size + candidate_src_rank)
                            if candidate_global_page < effective_kv_pages:
                                last_store_src_rank = candidate_src_rank

                for kv_page_seq_idx in range(scheduled_kv_pages):
                    if kv_pages_per_block == 1:
                        src_rank = kv_page_seq_idx % pcp_size
                        local_page_start = kv_page_seq_idx // pcp_size
                    else:
                        src_rank = kv_page_seq_idx % pcp_size
                        local_page_start = (kv_page_seq_idx // pcp_size *
                                            kv_pages_per_block)
                    global_page = local_page_start * pcp_size + src_rank
                    global_token_start = global_page * page_size
                    valid_tokens = 0
                    page_indices = []
                    for page_offset in range(kv_pages_per_block):
                        page_global = ((local_page_start + page_offset) *
                                       pcp_size + src_rank)
                        if page_global < effective_kv_pages:
                            local_page_index = page_global // pcp_size
                            if local_page_index >= block_tables.shape[1]:
                                raise ValueError(
                                    "block_tables does not cover requested "
                                    "KV page.")
                            physical_page = int(block_tables[req_idx,
                                                             local_page_index])
                            page_valid = min(
                                page_size,
                                kv_len - page_global * page_size,
                            )
                            valid_tokens += max(page_valid, 0)
                        else:
                            physical_page = 0
                        page_indices.append(physical_page)
                    if valid_tokens == 0:
                        lane_entries[target_lane].append(
                            _Entry(
                                req_id=-1,
                                kv_page_rank=src_rank,
                                kv_page_idx=0,
                                is_first_kv=0,
                                is_last_kv=0,
                                load_q=0,
                                q_global_start=q_global,
                                kv_global_start=global_token_start,
                                kv_valid_len=0,
                                q_hbm_offset=q_hbm_offset,
                                q_tile_size=tile_len,
                                o_hbm_offset=q_hbm_offset,
                                kv_page_indices=tuple(page_indices),
                            ))
                        lane_lengths[target_lane] += 1
                        continue
                    if kv_pages_per_block == 1:
                        is_first = int(kv_page_seq_idx == 0)
                        is_last = int(kv_page_seq_idx == effective_kv_pages - 1)
                    else:
                        cur_block = kv_page_seq_idx // pcp_size
                        is_first = int(cur_block == 0 and src_rank == 0)
                        is_last = int(cur_block == last_block
                                      and src_rank == last_store_src_rank)
                    lane_entries[target_lane].append(
                        _Entry(
                            req_id=req_idx,
                            kv_page_rank=src_rank,
                            kv_page_idx=page_indices[0],
                            is_first_kv=is_first,
                            is_last_kv=is_last,
                            load_q=is_first,
                            q_global_start=q_global,
                            kv_global_start=global_token_start,
                            kv_valid_len=valid_tokens,
                            q_hbm_offset=q_hbm_offset,
                            q_tile_size=tile_len,
                            o_hbm_offset=q_hbm_offset,
                            kv_page_indices=tuple(page_indices),
                        ))
                    lane_lengths[target_lane] += 1

                rank_q_offset += tile_len

        actual_steps[consumer_rank] = int(lane_lengths.max(initial=0))
        schedules.append(lane_entries)

    actual_max_steps = int(actual_steps.max(initial=0))
    max_steps = actual_max_steps
    if pad_steps_to is not None:
        pad_steps_to = int(pad_steps_to)
        if pad_steps_to < actual_max_steps:
            raise ValueError("pad_steps_to must be >= generated schedule steps: "
                             f"{pad_steps_to=} {actual_max_steps=}.")
        max_steps = pad_steps_to
    shape = (pcp_size, max_steps, num_lanes)

    req_id = np.full(shape, -1, dtype=np.int32)
    kv_page_indices = None
    if kv_pages_per_block > 1:
        kv_page_indices = np.zeros(shape + (kv_pages_per_block, ),
                                   dtype=np.int32)
    fields = {
        "kv_page_rank": np.full(shape, -1, dtype=np.int32),
        "kv_page_idx": np.full(shape, -1, dtype=np.int32),
        "is_first_kv": np.zeros(shape, dtype=np.int32),
        "is_last_kv": np.zeros(shape, dtype=np.int32),
        "load_q": np.zeros(shape, dtype=np.int32),
        "q_global_start": np.zeros(shape, dtype=np.int32),
        "kv_global_start": np.zeros(shape, dtype=np.int32),
        "kv_valid_len": np.zeros(shape, dtype=np.int32),
        "q_hbm_offset": np.zeros(shape, dtype=np.int32),
        "q_tile_size": np.zeros(shape, dtype=np.int32),
        "o_hbm_offset": np.zeros(shape, dtype=np.int32),
    }

    for consumer_rank, lane_entries in enumerate(schedules):
        for lane, entries in enumerate(lane_entries):
            for step, entry in enumerate(entries):
                req_id[consumer_rank, step, lane] = entry.req_id
                fields["kv_page_rank"][consumer_rank, step,
                                       lane] = entry.kv_page_rank
                fields["kv_page_idx"][consumer_rank, step,
                                      lane] = entry.kv_page_idx
                fields["is_first_kv"][consumer_rank, step,
                                      lane] = entry.is_first_kv
                fields["is_last_kv"][consumer_rank, step,
                                     lane] = entry.is_last_kv
                fields["load_q"][consumer_rank, step, lane] = entry.load_q
                fields["q_global_start"][consumer_rank, step,
                                         lane] = entry.q_global_start
                fields["kv_global_start"][consumer_rank, step,
                                          lane] = entry.kv_global_start
                fields["kv_valid_len"][consumer_rank, step,
                                       lane] = entry.kv_valid_len
                fields["q_hbm_offset"][consumer_rank, step,
                                       lane] = entry.q_hbm_offset
                fields["q_tile_size"][consumer_rank, step,
                                      lane] = entry.q_tile_size
                fields["o_hbm_offset"][consumer_rank, step,
                                       lane] = entry.o_hbm_offset
                if kv_page_indices is not None:
                    # The first page id is also present in kv_page_idx for
                    # compatibility with the single-page schedule fields.
                    page_indices = entry.kv_page_indices
                    if page_indices is None:
                        page_indices = (entry.kv_page_idx, )
                    kv_page_indices[consumer_rank, step, lane, :len(
                        page_indices)] = page_indices

    packed_schedule = pack_pcp_streaming_schedule_fields(req_id=req_id,
                                                         **fields)
    if kv_page_indices is not None:
        packed_schedule[
            ...,
            ScheduleField.KV_PAGE_INDICES_START:
            ScheduleField.KV_PAGE_INDICES_START + kv_pages_per_block,
        ] = np.transpose(kv_page_indices, (1, 0, 2, 3))

    return PcpStreamingSchedule(
        req_id=req_id,
        actual_steps=actual_steps,
        global_actual_steps=np.array([actual_max_steps], dtype=np.int32),
        packed_schedule=packed_schedule,
        kv_page_indices=kv_page_indices,
        **fields,
    )


def validate_pcp_streaming_schedule(schedule: PcpStreamingSchedule) -> None:
    """Validate per-lane online softmax state invariants."""
    for consumer_rank in range(schedule.pcp_size):
        for lane in range(schedule.num_lanes):
            in_tile = False
            cur_req_id = -1
            cur_q_offset = -1
            prev_kv_start = -1
            saw_last = False
            for step in range(int(schedule.actual_steps[consumer_rank])):
                req_id = int(schedule.req_id[consumer_rank, step, lane])
                if req_id == -1:
                    continue
                is_first = bool(schedule.is_first_kv[consumer_rank, step,
                                                     lane])
                is_last = bool(schedule.is_last_kv[consumer_rank, step, lane])
                if is_first:
                    if in_tile:
                        raise ValueError(
                            "new q_tile before previous is_last_kv.")
                    in_tile = True
                    saw_last = False
                    cur_req_id = req_id
                    cur_q_offset = int(
                        schedule.q_hbm_offset[consumer_rank, step, lane])
                    prev_kv_start = -1
                if not in_tile:
                    raise ValueError("entry outside q_tile boundary.")
                if req_id != cur_req_id:
                    raise ValueError("req_id changed inside q_tile.")
                if int(schedule.q_hbm_offset[consumer_rank, step,
                                             lane]) != cur_q_offset:
                    raise ValueError("q_hbm_offset changed inside q_tile.")
                kv_start = int(schedule.kv_global_start[consumer_rank, step,
                                                        lane])
                if kv_start <= prev_kv_start:
                    raise ValueError(
                        "kv_global_start must increase inside q_tile.")
                prev_kv_start = kv_start
                if is_last:
                    in_tile = False
                    saw_last = True
            if in_tile or not saw_last and schedule.actual_steps[
                    consumer_rank] > 0 and np.any(
                        schedule.req_id[consumer_rank, :, lane] != -1):
                raise ValueError("q_tile not closed at end of schedule.")
