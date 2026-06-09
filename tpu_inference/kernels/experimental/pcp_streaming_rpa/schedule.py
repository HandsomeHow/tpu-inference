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


def build_pcp_streaming_active_page_groups(
    schedule: PcpStreamingSchedule,
) -> np.ndarray:
    """Build kernel active-page-group metadata from a generated schedule."""
    global_actual_steps = np.asarray(schedule.global_actual_steps)
    if global_actual_steps.ndim != 1 or global_actual_steps.size != 1:
        raise ValueError(
            "PCP streaming schedule must have one global_actual_steps value.")
    actual_steps = int(global_actual_steps[0])
    if actual_steps % schedule.pcp_size != 0:
        raise ValueError("PCP streaming schedule steps must be padded to a "
                         "PCP page group.")
    return np.array([actual_steps // schedule.pcp_size], dtype=np.int32)


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


def _single_active_aligned_request_params(
    *,
    q_lens: np.ndarray,
    q_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    pcp_size: int,
    interleave_size: int,
    path_name: str,
    q_start_name: str,
) -> tuple[int, int]:
    if q_lens.ndim != 1:
        raise ValueError("q_lens must be a 1D array.")
    if q_start_offsets.ndim != 1:
        raise ValueError("q_start_offsets must be a 1D array.")
    if q_start_offsets.size < q_lens.size:
        raise ValueError("q_start_offsets must cover every request.")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D array.")
    if block_tables.shape[0] < 1 or block_tables.shape[1] == 0:
        raise ValueError("block_tables must cover the active request.")

    active = np.flatnonzero(q_lens > 0)
    if active.size != 1 or int(active[0]) != 0:
        raise NotImplementedError(
            f"{path_name} only supports one active request at index 0, got "
            f"active request indices {active.tolist()}.")

    q_len = int(q_lens[0])
    q_start = int(q_start_offsets[0])
    cycle = pcp_size * interleave_size
    if q_len <= 0:
        raise NotImplementedError(f"{path_name} requires a positive q_len.")
    if q_start < 0:
        raise ValueError(f"{q_start_name} must be non-negative.")
    if q_len % cycle != 0:
        raise NotImplementedError(
            f"{path_name} requires q_len to be a multiple of "
            f"pcp_size * interleave_size, got {q_len=} {pcp_size=} "
            f"{interleave_size=}.")
    if q_start % cycle != 0:
        raise NotImplementedError(
            f"{path_name} requires {q_start_name} to be aligned to "
            "pcp_size * interleave_size, got "
            f"{q_start=} {pcp_size=} {interleave_size=}.")
    return q_len, q_start


def _single_aligned_request_params(
    *,
    kv_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    q_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool,
) -> tuple[int, int]:
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    if num_lanes != 1:
        raise NotImplementedError(
            "PCP streaming vectorized schedule only supports num_lanes == 1, "
            f"got {num_lanes}.")
    if not pad_kv_pages_to_pcp_group:
        raise NotImplementedError("PCP streaming vectorized schedule requires "
                                  "pad_kv_pages_to_pcp_group=True.")
    if page_size != interleave_size:
        raise NotImplementedError(
            "PCP streaming vectorized schedule requires "
            f"page_size == interleave_size, got {page_size} != "
            f"{interleave_size}.")
    if bq_sz % interleave_size != 0:
        raise NotImplementedError(
            "PCP streaming vectorized schedule requires bq_sz to be a "
            f"multiple of interleave_size, got {bq_sz=} "
            f"{interleave_size=}.")

    q_len, q_start = _single_active_aligned_request_params(
        q_lens=q_lens,
        q_start_offsets=q_start_offsets,
        block_tables=block_tables,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        path_name="PCP streaming vectorized schedule",
        q_start_name="q_start_offset",
    )
    kv_len = int(kv_lens[0])
    if kv_len < q_start + q_len:
        raise ValueError("kv_lens must include all scheduled Q tokens.")
    return q_len, q_start


def validate_pcp_streaming_local_padded_tokens(
    *,
    padded_num_tokens: int,
    pcp_size: int,
    q_block_size: int,
) -> None:
    local_padded_num_tokens = padded_num_tokens // pcp_size
    if local_padded_num_tokens % q_block_size != 0:
        raise ValueError(
            "PCP streaming schedule requires local padded tokens to be a "
            "multiple of q_block_size: got "
            f"{local_padded_num_tokens=} and {q_block_size=}.")


def estimate_pcp_streaming_schedule_steps_ub(
    q_lens: np.ndarray,
    capacity_tokens: int,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    q_block_size: int,
    kv_pages_per_block: int = 1,
) -> int:
    max_global_pages = _cdiv(capacity_tokens, block_size)
    kv_pages_per_block = max(1, int(kv_pages_per_block))
    max_steps = 0
    for consumer_rank in range(pcp_size):
        lane_lengths = np.zeros(num_lanes, dtype=np.int64)
        for q_len in q_lens:
            q_len = int(q_len)
            if q_len <= 0:
                continue
            q_global_base = max(0, capacity_tokens - q_len)
            for q_global, tile_len in _iter_pcp_q_tiles(
                    q_len,
                    q_global_base,
                    consumer_rank,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                    bq_sz=q_block_size):
                q_global_last = _q_global_last_for_strided_tile(
                    q_global,
                    tile_len,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size)
                effective_pages = min(max_global_pages,
                                      q_global_last // block_size + 1)
                scheduled_pages = (
                    _cdiv(effective_pages, pcp_size * kv_pages_per_block) *
                    pcp_size)
                target_lane = int(np.argmin(lane_lengths))
                lane_lengths[target_lane] += scheduled_pages
        max_steps = max(max_steps, int(lane_lengths.max(initial=0)))
    return max_steps


def build_pcp_streaming_local_slot_ids(
    q_lens: np.ndarray,
    seq_lens: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
) -> np.ndarray:
    """Build rank-major local cache slot ids for the supported PCP fast path."""
    if block_size <= 0:
        raise ValueError(f"Expected positive block_size, got {block_size}.")
    if pcp_size <= 0:
        raise ValueError(f"Expected positive pcp_size, got {pcp_size}.")
    if interleave_size <= 0:
        raise ValueError(
            f"Expected positive interleave_size, got {interleave_size}.")
    if padded_num_tokens % pcp_size != 0:
        raise ValueError(
            f"{padded_num_tokens=} must be divisible by {pcp_size=}.")
    if block_size != interleave_size:
        raise NotImplementedError(
            "PCP local slot id vectorized path only supports "
            f"block_size == interleave_size, got {block_size=} "
            f"{interleave_size=}.")

    q_lens = np.asarray(q_lens, dtype=np.int64)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be a 1D array.")
    if q_lens.size != seq_lens.size:
        raise ValueError("q_lens and seq_lens must have the same size.")
    if np.any(seq_lens < q_lens):
        raise ValueError("seq_lens_per_req must be >= "
                         "num_scheduled_tokens_per_req.")

    q_start_offsets = seq_lens - q_lens
    q_len, q_global_base = _single_active_aligned_request_params(
        q_lens=q_lens,
        q_start_offsets=q_start_offsets,
        block_tables=block_tables,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        path_name="PCP local slot id vectorized path",
        q_start_name="q_start",
    )
    if padded_num_tokens != q_len:
        raise NotImplementedError(
            "PCP local slot id vectorized path only supports "
            f"padded_num_tokens == q_len, got {padded_num_tokens=} "
            f"{q_len=}.")

    cycle = pcp_size * interleave_size
    local_padded_num_tokens = padded_num_tokens // pcp_size
    num_cycles = q_len // cycle
    if local_padded_num_tokens != num_cycles * interleave_size:
        raise ValueError("PCP local slot count does not match padded shape.")

    rank_offsets = (np.arange(pcp_size, dtype=np.int64)[:, None, None] *
                    interleave_size)
    cycle_offsets = (np.arange(num_cycles, dtype=np.int64)[None, :, None] *
                     cycle)
    within_chunk = np.arange(interleave_size, dtype=np.int64)[None, None, :]
    positions = (q_global_base + rank_offsets + cycle_offsets +
                 within_chunk).reshape(pcp_size, local_padded_num_tokens)

    virtual_block_size = block_size * pcp_size
    block_indices = positions // virtual_block_size
    max_block_index = int(block_indices.max(initial=-1))
    if max_block_index >= block_tables.shape[1]:
        raise ValueError("block_tables does not cover requested PCP slots.")

    virtual_offsets = positions - block_indices * virtual_block_size
    local_offsets = ((virtual_offsets // cycle) * interleave_size +
                     (virtual_offsets % interleave_size))
    block_numbers = block_tables[0, block_indices].astype(np.int64)
    slot_ids = (block_numbers * block_size + local_offsets).astype(np.int32)
    return slot_ids.reshape(-1)


def _last_group_info(effective_kv_pages: int, pcp_size: int,
                     kv_pages_per_block: int) -> tuple[int, int]:
    last_global_page = effective_kv_pages - 1
    last_local_page = last_global_page // pcp_size
    last_block = last_local_page // kv_pages_per_block
    last_local_page_start = last_block * kv_pages_per_block
    last_store_src_rank = 0
    for candidate_src_rank in range(pcp_size):
        for page_offset in range(kv_pages_per_block):
            candidate_global_page = (
                (last_local_page_start + page_offset) * pcp_size +
                candidate_src_rank)
            if candidate_global_page < effective_kv_pages:
                last_store_src_rank = candidate_src_rank
    return last_block, last_store_src_rank


def _fill_pcp_streaming_schedule_rows_vectorized(
    packed_schedule: np.ndarray,
    *,
    start_step: int,
    num_steps: int,
    consumer_rank: int,
    lane: int,
    req_id: int,
    q_global_start: int,
    q_tile_size: int,
    q_hbm_offset: int,
    kv_len: int,
    effective_kv_pages: int,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    kv_pages_per_block: int,
) -> None:
    if num_steps <= 0:
        return

    kv_page_seq_idx = np.arange(num_steps, dtype=np.int32)
    src_rank = kv_page_seq_idx % pcp_size
    local_page_start = (kv_page_seq_idx // pcp_size) * kv_pages_per_block
    global_page = local_page_start * pcp_size + src_rank

    page_offsets = np.arange(kv_pages_per_block, dtype=np.int32)
    page_global = (
        (local_page_start[:, None] + page_offsets[None, :]) * pcp_size +
        src_rank[:, None])
    valid_pages = page_global < effective_kv_pages
    local_page_idx = page_global // pcp_size
    if np.any(valid_pages) and int(
            local_page_idx[valid_pages].max()) >= block_tables.shape[1]:
        raise ValueError("block_tables does not cover requested KV page.")
    safe_page_idx = np.minimum(local_page_idx, block_tables.shape[1] - 1)
    page_indices = np.where(valid_pages, block_tables[req_id, safe_page_idx],
                            0).astype(np.int32)

    page_valid = np.minimum(page_size, kv_len - page_global * page_size)
    kv_valid_len = np.where(valid_pages, np.maximum(page_valid, 0),
                            0).sum(axis=1).astype(np.int32)

    rows = packed_schedule[start_step:start_step + num_steps, consumer_rank,
                           lane, :]
    rows.fill(0)
    rows[:, ScheduleField.REQ_ID] = np.where(kv_valid_len > 0, req_id, -1)
    rows[:, ScheduleField.KV_PAGE_RANK] = src_rank
    rows[:, ScheduleField.KV_PAGE_IDX] = np.where(kv_valid_len > 0,
                                                  page_indices[:, 0], 0)

    last_block, last_store_src_rank = _last_group_info(effective_kv_pages,
                                                       pcp_size,
                                                       kv_pages_per_block)
    cur_block = kv_page_seq_idx // pcp_size
    valid = kv_valid_len > 0
    is_first = valid & (cur_block == 0) & (src_rank == 0)
    is_last = valid & (cur_block == last_block) & (src_rank
                                                   == last_store_src_rank)
    rows[:, ScheduleField.IS_FIRST_KV] = is_first.astype(np.int32)
    rows[:, ScheduleField.IS_LAST_KV] = is_last.astype(np.int32)
    rows[:, ScheduleField.LOAD_Q] = is_first.astype(np.int32)
    rows[:, ScheduleField.Q_GLOBAL_START] = q_global_start
    rows[:, ScheduleField.KV_GLOBAL_START] = global_page * page_size
    rows[:, ScheduleField.KV_VALID_LEN] = kv_valid_len
    rows[:, ScheduleField.Q_HBM_OFFSET] = q_hbm_offset
    rows[:, ScheduleField.Q_TILE_SIZE] = q_tile_size
    rows[:, ScheduleField.O_HBM_OFFSET] = q_hbm_offset
    if kv_pages_per_block > 1:
        rows[
            :,
            ScheduleField.KV_PAGE_INDICES_START:
            ScheduleField.KV_PAGE_INDICES_START + kv_pages_per_block,
        ] = page_indices


def _packed_field(packed_schedule: np.ndarray, field: int) -> np.ndarray:
    return np.transpose(packed_schedule[..., field], (1, 0, 2)).copy()


def _generate_pcp_streaming_schedule_vectorized_aligned(
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
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    cu_q_lens = np.asarray(cu_q_lens, dtype=np.int64)
    q_start_offsets = np.asarray(q_start_offsets, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    _validate_inputs(
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
    q_len, q_global_base = _single_aligned_request_params(
        kv_lens=kv_lens,
        cu_q_lens=cu_q_lens,
        q_start_offsets=q_start_offsets,
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=bq_sz,
        pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
    )

    kv_len = int(kv_lens[0])
    num_kv_pages = _cdiv(kv_len, page_size)
    actual_steps = np.zeros(pcp_size, dtype=np.int32)
    tile_plans: list[list[tuple[int, int, int, int, int, int]]] = []
    actual_max_steps = 0
    for consumer_rank in range(pcp_size):
        rank_q_offset = 0
        rank_plans = []
        rank_steps = 0
        for q_global, q_tile_size in _iter_pcp_q_tiles(
                q_len,
                q_global_base,
                consumer_rank,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                bq_sz=bq_sz):
            q_global_last = _q_global_last_for_strided_tile(
                q_global,
                q_tile_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size)
            effective_kv_pages = min(num_kv_pages,
                                     q_global_last // page_size + 1)
            if kv_pages_per_block == 1:
                scheduled_steps = effective_kv_pages
                if pad_kv_pages_to_pcp_group:
                    scheduled_steps = _cdiv(effective_kv_pages,
                                            pcp_size) * pcp_size
            else:
                scheduled_steps = (
                    _cdiv(effective_kv_pages, pcp_size * kv_pages_per_block) *
                    pcp_size)
            rank_plans.append((rank_steps, int(q_global), int(q_tile_size),
                               int(rank_q_offset), int(effective_kv_pages),
                               int(scheduled_steps)))
            rank_steps += scheduled_steps
            rank_q_offset += int(q_tile_size)
        actual_steps[consumer_rank] = rank_steps
        actual_max_steps = max(actual_max_steps, rank_steps)
        tile_plans.append(rank_plans)

    max_steps = actual_max_steps
    if pad_steps_to is not None:
        pad_steps_to = int(pad_steps_to)
        if pad_steps_to < actual_max_steps:
            raise ValueError(
                "pad_steps_to must be >= generated schedule steps: "
                f"{pad_steps_to=} {actual_max_steps=}.")
        max_steps = pad_steps_to

    packed_schedule = np.zeros(
        (max_steps, pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
        dtype=np.int32)
    packed_schedule[..., ScheduleField.REQ_ID] = -1
    packed_schedule[..., ScheduleField.KV_PAGE_RANK] = -1
    packed_schedule[..., ScheduleField.KV_PAGE_IDX] = -1

    for consumer_rank, rank_plans in enumerate(tile_plans):
        for (start_step, q_global, q_tile_size, q_hbm_offset,
             effective_kv_pages, scheduled_steps) in rank_plans:
            _fill_pcp_streaming_schedule_rows_vectorized(
                packed_schedule,
                start_step=start_step,
                num_steps=scheduled_steps,
                consumer_rank=consumer_rank,
                lane=0,
                req_id=0,
                q_global_start=q_global,
                q_tile_size=q_tile_size,
                q_hbm_offset=q_hbm_offset,
                kv_len=kv_len,
                effective_kv_pages=effective_kv_pages,
                block_tables=block_tables,
                page_size=page_size,
                pcp_size=pcp_size,
                kv_pages_per_block=kv_pages_per_block,
            )

    fields = {
        name: _packed_field(packed_schedule,
                            getattr(ScheduleField, name.upper()))
        for name in _PACKED_FIELD_NAMES[1:]
    }
    req_id = _packed_field(packed_schedule, ScheduleField.REQ_ID)
    kv_page_indices = None
    if kv_pages_per_block > 1:
        kv_page_indices = np.transpose(
            packed_schedule[
                ...,
                ScheduleField.KV_PAGE_INDICES_START:
                ScheduleField.KV_PAGE_INDICES_START + kv_pages_per_block,
            ],
            (1, 0, 2, 3),
        ).copy()

    return PcpStreamingSchedule(
        req_id=req_id,
        actual_steps=actual_steps,
        global_actual_steps=np.array([actual_max_steps], dtype=np.int32),
        packed_schedule=packed_schedule,
        kv_page_indices=kv_page_indices,
        **fields,
    )


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
    """Generate the supported vectorized PCP streaming schedule.

    The production path is intentionally fail-closed for the first optimized
    deployment: one aligned active request, one lane, and PCP page-group padding.
    Unsupported shapes raise instead of silently falling back to another host
    generator.
    """
    return _generate_pcp_streaming_schedule_vectorized_aligned(
        kv_lens=kv_lens,
        cu_q_lens=cu_q_lens,
        q_start_offsets=q_start_offsets,
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=bq_sz,
        pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
        pad_steps_to=pad_steps_to,
        kv_pages_per_block=kv_pages_per_block,
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
