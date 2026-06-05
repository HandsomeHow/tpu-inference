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


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


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
    actual_steps: np.ndarray
    global_actual_steps: np.ndarray

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
) -> PcpStreamingSchedule:
    """Generate a replicated PCP streaming schedule for page-aligned PCP.

    Q ownership follows PCP interleave chunks. KV page ownership follows the
    fast path where page_size == interleave_size, so each global KV page belongs
    to exactly one PCP rank.
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
    )

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

            for chunk_start, chunk_end in pcp_query_chunk_ranges(
                    q_len, q_global_base, consumer_rank, pcp_size,
                    interleave_size):
                chunk_len = chunk_end - chunk_start
                num_tiles = _cdiv(chunk_len, bq_sz)

                for tile_idx in range(num_tiles):
                    tile_start = tile_idx * bq_sz
                    tile_len = min(bq_sz, chunk_len - tile_start)
                    q_global = chunk_start + tile_start
                    q_global_last = q_global + tile_len - 1
                    q_hbm_offset = rank_q_offset
                    target_lane = int(np.argmin(lane_lengths))
                    effective_kv_pages = min(num_kv_pages,
                                             q_global_last // page_size + 1)

                    for kv_page_seq_idx in range(effective_kv_pages):
                        global_token_start = kv_page_seq_idx * page_size
                        global_page = global_token_start // page_size
                        src_rank = global_page % pcp_size
                        local_page_index = global_page // pcp_size
                        if local_page_index >= block_tables.shape[1]:
                            raise ValueError(
                                "block_tables does not cover requested KV page.")
                        physical_page = int(block_tables[req_idx,
                                                         local_page_index])
                        kv_valid = min(page_size,
                                       kv_len - global_token_start)
                        lane_entries[target_lane].append(
                            _Entry(
                                req_id=req_idx,
                                kv_page_rank=src_rank,
                                kv_page_idx=physical_page,
                                is_first_kv=int(kv_page_seq_idx == 0),
                                is_last_kv=int(
                                    kv_page_seq_idx == effective_kv_pages - 1),
                                load_q=int(kv_page_seq_idx == 0),
                                q_global_start=q_global,
                                kv_global_start=global_token_start,
                                kv_valid_len=kv_valid,
                                q_hbm_offset=q_hbm_offset,
                                q_tile_size=tile_len,
                                o_hbm_offset=q_hbm_offset,
                            ))
                        lane_lengths[target_lane] += 1

                    rank_q_offset += tile_len

        actual_steps[consumer_rank] = int(lane_lengths.max(initial=0))
        schedules.append(lane_entries)

    max_steps = int(actual_steps.max(initial=0))
    shape = (pcp_size, max_steps, num_lanes)

    req_id = np.full(shape, -1, dtype=np.int32)
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

    return PcpStreamingSchedule(
        req_id=req_id,
        actual_steps=actual_steps,
        global_actual_steps=np.array([max_steps], dtype=np.int32),
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
                    if in_tile:
                        raise ValueError("idle entry inside active q_tile.")
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
