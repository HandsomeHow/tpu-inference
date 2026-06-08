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

import numpy as np

from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField)
from tpu_inference.runner.tpu_runner import (
    _PCPStreamingScheduleTemplateCache, _build_pcp_attention_metadata,
    _prewarm_pcp_streaming_schedule_template_cache)

PCP_SIZE = 8
BLOCK_SIZE = 4
PAGES_PER_SEQ = 64
Q_LEN = 32
SEQ_LEN = 256


def _block_tables(offset: int = 0) -> np.ndarray:
    return np.arange(offset, offset + PAGES_PER_SEQ,
                     dtype=np.int32).reshape(1, PAGES_PER_SEQ)


def _block_tables_for_reqs(num_reqs: int, offset: int = 0) -> np.ndarray:
    return np.arange(offset, offset + num_reqs * PAGES_PER_SEQ,
                     dtype=np.int32).reshape(num_reqs, PAGES_PER_SEQ)


def _build_metadata(
    *,
    block_tables: np.ndarray | None = None,
    seq_len: int = SEQ_LEN,
    q_len: int = Q_LEN,
    kv_pages_per_block: int = 8,
    cache: _PCPStreamingScheduleTemplateCache | None = None,
):
    if block_tables is None:
        block_tables = _block_tables()
    return _build_pcp_attention_metadata(
        num_scheduled_tokens_per_req=[q_len],
        seq_lens_per_req=[seq_len],
        block_tables=block_tables,
        pcp_size=PCP_SIZE,
        interleave_size=BLOCK_SIZE,
        padded_num_tokens=Q_LEN,
        max_num_reqs_per_dp_rank=block_tables.shape[0],
        block_size=BLOCK_SIZE,
        build_streaming_schedule=True,
        streaming_num_lanes=1,
        streaming_q_block_size=4,
        streaming_kv_pages_per_block=kv_pages_per_block,
        streaming_schedule_template_cache=cache,
    )


def _valid_page_mask(schedule: np.ndarray, page_offset: int) -> np.ndarray:
    return np.logical_and(
        schedule[..., ScheduleField.REQ_ID] >= 0,
        schedule[..., ScheduleField.KV_VALID_LEN] > page_offset * BLOCK_SIZE,
    )


def test_template_cache_matches_full_schedule_with_grouped_kv_pages():
    full = _build_metadata(cache=None, kv_pages_per_block=8)
    cache = _PCPStreamingScheduleTemplateCache()
    cached = _build_metadata(cache=cache, kv_pages_per_block=8)

    assert full.streaming_schedule is not None
    assert cached.streaming_schedule is not None
    np.testing.assert_array_equal(cached.streaming_schedule,
                                  full.streaming_schedule)
    np.testing.assert_array_equal(cached.streaming_active_page_groups,
                                  full.streaming_active_page_groups)
    np.testing.assert_array_equal(cached.slot_ids, full.slot_ids)
    assert cache.misses == 1


def test_template_cache_hits_for_identical_logical_schedule():
    cache = _PCPStreamingScheduleTemplateCache()

    first = _build_metadata(cache=cache, kv_pages_per_block=8)
    second = _build_metadata(cache=cache, kv_pages_per_block=8)

    assert first.streaming_schedule is not None
    assert second.streaming_schedule is not None
    np.testing.assert_array_equal(second.streaming_schedule,
                                  first.streaming_schedule)
    assert cache.misses == 1
    assert len(cache) == 1


def test_template_cache_keys_chunk_position_by_seq_len():
    cache = _PCPStreamingScheduleTemplateCache()

    first = _build_metadata(cache=cache, seq_len=128, kv_pages_per_block=1)
    second = _build_metadata(cache=cache, seq_len=256, kv_pages_per_block=1)

    assert first.streaming_schedule is not None
    assert second.streaming_schedule is not None
    assert cache.misses == 2
    assert len(cache) == 2
    assert (first.streaming_active_page_groups[0]
            < second.streaming_active_page_groups[0])
    assert not np.array_equal(first.streaming_schedule,
                              second.streaming_schedule)


def test_template_cache_patches_runtime_physical_page_ids():
    cache = _PCPStreamingScheduleTemplateCache()

    first = _build_metadata(block_tables=_block_tables(0),
                            cache=cache,
                            kv_pages_per_block=8)
    second = _build_metadata(block_tables=_block_tables(1000),
                             cache=cache,
                             kv_pages_per_block=8)

    assert first.streaming_schedule is not None
    assert second.streaming_schedule is not None
    assert cache.misses == 1
    assert len(cache) == 1
    assert not np.array_equal(first.streaming_schedule,
                              second.streaming_schedule)

    first_schedule = first.streaming_schedule
    second_schedule = second.streaming_schedule
    mask = _valid_page_mask(first_schedule, page_offset=0)
    assert np.any(mask)
    np.testing.assert_array_equal(
        second_schedule[..., ScheduleField.KV_PAGE_IDX][mask],
        first_schedule[..., ScheduleField.KV_PAGE_IDX][mask] + 1000,
    )
    for page_offset in range(8):
        field = ScheduleField.KV_PAGE_INDICES_START + page_offset
        mask = _valid_page_mask(first_schedule, page_offset)
        if not np.any(mask):
            continue
        np.testing.assert_array_equal(
            second_schedule[..., field][mask],
            first_schedule[..., field][mask] + 1000,
        )


def test_prewarmed_template_cache_covers_single_request_prefill_chunks():
    cache = _PCPStreamingScheduleTemplateCache()
    max_num_reqs_per_dp_rank = 2

    generated = _prewarm_pcp_streaming_schedule_template_cache(
        cache,
        max_sequence_len=SEQ_LEN,
        chunk_size=Q_LEN,
        pages_per_seq=PAGES_PER_SEQ,
        max_num_reqs_per_dp_rank=max_num_reqs_per_dp_rank,
        num_token_paddings_per_dp=[Q_LEN],
        pcp_size=PCP_SIZE,
        block_size=BLOCK_SIZE,
        interleave_size=BLOCK_SIZE,
        streaming_num_lanes=1,
        streaming_q_block_size=4,
        streaming_kv_pages_per_block=8,
    )

    assert generated == SEQ_LEN // Q_LEN
    assert len(cache) == generated

    misses_after_prewarm = cache.misses
    runtime_block_tables = _block_tables_for_reqs(max_num_reqs_per_dp_rank,
                                                  offset=1000)
    cached = _build_pcp_attention_metadata(
        num_scheduled_tokens_per_req=[Q_LEN],
        seq_lens_per_req=[SEQ_LEN // 2],
        block_tables=runtime_block_tables,
        pcp_size=PCP_SIZE,
        interleave_size=BLOCK_SIZE,
        padded_num_tokens=Q_LEN,
        max_num_reqs_per_dp_rank=max_num_reqs_per_dp_rank,
        block_size=BLOCK_SIZE,
        build_streaming_schedule=True,
        streaming_num_lanes=1,
        streaming_q_block_size=4,
        streaming_kv_pages_per_block=8,
        streaming_schedule_template_cache=cache,
    )
    full = _build_pcp_attention_metadata(
        num_scheduled_tokens_per_req=[Q_LEN],
        seq_lens_per_req=[SEQ_LEN // 2],
        block_tables=runtime_block_tables,
        pcp_size=PCP_SIZE,
        interleave_size=BLOCK_SIZE,
        padded_num_tokens=Q_LEN,
        max_num_reqs_per_dp_rank=max_num_reqs_per_dp_rank,
        block_size=BLOCK_SIZE,
        build_streaming_schedule=True,
        streaming_num_lanes=1,
        streaming_q_block_size=4,
        streaming_kv_pages_per_block=8,
        streaming_schedule_template_cache=None,
    )

    assert cache.misses == misses_after_prewarm
    assert cached.streaming_schedule is not None
    assert full.streaming_schedule is not None
    np.testing.assert_array_equal(cached.streaming_schedule,
                                  full.streaming_schedule)
