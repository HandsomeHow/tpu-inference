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

import dataclasses

import numpy as np
import pytest

from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField, generate_pcp_streaming_schedule,
    unpack_pcp_streaming_schedule_field, validate_pcp_streaming_schedule)


def test_generate_schedule_uses_interleave_q_ownership_and_page_mapping():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    assert schedule.req_id.shape == (4, 6, 1)
    np.testing.assert_array_equal(schedule.actual_steps,
                                  np.array([5, 6, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(schedule.global_actual_steps,
                                  np.array([6], dtype=np.int32))

    # PCP=4/interleave=2/q_global_base=5 assigns Q chunks:
    # rank0 [8,10), rank1 [10,12), rank2 [5,6), rank3 [6,8).
    np.testing.assert_array_equal(schedule.q_global_start[:, 0, 0],
                                  np.array([8, 10, 5, 6], dtype=np.int32))
    np.testing.assert_array_equal(schedule.q_tile_size[:, 0, 0],
                                  np.array([2, 2, 1, 2], dtype=np.int32))

    np.testing.assert_array_equal(schedule.kv_page_rank[0, :5, 0],
                                  np.array([0, 1, 2, 3, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.kv_page_idx[0, :5, 0],
        np.array([100, 100, 100, 100, 101], dtype=np.int32),
    )
    np.testing.assert_array_equal(schedule.kv_global_start[0, :5, 0],
                                  np.array([0, 2, 4, 6, 8], dtype=np.int32))
    np.testing.assert_array_equal(schedule.is_first_kv[0, :5, 0],
                                  np.array([1, 0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(schedule.is_last_kv[0, :5, 0],
                                  np.array([0, 0, 0, 0, 1], dtype=np.int32))


def test_schedule_q_offsets_follow_rank_major_packed_order_across_requests():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[12, 3],
        cu_q_lens=[0, 7, 10],
        q_start_offsets=[5, 0],
        block_tables=np.array([[100, 101], [200, 0]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    # Consumer rank 0 first handles request 0 chunk [8,10) at local offset 0.
    assert schedule.req_id[0, 0, 0] == 0
    assert schedule.q_hbm_offset[0, 0, 0] == 0
    assert schedule.q_tile_size[0, 0, 0] == 2

    # Then request 1 chunk [0,2) is appended after rank 0's first two Q rows.
    assert schedule.req_id[0, 5, 0] == 1
    assert schedule.q_global_start[0, 5, 0] == 0
    assert schedule.q_hbm_offset[0, 5, 0] == 2
    assert schedule.o_hbm_offset[0, 5, 0] == 2


def test_validate_schedule_lane_invariant_accepts_generated_schedule():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[16],
        cu_q_lens=[0, 16],
        q_start_offsets=[0],
        block_tables=np.array([[10, 11]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=2,
        bq_sz=2,
    )

    validate_pcp_streaming_schedule(schedule)


def test_schedule_packed_fields_match_unpacked_arrays():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    assert schedule.packed_schedule.shape == (
        6,
        4,
        1,
        ScheduleField.PACKED_NUM_FIELDS,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.REQ_ID),
        schedule.req_id,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.KV_PAGE_RANK),
        schedule.kv_page_rank,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.KV_PAGE_IDX),
        schedule.kv_page_idx,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.Q_GLOBAL_START),
        schedule.q_global_start,
    )
    np.testing.assert_array_equal(
        schedule.packed_schedule[0, :, 0, ScheduleField.Q_TILE_SIZE],
        schedule.q_tile_size[:, 0, 0],
    )


def test_schedule_packed_field_rejects_invalid_field_index():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    with pytest.raises(ValueError, match="invalid schedule field"):
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.NUM_FIELDS)


def test_validate_schedule_lane_invariant_rejects_q_offset_change_inside_tile():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )
    q_hbm_offset = schedule.q_hbm_offset.copy()
    q_hbm_offset[0, 1, 0] = 99
    bad_schedule = dataclasses.replace(schedule, q_hbm_offset=q_hbm_offset)

    with pytest.raises(ValueError, match="q_hbm_offset changed"):
        validate_pcp_streaming_schedule(bad_schedule)


def test_generate_schedule_rejects_non_aligned_page_and_interleave_size():
    with pytest.raises(NotImplementedError, match="page_size == interleave"):
        generate_pcp_streaming_schedule(
            kv_lens=[12],
            cu_q_lens=[0, 7],
            q_start_offsets=[5],
            block_tables=np.array([[100, 101]], dtype=np.int32),
            page_size=4,
            pcp_size=4,
            interleave_size=2,
            num_lanes=1,
            bq_sz=2,
        )
