# Copyright 2025 Google LLC
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

from tpu_inference.layers.common.pcp_layout import (
    build_pcp_rank_major_token_order, pcp_local_token_counts,
    pcp_query_chunk_ranges)


def test_pcp_query_chunk_ranges_use_interleave_ownership_with_global_offset():
    chunks_by_rank = [
        list(
            pcp_query_chunk_ranges(
                q_len=7,
                q_global_base=5,
                pcp_rank=rank,
                pcp_size=4,
                interleave_size=2,
            )) for rank in range(4)
    ]

    assert chunks_by_rank == [[(8, 10)], [(10, 12)], [(5, 6)], [(6, 8)]]


def test_pcp_local_token_counts_follow_interleave_ownership():
    counts = pcp_local_token_counts(
        [7],
        pcp_size=4,
        interleave_size=2,
        token_start_offsets_per_req=[5],
    )

    np.testing.assert_array_equal(counts, np.array([2, 2, 1, 2],
                                                   dtype=np.int32))


def test_build_pcp_rank_major_token_order_uses_interleave_chunks():
    order, inverse = build_pcp_rank_major_token_order(
        [7],
        pcp_size=4,
        interleave_size=2,
        padded_num_tokens=8,
        token_start_offsets_per_req=[5],
    )

    np.testing.assert_array_equal(order, np.array([3, 4, 5, 6, 0, -1, 1, 2]))
    np.testing.assert_array_equal(inverse, np.array([4, 6, 7, 0, 1, 2, 3]))
