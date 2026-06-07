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

import enum
import functools
from dataclasses import dataclass, field
from typing import Any

import jax


class PcpMode(enum.Enum):
    """Static PCP attention execution mode."""

    DISABLED = 0
    PREFILL_STREAMING = 1
    DECODE_SHARDED_KV = 2


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "pcp_slot_ids",
        "pcp_source_block_tables",
        "pcp_streaming_schedule",
        "pcp_streaming_active_page_groups",
        "pcp_gdn_reorder_indices",
    ],
    meta_fields=["padded_num_reqs"],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    # None for pooling models that using no KV cache
    block_tables: jax.Array | None = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 — physical slot id (∈ [0, _mamba_num_blocks))
    # in the mamba kv-cache for the request currently in each persistent-
    # batch position. Used by mamba/GDN ops to read/write recurrent state
    # without going through `block_tables`, since the mamba pool is
    # smaller than the attention pool under compact-mamba sizing.
    # None for models without mamba layers; pure-mamba models would also
    # use this field, only hybrid models exercise it today.
    mamba_state_indices: jax.Array | None = None
    # Optional PCP metadata. PCP prefill is supported only through the streaming
    # RPA kernel; materialized decode KV is retained for correctness testing.
    pcp_slot_ids: jax.Array | None = None
    pcp_source_block_tables: jax.Array | None = None
    pcp_streaming_schedule: jax.Array | None = None
    pcp_streaming_active_page_groups: jax.Array | None = None
    # (padded_total_num_scheduled_tokens,) int32 — maps packed rank-major
    # token position → original sequential position. Used by GDN layers
    # under PCP prefill to AllGather + reorder tokens to sequential order
    # before running the recurrent scan.  -1 marks padding slots.
    # None when PCP is disabled or model has no mamba/GDN layers.
    pcp_gdn_reorder_indices: jax.Array | None = None

    # The actual number of requests padded to the compiled buckets. The bucket
    # contains only max_reqs by default to reduce model precompilation time.
    # If env var ATTN_BUCKETIZED_NUM_REQS=true, the buckets are the
    # power of 2 between min and max requests.
    # Env var ATTN_CUSTOM_NUM_REQS_BUCKETS can manually override the buckets.
    padded_num_reqs: int = -1

    query_start_loc_cpu: Any = field(init=False)
