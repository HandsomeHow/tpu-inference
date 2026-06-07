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

import functools
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import vllm.envs as vllm_envs
from flax import nnx
from jax._src import mesh as mesh_lib
from jax._src.pallas.utils import next_power_of_2
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.parallel import ParallelConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.forward_context import set_forward_context
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             DraftTokenIds, KVConnectorOutput, LogprobsLists,
                             LogprobsTensors, ModelRunnerOutput)
from vllm.v1.request import Request
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.kv_connector_model_runner_mixin import \
    KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

import tpu_inference.envs as envs
from tpu_inference import utils as common_utils
from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField, _iter_pcp_q_tiles, _q_global_last_for_strided_tile,
    generate_pcp_streaming_schedule)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.pcp_layout import (
    apply_pcp_rank_major_token_order as _apply_pcp_rank_major_token_order,
    build_pcp_logits_indices as _build_pcp_logits_indices,
    build_pcp_rank_major_token_order as _build_pcp_rank_major_token_order,
    pcp_local_token_counts as _pcp_local_token_counts,
    pcp_query_chunk_ranges as _pcp_query_chunk_ranges,
    pcp_query_start_offsets as _pcp_query_start_offsets,
)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  MESH_AXIS_NAMES_2D,
                                                  ShardingAxisName,
                                                  ShardingConfigManager)
from tpu_inference.layers.jax.sample.rejection_sampler import RejectionSampler
from tpu_inference.layers.jax.sample.sampling import (compute_logprobs,
                                                      gather_logprobs, sample)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    shard_put, transfer_state_with_mappings)
from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.compilation_manager import CompilationManager
from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.kv_cache_manager import KVCacheManager
from tpu_inference.runner.lora_utils import LoraUtils
from tpu_inference.runner.multimodal_manager import MultiModalManager
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager
from tpu_inference.runner.speculative_decoding_manager import (
    SpecDecodeMetadata, SpeculativeDecodingManager)
from tpu_inference.runner.structured_decoding_manager import \
    StructuredDecodingManager
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.spec_decode.jax.utils import (
    concat_last_sampled_tokens_and_draft_tokens, extract_last_sampled_tokens)
from tpu_inference.utils import (device_array, make_optimized_mesh,
                                 time_function, to_jax_dtype, to_torch_dtype)

logger = init_logger(__name__)

logging.getLogger("torchax.tensor").setLevel(logging.ERROR)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Holds asynchronous model output specifically from a TPU runner.

    This class acts as a wrapper around the standard ModelRunnerOutput. Its
    primary purpose is to hold references to data still on the TPU device
    (like the `next_tokens` JAX array) without blocking the main thread.

    The `get_output()` method is called to resolve these async results,
    triggering the JAX device-to-host (CPU) data transfer and populating
    the final `ModelRunnerOutput` object.
    """

    def __init__(self,
                 model_runner_output: ModelRunnerOutput,
                 next_tokens: jax.Array,
                 num_reqs: int,
                 discard_sampled_tokens_req_indices: list[int],
                 logits_indices_selector: Optional[List[int]] = None,
                 logprobs_tensors: Optional[LogprobsTensors] = None,
                 expert_indices: Optional[jax.Array] = None,
                 total_num_scheduled_tokens: int = 0,
                 spec_decode_metadata: Optional[SpecDecodeMetadata] = None,
                 runner=None):
        self._model_runner_output = model_runner_output
        self._next_tokens = next_tokens
        self._num_reqs = num_reqs
        self._discard_sampled_tokens_req_indices = discard_sampled_tokens_req_indices
        self.logits_indices_selector: list[int] = logits_indices_selector
        self._logprobs_tensors = logprobs_tensors
        self._expert_indices = expert_indices
        self._total_num_scheduled_tokens = total_num_scheduled_tokens
        self._spec_decode_metadata = spec_decode_metadata
        self._runner = runner

    def get_output(self) -> ModelRunnerOutput:
        valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
            self._runner, self._spec_decode_metadata, self._next_tokens,
            self.logits_indices_selector,
            self._discard_sampled_tokens_req_indices, self._num_reqs)

        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids

        if self._logprobs_tensors is not None:
            # Use materialize to ensure logprobs are ready on host when we return async results
            self._model_runner_output.logprobs = _jax_logprobs_materialize(
                self._logprobs_tensors, self.logits_indices_selector)

        if self._expert_indices is not None:
            expert_indices_cpu = np.asarray(
                jax.device_get(self._expert_indices))
            expert_indices_cpu = expert_indices_cpu[:, :self.
                                                    _total_num_scheduled_tokens, :]
            self._model_runner_output.expert_indices = expert_indices_cpu

        return self._model_runner_output


@dataclass
class AsyncPreResults:
    req_ids: list[str]
    next_tokens: jax.Array
    request_seq_lens: list[tuple[int, CachedRequestState, int]]
    discard_sampled_tokens_req_indices: list[int]
    placeholder_req_id_to_index: dict[str, int]
    logits_indices_selector: Optional[List[int]] = None

    # Only when spec decoding is enabled, the follow variables
    # are populated.
    spec_decode_next_tokens: Optional[
        jax.Array] = None  # [max_num_reqs * (gamma + 1)]
    spec_decode_num_rejected_tokens: Optional[
        jax.Array] = None  # [max_num_reqs]
    spec_decode_metadata: Optional[SpecDecodeMetadata] = None


@dataclass
class ExecuteModelState:
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "VllmSchedulerOutput"
    attn_metadata: AttentionMetadata
    sampling_metadata: TPUSupportedSamplingMetadata
    input_ids: Optional[jax.Array]
    hidden_states: jax.Array
    logits: jax.Array
    aux_hidden_states: Optional[jax.Array]
    spec_decode_metadata: Optional[SpecDecodeMetadata]
    kv_connector_output: Optional[KVConnectorOutput]
    logits_indices_selector: Optional[List[int]] = None
    padded_num_reqs: Optional[int] = None
    expert_indices: Optional[jax.Array] = None
    full_hidden_states: Optional[jax.Array] = None


@jax.jit(donate_argnums=(0, 1, 2))
def _substitute_placeholder_token(
        input_ids: jax.Array, token_in_tpu_cur_input_indices: jax.Array,
        token_in_tpu_pre_next_tokens_indices: jax.Array,
        next_tokens: jax.Array, placeholder_num: int):
    """Substitute placeholder tokens from TPU for async scheduler

    Padding for parallelisation of the substitute_placeholder_token_fn
    [1, 3] => [1, 3, 0, 2, 4, 5, 6, 7, 8]
    The reason for such a special padding instead of padding with -1 is:
    An edge case when the end index needs to be updated and padding is required.
    If we pad the array with -1, the _substitute_placeholder_token_fn will repeatedly update the end element with the original value
    Although such a scenario is unlikely to happen in vLLM, it is best to eliminate any potential risks.

    Args:
        input_ids: possible input_ids size
        token_in_tpu_cur_input_indices: replace holder idx in input_ids. Length the same to input_ids.
        token_in_tpu_pre_next_tokens_indices: value idx in next_tokens. Length the same to input_ids.
        next_tokens: next tokens on the TPU from previous step.
        placeholder_num: number of placeholders. placeholder_num <= len(token_in_tpu_cur_input_indices)
    Return:
        input_ids after replace placeholder tokens
    """
    assert input_ids.shape == token_in_tpu_cur_input_indices.shape == token_in_tpu_pre_next_tokens_indices.shape, \
        f"Shape mismatch: input_ids and index arrays must have identical shapes due to precompilation assumptions. " \
        f"Got: {input_ids.shape=}, {token_in_tpu_cur_input_indices.shape=}, {token_in_tpu_pre_next_tokens_indices.shape=}"

    # updates the input_ids for all placeholders.
    mask = jnp.arange(input_ids.shape[0]) < placeholder_num
    new_token_values = next_tokens[token_in_tpu_pre_next_tokens_indices]
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = jnp.where(mask, new_token_values, original_values)
    return input_ids.at[token_in_tpu_cur_input_indices].set(update_values)


@jax.jit(donate_argnums=(0, 1))
def _subtract_num_rejected_tokens_fn(seq_lens: jax.Array, positions: jax.Array,
                                     num_rejected_tokens: jax.Array,
                                     seq_lens_subtract_indices: jax.Array,
                                     positions_subtract_indices: jax.Array):
    """Subtract the previous step's rejected-token counts from `seq_lens` and
    `positions`.
    """
    seq_valid = seq_lens_subtract_indices >= 0
    seq_subtract = jnp.where(seq_valid,
                             num_rejected_tokens[seq_lens_subtract_indices], 0)
    seq_lens = seq_lens - seq_subtract

    pos_valid = positions_subtract_indices >= 0
    pos_subtract = jnp.where(pos_valid,
                             num_rejected_tokens[positions_subtract_indices],
                             0)
    positions = positions - pos_subtract
    return seq_lens, positions


def _jax_logprobs_copy_to_host_async(
        logprobs_tensors: LogprobsTensors) -> LogprobsTensors:
    """Initiate non-blocking TPU-to-host copies for all logprobs arrays."""
    return LogprobsTensors(
        logprob_token_ids=jax.copy_to_host_async(
            logprobs_tensors.logprob_token_ids),
        logprobs=jax.copy_to_host_async(logprobs_tensors.logprobs),
        selected_token_ranks=jax.copy_to_host_async(
            logprobs_tensors.selected_token_ranks),
    )


def _jax_logprobs_materialize(
        logprobs_tensors: LogprobsTensors,
        logits_indices_selector: Optional[List[int]] = None,
        cu_num_generated_tokens: Optional[Any] = None) -> LogprobsLists:
    """Materializes logprobs from JAX arrays into NumPy-backed LogprobsLists."""
    log_token_ids = np.asarray(
        jax.device_get(logprobs_tensors.logprob_token_ids))
    logprobs_arr = np.asarray(jax.device_get(logprobs_tensors.logprobs))
    selected_token_ranks = np.asarray(
        jax.device_get(logprobs_tensors.selected_token_ranks))

    if logits_indices_selector is not None:
        log_token_ids = log_token_ids[logits_indices_selector]
        logprobs_arr = logprobs_arr[logits_indices_selector]
        selected_token_ranks = selected_token_ranks[logits_indices_selector]

    return LogprobsLists(
        logprob_token_ids=np.array(log_token_ids.tolist()),
        logprobs=np.array(logprobs_arr.tolist()),
        sampled_token_ranks=np.array(selected_token_ranks.tolist()),
        cu_num_generated_tokens=cu_num_generated_tokens,
    )


def _get_pcp_parallel_config(vllm_config: VllmConfig) -> tuple[int, int]:
    parallel_config = vllm_config.parallel_config
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    interleave_size = getattr(parallel_config, "cp_kv_cache_interleave_size",
                              1)
    if not isinstance(pcp_size, int):
        pcp_size = 1
    if not isinstance(interleave_size, int):
        interleave_size = 1
    return pcp_size, interleave_size


def _kv_cache_group_supports_pcp_attention_metadata(kv_cache_group: Any) -> bool:
    kv_cache_spec = getattr(kv_cache_group, "kv_cache_spec", None)
    return isinstance(kv_cache_spec, AttentionSpec)


def _scheduled_token_span(
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    req_id: str,
) -> tuple[int, int, int]:
    req_index = input_batch.req_id_to_index[req_id]
    return (
        int(input_batch.num_computed_tokens_cpu[req_index]),
        int(scheduler_output.num_scheduled_tokens[req_id]),
        int(input_batch.num_prompt_tokens[req_index]),
    )


def _request_uses_initial_pcp_prefill(
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    req_id: str,
) -> bool:
    computed_tokens, scheduled_tokens, prompt_tokens = _scheduled_token_span(
        input_batch, scheduler_output, req_id)
    return (scheduled_tokens > 1 and computed_tokens == 0
            and scheduled_tokens <= prompt_tokens)


def _request_uses_pcp_prefill(
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    req_id: str,
) -> bool:
    computed_tokens, scheduled_tokens, prompt_tokens = _scheduled_token_span(
        input_batch, scheduler_output, req_id)
    return (scheduled_tokens > 1 and computed_tokens < prompt_tokens
            and computed_tokens + scheduled_tokens <= prompt_tokens)


def _request_uses_pcp_materialized_kv(
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    req_id: str,
) -> bool:
    computed_tokens, scheduled_tokens, prompt_tokens = _scheduled_token_span(
        input_batch, scheduler_output, req_id)
    if scheduled_tokens <= 0:
        return False
    if computed_tokens < prompt_tokens:
        if computed_tokens + scheduled_tokens > prompt_tokens:
            return False
        return scheduled_tokens == 1
    return scheduled_tokens == 1


def _batch_has_unsupported_pcp_mix(
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    num_reqs: int,
) -> bool:
    has_prompt = False
    has_decode = False
    for req_id in input_batch.req_ids[:num_reqs]:
        if _request_uses_pcp_prefill(input_batch, scheduler_output, req_id):
            has_prompt = True
            continue
        if _request_uses_pcp_materialized_kv(input_batch, scheduler_output,
                                             req_id):
            has_decode = True
            continue
        if scheduler_output.num_scheduled_tokens[req_id] > 0:
            return True
    return has_prompt and has_decode


def _batch_uses_pcp_prefill(
    vllm_config: VllmConfig,
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    num_reqs: int,
) -> bool:
    """Return whether this scheduled batch should use PCP attention."""
    pcp_size, _ = _get_pcp_parallel_config(vllm_config)
    if pcp_size <= 1 or num_reqs <= 0:
        return False

    for req_id in input_batch.req_ids[:num_reqs]:
        if not _request_uses_pcp_prefill(input_batch, scheduler_output,
                                         req_id):
            return False
    return True


def _batch_uses_pcp_decode(
    vllm_config: VllmConfig,
    input_batch: InputBatch,
    scheduler_output: VllmSchedulerOutput,
    num_reqs: int,
) -> bool:
    """Return whether this batch should materialize PCP KV before attention."""
    pcp_size, _ = _get_pcp_parallel_config(vllm_config)
    if pcp_size <= 1 or num_reqs <= 0:
        return False

    for req_id in input_batch.req_ids[:num_reqs]:
        if not _request_uses_pcp_materialized_kv(input_batch, scheduler_output,
                                                 req_id):
            return False
    return True


def _attention_metadata_uses_pcp(
        attn_metadata: AttentionMetadata | dict[str, AttentionMetadata]) -> bool:
    if isinstance(attn_metadata, dict):
        return any(_attention_metadata_uses_pcp(md)
                   for md in attn_metadata.values())
    return getattr(attn_metadata, "pcp_slot_ids", None) is not None


def _logits_indices_require_global_gather(
    vllm_config: VllmConfig,
    attn_metadata: AttentionMetadata | dict[str, AttentionMetadata],
) -> bool:
    pcp_size, _ = _get_pcp_parallel_config(vllm_config)
    return pcp_size > 1 or _attention_metadata_uses_pcp(attn_metadata)


@dataclass(frozen=True)
class _PCPAttentionMetadataHost:
    slot_ids: np.ndarray
    streaming_schedule: np.ndarray | None = None
    streaming_active_page_groups: np.ndarray | None = None


def _build_pcp_local_slot_ids(
    q_lens: np.ndarray,
    seq_lens: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
) -> np.ndarray:
    """Build rank-major local cache slot ids for PCP prefill K/V writes."""
    if block_size <= 0:
        raise ValueError(f"Expected positive block_size, got {block_size}.")
    local_padded_num_tokens = padded_num_tokens // pcp_size
    slot_ids = np.full(padded_num_tokens, -1, dtype=np.int32)
    rank_offsets = np.zeros(pcp_size, dtype=np.int64)
    virtual_block_size = block_size * pcp_size

    for req_idx, q_len in enumerate(q_lens):
        q_len = int(q_len)
        seq_len = int(seq_lens[req_idx])
        q_global_base = seq_len - q_len
        if q_global_base < 0:
            raise ValueError("seq_lens_per_req must be >= "
                             "num_scheduled_tokens_per_req.")
        for pcp_rank in range(pcp_size):
            for chunk_start, chunk_end in _pcp_query_chunk_ranges(
                    q_len, q_global_base, pcp_rank, pcp_size,
                    interleave_size):
                positions = np.arange(chunk_start, chunk_end, dtype=np.int64)
                block_indices = positions // virtual_block_size
                virtual_offsets = positions - block_indices * virtual_block_size
                is_local = ((virtual_offsets // interleave_size) %
                            pcp_size) == pcp_rank
                if not np.all(is_local):
                    raise ValueError("PCP slot mapping produced a non-local "
                                     "token for its packed rank.")
                local_offsets = (
                    (virtual_offsets //
                     (pcp_size * interleave_size)) * interleave_size +
                    (virtual_offsets % interleave_size))
                block_numbers = block_tables[req_idx,
                                             block_indices].astype(np.int32)
                local_slots = (block_numbers * block_size +
                               local_offsets).astype(np.int32)

                dst_start = (pcp_rank * local_padded_num_tokens +
                             rank_offsets[pcp_rank])
                dst_end = dst_start + local_slots.shape[0]
                if dst_end > (pcp_rank + 1) * local_padded_num_tokens:
                    raise ValueError(
                        "PCP local slot count exceeds padded local capacity.")
                slot_ids[dst_start:dst_end] = local_slots
                rank_offsets[pcp_rank] += local_slots.shape[0]

    return slot_ids


def _estimate_pcp_streaming_schedule_steps_ub(
    q_lens: np.ndarray,
    capacity_tokens: int,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    q_block_size: int,
    kv_pages_per_block: int = 1,
) -> int:
    max_global_pages = cdiv(capacity_tokens, block_size)
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
                    cdiv(effective_pages,
                         pcp_size * kv_pages_per_block) * pcp_size)
                target_lane = int(np.argmin(lane_lengths))
                lane_lengths[target_lane] += scheduled_pages
        max_steps = max(max_steps, int(lane_lengths.max(initial=0)))
    return max_steps


def _build_pcp_decode_attention_metadata(
    seq_lens_per_req: list[int] | np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    max_num_reqs_per_dp_rank: int,
    num_scheduled_tokens_per_req: list[int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build runner-owned replicated-Q/local-KV metadata for PCP materialize."""
    if block_size <= 0:
        raise ValueError("PCP decode metadata requires block_size > 0.")
    if pcp_size <= 1:
        raise ValueError("PCP decode metadata requires pcp_size > 1.")
    if interleave_size <= 0:
        raise ValueError(
            "PCP decode metadata requires interleave_size > 0.")
    if block_size % interleave_size != 0:
        raise ValueError(
            "PCP decode metadata requires block_size % interleave_size == 0.")
    seq_lens = np.asarray(seq_lens_per_req, dtype=np.int32)
    if seq_lens.size > max_num_reqs_per_dp_rank:
        raise ValueError(
            "active request count exceeds max_num_reqs_per_dp_rank.")
    if num_scheduled_tokens_per_req is None:
        q_lens = np.ones(seq_lens.size, dtype=np.int32)
    else:
        q_lens = np.asarray(num_scheduled_tokens_per_req, dtype=np.int32)
        if q_lens.size != seq_lens.size:
            raise ValueError("num_scheduled_tokens_per_req and "
                             "seq_lens_per_req must have the same number of "
                             "active requests.")
        if np.any(q_lens <= 0):
            raise ValueError("num_scheduled_tokens_per_req must be positive.")
        if np.any(seq_lens < q_lens):
            raise ValueError("seq_lens_per_req must be >= "
                             "num_scheduled_tokens_per_req.")

    seq_lens_full = np.zeros(max_num_reqs_per_dp_rank, dtype=np.int32)
    q_lens_full = np.zeros(max_num_reqs_per_dp_rank, dtype=np.int32)
    seq_lens_full[:seq_lens.size] = seq_lens
    q_lens_full[:q_lens.size] = q_lens

    block_tables = np.asarray(block_tables, dtype=np.int32)
    if block_tables.ndim == 1:
        if block_tables.size % max_num_reqs_per_dp_rank != 0:
            raise ValueError("flat block_tables size must be divisible by "
                             "max_num_reqs_per_dp_rank.")
        pages_per_seq = block_tables.size // max_num_reqs_per_dp_rank
        block_tables = block_tables.reshape(max_num_reqs_per_dp_rank,
                                            pages_per_seq)
    elif block_tables.ndim == 2:
        if block_tables.shape[0] != max_num_reqs_per_dp_rank:
            raise ValueError("block_tables first dimension must equal "
                             "max_num_reqs_per_dp_rank.")
        pages_per_seq = block_tables.shape[1]
    else:
        raise ValueError("block_tables must be rank 1 or 2.")
    max_capacity_tokens = pages_per_seq * block_size
    if np.any(seq_lens > max_capacity_tokens):
        raise ValueError(
            "seq_lens_per_req exceeds PCP decode block table capacity: "
            f"seq_lens_per_req={seq_lens.tolist()}, "
            f"max_capacity_tokens={int(max_capacity_tokens)}, "
            f"pages_per_seq={int(pages_per_seq)}, "
            f"block_size={int(block_size)}, "
            f"block_tables_shape={tuple(block_tables.shape)}.")
    virtual_blocks_per_req = cdiv(pages_per_seq, pcp_size)
    source_block_tables = block_tables[:, :virtual_blocks_per_req]

    rank_slot_ids = []
    virtual_block_size = block_size * pcp_size
    for pcp_rank in range(pcp_size):
        rank_slots = np.full(padded_num_tokens, -1, dtype=np.int32)
        token_offset = 0
        for req_idx, q_len in enumerate(q_lens_full):
            q_len = int(q_len)
            if q_len <= 0:
                continue
            q_global_base = int(seq_lens_full[req_idx] - q_len)
            for local_pos in range(q_len):
                dst_index = token_offset + local_pos
                position = q_global_base + local_pos
                block_index = position // virtual_block_size
                virtual_offset = position - block_index * virtual_block_size
                owner_rank = (virtual_offset // interleave_size) % pcp_size
                if owner_rank != pcp_rank:
                    continue
                local_offset = (
                    (virtual_offset //
                     (pcp_size * interleave_size)) * interleave_size +
                    (virtual_offset % interleave_size))
                block_number = block_tables[req_idx, block_index]
                rank_slots[dst_index] = np.int32(block_number * block_size +
                                                 local_offset)
            token_offset += q_len
        rank_slot_ids.append(rank_slots)

    return {
        "slot_ids": np.concatenate(rank_slot_ids).astype(np.int32),
        "source_block_tables": source_block_tables.astype(np.int32),
    }


def _build_pcp_attention_metadata(
    num_scheduled_tokens_per_req: list[int] | np.ndarray,
    seq_lens_per_req: list[int] | np.ndarray,
    block_tables: np.ndarray,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    max_num_reqs_per_dp_rank: int,
    block_size: int,
    build_streaming_schedule: bool = False,
    streaming_num_lanes: int = 1,
    streaming_q_block_size: int = 256,
    streaming_kv_pages_per_block: int = 1,
) -> _PCPAttentionMetadataHost:
    """Build runner-owned local-Q/full-KV metadata for one DP rank."""
    if pcp_size <= 1:
        raise ValueError("PCP attention metadata requires pcp_size > 1.")
    if padded_num_tokens % pcp_size != 0:
        raise ValueError(
            f"{padded_num_tokens=} must be divisible by {pcp_size=}.")

    q_lens = np.asarray(num_scheduled_tokens_per_req, dtype=np.int32)
    seq_lens = np.asarray(seq_lens_per_req, dtype=np.int32)
    if q_lens.size != seq_lens.size:
        raise ValueError("num_scheduled_tokens_per_req and seq_lens_per_req "
                         "must have the same number of active requests.")
    if np.any(seq_lens < q_lens):
        raise ValueError("seq_lens_per_req must be >= "
                         "num_scheduled_tokens_per_req.")
    if q_lens.size > max_num_reqs_per_dp_rank:
        raise ValueError(
            "active request count exceeds max_num_reqs_per_dp_rank.")

    q_lens_full = np.zeros(max_num_reqs_per_dp_rank, dtype=np.int32)
    kv_lens_full = np.zeros(max_num_reqs_per_dp_rank, dtype=np.int32)
    q_lens_full[:q_lens.size] = q_lens
    kv_lens_full[:seq_lens.size] = seq_lens

    block_tables = np.asarray(block_tables, dtype=np.int32)
    if block_tables.ndim == 1:
        if block_tables.size % max_num_reqs_per_dp_rank != 0:
            raise ValueError("flat block_tables size must be divisible by "
                             "max_num_reqs_per_dp_rank.")
        pages_per_seq = block_tables.size // max_num_reqs_per_dp_rank
        block_tables = block_tables.reshape(max_num_reqs_per_dp_rank,
                                            pages_per_seq)
    elif block_tables.ndim == 2:
        if block_tables.shape[0] != max_num_reqs_per_dp_rank:
            raise ValueError("block_tables first dimension must equal "
                             "max_num_reqs_per_dp_rank.")
        pages_per_seq = block_tables.shape[1]
    else:
        raise ValueError("block_tables must be rank 1 or 2.")

    q_global_base = kv_lens_full - q_lens_full

    streaming_schedule = None
    streaming_active_page_groups = None
    if build_streaming_schedule:
        local_padded_num_tokens = padded_num_tokens // pcp_size
        if local_padded_num_tokens % streaming_q_block_size != 0:
            raise ValueError(
                "PCP streaming schedule requires local padded tokens to be a "
                "multiple of streaming_q_block_size: got "
                f"{local_padded_num_tokens=} and {streaming_q_block_size=}.")
        virtual_blocks_per_req = cdiv(pages_per_seq, pcp_size)
        source_block_tables = block_tables[:, :virtual_blocks_per_req]
        max_streaming_steps = _estimate_pcp_streaming_schedule_steps_ub(
            q_lens_full,
            capacity_tokens=virtual_blocks_per_req * pcp_size * block_size,
            block_size=block_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            num_lanes=streaming_num_lanes,
            q_block_size=streaming_q_block_size,
            kv_pages_per_block=streaming_kv_pages_per_block,
        )
        cu_q_lens = np.pad(np.cumsum(q_lens_full, dtype=np.int32), (1, 0))
        streaming_schedule_host = generate_pcp_streaming_schedule(
            kv_lens=kv_lens_full,
            cu_q_lens=cu_q_lens,
            q_start_offsets=q_global_base,
            block_tables=source_block_tables,
            page_size=block_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            num_lanes=streaming_num_lanes,
            bq_sz=streaming_q_block_size,
            pad_kv_pages_to_pcp_group=True,
            pad_steps_to=max_streaming_steps,
            kv_pages_per_block=streaming_kv_pages_per_block,
        )
        actual_steps = int(streaming_schedule_host.global_actual_steps[0])
        if actual_steps % pcp_size != 0:
            raise ValueError("PCP streaming schedule steps must be padded to a "
                             "PCP page group.")
        streaming_schedule = streaming_schedule_host.packed_schedule
        streaming_active_page_groups = np.array([actual_steps // pcp_size],
                                                dtype=np.int32)

    return _PCPAttentionMetadataHost(
        slot_ids=_build_pcp_local_slot_ids(
            q_lens_full,
            kv_lens_full,
            block_tables,
            block_size,
            pcp_size,
            interleave_size,
            padded_num_tokens,
        ),
        streaming_schedule=streaming_schedule,
        streaming_active_page_groups=streaming_active_page_groups,
    )


def _merge_pcp_attention_metadata(
    metadata_per_dp: list[_PCPAttentionMetadataHost],
) -> _PCPAttentionMetadataHost:
    streaming_schedules = [m.streaming_schedule for m in metadata_per_dp]
    streaming_active_page_groups = [
        m.streaming_active_page_groups for m in metadata_per_dp
    ]
    if all(schedule is None for schedule in streaming_schedules):
        merged_streaming_schedule = None
        merged_streaming_active_page_groups = None
    elif any(schedule is None for schedule in streaming_schedules):
        raise ValueError("PCP streaming schedules must be present for every "
                         "DP rank or for none.")
    elif any(active_page_groups is None
             for active_page_groups in streaming_active_page_groups):
        raise ValueError("PCP streaming active page groups must be present for "
                         "every DP rank or for none.")
    else:
        max_steps = max(schedule.shape[0] for schedule in streaming_schedules)
        first_schedule = streaming_schedules[0]
        schedule_shape = (len(streaming_schedules), max_steps,
                          *first_schedule.shape[1:])
        merged_streaming_schedule = np.zeros(schedule_shape, dtype=np.int32)
        merged_streaming_schedule[..., ScheduleField.REQ_ID] = -1
        for dp_rank, schedule in enumerate(streaming_schedules):
            if schedule.shape[1:] != first_schedule.shape[1:]:
                raise ValueError("PCP streaming schedule static dimensions "
                                 "must match across DP ranks.")
            merged_streaming_schedule[dp_rank, :schedule.shape[0]] = schedule
        merged_streaming_active_page_groups = np.stack(
            streaming_active_page_groups, axis=0).astype(np.int32)

    return _PCPAttentionMetadataHost(
        slot_ids=np.concatenate([m.slot_ids for m in metadata_per_dp]),
        streaming_schedule=merged_streaming_schedule,
        streaming_active_page_groups=merged_streaming_active_page_groups,
    )


class TPUModelRunner(KVConnectorModelRunnerMixin, LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        devices: List[Any],
        rank: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # TODO(jevinjiang): override block size based on RPA v3.
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config: ParallelConfig = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        self.devices = devices
        self.dtype = self.model_config.dtype
        self.maybe_forbid_compile = runner_utils.ForbidCompile(
        ) if envs.VLLM_XLA_CHECK_RECOMPILATION else nullcontext()
        self.dp_size = self.vllm_config.sharding_config.total_dp_size
        self.rank = rank
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank

        self._init_random()
        self._init_mesh()
        self._init_phased_profiling()
        self._init_aggregated_stats_logging()
        self._init_mm()
        self._init_inputs()
        self._init_speculative_decoding()

        # Delegate functions to specific manager classes.
        self.compilation_manager = CompilationManager(self)
        if self.is_last_rank:
            self.speculative_decoding_manager = SpeculativeDecodingManager(
                self)
            self.structured_decoding_manager = StructuredDecodingManager(self)
        self.kv_cache_manager = KVCacheManager(self)
        self.mm_manager = MultiModalManager(self)
        self.persistent_batch_manager = PersistentBatchManager(
            self.requests, self.input_batch, self.encoder_cache,
            self.uses_mrope, self.model_config, self.is_last_rank)
        self.lora_utils = LoraUtils(self)

        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype == "auto":
            cache_dtype = self.dtype
        self.kv_cache_dtype = to_torch_dtype(cache_dtype)

        self._pre_async_results: AsyncPreResults | None = None
        self._substitute_placeholder_token_fn = _substitute_placeholder_token
        self.execute_model_state: ExecuteModelState | None = None
        self.batch_counter = 0

        self.kv_caches: list[jax.Array] = []
        self.layer_name_to_kvcache_index: dict[str, int] = {}

        self.is_pooling_model: bool = self.model_config.runner_type == "pooling"
        """Generative model or pooling model select different computations."""

    def _init_random(self):
        if self.model_config.seed is None:
            self.model_config.seed = 0
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        self.rng_key = jax.random.key(self.model_config.seed)

    def _init_mesh(self) -> None:
        if envs.NEW_MODEL_DESIGN:
            self.mesh = self._create_new_model_mesh()
        else:
            # NOTE(wenxindongwork): The new MoE kernel expects a 2D mesh, so we need
            # to create a 2D mesh for now. We should make the new_model_mesh as the default
            # in the future.
            self.mesh = self._create_2d_mesh()

        logger.info(f"Init mesh | mesh={self.mesh}")

    def _create_new_model_mesh(self) -> jax.sharding.Mesh:
        num_slices = envs.NUM_SLICES

        logger.info(f"Creating new model mesh | devices={len(self.devices)}, "
                    f"num_slices={num_slices}")

        if num_slices == 1:
            devices_array = self._create_single_slice_mesh()
        else:
            devices_array = self._create_multi_slice_mesh(num_slices)

        return jax.sharding.Mesh(devices_array, MESH_AXIS_NAMES)

    def _create_single_slice_mesh(self) -> jax.Array:
        sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
        mesh_shape = (
            sharding_config.model_dp_size,
            sharding_config.attn_dp_size,
            sharding_config.attn_dp_expert_size,
            sharding_config.expert_size,
            sharding_config.tp_size,
            sharding_config.decode_cp_size,
            sharding_config.prefill_cp_size,
        )

        # Attempt to create a physically optimized mesh. Fall back to a simple
        # logical reshape for non-power-of-two device counts (e.g., DP=6) to
        # bypass strict physical topology constraints.
        try:
            return mesh_utils.create_device_mesh(
                mesh_shape,
                self.devices,
                allow_split_physical_axes=True,
            )
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning(
                "Physical mesh creation failed (shape=%s, devices=%d). "
                "Falling back to logical reshape. Error: %s", mesh_shape,
                len(self.devices), e)
            return np.array(self.devices).reshape(mesh_shape)

    def _create_multi_slice_mesh(self, num_slices: int) -> jax.Array:
        sharding_config: ShardingConfigManager = self.vllm_config.sharding_config
        dp_inner = sharding_config.model_dp_size // num_slices

        # Splits data parallelism across multiple slices.
        ici_mesh_shape = (
            dp_inner,
            sharding_config.attn_dp_size,
            sharding_config.attn_dp_expert_size,
            sharding_config.expert_size,
            sharding_config.tp_size,
            sharding_config.decode_cp_size,
            sharding_config.prefill_cp_size,
        )
        dcn_mesh_shape = (num_slices, 1, 1, 1, 1, 1, 1)

        # Attempt to create a physically optimized hybrid mesh (ICI + DCN).
        # Fall back to a logical reshape for non-power-of-two device counts
        # to bypass strict hardware topology constraints across slices.
        try:
            return mesh_utils.create_hybrid_device_mesh(
                mesh_shape=ici_mesh_shape,
                dcn_mesh_shape=dcn_mesh_shape,
                devices=self.devices,
                allow_split_physical_axes=True,
            )
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning(
                "Hybrid physical mesh creation failed. Falling back to logical reshape. "
                "ICI shape: %s, DCN shape: %s, Error: %s", ici_mesh_shape,
                dcn_mesh_shape, e)
            return np.array(self.devices).reshape(
                tuple(i * d for i, d in zip(ici_mesh_shape, dcn_mesh_shape)))

    def _create_2d_mesh(self) -> jax.sharding.Mesh:

        sharding_strategy: ShardingConfigManager = self.vllm_config.sharding_config
        mesh_shape = (
            sharding_strategy.model_dp_size,
            sharding_strategy.tp_size,
        )

        enforce_device_order = (
            self.vllm_config.sharding_config.device_indexes is not None
            and len(self.vllm_config.sharding_config.device_indexes) > 0)

        if enforce_device_order:
            axis_types = (mesh_lib.AxisType.Auto, ) * len(mesh_shape)
            return jax.make_mesh(mesh_shape,
                                 MESH_AXIS_NAMES_2D,
                                 axis_types,
                                 devices=self.devices)
        else:
            return make_optimized_mesh(mesh_shape,
                                       MESH_AXIS_NAMES_2D,
                                       devices=self.devices)

    def _init_phased_profiling(self) -> None:
        self.phased_profiling_dir = envs.PHASED_PROFILING_DIR
        self.phase_based_profiler = None
        if self.phased_profiling_dir:
            # Under MPMD each DP rank runs its own profiler on the same host
            # and produces an identically-named xplane.pb (same hostname,
            # same JAX worker id). Without a per-rank segment they collide on
            # the second-resolution timestamp dir and one rank's capture
            # overwrites another's.
            dp_rank = self.parallel_config.data_parallel_index if envs.TPU_MULTIPROCESS_DP else 0
            self.phase_based_profiler = runner_utils.PhasedBasedProfiler(
                self.phased_profiling_dir, worker_rank=dp_rank)

    def _init_aggregated_stats_logging(self) -> None:
        self.aggregated_stats_dir = envs.AGGREGATED_STATS_DIR
        self.aggregated_stats_logger = None
        # Only enable stats aggregation on one worker to avoid duplicate records
        if self.aggregated_stats_dir and self.rank == 0:
            self.aggregated_stats_logger = runner_utils.AggregatedStatsLogger(
                self.aggregated_stats_dir)

    def _init_mm(self) -> None:
        self.is_multimodal_model = None
        self.uses_mrope = self.model_config.uses_mrope
        self.supports_mm_inputs = True

    def _init_speculative_decoding(self) -> None:
        self.drafter = None
        if self.speculative_config:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = Eagle3Proposer(self.vllm_config, self)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

    def _init_inputs(self) -> None:
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        # The total number of requests is dp_size * max_num_seqs
        self.max_num_reqs = max(self.dp_size * scheduler_config.max_num_seqs,
                                MIN_NUM_SEQS)

        additional_sizes = self.vllm_config.additional_config.get(
            "compilation_sizes", [])
        # [16, 32, 64, 128, 256, 512, 1024, 2048]
        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype == "auto":
            cache_dtype = self.dtype
        kv_cache_dtype = to_jax_dtype(cache_dtype)
        kv_packing = common_utils.get_dtype_packing(kv_cache_dtype)
        self.num_tokens_paddings = runner_utils.get_token_paddings(
            min_token_size=max(16, next_power_of_2(self.dp_size * kv_packing)),
            max_token_size=scheduler_config.max_num_batched_tokens *
            self.dp_size,
            padding_gap=vllm_envs.VLLM_TPU_BUCKET_PADDING_GAP)
        self.num_tokens_paddings = sorted(self.num_tokens_paddings +
                                          additional_sizes)
        self.num_tokens_paddings_per_dp = [
            padding // self.dp_size for padding in self.num_tokens_paddings
        ]
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        self.requests: dict[str, CachedRequestState] = {}
        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, jax.Array] = {}

        # Use parallel_config as the primary source of truth during initialization
        # to avoid AttributeError when self.mesh is not yet assigned (common in unit tests).
        tp_size = 1
        if self.vllm_config.parallel_config is not None:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        elif hasattr(self, 'mesh') and self.mesh is not None:
            tp_size = self.mesh.shape.get(ShardingAxisName.MODEL, 1)

        self.vocab_size = common_utils.align_to(model_config.get_vocab_size(),
                                                tp_size)
        num_speculative_tokens = 0
        if self.vllm_config.speculative_config:
            num_speculative_tokens = self.vllm_config.speculative_config.num_speculative_tokens

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            pin_memory=False,
            vocab_size=self.vocab_size,
            block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            num_speculative_tokens=num_speculative_tokens,
            dp_size=self.dp_size,
        )

        self.positions_cpu = np.zeros(self.max_num_tokens, dtype=np.int32)
        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_cpu = np.arange(self.max_num_tokens, dtype=np.int64)
        min_num_reqs = max(MIN_NUM_SEQS, next_power_of_2(self.dp_size))
        self.num_reqs_paddings = runner_utils.get_req_paddings(
            min_req_size=min_num_reqs, max_req_size=self.max_num_reqs)

        # The num_reqs paddings for attention only, by default it is
        # the max_reqs. If ATTN_BUCKETIZE_NUM_REQS=true, it is the
        # power-of-two between min and max reqs.
        # User can set ATTN_CUSTOM_NUM_REQS_BUCKETS to provide custom buckets.
        self.attn_num_reqs_paddings_per_dp = runner_utils.get_attn_req_paddings(
            min_req_size=MIN_NUM_SEQS,
            max_req_size=scheduler_config.max_num_seqs)
        self.attn_num_reqs_paddings = [
            padding * self.dp_size
            for padding in self.attn_num_reqs_paddings_per_dp
        ]

        self.num_reqs_paddings_per_dp = [
            padding // self.dp_size for padding in self.num_reqs_paddings
        ]

        # Padding for logits. Without speculative decoding, each request has one position to select from.
        # With speculative decoding, each request has multiple positions to select from.
        max_logits_per_req = 1
        if self.speculative_config:
            max_logits_per_req = self.speculative_config.num_speculative_tokens + 1  # Including bonus token
            self.num_logits_paddings = runner_utils.get_token_paddings(
                min_token_size=MIN_NUM_SEQS,
                max_token_size=self.max_num_reqs * max_logits_per_req,
                padding_gap=0)
        else:
            self.num_logits_paddings = None

        # tensors for structured decoding
        self.grammar_bitmask_cpu = np.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=np.int32,
        )
        self.require_structured_out_cpu = np.zeros(
            (self.max_num_reqs, 1),
            dtype=np.bool_,
        )
        self.structured_decode_arange = np.arange(0, 32, dtype=np.int32)

        # multi-modal support
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)

        # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
        # the modality of inputs. For text-only inputs, each dimension has
        # identical position IDs, making M-RoPE functionally equivalent to
        # 1D-RoPE.
        # See page 5 of https://arxiv.org/abs/2409.12191
        self.mrope_positions_cpu = np.zeros((3, self.max_num_tokens),
                                            dtype=np.int64)

        # Contiguous buffer for metadata array. By using a single buffer, we are
        # able to avoid the overhead of multiple device_put operations.
        # Initialize to a constant size, and resize later after kv cache size is
        # known.
        self.device_buffer = common_utils.DeviceBuffer(initial_capacity=1024)

    def load_model(self):
        with set_current_vllm_config(self.vllm_config):
            model = get_model(
                self.vllm_config,
                self.rng_key,
                self.mesh,
            )

            # Only pre-warm torchax for pooling tasks to prevent JIT during inference
            # while avoiding unnecessary overhead for standard generative models.
            if self.is_pooling_model:
                import torchax
                from torchax.interop import torch_view

                h_size = self.model_config.get_hidden_size()
                # Use the actual model mesh and ATTN_DATA axis for sharding alignment
                pooling_sharding = NamedSharding(
                    self.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))

                logger.info(
                    f"Pre-warming StepPooler for shapes: {self.num_tokens_paddings}"
                )
                with torchax.default_env():
                    for max_tokens in self.num_tokens_paddings:
                        # Create a dummy JAX array with exact shape and sharding metadata
                        dummy_jax = jnp.zeros((max_tokens, h_size),
                                              dtype=jnp.bfloat16)
                        # Shard the array to match real-time hidden_states
                        dummy_jax = jax.device_put(dummy_jax, pooling_sharding)

                        # Trigger the casting and host-transfer kernels
                        # Using non_blocking=False to ensure AOT compilation completes during load
                        _ = torch_view(dummy_jax).to('cpu', non_blocking=False)
                logger.debug("Universal StepPooler pre-warming successful.")

        self.model_fn = model.model_fn
        self.compute_logits_fn = model.compute_logits_fn
        self.pooler_fn = model.pooler_fn
        self.combine_hidden_states_fn = model.combine_hidden_states_fn
        self.state = model.state
        # For the flax_nnx path, `model_fn` (== `run_model`) accepts a flat
        # tuple of array leaves and reconstructs the nnx.State inside the
        # jit. Pre-flatten here so subsequent dispatches skip the per-call
        # walk of `nnx.Variable` wrappers
        self.state_leaves = model.state_leaves
        self.lora_manager = model.lora_manager
        self.model = model.model

        self.precompile_vision_encoder_fn = model.multimodal_fns.precompile_vision_encoder_fn
        self.embed_multimodal_fn = model.multimodal_fns.embed_multimodal_fn
        self.embed_input_ids_fn = model.multimodal_fns.embed_input_ids_fn
        self.get_mrope_input_positions_fn = model.multimodal_fns.get_mrope_input_positions_fn

        if self.drafter is not None:
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.state)

        rng_key = nnx.Rngs(jax.random.key(self.model_config.seed)).params()
        self.rng_params_for_sampling = device_array(self.mesh,
                                                    rng_key,
                                                    sharding=NamedSharding(
                                                        self.mesh,
                                                        PartitionSpec()))
        # This allows a multi-modal model to be used as text-only, assuming the user
        # passes the following to vLLM (on the CLI):
        # --limit-mm-per-prompt '{"image": 0, "video": 0}'
        disable_mm_from_limits = False
        if self.model_config.is_multimodal_model:
            mm_limits = self.model_config.multimodal_config.limit_per_prompt
            # According to https://github.com/vllm-project/vllm/blob/21d2b53f88d99f9ab369444f6d53ed2b9c260e4f/vllm/config/multimodal.py#L79-L95
            # if a modality limit is missing, we should treat count as 999. So here we disable multi-modality only when all limits are set to 0.
            if mm_limits and all(limit.count == 0
                                 for limit in mm_limits.values()):
                disable_mm_from_limits = True

            if disable_mm_from_limits:
                logger.warning(
                    f"Disabling multi-modality for model because limits are set to 0. {mm_limits=}"
                )

        self.is_multimodal_model = (self.model_config.is_multimodal_model
                                    and self.embed_multimodal_fn is not None
                                    and hasattr(self.model_config.hf_config,
                                                "architectures")
                                    and not disable_mm_from_limits)

        # Clear JIT compilation caches from weight loading to free XLA
        # program reservations (bytes_reserved) on TPU HBM.
        jax.clear_caches()
        logger.info("Cleared JIT caches after weight loading")

        logger.info(f"Init model | "
                    f"hbm={common_utils.hbm_usage_gb(self.devices)}GiB")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        runner_type = self.model_config.runner_type
        if runner_type == "generate":
            return ("generate", )
        if runner_type == "pooling":
            return ("embed", )
        assert False, f"unsupported runner type: {runner_type}"

    def get_kv_cache_spec(self):
        return self.kv_cache_manager.get_kv_cache_spec()

    def get_kv_cache_layout(self):
        return self.kv_cache_manager.get_kv_cache_layout()

    def initialize_kv_cache(self,
                            kv_cache_config: KVCacheConfig,
                            topology_order_id: int = 0) -> None:
        self.topology_order_id = topology_order_id
        self.kv_cache_config = kv_cache_config
        self.use_hybrid_kvcache = len(kv_cache_config.kv_cache_groups) > 1
        self.kv_cache_manager.initialize_kv_cache(kv_cache_config)

        if self.kv_cache_manager.actual_mamba_num_blocks is not None:
            self.input_batch.init_mamba_pools(
                self.kv_cache_manager.actual_mamba_num_blocks)

        # This buffer grows dynamically to accommodate metadata and block tables.
        # We re-initialize with a precise capacity now that kv_cache_config is known.
        num_kv_groups = len(kv_cache_config.kv_cache_groups)
        sampling_params_size = 3 * self.max_num_reqs
        input_ids_size = self.max_num_tokens
        query_start_loc_size = self.max_num_reqs + self.dp_size
        seq_lens_size = self.max_num_reqs
        logits_indices_size = self.max_num_reqs
        # Block tables for each KV cache group.
        block_tables_size = num_kv_groups * self.max_num_reqs * self.max_num_blocks_per_req

        initial_capacity = (sampling_params_size + input_ids_size +
                            query_start_loc_size + seq_lens_size +
                            logits_indices_size + block_tables_size)
        self.device_buffer = common_utils.DeviceBuffer(
            initial_capacity=initial_capacity)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_runner(self)

    def delete_kv_cache(self) -> None:
        self.kv_cache_manager.delete_kv_cache()

    def reinitialize_kv_cache(self) -> None:
        self.kv_cache_manager.reinitialize_kv_cache()

    def capture_model(self) -> None:
        self.compilation_manager.capture_model()

    @time_function
    def execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
    ) -> ModelRunnerOutput | JaxIntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        reqs = self.input_batch.num_reqs
        toks = scheduler_output.total_num_scheduled_tokens
        with jax.set_mesh(self.mesh), jax.profiler.TraceAnnotation(
                f"execute_model: {reqs} reqs, {toks} toks"):
            output = self._execute_model(scheduler_output,
                                         intermediate_tensors)
        return output

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            # This can happen in pipeline parallel case.
            return EMPTY_MODEL_RUNNER_OUTPUT

        (scheduler_output, attn_metadata, sampling_metadata, input_ids,
         hidden_states, logits, aux_hidden_states, spec_decode_metadata,
         kv_connector_output, logits_indices_selector, padded_num_reqs,
         expert_indices, full_hidden_states) = (
             self.execute_model_state.scheduler_output,
             self.execute_model_state.attn_metadata,
             self.execute_model_state.sampling_metadata,
             self.execute_model_state.input_ids,
             self.execute_model_state.hidden_states,
             self.execute_model_state.logits,
             self.execute_model_state.aux_hidden_states,
             self.execute_model_state.spec_decode_metadata,
             self.execute_model_state.kv_connector_output,
             self.execute_model_state.logits_indices_selector,
             self.execute_model_state.padded_num_reqs,
             self.execute_model_state.expert_indices,
             self.execute_model_state.full_hidden_states)
        self.execute_model_state = None

        if grammar_output is not None:
            (
                require_struct_decoding, grammar_bitmask_padded, arange
            ) = self.structured_decoding_manager.prepare_structured_decoding_input(
                logits, grammar_output)
            logits = self.structured_decoding_manager.structured_decode_fn(
                require_struct_decoding,
                grammar_bitmask_padded,
                logits,
                arange,
            )
        return self._sample_from_logits(
            scheduler_output, attn_metadata, sampling_metadata, input_ids,
            hidden_states, logits, aux_hidden_states, spec_decode_metadata,
            kv_connector_output, logits_indices_selector, padded_num_reqs,
            expert_indices, full_hidden_states)

    def _modify_prev_results(self):
        # If copy to host has not been done, we just wait.
        # device_get should return immediately as we have scheduled it in previous function call.
        assert self._pre_async_results is not None, "When we call _modify_prev_results(), self._pre_async_results should already exist"
        pre_req_ids = self._pre_async_results.req_ids
        pre_num_reqs = len(pre_req_ids)
        pre_next_tokens = self._pre_async_results.next_tokens
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = self._pre_async_results.discard_sampled_tokens_req_indices
        pre_logits_indices_selector = self._pre_async_results.logits_indices_selector
        pre_spec_decode_metadata = self._pre_async_results.spec_decode_metadata

        valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
            self, pre_spec_decode_metadata, pre_next_tokens,
            pre_logits_indices_selector,
            pre_discard_sampled_tokens_req_indices, pre_num_reqs)

        # Append sampled tokens
        for pre_req_idx, req_state, _ in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if not sampled_ids:
                continue

            # If request not active in the *current* batch (e.g. finished or evicted), skip it.
            req_id = pre_req_ids[pre_req_idx]
            if req_id not in self.input_batch.req_id_to_index:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]
            assert req_state is self.requests[
                req_id], "The req_state should be valid and identical"

            # Updated on previous execute
            pre_num_placeholder_tokens = 1
            if pre_spec_decode_metadata is not None:
                pre_num_placeholder_tokens += pre_spec_decode_metadata.draft_lengths_cpu[
                    pre_req_idx]
            end_idx = self.input_batch.num_tokens_no_spec[req_idx]
            num_sampled_tokens = len(sampled_ids)
            assert num_sampled_tokens <= pre_num_placeholder_tokens
            start_idx = end_idx - pre_num_placeholder_tokens
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx, start_idx:start_idx +
                                           num_sampled_tokens] = sampled_ids
            self.input_batch.num_tokens_no_spec[
                req_idx] = start_idx + num_sampled_tokens
            self.input_batch.num_tokens[
                req_idx] = start_idx + num_sampled_tokens
            # Replace previous placeholder
            for j in range(pre_num_placeholder_tokens):
                req_state.output_token_ids.pop()
            req_state.output_token_ids.extend(sampled_ids)

    def _update_placeholder(self,
                            discard_sampled_tokens_req_indices,
                            request_seq_lens,
                            spec_decode_metadata,
                            logits_indices_selector=None):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_sampled_tokens_req_indices_set = set(
            discard_sampled_tokens_req_indices)
        for req_idx, req_state, _ in request_seq_lens:
            if req_idx in discard_sampled_tokens_req_indices_set:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + 1
            if spec_decode_metadata is not None:
                end_idx += spec_decode_metadata.draft_lengths_cpu[req_idx]
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            # Update cpu tokens at next execute and prepare input from tpu
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            # For placeholder, should be update on next execute.
            req_state.output_token_ids.extend([0] * (end_idx - start_idx))
            if logits_indices_selector is None:
                placeholder_req_id_to_index[req_state.req_id] = req_idx
            else:
                placeholder_req_id_to_index[
                    req_state.req_id] = logits_indices_selector[req_idx]
        return placeholder_req_id_to_index

    def _execute_model(
        self,
        scheduler_output: "VllmSchedulerOutput",
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
    ) -> JaxIntermediateTensors | ModelRunnerOutput | None:
        self.persistent_batch_manager.update_states(
            scheduler_output, self.get_mrope_input_positions_fn)
        if not scheduler_output.total_num_scheduled_tokens:
            if has_kv_transfer_group():
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            # Return empty ModelRunnerOutput if there's no work to do.
            # TODO(fhzhang): We rely on empty cycles to remove requests in input batch. Fix it to reduce overhead.
            logger.debug(f"Nothing scheduled: {scheduler_output}!")
            # NOTE(pooyam): There is no guarantee that scheduler is not sending empty output: https://github.com/vllm-project/vllm/blob/7cfea0df390c154c1026f77d3682e2733ca4aca8/vllm/v1/engine/core.py#L275
            # Why they are not preventing that is not clear to me.
            if len(scheduler_output.finished_req_ids) == 0:
                logger.warning(
                    "Should not schedule a request that does nothing!")
                # raise Exception(
                #     "Should not schedule a request that does nothing!")
            return EMPTY_MODEL_RUNNER_OUTPUT

        # TODO(pooyam): I guess we can remove returning sampling_metadata in `_prepare_inputs` after https://github.com/njhill/vllm/commit/b7433ca1a47732394b1bdea4099d98389515954b
        (
            input_ids,
            input_positions,
            attn_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
            req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank,
        ) = self._prepare_inputs(scheduler_output)

        # multi-modal support
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            # We have the modality embeds at this time.
            self.mm_manager.execute_mm_encoder(scheduler_output)
            mm_embeds, is_mm_embed = self.mm_manager.gather_mm_embeddings(
                scheduler_output, input_ids.shape[0], req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank)
        else:
            mm_embeds, is_mm_embed = None, None

        # NOTE(Wenlong): For multi-modal model,
        # it will embed the text tokens and merge with the existing modality embeds
        # Later, the multi-modality model will take the embedding as the input.
        # For text-only model, this does nothing. It will input the input_ids and
        # leave the embedding job inside the forward pass
        input_ids, inputs_embeds = self._get_input_ids_embeds(
            input_ids, mm_embeds, is_mm_embed)

        lora_metadata = self.lora_utils.extract_lora_metadata()
        # TODO: make _get_input_ids_embeds within this context
        # NOTE: right now, mm model will use embeddings as the input,
        # but text-only model will use input_ids
        with self.maybe_forbid_compile:

            with set_forward_context(
                    None,
                    self.vllm_config,
            ), self.maybe_get_kv_connector_output(
                    scheduler_output) as kv_connector_output:
                # NOTE(Wenlong): It takes both `input_ids` and `inputs_embeds`,
                # but one of them would be `None`
                (self.kv_caches, hidden_states, aux_hidden_states,
                 expert_indices) = self.model_fn(
                     self.state_leaves,
                     self.kv_caches,
                     input_ids,
                     attn_metadata,
                     inputs_embeds,
                     input_positions,
                     tuple(self.layer_name_to_kvcache_index.items()),
                     lora_metadata,
                     intermediate_tensors,
                     self.is_first_rank,
                     self.is_last_rank,
                 )
            if not self.is_last_rank:
                assert isinstance(hidden_states, JaxIntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                hidden_states.expert_indices = expert_indices
                return hidden_states

        if self.is_pooling_model:
            num_reqs = self.input_batch.num_reqs

            # Retrieve sequence lengths
            seq_lens_view = self.device_buffer.get_view((self.max_num_reqs, ),
                                                        key="seq_lens")
            seq_lens = seq_lens_view[:num_reqs]

            pooling_metadata = self.input_batch.get_pooling_metadata()

            # Extract scheduled token counts for the current chunk
            num_scheduled_tokens = np.array([
                scheduler_output.num_scheduled_tokens[req_id]
                for req_id in self.input_batch.req_ids[:num_reqs]
            ],
                                            dtype=np.int32)

            # Call the pooler with the decoupled interface
            pooler_output = self.pooler_fn(
                hidden_states,
                pooling_metadata,
                seq_lens,
                num_scheduled_tokens,
            )

            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=pooler_output,
            )

        full_hidden_states = hidden_states
        if _logits_indices_require_global_gather(self.vllm_config,
                                                 attn_metadata):
            hidden_states = hidden_states[logits_indices]
        else:
            hidden_states = self._select_from_array_fn(hidden_states,
                                                       logits_indices)
        logits = self.compute_logits_fn(
            self.state_leaves,
            hidden_states,
            lora_metadata,
        )

        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            attn_metadata=attn_metadata,
            sampling_metadata=sampling_metadata,
            input_ids=input_ids,
            hidden_states=hidden_states,
            logits=logits,
            aux_hidden_states=aux_hidden_states,
            spec_decode_metadata=spec_decode_metadata,
            kv_connector_output=kv_connector_output,
            logits_indices_selector=logits_indices_selector,
            padded_num_reqs=padded_num_reqs,
            expert_indices=expert_indices,
            full_hidden_states=full_hidden_states)
        return None

    def _sample_from_logits(
        self,
        scheduler_output: "VllmSchedulerOutput",
        attn_metadata: AttentionMetadata,
        tpu_sampling_metadata: TPUSupportedSamplingMetadata,
        input_ids: Optional[jax.Array],
        hidden_states: jax.Array,
        logits: jax.Array,
        aux_hidden_states: Optional[jax.Array],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        kv_connector_output: Optional[KVConnectorOutput],
        logits_indices_selector: Optional[List[int]] = None,
        padded_num_reqs: Optional[int] = None,
        expert_indices: Optional[jax.Array] = None,
        full_hidden_states: Optional[jax.Array] = None,
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if padded_num_reqs is None:
            padded_num_reqs = runner_utils.get_padded_num_reqs_with_upper_limit(
                self.input_batch.num_reqs, self.max_num_reqs)

        if tpu_sampling_metadata.do_sampling:
            self.rng_params_for_sampling, step_rng = jax.random.split(
                self.rng_params_for_sampling)
        else:
            step_rng = self.rng_params_for_sampling

        if spec_decode_metadata is None:
            logits = logits.astype(jnp.float32)
            with self.maybe_forbid_compile:
                next_tokens, processed_logits = sample(
                    step_rng,
                    self.mesh,
                    logits,
                    tpu_sampling_metadata,
                )
        else:
            # TODO(gxd3): wrap the spec decode sampling code block
            # under maybe_forbid_compile as well.
            # Currently when spec-decoding is enabled, serving-time
            # jit-recompile might still happen.
            if tpu_sampling_metadata.do_sampling:
                bonus_rng, rejection_rng = jax.random.split(step_rng)
            else:
                bonus_rng = step_rng
                rejection_rng = step_rng
            bonus_logits = self._select_from_array_fn(
                logits, spec_decode_metadata.bonus_logits_indices)
            bonus_token_ids, _ = sample(
                bonus_rng,
                self.mesh,
                bonus_logits,
                tpu_sampling_metadata,
            )
            target_logits = self._select_from_array_fn(
                logits, spec_decode_metadata.target_logits_indices)
            next_tokens = self.rejection_sampler(
                draft_token_ids=spec_decode_metadata.draft_token_ids,
                num_draft_tokens=spec_decode_metadata.draft_lengths,
                draft_probs=None,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                sampling_metadata=tpu_sampling_metadata,
                key=rejection_rng,
            )

        logits = logits.astype(jnp.float32)
        with self.maybe_forbid_compile:

            if tpu_sampling_metadata.logprobs:
                logits = processed_logits if self.model_config.logprobs_mode == "processed_logprobs" else logits
                logprobs = self._compute_and_gather_logprobs(
                    logits, next_tokens, self.model_config.max_logprobs)
                logprobs = _jax_logprobs_copy_to_host_async(logprobs)
            else:
                logprobs = None

        num_reqs = self.input_batch.num_reqs

        # Update the cache state concurrently. Code above will not block until
        # We use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        spec_decode_last_sampled_token_id = None
        spec_decode_num_rejected_tokens = None
        if self.speculative_config:
            with self.maybe_forbid_compile, jax.set_mesh(self.mesh):
                last_sampled_token_id, num_rejected_tokens = extract_last_sampled_tokens(
                    spec_decode_metadata, next_tokens,
                    self.speculative_config.num_speculative_tokens,
                    self.input_batch.vocab_size, self.max_num_reqs)
                self.speculative_decoding_manager.propose_draft_token_ids(
                    next_tokens,
                    logits_indices_selector,
                    last_sampled_token_id,
                    num_rejected_tokens,
                    discard_sampled_tokens_req_indices,
                    aux_hidden_states,
                    attn_metadata,
                    bool(self.scheduler_config.async_scheduling),
                    spec_decode_metadata,
                    scheduler_output,
                    input_ids,
                    full_hidden_states,
                )
                spec_decode_last_sampled_token_id = last_sampled_token_id
                spec_decode_num_rejected_tokens = num_rejected_tokens

        # If async scheduler enabled
        if self.scheduler_config.async_scheduling:
            # Get previous results from TPU and replace the placeholder.
            if self._pre_async_results is not None:
                self._modify_prev_results()

            # Set placeholder for next tokens that is not yet generated
            placeholder_req_id_to_index: dict[
                str, int] = self._update_placeholder(
                    discard_sampled_tokens_req_indices, request_seq_lens,
                    spec_decode_metadata, logits_indices_selector)

            spec_decode_next_tokens = None
            if self.speculative_config:
                assert spec_decode_last_sampled_token_id is not None
                with self.maybe_forbid_compile, jax.set_mesh(self.mesh):
                    spec_decode_next_tokens = concat_last_sampled_tokens_and_draft_tokens(
                        spec_decode_last_sampled_token_id,
                        self.speculative_decoding_manager._draft_token_ids)

            # Save the previous results
            next_tokens = jax.copy_to_host_async(next_tokens)
            self._pre_async_results = AsyncPreResults(
                req_ids=req_ids,
                next_tokens=next_tokens,
                request_seq_lens=request_seq_lens,
                discard_sampled_tokens_req_indices=
                discard_sampled_tokens_req_indices,
                placeholder_req_id_to_index=placeholder_req_id_to_index,
                logits_indices_selector=logits_indices_selector,
                spec_decode_next_tokens=spec_decode_next_tokens,
                spec_decode_num_rejected_tokens=spec_decode_num_rejected_tokens,
                spec_decode_metadata=spec_decode_metadata,
            )

            # Return Model output to executor
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index=self.input_batch.req_id_to_index.copy(),
                sampled_token_ids=[],  # Fill in async get
                logprobs=None,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                kv_connector_output=kv_connector_output,
            )
            # Return async_model_runner_output
            async_model_runner_output = AsyncTPUModelRunnerOutput(
                model_runner_output,
                next_tokens,
                num_reqs,
                discard_sampled_tokens_req_indices,
                logits_indices_selector,
                logprobs_tensors=logprobs,
                expert_indices=expert_indices,
                total_num_scheduled_tokens=scheduler_output.
                total_num_scheduled_tokens,
                spec_decode_metadata=spec_decode_metadata,
                runner=self)
            return async_model_runner_output

        valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
            self, spec_decode_metadata, next_tokens, logits_indices_selector,
            discard_sampled_tokens_req_indices, num_reqs)

        # Append sampled tokens
        for req_idx, req_state, _ in request_seq_lens:
            sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_state.output_token_ids.extend(sampled_ids)

        if logprobs is not None:
            # Use materialize to ensure logprobs are ready on host when we return async results
            logprobs_lists = _jax_logprobs_materialize(
                logprobs, logits_indices_selector)
        else:
            logprobs_lists = None

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

        if expert_indices is not None:
            expert_indices_cpu = np.asarray(jax.device_get(expert_indices))

            routed_experts_dict = {}
            current_token_offset = 0
            for req_id in self.input_batch.req_ids[:num_reqs]:
                req_state = self.requests[req_id]
                num_tokens_scheduled = scheduler_output.num_scheduled_tokens[
                    req_id]
                start_idx = current_token_offset
                end_idx = start_idx + num_tokens_scheduled
                current_token_offset = end_idx

                # Shape: (num_tokens_scheduled, num_layers, top_k)
                step_experts = expert_indices_cpu[:, start_idx:
                                                  end_idx, :].transpose(
                                                      1, 0, 2)

                if not hasattr(req_state, "_routed_experts_buf"):
                    _, layers, top_k = step_experts.shape
                    req_state._routed_experts_buf = np.zeros(
                        (self.max_model_len, layers, top_k),
                        dtype=step_experts.dtype)
                    req_state._routed_experts_len = 0

                offset = req_state._routed_experts_len
                allowed = min(num_tokens_scheduled,
                              self.max_model_len - offset)
                if allowed > 0:
                    req_state._routed_experts_buf[offset:offset + allowed] = (
                        step_experts[:allowed])
                    req_state._routed_experts_len += allowed

                routed_experts_dict[
                    req_id] = req_state._routed_experts_buf[:req_state.
                                                            _routed_experts_len]

            model_runner_output.routed_experts_dict = routed_experts_dict

        return model_runner_output

    @jax.jit(static_argnums=(0, ))
    def _select_from_array_fn(self, array, indices_to_select):

        def select_local_fn(local_array, local_indices):
            return local_array[local_indices]

        ret = jax.shard_map(
            select_local_fn,
            mesh=self.mesh,
            in_specs=(PartitionSpec(ShardingAxisName.ATTN_DATA),
                      PartitionSpec(ShardingAxisName.ATTN_DATA)),
            out_specs=PartitionSpec(ShardingAxisName.ATTN_DATA))(
                array, indices_to_select)

        return ret

    @staticmethod
    @jax.jit(static_argnames=("max_logprobs", ))
    def _compute_and_gather_logprobs(logits, next_tokens, max_logprobs):
        logprobs = compute_logprobs(logits)
        return gather_logprobs(logprobs, next_tokens, max_logprobs)

    def _prepare_input_metadata(self,
                                scheduler_output: "VllmSchedulerOutput",
                                use_pcp: bool | None = None):

        dp_size = self.dp_size
        num_reqs = self.input_batch.num_reqs
        max_num_reqs_per_dp_rank = self.max_num_reqs // dp_size
        req_ids_dp = {dp_rank: [] for dp_rank in range(dp_size)}
        req_indices_dp = {dp_rank: [] for dp_rank in range(dp_size)}
        num_scheduled_tokens_per_dp_rank = {
            dp_rank: 0
            for dp_rank in range(dp_size)
        }
        scheduled_tokens_per_dp_rank = {
            dp_rank: []
            for dp_rank in range(dp_size)
        }
        num_req_per_dp_rank = {dp_rank: 0 for dp_rank in range(dp_size)}

        for req_id in self.input_batch.req_ids[:num_reqs]:
            dp_rank = (scheduler_output.assigned_dp_rank[req_id]
                       if dp_size > 1 else 0)
            req_ids_dp[dp_rank].append(req_id)
            req_indices_dp[dp_rank].append(
                self.input_batch.req_id_to_index[req_id])
            num_scheduled_tokens_per_dp_rank[
                dp_rank] += scheduler_output.num_scheduled_tokens[req_id]
            scheduled_tokens_per_dp_rank[dp_rank].append(
                scheduler_output.num_scheduled_tokens[req_id])
            num_req_per_dp_rank[dp_rank] += 1

        # Find maximum number of scheduled tokens across DP ranks
        max_num_scheduled_tokens_across_dp = max(
            num_scheduled_tokens_per_dp_rank.values())
        pcp_size, interleave_size = _get_pcp_parallel_config(self.vllm_config)
        if use_pcp is None:
            use_pcp = _batch_uses_pcp_prefill(self.vllm_config,
                                              self.input_batch,
                                              scheduler_output, num_reqs)
        if use_pcp and interleave_size <= 0:
            raise ValueError(
                "PCP runner path requires cp_kv_cache_interleave_size > 0.")
        if use_pcp:
            max_local_tokens_across_pcp = 0
            for dp_rank in range(dp_size):
                req_indices = np.asarray(req_indices_dp[dp_rank],
                                         dtype=np.int64)
                token_start_offsets = (
                    self.input_batch.num_computed_tokens_cpu[req_indices]
                    if req_indices.size else np.array([], dtype=np.int32))
                local_counts = _pcp_local_token_counts(
                    scheduled_tokens_per_dp_rank[dp_rank],
                    pcp_size,
                    interleave_size,
                    token_start_offsets_per_req=token_start_offsets,
                )
                max_local_tokens_across_pcp = max(max_local_tokens_across_pcp,
                                                  int(local_counts.max()))
            max_num_scheduled_tokens_across_dp = max(
                max_num_scheduled_tokens_across_dp,
                max_local_tokens_across_pcp * pcp_size)

        padded_num_scheduled_tokens_per_dp_rank = runner_utils.get_padded_token_len(
            self.num_tokens_paddings_per_dp,
            max_num_scheduled_tokens_across_dp)
        if use_pcp:
            padded_num_scheduled_tokens_per_dp_rank = common_utils.align_to(
                padded_num_scheduled_tokens_per_dp_rank, pcp_size)

        padded_total_num_scheduled_tokens = (
            padded_num_scheduled_tokens_per_dp_rank * dp_size)

        assert max_num_scheduled_tokens_across_dp > 0

        # Find maximum number of requests across DP ranks
        max_num_reqs_across_dp = max(
            len(req_ids) for req_ids in req_ids_dp.values())
        padded_num_reqs_per_dp_rank = runner_utils.get_padded_token_len(
            self.num_reqs_paddings_per_dp, max_num_reqs_across_dp)
        padded_num_reqs = padded_num_reqs_per_dp_rank * dp_size
        attn_padded_num_reqs = runner_utils.get_padded_token_len(
            self.attn_num_reqs_paddings_per_dp,
            max_num_reqs_across_dp) * dp_size

        # logits_indices_selector reorders per-rank outputs back to the
        # original batch ordering; with a single rank the ordering is already
        # the input-batch ordering, so no selector is needed.
        if dp_size > 1:
            all_req_indices = np.concatenate(
                [req_indices_dp[dp_rank] for dp_rank in range(dp_size)])
            all_positions = np.concatenate([
                np.arange(len(req_indices_dp[dp_rank])) +
                padded_num_reqs_per_dp_rank * dp_rank
                for dp_rank in range(dp_size)
            ])
            sorted_indices = np.argsort(all_req_indices)
            logits_indices_selector = all_positions[sorted_indices]
        else:
            logits_indices_selector = None

        return (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
                scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
                padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
                attn_padded_num_reqs, padded_total_num_scheduled_tokens,
                padded_num_reqs_per_dp_rank, logits_indices_selector,
                max_num_reqs_per_dp_rank)

    def _prepare_async_token_substitution_indices(
            self, req_ids_dp, scheduled_tokens_per_dp_rank,
            padded_num_scheduled_tokens_per_dp_rank,
            num_draft_tokens_per_dp_rank, dp_size):
        """Prepare token substitution indices for async scheduling."""
        # For input_ids substitution.
        token_in_tpu_cur_input_indices_dp = {}
        token_in_tpu_pre_next_tokens_indices_dp = {}
        # For SpecDecodeMetadata.draft_token_ids substitution.
        draft_token_in_tpu_cur_indices_dp = {}
        draft_token_in_prev_next_tokens_indices_dp = {}
        spec_decode_enabled = (self.speculative_config is not None)

        for dp_rank in range(dp_size):
            token_in_tpu_cur_input_indices_dp[dp_rank] = []
            token_in_tpu_pre_next_tokens_indices_dp[dp_rank] = []
            draft_token_in_tpu_cur_indices_dp[dp_rank] = []
            draft_token_in_prev_next_tokens_indices_dp[dp_rank] = []

            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]
            num_draft_tokens = num_draft_tokens_per_dp_rank.get(dp_rank, {})
            token_in_tpu_cur_input_indices_list = token_in_tpu_cur_input_indices_dp[
                dp_rank]
            token_in_tpu_pre_next_tokens_indices_list = token_in_tpu_pre_next_tokens_indices_dp[
                dp_rank]
            draft_token_in_tpu_cur_indices_list = draft_token_in_tpu_cur_indices_dp[
                dp_rank]
            draft_token_in_prev_next_tokens_indices_list = draft_token_in_prev_next_tokens_indices_dp[
                dp_rank]

            token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            acc_cur_len = token_offset
            # TODO(gxd3): support spec-decoding with DP.
            draft_tokens_acc_cur_len = 0

            for i, req_id in enumerate(req_ids_dp[dp_rank]):
                acc_cur_len += num_scheduled_tokens_per_req[i]
                if dp_rank == 0:
                    draft_tokens_acc_cur_len += num_draft_tokens[i]
                if req_id not in self._pre_async_results.placeholder_req_id_to_index:
                    continue

                if not spec_decode_enabled:
                    token_in_tpu_cur_input_indices_list.append(acc_cur_len - 1)
                    token_in_tpu_pre_next_tokens_indices_list.append(
                        self._pre_async_results.
                        placeholder_req_id_to_index[req_id])
                else:
                    max_num_spec_tokens = self.speculative_config.num_speculative_tokens
                    assert num_scheduled_tokens_per_req[
                        i] <= max_num_spec_tokens + 1
                    idx = self._pre_async_results.placeholder_req_id_to_index[
                        req_id]

                    base_offset = acc_cur_len - num_scheduled_tokens_per_req[i]
                    for j in range(num_scheduled_tokens_per_req[i]):
                        token_in_tpu_cur_input_indices_list.append(
                            base_offset + j)
                        token_in_tpu_pre_next_tokens_indices_list.append(
                            idx * (max_num_spec_tokens + 1) + j)

                    draft_base_offset = draft_tokens_acc_cur_len - num_draft_tokens[
                        i]
                    for j in range(num_draft_tokens[i]):
                        draft_token_in_tpu_cur_indices_list.append(
                            draft_base_offset + j)
                        draft_token_in_prev_next_tokens_indices_list.append(
                            idx * (max_num_spec_tokens + 1) + j + 1)

        return token_in_tpu_cur_input_indices_dp, token_in_tpu_pre_next_tokens_indices_dp, draft_token_in_tpu_cur_indices_dp, draft_token_in_prev_next_tokens_indices_dp

    def _apply_async_token_substitution(self, input, next_tokens_in_tpu,
                                        token_in_tpu_cur_input_indices,
                                        token_in_tpu_pre_next_tokens_indices):
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input

        idx_pad_len = len(input) - len(token_in_tpu_cur_input_indices)

        # Pad according to the instructions written inside self._substitute_placeholder_token_fn
        full_range = np.arange(0, len(input), dtype=np.int32)
        missing_values = np.setdiff1d(full_range,
                                      token_in_tpu_cur_input_indices)
        padded_token_in_tpu_cur_input_indices = np.concatenate(
            (token_in_tpu_cur_input_indices, missing_values))

        padded_token_in_tpu_pre_next_tokens_indices = np.pad(
            token_in_tpu_pre_next_tokens_indices, (0, idx_pad_len),
            mode='constant',
            constant_values=-1).astype(np.int32)
        placeholder_num = np.array([len(token_in_tpu_cur_input_indices)
                                    ]).astype(np.int32)

        (padded_token_in_tpu_cur_input_indices,
         padded_token_in_tpu_pre_next_tokens_indices,
         placeholder_num) = device_array(
             self.mesh,
             (padded_token_in_tpu_cur_input_indices,
              padded_token_in_tpu_pre_next_tokens_indices, placeholder_num))

        with self.maybe_forbid_compile:
            input = self._substitute_placeholder_token_fn(
                input, padded_token_in_tpu_cur_input_indices,
                padded_token_in_tpu_pre_next_tokens_indices,
                next_tokens_in_tpu, placeholder_num)
        return input

    def _subtract_num_rejected_tokens(self, seq_lens, positions,
                                      num_scheduled_tokens_per_req):
        """Apply rejection-count subtraction to seq_lens and positions if needed.

        `num_computed_tokens_cpu` was advanced on the host assuming every
        speculatively proposed token from the previous step was accepted. Here
        we subtract the actual rejection counts on TPU for the requests that
        ran spec decoding in the previous step.
        """
        assert self._pre_async_results is not None
        assert self._pre_async_results.spec_decode_num_rejected_tokens is not None
        num_reqs = len(num_scheduled_tokens_per_req)
        seq_lens_subtract_indices = np.full(self.max_num_reqs,
                                            -1,
                                            dtype=np.int32)
        positions_subtract_indices = np.full(positions.size,
                                             -1,
                                             dtype=np.int32)

        acc_cur_len = 0
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            acc_cur_len += num_scheduled_tokens_per_req[i]
            assert req_id is not None
            if req_id not in self._pre_async_results.placeholder_req_id_to_index:
                continue
            idx = self._pre_async_results.placeholder_req_id_to_index[req_id]
            seq_lens_subtract_indices[i] = idx
            base_offset = acc_cur_len - num_scheduled_tokens_per_req[i]
            for j in range(num_scheduled_tokens_per_req[i]):
                positions_subtract_indices[base_offset + j] = idx

        seq_lens_subtract_indices, positions_subtract_indices = device_array(
            self.mesh, (seq_lens_subtract_indices, positions_subtract_indices))

        with self.maybe_forbid_compile:
            seq_lens, positions = _subtract_num_rejected_tokens_fn(
                seq_lens, positions,
                self._pre_async_results.spec_decode_num_rejected_tokens,
                seq_lens_subtract_indices, positions_subtract_indices)

        return seq_lens, positions

    def _prepare_inputs(self, scheduler_output: "VllmSchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        dp_size = self.dp_size
        if self.speculative_config and dp_size > 1:
            assert "Spec decoding not yet support when dp > 1"

        pcp_size, cp_kv_cache_interleave_size = _get_pcp_parallel_config(
            self.vllm_config)
        use_pcp = _batch_uses_pcp_prefill(self.vllm_config, self.input_batch,
                                          scheduler_output, num_reqs)
        use_pcp_decode = _batch_uses_pcp_decode(
            self.vllm_config, self.input_batch, scheduler_output, num_reqs)
        scheduled_counts = [
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in self.input_batch.req_ids[:num_reqs]
        ]
        positive_scheduled_counts = [n for n in scheduled_counts if n > 0]
        if (pcp_size > 1 and positive_scheduled_counts
                and _batch_has_unsupported_pcp_mix(self.input_batch,
                                                   scheduler_output,
                                                   num_reqs)):
            raise NotImplementedError(
                "PCP runner path does not support mixed prompt/decode "
                "batches or prompt/decode boundary-crossing schedules yet.")
        if use_pcp or use_pcp_decode:
            if self.parallel_config.decode_context_parallel_size > 1:
                raise NotImplementedError(
                    "PCP runner path does not support DCP yet.")
            if self.speculative_config is not None:
                raise NotImplementedError(
                    "PCP runner path does not support speculative decoding yet."
                )
            if self.scheduler_config.async_scheduling:
                raise NotImplementedError(
                    "PCP runner path does not support async scheduling yet.")
            if cp_kv_cache_interleave_size <= 0:
                raise ValueError(
                    "PCP runner path requires cp_kv_cache_interleave_size > 0."
                )
            if use_pcp and not envs.USE_PCP_STREAMING_RPA_KERNEL:
                raise NotImplementedError(
                    "PCP prefill requires USE_PCP_STREAMING_RPA_KERNEL=1. "
                    "Only the materialized decode KV path is supported "
                    "without the streaming prefill kernel.")

        token_data_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))
        metadata_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.BATCH))
        source_block_tables_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.BATCH, None))
        pcp_streaming_schedule_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(ShardingAxisName.BATCH, None, None, None, None))
        pcp_streaming_active_page_groups_sharding = NamedSharding(
            self.mesh, PartitionSpec(ShardingAxisName.BATCH, None))

        (req_ids_dp, req_indices_dp, num_scheduled_tokens_per_dp_rank,
         scheduled_tokens_per_dp_rank, num_req_per_dp_rank,
         padded_num_scheduled_tokens_per_dp_rank, padded_num_reqs,
         attn_padded_num_reqs, padded_total_num_scheduled_tokens,
         padded_num_reqs_per_dp_rank, logits_indices_selector,
         max_num_reqs_per_dp_rank
         ) = self._prepare_input_metadata(scheduler_output, use_pcp=use_pcp)
        # Multi-modal support
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self.mm_manager.calc_mrope_positions(
                scheduler_output, req_ids_dp,
                padded_num_scheduled_tokens_per_dp_rank)

        # Async scheduling: prepare token substitution indices for DP
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for (req_id, draft_token_ids
             ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)
        token_in_tpu_cur_input_indices_dp = {}
        token_in_tpu_pre_next_tokens_indices_dp = {}
        draft_token_in_tpu_cur_indices_dp = {}
        draft_token_in_prev_next_tokens_indices_dp = {}
        if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            # If async previous results exists, we will prepare for the token substitution here
            # The actual substitution will be performed in tpu during later parts of this function.
            (token_in_tpu_cur_input_indices_dp,
             token_in_tpu_pre_next_tokens_indices_dp,
             draft_token_in_tpu_cur_indices_dp,
             draft_token_in_prev_next_tokens_indices_dp
             ) = self._prepare_async_token_substitution_indices(
                 req_ids_dp, scheduled_tokens_per_dp_rank,
                 padded_num_scheduled_tokens_per_dp_rank,
                 {0: num_draft_tokens}, dp_size)

        self.device_buffer.reset()

        input_ids_view = self.device_buffer.get_view(
            (padded_total_num_scheduled_tokens, ), key="input_ids")
        query_start_loc_view = self.device_buffer.get_view(
            (self.max_num_reqs + dp_size, ), key="query_start_loc")
        seq_lens_view = self.device_buffer.get_view((self.max_num_reqs, ),
                                                    key="seq_lens")

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0

        if use_spec_decode:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for (
                    req_id,
                    draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            num_sampled_tokens = num_draft_tokens + 1
            total_sampled_tokens = np.sum(num_sampled_tokens)
            padded_logits_length = runner_utils.get_padded_token_len(
                self.num_logits_paddings, total_sampled_tokens)
            logits_indices_shape = (padded_logits_length, )
        else:
            logits_indices_shape = (padded_num_reqs, )

        logits_indices_view = self.device_buffer.get_view(logits_indices_shape,
                                                          key="logits_indices")

        pcp_inverse_order_by_dp_rank: dict[int, np.ndarray] = {}

        # Populates input_ids and positions
        for dp_rank in range(dp_size):
            if num_req_per_dp_rank[dp_rank] == 0:
                continue
            token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]
            total_num_scheduled_tokens = num_scheduled_tokens_per_dp_rank[
                dp_rank]
            input_ids_cpu = input_ids_view[
                token_offset:token_offset +
                padded_num_scheduled_tokens_per_dp_rank]
            positions_cpu = self.positions_cpu[
                token_offset:token_offset +
                padded_num_scheduled_tokens_per_dp_rank]
            # Get request indices.
            # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
            # For each scheduled token, what are the corresponding req index.
            req_indices_per_req = np.asarray(req_indices_dp[dp_rank],
                                             dtype=np.int64)
            req_indices = np.repeat(req_indices_per_req,
                                    num_scheduled_tokens_per_req)
            # Get batched arange.
            # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # For each scheduled token, what is its position in corresponding req.
            arange = np.concatenate(
                [self.arange_cpu[:n] for n in num_scheduled_tokens_per_req])
            # Get positions.
            positions_np = positions_cpu[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )
            # Get token indices.
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
            # where M is the max_model_len.
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])
            np.take(
                self.input_batch.token_ids_cpu.ravel(),
                token_indices,
                out=input_ids_cpu[:total_num_scheduled_tokens],
            )

            if use_pcp:
                if any(n <= 0 for n in num_scheduled_tokens_per_req):
                    raise NotImplementedError(
                        "PCP runner path requires scheduled prompt tokens.")
                token_start_offsets = self.input_batch.num_computed_tokens_cpu[
                    req_indices_per_req]
                prompt_lens = self.input_batch.num_prompt_tokens[
                    req_indices_per_req]
                if np.any(token_start_offsets +
                          np.asarray(num_scheduled_tokens_per_req,
                                     dtype=np.int32) > prompt_lens):
                    raise NotImplementedError(
                        "PCP runner path currently supports prompt-only "
                        "prefill chunks.")
                mrope_slice = None
                if self.uses_mrope:
                    mrope_start = token_offset
                    mrope_end = token_offset + padded_num_scheduled_tokens_per_dp_rank
                    mrope_slice = self.mrope_positions_cpu[:, mrope_start:
                                                           mrope_end]
                # This helper rewrites the full padded per-DP slice and clears
                # padding entries before placing valid rank-major PCP tokens.
                pcp_inverse_order_by_dp_rank[
                    dp_rank] = _apply_pcp_rank_major_token_order(
                        input_ids_cpu,
                        positions_cpu,
                        num_scheduled_tokens_per_req,
                        pcp_size,
                        cp_kv_cache_interleave_size,
                        padded_num_scheduled_tokens_per_dp_rank,
                        mrope_slice,
                        token_start_offsets_per_req=token_start_offsets,
                    )
            else:
                input_ids_cpu[total_num_scheduled_tokens:] = 0

        # Prepare the attention metadata (query_start_loc_cpu, seq_lens_cpu)
        for dp_rank in range(dp_size):
            req_offset = dp_rank * max_num_reqs_per_dp_rank
            query_start_loc_cpu = query_start_loc_view[
                req_offset + dp_rank:req_offset + max_num_reqs_per_dp_rank +
                dp_rank + 1]
            seq_lens_cpu = seq_lens_view[req_offset:req_offset +
                                         max_num_reqs_per_dp_rank]
            _num_reqs = num_req_per_dp_rank[dp_rank]
            req_indices = req_indices_dp[dp_rank]
            num_scheduled_tokens_per_req = scheduled_tokens_per_dp_rank[
                dp_rank]

            if _num_reqs == 0:
                query_start_loc_cpu[:] = 0
                seq_lens_cpu[:] = 0
                continue

            # After buffer.reset(), the buffer is still dirty, so we need to zero
            # Out the starting index.
            query_start_loc_cpu[0] = 0
            np.cumsum(
                num_scheduled_tokens_per_req,
                out=query_start_loc_cpu[1:_num_reqs + 1],
            )
            query_start_loc_cpu[_num_reqs + 1:] = 1

            seq_lens_cpu[:_num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[req_indices] +
                num_scheduled_tokens_per_req)
            seq_lens_cpu[_num_reqs:] = 0

        # populate logits_indices
        for dp_rank in range(dp_size):
            req_offset = dp_rank * padded_num_reqs_per_dp_rank
            query_loc_req_offset = dp_rank * (max_num_reqs_per_dp_rank + 1)
            _num_reqs = num_req_per_dp_rank[dp_rank]

            logits_indices_cpu = logits_indices_view[
                req_offset:req_offset + padded_num_reqs_per_dp_rank]
            if use_pcp and _num_reqs > 0:
                token_offset = padded_num_scheduled_tokens_per_dp_rank * dp_rank
                local_request_ends = np.cumsum(
                    scheduled_tokens_per_dp_rank[dp_rank],
                    dtype=np.int64) - 1
                logits_indices_cpu[:_num_reqs] = (
                    pcp_inverse_order_by_dp_rank[dp_rank][local_request_ends] +
                    token_offset)
            else:
                local_logits_indices = (
                    query_start_loc_view[query_loc_req_offset +
                                         1:query_loc_req_offset + _num_reqs +
                                         1] - 1)
                if pcp_size > 1:
                    local_logits_indices = (
                        local_logits_indices +
                        padded_num_scheduled_tokens_per_dp_rank * dp_rank)
                logits_indices_cpu[:_num_reqs] = local_logits_indices
            logits_indices_cpu[_num_reqs:] = -1

            # Calculate batch composition statistics for active hardware profilers
            # and/or continuous batch logging.
        if self.phase_based_profiler or self.aggregated_stats_logger:
            self.batch_counter += 1
            batch_composition_stats = runner_utils.get_batch_composition_stats(
                self.batch_counter, self.input_batch,
                total_num_scheduled_tokens, num_reqs,
                padded_total_num_scheduled_tokens, scheduler_output)

            if self.phase_based_profiler:
                self.phase_based_profiler.step(batch_composition_stats)
            if self.aggregated_stats_logger:
                self.aggregated_stats_logger.log(batch_composition_stats)

        positions = self.positions_cpu[:padded_total_num_scheduled_tokens]
        mrope_positions = self.mrope_positions_cpu[:, :
                                                   padded_total_num_scheduled_tokens]
        _request_distribution = []
        for dp_rank in range(dp_size):
            _num_reqs = num_req_per_dp_rank[dp_rank]
            # The batch has been reordered by _reorder_batch so single-token
            # requests come first. They use decode-shaped RPA even if the token
            # is a prompt continuation.
            num_decode_in_dp_rank = 0
            for req_id in req_ids_dp[dp_rank]:
                if scheduler_output.num_scheduled_tokens[req_id] == 1:
                    num_decode_in_dp_rank += 1
            _request_distribution.append(
                [num_decode_in_dp_rank, num_decode_in_dp_rank, _num_reqs])
        request_distribution = np.array(_request_distribution,
                                        dtype=np.int32).ravel()

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        spec_decode_metadata = None
        if use_spec_decode:
            spec_decode_metadata = (
                self.speculative_decoding_manager.get_spec_decode_metadata(
                    num_draft_tokens,
                    query_start_loc_view[1:num_reqs + 1],
                    padded_num_reqs,
                    input_ids_view,
                ))
            logits_indices_view[:] = spec_decode_metadata.final_logits_indices

        # Put to device
        sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
            self.mesh,
            self.input_batch,
            padded_num_reqs,
            sharding=metadata_sharding,
        )

        if self.uses_mrope:
            # M-RoPE positions are of the shape (3, max_num_tokens).
            # https://github.com/vllm-project/tpu-inference/blob/efc9608acd925bb3b64db6fda509514f799ab7be/tpu_inference/runner/tpu_runner.py#L555
            # Shard the positions accordingly.
            mrope_sharding = NamedSharding(
                self.mesh, PartitionSpec(None, ShardingAxisName.ATTN_DATA))
            positions = device_array(self.mesh,
                                     mrope_positions,
                                     sharding=mrope_sharding)
        else:
            positions = device_array(self.mesh,
                                     positions,
                                     sharding=token_data_sharding)

        block_table_views_by_gid: dict[int, np.ndarray] = {}

        # Collect block tables host arrays loops zone presence zones legality
        def build_block_table_host(kv_cache_gid: int) -> None:

            block_table_obj = self.input_batch.block_table[kv_cache_gid]
            block_tables_view = self.device_buffer.get_view(
                (self.max_num_reqs, block_table_obj.max_num_blocks_per_req),
                key=f"block_tables_gid_{kv_cache_gid}")
            block_table_views_by_gid[kv_cache_gid] = block_tables_view

            # Zero out the view once for correct padding
            block_tables_view.fill(0)

            cpu_tensor = block_table_obj.get_cpu_tensor()
            for dp_rank in range(dp_size):
                _num_reqs = num_req_per_dp_rank[dp_rank]
                if _num_reqs == 0:
                    continue

                req_offset = dp_rank * max_num_reqs_per_dp_rank
                # Use np.take with out= to avoid intermediate copies from advanced indexing
                np.take(cpu_tensor,
                        req_indices_dp[dp_rank],
                        axis=0,
                        out=block_tables_view[req_offset:req_offset +
                                              _num_reqs])

        if len(self.kv_cache_config.kv_cache_groups) <= 1:
            no_kv_cache = len(self.kv_cache_config.kv_cache_groups) == 0
            if not no_kv_cache:
                build_block_table_host(0)
        else:
            for gid, kv_cache_group in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                build_block_table_host(gid)

        pcp_attention_metadata_by_gid: dict[int, dict[str, jax.Array]] = {}
        if (use_pcp or use_pcp_decode) and block_table_views_by_gid:
            build_pcp_streaming_schedule = envs.USE_PCP_STREAMING_RPA_KERNEL
            pcp_streaming_num_lanes = envs.PCP_STREAMING_RPA_NUM_LANES
            pcp_streaming_q_block_size = envs.PCP_STREAMING_RPA_Q_BLOCK_SIZE
            pcp_streaming_kv_pages_per_block = max(
                1,
                min(
                    ScheduleField.MAX_KV_PAGES_PER_BLOCK,
                    envs.PCP_STREAMING_RPA_KV_BLOCK_SIZE // self.block_size,
                ),
            )
            for gid, block_tables_view in block_table_views_by_gid.items():
                kv_cache_group = self.kv_cache_config.kv_cache_groups[gid]
                if not _kv_cache_group_supports_pcp_attention_metadata(
                        kv_cache_group):
                    continue
                if use_pcp:
                    metadata_per_dp = []
                    for dp_rank in range(dp_size):
                        req_offset = dp_rank * max_num_reqs_per_dp_rank
                        _num_reqs = num_req_per_dp_rank[dp_rank]
                        metadata_per_dp.append(
                            _build_pcp_attention_metadata(
                                scheduled_tokens_per_dp_rank[dp_rank],
                                seq_lens_view[req_offset:req_offset +
                                              _num_reqs].copy(),
                                block_tables_view[req_offset:req_offset +
                                                  max_num_reqs_per_dp_rank],
                                pcp_size,
                                cp_kv_cache_interleave_size,
                                padded_num_scheduled_tokens_per_dp_rank,
                                max_num_reqs_per_dp_rank,
                                self.block_size,
                                build_streaming_schedule=(
                                    build_pcp_streaming_schedule),
                                streaming_num_lanes=pcp_streaming_num_lanes,
                                streaming_q_block_size=(
                                    pcp_streaming_q_block_size),
                                streaming_kv_pages_per_block=(
                                    pcp_streaming_kv_pages_per_block),
                            ))
                    host_pcp_metadata = _merge_pcp_attention_metadata(
                        metadata_per_dp)
                    if host_pcp_metadata.streaming_schedule is None:
                        raise ValueError(
                            "PCP prefill metadata is missing the streaming "
                            "schedule.")
                    pcp_slot_ids = device_array(
                        self.mesh,
                        host_pcp_metadata.slot_ids,
                        sharding=token_data_sharding)
                    pcp_streaming_schedule = device_array(
                        self.mesh,
                        host_pcp_metadata.streaming_schedule,
                        sharding=pcp_streaming_schedule_sharding)
                    pcp_streaming_active_page_groups = device_array(
                        self.mesh,
                        host_pcp_metadata.streaming_active_page_groups,
                        sharding=pcp_streaming_active_page_groups_sharding)
                    pcp_attention_metadata_by_gid[gid] = {
                        "pcp_slot_ids": pcp_slot_ids,
                        "pcp_streaming_schedule": pcp_streaming_schedule,
                        "pcp_streaming_active_page_groups": (
                            pcp_streaming_active_page_groups),
                    }
                else:
                    metadata_per_dp = []
                    for dp_rank in range(dp_size):
                        req_offset = dp_rank * max_num_reqs_per_dp_rank
                        _num_reqs = num_req_per_dp_rank[dp_rank]
                        metadata_per_dp.append(
                            _build_pcp_decode_attention_metadata(
                                seq_lens_view[req_offset:req_offset +
                                              _num_reqs].copy(),
                                block_tables_view[req_offset:req_offset +
                                                  max_num_reqs_per_dp_rank],
                                self.block_size,
                                pcp_size,
                                cp_kv_cache_interleave_size,
                                padded_num_scheduled_tokens_per_dp_rank,
                                max_num_reqs_per_dp_rank,
                                scheduled_tokens_per_dp_rank[dp_rank],
                            ))
                    host_slot_ids = np.concatenate(
                        [m["slot_ids"] for m in metadata_per_dp])
                    host_source_block_tables = np.concatenate(
                        [m["source_block_tables"] for m in metadata_per_dp],
                        axis=0)
                    pcp_slot_ids = device_array(
                        self.mesh,
                        host_slot_ids,
                        sharding=token_data_sharding)
                    pcp_source_block_tables = device_array(
                        self.mesh,
                        host_source_block_tables,
                        sharding=source_block_tables_sharding)
                    pcp_attention_metadata_by_gid[gid] = {
                        "pcp_slot_ids": pcp_slot_ids,
                        "pcp_source_block_tables": pcp_source_block_tables,
                    }

        metadata_blob, metadata_layout = self.device_buffer.build()

        # Mamba slot ids are only consumed by hybrid attn+mamba models; for
        # pure-attention models, leaving the field None keeps AttentionMetadata
        # byte-identical to the pre-compact-mamba layout (so the model_fn
        # signature on those models is unchanged).
        if self.kv_cache_config.has_mamba_layers:
            # Reorder mamba_state_indices per DP rank (like block_tables)
            # and convert global slot ids to rank-local indices so they
            # index correctly into the per-rank shard of the mamba state.
            local_slots = self.input_batch._mamba_local_slots
            mamba_state_indices_cpu = np.zeros(self.max_num_reqs,
                                               dtype=np.int32)
            for dp_rank in range(dp_size):
                _num_reqs = num_req_per_dp_rank[dp_rank]
                if _num_reqs == 0:
                    continue
                req_offset = dp_rank * max_num_reqs_per_dp_rank
                global_slots = self.input_batch.mamba_state_indices_cpu[
                    req_indices_dp[dp_rank]]
                mamba_state_indices_cpu[req_offset:req_offset +
                                        _num_reqs] = (global_slots %
                                                      local_slots)
            (request_distribution, mamba_state_indices,
             dev_arrays_payload) = device_array(
                 self.mesh, (request_distribution, mamba_state_indices_cpu,
                             metadata_blob),
                 sharding=metadata_sharding)
        else:
            mamba_state_indices = None
            (request_distribution, dev_arrays_payload) = device_array(
                self.mesh, (request_distribution, metadata_blob),
                sharding=metadata_sharding)

        metadata = common_utils.DeviceBuffer.unpack_arrays(
            dev_arrays_payload, metadata_layout)
        input_ids = metadata["input_ids"]
        if use_pcp:
            input_ids = jax.device_put(input_ids, token_data_sharding)
        query_start_loc = metadata["query_start_loc"]
        seq_lens = metadata["seq_lens"]
        logits_indices = metadata["logits_indices"]

        # The host-side `num_computed_tokens_cpu` assumes all speculatively
        # proposed tokens from the previous step were accepted. Subtract the
        # actual rejection counts from `seq_lens` and `positions` on TPU.
        if self.speculative_config and self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            seq_lens, positions = self._subtract_num_rejected_tokens(
                seq_lens, positions, scheduled_tokens_per_dp_rank[0])

        # Build GDN reorder indices for PCP prefill. Each DP rank's token_order
        # maps packed-rank-major position → original sequential position.
        pcp_gdn_reorder_indices: jax.Array | None = None
        if use_pcp and self.kv_cache_config.has_mamba_layers:
            gdn_reorder_parts = []
            for dp_rank in range(dp_size):
                token_order, _ = _build_pcp_rank_major_token_order(
                    scheduled_tokens_per_dp_rank[dp_rank],
                    pcp_size,
                    cp_kv_cache_interleave_size,
                    padded_num_scheduled_tokens_per_dp_rank,
                )
                gdn_reorder_parts.append(token_order.astype(np.int32))
            pcp_gdn_reorder_cpu = np.concatenate(gdn_reorder_parts)
            pcp_gdn_reorder_indices = device_array(
                self.mesh, pcp_gdn_reorder_cpu,
                sharding=token_data_sharding)

        def build_attn(block_tables: jax.Array | None,
                       gid: int | None = None) -> AttentionMetadata:
            pcp_metadata = (pcp_attention_metadata_by_gid.get(gid, {})
                            if gid is not None else {})
            attention_metadata_gid = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                mamba_state_indices=mamba_state_indices,
                pcp_slot_ids=pcp_metadata.get("pcp_slot_ids"),
                pcp_source_block_tables=pcp_metadata.get(
                    "pcp_source_block_tables"),
                pcp_streaming_schedule=pcp_metadata.get(
                    "pcp_streaming_schedule"),
                pcp_streaming_active_page_groups=pcp_metadata.get(
                    "pcp_streaming_active_page_groups"),
                pcp_gdn_reorder_indices=pcp_gdn_reorder_indices,
                padded_num_reqs=attn_padded_num_reqs,
            )

            # This is for making these cpu buffers hidden during tracing
            attention_metadata_gid.query_start_loc_cpu = query_start_loc_view
            return attention_metadata_gid

        attention_metadata: AttentionMetadata | dict[str, AttentionMetadata]
        if len(self.kv_cache_config.kv_cache_groups) <= 1:
            # Pooling model will not using kv cache
            no_kv_cache = len(self.kv_cache_config.kv_cache_groups) == 0
            block_tables = metadata.get(
                "block_tables_gid_0") if not no_kv_cache else None
            attention_metadata = build_attn(block_tables,
                                            None if no_kv_cache else 0)
        else:
            attention_metadata = {
                name: build_attn(metadata[f"block_tables_gid_{gid}"], gid)
                for gid, kv_cache_group in enumerate(
                    self.kv_cache_config.kv_cache_groups)
                for name in kv_cache_group.layer_names
            }

        # Async scheduling: substitute placeholder tokens for DP
        if self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            # Collect all token indices that need substitution across all DP ranks
            all_token_indices_to_substitute = []
            all_pre_next_tokens_indices = []
            draft_all_token_indices_to_substitute = []
            draft_all_pre_next_tokens_indices = []

            for dp_rank in range(dp_size):
                cur_indices = token_in_tpu_cur_input_indices_dp[dp_rank]
                pre_indices = token_in_tpu_pre_next_tokens_indices_dp[dp_rank]
                all_token_indices_to_substitute.extend(cur_indices)
                all_pre_next_tokens_indices.extend(pre_indices)
                draft_all_token_indices_to_substitute.extend(
                    draft_token_in_tpu_cur_indices_dp[dp_rank])
                draft_all_pre_next_tokens_indices.extend(
                    draft_token_in_prev_next_tokens_indices_dp[dp_rank])

            if self.scheduler_config.async_scheduling and self._pre_async_results:
                if self.speculative_config:
                    next_tokens = self._pre_async_results.spec_decode_next_tokens
                else:
                    next_tokens = self._pre_async_results.next_tokens
                token_in_tpu_cur_input_indices = np.array(
                    all_token_indices_to_substitute)
                token_in_tpu_pre_next_tokens_indices = np.array(
                    all_pre_next_tokens_indices)
                input_ids = self._apply_async_token_substitution(
                    input_ids, next_tokens, token_in_tpu_cur_input_indices,
                    token_in_tpu_pre_next_tokens_indices)
                if spec_decode_metadata:
                    draft_token_in_tpu_cur_input_indices = np.array(
                        draft_all_token_indices_to_substitute)
                    draft_token_in_tpu_pre_next_tokens_indices = np.array(
                        draft_all_pre_next_tokens_indices)
                    draft_token_ids = self._apply_async_token_substitution(
                        spec_decode_metadata.draft_token_ids,
                        self._pre_async_results.spec_decode_next_tokens,
                        draft_token_in_tpu_cur_input_indices,
                        draft_token_in_tpu_pre_next_tokens_indices)
                    new_md = replace(spec_decode_metadata,
                                     draft_token_ids=draft_token_ids)
                    new_md.draft_lengths_cpu = spec_decode_metadata.draft_lengths_cpu
                    spec_decode_metadata = new_md

        num_scheduled_tokens_per_req = np.concatenate([
            np.array(scheduled_tokens_per_dp_rank[dp_rank], dtype=np.int32)
            for dp_rank in range(dp_size)
        ])
        if self.lora_config is not None:
            self.lora_utils.set_active_loras(
                num_scheduled_tokens_per_req,
                total_num_scheduled_tokens,
                padded_total_num_scheduled_tokens,
            )

        return (
            input_ids,
            positions,
            attention_metadata,
            sampling_metadata,
            logits_indices,
            spec_decode_metadata,
            logits_indices_selector,
            padded_num_reqs,
            req_ids_dp,
            padded_num_scheduled_tokens_per_dp_rank,
        )

    def _get_input_ids_embeds(self, input_ids: jax.Array,
                              mm_embeds: list[jax.Array] | None,
                              is_mm_embed: jax.Array | None):
        # Prevent the cost of calling additional function.
        if self.is_multimodal_model and mm_embeds is not None:
            assert self.embed_input_ids_fn is not None
            inputs_embeds = self.embed_input_ids_fn(
                self.state_leaves,
                input_ids,
                mm_embeds,
                is_multimodal=is_mm_embed,
            )
            return None, inputs_embeds
        else:
            return input_ids, None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.speculative_decoding_manager.take_draft_token_ids()

    ###### Local disagg utilities ######

    def get_kv_cache_for_block_ids(
        self,
        block_ids: List[int],
    ) -> List[jax.Array]:
        return self.kv_cache_manager.get_kv_cache_for_block_ids(block_ids)

    def transfer_kv_cache(self,
                          kv_cache_slices: List[jax.Array]) -> List[jax.Array]:
        return self.kv_cache_manager.transfer_kv_cache(kv_cache_slices)

    def insert_request_with_kv_cache(
        self,
        request: "Request",
        kv_cache_slices: List[jax.Array],
        block_ids: List[List[int]],
    ):
        return self.kv_cache_manager.insert_request_with_kv_cache(
            request, kv_cache_slices, block_ids)

    ###### RL framework integration ######

    def _sync_weights(
        self,
        updated_weights: jaxtyping.PyTree,
        mappings: Dict[str, Tuple[str, Tuple[str]]],
        transpose_keys: Dict[str, Tuple[int]],
        reshard_fn: Callable[[jaxtyping.PyTree, jaxtyping.PyTree],
                             jaxtyping.PyTree] = None
    ) -> None:
        """For RL framework integration."""
        if reshard_fn is not None:
            updated_weights = reshard_fn(updated_weights, self.state)
            shard = None
        else:
            shard = functools.partial(shard_put, mesh=self.mesh)
        self.state = transfer_state_with_mappings(
            src_state=updated_weights,
            tgt_state=self.state,
            mappings=mappings,
            transpose_keys=transpose_keys,
            shard=shard)
        # Keep the dispatch-side view in sync with the updated state so
        # subsequent jit dispatches see the new weights.
        if isinstance(self.state, nnx.State):
            self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
        else:
            self.state_leaves = self.state

    def _get_padded_total_tokens(
            self, scheduler_output: "VllmSchedulerOutput") -> int:
        num_tokens = scheduler_output.total_num_scheduled_tokens

        # Determine the capacity per rank (max tokens assigned to any single device)
        max_tokens_per_rank = getattr(
            scheduler_output, "max_num_scheduled_tokens_per_dp_rank",
            (num_tokens + self.dp_size - 1) // self.dp_size)

        # Map to the next local bucket and multiply by world size to get global shape
        padded_per_rank = runner_utils.get_padded_token_len(
            self.num_tokens_paddings_per_dp, max_tokens_per_rank)

        return padded_per_rank * self.dp_size

    def get_intermediate_tensor_spec(self,
                                     scheduler_output: "VllmSchedulerOutput"):
        jax_dtype = to_jax_dtype(self.dtype)
        num_padded_tokens = self._get_padded_total_tokens(scheduler_output)

        if self.dp_size > 1:
            sharding = NamedSharding(
                self.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
        else:
            sharding = NamedSharding(self.mesh, PartitionSpec())
        hidden_size = self.model_config.get_hidden_size()
        spec = jax.ShapeDtypeStruct(shape=(num_padded_tokens, hidden_size),
                                    dtype=jax_dtype,
                                    sharding=sharding)
        tensor_spec = {"hidden_states": spec, "residual": spec}
        return tensor_spec

    def get_uuid_for_jax_transfer(self,
                                  scheduler_output: "VllmSchedulerOutput",
                                  rank: int, step: int) -> int:
        '''
        Get a uuid for jax.transfer, here we use the hash of
        scheduler_output + counter_step + sender's rank
        '''
        scheduler_output_str = ""
        if not scheduler_output.num_scheduled_tokens:
            scheduler_output_str = "empty_batch"
        else:
            scheduler_output_str = str(
                sorted(scheduler_output.num_scheduled_tokens.items()))
        unique_str = f'{scheduler_output_str} {step} {rank}'
        import hashlib
        hasher = hashlib.sha1()
        hasher.update(unique_str.encode('utf-8'))
        return int.from_bytes(hasher.digest()[:8], 'big')
