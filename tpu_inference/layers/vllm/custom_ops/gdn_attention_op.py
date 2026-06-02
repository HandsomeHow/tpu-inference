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

import re

import jax
import jax.numpy as jnp
import torch
# NOTE: we don't specify this in our requirements.txt but it should be coming
# from upstream vLLM
from einops import rearrange
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention

from tpu_inference import envs
from tpu_inference.layers.common.gdn_attention import (
    GdnAttentionConfig,
    run_jax_gdn_attention,
    run_jax_gdn_attention_pcp_tp_prefill,
)
from tpu_inference.layers.common.ragged_gated_delta_rule_wrapper import \
    RaggedGatedDeltaRuleImpl
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    reorder_concatenated_tensor_for_sharding, truncate_sharded_tensor)
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)


def _gdn_trace_stage_prefix(layer_name: str) -> str:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match is None:
        return "linear_attention.gdn"
    return f"layer.{match.group(1)}.linear_attention.gdn"


def _jax_dtype_for_torch_dtype(torch_dtype: torch.dtype):
    torch_to_jax_dtype = {
        torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        torch.int8: jnp.int8,
        torch.int16: jnp.int16,
        torch.int32: jnp.int32,
        torch.int64: jnp.int64,
    }
    return torch_to_jax_dtype.get(torch_dtype)


def _cast_jax_output_to_buffer_dtype(j_output, output_buffer: torch.Tensor):
    """Cast JAX output to the destination TorchAX buffer dtype before copy_."""
    if not hasattr(j_output, "astype"):
        return j_output
    target_dtype = _jax_dtype_for_torch_dtype(
        getattr(output_buffer, "dtype", None))
    if target_dtype is None:
        try:
            target_dtype = jax_view(output_buffer).dtype
        except Exception:
            return j_output
    if getattr(j_output, "dtype", None) == target_dtype:
        return j_output
    return j_output.astype(target_dtype)


def _cast_torchax_tensor_to_torch_dtype(tensor: torch.Tensor,
                                        torch_dtype: torch.dtype):
    target_dtype = _jax_dtype_for_torch_dtype(torch_dtype)
    if target_dtype is None:
        return tensor
    try:
        arr = jax_view(tensor)
        # Some TorchAX/JAX fused paths can keep a bf16-typed value as an
        # unrounded fp32 expression when all downstream users immediately
        # upcast it again. Force a real round-trip at this boundary so PCP and
        # non-PCP paths observe the same dtype semantics.
        if target_dtype in (jnp.bfloat16, jnp.float16):
            casted = arr.astype(jnp.float32).astype(target_dtype)
        else:
            casted = arr.astype(target_dtype)
        return torch_view(jax.lax.optimization_barrier(casted))
    except Exception:
        return tensor


def gdn_attention_core_tpu(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
    mesh: jax.sharding.Mesh,
) -> None:
    """
    This acts as main bridge between PyTorch and JAX for the GDN core attention.
    Uses a robust, token-by-token scan to inherently handle any mix of
    ragged prefill and decode sequences without dynamic shape compilation errors.

    Some key details:
    1. Cache Mapping: We'll read vLLM's  `block_tables` and `query_start_loc`
       and translate them into static index arrays (`req_indices` and `state_indices`).
    2. JAX Scan: We use `jax.lax.scan` to perform a robust, token-by-token loop over
       the flat inputs. This allows us to handle ANY mix of prefill and decode tokens
       in a single compiled XLA graph.
    3. Conditional Updates: The `valid_mask` ensures that padded dummy tokens
       (used to keep the tensor shape static) do not corrupt the recurrent state
       in the cache.
    """
    fc = get_forward_context()
    attn_metadata = fc.attn_metadata[layer_name]

    layer_module = fc.no_compile_layers[layer_name]
    vllm_context = get_vllm_model_wrapper_context()

    n_kq = layer_module.num_k_heads
    n_v = layer_module.num_v_heads
    d_k = layer_module.head_k_dim
    d_v = layer_module.head_v_dim
    kernel_size = layer_module.conv_kernel_size

    j_mixed_qkv = jax_view(mixed_qkv)  # [num_tokens, dim]
    j_b = jax_view(b)
    j_a = jax_view(a)

    j_conv_weight = jax_view(layer_module.conv1d.weight)
    j_conv_bias = jax_view(layer_module.conv1d.bias
                           ) if layer_module.conv1d.bias is not None else None
    j_A_log = jax_view(layer_module.A_log)
    j_dt_bias = jax_view(layer_module.dt_bias)

    # The j_mixed_qkv and j_conv_weight are not in an interleaved layout.
    # E.g. they are in [Q Q | K K | V V] layout. We need [Q K | Q K | Q K] layout.
    # Use reorder_concatenated_tensor_for_sharding to reorder into correct layout
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    dp_size = get_mesh_shape_product(mesh, ShardingAxisName.BATCH)
    pcp_size = get_mesh_shape_product(mesh, ShardingAxisName.PREFILL_CONTEXT)
    use_pcp_prefill = (pcp_size > 1
                       and attn_metadata.pcp_gdn_reorder_indices is not None)
    effective_tp = tp_size * pcp_size if use_pcp_prefill else tp_size
    assert n_kq % effective_tp == 0, (
        f"n_kq={n_kq} must be divisible by effective_tp={effective_tp} "
        f"(tp_size={tp_size}, pcp_size={pcp_size})")
    assert n_v % effective_tp == 0, (
        f"n_v={n_v} must be divisible by effective_tp={effective_tp} "
        f"(tp_size={tp_size}, pcp_size={pcp_size})")

    j_mixed_qkv = reorder_concatenated_tensor_for_sharding(
        j_mixed_qkv, [key_dim, key_dim, value_dim], effective_tp, -1)
    j_conv_weight = reorder_concatenated_tensor_for_sharding(
        j_conv_weight, [key_dim, key_dim, value_dim], effective_tp, 0)
    if j_conv_bias is not None:
        j_conv_bias = reorder_concatenated_tensor_for_sharding(
            j_conv_bias, [key_dim, key_dim, value_dim], effective_tp, 0)

    layer_idx = vllm_context.layer_name_to_kvcache_index[layer_name]
    conv_state, recurrent_state = vllm_context.kv_caches[layer_idx]
    state_len = conv_state.shape[1]
    if state_len > kernel_size - 1:
        conv_state_in = conv_state[:, :kernel_size - 1, :]
    else:
        conv_state_in = conv_state

    # Index mamba state by the per-request slot id from
    # `InputBatch.mamba_state_indices_cpu`, not by `block_tables[:, 0]`
    # (vLLM's GPU convention). Two reasons:
    #
    #  1. `_maybe_set_compact_mamba_num_blocks_override` caps the mamba
    #     pool at `max_num_seqs + 1` while the attention pool is much
    #     larger; using `block_tables[:, 0]` (a value in the attention
    #     range) would walk off the end of the mamba arrays.
    #  2. When vLLM's input batch runs `condense` to compact the persistent
    #     batch (https://github.com/vllm-project/vllm/blob/de3da0b/vllm/v1/worker/gpu_input_batch.py#L662 — moves
    #     requests into lower-index slots after earlier ones finish), the
    #     slot id moves with the request so the kernel still reads/writes
    #     the slot that holds this request's real state.
    state_indices = attn_metadata.mamba_state_indices.astype(jnp.int32)

    config = GdnAttentionConfig(
        ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl(
            envs.RAGGED_GATED_DELTA_RULE_IMPL))
    logger.info_once(f"GDN Attention Config: {config}")

    padded_num_reqs_per_dp = attn_metadata.padded_num_reqs // dp_size

    # Slice the state indices to the padded_num_reqs, which is the actual number
    # of requests padded to the bucket.
    state_indices_sliced = truncate_sharded_tensor(state_indices,
                                                   padded_num_reqs_per_dp,
                                                   dp_size)
    query_start_loc_sliced = truncate_sharded_tensor(
        attn_metadata.query_start_loc, padded_num_reqs_per_dp + 1, dp_size)
    seq_lens_sliced = truncate_sharded_tensor(attn_metadata.seq_lens,
                                              padded_num_reqs_per_dp, dp_size)

    if use_pcp_prefill:
        (new_conv_state_extracted,
         new_recurrent_state), j_output = run_jax_gdn_attention_pcp_tp_prefill(
             j_mixed_qkv,
             j_b,
             j_a,
             conv_state_in,
             recurrent_state,
             j_conv_weight,
             j_conv_bias,
             j_A_log,
             j_dt_bias,
             state_indices_sliced,
             query_start_loc_sliced,
             attn_metadata.request_distribution,
             seq_lens_sliced,
             attn_metadata.pcp_gdn_reorder_indices,
             n_kq,
             n_v,
             d_k,
             d_v,
             kernel_size,
             pcp_size=pcp_size,
             mesh=mesh,
             config=config)
    else:
        (new_conv_state_extracted,
         new_recurrent_state), j_output = run_jax_gdn_attention(
             j_mixed_qkv,
             j_b,
             j_a,
             conv_state_in,
             recurrent_state,
             j_conv_weight,
             j_conv_bias,
             j_A_log,
             j_dt_bias,
             state_indices_sliced,
             query_start_loc_sliced,
             attn_metadata.request_distribution,
             seq_lens_sliced,
             n_kq,
             n_v,
             d_k,
             d_v,
             kernel_size,
             mesh=mesh,
             config=config)
    if state_len > kernel_size - 1:
        remaining_old_state = conv_state[:, kernel_size - 1:, :]
        new_conv_state = jnp.concatenate(
            [new_conv_state_extracted, remaining_old_state], axis=1)
    else:
        new_conv_state = new_conv_state_extracted

    vllm_context.kv_caches[layer_idx] = (new_conv_state, new_recurrent_state)

    j_output_flat = j_output.reshape(core_attn_out.shape)
    j_output_flat = _cast_jax_output_to_buffer_dtype(j_output_flat,
                                                     core_attn_out)
    core_attn_out.copy_(torch_view(j_output_flat))


@QwenGatedDeltaNetAttention.register_oot
class VllmGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Implements the exact same logic as in vLLM (https://github.com/vllm-project/vllm/blob/9c81f35/vllm/model_executor/layers/mamba/gdn_linear_attn.py#L508)
        but omits the reshape in Part 3 for z/core_attn_out that is causing an unnecessary all-gather.

        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        mesh = vllm_model_wrapper_context.mesh
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            mixed_qkv = _cast_torchax_tensor_to_torch_dtype(
                mixed_qkv, hidden_states.dtype)
            ba = _cast_torchax_tensor_to_torch_dtype(ba, hidden_states.dtype)
            z = _cast_torchax_tensor_to_torch_dtype(z, hidden_states.dtype)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            mixed_qkvz = _cast_torchax_tensor_to_torch_dtype(
                mixed_qkvz, hidden_states.dtype)
            ba = _cast_torchax_tensor_to_torch_dtype(ba, hidden_states.dtype)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba)
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value))
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        gdn_attention_core_tpu(mixed_qkv,
                               b,
                               a,
                               core_attn_out,
                               self.prefix,
                               mesh=mesh)
        core_attn_out = _cast_torchax_tensor_to_torch_dtype(
            core_attn_out, hidden_states.dtype)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        projected, _ = self.out_proj(core_attn_out)
        output[:num_tokens] = projected
