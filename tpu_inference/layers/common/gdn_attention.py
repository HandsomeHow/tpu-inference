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
"""
Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl

"""
import dataclasses
import enum
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpu_inference.layers.common.ragged_gated_delta_rule_wrapper as ragged_gated_delta_rule_wrapper
from tpu_inference.layers.common.ragged_conv1d_jax import \
    ragged_conv1d as ragged_conv1d_jax
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    inverse_reorder_for_sharding, reorder_concatenated_tensor_for_sharding)
from tpu_inference.utils import get_mesh_shape_product


class RaggedConv1dImpl(enum.Enum):
    JAX = "ragged_conv1d_jax"


RaggedGatedDeltaRuleImpl = ragged_gated_delta_rule_wrapper.RaggedGatedDeltaRuleImpl


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GdnAttentionConfig:
    ragged_conv1d_impl: RaggedConv1dImpl = RaggedConv1dImpl.JAX
    ragged_gated_delta_rule_impl: RaggedGatedDeltaRuleImpl = (
        RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_PD)


def run_jax_gdn_attention_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local JAX GDN attention mechanism with combined QKV tensors.

    Args:
        mixed_qkv: Combined QKV tensor of shape `(num_tokens, dim)`.
        b: B tensor of shape `(num_tokens, n_v)`.
        a: A tensor of shape `(num_tokens, n_v)`.
        conv_state: Combined convolutional state of shape `(num_blocks,
          kernel_size - 1, dim)`. `num_blocks` is always equal or larger than
          `max_seqs + 1`. The first block is a null_block and only used for
          padded / invalid tokens.
        recurrent_state: Recurrent state of shape `(num_blocks, n_v, d_k, d_v)`.
        conv_weight: Combined convolutional weight of shape `(dim, 1,
          kernel_size)`.
        conv_bias: Optional combined convolutional bias of shape `(dim,)`.
        A_log: Log of A parameter of shape `(n_v,)`.
        dt_bias: Delta T bias of shape `(n_v,)`.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used to derive
          ``has_initial_state`` so brand-new prefills don't read stale state
          from a reused mamba slot, mirroring GPU's
          ``initial_state[~has_initial_state, ...] = 0`` in
          ``gdn_linear_attn._forward_core``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    # has_initial_state[i] = True iff request i already has computed
    # tokens in its mamba slot (chunked-prefill continuation, prefix-cache
    # hit, or running decode). False for brand-new prefills, in which
    # case the conv1d, the chunked / ref delta-rule impls, and the fused
    # Pallas recurrent kernel all zero the slot's prior state before
    # the update so a freshly-allocated mamba slot can't leak its
    # previous tenant's state. context_len = seq_len - query_len.
    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    # TODO: Switch conv implementaion based on config once we have more than 1 impl
    conv_impl = ragged_conv1d_jax

    out_mixed_qkv, new_conv_state = conv_impl(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

    if config.ragged_gated_delta_rule_impl == RaggedGatedDeltaRuleImpl.REF:
        ragged_gdn_impl = functools.partial(
            ragged_gated_delta_rule_ref,
            has_initial_state=has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        new_recurrent_state, output = ragged_gdn_impl(
            out_mixed_qkv,
            b,
            a,
            recurrent_state,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
        )
    else:
        wrapper_config = config.ragged_gated_delta_rule_impl.to_config()
        new_recurrent_state, output = ragged_gated_delta_rule_wrapper.ragged_gated_delta_rule_wrapper(
            mixed_qkv=out_mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            config=wrapper_config,
            chunk_size=32,
            has_initial_state=has_initial_state,
        )

    return (new_conv_state, new_recurrent_state), output


def run_jax_gdn_attention(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the Jax GDN attention mechanism.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        conv_state: Convolutional state tensor of shape `(num_blocks, kernel_size
          - 1, dim)`. `num_blocks` is always equal or larger than `max_seqs +
          1`. The first block is a null_block and only used for padded / invalid
          tokens.
        recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
          d_v)`.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used inside the local function
          to derive ``has_initial_state``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(num_blocks, kernel_size - 1, dim)`
          - new_recurrent_state: `(num_blocks, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    in_specs = (
        P(ShardingAxisName.BATCH,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD),  # j_b
        P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD),  # j_a
        P(ShardingAxisName.BATCH, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state
        P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD, None,
          None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(ShardingAxisName.BATCH),  # query_start_loc
        P(ShardingAxisName.BATCH),  # state_indices
        P(ShardingAxisName.BATCH),  # distribution
        P(ShardingAxisName.BATCH),  # seq_lens
    )

    out_specs = (
        (
            P(ShardingAxisName.BATCH, None,
              ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)

    p_run_jax_gdn_attention_local = functools.partial(
        run_jax_gdn_attention_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        config=config,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )

    return (new_conv_state, new_recurrent_state), output


def _slice_dim(tensor: jnp.ndarray, slice_index: jnp.ndarray | int,
               num_slices: int, axis: int) -> jnp.ndarray:
    if axis < 0:
        axis += tensor.ndim
    assert tensor.shape[axis] % num_slices == 0
    slice_size = tensor.shape[axis] // num_slices
    start_index = slice_index * slice_size
    return jax.lax.dynamic_slice_in_dim(tensor,
                                        start_index,
                                        slice_size,
                                        axis=axis)


def run_jax_gdn_attention_pcp_tp_prefill(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    reorder_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    mesh: jax.sharding.Mesh,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """GDN PCP prefill with PCP ranks acting as extra head shards.

    During PCP prefill, tokens are distributed across PCP ranks in rank-major
    interleaved order. GDN has sequential state dependencies, so we must
    reconstruct the full ordered sequence before running the recurrent scan.

    Strategy: AllGather tokens across PCP -> reorder to original sequential
    order -> slice this PCP rank's head shard -> run local GDN -> gather output
    and state heads back to TP-local full-head layout -> reorder output back ->
    take local token slice.

    Args:
        reorder_indices: (padded_num_tokens_per_dp,) int32 maps packed
            rank-major position -> original sequential position. -1 = padding.
        pcp_size: Number of PCP ranks.
        Other args: same as run_jax_gdn_attention.
    """
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT

    # Token arrays include PCP in their sharding (ATTN_DATA = BATCH + pcp).
    # State/metadata use BATCH only (replicated across PCP ranks).
    in_specs = (
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv (pcp-split)
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_b (pcp-split)
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_a (pcp-split)
        P(ShardingAxisName.BATCH, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state (pcp-replicated)
        P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD,
          None, None),  # recurrent_state (pcp-replicated)
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(ShardingAxisName.BATCH),  # query_start_loc (pcp-replicated)
        P(ShardingAxisName.BATCH),  # state_indices (pcp-replicated)
        P(ShardingAxisName.BATCH),  # distribution (pcp-replicated)
        P(ShardingAxisName.BATCH),  # seq_lens (pcp-replicated)
        P(ShardingAxisName.ATTN_DATA),  # reorder_indices (pcp-split)
    )

    out_specs = (
        (
            P(ShardingAxisName.BATCH, None,
              ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD,
              None, None),  # new_recurrent_state
        ),
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # output (pcp-split)
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    effective_tp = tp_size * pcp_size
    assert n_kq % effective_tp == 0, (
        f"n_kq={n_kq} must be divisible by effective_tp={effective_tp}")
    assert n_v % effective_tp == 0, (
        f"n_v={n_v} must be divisible by effective_tp={effective_tp}")
    local_n_kq = n_kq // effective_tp
    local_n_v = n_v // effective_tp
    local_key_dim = n_kq * d_k // tp_size
    local_value_dim = n_v * d_v // tp_size

    def _pcp_prefill_fn(
        local_qkv, local_b, local_a,
        conv_state_, recurrent_state_,
        conv_weight_, conv_bias_,
        A_log_, dt_bias_,
        query_start_loc_, state_indices_, distribution_, seq_lens_,
        local_reorder_indices,
    ):
        # AllGather token data across PCP ranks
        full_qkv = jax.lax.all_gather(
            local_qkv, axis_name=pcp_axis, axis=0, tiled=True)
        full_b = jax.lax.all_gather(
            local_b, axis_name=pcp_axis, axis=0, tiled=True)
        full_a = jax.lax.all_gather(
            local_a, axis_name=pcp_axis, axis=0, tiled=True)
        full_reorder = jax.lax.all_gather(
            local_reorder_indices, axis_name=pcp_axis, axis=0, tiled=True)

        # Reorder from packed rank-major to original sequential order.
        # full_reorder[i] = original position for packed position i, or -1.
        valid_mask = full_reorder >= 0
        scatter_indices = jnp.where(valid_mask, full_reorder, full_reorder.size)
        gather_indices = jnp.where(valid_mask, full_reorder, 0)

        seq_qkv = jnp.zeros_like(full_qkv)
        seq_b = jnp.zeros_like(full_b)
        seq_a = jnp.zeros_like(full_a)

        # Scatter packed -> sequential. Invalid padding entries use an
        # out-of-bounds index and are dropped instead of overwriting token 0.
        seq_qkv = seq_qkv.at[scatter_indices].set(full_qkv, mode="drop")
        seq_b = seq_b.at[scatter_indices].set(full_b, mode="drop")
        seq_a = seq_a.at[scatter_indices].set(full_a, mode="drop")

        rank = jax.lax.axis_index(pcp_axis)
        qkv_shard = _slice_dim(seq_qkv, rank, pcp_size, axis=-1)
        b_shard = _slice_dim(seq_b, rank, pcp_size, axis=-1)
        a_shard = _slice_dim(seq_a, rank, pcp_size, axis=-1)
        weight_shard = _slice_dim(conv_weight_, rank, pcp_size, axis=0)
        bias_shard = (None if conv_bias_ is None else
                      _slice_dim(conv_bias_, rank, pcp_size, axis=0))
        A_shard = _slice_dim(A_log_, rank, pcp_size, axis=0)
        dt_shard = _slice_dim(dt_bias_, rank, pcp_size, axis=0)

        conv_state_interleaved = reorder_concatenated_tensor_for_sharding(
            conv_state_,
            [local_key_dim, local_key_dim, local_value_dim],
            pcp_size,
            -1,
        )
        conv_state_shard = _slice_dim(
            conv_state_interleaved, rank, pcp_size, axis=-1)
        recurrent_state_shard = _slice_dim(
            recurrent_state_, rank, pcp_size, axis=1)

        (new_conv_shard, new_rec_shard), seq_output_shard = (
            run_jax_gdn_attention_local(
                qkv_shard, b_shard, a_shard,
                conv_state_shard, recurrent_state_shard,
                weight_shard, bias_shard,
                A_shard, dt_shard,
                query_start_loc_, state_indices_, distribution_, seq_lens_,
                n_kq=local_n_kq,
                n_v=local_n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                config=config,
            ))

        seq_output = jax.lax.all_gather(seq_output_shard,
                                        axis_name=pcp_axis,
                                        axis=-1,
                                        tiled=True)
        new_conv_gathered = jax.lax.all_gather(new_conv_shard,
                                               axis_name=pcp_axis,
                                               axis=-1,
                                               tiled=True)
        new_rec = jax.lax.all_gather(new_rec_shard,
                                     axis_name=pcp_axis,
                                     axis=1,
                                     tiled=True)
        new_conv = inverse_reorder_for_sharding(
            new_conv_gathered,
            [local_key_dim, local_key_dim, local_value_dim],
            pcp_size,
            -1,
        )

        # Gather output back: sequential -> packed rank-major
        full_output = seq_output[gather_indices]
        full_output = jnp.where(valid_mask[:, None], full_output, 0.0)

        # Take local slice for this PCP rank
        local_tokens = full_output.shape[0] // pcp_size
        local_output = jax.lax.dynamic_slice_in_dim(
            full_output, rank * local_tokens, local_tokens, axis=0)

        return (new_conv, new_rec), local_output

    mapped_fn = jax.shard_map(
        _pcp_prefill_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        reorder_indices,
    )

    return (new_conv_state, new_recurrent_state), output


def run_jax_gdn_attention_pcp_prefill(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    reorder_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    mesh: jax.sharding.Mesh,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return run_jax_gdn_attention_pcp_tp_prefill(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        reorder_indices,
        n_kq,
        n_v,
        d_k,
        d_v,
        kernel_size,
        pcp_size,
        mesh,
        config,
    )
