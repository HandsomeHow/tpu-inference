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
import inspect
import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import Sharding

import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa_hd64
from tpu_inference import envs
from tpu_inference.kernels.experimental.batched_rpa import \
    wrapper as batched_rpa_wrapper
from tpu_inference.kernels.experimental.pcp_streaming_rpa import (
    pcp_streaming_attention_page_groups_packed_local)
from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField)
from tpu_inference.kernels.flash_attention.kernel import flash_attention
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.layers.common.attention_metadata import (AttentionMetadata,
                                                            PcpMode)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_megacore, get_mesh_shape_product

logger = init_logger(__name__)

MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.

# NOTE: this kernel is experimental and not fully tested.  See
# tpu-inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py
# for details
if envs.USE_BATCHED_RPA_KERNEL:
    import tpu_inference.kernels.experimental.batched_rpa.wrapper as rpa
    logger.info_once("Using experimental batched RPA kernel")
else:
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
    logger.info_once("Using default RPA kernel")

ragged_paged_attention = rpa.ragged_paged_attention
get_kv_cache_shape = rpa.get_kv_cache_shape

ragged_paged_attention_hd64 = rpa_hd64.ragged_paged_attention_hd64
get_kv_cache_shape_hd64 = rpa_hd64.get_kv_cache_shape


@functools.lru_cache(maxsize=None)
def _ragged_paged_attention_accepts_pcp_metadata(
        func: Callable[..., Any]) -> bool:
    rpa_params = inspect.signature(func).parameters
    return ("q_start_offsets" in rpa_params
            or any(param.kind == inspect.Parameter.VAR_KEYWORD
                   for param in rpa_params.values()))


def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    vmem_limit_bytes: int | None = None,
    use_attention_bias: bool = False,
) -> Callable[..., Any]:
    if use_attention_bias:
        in_specs = (
            P("data", "model", None, None),  # q
            P("data", "model", None, None),  # k
            P("data", "model", None, None),  # v
            P("data", "model", None, None),  # attention_bias
            P("data", None),  # segment_ids (B matches q's B, so shard 'data')
        )
        out_specs = P("data", "model", None, None)

        def _flash_attention_use_ab(q, k, v, attention_bias, segment_ids):
            return flash_attention(q,
                                   k,
                                   v,
                                   ab=attention_bias,
                                   segment_ids=segment_ids,
                                   sm_scale=sm_scale,
                                   causal=causal,
                                   vmem_limit_bytes=vmem_limit_bytes)

        attn_fn = _flash_attention_use_ab
    else:
        in_specs = (
            P("data", "model", None, None),  # q
            P("data", "model", None, None),  # k
            P("data", "model", None, None),  # v
            P("data", None),  # segment_ids (B matches q's B, so shard 'data')
        )
        out_specs = P("data", "model", None, None)

        def _flash_attention(q, k, v, segment_ids):
            return flash_attention(q,
                                   k,
                                   v,
                                   segment_ids=segment_ids,
                                   sm_scale=sm_scale,
                                   causal=causal,
                                   vmem_limit_bytes=vmem_limit_bytes)

        attn_fn = _flash_attention

    return jax.jit(
        jax.shard_map(attn_fn,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: Optional[float] = None,
) -> Callable[..., Any]:
    """Shards GQA PagedAttention along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P("model", None, None, None),  # k
        P("model", None, None, None),  # v
        P(),  # lengths
        P(),  # page_indices
    )
    out_specs = P(None, "model", None)

    def _paged_attention_fn(q, k, v, lengths, page_indices):
        if page_indices.size > MAX_ALLOWED_PAGE_INDICES_N:
            raise ValueError(
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` to run with minibatches."
            )
        return paged_attention(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        jax.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


# TODO(xiangxu): merge this with sharded_paged_attention
@jax.jit(static_argnames=["paged_attention_kernel"])
def paged_attention_with_guarded_smem(
    paged_attention_kernel: Callable,
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
):
    # Addresses b/336316706. Summary:
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent. Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths,
                                      page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(q[i], k_pages, v_pages, seq_lens[i],
                                   block_indices[i]))

    outputs = outputs.reshape((batch_size, *outputs.shape[2:]))

    return outputs


# ruff: noqa: E741
def update_cache(
    is_prefill,
    cache,
    indices,
    operand,
    prefill_seq_len=None,
    sliding_window=None,
) -> jax.Array:

    # (8, 55640, 32, 128) (1, 8, 256, 128) -> K (8, 8, 32, 128)
    # I = B * T // S
    # k cache, operand

    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op inserted by
    #    the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from actual prompt, not from the padded tokens.
        if sliding_window and T > sliding_window:
            assert B == 1
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            operand = jax.lax.dynamic_slice_in_dim(
                operand, start_index, sliding_window,
                axis=2)  # TODO: @pooyam Perf check this.
            T = sliding_window

        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)
        indices = indices.reshape(I)
        cache = cache.at[:, indices, :, :].set(operand)
    else:
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        cache = cache.reshape(K, L * S, H)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache


@jax.jit(static_argnames=["window_size", "attn_logits_soft_cap", "is_mqa"])
def apply_splash(q, k, v, window_size, attn_logits_soft_cap,
                 is_mqa) -> jax.Array:
    # q: (batch_size, num_heads, seq_len, head_dim)
    num_heads = q.shape[1]
    q_seq_len = q.shape[2]
    kv_seq_len = k.shape[2]
    assert kv_seq_len >= q_seq_len

    masks = [
        mask_lib.LocalMask((q_seq_len, kv_seq_len), (window_size, 0),
                           kv_seq_len - q_seq_len) for _ in range(num_heads)
    ]
    mask = mask_lib.MultiHeadMask(tuple((m for m in masks)))
    block_sizes = splash.BlockSizes.get_default()

    if is_mqa:
        attn = splash.make_splash_mqa_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    else:
        attn = splash.make_splash_mha_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    attn = jax.vmap(attn)
    outputs = attn(q, k, v, None)

    return outputs


def sharded_splash_attention(
    mesh: Mesh,
    window_size: Optional[int] = None,
    attn_logits_soft_cap: Optional[float] = None,
    is_mqa: bool = False,
) -> Callable[..., Any]:
    in_specs = (
        P("data", "model", None, None),  # q
        P("data", "model", None, None),  # k
        P("data", "model", None, None),  # vx
    )
    out_specs = P("data", "model", None, None)
    return jax.jit(
        jax.shard_map(
            functools.partial(
                apply_splash,
                window_size=window_size,
                attn_logits_soft_cap=attn_logits_soft_cap,
                is_mqa=is_mqa,
            ),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        ))


def sharded_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    pcp_mode: PcpMode = PcpMode.DISABLED,
    use_pcp: bool = False,
    use_pcp_decode: bool = False,
    shard_pcp_axis: bool = True,
    cp_kv_cache_interleave_size: int = 0,
    pcp_kv_lens: jax.Array | None = None,
    pcp_page_indices: jax.Array | None = None,
    pcp_query_start_loc: jax.Array | None = None,
    pcp_request_distribution: jax.Array | None = None,
    pcp_q_start_offsets: jax.Array | None = None,
    pcp_cu_k_lens: jax.Array | None = None,
    pcp_slot_ids: jax.Array | None = None,
    pcp_source_block_tables: jax.Array | None = None,
    pcp_streaming_schedule: jax.Array | None = None,
    pcp_streaming_active_page_groups: jax.Array | None = None,
):
    """Shards along KV heads."""
    if use_pcp_decode:
        if pcp_mode not in (PcpMode.DISABLED, PcpMode.DECODE_SHARDED_KV):
            raise ValueError("Conflicting PCP mode and use_pcp_decode=True.")
        pcp_mode = PcpMode.DECODE_SHARDED_KV
    elif use_pcp:
        if pcp_mode not in (PcpMode.DISABLED, PcpMode.PREFILL_LOCAL_Q_FULL_KV):
            raise ValueError("Conflicting PCP mode and use_pcp=True.")
        pcp_mode = PcpMode.PREFILL_LOCAL_Q_FULL_KV

    # Handle GQA/MQA where num_kv_heads < tp_size
    # We replicate KV heads to match tp_size so that we can shard them evenly.
    # TODO (ranlihao): This is not performant and introduces extra overhead during inference. We need to handle this during weight loading
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    if tp_size > 1:
        num_kv_heads = k.shape[1]
        if num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"For GQA/MQA, tp_size {tp_size} must be divisible by num_kv_heads {num_kv_heads}"
                )
            factor = tp_size // num_kv_heads
            k = jnp.repeat(k, factor, axis=1)
            v = jnp.repeat(v, factor, axis=1)

    if pcp_mode == PcpMode.DECODE_SHARDED_KV:
        return sharded_pcp_decode_ragged_paged_attention(
            mesh=mesh,
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=attention_sink,
            sm_scale=sm_scale,
            attention_chunk_size=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            update_kv_cache=update_kv_cache,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            pcp_slot_ids=pcp_slot_ids,
            pcp_source_block_tables=pcp_source_block_tables,
        )

    if pcp_mode == PcpMode.PREFILL_LOCAL_Q_FULL_KV:
        return sharded_pcp_ragged_paged_attention(
            mesh=mesh,
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=attention_sink,
            sm_scale=sm_scale,
            attention_chunk_size=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            update_kv_cache=update_kv_cache,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            pcp_kv_lens=pcp_kv_lens,
            pcp_page_indices=pcp_page_indices,
            pcp_query_start_loc=pcp_query_start_loc,
            pcp_request_distribution=pcp_request_distribution,
            pcp_q_start_offsets=pcp_q_start_offsets,
            pcp_cu_k_lens=pcp_cu_k_lens,
            pcp_slot_ids=pcp_slot_ids,
            pcp_streaming_schedule=pcp_streaming_schedule,
            pcp_streaming_active_page_groups=(
                pcp_streaming_active_page_groups),
        )

    data_axis = (ShardingAxisName.ATTN_DATA
                 if shard_pcp_axis else ShardingAxisName.BATCH)
    qkv_spec = P(data_axis, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(data_axis, None, ShardingAxisName.KV_CACHE_HEAD, None,
                      None)
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        kv_cache_spec,  # kv cache
        P(data_axis),  # kv_lens
        P(data_axis),  # page_indices
        P(data_axis),  # cu_q_lens
        P(data_axis),  # distribution
    )
    out_specs = (qkv_spec, kv_cache_spec)

    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    use_hd64 = q.shape[-1] == 64
    func = ragged_paged_attention_hd64 if use_hd64 else ragged_paged_attention

    if attention_sink is not None:
        if not use_hd64:
            raise NotImplementedError(
                "Attention sink support is only available when head_dim==64")

        in_specs += (P(ShardingAxisName.ATTN_HEAD), )
        args += (attention_sink, )

    # update_kv_cache=False (KV-share) is supported by the v3 default RPA
    # kernel and by the experimental batched RPA kernel. The hd64 path
    # doesn't accept it; fail loud rather than silently ignoring.
    if use_hd64 and not update_kv_cache:
        raise NotImplementedError(
            "update_kv_cache=False (KV-share) is not supported on the "
            "head_dim==64 RPA kernel.")

    def _ragged_paged_attention(*args):
        kwargs = dict(
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # update_kv_cache is supported by both the v3 default and batched
        # RPA kernels; only the hd64 path doesn't accept it. Default True
        # is a no-op so we don't forward it to the hd64 signature.
        if not use_hd64:
            kwargs["update_kv_cache"] = update_kv_cache
        return func(*args, **kwargs)

    return jax.shard_map(
        _ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def _pcp_chunks_per_seq(max_num_tokens: int, pcp_size: int,
                        interleave_size: int) -> int:
    if interleave_size <= 0:
        raise ValueError("PCP requires cp_kv_cache_interleave_size > 0.")
    return max(1, math.ceil(max_num_tokens / (pcp_size * interleave_size)))


def _make_pcp_interleaved_token_indices(
    seq_lens: jax.Array,
    local_num_tokens: int,
    interleave_size: int,
    *,
    pcp_size: int,
) -> jax.Array:
    """Map contiguous sequence token order to rank-major interleaved order."""
    max_num_seqs = seq_lens.shape[0]
    max_total_tokens = local_num_tokens * pcp_size
    chunks_per_seq = _pcp_chunks_per_seq(max_total_tokens, pcp_size,
                                         interleave_size)
    token_idx = jnp.arange(max_total_tokens, dtype=jnp.int32)

    seq_starts = jnp.pad(jnp.cumsum(seq_lens), (1, 0))[:-1]
    seq_ends = seq_starts + seq_lens
    seq_mask = ((token_idx[:, None] >= seq_starts[None, :])
                & (token_idx[:, None] < seq_ends[None, :]))
    valid = jnp.any(seq_mask, axis=1)
    seq_idx = jnp.argmax(seq_mask.astype(jnp.int32), axis=1)

    token_offset = jnp.where(valid, token_idx - seq_starts[seq_idx], 0)
    token_rank = (token_offset // interleave_size) % pcp_size
    token_chunk = token_offset // (interleave_size * pcp_size)
    token_chunk = jnp.minimum(token_chunk, chunks_per_seq - 1)
    token_offset_in_chunk = token_offset % interleave_size

    ranks = jnp.arange(pcp_size, dtype=jnp.int32)[:, None, None]
    chunks = jnp.arange(chunks_per_seq, dtype=jnp.int32)[None, None, :]
    chunk_starts = (chunks * pcp_size + ranks) * interleave_size
    chunk_lens = jnp.clip(seq_lens[None, :, None] - chunk_starts, 0,
                          interleave_size)
    local_cu_lens_by_rank = jnp.pad(
        jnp.cumsum(chunk_lens.reshape(pcp_size, -1), axis=1),
        ((0, 0), (1, 0)),
    )

    chunk_seq_idx = seq_idx * chunks_per_seq + token_chunk
    local_offset = (local_cu_lens_by_rank[token_rank, chunk_seq_idx] +
                    token_offset_in_chunk)
    src_idx = token_rank * local_num_tokens + local_offset
    return jnp.where(valid, src_idx, 0)


def _make_pcp_interleaved_metadata(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    local_num_tokens: int,
    interleave_size: int,
    *,
    pcp_size: int,
    axis_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build chunked local-Q/full-KV metadata for one PCP shard."""
    pcp_rank = jax.lax.axis_index(axis_name)
    max_num_seqs = kv_lens.shape[0]
    chunks_per_seq = _pcp_chunks_per_seq(local_num_tokens * pcp_size, pcp_size,
                                         interleave_size)

    global_q_starts = cu_q_lens[:-1]
    global_q_ends = cu_q_lens[1:]
    global_q_lens = global_q_ends - global_q_starts

    chunk_ids = jnp.arange(chunks_per_seq, dtype=jnp.int32)
    chunk_offsets = (chunk_ids * pcp_size + pcp_rank) * interleave_size
    local_q_lens = jnp.clip(global_q_lens[:, None] - chunk_offsets[None, :], 0,
                            interleave_size)

    # TODO(xiaohao.yxh): Treating every interleaved chunk as a pseudo sequence
    # can create many tiny sequences when the interleave size is small. That
    # hurts RPA schedule overhead and tile utilization; replace this with native
    # multi-chunk-per-sequence support in the kernel scheduler.
    flat_local_q_lens = local_q_lens.reshape(-1)
    local_cu_q_lens = jnp.pad(jnp.cumsum(flat_local_q_lens), (1, 0))

    q_start_offsets = ((kv_lens - global_q_lens)[:, None] +
                       chunk_offsets[None, :])
    q_start_offsets = jnp.where(local_q_lens > 0, q_start_offsets, 0)
    q_start_offsets = q_start_offsets.reshape(-1)

    expanded_kv_lens = jnp.where(local_q_lens > 0, kv_lens[:, None],
                                 0).reshape(-1)
    k_start_offsets = jnp.pad(jnp.cumsum(kv_lens), (1, 0))[:-1]
    cu_k_lens = jnp.where(local_q_lens > 0, k_start_offsets[:, None],
                          0).reshape(-1)
    cu_k_lens = jnp.concatenate(
        [cu_k_lens,
         jnp.array([jnp.sum(kv_lens)], dtype=cu_k_lens.dtype)])

    pages_per_seq = page_indices.shape[0] // max_num_seqs
    page_indices_by_seq = page_indices.reshape(max_num_seqs, pages_per_seq)
    expanded_page_indices = jnp.broadcast_to(
        page_indices_by_seq[:, None, :],
        (max_num_seqs, chunks_per_seq, pages_per_seq),
    ).reshape(-1)
    expanded_distribution = jnp.array([0, 0, max_num_seqs * chunks_per_seq],
                                      dtype=kv_lens.dtype)
    return (expanded_kv_lens, expanded_page_indices, local_cu_q_lens,
            expanded_distribution, q_start_offsets, cu_k_lens)


def _update_local_paged_kv_cache(
    kv_cache: jax.Array,
    k: jax.Array,
    v: jax.Array,
    slot_ids: jax.Array,
) -> jax.Array:
    """Write local K/V rows into the local paged KV cache shard."""
    dummy_q = jnp.zeros_like(k)
    _, packed_kv = batched_rpa_wrapper.prepare_inputs(dummy_q, k, v, k.dtype,
                                                      kv_cache.dtype)
    page_size = kv_cache.shape[1]
    valid = slot_ids >= 0
    page = jnp.where(valid, slot_ids // page_size, kv_cache.shape[0])
    offset = jnp.where(valid, slot_ids % page_size, 0)
    return kv_cache.at[page, offset].set(packed_kv, mode="drop")


def _materialize_gathered_pcp_kv_for_decode(
    gathered_kv_cache: jax.Array,
    kv_lens: jax.Array,
    source_block_tables: jax.Array,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Repack gathered PCP KV cache into standard paged KV layout."""
    if pcp_size <= 1:
        raise ValueError("PCP decode materialize requires pcp_size > 1.")
    if gathered_kv_cache.shape[0] != pcp_size:
        raise ValueError(
            "PCP decode materialize requires gathered PCP rank dimension "
            f"to match pcp_size: got {gathered_kv_cache.shape[0]} vs "
            f"{pcp_size}.")
    if gathered_kv_cache.shape[1] != source_block_tables.size:
        raise ValueError(
            "PCP decode materialize requires compact gathered page dimension "
            "to match source_block_tables.size: got "
            f"{gathered_kv_cache.shape[1]} vs {source_block_tables.size}.")
    if interleave_size <= 0:
        raise ValueError(
            "PCP decode materialize requires interleave_size > 0.")
    if page_size % interleave_size != 0:
        raise ValueError(
            "PCP decode materialize requires page_size % interleave_size == 0."
        )

    max_num_reqs = kv_lens.shape[0]
    virtual_blocks_per_req = source_block_tables.shape[1]
    standard_pages_per_req = virtual_blocks_per_req * pcp_size
    max_tokens_per_req = standard_pages_per_req * page_size
    max_total_dst_pages = max_num_reqs * standard_pages_per_req

    req_indices = jnp.arange(max_num_reqs, dtype=jnp.int32)[:, None]
    positions = jnp.arange(max_tokens_per_req, dtype=jnp.int32)[None, :]
    valid = positions < kv_lens[:, None]

    virtual_block_size = page_size * pcp_size
    virtual_blocks = positions // virtual_block_size
    virtual_offsets = positions % virtual_block_size
    src_ranks = (virtual_offsets // interleave_size) % pcp_size
    src_offsets = (
        (virtual_offsets // (pcp_size * interleave_size)) * interleave_size +
        (virtual_offsets % interleave_size))
    compact_src_pages = req_indices * virtual_blocks_per_req + virtual_blocks
    values = gathered_kv_cache[src_ranks, compact_src_pages, src_offsets]

    dst_page_indices = positions // page_size
    dst_pages = req_indices * standard_pages_per_req + dst_page_indices
    dst_offsets = positions % page_size
    dst_pages = jnp.where(valid, dst_pages, max_total_dst_pages)

    full_kv_cache = jnp.zeros(
        (max_total_dst_pages, page_size, *gathered_kv_cache.shape[3:]),
        dtype=gathered_kv_cache.dtype,
    )
    full_kv_cache = full_kv_cache.at[dst_pages, dst_offsets].set(values,
                                                                 mode="drop")
    full_page_indices_2d = (
        jnp.arange(max_num_reqs, dtype=jnp.int32)[:, None] *
        standard_pages_per_req +
        jnp.arange(standard_pages_per_req, dtype=jnp.int32)[None, :])
    return full_kv_cache, kv_lens, full_page_indices_2d.reshape(-1)


def materialize_pcp_kv_for_decode(
    pcp_kv_cache: jax.Array,
    kv_lens: jax.Array,
    source_block_tables: jax.Array,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    pcp_axis_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Materialize PCP-sharded KV cache into standard PCP=1 decode layout."""
    if pcp_size <= 1:
        raise ValueError("PCP decode materialize requires pcp_size > 1.")
    if interleave_size <= 0:
        raise ValueError(
            "PCP decode materialize requires interleave_size > 0.")
    if page_size % interleave_size != 0:
        raise ValueError(
            "PCP decode materialize requires page_size % interleave_size == 0."
        )

    compact_source_pages = source_block_tables.reshape(-1)
    compact_pcp_kv_cache = pcp_kv_cache[compact_source_pages]
    gathered_kv_cache = jax.lax.all_gather(
        compact_pcp_kv_cache,
        axis_name=pcp_axis_name,
        axis=0,
        tiled=False,
    )
    return _materialize_gathered_pcp_kv_for_decode(
        gathered_kv_cache,
        kv_lens,
        source_block_tables,
        page_size,
        pcp_size,
        interleave_size,
    )


def _pcp_lse_merge_weight(partial_lse: jax.Array,
                          all_lses: jax.Array) -> jax.Array:
    max_lse = jnp.max(all_lses, axis=0)
    valid = max_lse != -jnp.inf
    safe_local_diff = jnp.where(valid, partial_lse - max_lse, 0.0)
    local_exp = jnp.exp(safe_local_diff)

    safe_all_diffs = jnp.where(valid[None], all_lses - max_lse[None], 0.0)
    denom = jnp.sum(jnp.exp(safe_all_diffs), axis=0)
    weight = local_exp / jnp.maximum(denom, 1e-30)
    return jnp.where(valid, weight, 0.0)


def pcp_lse_merge(
    partial_out: jax.Array,
    partial_lse: jax.Array,
    pcp_axis_name: str,
) -> jax.Array:
    """Merge PCP partial attention outputs using log-sum-exp weights.

    This runs inside a shard_map over the PCP axis. Each rank supplies the
    attention result for its local KV shard. Rows where all ranks are empty
    produce zero output instead of NaNs.
    """
    all_lses = jax.lax.all_gather(partial_lse,
                                  axis_name=pcp_axis_name,
                                  axis=0,
                                  tiled=False)
    weight = _pcp_lse_merge_weight(partial_lse, all_lses)
    weighted_out = partial_out.astype(jnp.float32) * weight[..., None]
    merged = jax.lax.psum(weighted_out, axis_name=pcp_axis_name)
    return merged.astype(partial_out.dtype)


def compute_pcp_local_mapping(
    positions: jax.Array,
    token_req_indices: jax.Array,
    block_tables: jax.Array,
    block_size: int,
    cp_size: int,
    cp_rank: int,
    interleave_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute vLLM-compatible local CP slot mapping for one PCP rank."""
    if block_size <= 0:
        raise ValueError(f"Expected positive block_size, got {block_size}.")
    if cp_size <= 0:
        raise ValueError(f"Expected positive cp_size, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"Expected cp_rank in [0, {cp_size}), got {cp_rank}.")
    if interleave_size <= 0:
        raise ValueError(
            f"Expected positive interleave_size, got {interleave_size}.")

    positions = positions.astype(jnp.int32)
    token_req_indices = token_req_indices.astype(jnp.int32)
    virtual_block_size = block_size * cp_size
    block_indices = positions // virtual_block_size
    block_numbers = block_tables[token_req_indices,
                                 block_indices].astype(jnp.int32)

    virtual_offsets = positions - block_indices * virtual_block_size
    is_local = ((virtual_offsets // interleave_size) % cp_size) == cp_rank
    local_offsets = ((virtual_offsets //
                      (cp_size * interleave_size)) * interleave_size +
                     (virtual_offsets % interleave_size))
    slot_ids = block_numbers * block_size + local_offsets
    slot_ids = jnp.where(is_local, slot_ids, -1)
    return is_local, slot_ids.astype(jnp.int32)


def sharded_pcp_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    cp_kv_cache_interleave_size: int = 0,
    pcp_kv_lens: jax.Array | None = None,
    pcp_page_indices: jax.Array | None = None,
    pcp_query_start_loc: jax.Array | None = None,
    pcp_request_distribution: jax.Array | None = None,
    pcp_q_start_offsets: jax.Array | None = None,
    pcp_cu_k_lens: jax.Array | None = None,
    pcp_slot_ids: jax.Array | None = None,
    pcp_streaming_schedule: jax.Array | None = None,
    pcp_streaming_active_page_groups: jax.Array | None = None,
):
    """Runs local-query/full-KV RPA over the prefill context axis."""
    if attention_sink is not None:
        raise NotImplementedError("PCP RPA does not support attention sinks.")
    if q.shape[-1] == 64:
        raise NotImplementedError("PCP RPA does not support head_dim==64.")
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError("PCP RPA requires cp_kv_cache_interleave_size > 0.")
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT
    if pcp_axis is None:
        raise NotImplementedError("PCP requires a named prefill context axis.")
    pcp_size = mesh.shape[pcp_axis]
    if get_mesh_shape_product(mesh, ShardingAxisName.CONTEXT) > 1:
        raise NotImplementedError("PCP RPA does not support DCP yet.")

    precomputed_pcp_metadata = (
        pcp_kv_lens,
        pcp_page_indices,
        pcp_query_start_loc,
        pcp_request_distribution,
        pcp_q_start_offsets,
        pcp_cu_k_lens,
        pcp_slot_ids,
    )
    has_precomputed_pcp_metadata = any(x is not None
                                       for x in precomputed_pcp_metadata)
    if has_precomputed_pcp_metadata and not all(
            x is not None for x in precomputed_pcp_metadata):
        raise ValueError(
            "PCP RPA requires all precomputed PCP metadata fields when any "
            "one of them is provided.")

    streaming_requested = (envs.USE_PCP_STREAMING_RPA_KERNEL
                           and pcp_streaming_schedule is not None)
    if streaming_requested:
        if pcp_streaming_active_page_groups is None:
            raise ValueError("PCP streaming RPA requires active page groups.")
        if not has_precomputed_pcp_metadata:
            raise ValueError(
                "PCP streaming RPA requires precomputed PCP metadata from "
                "the runner.")
        if attention_chunk_size is not None:
            raise NotImplementedError(
                "PCP streaming RPA supports full attention only.")
        if q_scale is not None or k_scale is not None or v_scale is not None:
            raise NotImplementedError(
                "PCP streaming RPA does not support quantized Q/K/V scales.")
        if not update_kv_cache:
            raise NotImplementedError(
                "PCP streaming RPA requires update_kv_cache=True.")
        if cp_kv_cache_interleave_size != kv_cache.shape[1]:
            raise NotImplementedError(
                "PCP streaming RPA currently requires "
                "cp_kv_cache_interleave_size == page_size.")
    elif not _ragged_paged_attention_accepts_pcp_metadata(
            ragged_paged_attention):
        raise NotImplementedError(
            "PCP RPA requires the batched RPA wrapper with local-Q/full-KV "
            "metadata support.")

    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.KV_CACHE_HEAD,
                 None)
    kv_cache_spec = P(ShardingAxisName.KV_CACHE_BLOCK, None,
                      ShardingAxisName.KV_CACHE_HEAD, None, None)
    metadata_spec = P(ShardingAxisName.BATCH)
    pcp_metadata_spec = P(ShardingAxisName.ATTN_DATA)
    if pcp_streaming_schedule is not None and pcp_streaming_schedule.ndim == 4:
        pcp_streaming_schedule_spec = P(None, None, None, None)
    else:
        pcp_streaming_schedule_spec = P(ShardingAxisName.BATCH, None, None,
                                        None, None)
    if (pcp_streaming_active_page_groups is not None
            and pcp_streaming_active_page_groups.ndim == 1):
        pcp_streaming_active_page_groups_spec = P(None)
    else:
        pcp_streaming_active_page_groups_spec = P(ShardingAxisName.BATCH, None)
    in_specs = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        kv_cache_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
    )
    out_specs = (qkv_spec, kv_cache_spec)
    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)
    if has_precomputed_pcp_metadata:
        in_specs += (
            pcp_metadata_spec,  # pcp_kv_lens
            pcp_metadata_spec,  # pcp_page_indices
            pcp_metadata_spec,  # pcp_query_start_loc
            pcp_metadata_spec,  # pcp_request_distribution
            pcp_metadata_spec,  # pcp_q_start_offsets
            pcp_metadata_spec,  # pcp_cu_k_lens
            pcp_metadata_spec,  # pcp_slot_ids
        )
        args += precomputed_pcp_metadata
    if streaming_requested:
        in_specs += (pcp_streaming_schedule_spec,
                     pcp_streaming_active_page_groups_spec)
        args += (pcp_streaming_schedule, pcp_streaming_active_page_groups)

    def _pcp_ragged_paged_attention(q_local, k_local, v_local, kv_cache,
                                    kv_lens, page_indices, cu_q_lens,
                                    distribution, *extra_pcp_args):
        num_pcp_metadata_args = (len(precomputed_pcp_metadata)
                                 if has_precomputed_pcp_metadata else 0)
        pcp_metadata_args = extra_pcp_args[:num_pcp_metadata_args]
        streaming_schedule_arg = (extra_pcp_args[num_pcp_metadata_args]
                                  if streaming_requested else None)
        streaming_active_page_groups_arg = (
            extra_pcp_args[num_pcp_metadata_args + 1]
            if streaming_requested else None)
        kv_indices = _make_pcp_interleaved_token_indices(
            kv_lens,
            k_local.shape[0],
            cp_kv_cache_interleave_size,
            pcp_size=pcp_size,
        )
        if pcp_metadata_args:
            (expanded_kv_lens, expanded_page_indices, local_cu_q_lens,
             expanded_distribution, q_start_offsets, cu_k_lens,
             slot_ids) = pcp_metadata_args
        else:
            if update_kv_cache:
                raise ValueError(
                    "PCP local KV cache updates require precomputed "
                    "pcp_slot_ids from the runner.")
            (expanded_kv_lens, expanded_page_indices, local_cu_q_lens,
             expanded_distribution, q_start_offsets,
             cu_k_lens) = _make_pcp_interleaved_metadata(
                 cu_q_lens,
                 kv_lens,
                 page_indices,
                 local_num_tokens=q_local.shape[0],
                 interleave_size=cp_kv_cache_interleave_size,
                 pcp_size=pcp_size,
                 axis_name=pcp_axis,
             )
            slot_ids = None
        if update_kv_cache:
            kv_cache = _update_local_paged_kv_cache(kv_cache, k_local, v_local,
                                                    slot_ids)
        if streaming_requested:
            if q_local.shape[1] % k_local.shape[1] != 0:
                raise ValueError("Q heads must be divisible by KV heads.")
            q_per_kv = q_local.shape[1] // k_local.shape[1]
            q_streaming = q_local.reshape(q_local.shape[0], k_local.shape[1],
                                          q_per_kv, q_local.shape[2])
            if streaming_schedule_arg.ndim == 5:
                streaming_schedule_arg = streaming_schedule_arg[0]
            if streaming_active_page_groups_arg.ndim == 2:
                streaming_active_page_groups_arg = (
                    streaming_active_page_groups_arg[0])
            output = pcp_streaming_attention_page_groups_packed_local(
                q_streaming,
                kv_cache,
                streaming_schedule_arg,
                streaming_active_page_groups_arg,
                pcp_size=pcp_size,
                q_block_size=envs.PCP_STREAMING_RPA_Q_BLOCK_SIZE,
                sm_scale=sm_scale,
                collective_id=23,
                kv_pages_per_block=max(
                    1,
                    min(
                        ScheduleField.MAX_KV_PAGES_PER_BLOCK,
                        envs.PCP_STREAMING_RPA_KV_BLOCK_SIZE //
                        kv_cache.shape[1],
                    ),
                ),
                mesh_axis_names=tuple(mesh.axis_names),
                pcp_axis_name=pcp_axis,
            )
            return output.reshape(q_local.shape), kv_cache
        full_k = jax.lax.all_gather(k_local,
                                    axis_name=pcp_axis,
                                    axis=0,
                                    tiled=True)
        full_v = jax.lax.all_gather(v_local,
                                    axis_name=pcp_axis,
                                    axis=0,
                                    tiled=True)
        full_k = full_k[kv_indices]
        full_v = full_v[kv_indices]
        return ragged_paged_attention(
            q_local,
            full_k,
            full_v,
            kv_cache,
            expanded_kv_lens,
            expanded_page_indices,
            local_cu_q_lens,
            expanded_distribution,
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            update_kv_cache=False,
            q_start_offsets=q_start_offsets,
            cu_k_lens=cu_k_lens,
        )

    return jax.shard_map(
        _pcp_ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def sharded_pcp_decode_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    cp_kv_cache_interleave_size: int = 0,
    pcp_slot_ids: jax.Array | None = None,
    pcp_source_block_tables: jax.Array | None = None,
):
    """Runs replicated-query/local-KV decode over the prefill context axis."""
    if attention_sink is not None:
        raise NotImplementedError("PCP decode RPA does not support sinks.")
    if attention_chunk_size is not None:
        raise NotImplementedError(
            "PCP decode materialize path supports full attention only.")
    if q.shape[-1] == 64:
        raise NotImplementedError(
            "PCP decode RPA does not support head_dim==64.")
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError(
            "PCP decode RPA requires cp_kv_cache_interleave_size > 0.")
    pcp_axis = ShardingAxisName.PREFILL_CONTEXT
    if pcp_axis is None:
        raise NotImplementedError("PCP decode requires a named PCP axis.")
    if get_mesh_shape_product(mesh, ShardingAxisName.CONTEXT) > 1:
        raise NotImplementedError("PCP decode RPA does not support DCP yet.")
    pcp_size = get_mesh_shape_product(mesh, pcp_axis)
    if pcp_slot_ids is None or pcp_source_block_tables is None:
        raise ValueError(
            "PCP decode materialize path requires pcp_slot_ids and "
            "pcp_source_block_tables from the runner.")

    qkv_spec = P(ShardingAxisName.BATCH, ShardingAxisName.KV_CACHE_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.KV_CACHE_BLOCK, None,
                      ShardingAxisName.KV_CACHE_HEAD, None, None)
    metadata_spec = P(ShardingAxisName.BATCH)
    source_block_tables_spec = P(ShardingAxisName.BATCH, None)
    pcp_metadata_spec = P(ShardingAxisName.ATTN_DATA)
    in_specs = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        kv_cache_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
        source_block_tables_spec,
        pcp_metadata_spec,
    )
    out_specs = (qkv_spec, kv_cache_spec)
    args = (q, k, v, kv_cache, cu_q_lens, distribution, kv_lens,
            pcp_source_block_tables, pcp_slot_ids)

    def _pcp_decode_ragged_paged_attention(q_replicated, k_replicated,
                                           v_replicated, kv_cache, cu_q_lens,
                                           distribution, global_kv_lens,
                                           source_block_tables,
                                           local_slot_ids):
        if update_kv_cache:
            kv_cache = _update_local_paged_kv_cache(kv_cache, k_replicated,
                                                    v_replicated,
                                                    local_slot_ids)
        full_kv_cache, full_kv_lens, full_page_indices = (
            materialize_pcp_kv_for_decode(
                kv_cache,
                global_kv_lens,
                source_block_tables,
                kv_cache.shape[1],
                pcp_size,
                cp_kv_cache_interleave_size,
                pcp_axis,
            ))
        output, _ = ragged_paged_attention(
            q_replicated,
            k_replicated,
            v_replicated,
            full_kv_cache,
            full_kv_lens,
            full_page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=None,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            update_kv_cache=False,
        )
        return output, kv_cache

    return jax.shard_map(
        _pcp_decode_ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,  # before padding,
    sm_scale: float | None = None,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sinks: jax.Array | None = None,
    update_kv_cache: bool = True,
    pcp_mode: PcpMode = PcpMode.DISABLED,
    use_pcp: bool = False,
    use_pcp_decode: bool = False,
    shard_pcp_axis: bool = True,
    cp_kv_cache_interleave_size: int = 0,
) -> Tuple[jax.Array, jax.Array]:
    # T: seq_len
    # N: num_heads
    # K: num_kv_heads
    # D: hidden_size
    # H: head_dim
    # L: num_blocks
    # S: block_size

    # TODO(jevinjiang, cuiq): transpose q weight offline.
    # q: (T, N, H)
    # k,v: (T, K, H)

    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata

    # (T, N, H)
    output, kv_cache = sharded_ragged_paged_attention(
        mesh,
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        sinks,
        sm_scale=sm_scale,
        attention_chunk_size=attention_chunk_size,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        update_kv_cache=update_kv_cache,
        pcp_mode=pcp_mode,
        use_pcp=use_pcp,
        use_pcp_decode=use_pcp_decode,
        shard_pcp_axis=shard_pcp_axis,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        pcp_kv_lens=md.pcp_kv_lens,
        pcp_page_indices=md.pcp_page_indices,
        pcp_query_start_loc=md.pcp_query_start_loc,
        pcp_request_distribution=md.pcp_request_distribution,
        pcp_q_start_offsets=md.pcp_q_start_offsets,
        pcp_cu_k_lens=md.pcp_cu_k_lens,
        pcp_slot_ids=md.pcp_slot_ids,
        pcp_source_block_tables=md.pcp_source_block_tables,
        pcp_streaming_schedule=md.pcp_streaming_schedule,
        pcp_streaming_active_page_groups=(
            md.pcp_streaming_active_page_groups),
    )

    return kv_cache, output


def mla_attention(
        q_NTA: jax.Array,
        q_rope_TNH: jax.Array,
        k_SA: jax.Array,
        k_rope_SH: jax.Array,
        kv_cache: jax.Array,
        md: AttentionMetadata,
        mesh: Mesh,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        query_nth_sharding: Sharding | None = None,
        query_tnh_sharding: Sharding | None = None,
        keyvalue_skh_sharding: Sharding | None = None,
        attn_o_nth_sharding: Sharding | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        sm_scale: float | None = None) -> Tuple[jax.Array, jax.Array]:
    """
    Main shared interface for MLA attention.  Computes the sharded attention
    output and kv cache update.

    Args:
        q_NTA: (num_query_heads, tokens_query, q_lora_rank) # head-major output from q_nope einsum projection.
        q_rope_TNH: (tokens_query, num_query_heads, head_dim)
        k_SA: (tokens_kv, q_lora_rank)
        k_rope_SH: (tokens_kv, head_dim)
        kv_cache: KV cache to be retrieved from/updated
        md: attention metadata
        mesh: Mesh
        num_attention_heads: number of attention heads
        qk_nope_head_dim: head dim for QK without rope
        query_nth_sharding: sharding to use for q_nope for the shard map (MLA kernel)
        query_tnh_sharding: sharding to use for q_rope for the shard map (MLA kernel)
        keyvalue_skh_sharding: sharding to use for k/k_rope for the shard map (MLA kernel)
        attn_o_nth_sharding: sharding to use for the attention output for the shard map (MLA kernel)
        q_scale: scale to apply to q (if quantized)
        k_scale: scale to apply to k (if quantized)
        v_scale: scale to apply to v (if quantized)
        sm_scale: softmax scale
    """
    in_specs = (
        query_nth_sharding
        or P(None, ShardingAxisName.MLP_TENSOR, None),  # q (head-major)
        query_tnh_sharding
        or P(ShardingAxisName.MLP_TENSOR, None, None),  # q_rope (token-major)
        keyvalue_skh_sharding or P(ShardingAxisName.MLP_TENSOR, None),  # k
        keyvalue_skh_sharding
        or P(ShardingAxisName.MLP_TENSOR, None),  # k_rope
        P(ShardingAxisName.BATCH),  # kv_cache
        P(ShardingAxisName.ATTN_DATA),  # md.seq_lens
        P(ShardingAxisName.ATTN_DATA),  # md.page_indices_flat
        P(ShardingAxisName.ATTN_DATA),  # md.query_start_loc
        P(ShardingAxisName.ATTN_DATA),  # md.distribution
    )
    out_specs = (
        P(ShardingAxisName.BATCH),  # kv cache
        attn_o_nth_sharding
        or P(None, ShardingAxisName.MLP_TENSOR, None)  # attn output
    )

    def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, *args):
        # TODO: use auto tuner to find the best block sizes.
        num_kv_pages_per_block = (3, 1, 1)
        num_queries_per_block = (1, 16, 16)
        decode_batch_size = 4

        out, new_cache = mla_ragged_paged_attention(
            q,
            q_rope,
            k,
            k_rope,
            cache,
            *args,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            decode_batch_size=decode_batch_size,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale)

        return new_cache, out

    kv_cache, output_TNA = jax.jit(
        jax.shard_map(_mla_ragged_paged_attention,
                      mesh=mesh,
                      in_specs=in_specs,
                      out_specs=out_specs,
                      check_vma=False))(q_NTA, q_rope_TNH, k_SA, k_rope_SH,
                                        kv_cache, md.seq_lens, md.block_tables,
                                        md.query_start_loc,
                                        md.request_distribution)
    return kv_cache, output_TNA
