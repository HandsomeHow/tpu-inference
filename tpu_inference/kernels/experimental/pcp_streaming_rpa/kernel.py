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

"""PCP streaming prefill RPA kernels.

This module currently contains the first production-shaped MVP kernel for the
RingAttention-style PCP path. It intentionally supports only one page group,
one lane, one KV head, and one Q head per KV head. The goal is to keep the first
kernel small while validating the real schedule staging + ring communication +
online softmax control flow.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util
from tpu_inference.kernels.experimental.batched_rpa.utils import (
    broadcast_minor)
from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField)

P = jax.sharding.PartitionSpec
AXIS = "pcp"


def _consume_scheduled_kv_page(
    q_vmem_ref,
    kv_vmem_ref,
    sched_vmem_ref,
    slot,
    consumer_rank,
    m,
    l,
    acc,
    *,
    sm_scale,
):
    q = q_vmem_ref[...].astype(jnp.float32)
    k = kv_vmem_ref.at[slot, :, 0, :][...].astype(jnp.float32)
    v = kv_vmem_ref.at[slot, :, 1, :][...].astype(jnp.float32)

    q_global_start = sched_vmem_ref[
        consumer_rank, 0, ScheduleField.Q_GLOBAL_START]
    kv_global_start = sched_vmem_ref[
        consumer_rank, 0, ScheduleField.KV_GLOBAL_START]
    req_id = sched_vmem_ref[consumer_rank, 0, ScheduleField.REQ_ID]
    kv_valid_len = sched_vmem_ref[consumer_rank, 0,
                                  ScheduleField.KV_VALID_LEN]
    q_tile_size = sched_vmem_ref[consumer_rank, 0, ScheduleField.Q_TILE_SIZE]

    scores = jnp.matmul(q, k.T, preferred_element_type=jnp.float32) * sm_scale
    q_pos = q_global_start + lax.broadcasted_iota(jnp.int32, scores.shape, 0)
    kv_pos = kv_global_start + lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    kv_valid = lax.broadcasted_iota(jnp.int32, scores.shape, 1) < kv_valid_len
    q_valid = lax.broadcasted_iota(jnp.int32, scores.shape, 0) < q_tile_size
    entry_valid = req_id != -1
    row_active = jnp.logical_and(entry_valid,
                                 q_valid[:, :1])
    mask = jnp.logical_and(jnp.logical_and(q_pos >= kv_pos, kv_valid),
                           q_valid)
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jnp.where(row_active, scores, 0.0)

    m_curr = jnp.max(scores, axis=1, keepdims=True)
    m_next = jnp.where(row_active, jnp.maximum(m, m_curr), m)
    p = jnp.where(row_active,
                  jnp.exp(scores - broadcast_minor(m_next, scores.shape)),
                  0.0)
    alpha = jnp.where(row_active, jnp.exp(m - m_next), 1.0)
    l_next = alpha * l + jnp.sum(p, axis=1, keepdims=True)
    pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
    acc_next = broadcast_minor(alpha, acc.shape) * acc + pv
    return m_next, l_next, acc_next


def _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sem, step):
    load_op = pltpu.make_async_copy(
        src_ref=packed_schedule_ref.at[step],
        dst_ref=sched_vmem_ref.at[:, :, :],
        sem=sem,
    )
    load_op.start()
    load_op.wait()


def _source_page_idx_from_staged_schedule(sched_vmem_ref, source_rank,
                                          pcp_size):
    page_idx = jnp.array(0, dtype=jnp.int32)
    for consumer_rank in range(pcp_size):
        req_id = sched_vmem_ref[consumer_rank, 0, ScheduleField.REQ_ID]
        kv_page_rank = sched_vmem_ref[consumer_rank, 0,
                                      ScheduleField.KV_PAGE_RANK]
        candidate = jnp.logical_and(req_id != -1, kv_page_rank == source_rank)
        page_idx = jnp.where(candidate,
                             sched_vmem_ref[consumer_rank, 0,
                                            ScheduleField.KV_PAGE_IDX],
                             page_idx)
    return page_idx


def _pcp_streaming_attention_single_page_group_kernel(
    q_ref,
    kv_cache_ref,
    packed_schedule_ref,
    o_ref,
    sched_dma_sem,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    o_vmem_ref,
    *,
    pcp_size,
    q_block_size,
    page_size,
    sm_scale,
):
    my_id = lax.axis_index(AXIS)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)

    _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sched_dma_sem, 0)
    q_hbm_offset = sched_vmem_ref[my_id, 0, ScheduleField.Q_HBM_OFFSET]
    q_load = pltpu.make_async_copy(
        src_ref=q_ref.at[
            0,
            pl.ds(q_hbm_offset, q_block_size),
            0,
            0,
            :,
        ],
        dst_ref=q_vmem_ref.at[:, :],
        sem=local_dma_sem,
    )
    q_load.start()
    q_load.wait()

    _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sched_dma_sem,
                        my_id)
    local_page_idx = _source_page_idx_from_staged_schedule(
        sched_vmem_ref, my_id, pcp_size)
    kv_load = pltpu.make_async_copy(
        src_ref=kv_cache_ref.at[0, local_page_idx, :, 0, :, :],
        dst_ref=kv_vmem_ref.at[0],
        sem=local_dma_sem,
    )
    kv_load.start()
    kv_load.wait()

    o_vmem_ref[...] = jnp.zeros_like(o_vmem_ref)
    zero_store = pltpu.make_async_copy(
        src_ref=o_vmem_ref.at[:, :],
        dst_ref=o_ref.at[0, :, 0, 0, :],
        sem=local_dma_sem,
    )
    zero_store.start()
    zero_store.wait()

    util.local_barrier(prev_rank, next_rank)

    m = jnp.full((q_block_size, 128), -jnp.inf, dtype=jnp.float32)
    l = jnp.zeros((q_block_size, 128), dtype=jnp.float32)
    acc = jnp.zeros((q_block_size, q_vmem_ref.shape[1]), dtype=jnp.float32)

    for round_idx in range(pcp_size):
        curr_slot = round_idx % 2
        next_slot = 1 - curr_slot
        src_rank = lax.rem(my_id + pcp_size - round_idx, pcp_size)

        _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                            sched_dma_sem, src_rank)

        if round_idx < pcp_size - 1:
            remote_op = pltpu.make_async_remote_copy(
                src_ref=kv_vmem_ref.at[curr_slot],
                dst_ref=kv_vmem_ref.at[next_slot],
                send_sem=remote_send_sems.at[round_idx],
                recv_sem=remote_recv_sems.at[round_idx],
                device_id=(next_rank, ),
                device_id_type=pl.DeviceIdType.MESH,
            )
            remote_op.start()

        m, l, acc = _consume_scheduled_kv_page(
            q_vmem_ref,
            kv_vmem_ref,
            sched_vmem_ref,
            curr_slot,
            my_id,
            m,
            l,
            acc,
            sm_scale=sm_scale,
        )

        if round_idx < pcp_size - 1:
            remote_op.wait()

    l_broadcast = broadcast_minor(l, acc.shape)
    o_vmem_ref[...] = jnp.where(l_broadcast > 0, acc / l_broadcast,
                                0.0).astype(
        o_vmem_ref.dtype)
    _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sched_dma_sem,
                        pcp_size - 1)
    req_id = sched_vmem_ref[my_id, 0, ScheduleField.REQ_ID]
    o_hbm_offset = sched_vmem_ref[my_id, 0, ScheduleField.O_HBM_OFFSET]

    @pl.when(req_id != -1)
    def _store_output():
        o_store = pltpu.make_async_copy(
            src_ref=o_vmem_ref.at[:, :],
            dst_ref=o_ref.at[
                0,
                pl.ds(o_hbm_offset, q_block_size),
                0,
                0,
                :,
            ],
            sem=local_dma_sem,
        )
        o_store.start()
        o_store.wait()


def _validate_single_page_group_inputs(q_by_rank, kv_cache_by_rank,
                                       packed_schedule, pcp_size):
    if q_by_rank.ndim != 5:
        raise ValueError("q_by_rank must have shape "
                         "[pcp, local_tokens, kv_heads, q_per_kv, head_dim].")
    if kv_cache_by_rank.ndim != 6:
        raise ValueError("kv_cache_by_rank must have shape "
                         "[pcp, pages, page_size, kv_heads, 2, head_dim].")
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must have shape "
                         "[steps, pcp, lanes, packed_fields].")
    if q_by_rank.shape[0] != pcp_size or kv_cache_by_rank.shape[0] != pcp_size:
        raise ValueError("q_by_rank and kv_cache_by_rank must be sharded over "
                         "pcp_size ranks.")
    if packed_schedule.shape[0] != pcp_size:
        raise NotImplementedError(
            "single-page-group MVP requires exactly pcp_size schedule steps.")
    if packed_schedule.shape[1] != pcp_size or packed_schedule.shape[2] != 1:
        raise NotImplementedError(
            "single-page-group MVP supports exactly one lane.")
    if packed_schedule.shape[3] != ScheduleField.PACKED_NUM_FIELDS:
        raise ValueError("packed_schedule must use padded packed fields.")
    if q_by_rank.shape[2] != 1 or q_by_rank.shape[3] != 1:
        raise NotImplementedError(
            "single-page-group MVP supports kv_heads=1 and q_per_kv=1.")
    if kv_cache_by_rank.shape[3] != 1 or kv_cache_by_rank.shape[4] != 2:
        raise NotImplementedError(
            "single-page-group MVP expects KV cache shape [..., 1, 2, head_dim]."
        )
    if q_by_rank.shape[-1] != kv_cache_by_rank.shape[-1]:
        raise ValueError("Q and KV head_dim must match.")
    if q_by_rank.shape[-1] % 128 != 0:
        raise NotImplementedError(
            "single-page-group MVP requires head_dim to be 128-aligned.")


def pcp_streaming_attention_single_page_group(
    q_by_rank,
    kv_cache_by_rank,
    packed_schedule,
    *,
    pcp_size: int,
    sm_scale: float,
    collective_id: int | None = 13,
):
    """Run one RingAttention-style PCP page group.

    Args:
        q_by_rank: [pcp, local_tokens, 1, 1, head_dim].
        kv_cache_by_rank: [pcp, pages, page_size, 1, 2, head_dim].
        packed_schedule: [pcp, pcp, 1, 128] schedule for one page group.
        pcp_size: Number of PCP ranks.
        sm_scale: Attention softmax scale.
        collective_id: Pallas collective id used by the local ring barrier.

    Returns:
        Rank-local packed output with the same shape as q_by_rank.
    """
    _validate_single_page_group_inputs(q_by_rank, kv_cache_by_rank,
                                       packed_schedule, pcp_size)

    q_block_size = q_by_rank.shape[1]
    page_size = kv_cache_by_rank.shape[2]
    head_dim = q_by_rank.shape[-1]

    def _call(q, kv_cache, schedule):
        return pl.pallas_call(
            functools.partial(
                _pcp_streaming_attention_single_page_group_kernel,
                pcp_size=pcp_size,
                q_block_size=q_block_size,
                page_size=page_size,
                sm_scale=sm_scale,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (1, q_block_size, 1, 1, head_dim),
                jnp.float32,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA((pcp_size - 1, )),
                    pltpu.SemaphoreType.DMA((pcp_size - 1, )),
                    pltpu.VMEM((pcp_size, 1,
                                ScheduleField.PACKED_NUM_FIELDS),
                               schedule.dtype),
                    pltpu.VMEM((q_block_size, head_dim), q.dtype),
                    pltpu.VMEM((2, page_size, 2, head_dim), kv_cache.dtype),
                    pltpu.VMEM((q_block_size, head_dim), jnp.float32),
                ),
                grid=(1, ),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=8 * 1024 * 1024,
            ),
            name="pcp_streaming_attention_single_page_group",
        )(q, kv_cache, schedule)

    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))
    shard_map_kernel = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None, None),
            check_vma=False,
        ))
    return shard_map_kernel(q_by_rank, kv_cache_by_rank, packed_schedule)
