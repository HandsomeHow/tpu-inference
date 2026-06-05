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

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.pcp_streaming_rpa.kernel import (
    pcp_streaming_attention_page_groups,
    pcp_streaming_attention_page_groups_local,
    pcp_streaming_attention_page_groups_packed_local,
    pcp_streaming_attention_single_page_group)
from tpu_inference.kernels.experimental.pcp_streaming_rpa.reference import (
    execute_pcp_streaming_reference)
from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    PcpStreamingSchedule, generate_pcp_streaming_schedule,
    pack_pcp_streaming_schedule_fields)

PCP_SIZE = 4
Q_TILE = 8
PAGE_SIZE = 128
HEAD_DIM = 128
P = jax.sharding.PartitionSpec
AXIS = "pcp"

pytestmark = pytest.mark.skipif(
    jax.local_device_count() < PCP_SIZE
    or jax.local_devices()[0].platform != "tpu",
    reason="PCP streaming kernel test requires four TPU devices.",
)


def _make_single_page_group_schedule():
    shape = (PCP_SIZE, PCP_SIZE, 1)
    req_id = np.zeros(shape, dtype=np.int32)
    kv_page_rank = np.zeros(shape, dtype=np.int32)
    kv_page_idx = np.zeros(shape, dtype=np.int32)
    is_first_kv = np.zeros(shape, dtype=np.int32)
    is_last_kv = np.zeros(shape, dtype=np.int32)
    load_q = np.zeros(shape, dtype=np.int32)
    q_global_start = np.full(shape, 3 * PAGE_SIZE, dtype=np.int32)
    kv_global_start = np.zeros(shape, dtype=np.int32)
    kv_valid_len = np.full(shape, PAGE_SIZE, dtype=np.int32)
    q_hbm_offset = np.zeros(shape, dtype=np.int32)
    q_tile_size = np.full(shape, Q_TILE, dtype=np.int32)
    o_hbm_offset = np.zeros(shape, dtype=np.int32)

    for step in range(PCP_SIZE):
        kv_page_rank[:, step, 0] = step
        kv_global_start[:, step, 0] = step * PAGE_SIZE
    is_first_kv[:, 0, 0] = 1
    is_last_kv[:, PCP_SIZE - 1, 0] = 1
    load_q[:, 0, 0] = 1

    packed_schedule = pack_pcp_streaming_schedule_fields(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
    )
    return PcpStreamingSchedule(
        req_id=req_id,
        kv_page_rank=kv_page_rank,
        kv_page_idx=kv_page_idx,
        is_first_kv=is_first_kv,
        is_last_kv=is_last_kv,
        load_q=load_q,
        q_global_start=q_global_start,
        kv_global_start=kv_global_start,
        kv_valid_len=kv_valid_len,
        q_hbm_offset=q_hbm_offset,
        q_tile_size=q_tile_size,
        o_hbm_offset=o_hbm_offset,
        packed_schedule=packed_schedule,
        actual_steps=np.full((PCP_SIZE, ), PCP_SIZE, dtype=np.int32),
        global_actual_steps=np.array([PCP_SIZE], dtype=np.int32),
    )


def _truncate_schedule(schedule, steps, actual_steps):
    return PcpStreamingSchedule(
        req_id=schedule.req_id[:, :steps].copy(),
        kv_page_rank=schedule.kv_page_rank[:, :steps].copy(),
        kv_page_idx=schedule.kv_page_idx[:, :steps].copy(),
        is_first_kv=schedule.is_first_kv[:, :steps].copy(),
        is_last_kv=schedule.is_last_kv[:, :steps].copy(),
        load_q=schedule.load_q[:, :steps].copy(),
        q_global_start=schedule.q_global_start[:, :steps].copy(),
        kv_global_start=schedule.kv_global_start[:, :steps].copy(),
        kv_valid_len=schedule.kv_valid_len[:, :steps].copy(),
        q_hbm_offset=schedule.q_hbm_offset[:, :steps].copy(),
        q_tile_size=schedule.q_tile_size[:, :steps].copy(),
        o_hbm_offset=schedule.o_hbm_offset[:, :steps].copy(),
        packed_schedule=schedule.packed_schedule[:steps].copy(),
        actual_steps=np.asarray(actual_steps, dtype=np.int32),
        global_actual_steps=np.array([steps], dtype=np.int32),
    )


def _run_page_groups_local(q_global, kv_cache_by_rank, packed_schedule, *,
                           sm_scale, collective_id):
    mesh = jax.sharding.Mesh(jax.local_devices()[:PCP_SIZE], (AXIS, ))

    def _call(q_local, kv_cache_local, schedule):
        return pcp_streaming_attention_page_groups_local(
            q_local,
            kv_cache_local[0],
            schedule,
            pcp_size=PCP_SIZE,
            q_block_size=Q_TILE,
            sm_scale=sm_scale,
            collective_id=collective_id,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    return fn(q_global, kv_cache_by_rank, packed_schedule)


def _run_page_groups_packed_local(q_global, kv_cache_by_rank, packed_schedule,
                                  *, sm_scale, collective_id):
    mesh = jax.sharding.Mesh(jax.local_devices()[:PCP_SIZE], (AXIS, ))

    def _call(q_local, kv_cache_local, schedule):
        return pcp_streaming_attention_page_groups_packed_local(
            q_local,
            kv_cache_local[0],
            schedule,
            pcp_size=PCP_SIZE,
            q_block_size=Q_TILE,
            sm_scale=sm_scale,
            collective_id=collective_id,
        )

    fn = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))
    return fn(q_global, kv_cache_by_rank, packed_schedule)


def _pack_native_kv_cache(kv_cache, kv_packing):
    pcp_size, pages, page_size, kv_heads, kv_pair, head_dim = kv_cache.shape
    if kv_pair != 2:
        raise ValueError("native KV cache must have a K/V pair axis.")
    aligned_kv_heads_x2 = math.ceil(kv_heads * 2 / kv_packing) * kv_packing
    flat = np.zeros((pcp_size, pages, page_size, aligned_kv_heads_x2,
                     head_dim),
                    dtype=kv_cache.dtype)
    flat[..., :kv_heads * 2, :] = kv_cache.reshape(pcp_size, pages,
                                                   page_size, kv_heads * 2,
                                                   head_dim)
    return flat.reshape(pcp_size, pages, page_size,
                        aligned_kv_heads_x2 // kv_packing, kv_packing,
                        head_dim)


def test_single_page_group_kernel_matches_reference():
    rng = np.random.default_rng(1234)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_single_page_group(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        sm_scale=sm_scale,
        collective_id=14,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_single_page_group_kernel_handles_idle_consumer_ranks():
    rng = np.random.default_rng(5678)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    generated = generate_pcp_streaming_schedule(
        kv_lens=[PCP_SIZE * PAGE_SIZE],
        cu_q_lens=[0, 32],
        q_start_offsets=[3 * PAGE_SIZE],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
    )
    schedule = _truncate_schedule(generated,
                                  steps=PCP_SIZE,
                                  actual_steps=[0, 0, 0, PCP_SIZE])
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_single_page_group(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        sm_scale=sm_scale,
        collective_id=15,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_matches_generated_multi_tile_schedule():
    rng = np.random.default_rng(9012)
    q_by_rank = rng.normal(size=(PCP_SIZE, 4 * Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[PCP_SIZE * PAGE_SIZE],
        cu_q_lens=[0, 32],
        q_start_offsets=[3 * PAGE_SIZE],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=16,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_handles_partial_q_tile_and_partial_kv_page():
    rng = np.random.default_rng(3456)
    q_by_rank = rng.normal(size=(PCP_SIZE, 4 * Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[3 * PAGE_SIZE + 116],
        cu_q_lens=[0, 28],
        q_start_offsets=[3 * PAGE_SIZE],
        block_tables=np.array([[0]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=17,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_carries_online_state_across_kv_groups():
    rng = np.random.default_rng(7890)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 2, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[5 * PAGE_SIZE],
        cu_q_lens=[0, Q_TILE],
        q_start_offsets=[4 * PAGE_SIZE],
        block_tables=np.array([[0, 1]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=1,
        bq_sz=Q_TILE,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=18,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_handles_two_lanes():
    rng = np.random.default_rng(2468)
    q_by_rank = rng.normal(size=(PCP_SIZE, 4 * Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 2, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[5 * PAGE_SIZE],
        cu_q_lens=[0, 4 * Q_TILE],
        q_start_offsets=[4 * PAGE_SIZE],
        block_tables=np.array([[0, 1]], dtype=np.int32),
        page_size=PAGE_SIZE,
        pcp_size=PCP_SIZE,
        interleave_size=PAGE_SIZE,
        num_lanes=2,
        bq_sz=Q_TILE,
        pad_kv_pages_to_pcp_group=True,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=19,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_kernel_handles_multiple_kv_and_q_heads():
    rng = np.random.default_rng(1357)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 2, 2,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 2, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = pcp_streaming_attention_page_groups(
        jnp.asarray(q_by_rank),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        pcp_size=PCP_SIZE,
        q_block_size=Q_TILE,
        sm_scale=sm_scale,
        collective_id=20,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_local_kernel_runs_inside_existing_pcp_shard_map():
    rng = np.random.default_rng(9753)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 1, 1,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * Q_TILE, 1, 1, HEAD_DIM)
    kv_cache = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 1, 2,
                                HEAD_DIM)).astype(np.float32) * 0.1
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_local(
        jnp.asarray(q_global),
        jnp.asarray(kv_cache),
        jnp.asarray(schedule.packed_schedule),
        sm_scale=sm_scale,
        collective_id=21,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               kv_cache,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(PCP_SIZE * Q_TILE, 1, 1, HEAD_DIM)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-4,
                               atol=5e-5)


def test_page_group_packed_local_kernel_consumes_batched_rpa_kv_layout():
    rng = np.random.default_rng(8642)
    q_by_rank = rng.normal(size=(PCP_SIZE, Q_TILE, 2, 2,
                                 HEAD_DIM)).astype(np.float32) * 0.1
    q_global = q_by_rank.reshape(PCP_SIZE * Q_TILE, 2, 2, HEAD_DIM)
    native_kv_np = rng.normal(size=(PCP_SIZE, 1, PAGE_SIZE, 2, 2,
                                    HEAD_DIM)).astype(np.float32) * 0.1
    native_kv = jnp.asarray(native_kv_np, dtype=jnp.bfloat16)
    native_kv_host = np.asarray(jax.device_get(native_kv))
    packed_kv = jnp.asarray(_pack_native_kv_cache(native_kv_host, 2),
                            dtype=jnp.bfloat16)
    schedule = _make_single_page_group_schedule()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _run_page_groups_packed_local(
        jnp.asarray(q_global),
        packed_kv,
        jnp.asarray(schedule.packed_schedule),
        sm_scale=sm_scale,
        collective_id=22,
    )
    out.block_until_ready()

    expected = execute_pcp_streaming_reference(q_by_rank,
                                               native_kv_host,
                                               schedule,
                                               sm_scale=sm_scale)
    expected = expected.reshape(PCP_SIZE * Q_TILE, 2, 2, HEAD_DIM)
    np.testing.assert_allclose(np.asarray(jax.device_get(out)),
                               expected,
                               rtol=5e-3,
                               atol=5e-4)
