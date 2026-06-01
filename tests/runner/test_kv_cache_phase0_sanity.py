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

from itertools import product
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from tpu_inference.core.sched.dp_scheduler import _get_per_rank_num_blocks
from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.runner.kv_cache import (create_kv_caches,
                                           get_attention_kv_cache_sizing)


def _fake_mesh(**overrides):
    shape = {
        "data": 1,
        "attn_dp": 1,
        "attn_dp_expert": 1,
        "pcp": 1,
        "dcp": 1,
        "model": 1,
        "expert": 1,
    }
    shape.update(overrides)
    return SimpleNamespace(shape=shape)


@pytest.fixture(autouse=True)
def _use_new_model_sharding(monkeypatch):
    monkeypatch.setattr(sharding_mod.ShardingAxisName, "_cls",
                        sharding_mod.ShardingAxisNameBase)


@pytest.mark.parametrize(
    ("mesh_overrides", "expected_block_shards"),
    [
        ({
            "pcp": 4
        }, 4),
        ({
            "data": 2,
            "pcp": 4
        }, 8),
        ({
            "attn_dp": 2,
            "attn_dp_expert": 2,
            "pcp": 2
        }, 8),
    ],
)
def test_phase0_sizing_uses_full_kv_cache_block_axis_product(
        mesh_overrides, expected_block_shards):
    mesh = _fake_mesh(**mesh_overrides)

    sizing = get_attention_kv_cache_sizing(
        mesh,
        spec_block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    local_num_blocks = 13
    aggregate_available_memory = (local_num_blocks *
                                  sizing.vllm_page_size_padded)
    core_num_blocks = sizing.vllm_num_blocks(aggregate_available_memory)
    tensor_size = sizing.vllm_tensor_size(core_num_blocks)
    jax_global_num_blocks = sizing.global_num_blocks_from_tensor_size(
        tensor_size)

    assert sizing.kv_cache_block_shard_count == expected_block_shards
    assert core_num_blocks == local_num_blocks
    assert jax_global_num_blocks == local_num_blocks * expected_block_shards
    assert sizing.local_num_blocks_from_global(
        jax_global_num_blocks) == local_num_blocks
    assert max(0, local_num_blocks - 1) < core_num_blocks


def test_phase0_dcp_expands_global_block_dim_but_preserves_local_block_size():
    mesh = _fake_mesh(pcp=2, dcp=2)
    spec_block_size = 16

    sizing = get_attention_kv_cache_sizing(
        mesh,
        spec_block_size=spec_block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    local_num_blocks = 7
    jax_global_num_blocks = sizing.global_num_blocks_from_tensor_size(
        sizing.vllm_tensor_size(local_num_blocks))

    assert sizing.allocation_block_size == spec_block_size * mesh.shape["dcp"]
    assert jax_global_num_blocks == local_num_blocks * mesh.shape["pcp"]
    assert sizing.allocation_block_size // mesh.shape["dcp"] == spec_block_size


def test_phase0_uninflated_page_size_would_overestimate_core_num_blocks():
    mesh = _fake_mesh(pcp=8)
    sizing = get_attention_kv_cache_sizing(
        mesh,
        spec_block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    local_num_blocks = 11
    aggregate_available_memory = (local_num_blocks *
                                  sizing.vllm_page_size_padded)

    uninflated_core_num_blocks = (aggregate_available_memory //
                                  sizing.allocation_page_size_bytes)

    assert sizing.vllm_num_blocks(
        aggregate_available_memory) == local_num_blocks
    assert uninflated_core_num_blocks == (local_num_blocks *
                                          sizing.kv_cache_block_shard_count)


def test_phase0_same_worker_dp_must_not_be_divided_twice():
    mesh = _fake_mesh(data=2, pcp=4)
    sizing = get_attention_kv_cache_sizing(
        mesh,
        spec_block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    local_num_blocks = 17
    core_num_blocks = sizing.vllm_num_blocks(local_num_blocks *
                                             sizing.vllm_page_size_padded)

    assert core_num_blocks == local_num_blocks
    assert core_num_blocks // mesh.shape["data"] != local_num_blocks


@pytest.mark.parametrize(("data", "pcp", "dcp"),
                         list(product([1, 2], [1, 2, 4], [1, 2])))
def test_phase0_sizing_matrix_returns_local_pages(data, pcp, dcp):
    mesh = _fake_mesh(data=data, pcp=pcp, dcp=dcp)
    spec_block_size = 16

    sizing = get_attention_kv_cache_sizing(
        mesh,
        spec_block_size=spec_block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    local_num_blocks = 19
    aggregate_available_memory = (local_num_blocks *
                                  sizing.vllm_page_size_padded)
    core_num_blocks = sizing.vllm_num_blocks(aggregate_available_memory)
    tensor_size = sizing.vllm_tensor_size(core_num_blocks)
    global_num_blocks = sizing.global_num_blocks_from_tensor_size(tensor_size)
    expected_block_shards = data * pcp
    block_table = np.arange(core_num_blocks, dtype=np.int32)

    assert sizing.allocation_block_size == spec_block_size * dcp
    assert sizing.kv_cache_block_shard_count == expected_block_shards
    assert sizing.vllm_page_size_padded == (sizing.allocation_page_size_bytes *
                                            expected_block_shards)
    assert core_num_blocks == local_num_blocks
    assert global_num_blocks == local_num_blocks * expected_block_shards
    assert sizing.local_num_blocks_from_global(
        global_num_blocks) == local_num_blocks
    assert block_table.max() < core_num_blocks


def test_phase0_dp_scheduler_preserves_tpu_local_num_blocks():
    manager = sharding_mod.ShardingConfigManager(
        sharding_mod.ShardingStrategy(data_parallelism=2,
                                      prefill_context_parallelism=4))
    vllm_config = SimpleNamespace(sharding_config=manager)

    assert manager.kv_cache_num_blocks_are_per_dp_rank is True
    assert _get_per_rank_num_blocks(vllm_config, num_blocks=17,
                                    dp_size=2) == 17


def test_phase0_dp_scheduler_can_still_split_global_num_blocks():
    vllm_config = SimpleNamespace(sharding_config=SimpleNamespace())

    assert _get_per_rank_num_blocks(vllm_config, num_blocks=18, dp_size=2) == 9


def test_phase0_allocated_cache_shapes_match_sizing_terms():
    if len(jax.local_devices()) < 4:
        pytest.skip("requires at least 4 local devices")

    devices = np.array(jax.local_devices()[:4]).reshape((1, 1, 1, 1, 1, 2, 2))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    spec_block_size = 16
    local_num_blocks = 5

    sizing = get_attention_kv_cache_sizing(
        mesh,
        spec_block_size=spec_block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=jnp.bfloat16,
    )
    global_num_blocks = sizing.global_num_blocks_from_tensor_size(
        sizing.vllm_tensor_size(local_num_blocks))

    kv_cache = create_kv_caches(
        num_blocks=global_num_blocks,
        block_size=sizing.allocation_block_size,
        num_kv_heads=4,
        head_size=128,
        mesh=mesh,
        layer_names=["layer.0"],
        cache_dtype=jnp.bfloat16,
    )[0]

    assert sizing.kv_cache_block_shard_count == 2
    assert sizing.allocation_block_size == spec_block_size * 2
    assert kv_cache.shape[0] == local_num_blocks * 2
    assert kv_cache.shape[1] == sizing.allocation_block_size
    local_shapes = [shard.data.shape for shard in kv_cache.addressable_shards]
    assert {shape[0] for shape in local_shapes} == {local_num_blocks}
    assert {shape[1] for shape in local_shapes} == {spec_block_size}
