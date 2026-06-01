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

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu
from jax.sharding import Mesh
from transformers import AutoConfig

from tpu_inference.layers.common.attention_metadata import PcpMode
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.kernels.experimental.batched_rpa import \
    wrapper as batched_rpa_wrapper
from tpu_inference.kernels.experimental.batched_rpa.wrapper import \
    get_kv_cache_shape as get_batched_rpa_kv_cache_shape
from tpu_inference.layers.vllm.backends.flash_attn import _jax_attn_func
from tpu_inference.runner.tpu_runner import (TPUModelRunner,
                                             _build_pcp_rank_major_token_order,
                                             _pcp_local_token_counts)
from tpu_inference.utils import DeviceBuffer

QWEN3_06B_PATH = Path("/mnt/data/xiaohao/workspace/models/Qwen3-0.6B")
NUM_SYNTHETIC_LAYERS = 4


def _load_qwen3_06b_four_layer_config():
    if not QWEN3_06B_PATH.exists():
        pytest.skip(f"Qwen3-0.6B config not found: {QWEN3_06B_PATH}")
    config = AutoConfig.from_pretrained(QWEN3_06B_PATH, local_files_only=True)
    config.num_hidden_layers = NUM_SYNTHETIC_LAYERS
    return config


def _synthetic_hidden_from_input_ids(input_ids: jax.Array,
                                     hidden_size: int) -> jax.Array:
    dims = jnp.arange(hidden_size, dtype=jnp.float32)
    ids = input_ids.astype(jnp.float32)[:, None]
    return 0.1 * jnp.sin(ids * (dims[None, :] + 1.0) * 0.0003)


def _make_synthetic_qwen3_attention_weights(
        config) -> list[dict[str, jax.Array]]:
    rng = np.random.default_rng(123)
    hidden_size = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    weights = []
    for _ in range(config.num_hidden_layers):
        weights.append({
            "q":
            jnp.array(rng.normal(scale=0.02, size=(hidden_size, q_dim)),
                      dtype=jnp.float32),
            "k":
            jnp.array(rng.normal(scale=0.02, size=(hidden_size, kv_dim)),
                      dtype=jnp.float32),
            "v":
            jnp.array(rng.normal(scale=0.02, size=(hidden_size, kv_dim)),
                      dtype=jnp.float32),
            "o":
            jnp.array(rng.normal(scale=0.02, size=(q_dim, hidden_size)),
                      dtype=jnp.float32),
        })
    return weights


def _ref_flat_causal_attention(q_flat, k_flat, v_flat, *, num_heads,
                               num_kv_heads, head_dim, scale):
    q = q_flat.reshape(q_flat.shape[0], num_heads,
                       head_dim).astype(jnp.float32)
    k = k_flat.reshape(k_flat.shape[0], num_kv_heads,
                       head_dim).astype(jnp.float32)
    v = v_flat.reshape(v_flat.shape[0], num_kv_heads,
                       head_dim).astype(jnp.float32)
    num_queries_per_kv = num_heads // num_kv_heads
    k = jnp.repeat(k, num_queries_per_kv, axis=1)
    v = jnp.repeat(v, num_queries_per_kv, axis=1)

    logits = jnp.einsum("qhd,khd->hqk", q, k) * scale
    q_pos = jnp.arange(q.shape[0], dtype=jnp.int32)
    k_pos = jnp.arange(k.shape[0], dtype=jnp.int32)
    causal_mask = q_pos[None, :, None] >= k_pos[None, None, :]
    logits = jnp.where(causal_mask, logits, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hqk,khd->qhd", probs,
                      v).reshape(q_flat.shape[0], num_heads * head_dim)


def _ref_segmented_causal_attention(q, k, v, seq_lens, config):
    outputs = []
    start = 0
    scale = config.head_dim**-0.5
    for seq_len in seq_lens:
        end = start + int(seq_len)
        outputs.append(
            _ref_flat_causal_attention(
                q[start:end],
                k[start:end],
                v[start:end],
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                scale=scale,
            ))
        start = end
    return jnp.concatenate(outputs, axis=0)


def _run_reference_qwen3_attention_stack(input_ids, seq_lens, weights, config):
    hidden = _synthetic_hidden_from_input_ids(
        jnp.array(input_ids, dtype=jnp.int32), config.hidden_size)
    for layer_weights in weights:
        q = hidden @ layer_weights["q"]
        k = hidden @ layer_weights["k"]
        v = hidden @ layer_weights["v"]
        attn_out = _ref_segmented_causal_attention(q, k, v, seq_lens, config)
        hidden = hidden + attn_out @ layer_weights["o"]
    return hidden


def _make_pcp_runner(
        mesh: Mesh,
        seq_lens: list[int],
        interleave_size: int,
        assigned_dp_rank: dict[str, int] | None = None) -> SimpleNamespace:
    runner = SimpleNamespace()
    pcp_size = mesh.shape["pcp"]
    dp_size = mesh.shape["data"]
    if assigned_dp_rank is None:
        assigned_dp_rank = {f"req{i}": 0 for i in range(len(seq_lens))}

    per_dp_token_capacity = 1
    for dp_rank in range(dp_size):
        dp_seq_lens = [
            seq_lens[i] for i in range(len(seq_lens))
            if assigned_dp_rank[f"req{i}"] == dp_rank
        ]
        if not dp_seq_lens:
            continue
        local_counts = _pcp_local_token_counts(dp_seq_lens, pcp_size,
                                               interleave_size)
        per_dp_token_capacity = max(per_dp_token_capacity, sum(dp_seq_lens),
                                    int(local_counts.max()) * pcp_size)
    per_dp_token_capacity = ((per_dp_token_capacity + pcp_size - 1) //
                             pcp_size * pcp_size)

    parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        cp_kv_cache_interleave_size=interleave_size,
        decode_context_parallel_size=1,
    )
    runner.vllm_config = SimpleNamespace(parallel_config=parallel_config)
    runner.parallel_config = parallel_config
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.speculative_config = None
    runner.lora_config = None
    runner.mesh = mesh
    runner.dp_size = dp_size
    runner.max_num_reqs = 8
    runner.max_num_tokens = per_dp_token_capacity * dp_size
    runner.max_model_len = 64
    runner.block_size = 16
    runner.max_num_blocks_per_req = 4
    runner.num_tokens_paddings_per_dp = [per_dp_token_capacity]
    runner.num_reqs_paddings_per_dp = [runner.max_num_reqs // dp_size]
    runner.attn_num_reqs_paddings_per_dp = [runner.max_num_reqs // dp_size]
    runner.positions_cpu = np.zeros(runner.max_num_tokens, dtype=np.int32)
    runner.mrope_positions_cpu = np.zeros((3, runner.max_num_tokens),
                                          dtype=np.int64)
    runner.arange_cpu = np.arange(runner.max_model_len, dtype=np.int64)
    runner.uses_mrope = False
    runner.device_buffer = DeviceBuffer(initial_capacity=4096)
    runner.phase_based_profiler = None
    runner.aggregated_stats_logger = None
    runner.batch_counter = 0
    runner.lora_utils = SimpleNamespace(
        extract_lora_metadata=lambda: None,
        set_active_loras=lambda *args, **kwargs: None,
    )
    runner.persistent_batch_manager = SimpleNamespace(
        update_states=lambda *args, **kwargs: None)
    runner.get_mrope_input_positions_fn = None
    runner.maybe_forbid_compile = nullcontext()
    runner.maybe_get_kv_connector_output = lambda *args, **kwargs: nullcontext(
        None)
    runner.is_multimodal_model = False
    runner.is_pooling_model = False
    runner.is_first_rank = True
    runner.is_last_rank = True
    runner.state_leaves = None
    runner.execute_model_state = None

    input_batch = SimpleNamespace()
    input_batch.num_reqs = len(seq_lens)
    input_batch.req_ids = [f"req{i}" for i in range(len(seq_lens))]
    input_batch.req_id_to_index = {
        req_id: i
        for i, req_id in enumerate(input_batch.req_ids)
    }
    input_batch.num_computed_tokens_cpu = np.zeros(len(seq_lens),
                                                   dtype=np.int32)
    input_batch.token_ids_cpu = np.zeros((len(seq_lens), runner.max_model_len),
                                         dtype=np.int32)
    for req_idx, seq_len in enumerate(seq_lens):
        start = 100 * (req_idx + 1)
        input_batch.token_ids_cpu[req_idx, :seq_len] = np.arange(
            start, start + seq_len, dtype=np.int32)
    input_batch.mamba_state_indices_cpu = np.zeros(runner.max_num_reqs,
                                                   dtype=np.int32)

    block_tables_cpu = np.zeros((len(seq_lens), runner.max_num_blocks_per_req),
                                dtype=np.int32)
    block_size = 16
    next_block = 0
    for req_idx, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_tables_cpu[req_idx, :num_blocks] = np.arange(next_block,
                                                           next_block +
                                                           num_blocks,
                                                           dtype=np.int32)
        next_block += num_blocks
    block_table = SimpleNamespace(
        max_num_blocks_per_req=runner.max_num_blocks_per_req,
        get_cpu_tensor=lambda: block_tables_cpu,
    )
    input_batch.block_table = [block_table]
    runner.input_batch = input_batch

    kv_cache_group = SimpleNamespace(
        layer_names=[f"layer.{i}" for i in range(NUM_SYNTHETIC_LAYERS)])
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[kv_cache_group],
                                             has_mamba_layers=False)
    runner.use_hybrid_kvcache = False

    runner._prepare_inputs = TPUModelRunner._prepare_inputs.__get__(runner)
    runner._prepare_input_metadata = (
        TPUModelRunner._prepare_input_metadata.__get__(runner))
    runner._get_input_ids_embeds = TPUModelRunner._get_input_ids_embeds.__get__(
        runner)
    runner._execute_model = TPUModelRunner._execute_model.__get__(runner)
    runner.execute_model = TPUModelRunner.execute_model.__get__(runner)
    return runner


def _make_scheduler_output(
        seq_lens: list[int],
        assigned_dp_rank: dict[str, int] | None = None) -> SimpleNamespace:
    num_scheduled_tokens = {
        f"req{i}": int(seq_len)
        for i, seq_len in enumerate(seq_lens)
    }
    if assigned_dp_rank is None:
        assigned_dp_rank = {req_id: 0 for req_id in num_scheduled_tokens}
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(seq_lens),
        num_scheduled_tokens=num_scheduled_tokens,
        assigned_dp_rank=assigned_dp_rank,
        scheduled_spec_decode_tokens={},
        grammar_bitmask=None,
        finished_req_ids=[],
    )


def _unpack_dp_pcp_output(packed: np.ndarray, seq_lens: list[int],
                          assigned_dp_rank: dict[str, int], dp_size: int,
                          pcp_size: int, interleave_size: int,
                          padded_num_tokens_per_dp: int) -> np.ndarray:
    req_starts = np.pad(np.cumsum(seq_lens, dtype=np.int64), (1, 0))[:-1]
    unpacked = np.empty((sum(seq_lens), packed.shape[-1]), dtype=packed.dtype)

    for dp_rank in range(dp_size):
        req_indices = [
            i for i in range(len(seq_lens))
            if assigned_dp_rank[f"req{i}"] == dp_rank
        ]
        dp_seq_lens = [seq_lens[i] for i in req_indices]
        if not dp_seq_lens:
            continue
        _, inverse_order = _build_pcp_rank_major_token_order(
            dp_seq_lens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            padded_num_tokens=padded_num_tokens_per_dp,
        )
        dp_start = dp_rank * padded_num_tokens_per_dp
        dp_packed = packed[dp_start:dp_start + padded_num_tokens_per_dp]
        dp_natural = dp_packed[inverse_order]

        local_start = 0
        for req_idx, seq_len in zip(req_indices, dp_seq_lens):
            local_end = local_start + seq_len
            req_start = req_starts[req_idx]
            unpacked[req_start:req_start +
                     seq_len] = dp_natural[local_start:local_end]
            local_start = local_end
    return unpacked


def _install_synthetic_qwen3_attention_stack(runner: SimpleNamespace,
                                             mesh: Mesh, config, weights,
                                             interleave_size: int) -> None:

    def synthetic_model_fn(_state_leaves, kv_caches, input_ids,
                           attention_metadata, _inputs_embeds,
                           _input_positions, _layer_name_to_kvcache_index,
                           _lora_metadata, _intermediate_tensors,
                           _is_first_rank, _is_last_rank):
        hidden = _synthetic_hidden_from_input_ids(input_ids,
                                                  config.hidden_size)
        new_kv_caches = []
        for layer_idx, layer_weights in enumerate(weights):
            q = hidden @ layer_weights["q"]
            k = hidden @ layer_weights["k"]
            v = hidden @ layer_weights["v"]
            new_kv_cache, attn_out = _jax_attn_func(
                kv_caches[layer_idx],
                q,
                k,
                v,
                None,
                attention_metadata,
                mesh,
                config.head_dim**-0.5,
                config.head_dim,
                config.num_attention_heads,
                config.num_key_value_heads,
                None,
                None,
                None,
                None,
                PcpMode.PREFILL_LOCAL_Q_FULL_KV,
                True,
                interleave_size,
            )
            hidden = hidden + attn_out @ layer_weights["o"]
            new_kv_caches.append(new_kv_cache)
        return new_kv_caches, hidden, [], None

    runner.model_fn = synthetic_model_fn
    runner.compute_logits_fn = lambda _state, hidden, _lora: hidden[:, :8]


def test_runner_to_four_layer_qwen3_pcp_attention_output(monkeypatch):
    if not jtu.is_device_tpu_at_least(version=4):
        pytest.skip("Batched RPA requires TPUv4+")
    if len(jax.local_devices()) < 2:
        pytest.skip("PCP runner integration test requires 2 local devices")

    config = _load_qwen3_06b_four_layer_config()
    seq_lens = [16, 16]
    interleave_size = 4
    pcp_size = 2
    total_tokens = sum(seq_lens)

    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        batched_rpa_wrapper.ragged_paged_attention,
    )
    monkeypatch.setattr(
        "tpu_inference.runner.tpu_runner.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    devices = np.array(jax.local_devices()[:pcp_size]).reshape(
        (1, 1, 1, 1, 1, 1, pcp_size))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    runner = _make_pcp_runner(mesh, seq_lens, interleave_size)
    scheduler_output = _make_scheduler_output(seq_lens)
    weights = _make_synthetic_qwen3_attention_weights(config)

    num_pages = 8
    block_size = 16
    kv_cache_shape = get_batched_rpa_kv_cache_shape(
        total_num_pages=num_pages,
        page_size=block_size,
        actual_num_kv_heads=config.num_key_value_heads,
        actual_head_dim=config.head_dim,
        kv_dtype=jnp.float32,
    )
    runner.kv_caches = [
        jnp.zeros(kv_cache_shape, dtype=jnp.float32)
        for _ in range(config.num_hidden_layers)
    ]
    runner.layer_name_to_kvcache_index = {
        f"layer.{i}": i
        for i in range(config.num_hidden_layers)
    }

    _install_synthetic_qwen3_attention_stack(runner, mesh, config, weights,
                                             interleave_size)

    with patch("tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata"
               ) as mock_sampling_metadata:
        mock_sampling_metadata.from_input_batch.return_value = MagicMock()
        with jax.set_mesh(mesh):
            assert runner.execute_model(scheduler_output) is None

    original_input_ids = np.concatenate([
        runner.input_batch.token_ids_cpu[i, :seq_lens[i]]
        for i in range(len(seq_lens))
    ])
    expected = _run_reference_qwen3_attention_stack(original_input_ids,
                                                    seq_lens, weights, config)
    expected = np.asarray(jax.device_get(expected))

    _, inverse_order = _build_pcp_rank_major_token_order(
        seq_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=total_tokens,
    )
    actual_packed = runner.execute_model_state.full_hidden_states
    actual_packed = jax.device_get(jax.block_until_ready(actual_packed))
    actual = np.asarray(actual_packed)[inverse_order]

    np.testing.assert_allclose(actual, expected, rtol=0.12, atol=0.12)

    selected = jax.device_get(
        jax.block_until_ready(runner.execute_model_state.hidden_states))
    expected_last_tokens = expected[np.cumsum(seq_lens) - 1]
    np.testing.assert_allclose(np.asarray(selected)[:len(seq_lens)],
                               expected_last_tokens,
                               rtol=0.12,
                               atol=0.12)
    assert runner.execute_model_state.logits.shape == (runner.max_num_reqs, 8)


def test_runner_to_four_layer_qwen3_pcp_attention_output_dp2(monkeypatch):
    if not jtu.is_device_tpu_at_least(version=4):
        pytest.skip("Batched RPA requires TPUv4+")
    if len(jax.local_devices()) < 4:
        pytest.skip("PCP + DP integration test requires 4 local devices")

    config = _load_qwen3_06b_four_layer_config()
    seq_lens = [16, 16, 16, 16]
    assigned_dp_rank = {
        "req0": 0,
        "req1": 1,
        "req2": 0,
        "req3": 1,
    }
    interleave_size = 4
    pcp_size = 2
    dp_size = 2

    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        batched_rpa_wrapper.ragged_paged_attention,
    )
    monkeypatch.setattr(
        "tpu_inference.runner.tpu_runner.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    devices = np.array(jax.local_devices()[:dp_size * pcp_size]).reshape(
        (dp_size, 1, 1, 1, 1, 1, pcp_size))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    runner = _make_pcp_runner(mesh, seq_lens, interleave_size,
                              assigned_dp_rank)
    scheduler_output = _make_scheduler_output(seq_lens, assigned_dp_rank)
    weights = _make_synthetic_qwen3_attention_weights(config)

    num_pages = 8
    block_size = 16
    kv_cache_shape = get_batched_rpa_kv_cache_shape(
        total_num_pages=num_pages,
        page_size=block_size,
        actual_num_kv_heads=config.num_key_value_heads,
        actual_head_dim=config.head_dim,
        kv_dtype=jnp.float32,
    )
    runner.kv_caches = [
        jnp.zeros(kv_cache_shape, dtype=jnp.float32)
        for _ in range(config.num_hidden_layers)
    ]
    runner.layer_name_to_kvcache_index = {
        f"layer.{i}": i
        for i in range(config.num_hidden_layers)
    }
    _install_synthetic_qwen3_attention_stack(runner, mesh, config, weights,
                                             interleave_size)

    with patch("tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata"
               ) as mock_sampling_metadata:
        mock_sampling_metadata.from_input_batch.return_value = MagicMock()
        with jax.set_mesh(mesh):
            assert runner.execute_model(scheduler_output) is None

    original_input_ids = np.concatenate([
        runner.input_batch.token_ids_cpu[i, :seq_lens[i]]
        for i in range(len(seq_lens))
    ])
    expected = _run_reference_qwen3_attention_stack(original_input_ids,
                                                    seq_lens, weights, config)
    expected = np.asarray(jax.device_get(expected))

    actual_packed = runner.execute_model_state.full_hidden_states
    actual_packed = jax.device_get(jax.block_until_ready(actual_packed))
    actual = _unpack_dp_pcp_output(
        np.asarray(actual_packed),
        seq_lens,
        assigned_dp_rank,
        dp_size=dp_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens_per_dp=runner.max_num_tokens // dp_size,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.12, atol=0.12)

    attn_metadata = runner.execute_model_state.attn_metadata
    assert attn_metadata.pcp_slot_ids is not None
    assert attn_metadata.pcp_slot_ids.shape[0] == runner.max_num_tokens

    selected = jax.device_get(
        jax.block_until_ready(runner.execute_model_state.hidden_states))
    selector = runner.execute_model_state.logits_indices_selector
    assert selector is not None
    selected_original_order = np.asarray(selected)[selector]
    expected_last_tokens = expected[np.cumsum(seq_lens) - 1]
    np.testing.assert_allclose(selected_original_order[:len(seq_lens)],
                               expected_last_tokens,
                               rtol=0.12,
                               atol=0.12)
    assert runner.execute_model_state.logits.shape == (runner.max_num_reqs, 8)
