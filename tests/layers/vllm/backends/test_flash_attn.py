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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from jax.sharding import Mesh
from torchax.interop import torch_view
from vllm.v1.attention.backend import AttentionType

from tpu_inference.layers.common.attention_metadata import (AttentionMetadata,
                                                            PcpMode)
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.layers.vllm.backends.flash_attn import (
    PallasAttentionBackend, PallasAttentionBackendImpl, _jax_attn_func)
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from tpu_inference.runner.kv_cache import get_kv_cache_shape_with_mesh

# ---- Test Configuration & Constants ----

# Total number of tokens across all sequences in the batch
TOTAL_TOKENS = 10
# Number of sequences in the batch
NUM_SEQS = 2
# Padded maximum number of sequences
MAX_NUM_SEQS = 4
# Number of attention heads (Query)
NUM_HEADS = 8
# Number of attention heads (Key/Value) - for Grouped-Query Attention
NUM_KV_HEADS = 4
# Dimension of each attention head
HEAD_DIM = 128
# Total number of blocks in the KV cache
NUM_BLOCKS = 32
# Number of tokens per block
BLOCK_SIZE = 16
# Maximum number of blocks a single sequence can occupy
MAX_BLOCKS_PER_SEQ = 8


def create_inputs(
    mesh: Mesh,
    q_dtype: jnp.dtype = jnp.bfloat16,
    kv_dtype: jnp.dtype = jnp.bfloat16,
    total_tokens: int = TOTAL_TOKENS,
    num_seqs: int = NUM_SEQS,
    max_num_seqs: int = MAX_NUM_SEQS,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    max_blocks_per_seq: int = MAX_BLOCKS_PER_SEQ,
):
    key = jax.random.key(0)
    q = jax.random.uniform(key, (total_tokens, num_heads * head_dim),
                           dtype=q_dtype)
    k = jax.random.uniform(key, (total_tokens, num_kv_heads * head_dim),
                           dtype=q_dtype)
    v = jax.random.uniform(key, (total_tokens, num_kv_heads * head_dim),
                           dtype=q_dtype)
    q = torch_view(q)
    k = torch_view(k)
    v = torch_view(v)

    kv_cache_shape = get_kv_cache_shape_with_mesh(mesh, num_blocks, block_size,
                                                  num_kv_heads, head_dim,
                                                  kv_dtype)
    kv_cache = jax.random.normal(key, kv_cache_shape, dtype=kv_dtype)

    positions = jnp.ones((total_tokens, ), dtype=jnp.int32)
    block_tables = jnp.zeros((max_num_seqs * max_blocks_per_seq),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.array([5, 5, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )

    return q, k, v, kv_cache, metadata


def create_fake_vllm_config(prefill_context_parallel_size: int,
                            cp_kv_cache_interleave_size: int = 2):
    vllm_config = MagicMock()
    vllm_config.parallel_config.prefill_context_parallel_size = (
        prefill_context_parallel_size)
    vllm_config.parallel_config.cp_kv_cache_interleave_size = (
        cp_kv_cache_interleave_size)
    return vllm_config


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    # Create a mesh with available devices, useful for running on CPU/GPU/TPU
    # For this test, it will likely be a single CPU device.
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        # Add a mock device if no devices are present (e.g., in a CI environment)
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


class TestPallasAttentionBackend:

    def test_get_name(self):
        assert PallasAttentionBackend.get_name() == "FLASH_ATTN"

    def test_get_impl_cls(self):
        assert PallasAttentionBackend.get_impl_cls(
        ) == PallasAttentionBackendImpl


class TestPallasAttentionBackendImpl:

    def test_backend_declares_pcp_support(self):
        assert PallasAttentionBackendImpl.supports_pcp is True

    def test_init_valid_params(self):
        impl = PallasAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.088,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        assert impl.num_heads == 32
        assert impl.head_size == 128
        assert impl.scale == 0.088
        assert impl.num_kv_heads == 8
        assert impl.num_queries_per_kv == 4
        assert impl.sliding_window is None

    def test_init_with_alibi_slopes_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="Alibi slopes is not supported"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=[1.0, 2.0],
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.DECODER,
            )

    def test_init_with_encoder_attention_raises_error(self):
        with pytest.raises(NotImplementedError,
                           match="Encoder self-attention"):
            PallasAttentionBackendImpl(
                num_heads=32,
                head_size=128,
                scale=0.088,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
                attn_type=AttentionType.ENCODER,
            )

    def test_forward(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)
    def test_forward_with_3d_qkv(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            query = query.reshape(TOTAL_TOKENS, NUM_HEADS, HEAD_DIM)
            key = key.reshape(TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM)
            value = value.reshape(TOTAL_TOKENS, NUM_KV_HEADS, HEAD_DIM)

            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    @pytest.mark.parametrize(
        ("prefill_context_parallel_size", "has_pcp_metadata",
         "expected_pcp_mode", "expected_shard_pcp_axis"),
        [
            (1, False, PcpMode.DISABLED, True),
            (2, False, PcpMode.DISABLED, False),
            (2, True, PcpMode.PREFILL_STREAMING, True),
        ],
    )
    def test_forward_passes_pcp_mode_from_runtime_metadata(
            self, monkeypatch, mesh, prefill_context_parallel_size,
            has_pcp_metadata, expected_pcp_mode, expected_shard_pcp_axis):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"
        query, key, value, kv_cache, metadata = create_inputs(mesh)
        if has_pcp_metadata:
            metadata.pcp_streaming_schedule = jnp.zeros((1, 1, 1, 1, 128),
                                                        dtype=jnp.int32)
            metadata.pcp_streaming_active_page_groups = jnp.array(
                [1], dtype=jnp.int32)
        captured = {}

        def fake_jax_attn_func(kv_cache_arg, q_arg, k_arg, v_arg, sinks_arg,
                               metadata_arg, mesh_arg, scale_arg,
                               head_size_arg, num_heads_arg, num_kv_heads_arg,
                               q_scale_arg, k_scale_arg, v_scale_arg,
                               sliding_window_arg, pcp_mode_arg,
                               shard_pcp_axis_arg,
                               cp_kv_cache_interleave_size_arg):
            captured["pcp_mode"] = pcp_mode_arg
            captured["shard_pcp_axis"] = shard_pcp_axis_arg
            captured["cp_kv_cache_interleave_size"] = (
                cp_kv_cache_interleave_size_arg)
            captured["q_shape"] = q_arg.shape
            return kv_cache_arg, jnp.zeros_like(q_arg)

        monkeypatch.setattr(
            "tpu_inference.layers.vllm.backends.flash_attn._jax_attn_func",
            fake_jax_attn_func,
        )

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0},
                vllm_config=create_fake_vllm_config(
                    prefill_context_parallel_size),
        ):
            out = impl.forward(layer, query, key, value, torch.tensor([]),
                               metadata)

        assert captured["pcp_mode"] == expected_pcp_mode
        assert captured["shard_pcp_axis"] is expected_shard_pcp_axis
        assert captured["cp_kv_cache_interleave_size"] == (
            2 if expected_pcp_mode != PcpMode.DISABLED else 0)
        assert captured["q_shape"] == (TOTAL_TOKENS, NUM_HEADS * HEAD_DIM)
        assert tuple(out.shape) == (TOTAL_TOKENS, NUM_HEADS * HEAD_DIM)

    def test_forward_passes_pcp_decode_mode_from_runtime_metadata(
            self, monkeypatch, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )
        layer = MagicMock()
        layer.layer_name = "0"
        query, key, value, kv_cache, metadata = create_inputs(mesh)
        metadata.pcp_slot_ids = jnp.arange(TOTAL_TOKENS, dtype=jnp.int32)
        metadata.pcp_source_block_tables = jnp.zeros((MAX_NUM_SEQS, 1),
                                                     dtype=jnp.int32)
        captured = {}

        def fake_jax_attn_func(kv_cache_arg, q_arg, k_arg, v_arg, sinks_arg,
                               metadata_arg, mesh_arg, scale_arg,
                               head_size_arg, num_heads_arg, num_kv_heads_arg,
                               q_scale_arg, k_scale_arg, v_scale_arg,
                               sliding_window_arg, pcp_mode_arg,
                               shard_pcp_axis_arg,
                               cp_kv_cache_interleave_size_arg):
            captured["pcp_mode"] = pcp_mode_arg
            captured["shard_pcp_axis"] = shard_pcp_axis_arg
            captured["cp_kv_cache_interleave_size"] = (
                cp_kv_cache_interleave_size_arg)
            return kv_cache_arg, jnp.zeros_like(q_arg)

        monkeypatch.setattr(
            "tpu_inference.layers.vllm.backends.flash_attn._jax_attn_func",
            fake_jax_attn_func,
        )

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0},
                vllm_config=create_fake_vllm_config(2),
        ):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

        assert captured["pcp_mode"] == PcpMode.DECODE_SHARDED_KV
        assert captured["shard_pcp_axis"] is True
        assert captured["cp_kv_cache_interleave_size"] == 2

    def test_forward_with_fp8_kv_cache(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"
        layer._q_scale_float = None
        layer._k_scale_float = 1
        layer._v_scale_float = 1

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, kv_dtype=jnp.float8_e4m3fn)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_w8a8(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"
        layer._q_scale_float = 1
        layer._k_scale_float = 1
        layer._v_scale_float = 1

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, kv_dtype=jnp.float8_e4m3fn)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_vllm_kv_cache_raises_error(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh), pytest.raises(RuntimeError,
                                          match="should be empty but has"):
            impl.forward(layer, query, key, value, torch.tensor([1]), metadata)

    def test_forward_with_output_scale_raises_error(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)
        output_scale = torch.tensor([1.0])

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh), pytest.raises(NotImplementedError,
                                          match="fused output quantization"):
            impl.forward(layer,
                         query,
                         key,
                         value,
                         torch.tensor([]),
                         metadata,
                         output_scale=output_scale)

    def test_forward_with_output_block_scale_raises_error(self, mesh):
        impl = PallasAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_DIM,
            scale=0.088,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            attn_type=AttentionType.DECODER,
        )

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(mesh)
        output_block_scale = torch.tensor([1.0])

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh), pytest.raises(NotImplementedError,
                                          match="fused output quantization"):
            impl.forward(layer,
                         query,
                         key,
                         value,
                         torch.tensor([]),
                         metadata,
                         output_block_scale=output_block_scale)


def test_jax_attn_func_reshapes_flat_qkv_and_returns_local_flat_output(
        monkeypatch, mesh):
    _, _, _, kv_cache, metadata = create_inputs(mesh,
                                                total_tokens=4,
                                                num_seqs=1)
    q = jnp.ones((4, NUM_HEADS * HEAD_DIM), dtype=jnp.float32)
    k = jnp.ones((4, NUM_KV_HEADS * HEAD_DIM), dtype=jnp.float32)
    v = jnp.ones((4, NUM_KV_HEADS * HEAD_DIM), dtype=jnp.float32)
    captured = {}

    def fake_attention(kv_cache_arg, q_arg, k_arg, v_arg, metadata_arg,
                       mesh_arg, **kwargs):
        captured["q_shape"] = q_arg.shape
        captured["k_shape"] = k_arg.shape
        captured["v_shape"] = v_arg.shape
        captured["pcp_mode"] = kwargs["pcp_mode"]
        captured["shard_pcp_axis"] = kwargs["shard_pcp_axis"]
        captured["cp_kv_cache_interleave_size"] = (
            kwargs["cp_kv_cache_interleave_size"])
        output = jnp.full(q_arg.shape, 3, dtype=q_arg.dtype)
        return kv_cache_arg, output

    monkeypatch.setattr(
        "tpu_inference.layers.vllm.backends.flash_attn.attention",
        fake_attention,
    )
    fn = getattr(_jax_attn_func, "__wrapped__", _jax_attn_func)

    new_cache, output = fn(
        kv_cache,
        q,
        k,
        v,
        None,
        metadata,
        mesh,
        0.088,
        HEAD_DIM,
        NUM_HEADS,
        NUM_KV_HEADS,
        None,
        None,
        None,
        None,
        PcpMode.PREFILL_STREAMING,
        True,
        2,
    )

    assert new_cache.shape == kv_cache.shape
    assert output.shape == (4, NUM_HEADS * HEAD_DIM)
    assert captured["q_shape"] == (4, NUM_HEADS, HEAD_DIM)
    assert captured["k_shape"] == (4, NUM_KV_HEADS, HEAD_DIM)
    assert captured["v_shape"] == (4, NUM_KV_HEADS, HEAD_DIM)
    assert captured["pcp_mode"] == PcpMode.PREFILL_STREAMING
    assert captured["shard_pcp_axis"] is True
    assert captured["cp_kv_cache_interleave_size"] == 2
    np.testing.assert_array_equal(
        np.asarray(output),
        np.full(output.shape, 3, dtype=np.asarray(output).dtype))

    def test_forward_with_attention_sink(self, mesh):
        head_dim = 64
        sinks = torch.rand([NUM_HEADS], dtype=torch.float32)

        impl = PallasAttentionBackendImpl(num_heads=NUM_HEADS,
                                          head_size=head_dim,
                                          scale=0.088,
                                          num_kv_heads=NUM_KV_HEADS,
                                          alibi_slopes=None,
                                          sliding_window=None,
                                          kv_cache_dtype="auto",
                                          attn_type=AttentionType.DECODER,
                                          sinks=sinks)
        impl.process_weights_after_loading(torch.bfloat16)

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, head_dim=head_dim)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}):
            assert impl.sinks is not None
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)

    def test_forward_with_attention_sink_head_dim_128_raises_error(self, mesh):
        head_dim = 128
        sinks = torch.rand([NUM_HEADS], dtype=torch.float32)

        impl = PallasAttentionBackendImpl(num_heads=NUM_HEADS,
                                          head_size=head_dim,
                                          scale=0.088,
                                          num_kv_heads=NUM_KV_HEADS,
                                          alibi_slopes=None,
                                          sliding_window=None,
                                          kv_cache_dtype="auto",
                                          attn_type=AttentionType.DECODER,
                                          sinks=sinks)
        impl.process_weights_after_loading(torch.bfloat16)

        layer = MagicMock()
        layer.layer_name = "0"

        query, key, value, kv_cache, metadata = create_inputs(
            mesh, head_dim=head_dim)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={'0': 0}
        ), pytest.raises(
                NotImplementedError,
                match=
                "Attention sink support is only available when head_dim==64"):
            assert impl.sinks is not None
            impl.forward(layer, query, key, value, torch.tensor([]), metadata)
