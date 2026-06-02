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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from tpu_inference.layers.vllm.custom_ops import gdn_attention_op
from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


class TestVllmGatedDeltaNetAttention:

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.gdn_attention_core_tpu"
    )
    def test_forward_cuda_lora(self, mock_gdn_attention_core_tpu, mesh):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"

        # Mocks for LoRA path (uses in_proj_qkv and in_proj_z)
        attn.in_proj_qkv = MagicMock()
        attn.in_proj_z = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        attn.in_proj_qkv.return_value = (torch.randn(num_tokens, 96), None)
        attn.in_proj_z.return_value = (torch.randn(num_tokens, 64), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        with set_vllm_model_wrapper_context(kv_caches=[],
                                            mesh=mesh,
                                            layer_name_to_kvcache_index={}):
            attn.forward(hidden_states, output)

        attn.in_proj_qkv.assert_called_once_with(hidden_states)
        attn.in_proj_z.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert mock_gdn_attention_core_tpu.call_count == 1
        core_args = mock_gdn_attention_core_tpu.call_args[0]
        core_kwargs = mock_gdn_attention_core_tpu.call_args[1]

        assert core_args[0].shape == (num_tokens, 96)  # mixed_qkv
        assert core_args[1].shape == (num_tokens, 16)  # b
        assert core_args[2].shape == (num_tokens, 16)  # a
        assert core_args[3].shape == (num_tokens, 4, 16)  # core_attn_out
        assert core_args[3].dtype == hidden_states.dtype
        assert core_args[4] == "test_layer"
        assert core_kwargs["mesh"] == mesh

        attn.norm.assert_called_once()
        # Verify z was correctly reshaped: [num_tokens, -1, head_v_dim]
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        # Verify reshaped output from norm went to out_proj
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        # Check that output buffer was updated only up to num_tokens
        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.gdn_attention_core_tpu"
    )
    def test_forward_cuda_non_lora_no_gqa(self, mock_gdn_attention_core_tpu,
                                          mesh):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = False
        attn.key_dim = 32
        attn.value_dim = 64

        # Mocks for non-LoRA no GQA path
        attn.in_proj_qkvz = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size  # 128
        z_size = attn.value_dim // attn.tp_size  # 64
        mixed_qkvz = torch.randn(num_tokens, qkv_size + z_size)

        attn.in_proj_qkvz.return_value = (mixed_qkvz, None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        with set_vllm_model_wrapper_context(kv_caches=[],
                                            mesh=mesh,
                                            layer_name_to_kvcache_index={}):
            attn.forward(hidden_states, output)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert mock_gdn_attention_core_tpu.call_count == 1
        core_args = mock_gdn_attention_core_tpu.call_args[0]
        core_kwargs = mock_gdn_attention_core_tpu.call_args[1]

        # mixed_qkv should be separated accurately
        assert core_args[0].shape == (num_tokens, 128)
        assert core_args[1].shape == (num_tokens, 16)
        assert core_args[2].shape == (num_tokens, 16)
        assert core_args[3].shape == (num_tokens, 4, 16)
        assert core_args[4] == "test_layer"
        assert core_kwargs["mesh"] == mesh

        attn.norm.assert_called_once()
        # Verify z was split and reshaped correctly
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.gdn_attention_core_tpu"
    )
    def test_forward_cuda_non_lora_gqa(self, mock_gdn_attention_core_tpu,
                                       mesh):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True

        # Mocks for non-LoRA GQA path
        attn.in_proj_qkvz = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.fix_query_key_value_ordering = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        attn.in_proj_qkvz.return_value = (torch.randn(num_tokens, 192), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        query = torch.randn(num_tokens, 4, 8)
        key = torch.randn(num_tokens, 4, 8)
        value = torch.randn(num_tokens, 4, 8)
        z = torch.randn(num_tokens, 4, 16)
        b = torch.randn(num_tokens, 16)
        a = torch.randn(num_tokens, 16)

        attn.fix_query_key_value_ordering.return_value = (query, key, value, z,
                                                          b, a)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        with set_vllm_model_wrapper_context(kv_caches=[],
                                            mesh=mesh,
                                            layer_name_to_kvcache_index={}):
            attn.forward(hidden_states, output)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)
        attn.fix_query_key_value_ordering.assert_called_once()

        assert mock_gdn_attention_core_tpu.call_count == 1
        core_args = mock_gdn_attention_core_tpu.call_args[0]
        core_kwargs = mock_gdn_attention_core_tpu.call_args[1]

        # mixed_qkv should be cat of rearranged query, key, value
        # rearranged from "l p d -> l (p d)", e.g. 2x(4*8) = 2x32 -> cat into 2x96
        assert core_args[0].shape == (num_tokens, 96)
        assert core_args[1].shape == (num_tokens, 16)
        assert core_args[2].shape == (num_tokens, 16)
        assert core_args[3].shape == (num_tokens, 4, 16)
        assert core_args[4] == "test_layer"
        assert core_kwargs["mesh"] == mesh

        attn.norm.assert_called_once()
        # Verify unpacked z is natively used
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)


class TestGdnAttentionCoreRouting:

    def test_gdn_trace_stage_prefix_uses_layer_index(self):
        assert gdn_attention_op._gdn_trace_stage_prefix(
            "model.layers.12.linear_attn") == (
                "layer.12.linear_attention.gdn")

    def test_cast_jax_output_to_buffer_dtype(self, monkeypatch):
        class FakeJaxArray:

            dtype = np.dtype("float32")

            def __init__(self):
                self.cast_dtype = None

            def astype(self, dtype):
                self.cast_dtype = dtype
                return ("cast", dtype)

        target_dtype = np.dtype("float16")
        fake_output = FakeJaxArray()
        monkeypatch.setattr(gdn_attention_op, "jax_view",
                            lambda _value: SimpleNamespace(dtype=target_dtype))

        result = gdn_attention_op._cast_jax_output_to_buffer_dtype(
            fake_output, object())

        assert result == ("cast", target_dtype)
        assert fake_output.cast_dtype == target_dtype

    def test_cast_jax_output_prefers_torch_buffer_dtype(self, monkeypatch):
        class FakeJaxArray:

            dtype = jnp.float32

            def astype(self, dtype):
                return ("cast", dtype)

        monkeypatch.setattr(gdn_attention_op, "jax_view",
                            lambda _value: SimpleNamespace(dtype=jnp.float32))

        result = gdn_attention_op._cast_jax_output_to_buffer_dtype(
            FakeJaxArray(), SimpleNamespace(dtype=torch.bfloat16))

        assert result == ("cast", jnp.bfloat16)

    def test_cast_jax_output_keeps_matching_dtype(self, monkeypatch):
        class FakeJaxArray:

            dtype = np.dtype("float32")

            def astype(self, dtype):
                raise AssertionError(f"unexpected cast to {dtype}")

        fake_output = FakeJaxArray()
        monkeypatch.setattr(
            gdn_attention_op,
            "jax_view",
            lambda _value: SimpleNamespace(dtype=fake_output.dtype),
        )

        assert gdn_attention_op._cast_jax_output_to_buffer_dtype(
            fake_output, object()) is fake_output

    def test_cast_torchax_tensor_to_torch_dtype(self, monkeypatch):
        class FakeJaxArray:

            dtype = jnp.float32

            def astype(self, dtype):
                return FakeJaxArrayCast([dtype])

        class FakeJaxArrayCast:

            def __init__(self, dtypes):
                self.dtypes = dtypes

            def astype(self, dtype):
                return FakeJaxArrayCast(self.dtypes + [dtype])

        monkeypatch.setattr(gdn_attention_op, "jax_view",
                            lambda _value: FakeJaxArray())
        monkeypatch.setattr(gdn_attention_op, "torch_view",
                            lambda value: ("torch", value))
        monkeypatch.setattr(gdn_attention_op.jax.lax,
                            "optimization_barrier", lambda value: value)

        result = gdn_attention_op._cast_torchax_tensor_to_torch_dtype(
            object(), torch.bfloat16)

        assert result[0] == "torch"
        assert result[1].dtypes == [jnp.float32, jnp.bfloat16]

    def test_cast_torchax_tensor_forces_roundtrip_on_matching_dtype(
            self, monkeypatch):
        class FakeJaxArray:

            dtype = jnp.bfloat16

            def __init__(self, dtypes=()):
                self.dtypes = list(dtypes)

            def astype(self, dtype):
                return FakeJaxArray(self.dtypes + [dtype])

        monkeypatch.setattr(gdn_attention_op, "jax_view",
                            lambda _value: FakeJaxArray())
        monkeypatch.setattr(gdn_attention_op, "torch_view",
                            lambda value: ("torch", value))
        monkeypatch.setattr(gdn_attention_op.jax.lax,
                            "optimization_barrier", lambda value: value)

        result = gdn_attention_op._cast_torchax_tensor_to_torch_dtype(
            object(), torch.bfloat16)

        assert result[0] == "torch"
        assert result[1].dtypes == [jnp.float32, jnp.bfloat16]

    def _run_core_with_reorder_indices(self, monkeypatch, reorder_indices):
        calls = {"pcp": 0, "standard": 0}
        layer_name = "layer.0"
        num_tokens = 2
        n_kq = 2
        n_v = 2
        d_k = 4
        d_v = 4
        kernel_size = 4
        dim = n_kq * d_k + n_kq * d_k + n_v * d_v

        attn_metadata = SimpleNamespace(
            mamba_state_indices=np.array([1, 2], dtype=np.int32),
            query_start_loc=np.array([0, 1, 2], dtype=np.int32),
            seq_lens=np.array([8, 16], dtype=np.int32),
            request_distribution=np.array([2, 2, 2], dtype=np.int32),
            pcp_gdn_reorder_indices=reorder_indices,
            padded_num_reqs=2,
        )
        layer_module = SimpleNamespace(
            num_k_heads=n_kq,
            num_v_heads=n_v,
            head_k_dim=d_k,
            head_v_dim=d_v,
            conv_kernel_size=kernel_size,
            conv1d=SimpleNamespace(
                weight=torch.randn(dim, 1, kernel_size),
                bias=torch.randn(dim),
            ),
            A_log=torch.randn(n_v),
            dt_bias=torch.randn(n_v),
        )
        forward_context = SimpleNamespace(
            attn_metadata={layer_name: attn_metadata},
            no_compile_layers={layer_name: layer_module},
        )
        conv_state = torch.zeros(3, kernel_size - 1, dim)
        recurrent_state = torch.zeros(3, n_v, d_k, d_v)
        wrapper_context = SimpleNamespace(
            layer_name_to_kvcache_index={layer_name: 0},
            kv_caches=[(conv_state, recurrent_state)],
        )

        def fake_get_mesh_shape_product(_mesh, axis):
            if axis == gdn_attention_op.ShardingAxisName.PREFILL_CONTEXT:
                return 2
            return 1

        def fake_pcp(*args, **kwargs):
            calls["pcp"] += 1
            calls["pcp_reorder_indices"] = args[13]
            calls["pcp_size"] = kwargs["pcp_size"]
            output = torch.ones(num_tokens, n_v * d_v)
            return (conv_state + 1, recurrent_state + 1), output

        def fake_standard(*args, **kwargs):
            calls["standard"] += 1
            output = torch.ones(num_tokens, n_v * d_v) * 2
            return (conv_state + 2, recurrent_state + 2), output

        monkeypatch.setattr(gdn_attention_op, "get_forward_context",
                            lambda: forward_context)
        monkeypatch.setattr(gdn_attention_op,
                            "get_vllm_model_wrapper_context",
                            lambda: wrapper_context)
        monkeypatch.setattr(gdn_attention_op, "jax_view", lambda x: x)
        monkeypatch.setattr(gdn_attention_op, "torch_view", lambda x: x)
        monkeypatch.setattr(gdn_attention_op,
                            "reorder_concatenated_tensor_for_sharding",
                            lambda tensor, *_args, **_kwargs: tensor)
        monkeypatch.setattr(gdn_attention_op, "truncate_sharded_tensor",
                            lambda tensor, *_args, **_kwargs: tensor)
        monkeypatch.setattr(gdn_attention_op, "get_mesh_shape_product",
                            fake_get_mesh_shape_product)
        monkeypatch.setattr(gdn_attention_op,
                            "run_jax_gdn_attention_pcp_tp_prefill", fake_pcp)
        monkeypatch.setattr(gdn_attention_op, "run_jax_gdn_attention",
                            fake_standard)

        core_attn_out = torch.zeros(num_tokens, n_v, d_v)
        gdn_attention_op.gdn_attention_core_tpu(
            torch.randn(num_tokens, dim),
            torch.randn(num_tokens, n_v),
            torch.randn(num_tokens, n_v),
            core_attn_out,
            layer_name,
            mesh=object(),
        )
        return calls, core_attn_out, wrapper_context

    def test_core_routes_pcp_prefill_when_reorder_indices_present(
            self, monkeypatch):
        reorder_indices = np.array([0, 1], dtype=np.int32)

        calls, core_attn_out, wrapper_context = (
            self._run_core_with_reorder_indices(monkeypatch, reorder_indices))

        assert calls["pcp"] == 1
        assert calls["standard"] == 0
        assert calls["pcp_reorder_indices"] is reorder_indices
        assert calls["pcp_size"] == 2
        assert torch.all(core_attn_out == 1)
        assert torch.all(wrapper_context.kv_caches[0][0] == 1)

    def test_core_routes_standard_path_without_reorder_indices(
            self, monkeypatch):
        calls, core_attn_out, wrapper_context = (
            self._run_core_with_reorder_indices(monkeypatch, None))

        assert calls["pcp"] == 0
        assert calls["standard"] == 1
        assert torch.all(core_attn_out == 2)
        assert torch.all(wrapper_context.kv_caches[0][0] == 2)
