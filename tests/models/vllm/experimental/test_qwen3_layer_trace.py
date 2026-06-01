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

import importlib.util
from pathlib import Path
from types import MethodType

import numpy as np


def _load_qwen3_layer_trace():
    repo_root = Path(__file__).parents[4]
    trace_path = (
        repo_root /
        "tpu_inference/models/vllm/experimental/qwen3_layer_trace.py")
    spec = importlib.util.spec_from_file_location(
        "qwen3_layer_trace_under_test", trace_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qwen3_layer_trace = _load_qwen3_layer_trace()


class _FakeLayer:

    layer_idx = 3
    layer_type = "linear_attention"
    input_layernorm = object()
    post_attention_layernorm = object()
    mlp = object()
    linear_attn = object()

    def forward(self, *args, **kwargs):
        return "original"


class _FakeRouter:

    def select_experts(self, *, hidden_states, router_logits, input_ids=None):
        return f"{hidden_states}:weights", f"{router_logits}:ids"


class _FakeMoeRunner:

    def __init__(self):
        self.router = _FakeRouter()

    def _maybe_dispatch(self, layer, hidden_states, router_logits):
        return f"{hidden_states}:dispatch", f"{router_logits}:dispatch"

    def _apply_quant_method(self,
                            *,
                            layer,
                            hidden_states,
                            router_logits,
                            shared_experts_input,
                            input_ids=None):
        return shared_experts_input, f"{hidden_states}:fused"

    def _maybe_combine(self, shared_output, hidden_states):
        if shared_output is None:
            return f"{hidden_states}:combined"
        return f"{shared_output}+{hidden_states}", None


class _FakeExperts:

    is_internal_router = False

    def __init__(self):
        self.runner = _FakeMoeRunner()

    def __call__(self, *, hidden_states, router_logits):
        return f"{hidden_states}:{router_logits}:experts"


class _FakeSparseMoe:

    is_sequence_parallel = False

    def __init__(self):
        self.gate = object()
        self.shared_expert = _FakeQwenMoeMLP()
        self.experts = _FakeExperts()

    def forward(self, hidden_states):
        return f"original:{hidden_states}"


class _FakeQwenMoeMLP:

    gate_up_proj = object()
    down_proj = object()
    act_fn = object()

    def forward(self, hidden_states):
        return f"original:{hidden_states}"


class _FakeLayerWithMoe(_FakeLayer):

    def __init__(self):
        self.mlp = _FakeSparseMoe()


class _FakeModel:

    def __init__(self, layer):
        self.layer = layer

    def modules(self):
        return [self, self.layer]


def test_trace_torch_tensor_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(qwen3_layer_trace.TRACE_ENABLED_ENV, raising=False)
    value = object()

    assert qwen3_layer_trace.trace_torch_tensor("stage", value) is value


def test_trace_dump_stage_selection(monkeypatch, tmp_path):
    monkeypatch.setenv(qwen3_layer_trace.TRACE_DUMP_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(qwen3_layer_trace.TRACE_DUMP_STAGES_ENV, "a.b,c.d")

    assert qwen3_layer_trace._should_dump_stage("a.b")
    assert qwen3_layer_trace._should_dump_stage("c.d")
    assert not qwen3_layer_trace._should_dump_stage("x.y")


def test_write_record_includes_actual_dtype(monkeypatch, tmp_path):
    import json

    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(qwen3_layer_trace.TRACE_JSONL_ENV, str(path))

    qwen3_layer_trace._write_record(
        stage="stage",
        shape=(1, ),
        dtype="torch.bfloat16",
        actual_dtype="bfloat16",
        size=1,
        mean=0.0,
        std=0.0,
        min_value=0.0,
        max_value=0.0,
        max_abs=0.0,
        sample=[0.0],
    )

    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["dtype"] == "torch.bfloat16"
    assert record["actual_dtype"] == "bfloat16"


def test_gemma_rms_norm_reference_matches_manual_numpy():
    import jax.numpy as jnp

    x = jnp.array([[1.0, -2.0, 3.0], [0.5, 1.5, -2.5]],
                  dtype=jnp.float32)
    residual = jnp.array([[0.25, 0.5, -0.75], [1.0, -0.5, 0.25]],
                         dtype=jnp.float32)
    weight = jnp.array([0.0, 0.5, -0.25], dtype=jnp.float32)
    eps = 1e-6

    out, input_sum, variance = qwen3_layer_trace._gemma_rms_norm_reference(
        x, residual, weight, eps)

    expected_input = np.asarray(x + residual)
    expected_variance = np.mean(expected_input * expected_input,
                                axis=-1,
                                keepdims=True)
    expected_out = (expected_input / np.sqrt(expected_variance + eps) *
                    (np.asarray(weight) + 1.0))

    np.testing.assert_allclose(np.asarray(input_sum), expected_input)
    np.testing.assert_allclose(np.asarray(variance), expected_variance)
    np.testing.assert_allclose(np.asarray(out), expected_out, rtol=1e-6)


def test_trace_post_attention_norm_inputs_records_actual_inputs(monkeypatch):
    stages = []

    def fake_trace_torch_tensor(stage, value):
        stages.append((stage, value))
        return value

    monkeypatch.setattr(qwen3_layer_trace, "trace_torch_tensor",
                        fake_trace_torch_tensor)
    hidden_states = object()
    residual = object()

    qwen3_layer_trace.trace_post_attention_norm_inputs(
        "layer.00.linear_attention", hidden_states, residual)

    assert stages == [
        ("layer.00.linear_attention.post_attention_norm.input.hidden_states",
         hidden_states),
        ("layer.00.linear_attention.post_attention_norm.input.residual",
         residual),
    ]


def test_trace_post_attention_norm_raw_outputs_records_pre_cast_outputs(
        monkeypatch):
    stages = []

    def fake_trace_torch_tensor(stage, value):
        stages.append((stage, value))
        return value

    monkeypatch.setattr(qwen3_layer_trace, "trace_torch_tensor",
                        fake_trace_torch_tensor)
    hidden_states = object()
    residual = object()

    qwen3_layer_trace.trace_post_attention_norm_raw_outputs(
        "layer.00.linear_attention", hidden_states, residual)

    assert stages == [
        ("layer.00.linear_attention.post_attention_norm.raw.hidden_states",
         hidden_states),
        ("layer.00.linear_attention.post_attention_norm.raw.residual",
         residual),
    ]


def test_moe_runner_trace_wrappers_preserve_returns(monkeypatch):
    stages = []

    def fake_trace_torch_tensor(stage, value):
        stages.append((stage, value))
        return value

    monkeypatch.setattr(qwen3_layer_trace, "trace_torch_tensor",
                        fake_trace_torch_tensor)
    runner = _FakeMoeRunner()

    assert qwen3_layer_trace._patch_moe_runner_trace(
        runner, "layer.00.linear_attention.moe.runner")

    assert runner._maybe_dispatch("layer", "h", "r") == (
        "h:dispatch", "r:dispatch")
    assert runner.router.select_experts(hidden_states="h",
                                        router_logits="r") == ("h:weights",
                                                               "r:ids")
    assert runner._apply_quant_method(layer="layer",
                                      hidden_states="h",
                                      router_logits="r",
                                      shared_experts_input=None) == (
                                          None, "h:fused")
    assert runner._maybe_combine(None, "h") == "h:combined"

    traced_stage_names = [stage for stage, _ in stages]
    expected_stages = {
        "layer.00.linear_attention.moe.runner.dispatch.input.hidden_states",
        "layer.00.linear_attention.moe.runner.dispatch.output.router_logits",
        "layer.00.linear_attention.moe.runner.router.topk_weights",
        "layer.00.linear_attention.moe.runner.router.topk_ids",
        "layer.00.linear_attention.moe.runner.quant.output.fused",
        "layer.00.linear_attention.moe.runner.combine.output",
    }
    assert expected_stages.issubset(traced_stage_names)


def test_qwen_moe_mlp_trace_patch(monkeypatch):
    mlp = _FakeQwenMoeMLP()

    assert qwen3_layer_trace._patch_qwen_moe_mlp_trace(
        mlp, "layer.00.linear_attention.moe.shared")

    assert getattr(mlp, qwen3_layer_trace._PATCHED_ATTR)
    assert getattr(mlp, qwen3_layer_trace._TRACE_PREFIX_ATTR) == (
        "layer.00.linear_attention.moe.shared")
    assert isinstance(mlp.forward, MethodType)
    assert mlp.forward.__func__ is (
        qwen3_layer_trace._trace_qwen_moe_mlp_forward)


def test_qwen3_layer_trace_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(qwen3_layer_trace.TRACE_ENABLED_ENV, raising=False)
    layer = _FakeLayer()

    qwen3_layer_trace.maybe_apply_qwen3_layer_trace(_FakeModel(layer))

    assert not getattr(layer, qwen3_layer_trace._PATCHED_ATTR, False)


def test_qwen3_layer_trace_patches_decoder_layers(monkeypatch):
    monkeypatch.setenv(qwen3_layer_trace.TRACE_ENABLED_ENV, "1")
    layer = _FakeLayer()

    qwen3_layer_trace.maybe_apply_qwen3_layer_trace(_FakeModel(layer))

    assert getattr(layer, qwen3_layer_trace._PATCHED_ATTR)
    assert isinstance(layer.forward, MethodType)
    assert layer.forward.__func__ is (
        qwen3_layer_trace._trace_decoder_layer_forward)


def test_qwen3_layer_trace_patches_sparse_moe(monkeypatch):
    monkeypatch.setenv(qwen3_layer_trace.TRACE_ENABLED_ENV, "1")
    layer = _FakeLayerWithMoe()

    qwen3_layer_trace.maybe_apply_qwen3_layer_trace(_FakeModel(layer))

    assert getattr(layer.mlp, qwen3_layer_trace._PATCHED_ATTR)
    assert getattr(layer.mlp.experts.runner,
                   qwen3_layer_trace._MOE_RUNNER_PATCHED_ATTR)
    assert getattr(layer.mlp.experts.runner.router,
                   qwen3_layer_trace._MOE_ROUTER_PATCHED_ATTR)
    assert getattr(layer.mlp.shared_expert, qwen3_layer_trace._PATCHED_ATTR)
    assert getattr(layer.mlp,
                   qwen3_layer_trace._TRACE_PREFIX_ATTR) == (
                       "layer.03.linear_attention.moe")
    assert isinstance(layer.mlp.forward, MethodType)
    assert layer.mlp.forward.__func__ is (
        qwen3_layer_trace._trace_qwen3_sparse_moe_forward)
