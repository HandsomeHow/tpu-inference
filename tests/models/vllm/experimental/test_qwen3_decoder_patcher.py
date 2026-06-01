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

import torch
import torch.nn as nn

from tpu_inference.models.vllm.experimental import qwen3_decoder_patcher


def test_maybe_cast_attention_output_is_noop_when_disabled():
    hidden_states = object()
    module = SimpleNamespace()

    assert qwen3_decoder_patcher.maybe_cast_attention_output(
        module, hidden_states) is hidden_states


def test_maybe_cast_attention_output_casts_when_enabled(monkeypatch):
    hidden_states = SimpleNamespace(dtype="bf16")
    module = SimpleNamespace()
    setattr(module, qwen3_decoder_patcher._FORCE_OUTPUT_CAST_ATTR, True)

    monkeypatch.setattr(
        qwen3_decoder_patcher,
        "_cast_attention_output",
        lambda value, dtype: ("cast", value, dtype),
    )

    assert qwen3_decoder_patcher.maybe_cast_attention_output(
        module, hidden_states) == ("cast", hidden_states, "bf16")


def test_maybe_cast_residual_casts_when_enabled(monkeypatch):
    residual = SimpleNamespace(dtype="bf16")
    module = SimpleNamespace()
    setattr(module, qwen3_decoder_patcher._FORCE_OUTPUT_CAST_ATTR, True)

    monkeypatch.setattr(
        qwen3_decoder_patcher,
        "_cast_attention_output",
        lambda value, dtype: ("cast", value, dtype),
    )

    assert qwen3_decoder_patcher.maybe_cast_residual(
        module, residual) == ("cast", residual, "bf16")


def test_maybe_apply_qwen3_decoder_output_cast_installs_patch(monkeypatch):

    class FakeDecoderLayer(nn.Module):

        def forward(self):
            return "original"

    layer = FakeDecoderLayer()
    model = SimpleNamespace(modules=lambda: [model, layer])
    monkeypatch.setattr(qwen3_decoder_patcher,
                        "_get_qwen3_next_decoder_layer_cls",
                        lambda: FakeDecoderLayer)

    qwen3_decoder_patcher.maybe_apply_qwen3_decoder_output_cast(model,
                                                                enabled=True)

    assert getattr(FakeDecoderLayer, qwen3_decoder_patcher._PATCHED_ATTR)
    assert getattr(layer, qwen3_decoder_patcher._FORCE_OUTPUT_CAST_ATTR)
    assert getattr(FakeDecoderLayer,
                   qwen3_decoder_patcher._ORIGINAL_FORWARD_ATTR) is not None


def test_patched_forward_casts_post_norm_hidden_output(monkeypatch):

    class FakeDecoderLayer(nn.Module):

        def forward(self, *args, **kwargs):
            return "original"

    layer = FakeDecoderLayer()
    layer.layer_type = "linear_attention"
    layer.layer_scale = False
    layer.input_layernorm = lambda hidden_states: hidden_states
    layer.mlp = lambda hidden_states: hidden_states

    def linear_attn(*, hidden_states, output):
        output.copy_(hidden_states + 1)

    def post_attention_layernorm(hidden_states, residual):
        return hidden_states + 2, residual + 3

    layer.linear_attn = linear_attn
    layer.post_attention_layernorm = post_attention_layernorm
    model = SimpleNamespace(modules=lambda: [model, layer])
    monkeypatch.setattr(qwen3_decoder_patcher,
                        "_get_qwen3_next_decoder_layer_cls",
                        lambda: FakeDecoderLayer)

    cast_values = []

    def fake_cast(value, dtype):
        cast_values.append(value.clone())
        return value

    monkeypatch.setattr(qwen3_decoder_patcher, "_cast_attention_output",
                        fake_cast)

    qwen3_decoder_patcher.maybe_apply_qwen3_decoder_output_cast(model,
                                                                enabled=True)

    hidden_states, residual = layer.forward(torch.zeros(1, 2), None)

    assert len(cast_values) == 5
    assert torch.equal(cast_values[4], torch.full((1, 2), 3.0))
    assert torch.equal(hidden_states, torch.full((1, 2), 3.0))
    assert torch.equal(residual, torch.full((1, 2), 3.0))
