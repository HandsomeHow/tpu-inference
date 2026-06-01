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
import sys
import types
from pathlib import Path


def _load_qwen_mlp_patcher():
    repo_root = Path(__file__).parents[4]
    patcher_path = (
        repo_root /
        "tpu_inference/models/vllm/experimental/qwen_mlp_patcher.py")
    spec = importlib.util.spec_from_file_location("qwen_mlp_patcher_under_test",
                                                  patcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qwen_mlp_patcher = _load_qwen_mlp_patcher()


class _FakeLinear:

    def __init__(self, suffix):
        self.suffix = suffix

    def __call__(self, x):
        return f"{x}{self.suffix}", None


class _FakeQwen2MLP:

    def __init__(self):
        self.gate_up_proj = _FakeLinear(":gate")
        self.act_fn = lambda x: f"{x}:act"
        self.down_proj = _FakeLinear(":down")

    def forward(self, x):
        return f"original:{x}"


class _FakeExpertGate:

    def __call__(self, x):
        return f"{x}:expert_gate", None


class _FakeQwen2MoeMLP(_FakeQwen2MLP):

    def __init__(self):
        super().__init__()
        self.expert_gate = None


class _FakeScale:

    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        return f"{self.value}*{other}"


class _FakeModel:

    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return [self, *self._modules]


def _install_fake_vllm_qwen2(monkeypatch,
                             qwen2_mlp_cls,
                             qwen2_moe_mlp_cls=None):
    packages = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
    ]
    for package in packages:
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))

    qwen2_mod = types.ModuleType("vllm.model_executor.models.qwen2")
    qwen2_mod.Qwen2MLP = qwen2_mlp_cls
    monkeypatch.setitem(sys.modules, qwen2_mod.__name__, qwen2_mod)

    qwen2_moe_mod = types.ModuleType("vllm.model_executor.models.qwen2_moe")
    if qwen2_moe_mlp_cls is not None:
        qwen2_moe_mod.Qwen2MoeMLP = qwen2_moe_mlp_cls
    monkeypatch.setitem(sys.modules, qwen2_moe_mod.__name__, qwen2_moe_mod)


def test_qwen_mlp_patch_is_enabled_per_module(monkeypatch):
    class FakeQwen2MLP(_FakeQwen2MLP):
        pass

    class FakeQwen2MoeMLP(_FakeQwen2MoeMLP):
        pass

    _install_fake_vllm_qwen2(monkeypatch, FakeQwen2MLP,
                             FakeQwen2MoeMLP)
    mlp = FakeQwen2MLP()
    moe_mlp = FakeQwen2MoeMLP()

    qwen_mlp_patcher.maybe_apply_qwen_mlp_activation_barrier(
        _FakeModel([mlp, moe_mlp]), enabled=True)

    assert getattr(FakeQwen2MLP, "_tpu_inference_qwen_mlp_act_barrier_patched")
    assert getattr(FakeQwen2MoeMLP,
                   "_tpu_inference_qwen_mlp_act_barrier_patched")
    assert getattr(FakeQwen2MLP, "_tpu_inference_original_forward")
    assert getattr(mlp, "_tpu_inference_force_act_barrier") is True
    assert getattr(moe_mlp, "_tpu_inference_force_act_barrier") is True


def test_qwen_mlp_patch_preserves_original_path_when_disabled(monkeypatch):
    class FakeQwen2MLP(_FakeQwen2MLP):
        pass

    _install_fake_vllm_qwen2(monkeypatch, FakeQwen2MLP)
    mlp = FakeQwen2MLP()

    qwen_mlp_patcher.maybe_apply_qwen_mlp_activation_barrier(
        _FakeModel([mlp]), enabled=False)

    assert mlp.forward("x") == "x:gate:act:down"


def test_qwen_mlp_patch_applies_barrier_when_enabled(monkeypatch):
    class FakeQwen2MLP(_FakeQwen2MLP):
        pass

    _install_fake_vllm_qwen2(monkeypatch, FakeQwen2MLP)

    jax_mod = types.ModuleType("jax")
    jax_mod.lax = types.SimpleNamespace(
        optimization_barrier=lambda x: f"barrier({x})")
    torchax_mod = types.ModuleType("torchax")
    interop_mod = types.ModuleType("torchax.interop")
    interop_mod.jax_view = lambda x: f"jax({x})"
    interop_mod.torch_view = lambda x: f"torch({x})"
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "torchax", torchax_mod)
    monkeypatch.setitem(sys.modules, "torchax.interop", interop_mod)

    mlp = FakeQwen2MLP()
    qwen_mlp_patcher.maybe_apply_qwen_mlp_activation_barrier(
        _FakeModel([mlp]), enabled=True)

    assert mlp.forward("x") == "torch(barrier(jax(x:gate:act))):down"


def test_qwen_moe_mlp_patch_preserves_expert_gate(monkeypatch):
    class FakeQwen2MLP(_FakeQwen2MLP):
        pass

    class FakeQwen2MoeMLP(_FakeQwen2MoeMLP):
        pass

    _install_fake_vllm_qwen2(monkeypatch, FakeQwen2MLP,
                             FakeQwen2MoeMLP)
    monkeypatch.setattr(qwen_mlp_patcher, "F",
                        types.SimpleNamespace(
                            sigmoid=lambda x: _FakeScale(f"sigmoid({x})")))

    mlp = FakeQwen2MoeMLP()
    mlp.expert_gate = _FakeExpertGate()
    qwen_mlp_patcher.maybe_apply_qwen_mlp_activation_barrier(
        _FakeModel([mlp]), enabled=False)

    assert mlp.forward("x") == "sigmoid(x:expert_gate)*x:gate:act:down"
