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
"""Qwen MLP patches for the vLLM TorchAX path."""

import importlib
from typing import Any

import torch.nn as nn
import torch.nn.functional as F

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_PATCHED_ATTR = "_tpu_inference_qwen_mlp_act_barrier_patched"
_ORIGINAL_FORWARD_ATTR = "_tpu_inference_original_forward"
_FORCE_BARRIER_ATTR = "_tpu_inference_force_act_barrier"


def _get_qwen_mlp_cls(module_name: str,
                      class_name: str) -> type[nn.Module] | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("Unable to import %s for Qwen MLP patching: %s",
                     module_name, exc)
        return None

    mlp_cls = getattr(module, class_name, None)
    if mlp_cls is None:
        logger.debug("vLLM %s class was not found.", class_name)
        return None
    return mlp_cls


def _get_qwen_mlp_classes() -> tuple[type[nn.Module], ...]:
    classes = []
    for module_name, class_name in (
        ("vllm.model_executor.models.qwen2", "Qwen2MLP"),
        ("vllm.model_executor.models.qwen2_moe", "Qwen2MoeMLP"),
    ):
        mlp_cls = _get_qwen_mlp_cls(module_name, class_name)
        if mlp_cls is not None:
            classes.append(mlp_cls)
    return tuple(classes)


def _install_qwen_mlp_forward_patch(mlp_cls: type[nn.Module]) -> None:
    if getattr(mlp_cls, _PATCHED_ATTR, False):
        return

    original_forward = mlp_cls.forward

    def _tpu_inference_forward(self: Any, x: Any) -> Any:
        original_x = x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        if getattr(self, _FORCE_BARRIER_ATTR, False):
            import jax
            from torchax.interop import jax_view, torch_view

            x = torch_view(jax.lax.optimization_barrier(jax_view(x)))
        x, _ = self.down_proj(x)

        expert_gate = getattr(self, "expert_gate", None)
        if expert_gate is not None:
            x = F.sigmoid(expert_gate(original_x)[0]) * x
        return x

    setattr(mlp_cls, _ORIGINAL_FORWARD_ATTR, original_forward)
    mlp_cls.forward = _tpu_inference_forward
    setattr(mlp_cls, _PATCHED_ATTR, True)
    logger.info("Installed vLLM %s activation barrier patch.",
                mlp_cls.__name__)


def maybe_apply_qwen_mlp_activation_barrier(vllm_model: nn.Module,
                                            *,
                                            enabled: bool) -> None:
    """Enable Qwen MLP activation materialization for this model instance.

    Qwen3MLP is an alias of vLLM's Qwen2MLP, and Qwen3Next shared experts use
    Qwen2MoeMLP. On affected TPU/XLA stacks, fused bf16
    `silu(gate) * up -> down_proj` under PCP sharding can differ from
    materializing the activation before the down projection. The class patch is
    installed once, and the barrier itself is toggled per module instance so
    PCP=1 baseline runs can keep the original path.
    """
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        return

    for mlp_cls in mlp_classes:
        _install_qwen_mlp_forward_patch(mlp_cls)

    patched_modules = 0
    for module in vllm_model.modules():
        if isinstance(module, mlp_classes):
            setattr(module, _FORCE_BARRIER_ATTR, enabled)
            patched_modules += 1

    if patched_modules:
        logger.info(
            "Configured Qwen MLP activation barrier for %d module(s): "
            "enabled=%s", patched_modules, enabled)
