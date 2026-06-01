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
"""Qwen3 decoder patches for the vLLM TorchAX path."""

import importlib
from typing import Any

import torch.nn as nn

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_PATCHED_ATTR = "_tpu_inference_qwen3_decoder_output_cast_patched"
_ORIGINAL_FORWARD_ATTR = "_tpu_inference_original_forward"
_FORCE_OUTPUT_CAST_ATTR = "_tpu_inference_force_attention_output_cast"


def _get_qwen3_next_decoder_layer_cls() -> type[nn.Module] | None:
    try:
        qwen3_next_mod = importlib.import_module(
            "vllm.model_executor.models.qwen3_next")
    except Exception as exc:
        logger.debug("Unable to import vLLM Qwen3Next decoder: %s", exc)
        return None

    decoder_cls = getattr(qwen3_next_mod, "Qwen3NextDecoderLayer", None)
    if decoder_cls is None:
        logger.debug("vLLM Qwen3NextDecoderLayer class was not found.")
        return None
    return decoder_cls


def _cast_attention_output(hidden_states: Any, dtype: Any) -> Any:
    from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import (
        _cast_torchax_tensor_to_torch_dtype)

    return _cast_torchax_tensor_to_torch_dtype(hidden_states, dtype)


def maybe_cast_attention_output(module: Any, hidden_states: Any) -> Any:
    if not getattr(module, _FORCE_OUTPUT_CAST_ATTR, False):
        return hidden_states
    return _cast_attention_output(hidden_states, hidden_states.dtype)


def maybe_cast_residual(module: Any, residual: Any) -> Any:
    if residual is None or not getattr(module, _FORCE_OUTPUT_CAST_ATTR, False):
        return residual
    return _cast_attention_output(residual, residual.dtype)


def _install_qwen3_decoder_forward_patch(
        decoder_cls: type[nn.Module]) -> None:
    if getattr(decoder_cls, _PATCHED_ATTR, False):
        return

    original_forward = decoder_cls.forward

    def _tpu_inference_forward(
        self: Any,
        hidden_states: Any,
        residual: Any,
        positions: Any = None,
        **kwargs: object,
    ) -> tuple[Any, Any]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = maybe_cast_attention_output(self, hidden_states)
        residual = maybe_cast_residual(self, residual)

        import torch

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        hidden_states = maybe_cast_attention_output(self,
                                                    self_attention_output)
        residual = maybe_cast_residual(self, residual)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = maybe_cast_attention_output(self, hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(
                    self.ffn_layer_scale.shape), (
                        f"shape must be the same {len(hidden_states.shape)}, "
                        f"{len(self.ffn_layer_scale.shape)}")
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual

    setattr(decoder_cls, _ORIGINAL_FORWARD_ATTR, original_forward)
    decoder_cls.forward = _tpu_inference_forward
    setattr(decoder_cls, _PATCHED_ATTR, True)
    logger.info("Installed vLLM Qwen3 decoder output dtype patch.")


def maybe_apply_qwen3_decoder_output_cast(vllm_model: nn.Module,
                                          *,
                                          enabled: bool) -> None:
    decoder_cls = _get_qwen3_next_decoder_layer_cls()
    if decoder_cls is None:
        return

    _install_qwen3_decoder_forward_patch(decoder_cls)

    patched_modules = 0
    for module in vllm_model.modules():
        if isinstance(module, decoder_cls):
            setattr(module, _FORCE_OUTPUT_CAST_ATTR, enabled)
            patched_modules += 1

    if patched_modules:
        logger.info(
            "Configured Qwen3 decoder output dtype patch for %d module(s): "
            "enabled=%s", patched_modules, enabled)
