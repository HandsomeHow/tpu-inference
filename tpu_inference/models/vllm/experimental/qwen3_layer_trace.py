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
"""Runtime tensor tracing for Qwen3.5 vLLM models on the TorchAX path."""

from __future__ import annotations

import json
import os
import re
import threading
import types
from typing import Any

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

TRACE_ENABLED_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE"
TRACE_JSONL_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_JSONL"
TRACE_LABEL_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_LABEL"
TRACE_SAMPLE_SIZE_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_SAMPLE_SIZE"
TRACE_DUMP_DIR_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_DUMP_DIR"
TRACE_DUMP_STAGES_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_DUMP_STAGES"

_PATCHED_ATTR = "_tpu_inference_qwen3_layer_trace_patched"
_TRACE_PREFIX_ATTR = "_tpu_inference_qwen3_layer_trace_prefix"
_MOE_RUNNER_PATCHED_ATTR = "_tpu_inference_qwen3_moe_runner_trace_patched"
_MOE_ROUTER_PATCHED_ATTR = "_tpu_inference_qwen3_moe_router_trace_patched"
_QWEN_MLP_FORCE_BARRIER_ATTR = "_tpu_inference_force_act_barrier"
_TRACE_LOCK = threading.Lock()


def _env_enabled() -> bool:
    value = os.getenv(TRACE_ENABLED_ENV, "")
    return value.lower() in ("1", "true", "yes", "on")


def _sample_size() -> int:
    try:
        return max(0, int(os.getenv(TRACE_SAMPLE_SIZE_ENV, "8")))
    except ValueError:
        return 8


def _dump_stages() -> set[str]:
    value = os.getenv(TRACE_DUMP_STAGES_ENV, "")
    return {part.strip() for part in value.split(",") if part.strip()}


def _should_dump_stage(stage: str) -> bool:
    return stage in _dump_stages() and bool(os.getenv(TRACE_DUMP_DIR_ENV))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_record(
    *,
    stage: str,
    shape: tuple[int, ...],
    dtype: str,
    actual_dtype: str,
    size: int,
    mean: Any,
    std: Any,
    min_value: Any,
    max_value: Any,
    max_abs: Any,
    sample: Any,
) -> None:
    path = os.getenv(TRACE_JSONL_ENV)
    if not path:
        return

    import numpy as np

    record = {
        "label": os.getenv(TRACE_LABEL_ENV, ""),
        "stage": stage,
        "shape": list(shape),
        "dtype": dtype,
        "actual_dtype": actual_dtype,
        "size": size,
        "mean": float(np.asarray(mean)),
        "std": float(np.asarray(std)),
        "min": float(np.asarray(min_value)),
        "max": float(np.asarray(max_value)),
        "max_abs": float(np.asarray(max_abs)),
        "sample": np.asarray(sample, dtype=np.float32).tolist(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _TRACE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _write_tensor_dump(stage: str, value: Any) -> None:
    dump_dir = os.getenv(TRACE_DUMP_DIR_ENV)
    if not dump_dir:
        return

    import numpy as np

    label = os.getenv(TRACE_LABEL_ENV, "")
    name = f"{_safe_name(label)}.{_safe_name(stage)}.npy"
    path = os.path.join(dump_dir, name)
    os.makedirs(dump_dir, exist_ok=True)
    with _TRACE_LOCK:
        np.save(path, np.asarray(value, dtype=np.float32))


def trace_jax_array(stage: str,
                    arr: Any,
                    *,
                    dtype: str = "",
                    actual_dtype: str = "") -> None:
    """Append summary stats for a JAX array."""
    if not _env_enabled():
        return

    try:
        import jax
        import jax.numpy as jnp

        arr = jnp.asarray(arr)
        actual_dtype_name = actual_dtype or str(arr.dtype)
        arr = arr.astype(jnp.float32)
        size = int(arr.size)
        sample_count = min(_sample_size(), size)
        if size == 0:
            return

        flat = jnp.reshape(arr, (-1, ))
        finite = jnp.isfinite(flat)
        safe_flat = jnp.where(finite, flat, 0.0)
        finite_count = jnp.maximum(jnp.sum(finite), 1)
        mean = jnp.sum(safe_flat) / finite_count
        centered = jnp.where(finite, flat - mean, 0.0)
        std = jnp.sqrt(jnp.sum(centered * centered) / finite_count)
        min_value = jnp.min(jnp.where(finite, flat, jnp.inf))
        max_value = jnp.max(jnp.where(finite, flat, -jnp.inf))
        max_abs = jnp.max(jnp.abs(safe_flat))
        sample = flat[:sample_count]
        dtype_name = dtype or str(arr.dtype)

        def _callback(mean: Any, std: Any, min_value: Any, max_value: Any,
                      max_abs: Any, sample: Any) -> None:
            _write_record(
                stage=stage,
                shape=tuple(int(dim) for dim in arr.shape),
                dtype=dtype_name,
                actual_dtype=actual_dtype_name,
                size=size,
                mean=mean,
                std=std,
                min_value=min_value,
                max_value=max_value,
                max_abs=max_abs,
                sample=sample,
            )

        jax.debug.callback(
            _callback,
            mean,
            std,
            min_value,
            max_value,
            max_abs,
            sample,
        )
        if _should_dump_stage(stage):
            jax.debug.callback(lambda value: _write_tensor_dump(stage, value),
                               arr)
    except Exception as exc:  # pragma: no cover - tracing must be best effort.
        logger.warning_once("Unable to trace array %s: %s", stage, exc)


def trace_torch_tensor(stage: str, value: Any) -> Any:
    """Append summary stats for a TorchAX tensor and return it unchanged."""
    if not _env_enabled():
        return value
    if value is None or not hasattr(value, "shape"):
        return value

    try:
        from torchax.interop import jax_view

        trace_jax_array(stage, jax_view(value), dtype=str(value.dtype))
    except Exception as exc:  # pragma: no cover - tracing must be best effort.
        logger.warning_once("Unable to trace tensor %s: %s", stage, exc)

    return value


def _trace_optional_torch_tensor(stage: str, value: Any) -> Any:
    if value is not None:
        trace_torch_tensor(stage, value)
    return value


def _gemma_rms_norm_reference(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float,
) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(x)
    if residual is not None:
        x = x + jnp.asarray(residual)
    x_f32 = x.astype(jnp.float32)
    weight = jnp.asarray(weight, dtype=jnp.float32) + 1.0
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    out = x_f32 * jax.lax.rsqrt(variance + eps) * weight
    return out, x, variance


def trace_gemma_rms_norm_reference(
    stage: str,
    norm_layer: Any,
    hidden_states: Any,
    residual: Any,
) -> None:
    """Trace a plain JAX Gemma RMSNorm reference without changing inference."""
    if not _env_enabled():
        return
    weight = getattr(norm_layer, "weight", None)
    if weight is None:
        return

    try:
        import jax.numpy as jnp
        from torchax.interop import jax_view

        weight_value = getattr(weight, "data", weight)
        x = jax_view(hidden_states)
        residual_value = None if residual is None else jax_view(residual)
        weight_arr = jax_view(weight_value)
        eps = float(getattr(norm_layer, "variance_epsilon"))
        out, input_sum, variance = _gemma_rms_norm_reference(
            x, residual_value, weight_arr, eps)

        trace_jax_array(f"{stage}.reference",
                        out.astype(input_sum.dtype),
                        dtype=str(hidden_states.dtype))
        trace_jax_array(f"{stage}.reference_f32", out, dtype=str(jnp.float32))
        trace_jax_array(f"{stage}.input_sum", input_sum,
                        dtype=str(hidden_states.dtype))
        trace_jax_array(f"{stage}.variance", variance, dtype=str(jnp.float32))
        trace_jax_array(f"{stage}.weight", weight_arr, dtype=str(weight.dtype))
    except Exception as exc:  # pragma: no cover - tracing must be best effort.
        logger.warning_once("Unable to trace Gemma RMSNorm reference %s: %s",
                            stage, exc)


def trace_post_attention_norm_inputs(prefix: str, hidden_states: Any,
                                     residual: Any) -> None:
    trace_torch_tensor(f"{prefix}.post_attention_norm.input.hidden_states",
                       hidden_states)
    trace_torch_tensor(f"{prefix}.post_attention_norm.input.residual",
                       residual)


def trace_post_attention_norm_raw_outputs(prefix: str, hidden_states: Any,
                                          residual: Any) -> None:
    trace_torch_tensor(f"{prefix}.post_attention_norm.raw.hidden_states",
                       hidden_states)
    trace_torch_tensor(f"{prefix}.post_attention_norm.raw.residual", residual)


def _trace_moe_router_select_experts(
    self: Any,
    hidden_states: Any,
    router_logits: Any,
    input_ids: Any = None,
) -> tuple[Any, Any]:
    prefix = getattr(self, _TRACE_PREFIX_ATTR, "moe.runner.router")
    trace_torch_tensor(f"{prefix}.input.hidden_states", hidden_states)
    trace_torch_tensor(f"{prefix}.input.router_logits", router_logits)
    original = getattr(self, "_tpu_inference_original_select_experts")
    topk_weights, topk_ids = original(
        hidden_states=hidden_states,
        router_logits=router_logits,
        input_ids=input_ids,
    )
    trace_torch_tensor(f"{prefix}.topk_weights", topk_weights)
    trace_torch_tensor(f"{prefix}.topk_ids", topk_ids)
    return topk_weights, topk_ids


def _trace_moe_runner_maybe_dispatch(
    self: Any,
    layer: Any,
    hidden_states: Any,
    router_logits: Any,
) -> tuple[Any, Any]:
    prefix = getattr(self, _TRACE_PREFIX_ATTR, "moe.runner")
    trace_torch_tensor(f"{prefix}.dispatch.input.hidden_states",
                       hidden_states)
    trace_torch_tensor(f"{prefix}.dispatch.input.router_logits",
                       router_logits)
    original = getattr(self, "_tpu_inference_original_maybe_dispatch")
    hidden_states, router_logits = original(layer, hidden_states,
                                            router_logits)
    trace_torch_tensor(f"{prefix}.dispatch.output.hidden_states",
                       hidden_states)
    trace_torch_tensor(f"{prefix}.dispatch.output.router_logits",
                       router_logits)
    return hidden_states, router_logits


def _trace_moe_runner_apply_quant_method(
    self: Any,
    layer: Any,
    hidden_states: Any,
    router_logits: Any,
    shared_experts_input: Any,
    input_ids: Any = None,
) -> tuple[Any, Any]:
    prefix = getattr(self, _TRACE_PREFIX_ATTR, "moe.runner")
    trace_torch_tensor(f"{prefix}.quant.input.hidden_states", hidden_states)
    trace_torch_tensor(f"{prefix}.quant.input.router_logits", router_logits)
    _trace_optional_torch_tensor(f"{prefix}.quant.input.shared_experts",
                                 shared_experts_input)

    original = getattr(self, "_tpu_inference_original_apply_quant_method")
    shared_output, fused_output = original(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        shared_experts_input=shared_experts_input,
        input_ids=input_ids,
    )

    _trace_optional_torch_tensor(f"{prefix}.quant.output.shared",
                                 shared_output)
    trace_torch_tensor(f"{prefix}.quant.output.fused", fused_output)
    return shared_output, fused_output


def _trace_moe_runner_maybe_combine(
    self: Any,
    shared_output: Any,
    hidden_states: Any,
) -> Any:
    prefix = getattr(self, _TRACE_PREFIX_ATTR, "moe.runner")
    _trace_optional_torch_tensor(f"{prefix}.combine.input.shared",
                                 shared_output)
    trace_torch_tensor(f"{prefix}.combine.input.hidden_states",
                       hidden_states)
    original = getattr(self, "_tpu_inference_original_maybe_combine")
    result = original(shared_output, hidden_states)

    if isinstance(result, tuple):
        output_shared, output_hidden_states = result
        _trace_optional_torch_tensor(f"{prefix}.combine.output.shared",
                                     output_shared)
        trace_torch_tensor(f"{prefix}.combine.output.hidden_states",
                           output_hidden_states)
    else:
        trace_torch_tensor(f"{prefix}.combine.output", result)
    return result


def _patch_moe_router_trace(router: Any, prefix: str) -> bool:
    setattr(router, _TRACE_PREFIX_ATTR, prefix)
    if getattr(router, _MOE_ROUTER_PATCHED_ATTR, False):
        return False
    if not hasattr(router, "select_experts"):
        return False

    setattr(router, "_tpu_inference_original_select_experts",
            router.select_experts)
    router.select_experts = types.MethodType(_trace_moe_router_select_experts,
                                             router)
    setattr(router, _MOE_ROUTER_PATCHED_ATTR, True)
    return True


def _patch_moe_runner_trace(runner: Any, prefix: str) -> bool:
    setattr(runner, _TRACE_PREFIX_ATTR, prefix)
    patched = False

    router = getattr(runner, "router", None)
    if router is not None:
        patched = _patch_moe_router_trace(router,
                                         f"{prefix}.router") or patched

    if getattr(runner, _MOE_RUNNER_PATCHED_ATTR, False):
        return patched

    patch_specs = [
        ("_maybe_dispatch", "_tpu_inference_original_maybe_dispatch",
         _trace_moe_runner_maybe_dispatch),
        ("_apply_quant_method", "_tpu_inference_original_apply_quant_method",
         _trace_moe_runner_apply_quant_method),
        ("_maybe_combine", "_tpu_inference_original_maybe_combine",
         _trace_moe_runner_maybe_combine),
    ]
    for method_name, original_name, replacement in patch_specs:
        if not hasattr(runner, method_name):
            continue
        setattr(runner, original_name, getattr(runner, method_name))
        setattr(runner, method_name, types.MethodType(replacement, runner))
        patched = True

    if patched:
        setattr(runner, _MOE_RUNNER_PATCHED_ATTR, True)
    return patched


def _trace_qwen3_sparse_moe_forward(self: Any,
                                    hidden_states: Any) -> Any:
    prefix = getattr(self, _TRACE_PREFIX_ATTR, "moe")
    trace_torch_tensor(f"{prefix}.input", hidden_states)

    orig_shape = hidden_states.shape
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    trace_torch_tensor(f"{prefix}.view_input", hidden_states)

    if getattr(self, "is_sequence_parallel", False):
        from vllm.model_executor.models.utils import sequence_parallel_chunk

        hidden_states = sequence_parallel_chunk(hidden_states)
        trace_torch_tensor(f"{prefix}.sequence_parallel.input",
                           hidden_states)

    if self.experts.is_internal_router:
        trace_torch_tensor(f"{prefix}.internal_router.input", hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
        )
    else:
        router_logits, _ = self.gate(hidden_states)
        trace_torch_tensor(f"{prefix}.router_logits", router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

    trace_torch_tensor(f"{prefix}.experts.output", final_hidden_states)

    if getattr(self, "is_sequence_parallel", False):
        from vllm.distributed import tensor_model_parallel_all_gather

        final_hidden_states = tensor_model_parallel_all_gather(
            final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
        trace_torch_tensor(f"{prefix}.sequence_parallel.output",
                           final_hidden_states)

    final_hidden_states = final_hidden_states.view(orig_shape)
    trace_torch_tensor(f"{prefix}.output", final_hidden_states)
    return final_hidden_states


def _trace_qwen_moe_mlp_forward(self: Any, x: Any) -> Any:
    prefix = getattr(self, _TRACE_PREFIX_ATTR, "moe.shared")
    trace_torch_tensor(f"{prefix}.input", x)

    original_x = x
    gate_up, _ = self.gate_up_proj(x)
    trace_torch_tensor(f"{prefix}.gate_up", gate_up)

    x = self.act_fn(gate_up)
    trace_torch_tensor(f"{prefix}.act", x)
    if getattr(self, _QWEN_MLP_FORCE_BARRIER_ATTR, False):
        import jax
        from torchax.interop import jax_view, torch_view

        x = torch_view(jax.lax.optimization_barrier(jax_view(x)))
        trace_torch_tensor(f"{prefix}.act_barrier", x)

    x, _ = self.down_proj(x)
    trace_torch_tensor(f"{prefix}.down", x)

    expert_gate = getattr(self, "expert_gate", None)
    if expert_gate is not None:
        import torch.nn.functional as F

        gate_logits = expert_gate(original_x)[0]
        trace_torch_tensor(f"{prefix}.expert_gate.logits", gate_logits)
        gate = F.sigmoid(gate_logits)
        trace_torch_tensor(f"{prefix}.expert_gate.sigmoid", gate)
        x = gate * x

    trace_torch_tensor(f"{prefix}.output", x)
    return x


def _is_qwen_moe_mlp(module: Any) -> bool:
    return (hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")
            and hasattr(module, "act_fn"))


def _patch_qwen_moe_mlp_trace(module: Any, prefix: str) -> bool:
    if not _is_qwen_moe_mlp(module):
        return False
    setattr(module, _TRACE_PREFIX_ATTR, prefix)
    if getattr(module, _PATCHED_ATTR, False):
        return False

    module.forward = types.MethodType(_trace_qwen_moe_mlp_forward, module)
    setattr(module, _PATCHED_ATTR, True)
    return True


def _is_qwen3_sparse_moe_block(module: Any) -> bool:
    return (hasattr(module, "gate") and hasattr(module, "experts") and
            hasattr(module, "is_sequence_parallel") and
            hasattr(getattr(module, "experts"), "is_internal_router"))


def _patch_qwen3_sparse_moe_trace(module: Any, prefix: str) -> bool:
    if not _is_qwen3_sparse_moe_block(module):
        return False

    setattr(module, _TRACE_PREFIX_ATTR, prefix)
    patched = False

    runner = getattr(getattr(module, "experts", None), "runner", None)
    if runner is not None:
        patched = _patch_moe_runner_trace(runner,
                                         f"{prefix}.runner") or patched

    shared_expert = getattr(module, "shared_expert", None)
    if shared_expert is not None:
        patched = _patch_qwen_moe_mlp_trace(
            shared_expert, f"{prefix}.shared") or patched

    if not getattr(module, _PATCHED_ATTR, False):
        module.forward = types.MethodType(_trace_qwen3_sparse_moe_forward,
                                          module)
        setattr(module, _PATCHED_ATTR, True)
        patched = True

    return patched


def _trace_decoder_layer_forward(
    self: Any,
    hidden_states: Any,
    residual: Any,
    positions: Any = None,
    **kwargs: object,
) -> tuple[Any, Any]:
    import torch

    layer_idx = int(getattr(self, "layer_idx", -1))
    layer_type = getattr(self, "layer_type", "unknown")
    prefix = f"layer.{layer_idx:02d}.{layer_type}"

    trace_torch_tensor(f"{prefix}.input.hidden_states", hidden_states)
    trace_torch_tensor(f"{prefix}.input.residual", residual)

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states,
                                                       residual)
    if layer_type == "linear_attention":
        try:
            from tpu_inference.models.vllm.experimental.qwen3_decoder_patcher import (
                maybe_cast_attention_output, maybe_cast_residual)

            hidden_states = maybe_cast_attention_output(self, hidden_states)
            residual = maybe_cast_residual(self, residual)
        except Exception as exc:  # pragma: no cover - tracing must be best effort.
            logger.warning_once("Unable to cast traced input norm output: %s",
                                exc)

    trace_torch_tensor(f"{prefix}.input_norm.hidden_states", hidden_states)
    trace_torch_tensor(f"{prefix}.input_norm.residual", residual)

    self_attention_output = torch.empty_like(hidden_states)
    if layer_type == "linear_attention":
        self.linear_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
        )
    elif layer_type == "full_attention":
        self.self_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
            positions=positions,
        )
    else:
        raise ValueError("Invalid layer_type")

    hidden_states = self_attention_output
    if layer_type == "linear_attention":
        try:
            from tpu_inference.models.vllm.experimental.qwen3_decoder_patcher import (
                maybe_cast_attention_output, maybe_cast_residual)

            hidden_states = maybe_cast_attention_output(self, hidden_states)
            residual = maybe_cast_residual(self, residual)
        except Exception as exc:  # pragma: no cover - tracing must be best effort.
            logger.warning_once("Unable to cast traced attention output: %s",
                                exc)
    trace_torch_tensor(f"{prefix}.attention.output", hidden_states)

    if getattr(self, "layer_scale", False):
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype) + 1)
        trace_torch_tensor(f"{prefix}.attention_scaled.output",
                           hidden_states)

    trace_post_attention_norm_inputs(prefix, hidden_states, residual)
    trace_gemma_rms_norm_reference(
        f"{prefix}.post_attention_norm",
        self.post_attention_layernorm,
        hidden_states,
        residual,
    )
    hidden_states, residual = self.post_attention_layernorm(hidden_states,
                                                            residual)
    trace_post_attention_norm_raw_outputs(prefix, hidden_states, residual)
    if layer_type == "linear_attention":
        try:
            from tpu_inference.models.vllm.experimental.qwen3_decoder_patcher import (
                maybe_cast_attention_output)

            hidden_states = maybe_cast_attention_output(self, hidden_states)
        except Exception as exc:  # pragma: no cover - tracing must be best effort.
            logger.warning_once("Unable to cast traced post norm output: %s",
                                exc)
    trace_torch_tensor(f"{prefix}.post_attention_norm.hidden_states",
                       hidden_states)
    trace_torch_tensor(f"{prefix}.post_attention_norm.residual", residual)

    trace_torch_tensor(f"{prefix}.mlp.input", hidden_states)
    hidden_states = self.mlp(hidden_states)
    trace_torch_tensor(f"{prefix}.mlp.output", hidden_states)

    if getattr(self, "layer_scale", False):
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
        trace_torch_tensor(f"{prefix}.mlp_scaled.output", hidden_states)

    trace_torch_tensor(f"{prefix}.output.hidden_states", hidden_states)
    trace_torch_tensor(f"{prefix}.output.residual", residual)
    return hidden_states, residual


def _is_qwen3_decoder_layer(module: Any) -> bool:
    return (hasattr(module, "layer_idx") and hasattr(module, "layer_type")
            and hasattr(module, "input_layernorm")
            and hasattr(module, "post_attention_layernorm")
            and hasattr(module, "mlp") and
            (hasattr(module, "linear_attn") or hasattr(module, "self_attn")))


def maybe_apply_qwen3_layer_trace(vllm_model: Any) -> None:
    if not _env_enabled():
        return

    patched_layers = 0
    patched_moe = 0
    for module in vllm_model.modules():
        if not _is_qwen3_decoder_layer(module):
            continue

        layer_idx = int(getattr(module, "layer_idx", -1))
        layer_type = getattr(module, "layer_type", "unknown")
        prefix = f"layer.{layer_idx:02d}.{layer_type}"

        if _patch_qwen3_sparse_moe_trace(getattr(module, "mlp", None),
                                         f"{prefix}.moe"):
            patched_moe += 1

        if getattr(module, _PATCHED_ATTR, False):
            continue
        module.forward = types.MethodType(_trace_decoder_layer_forward, module)
        setattr(module, _PATCHED_ATTR, True)
        patched_layers += 1

    if patched_layers:
        logger.info("Installed Qwen3.5 layer trace hooks for %d layer(s).",
                    patched_layers)
    if patched_moe:
        logger.info("Installed Qwen3.5 MoE trace hooks for %d module(s).",
                    patched_moe)
