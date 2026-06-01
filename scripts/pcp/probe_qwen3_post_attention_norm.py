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
"""Probe Qwen3.5 post-attention GemmaRMSNorm under PCP meshes.

This narrows full-engine PCP correctness debugging to the boundary immediately
after linear attention: the same post_attention_layernorm inputs are replayed
under PCP=1 and PCP=N, then raw and dtype-cast outputs are compared.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torchax
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from scripts.pcp.probe_qwen3_in_proj_qkvz import (
    _ensure_vllm_single_rank_parallel,
    _make_probe_vllm_config,
    build_pcp_mesh,
    diff_stats,
    load_tensor_npy,
    mesh_summary,
    parse_torch_dtype,
)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import (
    _cast_torchax_tensor_to_torch_dtype,
)

P = PartitionSpec
_TORCHAX_GLOBAL_ENV: Any | None = None


def _set_default_env() -> None:
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")


def _ensure_torchax_global_env() -> None:
    global _TORCHAX_GLOBAL_ENV
    if _TORCHAX_GLOBAL_ENV is None:
        _TORCHAX_GLOBAL_ENV = torchax.default_env()
        _TORCHAX_GLOBAL_ENV.enable_torch_modes()


def _load_safetensor_tensor(model_dir: Path, index: dict[str, Any],
                            key: str) -> torch.Tensor:
    from safetensors.torch import safe_open

    filename = index["weight_map"][key]
    with safe_open(model_dir / filename, framework="pt",
                   device="cpu") as handle:
        return handle.get_tensor(key)


def load_qwen3_post_attention_norm_weight(model_dir: str | Path,
                                          layer_index: int,
                                          dtype: torch.dtype) -> torch.Tensor:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    suffix = f"layers.{layer_index}.post_attention_layernorm.weight"
    key = next((name for name in index["weight_map"] if name.endswith(suffix)),
               None)
    if key is None:
        raise KeyError(
            f"Could not find post_attention_layernorm weight for layer "
            f"{layer_index} in {index_path}.")
    return _load_safetensor_tensor(model_path, index, key).to(dtype)


def _to_torchax_tensor(tensor: torch.Tensor, mesh: jax.sharding.Mesh,
                       spec: PartitionSpec) -> torch.Tensor:
    jax_tensor = t2j(tensor, use_dlpack=False)
    jax_tensor = jax.device_put(jax_tensor, NamedSharding(mesh, spec))
    return torch_view(jax_tensor)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = jax_view(tensor).astype(jnp.float32)
    array.block_until_ready()
    return np.asarray(jax.device_get(array))


def _torch_dtype_to_jax(dtype: torch.dtype) -> jnp.dtype:
    if dtype is torch.bfloat16:
        return jnp.bfloat16
    if dtype is torch.float16:
        return jnp.float16
    if dtype is torch.float32:
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _torch_tensor_to_jax(tensor: torch.Tensor, dtype: torch.dtype) -> jax.Array:
    return jnp.asarray(tensor.float().cpu().numpy()).astype(
        _torch_dtype_to_jax(dtype))


def _cast_jax_array_to_torch_dtype(array: jax.Array,
                                   dtype: torch.dtype) -> jax.Array:
    return jax.lax.optimization_barrier(
        array.astype(jnp.float32).astype(_torch_dtype_to_jax(dtype)))


def run_post_attention_norm_reference(
    *,
    mesh: jax.sharding.Mesh,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    token_sharding = NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None))
    weight_sharding = NamedSharding(mesh, P(None))

    with jax.set_mesh(mesh):
        hidden_device = jax.device_put(
            _torch_tensor_to_jax(hidden_states, dtype), token_sharding)
        residual_device = jax.device_put(_torch_tensor_to_jax(residual, dtype),
                                         token_sharding)
        weight_device = jax.device_put(_torch_tensor_to_jax(weight, dtype),
                                       weight_sharding)

        raw_residual = hidden_device + residual_device
        residual_f32 = raw_residual.astype(jnp.float32)
        variance = jnp.mean(residual_f32 * residual_f32,
                            axis=-1,
                            keepdims=True)
        raw_hidden = (residual_f32 * jax.lax.rsqrt(variance + eps) *
                      (weight_device.astype(jnp.float32) + 1.0))
        raw_hidden = raw_hidden.astype(raw_residual.dtype)
        cast_hidden = _cast_jax_array_to_torch_dtype(raw_hidden, dtype)
        cast_residual = _cast_jax_array_to_torch_dtype(raw_residual, dtype)

        return {
            "input_hidden_states":
            np.asarray(jax.device_get(hidden_device.astype(jnp.float32))),
            "input_residual":
            np.asarray(jax.device_get(residual_device.astype(jnp.float32))),
            "raw_hidden_states":
            np.asarray(jax.device_get(raw_hidden.astype(jnp.float32))),
            "raw_residual":
            np.asarray(jax.device_get(raw_residual.astype(jnp.float32))),
            "cast_hidden_states":
            np.asarray(jax.device_get(cast_hidden.astype(jnp.float32))),
            "cast_residual":
            np.asarray(jax.device_get(cast_residual.astype(jnp.float32))),
        }


def run_post_attention_norm_vllm(
    *,
    mesh: jax.sharding.Mesh,
    pcp_size: int,
    model: str,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    _ensure_torchax_global_env()
    vllm_config = _make_probe_vllm_config(
        model=model,
        dtype=dtype,
        max_num_batched_tokens=int(hidden_states.shape[0]),
        pcp_size=pcp_size,
    )

    with set_current_vllm_config(vllm_config):
        norm = GemmaRMSNorm(int(hidden_states.shape[-1]), eps=eps)

    token_spec = P(ShardingAxisName.ATTN_DATA, None)
    hidden_device = _to_torchax_tensor(hidden_states, mesh, token_spec)
    residual_device = _to_torchax_tensor(residual, mesh, token_spec)
    weight_device = _to_torchax_tensor(weight, mesh, P(None))

    with torchax.default_env(), jax.set_mesh(mesh), set_current_vllm_config(
            vllm_config):
        norm.weight = nn.Parameter(weight_device, requires_grad=False)
        raw_hidden, raw_residual = norm(hidden_device, residual_device)
        cast_hidden = _cast_torchax_tensor_to_torch_dtype(
            raw_hidden, raw_hidden.dtype)
        cast_residual = _cast_torchax_tensor_to_torch_dtype(
            raw_residual, raw_residual.dtype)

        return {
            "input_hidden_states": _to_numpy(hidden_device),
            "input_residual": _to_numpy(residual_device),
            "raw_hidden_states": _to_numpy(raw_hidden),
            "raw_residual": _to_numpy(raw_residual),
            "cast_hidden_states": _to_numpy(cast_hidden),
            "cast_residual": _to_numpy(cast_residual),
        }


def run_post_attention_norm(
    *,
    impl: str,
    mesh: jax.sharding.Mesh,
    pcp_size: int,
    model: str,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    if impl == "reference":
        return run_post_attention_norm_reference(
            mesh=mesh,
            hidden_states=hidden_states,
            residual=residual,
            weight=weight,
            eps=eps,
            dtype=dtype,
        )
    if impl == "vllm":
        return run_post_attention_norm_vllm(
            mesh=mesh,
            pcp_size=pcp_size,
            model=model,
            hidden_states=hidden_states,
            residual=residual,
            weight=weight,
            eps=eps,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported --impl: {impl}")


def compare_norm_outputs(baseline: dict[str, np.ndarray],
                         candidate: dict[str, np.ndarray],
                         *,
                         topk: int) -> dict[str, Any]:
    return {
        "inputs": {
            "hidden_states":
            diff_stats(baseline["input_hidden_states"],
                       candidate["input_hidden_states"],
                       topk=topk),
            "residual":
            diff_stats(baseline["input_residual"],
                       candidate["input_residual"],
                       topk=topk),
        },
        "raw_outputs": {
            "hidden_states":
            diff_stats(baseline["raw_hidden_states"],
                       candidate["raw_hidden_states"],
                       topk=topk),
            "residual":
            diff_stats(baseline["raw_residual"],
                       candidate["raw_residual"],
                       topk=topk),
        },
        "as_current_engine": {
            "hidden_states":
            diff_stats(baseline["raw_hidden_states"],
                       candidate["cast_hidden_states"],
                       topk=topk),
            "residual":
            diff_stats(baseline["raw_residual"],
                       candidate["cast_residual"],
                       topk=topk),
        },
        "both_cast": {
            "hidden_states":
            diff_stats(baseline["cast_hidden_states"],
                       candidate["cast_hidden_states"],
                       topk=topk),
            "residual":
            diff_stats(baseline["cast_residual"],
                       candidate["cast_residual"],
                       topk=topk),
        },
        "cast_effect_baseline": {
            "hidden_states":
            diff_stats(baseline["raw_hidden_states"],
                       baseline["cast_hidden_states"],
                       topk=topk),
            "residual":
            diff_stats(baseline["raw_residual"],
                       baseline["cast_residual"],
                       topk=topk),
        },
        "cast_effect_candidate": {
            "hidden_states":
            diff_stats(candidate["raw_hidden_states"],
                       candidate["cast_hidden_states"],
                       topk=topk),
            "residual":
            diff_stats(candidate["raw_residual"],
                       candidate["cast_residual"],
                       topk=topk),
        },
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    _set_default_env()
    dtype = parse_torch_dtype(args.dtype)
    hidden_states = load_tensor_npy(args.hidden_states_npy, dtype)
    residual = load_tensor_npy(args.residual_npy, dtype)
    if hidden_states.shape != residual.shape:
        raise ValueError(f"Input shape mismatch: {tuple(hidden_states.shape)} "
                         f"vs {tuple(residual.shape)}.")

    if args.weight_npy:
        weight = load_tensor_npy(args.weight_npy, dtype).reshape(-1)
    else:
        if not args.model_dir:
            raise ValueError("--model-dir is required when --weight-npy is not "
                             "provided.")
        weight = load_qwen3_post_attention_norm_weight(
            args.model_dir, args.layer_index, dtype)
    if weight.ndim != 1 or weight.shape[0] != hidden_states.shape[-1]:
        raise ValueError(
            f"Weight shape {tuple(weight.shape)} does not match hidden size "
            f"{hidden_states.shape[-1]}.")

    init_config = _make_probe_vllm_config(
        model=args.model,
        dtype=dtype,
        max_num_batched_tokens=int(hidden_states.shape[0]),
        pcp_size=1,
    )
    _ensure_vllm_single_rank_parallel(init_config)

    baseline_mesh = build_pcp_mesh(
        expert_parallel_size=args.expert_parallel_size,
        pcp_size=args.baseline_pcp_size,
    )
    candidate_mesh = build_pcp_mesh(
        expert_parallel_size=args.expert_parallel_size,
        pcp_size=args.pcp_size,
    )

    baseline = run_post_attention_norm(
        impl=args.impl,
        mesh=baseline_mesh,
        pcp_size=args.baseline_pcp_size,
        model=args.model,
        hidden_states=hidden_states,
        residual=residual,
        weight=weight,
        eps=args.eps,
        dtype=dtype,
    )
    candidate = run_post_attention_norm(
        impl=args.impl,
        mesh=candidate_mesh,
        pcp_size=args.pcp_size,
        model=args.model,
        hidden_states=hidden_states,
        residual=residual,
        weight=weight,
        eps=args.eps,
        dtype=dtype,
    )

    payload = {
        "config": {
            "model": args.model,
            "impl": args.impl,
            "model_dir": args.model_dir,
            "layer_index": args.layer_index,
            "dtype": str(dtype),
            "eps": args.eps,
            "baseline_pcp_size": args.baseline_pcp_size,
            "pcp_size": args.pcp_size,
            "expert_parallel_size": args.expert_parallel_size,
            "hidden_states_npy": args.hidden_states_npy,
            "residual_npy": args.residual_npy,
            "weight_npy": args.weight_npy,
        },
        "data": {
            "hidden_states_shape": list(hidden_states.shape),
            "residual_shape": list(residual.shape),
            "weight_shape": list(weight.shape),
        },
        "meshes": {
            f"pcp{args.baseline_pcp_size}": mesh_summary(baseline_mesh),
            f"pcp{args.pcp_size}": mesh_summary(candidate_mesh),
        },
        "diffs": compare_norm_outputs(baseline, candidate, topk=args.topk),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Qwen3 post-attention GemmaRMSNorm inputs under "
        "PCP=1 and PCP=N.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--model-dir")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--hidden-states-npy", required=True)
    parser.add_argument("--residual-npy", required=True)
    parser.add_argument("--weight-npy")
    parser.add_argument("--impl", choices=("reference", "vllm"),
                        default="reference")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--baseline-pcp-size", type=int, default=1)
    parser.add_argument("--pcp-size", type=int, default=2)
    parser.add_argument("--expert-parallel-size", type=int, default=2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_probe(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
