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
"""Probe Qwen3.5 GDN ``in_proj_qkvz`` under PCP and non-PCP meshes.

This is intentionally narrower than the full vLLM engine smoke test: it builds
only the ``MergedColumnParallelLinear`` used by Qwen3.5 GDN ``in_proj_qkvz`` and
runs the same synthetic hidden states and weights on two JAX meshes.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    model_parallel_is_initialized,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

from tpu_inference.layers.common.process_weights.linear_weights import (
    get_model_matmul_fusion_assignment,
)
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedLinearMethod,
)

P = PartitionSpec

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_OUTPUT_SIZES = [2048, 2048, 4096, 4096]


@dataclass
class _ProbePassConfig:
    enable_sp: bool = False


@dataclass
class _ProbeCompilationConfig:
    pass_config: _ProbePassConfig = field(default_factory=_ProbePassConfig)
    custom_ops: list[str] = field(default_factory=lambda: ["all"])
    enabled_custom_ops: set[str] = field(default_factory=set)
    disabled_custom_ops: set[str] = field(default_factory=set)


@dataclass
class _ProbeModelConfig:
    model: str
    dtype: torch.dtype
    quantization: str | None = None
    runner_type: str = "generate"
    is_moe: bool = False


@dataclass
class _ProbeSchedulerConfig:
    max_num_batched_tokens: int


@dataclass
class _ProbeParallelConfig:
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_elastic_ep: bool = False
    enable_eplb: bool = False
    distributed_executor_backend: str | None = None
    nnodes: int = 1
    nnodes_within_dp: int = 1
    cpu_distributed_timeout_seconds: int | None = None
    prefill_context_parallel_size: int = 1


@dataclass
class _ProbeVllmConfig:
    model_config: _ProbeModelConfig
    scheduler_config: _ProbeSchedulerConfig
    parallel_config: _ProbeParallelConfig
    compilation_config: _ProbeCompilationConfig = field(
        default_factory=_ProbeCompilationConfig)


def _set_default_env() -> None:
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")


def parse_output_sizes(value: str) -> list[int]:
    output_sizes = [int(item) for item in value.split(",") if item.strip()]
    if not output_sizes or any(size <= 0 for size in output_sizes):
        raise ValueError("--output-sizes must be a comma-separated list of "
                         "positive integers.")
    return output_sizes


def parse_torch_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def build_pcp_mesh(*, expert_parallel_size: int, pcp_size: int) -> Mesh:
    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES

    if expert_parallel_size <= 0 or pcp_size <= 0:
        raise ValueError("expert_parallel_size and pcp_size must be positive.")

    required_devices = expert_parallel_size * pcp_size
    devices = sorted(jax.devices(), key=lambda device: device.id)
    if len(devices) < required_devices:
        raise RuntimeError(
            f"Need at least {required_devices} JAX devices for "
            f"expert_parallel_size={expert_parallel_size}, pcp_size={pcp_size}; "
            f"found {len(devices)}.")

    mesh_shape = (1, 1, 1, expert_parallel_size, 1, 1, pcp_size)
    axis_types = (AxisType.Auto, ) * len(MESH_AXIS_NAMES)
    return jax.make_mesh(mesh_shape,
                         MESH_AXIS_NAMES,
                         axis_types,
                         devices=devices[:required_devices])


def mesh_summary(mesh: Mesh) -> dict[str, Any]:
    return {
        "axis_names": list(mesh.axis_names),
        "shape": {name: int(size)
                  for name, size in mesh.shape.items()},
        "devices": [str(device) for device in np.asarray(mesh.devices).flat],
    }


def build_synthetic_tensors(
    *,
    tokens: int,
    hidden_size: int,
    output_sizes: list[int],
    dtype: torch.dtype,
    seed: int,
    input_scale: float,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    hidden_states = (
        torch.randn((tokens, hidden_size),
                    dtype=torch.float32,
                    generator=generator) * input_scale).to(dtype)
    weight = (
        torch.randn((sum(output_sizes), hidden_size),
                    dtype=torch.float32,
                    generator=generator) * weight_scale).to(dtype)
    return hidden_states, weight


def load_tensor_npy(path: str | Path, dtype: torch.dtype) -> torch.Tensor:
    array = np.load(path)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path} does not contain a numeric numpy array.")
    return torch.from_numpy(array.astype(np.float32, copy=False)).to(dtype)


def _load_safetensor_tensor(model_dir: Path, index: dict[str, Any],
                            key: str) -> torch.Tensor:
    from safetensors.torch import safe_open

    filename = index["weight_map"][key]
    with safe_open(model_dir / filename, framework="pt",
                   device="cpu") as handle:
        return handle.get_tensor(key)


def _find_layer_weight_keys(model_dir: Path,
                            layer_index: int) -> tuple[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    suffix_qkv = f"layers.{layer_index}.linear_attn.in_proj_qkv.weight"
    suffix_z = f"layers.{layer_index}.linear_attn.in_proj_z.weight"
    qkv_key = next((key for key in weight_map if key.endswith(suffix_qkv)),
                   None)
    z_key = next((key for key in weight_map if key.endswith(suffix_z)), None)
    if qkv_key is None or z_key is None:
        raise KeyError(
            f"Could not find Qwen3 in_proj_qkv/in_proj_z weights for "
            f"layer {layer_index} in {index_path}.")
    return qkv_key, z_key


def load_qwen3_in_proj_qkvz_weight(model_dir: str | Path, layer_index: int,
                                   dtype: torch.dtype) -> torch.Tensor:
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    qkv_key, z_key = _find_layer_weight_keys(model_path, layer_index)
    qkv_weight = _load_safetensor_tensor(model_path, index, qkv_key)
    z_weight = _load_safetensor_tensor(model_path, index, z_key)
    return torch.cat([qkv_weight, z_weight], dim=0).to(dtype)


def load_or_build_tensors(
    args: argparse.Namespace,
    *,
    output_sizes: list[int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source: dict[str, Any] = {}
    if args.hidden_states_npy:
        hidden_states = load_tensor_npy(args.hidden_states_npy, dtype)
        source["hidden_states"] = args.hidden_states_npy
    else:
        hidden_states = None

    if args.weight_npy:
        weight = load_tensor_npy(args.weight_npy, dtype)
        source["weight"] = args.weight_npy
    elif args.model_dir:
        weight = load_qwen3_in_proj_qkvz_weight(args.model_dir,
                                                args.layer_index, dtype)
        source["weight"] = {
            "model_dir": args.model_dir,
            "layer_index": args.layer_index,
        }
    else:
        weight = None

    if hidden_states is None or weight is None:
        synthetic_hidden, synthetic_weight = build_synthetic_tensors(
            tokens=args.tokens,
            hidden_size=args.hidden_size,
            output_sizes=output_sizes,
            dtype=dtype,
            seed=args.seed,
            input_scale=args.input_scale,
            weight_scale=args.weight_scale,
        )
        if hidden_states is None:
            hidden_states = synthetic_hidden
            source["hidden_states"] = "synthetic"
        if weight is None:
            weight = synthetic_weight
            source["weight"] = "synthetic"

    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden_states and weight must both be 2D tensors.")
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input dim mismatch: hidden_states={tuple(hidden_states.shape)}, "
            f"weight={tuple(weight.shape)}.")
    if weight.shape[0] != sum(output_sizes):
        raise ValueError(
            f"Weight output dim {weight.shape[0]} does not match "
            f"output_sizes={output_sizes}.")
    return hidden_states, weight, source


def _make_probe_vllm_config(
    *,
    model: str,
    dtype: torch.dtype,
    max_num_batched_tokens: int,
    pcp_size: int,
) -> _ProbeVllmConfig:
    return _ProbeVllmConfig(
        model_config=_ProbeModelConfig(model=model, dtype=dtype),
        scheduler_config=_ProbeSchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens),
        parallel_config=_ProbeParallelConfig(
            tensor_parallel_size=1,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def _ensure_vllm_single_rank_parallel(vllm_config: _ProbeVllmConfig) -> None:
    with set_current_vllm_config(vllm_config):
        if model_parallel_is_initialized():
            ensure_model_parallel_initialized(1, 1)
            return

        temp_fd, temp_file = tempfile.mkstemp()
        os.close(temp_fd)
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)


def _resolve_fuse_matmuls(
    *,
    requested: str,
    model: str,
    max_num_batched_tokens: int,
) -> bool:
    if requested == "true":
        return True
    if requested == "false":
        return False
    return bool(
        get_model_matmul_fusion_assignment(
            model,
            max_num_batched_tokens,
            1,
            "MergedColumnParallelLinear",
        ))


def run_in_proj_qkvz_linear(
    *,
    mesh: Mesh,
    pcp_size: int,
    model: str,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output_sizes: list[int],
    dtype: torch.dtype,
    fuse_matmuls: bool,
) -> np.ndarray:
    vllm_config = _make_probe_vllm_config(
        model=model,
        dtype=dtype,
        max_num_batched_tokens=int(hidden_states.shape[0]),
        pcp_size=pcp_size,
    )
    quant_config = VllmUnquantizedConfig()
    quant_config.set_configs(vllm_config, mesh)

    with set_current_vllm_config(vllm_config):
        layer = MergedColumnParallelLinear(
            input_size=int(hidden_states.shape[-1]),
            output_sizes=output_sizes,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix="model.layers.0.linear_attn.in_proj_qkvz",
        )
        assert isinstance(layer.quant_method, VllmUnquantizedLinearMethod)
        layer.quant_method.linear_config.fuse_matmuls = fuse_matmuls

    layer.weight.data = weight.clone()
    jax_input = torch_view(t2j(hidden_states, use_dlpack=False))
    jax_input.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, None)))

    with torchax.default_env(), jax.set_mesh(mesh), set_current_vllm_config(
            vllm_config):
        layer.quant_method.process_weights_after_loading(layer)
        output = layer(jax_input)
        output_jax = jax_view(output).astype(jnp.float32)
        output_jax.block_until_ready()
        return np.asarray(jax.device_get(output_jax))


def split_qkvz(output: np.ndarray, output_sizes: list[int],
               head_v_dim: int) -> dict[str, np.ndarray]:
    if len(output_sizes) != 4:
        return {"qkvz": output}
    q_size, k_size, v_size, z_size = output_sizes
    if output.shape[-1] != sum(output_sizes):
        raise ValueError(
            f"Output shape {output.shape} does not match output_sizes "
            f"{output_sizes}.")
    q_end = q_size
    k_end = q_end + k_size
    v_end = k_end + v_size
    q = output[:, :q_end]
    k = output[:, q_end:k_end]
    v = output[:, k_end:v_end]
    z_flat = output[:, v_end:v_end + z_size]
    result = {
        "qkvz": output,
        "mixed_qkv": output[:, :v_end],
        "q": q,
        "k": k,
        "v": v,
        "z_flat": z_flat,
    }
    if head_v_dim > 0 and z_size % head_v_dim == 0:
        result["z"] = z_flat.reshape(z_flat.shape[0], -1, head_v_dim)
    return result


def diff_stats(baseline: np.ndarray,
               candidate: np.ndarray,
               *,
               topk: int = 5) -> dict[str, Any]:
    if baseline.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {baseline.shape} vs "
                         f"{candidate.shape}.")

    baseline_f32 = baseline.astype(np.float32)
    candidate_f32 = candidate.astype(np.float32)
    diff = np.abs(candidate_f32 - baseline_f32)
    nonzero = np.flatnonzero(diff.reshape(-1) != 0)
    if diff.size == 0:
        max_abs = 0.0
        max_index: tuple[int, ...] = ()
    else:
        max_flat = int(np.argmax(diff))
        max_abs = float(diff.reshape(-1)[max_flat])
        max_index = tuple(int(i) for i in np.unravel_index(max_flat,
                                                           diff.shape))

    top: list[dict[str, Any]] = []
    if nonzero.size:
        order = nonzero[np.argsort(diff.reshape(-1)[nonzero])[::-1]]
        for flat_index in order[:topk]:
            index = tuple(
                int(i) for i in np.unravel_index(int(flat_index), diff.shape))
            top.append({
                "index": list(index),
                "baseline": float(baseline_f32[index]),
                "candidate": float(candidate_f32[index]),
                "abs_diff": float(diff[index]),
            })

    return {
        "shape": list(baseline.shape),
        "numel": int(diff.size),
        "nonzero_count": int(nonzero.size),
        "max_abs": max_abs,
        "max_index": list(max_index),
        "top": top,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    _set_default_env()

    output_sizes = parse_output_sizes(args.output_sizes)
    dtype = parse_torch_dtype(args.dtype)
    init_vllm_config = _make_probe_vllm_config(
        model=args.model,
        dtype=dtype,
        max_num_batched_tokens=args.tokens,
        pcp_size=1,
    )
    _ensure_vllm_single_rank_parallel(init_vllm_config)

    fuse_matmuls = _resolve_fuse_matmuls(
        requested=args.fuse_matmuls,
        model=args.model,
        max_num_batched_tokens=args.tokens,
    )
    hidden_states, weight, data_source = load_or_build_tensors(
        args, output_sizes=output_sizes, dtype=dtype)

    baseline_mesh = build_pcp_mesh(
        expert_parallel_size=args.expert_parallel_size,
        pcp_size=args.baseline_pcp_size,
    )
    candidate_mesh = build_pcp_mesh(
        expert_parallel_size=args.expert_parallel_size,
        pcp_size=args.pcp_size,
    )

    baseline = run_in_proj_qkvz_linear(
        mesh=baseline_mesh,
        pcp_size=args.baseline_pcp_size,
        model=args.model,
        hidden_states=hidden_states,
        weight=weight,
        output_sizes=output_sizes,
        dtype=dtype,
        fuse_matmuls=fuse_matmuls,
    )
    candidate = run_in_proj_qkvz_linear(
        mesh=candidate_mesh,
        pcp_size=args.pcp_size,
        model=args.model,
        hidden_states=hidden_states,
        weight=weight,
        output_sizes=output_sizes,
        dtype=dtype,
        fuse_matmuls=fuse_matmuls,
    )

    baseline_parts = split_qkvz(baseline, output_sizes, args.head_v_dim)
    candidate_parts = split_qkvz(candidate, output_sizes, args.head_v_dim)
    stats = {
        name: diff_stats(baseline_parts[name],
                         candidate_parts[name],
                         topk=args.topk)
        for name in baseline_parts
    }

    return {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "tokens": args.tokens,
        "actual_tokens": int(hidden_states.shape[0]),
        "hidden_size": int(hidden_states.shape[1]),
        "output_sizes": output_sizes,
        "head_v_dim": args.head_v_dim,
        "fuse_matmuls": fuse_matmuls,
        "data_source": data_source,
        "baseline": {
            "pcp_size": args.baseline_pcp_size,
            "mesh": mesh_summary(baseline_mesh),
        },
        "candidate": {
            "pcp_size": args.pcp_size,
            "mesh": mesh_summary(candidate_mesh),
        },
        "stats": stats,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3.5 in_proj_qkvz PCP linear output.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--output-sizes",
                        default=",".join(str(size)
                                         for size in DEFAULT_OUTPUT_SIZES))
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--weight-scale", type=float, default=0.1)
    parser.add_argument("--hidden-states-npy")
    parser.add_argument("--weight-npy")
    parser.add_argument("--model-dir")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--expert-parallel-size", type=int, default=2)
    parser.add_argument("--baseline-pcp-size", type=int, default=1)
    parser.add_argument("--pcp-size", type=int, default=2)
    parser.add_argument("--fuse-matmuls",
                        choices=["auto", "true", "false"],
                        default="auto")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_probe(args)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
    _print_summary(payload)


if __name__ == "__main__":
    main()
