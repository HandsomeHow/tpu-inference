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
"""Run a manual Qwen3 PCP correctness smoke test.

The script loads a baseline engine and a PCP engine sequentially, runs the same
greedy prompts, writes token/text/logprob summaries, and exits non-zero if the
generated token ids differ.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "What is 2 + 2?",
    "Please introduce yourself briefly.",
]

TRACE_ENABLED_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE"
TRACE_JSONL_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_JSONL"
TRACE_LABEL_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_LABEL"
TRACE_SAMPLE_SIZE_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_SAMPLE_SIZE"
TRACE_DUMP_DIR_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_DUMP_DIR"
TRACE_DUMP_STAGES_ENV = "TPU_INFERENCE_QWEN3_LAYER_TRACE_DUMP_STAGES"


def _set_default_env() -> None:
    os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
    os.environ.setdefault("USE_BATCHED_RPA_KERNEL", "1")
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)

    prompt_path = Path(path)
    raw = prompt_path.read_text(encoding="utf-8")
    if prompt_path.suffix == ".json":
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload = payload["prompts"]
        if not isinstance(payload, list) or not all(
                isinstance(item, str) for item in payload):
            raise ValueError("JSON prompt file must be a list of strings or "
                             "an object with a 'prompts' string list.")
        return payload

    prompts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if prompt_path.suffix == ".jsonl":
            item = json.loads(line)
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("prompt"),
                                                       str):
                prompts.append(item["prompt"])
            else:
                raise ValueError("JSONL prompt lines must be strings or "
                                 "objects with a string 'prompt' field.")
        else:
            prompts.append(line)
    return prompts


def _json_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _json_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _logprob_obj_to_json(obj: Any) -> dict[str, Any]:
    return {
        "logprob": _json_float(getattr(obj, "logprob", None)),
        "rank": _json_int(getattr(obj, "rank", None)),
        "decoded_token": getattr(obj, "decoded_token", None),
    }


def _logprobs_to_json(logprobs: Any) -> list[dict[str, Any]] | None:
    if logprobs is None:
        return None

    rows = []
    for row in logprobs:
        if row is None:
            rows.append({})
            continue
        rows.append({
            str(token_id): _logprob_obj_to_json(logprob_obj)
            for token_id, logprob_obj in row.items()
        })
    return rows


def summarize_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    summaries = []
    for output in outputs:
        completion = output.outputs[0]
        summaries.append({
            "prompt": output.prompt,
            "prompt_token_ids": list(output.prompt_token_ids),
            "generated_token_ids": list(completion.token_ids),
            "generated_text": completion.text,
            "logprobs": _logprobs_to_json(completion.logprobs),
        })
    return summaries


def _first_generated_logprob(summary: dict[str, Any]) -> float | None:
    token_ids = summary.get("generated_token_ids") or []
    logprobs = summary.get("logprobs") or []
    if not token_ids or not logprobs:
        return None
    row = logprobs[0]
    token_info = row.get(str(token_ids[0]))
    if token_info is None:
        return None
    logprob = token_info.get("logprob")
    return None if logprob is None else float(logprob)


def compare_summaries(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    logprob_abs_tol: float,
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        return {
            "ok": False,
            "error": f"result length mismatch: {len(baseline)} vs "
            f"{len(candidate)}",
            "items": [],
        }

    items = []
    ok = True
    for index, (base, cand) in enumerate(zip(baseline, candidate)):
        token_ids_match = (
            base["generated_token_ids"] == cand["generated_token_ids"])
        text_match = base["generated_text"] == cand["generated_text"]
        base_lp = _first_generated_logprob(base)
        cand_lp = _first_generated_logprob(cand)
        logprob_diff = None
        logprob_match = True
        if base_lp is not None and cand_lp is not None:
            logprob_diff = abs(base_lp - cand_lp)
            logprob_match = logprob_diff <= logprob_abs_tol
        item_ok = token_ids_match and text_match and logprob_match
        ok = ok and item_ok
        items.append({
            "index": index,
            "prompt": base["prompt"],
            "ok": item_ok,
            "token_ids_match": token_ids_match,
            "text_match": text_match,
            "first_generated_logprob_baseline": base_lp,
            "first_generated_logprob_candidate": cand_lp,
            "first_generated_logprob_abs_diff": logprob_diff,
        })

    return {"ok": ok, "items": items}


def _build_engine_kwargs(args: argparse.Namespace,
                         pcp_size: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "async_scheduling": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enable_expert_parallel": args.enable_expert_parallel,
        "prefill_context_parallel_size": pcp_size,
    }
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.num_gpu_blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.enable_prefix_caching:
        kwargs["enable_prefix_caching"] = True
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.mamba_cache_mode is not None:
        kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.load_format is not None:
        kwargs["load_format"] = args.load_format
    if args.all2all_backend is not None:
        kwargs["all2all_backend"] = args.all2all_backend
    if args.additional_config_json is not None:
        kwargs["additional_config"] = json.loads(args.additional_config_json)
    if pcp_size > 1:
        kwargs[
            "cp_kv_cache_interleave_size"] = args.cp_kv_cache_interleave_size
    return kwargs


def _trace_file(args: argparse.Namespace, pcp_size: int) -> Path:
    model_name = Path(args.model).name.replace("/", "_")
    trace_dir = Path(args.trace_output_dir or args.output_dir)
    run_id = getattr(args, "_trace_run_id", None)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        setattr(args, "_trace_run_id", run_id)
    return trace_dir / (
        f"{model_name}_pcp{pcp_size}_tp{args.tensor_parallel_size}_"
        f"dp{args.data_parallel_size}_{run_id}.jsonl")


def _trace_dump_dir(args: argparse.Namespace, pcp_size: int) -> Path:
    model_name = Path(args.model).name.replace("/", "_")
    trace_dir = Path(args.trace_dump_dir or args.trace_output_dir
                     or args.output_dir)
    run_id = getattr(args, "_trace_run_id", None)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        setattr(args, "_trace_run_id", run_id)
    return trace_dir / (
        f"{model_name}_pcp{pcp_size}_tp{args.tensor_parallel_size}_"
        f"dp{args.data_parallel_size}_{run_id}_tensors")


@contextmanager
def _temporary_env(updates: dict[str, str]):
    old_values = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _run_engine(args: argparse.Namespace, prompts: list[str],
                pcp_size: int) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    kwargs = _build_engine_kwargs(args, pcp_size)
    trace_env = {}
    if args.trace_qwen3_layers:
        trace_path = _trace_file(args, pcp_size)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
        trace_env = {
            TRACE_ENABLED_ENV: "1",
            TRACE_JSONL_ENV: str(trace_path),
            TRACE_LABEL_ENV: f"pcp{pcp_size}",
            TRACE_SAMPLE_SIZE_ENV: str(args.trace_sample_size),
        }
        if args.trace_dump_stages:
            dump_dir = _trace_dump_dir(args, pcp_size)
            dump_dir.mkdir(parents=True, exist_ok=True)
            for old_dump in dump_dir.glob("*.npy"):
                old_dump.unlink()
            trace_env[TRACE_DUMP_DIR_ENV] = str(dump_dir)
            trace_env[TRACE_DUMP_STAGES_ENV] = args.trace_dump_stages

    with _temporary_env(trace_env):
        llm = LLM(**kwargs)
        try:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                logprobs=args.logprobs if args.logprobs > 0 else None,
            )
            return summarize_outputs(llm.generate(prompts, sampling_params))
        finally:
            shutdown = getattr(llm, "shutdown", None)
            if shutdown is not None:
                shutdown()
            del llm
            gc.collect()


def _default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model).name.replace("/", "_")
    output_dir = Path(args.output_dir)
    return output_dir / (
        f"{model_name}_pcp{args.pcp_size}_tp{args.tensor_parallel_size}_"
        f"dp{args.data_parallel_size}_{timestamp}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3 greedy generation between PCP=1 and PCP=N.")
    parser.add_argument(
        "--model", default="/mnt/data/xiaohao/workspace/models/Qwen3-0.6B")
    parser.add_argument("--prompts-file")
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="pcp_correctness_runs")
    parser.add_argument("--trace-output-dir")
    parser.add_argument("--trace-qwen3-layers", action="store_true")
    parser.add_argument("--trace-sample-size", type=int, default=8)
    parser.add_argument("--trace-dump-dir")
    parser.add_argument("--trace-dump-stages")
    parser.add_argument("--pcp-size", type=int, default=8)
    parser.add_argument("--baseline-pcp-size", type=int, default=1)
    parser.add_argument("--cp-kv-cache-interleave-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mamba-cache-mode")
    parser.add_argument("--load-format")
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--logprob-abs-tol", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--all2all-backend")
    parser.add_argument("--additional-config-json")
    parser.add_argument("--candidate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    _set_default_env()
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    if args.prompt_limit is not None:
        if args.prompt_limit <= 0:
            raise ValueError("--prompt-limit must be positive when set.")
        prompts = prompts[:args.prompt_limit]
    if not prompts:
        raise ValueError("At least one prompt is required.")
    if args.pcp_size <= 1 and not args.candidate_only:
        raise ValueError(
            "--pcp-size must be > 1 when comparing with baseline.")
    if args.pcp_size > 1 and args.cp_kv_cache_interleave_size <= 0:
        raise ValueError("--cp-kv-cache-interleave-size must be positive.")
    args._trace_run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = Path(
        args.output) if args.output else _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "config": vars(args),
        "prompts": prompts,
        "baseline": None,
        "candidate": None,
        "comparison": None,
        "trace_files": {},
        "trace_dump_dirs": {},
    }

    if not args.candidate_only:
        payload["baseline"] = _run_engine(args, prompts,
                                          args.baseline_pcp_size)
        if args.trace_qwen3_layers:
            payload["trace_files"][f"pcp{args.baseline_pcp_size}"] = str(
                _trace_file(args, args.baseline_pcp_size))
            if args.trace_dump_stages:
                payload["trace_dump_dirs"][
                    f"pcp{args.baseline_pcp_size}"] = str(
                        _trace_dump_dir(args, args.baseline_pcp_size))
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["candidate"] = _run_engine(args, prompts, args.pcp_size)
    if args.trace_qwen3_layers:
        payload["trace_files"][f"pcp{args.pcp_size}"] = str(
            _trace_file(args, args.pcp_size))
        if args.trace_dump_stages:
            payload["trace_dump_dirs"][f"pcp{args.pcp_size}"] = str(
                _trace_dump_dir(args, args.pcp_size))

    ok = True
    if payload["baseline"] is not None:
        comparison = compare_summaries(
            payload["baseline"],
            payload["candidate"],
            logprob_abs_tol=args.logprob_abs_tol,
        )
        payload["comparison"] = comparison
        ok = bool(comparison["ok"])

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote PCP smoke result to {output_path}")
    if payload["comparison"] is not None:
        print(json.dumps(payload["comparison"], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
