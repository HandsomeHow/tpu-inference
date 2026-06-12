#!/usr/bin/env python3
"""Run Qwen3.5-397B 256K streaming TTFT measurements for prompt variants."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
VLLM_ROOT = WORKSPACE / "vllm"
PYTHON = WORKSPACE / ".venv-qwen35-pcp8" / "bin" / "python3"
MODEL = Path("/home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8")

PROMPT_LEN = 262_144
WARMUP_PROMPT_LEN = 32_768
MAX_MODEL_LEN = 263_168
CHUNKED_PREFILL_SIZE = 4_096
PORT = 18_200

REPEATED_TOKEN_ID = 23066  # " hello" for the Qwen3.5 tokenizer.
SEMANTIC_TEXT = (
    "System note: This benchmark prompt is a long coherent technical document "
    "about TPU inference. It explains how chunked prefill schedules, tensor "
    "parallel shards, expert parallel routing, KV cache pages, and batched "
    "ragged paged attention interact during one streaming completion request. "
    "The document keeps ordinary grammar and real relationships between ideas "
    "so the token stream resembles a production user document rather than "
    "random identifiers. Each section repeats the same operational theme: "
    "measure time to first token after a warmup, keep request parameters fixed, "
    "and compare PCP prefill with tensor parallel prefill under the same "
    "serving limits. "
)


def now_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VLLM_ROOT}:{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["JAX_PLATFORMS"] = "tpu,cpu"
    env["SKIP_JAX_PRECOMPILE"] = "1"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["NEW_MODEL_DESIGN"] = "1"
    return env


def mode_env(mode: str) -> dict[str, str]:
    env = base_env()
    if mode == "pcp8":
        env["USE_BATCHED_RPA_KERNEL"] = "0"
        env["USE_PCP_STREAMING_RPA_KERNEL"] = "1"
        env["PCP_STREAMING_RPA_NUM_LANES"] = "1"
        env["PCP_STREAMING_RPA_Q_BLOCK_SIZE"] = "256"
        env["PCP_STREAMING_RPA_KV_BLOCK_SIZE"] = "256"
    elif mode == "tp8":
        env["USE_BATCHED_RPA_KERNEL"] = "1"
        env["USE_PCP_STREAMING_RPA_KERNEL"] = "0"
        env["PCP_STREAMING_RPA_NUM_LANES"] = "1"
        env["PCP_STREAMING_RPA_Q_BLOCK_SIZE"] = "256"
        env["PCP_STREAMING_RPA_KV_BLOCK_SIZE"] = "256"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return env


def server_command(mode: str, port: int) -> tuple[str, list[str]]:
    if mode == "pcp8":
        served = "qwen397b-pcp8-ep8-tp1-256k"
        parallel_args = [
            "--prefill-context-parallel-size",
            "8",
            "--cp-kv-cache-interleave-size",
            "32",
        ]
    elif mode == "tp8":
        served = "qwen397b-tp8-ep8-256k-batched-rpa"
        parallel_args = ["--tensor-parallel-size", "8"]
    else:
        raise ValueError(f"unknown mode: {mode}")

    cmd = [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(MODEL),
        "--served-model-name",
        served,
        "--trust-remote-code",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--enable-expert-parallel",
        "--block-size",
        "32",
        "--gpu-memory-utilization",
        "0.7",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens",
        str(CHUNKED_PREFILL_SIZE),
        "--max-num-seqs",
        "1",
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        *parallel_args,
    ]
    return served, cmd


def prompt_ids(kind: str, length: int, tokenizer: Any) -> list[int]:
    if kind == "repeated":
        return [REPEATED_TOKEN_ID] * length
    if kind == "semantic":
        base = tokenizer.encode(SEMANTIC_TEXT, add_special_tokens=False)
        if not base:
            raise RuntimeError("semantic base text encoded to no tokens")
        repeats = (length + len(base) - 1) // len(base)
        return (base * repeats)[:length]
    raise ValueError(f"unknown prompt kind: {kind}")


def parse_token_id(token: str | None) -> int | None:
    if not token:
        return None
    prefix = "token_id:"
    if token.startswith(prefix):
        try:
            return int(token[len(prefix):])
        except ValueError:
            return None
    return None


def post_streaming_completion(
    *,
    endpoint: str,
    served_model_name: str,
    ids: list[int],
    prompt_kind: str,
    timeout_s: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": served_model_name,
        "prompt": ids,
        "add_special_tokens": False,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 1,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

    start = time.perf_counter()
    first_token_time = None
    first_choice: dict[str, Any] | None = None
    usage = None
    chunks: list[dict[str, Any]] = []
    generated = ""
    response_status = None

    with requests.post(
        endpoint,
        json=payload,
        headers=headers,
        stream=True,
        timeout=(30, timeout_s),
    ) as resp:
        response_status = resp.status_code
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line == "[DONE]":
                break
            data = json.loads(line)
            chunks.append(data)
            if data.get("usage") is not None:
                usage = data["usage"]
            choices = data.get("choices") or []
            if choices:
                choice = choices[0]
                text = choice.get("text") or ""
                generated += text
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    first_choice = choice

    end = time.perf_counter()
    if first_token_time is None or first_choice is None:
        raise RuntimeError("stream ended without a token-bearing choices chunk")

    logprobs = first_choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []

    return {
        "prompt_kind": prompt_kind,
        "prompt_len": len(ids),
        "response_status": response_status,
        "ttft_s": first_token_time - start,
        "latency_s": end - start,
        "first_token_text": first_choice.get("text"),
        "first_token_repr": tokens[0] if tokens else None,
        "first_token_id": parse_token_id(tokens[0]) if tokens else None,
        "first_token_logprob": token_logprobs[0] if token_logprobs else None,
        "finish_reason": first_choice.get("finish_reason"),
        "generated_text": generated,
        "usage": usage,
        "chunk_count": len(chunks),
        "request": {
            "endpoint": endpoint,
            "max_tokens": 1,
            "stream": True,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "add_special_tokens": False,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
        },
    }


def tail_file(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_for_health(proc: subprocess.Popen[Any], port: int, timeout_s: int,
                    log_path: Path) -> float:
    url = f"http://127.0.0.1:{port}/health"
    start = time.perf_counter()
    next_print = start
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited before health; rc={proc.returncode}\n"
                f"last log lines:\n{tail_file(log_path)}")
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return time.perf_counter() - start
        except requests.RequestException:
            pass
        now = time.perf_counter()
        if now - start > timeout_s:
            raise TimeoutError(
                f"server did not become healthy after {timeout_s}s\n"
                f"last log lines:\n{tail_file(log_path)}")
        if now >= next_print:
            print(f"[{now_stamp()}] waiting for {url} ({now - start:.1f}s)",
                  flush=True)
            next_print = now + 30
        time.sleep(5)


def terminate_server(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


def run_mode(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.out_dir) / now_stamp() / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    served_model_name, cmd = server_command(args.mode, args.port)
    env = mode_env(args.mode)
    env_record_keys = [
        "PYTHONPATH",
        "JAX_PLATFORMS",
        "SKIP_JAX_PRECOMPILE",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "NEW_MODEL_DESIGN",
        "USE_BATCHED_RPA_KERNEL",
        "USE_PCP_STREAMING_RPA_KERNEL",
        "PCP_STREAMING_RPA_NUM_LANES",
        "PCP_STREAMING_RPA_Q_BLOCK_SIZE",
        "PCP_STREAMING_RPA_KV_BLOCK_SIZE",
    ]

    prompt_kinds = [p.strip() for p in args.prompt_kinds.split(",") if p.strip()]
    log_path = run_dir / "server.log"
    summary: dict[str, Any] = {
        "mode": args.mode,
        "started_at": now_stamp(),
        "repo_root": str(REPO_ROOT),
        "model": str(MODEL),
        "served_model_name": served_model_name,
        "server_command": cmd,
        "server_env": {key: env.get(key) for key in env_record_keys},
        "prompt_len": PROMPT_LEN,
        "warmup_prompt_len": WARMUP_PROMPT_LEN,
        "prompt_kinds": prompt_kinds,
        "results": [],
        "server_log": str(log_path),
    }

    print(f"[{now_stamp()}] launching {args.mode} server", flush=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            health_s = wait_for_health(proc, args.port, args.server_timeout_s,
                                       log_path)
            summary["health_elapsed_s"] = health_s
            print(f"[{now_stamp()}] {args.mode} healthy after {health_s:.1f}s",
                  flush=True)

            endpoint = f"http://127.0.0.1:{args.port}/v1/completions"
            for kind in prompt_kinds:
                print(f"[{now_stamp()}] warmup {args.mode}/{kind}", flush=True)
                warm_ids = prompt_ids(kind, WARMUP_PROMPT_LEN, tokenizer)
                warmup = post_streaming_completion(
                    endpoint=endpoint,
                    served_model_name=served_model_name,
                    ids=warm_ids,
                    prompt_kind=kind,
                    timeout_s=args.request_timeout_s,
                )
                print(
                    f"[{now_stamp()}] measure {args.mode}/{kind} "
                    f"prompt_len={PROMPT_LEN}",
                    flush=True,
                )
                ids = prompt_ids(kind, PROMPT_LEN, tokenizer)
                preview = tokenizer.decode(ids[:128], skip_special_tokens=False)
                measure = post_streaming_completion(
                    endpoint=endpoint,
                    served_model_name=served_model_name,
                    ids=ids,
                    prompt_kind=kind,
                    timeout_s=args.request_timeout_s,
                )
                result = {
                    "mode": args.mode,
                    "prompt_kind": kind,
                    "prompt_preview_first_128_tokens": preview,
                    "warmup": warmup,
                    "measure": measure,
                }
                out_json = run_dir / f"{args.mode}_{kind}_prompt{PROMPT_LEN}.json"
                out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
                summary["results"].append({
                    "prompt_kind": kind,
                    "result_json": str(out_json),
                    "warmup_ttft_s": warmup["ttft_s"],
                    "measure_ttft_s": measure["ttft_s"],
                    "measure_latency_s": measure["latency_s"],
                    "measure_usage": measure["usage"],
                    "first_token_id": measure["first_token_id"],
                    "first_token_text": measure["first_token_text"],
                    "first_token_logprob": measure["first_token_logprob"],
                })
                print(
                    f"[{now_stamp()}] done {args.mode}/{kind}: "
                    f"ttft={measure['ttft_s']:.6f}s "
                    f"latency={measure['latency_s']:.6f}s",
                    flush=True,
                )
            return summary
        finally:
            print(f"[{now_stamp()}] stopping {args.mode} server", flush=True)
            terminate_server(proc)
            summary["server_returncode"] = proc.returncode
            summary_path = run_dir / f"{args.mode}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            print(f"[{now_stamp()}] wrote {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pcp8", "tp8"], required=True)
    parser.add_argument("--prompt-kinds", default="repeated,semantic")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "pcp_streaming_correctness_results" /
                    "ttft_256k_prompt_kinds"),
    )
    parser.add_argument("--server-timeout-s", type=int, default=3600)
    parser.add_argument("--request-timeout-s", type=int, default=2700)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_mode(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
