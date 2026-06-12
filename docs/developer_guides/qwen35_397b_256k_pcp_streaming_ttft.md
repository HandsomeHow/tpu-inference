# Qwen3.5-397B 256K PCP Streaming TTFT

This note records the exact launch and measurement setup used for the
Qwen3.5-397B FP8 256K prompt TTFT run with streaming PCP prefill.

The recorded run is the PCP case only:

- model: `/home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8`
- prompt length: `262144`
- chunked prefill size: `4096`
- max model length: `263168`
- PCP: `8`
- TP: `1`
- expert parallel: enabled
- block size: `32`
- PCP KV interleave size: `32`
- warmup prompt length: `32768`
- run directory:
  `/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_three_configs/20260608-070906`
- code revision after the template-prewarm fix:
  `bb004c60 Prewarm PCP streaming schedule templates`

## Checkout Layout

The run used this local layout:

```text
/mnt/data/workspace/llm/
  tpu-inference/
  vllm/
  .venv-qwen35-pcp8/

/home/xiaohao_yxh/workspace/models/
  Qwen3.5-397B-A17B-FP8/
```

## Top-Level Test Command

Run from the `tpu-inference` repo root:

```bash
cd /mnt/data/workspace/llm/tpu-inference

PYTHONPATH=/mnt/data/workspace/llm/vllm:/mnt/data/workspace/llm/tpu-inference \
  /mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  pcp_streaming_correctness_results/run_ttft_256k_three_configs.py \
  --only pcp8_ep8_tp1 \
  --warmup-prompt-len 32768 \
  --suffix _templcache_eager_warm32k_measure256k
```

The script launches the OpenAI API server, waits for `/health`, runs one
warmup request, runs the measured 256K request, writes JSON/log artifacts, and
then stops the server.

## Server Environment

The service process inherited the shell environment and explicitly set these
variables:

```bash
export PYTHONPATH="/mnt/data/workspace/llm/vllm:/mnt/data/workspace/llm/tpu-inference:${PYTHONPATH:-}"
export JAX_PLATFORMS=tpu,cpu
export SKIP_JAX_PRECOMPILE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NEW_MODEL_DESIGN=1

export USE_PCP_STREAMING_RPA_KERNEL=1
export PCP_STREAMING_RPA_NUM_LANES=1
export PCP_STREAMING_RPA_Q_BLOCK_SIZE=256
export PCP_STREAMING_RPA_KV_BLOCK_SIZE=256
```

## Server Command

The exact server command was:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 18200 \
  --model /home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8 \
  --served-model-name qwen397b-pcp8-ep8-tp1-256k \
  --trust-remote-code \
  --max-model-len 263168 \
  --enable-expert-parallel \
  --block-size 32 \
  --gpu-memory-utilization 0.7 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 32
```

## Measurement Command

The measurement subprocess used the same environment as the server and ran:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  /mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/measure_serving_ttft.py \
  --model-path /home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8 \
  --served-model-name qwen397b-pcp8-ep8-tp1-256k \
  --port 18200 \
  --out-json /mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_three_configs/20260608-070906/pcp8_ep8_tp1_prompt262144_templcache_eager_warm32k_measure256k.json \
  --mode pcp8_ep8_tp1_prompt262144_templcache_eager_warm32k_measure256k \
  --prompt-len 262144 \
  --timeout-s 2700 \
  --warmup \
  --warmup-prompt-len 32768
```

## Important Log Signals

The server log confirmed the production KV-cache capacity and eager schedule
template prewarm:

```text
Compact-mamba KV cache: num_gpu_blocks_override=9704 (attn), _mamba_num_blocks=9.
Hybrid KV cache layout: num_kv_cache_groups=4, num_kv_cache_tensors=15, kv_cache_config.num_blocks=9704, duplicate_shared_layers=True
Init kv-cache | num_total_layers=60 | ... regular_attn_layers=15 ...
Prewarmed PCP streaming schedule templates | templates=64 | generated=64 | max_model_len=263168 | chunk_size=4096 | pages_per_seq=[8224] | max_num_reqs_per_dp_rank=8 | pcp_size=8 | block_size=32 | interleave_size=32 | lanes=1 | q_block_size=256 | kv_pages_per_block=8 | elapsed_ms=8628.81
```

`pages_per_seq=[8224]` is the runtime block table capacity used by the runner
cache key. The eager prewarm must use this initialized block-table shape; using
only `ceil(max_model_len / block_size)` is not sufficient for this deployment.

## Result

Output files:

```text
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_three_configs/20260608-070906/pcp8_ep8_tp1_prompt262144_templcache_eager_warm32k_measure256k.json
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_three_configs/20260608-070906/pcp8_ep8_tp1_prompt262144_templcache_eager_warm32k_measure256k.stdout.log
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_three_configs/20260608-070906/pcp8_ep8_tp1_server.log
```

Measured values:

```text
warmup prompt length: 32768
warmup TTFT: 26.12567954370752 s

measure prompt length: 262144
measure TTFT: 28.477003111038357 s
measure latency: 28.47723402827978 s

first token id: 55404
first token text: " correctness"
first token logprob: -0.0005998004344291985
```

For comparison, the same 256K PCP service before eager template prewarm measured
about `35.43-35.72 s` TTFT with the same first token and logprob. The eager
prewarm run moved the roughly 64-template host generation cost into engine
initialization.

## 2026-06-12 Prompt Variants and TP=8 Batched-RPA Rerun

This follow-up reran the PCP=8 case with two explicit prompt construction
methods, then changed only the parallel serving mode from PCP=8 to TP=8 and
enabled the batched RPA kernel. The request shape and main serving limits were
kept the same.

- repo revision: `1a58adb4b487d6483f3b0e4540999ab02ad164c9`
- model: `/home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8`
- prompt length: `262144`
- warmup prompt length: `32768`
- chunked prefill size: `4096`
- max model length: `263168`
- max generated tokens: `1`
- stream: enabled
- expert parallel: enabled
- block size: `32`
- prompt variants:
  - `repeated`: token id `23066` repeated `262144` times
  - `semantic`: a deterministic coherent technical paragraph tokenized,
    repeated, and truncated to `262144` tokens
- measurement script:
  `pcp_streaming_correctness_results/run_qwen35_397b_256k_ttft_prompts.py`

### Result Summary

| Serving mode | Prompt kind | 256K TTFT | Status |
| --- | --- | ---: | --- |
| PCP=8, TP=1, PCP streaming RPA | `repeated` | `29.330640106927603 s` | success |
| PCP=8, TP=1, PCP streaming RPA | `semantic` | `28.02062379801646 s` | success |
| TP=8, PCP=1, batched RPA | `repeated` | n/a | EngineCore fatal before first token |
| TP=8, PCP=1, batched RPA | `semantic` | n/a | EngineCore fatal before first token |

The PCP=8 `semantic` TTFT is close to the earlier single-prompt result
`28.477003111038357 s`. The TP=8 batched-RPA configuration did not produce a
valid TTFT for either prompt kind because the engine terminated before any
token-bearing streaming chunk was returned.

The TP=8 logs confirmed the intended runtime mode:

```text
Using experimental batched RPA kernel
Init mesh | mesh=Mesh(... 'model': 8, ... 'pcp': 1, ...)
regular_attn_shape=(num_blocks, (32, 8, 2, 256))
```

Both TP=8 failures occurred while processing the 256K measured request after
`196608` prompt tokens had been computed and the next `4096` token chunk was
scheduled:

```text
num_computed_tokens=[196608]
total_num_scheduled_tokens=4096
jax.errors.JaxRuntimeError: INTERNAL: E0200: RuntimeUnexpectedCoreHalt
Detailed error: ... HLO: RPAm-p32-b2-q1024-k1024.1;
HLO computation: main.1247_spmd; HLO module: jit_step_fun
```

This should be treated as a TP=8 batched-RPA runtime/compiler/kernel failure
for this exact 256K configuration, not as a slow TTFT measurement.

### 128K Semantic Follow-Up

Because the 256K TP=8 batched-RPA runs failed after `196608` prompt tokens had
already been computed, a shorter real-semantic prompt was run to check whether
TP=8 can complete below that length. The same server limits and request shape
were kept, and only the measured prompt length changed to `131072`.

| Serving mode | Prompt kind | Prompt length | 128K TTFT | Status |
| --- | --- | ---: | ---: | --- |
| TP=8, PCP=1, batched RPA | `semantic` | `131072` | `12.996951603796333 s` | success |
| PCP=8, TP=1, PCP streaming RPA | `semantic` | `131072` | `10.917537875007838 s` | success |

Both runs returned the same first token id `7948` (`" warm"`). For this 128K
semantic prompt, PCP=8 was `2.079413728788495 s` faster than TP=8, about
`16.0%` lower TTFT relative to the TP=8 measurement. Equivalently, TP=8 was
about `19.0%` slower than PCP=8.

### Prompt and Request Code

The measured request was sent to `/v1/completions` with token ids directly in
the `prompt` field to avoid retokenization differences:

```python
payload = {
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
```

Prompt construction:

```python
REPEATED_TOKEN_ID = 23066

def prompt_ids(kind, length, tokenizer):
    if kind == "repeated":
        return [REPEATED_TOKEN_ID] * length
    if kind == "semantic":
        base = tokenizer.encode(SEMANTIC_TEXT, add_special_tokens=False)
        repeats = (length + len(base) - 1) // len(base)
        return (base * repeats)[:length]
    raise ValueError(f"unknown prompt kind: {kind}")
```

### Reproduction Commands

Run from the `tpu-inference` repo root.

The script defaults to `--prompt-len 262144`; shorter prompt scans can pass an
explicit `--prompt-len`.

PCP=8, both prompt variants:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  pcp_streaming_correctness_results/run_qwen35_397b_256k_ttft_prompts.py \
  --mode pcp8 \
  --prompt-kinds repeated,semantic \
  --port 18200
```

TP=8 batched-RPA, both prompt variants:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  pcp_streaming_correctness_results/run_qwen35_397b_256k_ttft_prompts.py \
  --mode tp8 \
  --prompt-kinds repeated,semantic \
  --port 18200
```

Because the first TP=8 measured prompt killed the engine before the second
prompt kind could run in the same process, the semantic TP=8 case was rerun as
a separate server process:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  pcp_streaming_correctness_results/run_qwen35_397b_256k_ttft_prompts.py \
  --mode tp8 \
  --prompt-kinds semantic \
  --port 18200
```

128K semantic TP=8 batched-RPA:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  pcp_streaming_correctness_results/run_qwen35_397b_256k_ttft_prompts.py \
  --mode tp8 \
  --prompt-kinds semantic \
  --prompt-len 131072 \
  --out-dir pcp_streaming_correctness_results/ttft_128k_prompt_kinds \
  --port 18200
```

128K semantic PCP=8:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  pcp_streaming_correctness_results/run_qwen35_397b_256k_ttft_prompts.py \
  --mode pcp8 \
  --prompt-kinds semantic \
  --prompt-len 131072 \
  --out-dir pcp_streaming_correctness_results/ttft_128k_prompt_kinds \
  --port 18200
```

### Server Environment

Common environment:

```bash
export PYTHONPATH="/mnt/data/workspace/llm/vllm:/mnt/data/workspace/llm/tpu-inference:${PYTHONPATH:-}"
export JAX_PLATFORMS=tpu,cpu
export SKIP_JAX_PRECOMPILE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NEW_MODEL_DESIGN=1
```

PCP=8 kernel environment:

```bash
export USE_BATCHED_RPA_KERNEL=0
export USE_PCP_STREAMING_RPA_KERNEL=1
export PCP_STREAMING_RPA_NUM_LANES=1
export PCP_STREAMING_RPA_Q_BLOCK_SIZE=256
export PCP_STREAMING_RPA_KV_BLOCK_SIZE=256
```

TP=8 batched-RPA kernel environment:

```bash
export USE_BATCHED_RPA_KERNEL=1
export USE_PCP_STREAMING_RPA_KERNEL=0
export PCP_STREAMING_RPA_NUM_LANES=1
export PCP_STREAMING_RPA_Q_BLOCK_SIZE=256
export PCP_STREAMING_RPA_KV_BLOCK_SIZE=256
```

### Server Commands

PCP=8 server command:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 18200 \
  --model /home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8 \
  --served-model-name qwen397b-pcp8-ep8-tp1-256k \
  --trust-remote-code \
  --max-model-len 263168 \
  --enable-expert-parallel \
  --block-size 32 \
  --gpu-memory-utilization 0.7 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --prefill-context-parallel-size 8 \
  --cp-kv-cache-interleave-size 32
```

TP=8 batched-RPA server command:

```bash
/mnt/data/workspace/llm/.venv-qwen35-pcp8/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 18200 \
  --model /home/xiaohao_yxh/workspace/models/Qwen3.5-397B-A17B-FP8 \
  --served-model-name qwen397b-tp8-ep8-256k-batched-rpa \
  --trust-remote-code \
  --max-model-len 263168 \
  --enable-expert-parallel \
  --block-size 32 \
  --gpu-memory-utilization 0.7 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --tensor-parallel-size 8
```

### Output Artifacts

```text
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_prompt_kinds/20260612-052702/pcp8/pcp8_summary.json
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_prompt_kinds/20260612-052702/pcp8/server.log

/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_prompt_kinds/20260612-053631/tp8/tp8_summary.json
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_prompt_kinds/20260612-053631/tp8/server.log

/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_prompt_kinds/20260612-054615/tp8/tp8_summary.json
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_256k_prompt_kinds/20260612-054615/tp8/server.log

/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_128k_prompt_kinds/20260612-075035/tp8/tp8_summary.json
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_128k_prompt_kinds/20260612-075035/tp8/server.log

/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_128k_prompt_kinds/20260612-075540/pcp8/pcp8_summary.json
/mnt/data/workspace/llm/tpu-inference/pcp_streaming_correctness_results/ttft_128k_prompt_kinds/20260612-075540/pcp8/server.log
```
