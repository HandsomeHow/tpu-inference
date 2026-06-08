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
