# Qwen3.5-397B FP8 TPU PCP/EP8 launch notes

This note records a working launch recipe for
`Qwen3.5-397B-A17B-FP8` on a single 8-chip TPU host.

The primary configuration is:

- `TP=2`
- `PCP=4`
- attention data parallelism = 2
- effective MoE expert parallelism = 8
- batched RPA attention kernel enabled

The main purpose is to let future agents reproduce the launch without relying
on machine-specific paths.

## Assumed checkout layout

Run commands from the `tpu-inference` repository root. The commands below
assume this sibling layout, but every path can be overridden with env vars.

```text
workspace/
  tpu-inference/
  vllm/
  models/
    Qwen3.5-397B-A17B-FP8/
```

If your layout differs, set `VLLM_DIR`, `MODEL_PATH`, and `PYTHON_BIN`
explicitly before running.

## Common environment

```bash
export TPU_INFERENCE_DIR="${TPU_INFERENCE_DIR:-$(pwd)}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${TPU_INFERENCE_DIR}/.." && pwd)}"
export VLLM_DIR="${VLLM_DIR:-${WORKSPACE_DIR}/vllm}"
export MODEL_PATH="${MODEL_PATH:-${WORKSPACE_DIR}/models/Qwen3.5-397B-A17B-FP8}"
export PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_DIR}/.venv-qwen3-tpu/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  export PYTHON_BIN=python3
fi

export PYTHONPATH="${TPU_INFERENCE_DIR}:${VLLM_DIR}:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_PLATFORMS=tpu,cpu
export USE_BATCHED_RPA_KERNEL=1
export SKIP_JAX_PRECOMPILE=1
```

## Primary launch: TP2 + PCP4 + attention-DP2

Use this when the model needs EP8 on one 8-chip host while also using PCP.

Important details:

- `NEW_MODEL_DESIGN=1` is required for the 7D mesh.
- `--tensor-parallel-size 2` plus
  `--additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}'`
  becomes `attention_data_parallelism=2` and model TP effectively 1.
- `--prefill-context-parallel-size 4` supplies the PCP axis.
- The TPU MoE path should shard experts over `attention-DP * PCP = 8`.
- Keep `--max-model-len 1024` and `--block-size 1024` to avoid the vLLM KV
  cache issue with mixed block sizes.

```bash
export NEW_MODEL_DESIGN=1

"${PYTHON_BIN}" examples/offline_inference.py \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size 2 \
  --prefill-context-parallel-size 4 \
  --cp-kv-cache-interleave-size 16 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 1024 \
  --block-size 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --no-async-scheduling \
  --disable-chunked-mm-input \
  --max-tokens 16 \
  --temperature 0 \
  --use-chat-template \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}'
```

Expected sanity signals in logs:

```text
Using experimental batched RPA kernel
Initialized sharding configuration: ... attention_data_parallelism=2 ... prefill_context_parallelism=4
Init mesh | mesh=Mesh('data': 1, 'attn_dp': 2, ... 'pcp': 4, ...)
[MoE]: Using GMM EP kernel
w13_weight_w1 shape after padding: (64, 4096, 1024)
regular_attn_shape=(num_blocks, (1024, 1, 4, 256))
```

The `(64, 4096, 1024)` local MoE weight shape is the key check: it means the
full expert set is being sharded across 8 physical shards.

## Chat correctness smoke test

Prefer chat API with thinking disabled. Raw prompt generation can repeatedly
emit `<think>` for this model; that is a known model/template behavior and is
not a good correctness signal for PCP/EP sharding.

```bash
export NEW_MODEL_DESIGN=1

"${PYTHON_BIN}" - <<'PY'
import os

from vllm import LLM, SamplingParams

prompts = [
    "Introduce Beijing in one short sentence.",
    "What is 2 + 3? Answer briefly.",
    "Translate to Chinese: The weather is nice today.",
]
messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

llm = LLM(
    model=os.environ["MODEL_PATH"],
    tensor_parallel_size=2,
    prefill_context_parallel_size=4,
    cp_kv_cache_interleave_size=16,
    enable_expert_parallel=True,
    kv_cache_dtype="fp8",
    max_model_len=1024,
    block_size=1024,
    max_num_seqs=1,
    max_num_batched_tokens=16384,
    gpu_memory_utilization=0.90,
    async_scheduling=False,
    disable_chunked_mm_input=True,
    additional_config={
        "sharding": {
            "sharding_strategy": {
                "enable_dp_attention": True,
            },
        },
    },
)

params = SamplingParams(max_tokens=16, temperature=0.0)
outputs = llm.chat(
    messages,
    params,
    chat_template_kwargs={"enable_thinking": False},
    use_tqdm=True,
)

for prompt, output in zip(prompts, outputs):
    print("PROMPT:", prompt)
    print("OUTPUT:", repr(output.outputs[0].text))
    print("---")
PY
```

## TP8 comparison baseline

Use TP8 as a semantic comparison baseline for the same prompts. This baseline
uses the default 2D mesh. Do not set `NEW_MODEL_DESIGN` for this baseline.

```bash
unset NEW_MODEL_DESIGN

"${PYTHON_BIN}" examples/offline_inference.py \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 1024 \
  --block-size 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --no-async-scheduling \
  --disable-chunked-mm-input \
  --max-tokens 16 \
  --temperature 0 \
  --use-chat-template \
  --chat-template-kwargs '{"enable_thinking": false}'
```

Expected sanity signals:

```text
Init mesh | mesh=Mesh('data': 1, 'model': 8, ...)
[MoE]: Using GMM EP kernel
w13_weight_w1 shape after padding: (64, 4096, 1024)
regular_attn_shape=(num_blocks, (1024, 8, 4, 256))
```

For a quick correctness check, run the same chat prompts under the TP2+PCP4
configuration and under TP8. With `temperature=0` and thinking disabled, the
answers should be semantically equivalent. In the reference run, both configs
returned the same answers for a Beijing sentence, `2 + 3`, and a short
English-to-Chinese translation.

To run the exact same chat smoke under TP8:

```bash
unset NEW_MODEL_DESIGN

"${PYTHON_BIN}" - <<'PY'
import os

from vllm import LLM, SamplingParams

prompts = [
    "Introduce Beijing in one short sentence.",
    "What is 2 + 3? Answer briefly.",
    "Translate to Chinese: The weather is nice today.",
]
messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

llm = LLM(
    model=os.environ["MODEL_PATH"],
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    kv_cache_dtype="fp8",
    max_model_len=1024,
    block_size=1024,
    max_num_seqs=1,
    max_num_batched_tokens=16384,
    gpu_memory_utilization=0.90,
    async_scheduling=False,
    disable_chunked_mm_input=True,
)

params = SamplingParams(max_tokens=16, temperature=0.0)
outputs = llm.chat(
    messages,
    params,
    chat_template_kwargs={"enable_thinking": False},
    use_tqdm=True,
)

for prompt, output in zip(prompts, outputs):
    print("PROMPT:", prompt)
    print("OUTPUT:", repr(output.outputs[0].text))
    print("---")
PY
```

## Troubleshooting

- If MoE local weight shape is `(256, 4096, 1024)`, the effective EP is only 2.
  Check that `NEW_MODEL_DESIGN=1`, `--prefill-context-parallel-size 4`, and
  `enable_dp_attention` are set for the TP2+PCP4 run.
- If `regular_attn_shape` does not use `block_size=1024`, confirm both
  `--max-model-len 1024` and `--block-size 1024` are present.
- If output repeatedly contains `<think>`, use chat API and pass
  `chat_template_kwargs={"enable_thinking": False}`.
- If the TP8 baseline fails in a 7D mesh with a QKV sharding axis error, unset
  `NEW_MODEL_DESIGN` and run it as the default 2D TP baseline.
