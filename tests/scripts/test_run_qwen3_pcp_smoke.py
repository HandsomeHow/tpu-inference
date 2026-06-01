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

import json
from types import SimpleNamespace

from scripts.pcp.run_qwen3_pcp_smoke import (_build_engine_kwargs,
                                             compare_summaries, load_prompts,
                                             summarize_outputs)


def test_load_prompts_from_json_list(tmp_path):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")

    assert load_prompts(str(prompt_file)) == ["a", "b"]


def test_load_prompts_from_jsonl_objects(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"prompt": "a"}\n"b"\n', encoding="utf-8")

    assert load_prompts(str(prompt_file)) == ["a", "b"]


def test_summarize_outputs_serializes_tokens_and_logprobs():
    logprob_obj = SimpleNamespace(logprob=-0.25, rank=1, decoded_token="token")
    completion = SimpleNamespace(token_ids=[7],
                                 text="out",
                                 logprobs=[{
                                     7: logprob_obj
                                 }])
    output = SimpleNamespace(prompt="prompt",
                             prompt_token_ids=[1, 2],
                             outputs=[completion])

    summary = summarize_outputs([output])[0]

    assert summary["prompt"] == "prompt"
    assert summary["prompt_token_ids"] == [1, 2]
    assert summary["generated_token_ids"] == [7]
    assert summary["generated_text"] == "out"
    assert summary["logprobs"][0]["7"]["logprob"] == -0.25


def test_build_engine_kwargs_passes_moe_options():
    args = SimpleNamespace(
        model="model-path",
        max_model_len=128,
        max_num_seqs=2,
        tensor_parallel_size=1,
        data_parallel_size=2,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        dtype="bfloat16",
        trust_remote_code=True,
        enable_expert_parallel=True,
        max_num_batched_tokens=None,
        num_gpu_blocks_override=None,
        enable_prefix_caching=True,
        block_size=256,
        mamba_cache_mode="align",
        load_format=None,
        all2all_backend="default",
        additional_config_json=(
            '{"sharding": {"sharding_strategy": {"expert_parallelism": 4}}}'),
        cp_kv_cache_interleave_size=16,
    )

    kwargs = _build_engine_kwargs(args, pcp_size=2)

    assert kwargs["enable_expert_parallel"]
    assert kwargs["enable_prefix_caching"]
    assert kwargs["all2all_backend"] == "default"
    assert kwargs["block_size"] == 256
    assert kwargs["mamba_cache_mode"] == "align"
    assert kwargs["additional_config"] == {
        "sharding": {
            "sharding_strategy": {
                "expert_parallelism": 4
            }
        }
    }
    assert kwargs["prefill_context_parallel_size"] == 2
    assert kwargs["cp_kv_cache_interleave_size"] == 16


def test_compare_summaries_checks_tokens_text_and_logprob_tolerance():
    baseline = [{
        "prompt": "a",
        "generated_token_ids": [1, 2],
        "generated_text": "xy",
        "logprobs": [{
            "1": {
                "logprob": -0.1
            }
        }],
    }]
    candidate = [{
        "prompt": "a",
        "generated_token_ids": [1, 2],
        "generated_text": "xy",
        "logprobs": [{
            "1": {
                "logprob": -0.12
            }
        }],
    }]

    comparison = compare_summaries(baseline, candidate, logprob_abs_tol=0.05)

    assert comparison["ok"]
    assert comparison["items"][0]["first_generated_logprob_abs_diff"] == (
        0.01999999999999999)


def test_compare_summaries_flags_token_mismatch():
    baseline = [{
        "prompt": "a",
        "generated_token_ids": [1],
        "generated_text": "x",
        "logprobs": None,
    }]
    candidate = [{
        "prompt": "a",
        "generated_token_ids": [2],
        "generated_text": "y",
        "logprobs": None,
    }]

    comparison = compare_summaries(baseline, candidate, logprob_abs_tol=0.05)

    assert not comparison["ok"]
    assert not comparison["items"][0]["token_ids_match"]
    assert not comparison["items"][0]["text_match"]
