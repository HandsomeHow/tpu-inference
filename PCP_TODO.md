# PCP TODO

更新时间：2026-06-01

## 当前已验证

- `Qwen3-0.6B` 真实权重 smoke：`DP=1, TP=1, PCP=8` 的 prefill 后接 decode/generate 可以跑通。
- 与 `PCP=1` 对比，3 条 prompt、每条 8 个 greedy decode token 完全一致。
- 首 token top-1 一致，首 token logprob 有约 `0.02` 以内 BF16 级别差异。
- Synthetic runner 集成测试已覆盖 `DP=1, PCP>1` 和 `DP=2, PCP=2`，并校验 `pcp_slot_ids` 在 `BATCH` 轴下按 DP slice 传递。
- `Qwen3-0.6B` 真实权重已验证 `DP=2, TP=1, PCP=2` generate smoke。4 条 prompt、每条 4 个 greedy decode token 与 `DP=1, TP=1, PCP=1` baseline 完全一致。
- `Qwen3-0.6B` 真实权重已验证 `PCP=8` 不再需要 `num_gpu_blocks_override` 即可完成 KV cache 初始化和单请求串行 generate smoke。当前 `determine_available_memory` 保持返回 worker aggregate HBM，sharded KV page sizing 负责把 vLLM block 数映射到每个 device 的 local page 数。
- `Qwen3-0.6B` 真实权重已验证 `DP=1, TP=2, PCP=2/4` 和 `DP=1, TP=4, PCP=2` generate smoke。3 条 prompt、每条 4 个 greedy decode token 与既有 baseline 完全一致。
- `Qwen3-0.6B` 真实权重已验证 `DP=2, TP=1, PCP=4` 和 `DP=4, TP=1, PCP=2` generate smoke。4 条 prompt、每条 4 个 greedy decode token 与同 DP 下的 `PCP=1` baseline 完全一致；首个生成 token logprob 最大差异约 `4.8e-7`。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp4_tp1_dp2_20260527_200101.json`、`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp2_tp1_dp4_20260527_200410.json`。
- PCP + async scheduling 已在 TPU platform config 检查阶段提前拒绝，避免等到 runner 输入准备阶段才失败。
- 首期 scope guard 已覆盖：PCP sharded KV cache 会拒绝 MLA、Mamba state cache、JAX KV-share，以及 compilation static context 中的 `kv_sharing_target_layer_name` KV-share，避免非 full-attention paged-KV 路径静默进入 PCP。
- batched RPA normal paged prefill 已给 `kv_cache_dtype=fp8` 保留额外 VMEM headroom，避免 baseline `RPAm-p256-b2-q512-k512` 在 XLA 编译阶段因 fp8 KV unpack spill 超过 64 MiB VMEM。实机 smoke：`Qwen3-0.6B, DP=1, TP=1, PCP=2, kv_cache_dtype=fp8, max_model_len=128, max_tokens=2, max_num_seqs=2` 通过，4 条 prompt token/text 与 `PCP=1` baseline 完全一致，首 token logprob diff 为 0。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp2_tp1_dp1_20260527_201838.json`。
- `Qwen3-0.6B` 当前代码实机 smoke：`DP=1, TP=1, PCP=8, max_model_len=512, max_tokens=8, max_num_seqs=4, no num_gpu_blocks_override` 通过，4 条 prompt token/text 与 `PCP=1` baseline 完全一致，首 token logprob 最大差异 `4.8e-7`。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp8_tp1_dp1_20260527_233242.json`。
- `Qwen3.5-397B-A17B-FP8` 配置中的 full-attention 大参数 UT 已覆盖：`32` Q heads、`2` KV heads、`head_dim=256`、`PCP=8`、512 token local-Q/full-KV prefill，`_jax_attn_func -> flash-attn -> batched RPA` 输出与 reference causal attention 对齐。测试：`tests/layers/vllm/backends/test_flash_attn.py::test_jax_attn_func_pcp_qwen35_full_attention_config_matches_reference`。
- `Qwen3-0.6B` 当前 `feature/pcp_support` + vLLM PCP 依赖 commit 已验证 `SKIP_JAX_PRECOMPILE=0` 的 PCP=8 预编译路径：`TP=1, DP=1, PCP=8, chunked-prefill=512` 可以在 init 阶段完成 backbone bucket 预编译，正式 `generate` 后没有再次触发完整 `jit(step_fun)` backbone 编译。
- `Qwen3-0.6B` 长 prompt prefill 首 token / logprob 已覆盖 `TP=1, DP=1, PCP=8, chunked-prefill=4K` 下 `2K/4K/8K/64K` 语义无关长 prompt。首 token top-1 稳定，概率差异主要发生在 top 候选接近的位置，未观察到概率质量跑到离谱 token。
- `Qwen3-0.6B` 已验证 `data=2, TP=2, PCP=2, chunked-prefill=2K` 下 `1K/2K/4K/16K` prefill 首 token/logprob smoke。该组合实际 mesh 为 `data=2, attn_dp=1, model=2, pcp=2`；严格的 `attn_dp=2, final TP=2, PCP=2` 在 Qwen3-0.6B + BF16 + 当前自动 sharding 规则下 8 卡不会被拆出来。
- `Qwen3-0.6B` 真实语义 decode smoke 已覆盖 `TP=1, DP=1, PCP=8, chunked-prefill=2K, max_tokens=256` 的 `4K/16K` prompt。4K 与 `PCP=1` 不 token-exact，但 quicksort 回答语义正常；16K 与 `PCP=1` 256 个输出 token 完全一致。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_semantic_decode_pcp1_chunk2k_1780307239.json`、`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_semantic_decode_pcp8_chunk2k_gmem015_1780307358.json`。
- `Qwen3-0.6B` 重复 token prompt 的 decode 对比显示 `PCP=8` 不保证与 `PCP=1` token-exact：`1K/2K/4K` 分别在第 `2/6/2` 个 decode step 分叉，`16K` 在首 token 分叉。但分叉点的候选通常在双方 top-2/top-3 内，margin 很小；当前归类为可接受的数值非 bit-exact，而非功能 blocker。对比文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/compare_qwen3_06b_decode_baseline_pcp1_chunk2k_1780306332_vs_qwen3_06b_decode_pcp8_chunk2k_gmem015_1780306573.json`。

## P0：正确性和真实运行阻塞

### 1. PCP 不支持 async scheduling（已加配置期 guard）

真实运行 `Qwen3-0.6B` 时，如果 `prefill_context_parallel_size=8` 且使用默认 async scheduling，会在 runner 输入准备阶段失败：

```text
NotImplementedError: PCP runner path does not support async scheduling yet.
```

当前要求：

```text
async_scheduling=False
```

已完成：

- 短期明确 PCP 不支持 async scheduling。
- 在 `TpuPlatform.check_and_update_config()` 中，当 `prefill_context_parallel_size > 1` 且 `async_scheduling=True` 时提前报错。
- 补了 platform 单测覆盖该配置检查。

后续需要做：

- 如果要支持，需要检查 async scheduler 下 `scheduler_output`、`assigned_dp_rank`、`num_computed_tokens`、logits selector 和 KV cache update 的一致性。

### 2. PCP=8 下 KV cache sizing 会 OOM（sharded KV sizing 已修复）

真实运行 `Qwen3-0.6B` 时，`PCP=8` 如果不指定 `num_gpu_blocks_override`，KV cache 初始化会 OOM。日志里可见 num blocks 按 8 个 device 的总 HBM 估算，但当前 KV cache sharding 不包含 `pcp` 轴，每个 device 都尝试分配过大的 cache。

失败现象：

```text
Attempting to allocate ~24G; free HBM only ~20-21G
regular_attn_sharding=P(('data', 'attn_dp', 'attn_dp_expert'), 'dcp', ('model', 'expert'))
```

当前修复方式：

- KV cache 第一维已改为 `KV_CACHE_BLOCK = ('data', 'attn_dp', 'attn_dp_expert', 'pcp')` sharding。
- TPU 上报给 vLLM 的 attention `page_size_padded` 使用 `allocation_page_bytes * kv_cache_block_shard_count`，让 vLLM 用 worker aggregate HBM 计算出每个 device 的 local block 数。
- `TPUWorker.determine_available_memory()` 不再为 PCP 做额外除法或 HBM reserve；它保持返回 worker 内所有 devices 的 aggregate HBM。
- Phase 0 sanity test 覆盖 `DP/PCP/DCP` 组合，验证 `core_num_blocks == device_local_num_pages`，并避免 DP/PCP 重复除法。
- 实机回归：`Qwen3-0.6B, TP=1, PCP=8, max_model_len=512, max_num_seqs=1, max_tokens=4, no num_gpu_blocks_override` 通过。日志目录：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_pcp8_no_reserve_maxseq1_20260527_175009/`。
- 对比试验：`max_num_seqs=4` 能完成 KV cache 初始化，但在 generate 阶段触发已知的 mixed prefill/decode guard。日志目录：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_pcp8_no_reserve_no_override_20260527_174908/`。

后续需要做：

- 将真实启动回归固化为可重复脚本或 e2e 测试，避免默认配置退化后只在手工测试里发现。
- PCP decode materialize/all-gather 对 KV cache block 数敏感。`Qwen3-0.6B, PCP=8, max_model_len=18432, max_tokens=64` 在 `gpu_memory_utilization=0.5` 下曾因 `jit(_jax_attn_func)/shard_map/all_gather` 临时量触发 compile-time HBM OOM；降到 `0.15` 后可跑完。上线配置需要限制 KV cache 规模，或后续优化 decode all-gather/materialize 的临时内存。

### 3. 真实模型的 PCP + DP>1 generate correctness（已修复并验证多组 smoke）

`PCP + DP>1` 已有 synthetic runner 集成测试，并已用真实 `Qwen3-0.6B` 权重跑过完整 prefill -> decode/generate smoke。

已完成：

- 发现 `DP=2, PCP=2` 下 decode 阶段第 2 条 prompt 会串到第 1 条 prompt 的 hidden state。根因是 PCP 配置下 sampling 前使用全局 `hidden_states[logits_indices]`，但非 prefill 分支仍写 per-DP local logits index，DP rank 1 会错误选择 DP rank 0 的 hidden state。
- 修复为 `pcp_size > 1` 时非 prefill logits index 也加上 per-DP `token_offset`，与全局 gather 语义一致。
- 单测覆盖 `DP=2, PCP=2` decode logits index。
- 实机 smoke：`DP=2, TP=1, PCP=2, cp_kv_cache_interleave_size=16, max_model_len=512, max_tokens=4, max_num_seqs=4`，4 条 prompt 的 token ids/text 均与 `DP=1, TP=1, PCP=1` baseline 一致。
- 实机 smoke：`DP=2, TP=1, PCP=4, cp_kv_cache_interleave_size=16, max_model_len=512, max_tokens=4, max_num_seqs=4`，4 条 prompt 的 token ids/text 均与 `DP=2, TP=1, PCP=1` baseline 一致。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp4_tp1_dp2_20260527_200101.json`。
- 实机 smoke：`DP=4, TP=1, PCP=2, cp_kv_cache_interleave_size=16, max_model_len=512, max_tokens=4, max_num_seqs=4`，4 条 prompt 的 token ids/text 均与 `DP=4, TP=1, PCP=1` baseline 一致。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp2_tp1_dp4_20260527_200410.json`。

后续需要做：

- 已补充 `TP=1, DP=1, PCP=8` 的长 prompt、chunked-prefill、64/256 token decode、prompt logprob/top-logprob 对比；仍需补充更多 batch 形态和自动化回归。
- 将真实启动回归固化为脚本或 e2e 测试。

### 4. 混合 prefill/decode batch 未支持（已在调度层规避）

当前 PCP runner path 主要按 initial prefill 设计。代码中对以下情况会拒绝：

- scheduled token 数量 `<= 1`
- `num_computed_tokens != 0`

这意味着新 prefill 请求和已有 decode 请求混在同一个 batch 时，PCP 路径还没有完整支持。

已完成：

- 在 `prefill_context_parallel_size > 1` 时，如果当前 batch 的正数 scheduled token 同时包含 prefill(`>1`) 和 decode(`=1`)，会在 runner 输入准备阶段直接报 `NotImplementedError`。
- 单测覆盖 mixed prefill/decode batch。
- 增加 `PcpAwareScheduler`：PCP 开启时，如果 scheduler 已有 running requests，本轮临时进入 `PAUSED_NEW`，只调度 running，不再同时拉新的 waiting prefill。这样避免 vLLM 默认 scheduler 在同一 forward 中混合 running decode 和新 prefill。
- `DP>1` 时，`DPScheduler` 的 per-rank inner scheduler 也会使用 `PcpAwareScheduler`，保证每个 DP rank 内部同样避免 mixed prefill/decode。
- 实机回归：`Qwen3-0.6B, TP=1, PCP=8, max_model_len=512, max_num_seqs=4, max_tokens=4, no num_gpu_blocks_override` 通过。日志目录：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_pcp8_pcp_scheduler_maxseq4_20260527_180019/`。

注意：

- scheduler 可能让部分 active request 在某轮没有 scheduled token。这种 `0/1` decode batch 不能按 mixed batch 拒绝，否则会挡住正常 generate。
- partial/chunked prefill 已在 `Qwen3-0.6B` 的单请求长 prompt 场景中做过实机 smoke：`PCP=8, chunked-prefill=2K/4K` 覆盖到 `4K/16K/64K` prompt。仍缺多请求、不同长度混合和在线流量形态。
- 当前方案是调度期规避 mixed，而不是实现真正的 mixed-mode PCP attention。代价是：当已有 running requests 时，新请求会等到 running 队列清空后再进入，不能做到 decode 与新 prefill 同步连续 batching。

需要做：

- 如果后续需要更高在线吞吐，需要实现真正的 mixed prefill/decode PCP attention，或者设计更细粒度的 scheduler batch split。

## P1：功能完整性

### 5. PCP + DCP 当前显式不支持

当前 PCP RPA 路径已加 guard：

```text
PCP RPA does not support DCP yet.
```

需要做：

- 后续如果要同时支持 PCP 和 DCP，需要重新定义 Q/KV 切分、KV all-gather 范围、metadata sharding 和 cache layout。

### 5a. MLA/Mamba/KV-share 首期不支持（已加 scope guard）

首期 PCP sharded KV cache 只支持 full attention paged KV。当前已在 KV cache spec 构建阶段拒绝：

- MLA KV cache。
- Mamba state cache。
- JAX 路径从 HF config 派生出的 KV-share。
- vLLM compilation static context 中通过 `kv_sharing_target_layer_name` 声明的 KV-share。

测试覆盖：

- `tests/runner/test_kv_cache_manager.py::TestKVCacheManager::test_get_kv_cache_spec_rejects_pcp_mla`
- `tests/runner/test_kv_cache_manager.py::TestKVCacheManager::test_get_kv_cache_spec_rejects_pcp_jax_kv_share`
- `tests/runner/test_kv_cache_manager.py::TestKVCacheManager::test_get_kv_cache_spec_rejects_pcp_static_kv_share`
- `tests/runner/test_kv_cache_manager.py::TestKVCacheManager::test_get_kv_cache_spec_rejects_pcp_mamba_state_cache`

### 6. PCP + TP/EP 组合还缺真实 generate 回归

目前真实权重 smoke 已覆盖 `TP=2, PCP=2`，大模型目标场景里仍会涉及 `TP=8`、`EP=8` 等组合。

已完成：

- 初次 `Qwen3-0.6B, DP=1, TP=2, PCP=2` 实机 smoke 暴露 `RPAm-p256-b2-q768-k768` compile vmem OOM。失败点在 PCP decode LSE merge 的 normal paged RPA path：`use_full_kv_inputs=false, return_lse=true`，静态 vmem 估算没有覆盖 compiler spill pressure。
- batched RPA 的 `return_lse=True` paged prefill/mixed block size 现在和 full-KV PCP prefill 一样保留额外 vmem headroom，避免继续选到 `bq/bkv=768`。
- 单测覆盖该 block-size 回归：`tests/kernels/batched_ragged_paged_attention_test.py::BatchedRaggedPagedAttentionTest::test_return_lse_paged_prefill_block_sizes_leave_tp_pcp_spill_headroom`。
- 实机回归：`Qwen3-0.6B, DP=1, TP=2, PCP=2, cp_kv_cache_interleave_size=16, max_model_len=512, max_tokens=4, max_num_seqs=4` 通过。日志目录：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_tp2_pcp2_lse_vmem_fix_20260527_181011/`。
- 8-device 组合回归：`DP=1, TP=2, PCP=4` 通过，日志目录：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_tp2_pcp4_smoke_20260527_181333/`。
- 8-device 组合回归：`DP=1, TP=4, PCP=2` 通过，日志目录：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/qwen3_06b_tp4_pcp2_smoke_20260527_181557/`。

需要做：

- `Qwen3-0.6B` 上已补充更长 prompt 和更长 decode token；仍需扩展更多 batch 形态、不同 prompt 长度混合和自动化回归。
- 大模型上再跑 full attention 层和 MoE 组合的 correctness/perf smoke。
- 特别检查 `RowParallelLinear` 之后的输出形状和后续 MoE/top-k 对局部 token 的依赖。

### 7. KV cache 回填重复 scatter（已修复）

Phase 5 已删除 replicated full-KV 回填路径。当前 prefill/decode 都通过 runner 生成的 local `pcp_slot_ids`，在 shard_map 内只把本 rank 负责的 K/V 写入 local sharded KV cache。

后续仍需关注：

- profile local scatter 本身在长 prefill 下的占比。
- decode owner-only write / PCP materialize path 已在 `Qwen3-0.6B` 真实语义 generate 上做过 256 token smoke；后续仍需在更大模型和更多 batch 形态继续回归。

### 8. Decode logits gather 条件偏宽

当前 `_logits_indices_require_global_gather` 对 `pcp_size > 1` 的配置会走 global gather，即使当前 batch 实际不是 PCP metadata。

需要做：

- 确认 decode-only batch 是否可以只根据 `_attention_metadata_uses_pcp(attn_metadata)` 决定。
- 对比 global indexing 和 `_select_from_array_fn` 在 decode 下的性能和正确性。

### 9. KV cache sharding spec 需要统一或加 guard（已统一 normal RPA spec）

PCP prefill 路径和 normal decode 路径对 KV cache head 轴的 spec 需要保持一致。DCP=1 时目前等价，但未来 DCP 或轴定义变化时可能踩坑。

已完成：

- normal RPA 路径的 `kv_cache_spec` 已统一使用 `KV_CACHE_HEAD`，与 PCP prefill/decode 路径和 KV cache allocation sharding 对齐。

需要做：

- 与 PCP + DCP 的规划一起处理。

## P2：性能和工程债

### 10. Batched RPA 将 interleaved chunk 当 pseudo sequence

当前实现为了尽快闭环，把每个 interleaved chunk 当成一个小 sequence 处理。

已在代码里标注：

```text
TODO(xiaohao.yxh): Treating every interleaved chunk as a pseudo sequence
can create many tiny sequences when the interleave size is small...
```

需要做：

- 在 kernel scheduler 内原生支持一个 sequence 的多段 local-Q chunk。
- 避免 chunk size 变小时产生大量小 sequence，减少 schedule overhead 和 tile 利用率损失。

### 11. RPA tile/block size 需要长期优化

当前 PCP local-Q/full-KV 能跑通，但还没有针对长 P、不同 batch、不同 interleave size 系统调优。

需要做：

- 对 P8K/P16K、batch sweep、interleave size sweep 做 profile。
- 重点看 RPA kernel tile utilization、KV tile 被 causal mask 完全跳过或浪费的比例。
- 评估 `bq_sz <= bkv_sz` 这类 block size 约束是否能提升性能。

### 12. PCP 下 block_size 放大影响 cache 利用率

PCP 下 page/block size 会随 `prefill_cp_size` 放大。短序列可能只使用 page 的一小部分，降低 KV cache 可支持并发。

需要做：

- 重新评估是否必须按 PCP size 放大 block size。
- 至少保证 `block_size` 与 `cp_kv_cache_interleave_size` 的关系正确，而不是无条件牺牲 cache 利用率。

### 13. 数值测试覆盖还需要扩大

已有测试覆盖了 core path，并补充了 `Qwen3-0.6B` 长 prompt prefill、真实语义 decode 和重复 token decode 对比。当前结论是 PCP 路径不承诺与 `PCP=1` bit-exact/token-exact；当 top 候选 margin 很小时可能发生 top-1 翻转，但真实语义输出目前未观察到明显质量问题。

已补：

- 更长 prompt、更长 decode token。
- `TP=1, DP=1, PCP=8` 下 `4K/16K` 真实语义 prompt 256-token decode。
- `TP=1, DP=1, PCP=8` 下重复 token prompt `1K/2K/4K/16K` 64-token decode 对比。

仍需补：

- 不同 `cp_kv_cache_interleave_size`。
- 不同 batch size 和不同 prompt 长度混合。
- `kv_cache_dtype=fp8` 的真实 generate correctness/perf 回归。
- sliding window / sinks / quantized QKV 等特定 attention 参数。

### 14. 退出时有 JAX allocator cleanup 噪声（已修复）

真实运行结束后，EngineCore shutdown 时会出现：

```text
RuntimeError: device_allocator INTERNAL ASSERT FAILED ... Allocator for jax is not a DeviceAllocator.
```

该问题出现在成功生成之后，暂时不影响结果，但会污染日志和自动化判断。

已完成：

- 在 TPU platform 初始化时将 `torch.accelerator.empty_cache()` patch 成 no-op。vLLM 的 CPU worker 已经采用同类处理；TPU/JAX 后端的 cache 不由 PyTorch accelerator allocator 管理，跳过该调用可以避免 PyTorch 检查 JAX allocator 时触发 assert。
- 补了幂等性单测：`tests/platforms/test_tpu_platform.py::TestTpuPlatform::test_tpu_platform_noops_torch_accelerator_empty_cache`。
- 真实 smoke 验证：`Qwen3-0.6B, DP=1, TP=1, PCP=2, max_model_len=128, max_num_seqs=1, max_tokens=1` 能正常 shutdown，不再出现 `Allocator for jax is not a DeviceAllocator`。结果文件：`/mnt/data/xiaohao/workspace/llm/pcp_correctness_runs/Qwen3-0.6B_pcp2_tp1_dp1_20260527_201150.json`。
