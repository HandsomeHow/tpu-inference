# PCP scripts

## PCP mask JIT-loop benchmark

`pcp_mask_jitloop_bench.py` is a standalone microbenchmark for the PCP
streaming attention page-group path. It creates synthetic Q/KV inputs, builds the
PCP schedule, runs the kernel inside an 8-device `shard_map`, and reports the
average per-iteration latency after JIT compilation.

Run from the repository root on an 8-device TPU machine:

```bash
PYTHONPATH=. python scripts/pcp/pcp_mask_jitloop_bench.py \
  --label no_split_fp32_ring_chunk32 \
  --collective-id 900 \
  --jit-loop-iters 10 \
  --rounds 5 \
  --q-start 126976 \
  --q-len 4096 \
  --q-block-size 256 \
  --kv-heads 2 \
  --q-per-kv 16 \
  --head-dim 256 \
  --kv-pages-per-block 8
```

The command above targets the 32nd 4K chunk in a 397B-style PCP=8 shape:

- `q_start=126976` is `(32 - 1) * 4096`.
- `q_len=4096` is one chunk.
- `kv_heads=2`, `q_per_kv=16`, and `head_dim=256` match the tested attention
  shape.
- `jit-loop-iters=10` runs ten kernel iterations inside one compiled function,
  reducing Python dispatch overhead in the reported per-iteration latency.

To exercise the split schedule path when the branch supports it, add
`--use-split` to the same command.

Use a fresh `--collective-id` when rerunning after another Pallas collective
benchmark in the same environment.
