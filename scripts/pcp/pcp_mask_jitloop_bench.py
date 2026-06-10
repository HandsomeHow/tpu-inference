import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from tpu_inference.kernels.experimental.pcp_streaming_rpa.kernel import (
    pcp_streaming_attention_page_groups_packed_local,
)
from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    build_pcp_streaming_active_page_groups,
    generate_pcp_streaming_schedule,
)
try:
    from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
        build_pcp_streaming_active_split_tiles,
    )
except ImportError:
    build_pcp_streaming_active_split_tiles = None


P = jax.sharding.PartitionSpec
AXIS = "pcp"


def pack_native_kv_cache(kv_cache, kv_packing):
    pcp_size, pages, page_size, kv_heads, kv_pair, head_dim = kv_cache.shape
    if kv_pair != 2:
        raise ValueError("native KV cache must have K/V pair axis.")
    aligned = math.ceil(kv_heads * 2 / kv_packing) * kv_packing
    flat = np.zeros((pcp_size, pages, page_size, aligned, head_dim),
                    dtype=kv_cache.dtype)
    flat[..., :kv_heads * 2, :] = kv_cache.reshape(
        pcp_size, pages, page_size, kv_heads * 2, head_dim)
    return flat.reshape(pcp_size, pages, page_size, aligned // kv_packing,
                        kv_packing, head_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--collective-id", type=int, required=True)
    parser.add_argument("--jit-loop-iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--q-start", type=int, default=126976)
    parser.add_argument("--q-len", type=int, default=4096)
    parser.add_argument("--q-block-size", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--kv-pages-per-block", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--q-per-kv", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--use-split", action="store_true")
    args = parser.parse_args()

    pcp_size = 8
    kv_len = args.q_start + args.q_len
    local_q_len = args.q_len // pcp_size
    local_pages = math.ceil(kv_len / (pcp_size * args.page_size))

    rng = np.random.default_rng(20260609)
    q_by_rank = (
        rng.normal(size=(pcp_size, local_q_len, args.kv_heads, args.q_per_kv,
                         args.head_dim)).astype(np.float32) * 0.01)
    q_global = q_by_rank.reshape(pcp_size * local_q_len, args.kv_heads,
                                 args.q_per_kv, args.head_dim)
    native_kv = (
        rng.normal(size=(pcp_size, local_pages, args.page_size, args.kv_heads,
                         2, args.head_dim)).astype(np.float32) * 0.01)
    packed_kv_np = pack_native_kv_cache(native_kv, 2)

    schedule = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, args.q_len],
        q_start_offsets=[args.q_start],
        block_tables=np.arange(local_pages, dtype=np.int32)[None, :],
        page_size=args.page_size,
        pcp_size=pcp_size,
        interleave_size=args.page_size,
        num_lanes=1,
        bq_sz=args.q_block_size,
        pad_kv_pages_to_pcp_group=True,
        kv_pages_per_block=args.kv_pages_per_block,
    )
    active = build_pcp_streaming_active_page_groups(schedule)
    split_tile_plan = getattr(schedule, "split_tile_plan", None)
    split_history_schedule = getattr(schedule, "split_history_packed_schedule",
                                    None)
    split_masked_schedule = getattr(schedule, "split_masked_packed_schedule",
                                   None)
    if split_tile_plan is None:
        split_tile_plan = np.zeros((1, pcp_size, 1, 128), dtype=np.int32)
    if split_history_schedule is None:
        split_history_schedule = np.zeros((pcp_size, pcp_size, 1, 128),
                                          dtype=np.int32)
    if split_masked_schedule is None:
        split_masked_schedule = np.zeros((pcp_size, pcp_size, 1, 128),
                                         dtype=np.int32)
    if build_pcp_streaming_active_split_tiles is None:
        active_split_tiles = np.array([0], dtype=np.int32)
    else:
        active_split_tiles = build_pcp_streaming_active_split_tiles(schedule)
    print(
        json.dumps(
            {
                "label": args.label,
                "schedule_shape": list(schedule.packed_schedule.shape),
                "active_page_groups": int(active[0]),
                "split_tile_plan_shape":
                    (list(split_tile_plan.shape)
                     if getattr(schedule, "split_tile_plan", None) is not None
                     else None),
                "split_history_groups":
                    (int(split_history_schedule.shape[0] //
                         pcp_size)
                     if getattr(schedule, "split_history_packed_schedule",
                                None) is not None
                     else None),
                "split_masked_groups":
                    (int(split_masked_schedule.shape[0] //
                         pcp_size)
                     if getattr(schedule, "split_masked_packed_schedule",
                                None) is not None
                     else None),
                "active_split_tiles": int(active_split_tiles[0]),
                "use_split": args.use_split,
                "local_pages": int(local_pages),
                "jit_loop_iters": args.jit_loop_iters,
            },
            sort_keys=True),
        flush=True)

    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))
    sm_scale = 1.0 / math.sqrt(args.head_dim)

    def call(q_local, kv_cache_local, packed_schedule, active_page_groups,
             split_tile_plan, split_history_schedule, split_masked_schedule,
             active_split_tiles):

        def body(_, carry):
            if args.use_split:
                return pcp_streaming_attention_page_groups_packed_local(
                    carry,
                    kv_cache_local[0],
                    packed_schedule,
                    active_page_groups,
                    pcp_size=pcp_size,
                    q_block_size=args.q_block_size,
                    sm_scale=sm_scale,
                    collective_id=args.collective_id,
                    kv_pages_per_block=args.kv_pages_per_block,
                    split_tile_plan=split_tile_plan,
                    split_history_packed_schedule=split_history_schedule,
                    split_masked_packed_schedule=split_masked_schedule,
                    active_split_tiles=active_split_tiles,
                )
            return pcp_streaming_attention_page_groups_packed_local(
                carry,
                kv_cache_local[0],
                packed_schedule,
                active_page_groups,
                pcp_size=pcp_size,
                q_block_size=args.q_block_size,
                sm_scale=sm_scale,
                collective_id=args.collective_id,
                kv_pages_per_block=args.kv_pages_per_block,
            )

        return lax.fori_loop(0, args.jit_loop_iters, body, q_local)

    fn = jax.jit(
        jax.shard_map(
            call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
                P(None),
                P(None, None, None, None),
                P(None, None, None, None),
                P(None, None, None, None),
                P(None),
            ),
            out_specs=P(AXIS, None, None, None),
            check_vma=False,
        ))

    q_dev = jnp.asarray(q_global, dtype=jnp.bfloat16)
    kv_dev = jnp.asarray(packed_kv_np, dtype=jnp.bfloat16)
    sched_dev = jnp.asarray(schedule.packed_schedule)
    active_dev = jnp.asarray(active)
    split_tile_plan_dev = jnp.asarray(split_tile_plan)
    split_history_dev = jnp.asarray(split_history_schedule)
    split_masked_dev = jnp.asarray(split_masked_schedule)
    active_split_tiles_dev = jnp.asarray(active_split_tiles)

    start = time.perf_counter()
    out = fn(q_dev, kv_dev, sched_dev, active_dev, split_tile_plan_dev,
             split_history_dev, split_masked_dev, active_split_tiles_dev)
    out.block_until_ready()
    compile_and_first_s = time.perf_counter() - start
    print(
        json.dumps({
            "label": args.label,
            "compile_and_first_loop_s": round(compile_and_first_s, 6)
        },
                   sort_keys=True),
        flush=True)

    times = []
    for _ in range(args.rounds):
        start = time.perf_counter()
        out = fn(q_dev, kv_dev, sched_dev, active_dev, split_tile_plan_dev,
                 split_history_dev, split_masked_dev, active_split_tiles_dev)
        out.block_until_ready()
        times.append(time.perf_counter() - start)

    per_iter = [t / args.jit_loop_iters for t in times]
    print(
        json.dumps(
            {
                "label":
                    args.label,
                "loop_times_s": [round(t, 6) for t in times],
                "per_iter_ms": [round(t * 1000, 4) for t in per_iter],
                "mean_per_iter_ms":
                    round(float(np.mean(per_iter)) * 1000, 4),
                "median_per_iter_ms":
                    round(float(np.median(per_iter)) * 1000, 4),
            },
            sort_keys=True),
        flush=True)


if __name__ == "__main__":
    main()
