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

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs, schedule
from tpu_inference.kernels.experimental.batched_rpa.wrapper import (
    _maybe_coalesce_pcp_pseudo_sequences, calculate_block_sizes,
    get_kv_cache_shape, ragged_paged_attention)

jax.config.parse_flags_with_absl()


def _ref_local_q_full_kv_attention(q, k, v, cu_q_lens, cu_k_lens, kv_lens,
                                   q_start_offsets, num_seqs):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    q_per_kv = num_q_heads // num_kv_heads
    outs = []

    for seq_idx in range(num_seqs):
        q_start = int(cu_q_lens[seq_idx])
        q_end = int(cu_q_lens[seq_idx + 1])
        k_start = int(cu_k_lens[seq_idx])
        q_global_start = int(q_start_offsets[seq_idx])
        kv_len = int(kv_lens[seq_idx])

        q_seq = q[q_start:q_end].astype(jnp.float32)
        k_seq = k[k_start:k_start + kv_len].astype(jnp.float32)
        v_seq = v[k_start:k_start + kv_len].astype(jnp.float32)
        k_seq = jnp.repeat(k_seq, q_per_kv, axis=1)
        v_seq = jnp.repeat(v_seq, q_per_kv, axis=1)

        attn = jnp.einsum("qhd,khd->hqk", q_seq, k_seq)
        q_pos = q_global_start + jnp.arange(q_seq.shape[0], dtype=jnp.int32)
        k_pos = jnp.arange(kv_len, dtype=jnp.int32)
        mask = q_pos[None, :, None] >= k_pos[None, None, :]
        attn = jnp.where(mask, attn, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(attn, axis=-1)
        outs.append(jnp.einsum("hqk,khd->qhd", probs, v_seq))

    return jnp.concatenate(outs, axis=0)


def _ref_local_q_full_kv_attention_and_lse(q, k, v, cu_q_lens, cu_k_lens,
                                           kv_lens, q_start_offsets, num_seqs):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    q_per_kv = num_q_heads // num_kv_heads
    outs = []
    lses = []

    for seq_idx in range(num_seqs):
        q_start = int(cu_q_lens[seq_idx])
        q_end = int(cu_q_lens[seq_idx + 1])
        k_start = int(cu_k_lens[seq_idx])
        q_global_start = int(q_start_offsets[seq_idx])
        kv_len = int(kv_lens[seq_idx])

        q_seq = q[q_start:q_end].astype(jnp.float32)
        k_seq = k[k_start:k_start + kv_len].astype(jnp.float32)
        v_seq = v[k_start:k_start + kv_len].astype(jnp.float32)
        k_seq = jnp.repeat(k_seq, q_per_kv, axis=1)
        v_seq = jnp.repeat(v_seq, q_per_kv, axis=1)

        attn = jnp.einsum("qhd,khd->hqk", q_seq, k_seq)
        q_pos = q_global_start + jnp.arange(q_seq.shape[0], dtype=jnp.int32)
        k_pos = jnp.arange(kv_len, dtype=jnp.int32)
        mask = q_pos[None, :, None] >= k_pos[None, None, :]
        attn = jnp.where(mask, attn, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(attn, axis=-1)
        outs.append(jnp.einsum("hqk,khd->qhd", probs, v_seq))
        lses.append(jax.nn.logsumexp(attn, axis=-1).T)

    return jnp.concatenate(outs, axis=0), jnp.concatenate(lses, axis=0)


def _build_pcp_interleaved_metadata(seq_lens,
                                    pcp_size,
                                    interleave_size,
                                    pcp_rank,
                                    page_size,
                                    *,
                                    max_num_reqs=None,
                                    active_distribution=False):
    active_num_reqs = len(seq_lens)
    if max_num_reqs is None:
        max_num_reqs = active_num_reqs
    if max_num_reqs < active_num_reqs:
        raise ValueError("max_num_reqs must cover seq_lens.")

    chunks_per_seq = max((seq_len + pcp_size * interleave_size - 1) //
                         (pcp_size * interleave_size) for seq_len in seq_lens)
    pages_per_seq = max(
        (seq_len + page_size - 1) // page_size for seq_len in seq_lens)

    kv_lens = []
    q_lens = []
    q_start_offsets = []
    cu_k_lens = []
    page_indices = []
    original_page_indices = []
    next_page = 0
    k_start = 0

    for req_idx in range(max_num_reqs):
        seq_len = seq_lens[req_idx] if req_idx < active_num_reqs else 0
        num_pages = (seq_len + page_size - 1) // page_size
        pages = np.arange(next_page, next_page + num_pages, dtype=np.int32)
        next_page += num_pages
        pages = np.pad(pages, (0, pages_per_seq - num_pages))
        original_page_indices.append(pages)

        for chunk_idx in range(chunks_per_seq):
            chunk_offset = (chunk_idx * pcp_size + pcp_rank) * interleave_size
            q_len = min(interleave_size, max(seq_len - chunk_offset, 0))
            q_lens.append(q_len)
            kv_lens.append(seq_len if q_len else 0)
            q_start_offsets.append(chunk_offset if q_len else 0)
            cu_k_lens.append(k_start if q_len else 0)
            page_indices.append(pages)
        k_start += seq_len

    cu_q_lens = np.pad(np.cumsum(q_lens, dtype=np.int32), (1, 0))
    cu_k_lens = np.concatenate([
        np.asarray(cu_k_lens, dtype=np.int32),
        np.asarray([sum(seq_lens)], dtype=np.int32)
    ])

    distribution_num_seqs = len(q_lens)
    if active_distribution:
        distribution_num_seqs = active_num_reqs * chunks_per_seq

    return {
        "chunks_per_seq":
        chunks_per_seq,
        "pages_per_seq":
        pages_per_seq,
        "total_pages":
        max(next_page, 1),
        "kv_lens":
        jnp.asarray(kv_lens, dtype=jnp.int32),
        "page_indices":
        jnp.asarray(np.stack(page_indices).reshape(-1), dtype=jnp.int32),
        "cu_q_lens":
        jnp.asarray(cu_q_lens, dtype=jnp.int32),
        "distribution":
        jnp.asarray([0, 0, distribution_num_seqs], dtype=jnp.int32),
        "q_start_offsets":
        jnp.asarray(q_start_offsets, dtype=jnp.int32),
        "cu_k_lens":
        jnp.asarray(cu_k_lens, dtype=jnp.int32),
        "original_page_indices":
        np.stack(original_page_indices),
    }


def _local_pcp_query(q_full, seq_lens, pcp_size, interleave_size, pcp_rank):
    chunks_per_seq = max((seq_len + pcp_size * interleave_size - 1) //
                         (pcp_size * interleave_size) for seq_len in seq_lens)
    q_parts = []
    seq_start = 0
    for seq_len in seq_lens:
        q_seq = q_full[seq_start:seq_start + seq_len]
        for chunk_idx in range(chunks_per_seq):
            chunk_offset = (chunk_idx * pcp_size + pcp_rank) * interleave_size
            q_len = min(interleave_size, max(seq_len - chunk_offset, 0))
            if q_len:
                q_parts.append(q_seq[chunk_offset:chunk_offset + q_len])
        seq_start += seq_len
    return jnp.concatenate(q_parts, axis=0)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class BatchedRaggedPagedAttentionTest(jtu.JaxTestCase):

    def _assert_rpa_close(self, actual, expected, dtype):
        if jnp.dtype(dtype) == jnp.dtype(jnp.float32):
            self.assertAllClose(actual, expected, atol=1e-2, rtol=1e-2)
        else:
            self.assertAllClose(actual, expected, atol=0.25, rtol=0.25)

    def test_default_full_kv_block_sizes_cover_large_pages(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        model_cfgs = configs.ModelConfigs(
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=128,
            mask_value=jnp.finfo(jnp.float32).min,
        )

        for page_size in (512, 1024, 2048):
            serve_cfgs = configs.ServingConfigs(
                num_seqs=32,
                page_size=page_size,
                total_q_tokens=64,
                num_page_indices=64,
                dtype_q=jnp.bfloat16,
                dtype_kv=jnp.bfloat16,
                dtype_out=jnp.bfloat16,
                use_full_kv_inputs=True,
            )
            decode_blocks, prefill_blocks = calculate_block_sizes(
                model_cfgs, serve_cfgs,
                pltpu.get_tpu_info().vmem_capacity_bytes)

            for block_sizes in (decode_blocks, prefill_blocks):
                self.assertGreaterEqual(block_sizes.bkv_sz, page_size)
                self.assertEqual(block_sizes.bkv_sz % page_size, 0)

    def test_full_kv_prefill_block_sizes_leave_tp_pcp_spill_headroom(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        model_cfgs = configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
            mask_value=jnp.finfo(jnp.float32).min,
        )
        serve_cfgs = configs.ServingConfigs(
            num_seqs=8,
            page_size=256,
            total_q_tokens=16,
            num_page_indices=16,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            use_full_kv_inputs=True,
        )

        _, prefill_blocks = calculate_block_sizes(
            model_cfgs, serve_cfgs,
            pltpu.get_tpu_info().vmem_capacity_bytes)

        self.assertLessEqual(prefill_blocks.bq_sz, 512)
        self.assertLessEqual(prefill_blocks.bkv_sz, 512)

    def test_return_lse_paged_prefill_block_sizes_leave_tp_pcp_spill_headroom(
            self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        model_cfgs = configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
            mask_value=jnp.finfo(jnp.float32).min,
        )
        serve_cfgs = configs.ServingConfigs(
            num_seqs=8,
            page_size=256,
            total_q_tokens=16,
            num_page_indices=16,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            use_full_kv_inputs=False,
            return_lse=True,
        )

        _, prefill_blocks = calculate_block_sizes(
            model_cfgs, serve_cfgs,
            pltpu.get_tpu_info().vmem_capacity_bytes)

        self.assertLessEqual(prefill_blocks.bq_sz, 512)
        self.assertLessEqual(prefill_blocks.bkv_sz, 512)

    def test_bfloat16_paged_prefill_block_sizes_leave_tp_spill_headroom(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        model_cfgs = configs.ModelConfigs(
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
            mask_value=jnp.finfo(jnp.float32).min,
        )
        serve_cfgs = configs.ServingConfigs(
            num_seqs=8,
            page_size=256,
            total_q_tokens=16,
            num_page_indices=16,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
            use_full_kv_inputs=False,
            return_lse=False,
        )

        _, prefill_blocks = calculate_block_sizes(
            model_cfgs, serve_cfgs,
            pltpu.get_tpu_info().vmem_capacity_bytes)

        self.assertLessEqual(prefill_blocks.bq_sz, 512)
        self.assertLessEqual(prefill_blocks.bkv_sz, 512)

    def test_fp8_kv_paged_prefill_block_sizes_leave_spill_headroom(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        model_cfgs = configs.ModelConfigs(
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=128,
            mask_value=jnp.finfo(jnp.float32).min,
        )
        serve_cfgs = configs.ServingConfigs(
            num_seqs=8,
            page_size=256,
            total_q_tokens=16,
            num_page_indices=8,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.bfloat16,
            use_full_kv_inputs=False,
            return_lse=False,
        )

        _, prefill_blocks = calculate_block_sizes(
            model_cfgs, serve_cfgs,
            pltpu.get_tpu_info().vmem_capacity_bytes)

        self.assertLessEqual(prefill_blocks.bq_sz, 256)
        self.assertLessEqual(prefill_blocks.bkv_sz, 256)

    def test_chunked_q_global_position_mapping(self):
        cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(
                bq_sz=128,
                bq_c_sz=64,
                bkv_sz=256,
                batch_size=1,
                n_buffer=2,
            ),
            model=configs.ModelConfigs(
                num_q_heads=4,
                num_kv_heads=1,
                head_dim=128,
                mask_value=jnp.finfo(jnp.float32).min,
            ),
            serve=configs.ServingConfigs(
                num_seqs=1,
                page_size=16,
                total_q_tokens=128,
                num_page_indices=64,
                dtype_q=jnp.float32,
                dtype_kv=jnp.float32,
                dtype_out=jnp.float32,
                use_full_kv_inputs=True,
                q_position_chunk_size=16,
                q_position_chunk_stride=128,
            ),
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=0,
        )

        local_offsets = jnp.asarray([0, 15, 16, 17, 31, 127], dtype=jnp.int32)
        actual = jax.vmap(lambda offset: schedule._q_global_position(
            jnp.asarray(5, dtype=jnp.int32), offset, cfgs))(local_offsets)
        expected = jnp.asarray([5, 20, 133, 134, 148, 916], dtype=jnp.int32)
        self.assertArraysEqual(actual, expected)

        no_chunk_cfgs = configs.RpaConfigs(
            block=cfgs.block,
            model=cfgs.model,
            serve=configs.ServingConfigs(
                num_seqs=1,
                page_size=16,
                total_q_tokens=128,
                num_page_indices=64,
                dtype_q=jnp.float32,
                dtype_kv=jnp.float32,
                dtype_out=jnp.float32,
                use_full_kv_inputs=True,
            ),
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=0,
        )
        actual_no_chunk = jax.vmap(lambda offset: schedule._q_global_position(
            jnp.asarray(5, dtype=jnp.int32), offset, no_chunk_cfgs))(
                local_offsets)
        self.assertArraysEqual(actual_no_chunk, local_offsets + 5)

    def test_coalesce_pcp_metadata_single_sequence(self):
        metadata = _build_pcp_interleaved_metadata(
            seq_lens=[1024],
            pcp_size=8,
            interleave_size=16,
            pcp_rank=0,
            page_size=16,
        )
        block_sizes = configs.BlockSizes(
            bq_sz=256,
            bq_c_sz=64,
            bkv_sz=256,
            batch_size=1,
            n_buffer=2,
        )

        result = _maybe_coalesce_pcp_pseudo_sequences(
            jnp.zeros((128, 4, 128), dtype=jnp.bfloat16),
            jnp.zeros((1024, 1, 128), dtype=jnp.bfloat16),
            metadata["kv_lens"],
            metadata["page_indices"],
            metadata["cu_q_lens"],
            metadata["distribution"],
            metadata["q_start_offsets"],
            metadata["cu_k_lens"],
            block_sizes,
            use_full_kv_inputs=True,
            update_kv_cache=False,
        )
        (kv_lens, page_indices, cu_q_lens, distribution, q_start_offsets,
         cu_k_lens, q_chunk_size, q_chunk_stride, coalesced_blocks) = result

        self.assertEqual(q_chunk_size, 16)
        self.assertEqual(q_chunk_stride, 128)
        self.assertEqual(coalesced_blocks.bq_sz, 128)
        self.assertEqual(coalesced_blocks.bq_c_sz, 64)
        self.assertArraysEqual(kv_lens, jnp.asarray([1024], dtype=jnp.int32))
        self.assertArraysEqual(cu_q_lens, jnp.asarray([0, 128],
                                                      dtype=jnp.int32))
        self.assertArraysEqual(distribution,
                               jnp.asarray([0, 0, 1], dtype=jnp.int32))
        self.assertArraysEqual(q_start_offsets,
                               jnp.asarray([0], dtype=jnp.int32))
        self.assertArraysEqual(cu_k_lens,
                               jnp.asarray([0, 1024], dtype=jnp.int32))
        self.assertArraysEqual(page_indices, jnp.arange(64, dtype=jnp.int32))

    def test_coalesce_pcp_metadata_uses_active_distribution_not_padding(self):
        metadata = _build_pcp_interleaved_metadata(
            seq_lens=[1024],
            pcp_size=8,
            interleave_size=16,
            pcp_rank=0,
            page_size=16,
            max_num_reqs=10,
            active_distribution=True,
        )
        block_sizes = configs.BlockSizes(
            bq_sz=256,
            bq_c_sz=64,
            bkv_sz=256,
            batch_size=1,
            n_buffer=2,
        )
        self.assertNotEqual(128 % metadata["kv_lens"].shape[0], 0)

        result = _maybe_coalesce_pcp_pseudo_sequences(
            jnp.zeros((128, 4, 128), dtype=jnp.bfloat16),
            jnp.zeros((1024, 1, 128), dtype=jnp.bfloat16),
            metadata["kv_lens"],
            metadata["page_indices"],
            metadata["cu_q_lens"],
            metadata["distribution"],
            metadata["q_start_offsets"],
            metadata["cu_k_lens"],
            block_sizes,
            use_full_kv_inputs=True,
            update_kv_cache=False,
        )
        (kv_lens, page_indices, cu_q_lens, distribution, q_start_offsets,
         cu_k_lens, q_chunk_size, q_chunk_stride, coalesced_blocks) = result

        self.assertEqual(q_chunk_size, 16)
        self.assertEqual(q_chunk_stride, 128)
        self.assertEqual(coalesced_blocks.bq_sz, 128)
        self.assertArraysEqual(distribution,
                               jnp.asarray([0, 0, 1], dtype=jnp.int32))
        self.assertEqual(kv_lens.shape[0], 10)
        self.assertEqual(page_indices.shape[0], 10 * 64)
        self.assertArraysEqual(kv_lens[:2],
                               jnp.asarray([1024, 0], dtype=jnp.int32))
        self.assertArraysEqual(cu_q_lens[:3],
                               jnp.asarray([0, 128, 128], dtype=jnp.int32))
        self.assertArraysEqual(q_start_offsets[:2],
                               jnp.asarray([0, 0], dtype=jnp.int32))
        self.assertArraysEqual(cu_k_lens[:3],
                               jnp.asarray([0, 0, 0], dtype=jnp.int32))

    def test_coalesce_pcp_metadata_multiple_sequences_nonzero_rank(self):
        metadata = _build_pcp_interleaved_metadata(
            seq_lens=[256, 256],
            pcp_size=4,
            interleave_size=8,
            pcp_rank=2,
            page_size=16,
        )
        block_sizes = configs.BlockSizes(
            bq_sz=64,
            bq_c_sz=32,
            bkv_sz=64,
            batch_size=1,
            n_buffer=2,
        )

        result = _maybe_coalesce_pcp_pseudo_sequences(
            jnp.zeros((128, 4, 128), dtype=jnp.float32),
            jnp.zeros((512, 1, 128), dtype=jnp.float32),
            metadata["kv_lens"],
            metadata["page_indices"],
            metadata["cu_q_lens"],
            metadata["distribution"],
            metadata["q_start_offsets"],
            metadata["cu_k_lens"],
            block_sizes,
            use_full_kv_inputs=True,
            update_kv_cache=False,
        )
        (kv_lens, page_indices, cu_q_lens, distribution, q_start_offsets,
         cu_k_lens, q_chunk_size, q_chunk_stride, coalesced_blocks) = result

        self.assertEqual(q_chunk_size, 8)
        self.assertEqual(q_chunk_stride, 32)
        self.assertEqual(coalesced_blocks.bq_sz, 32)
        self.assertEqual(coalesced_blocks.bq_c_sz, 32)
        self.assertArraysEqual(
            kv_lens, jnp.asarray([256, 256, 256, 256], dtype=jnp.int32))
        self.assertArraysEqual(
            cu_q_lens, jnp.asarray([0, 32, 64, 96, 128], dtype=jnp.int32))
        self.assertArraysEqual(distribution,
                               jnp.asarray([0, 0, 4], dtype=jnp.int32))
        self.assertArraysEqual(
            q_start_offsets, jnp.asarray([16, 144, 16, 144], dtype=jnp.int32))
        self.assertArraysEqual(
            cu_k_lens, jnp.asarray([0, 0, 256, 256, 512], dtype=jnp.int32))
        expected_pages = np.concatenate([
            np.arange(0, 16, dtype=np.int32),
            np.arange(0, 16, dtype=np.int32),
            np.arange(16, 32, dtype=np.int32),
            np.arange(16, 32, dtype=np.int32),
        ])
        self.assertArraysEqual(page_indices,
                               jnp.asarray(expected_pages, dtype=jnp.int32))

    def test_coalesce_pcp_metadata_noop_guards(self):
        metadata = _build_pcp_interleaved_metadata(
            seq_lens=[1024],
            pcp_size=8,
            interleave_size=16,
            pcp_rank=0,
            page_size=16,
        )
        q = jnp.zeros((128, 4, 128), dtype=jnp.bfloat16)
        k = jnp.zeros((1024, 1, 128), dtype=jnp.bfloat16)
        block_sizes = configs.BlockSizes(
            bq_sz=256,
            bq_c_sz=64,
            bkv_sz=256,
            batch_size=1,
            n_buffer=2,
        )

        guard_cases = [
            dict(
                testcase_name="regular_rpa_path",
                q=q,
                k=k,
                page_indices=metadata["page_indices"],
                use_full_kv_inputs=False,
                update_kv_cache=False,
            ),
            dict(
                testcase_name="updates_kv_cache",
                q=q,
                k=k,
                page_indices=metadata["page_indices"],
                use_full_kv_inputs=True,
                update_kv_cache=True,
            ),
            dict(
                testcase_name="non_integer_pcp_ratio",
                q=q,
                k=jnp.zeros((1000, 1, 128), dtype=jnp.bfloat16),
                page_indices=metadata["page_indices"],
                use_full_kv_inputs=True,
                update_kv_cache=False,
            ),
            dict(
                testcase_name="pcp_size_too_small_for_internal_coalesce",
                q=q,
                k=jnp.zeros((256, 1, 128), dtype=jnp.bfloat16),
                page_indices=metadata["page_indices"],
                use_full_kv_inputs=True,
                update_kv_cache=False,
            ),
            dict(
                testcase_name="invalid_page_table_shape",
                q=q,
                k=k,
                page_indices=metadata["page_indices"][:-1],
                use_full_kv_inputs=True,
                update_kv_cache=False,
            ),
            dict(
                testcase_name="non_all_prefill_distribution",
                q=q,
                k=k,
                page_indices=metadata["page_indices"],
                distribution=jnp.asarray([1, 1, 8], dtype=jnp.int32),
                use_full_kv_inputs=True,
                update_kv_cache=False,
            ),
        ]

        for case in guard_cases:
            with self.subTest(case["testcase_name"]):
                result = _maybe_coalesce_pcp_pseudo_sequences(
                    case["q"],
                    case["k"],
                    metadata["kv_lens"],
                    case["page_indices"],
                    metadata["cu_q_lens"],
                    case.get("distribution", metadata["distribution"]),
                    metadata["q_start_offsets"],
                    metadata["cu_k_lens"],
                    block_sizes,
                    use_full_kv_inputs=case["use_full_kv_inputs"],
                    update_kv_cache=case["update_kv_cache"],
                )
                (kv_lens, page_indices, cu_q_lens, distribution,
                 q_start_offsets, cu_k_lens, q_chunk_size, q_chunk_stride,
                 coalesced_blocks) = result
                self.assertEqual(q_chunk_size, 0)
                self.assertEqual(q_chunk_stride, 0)
                self.assertEqual(coalesced_blocks, block_sizes)
                self.assertArraysEqual(kv_lens, metadata["kv_lens"])
                self.assertArraysEqual(page_indices, case["page_indices"])
                self.assertArraysEqual(cu_q_lens, metadata["cu_q_lens"])
                self.assertArraysEqual(
                    distribution,
                    case.get("distribution", metadata["distribution"]))
                self.assertArraysEqual(q_start_offsets,
                                       metadata["q_start_offsets"])
                self.assertArraysEqual(cu_k_lens, metadata["cu_k_lens"])

        nonuniform_q_lens = np.full((metadata["kv_lens"].shape[0], ),
                                    16,
                                    dtype=np.int32)
        nonuniform_q_lens[0] = 8
        nonuniform_q_lens[1] = 24
        nonuniform_cu_q_lens = jnp.asarray(np.pad(
            np.cumsum(nonuniform_q_lens, dtype=np.int32), (1, 0)),
                                           dtype=jnp.int32)
        result = _maybe_coalesce_pcp_pseudo_sequences(
            q,
            k,
            metadata["kv_lens"],
            metadata["page_indices"],
            nonuniform_cu_q_lens,
            metadata["distribution"],
            metadata["q_start_offsets"],
            metadata["cu_k_lens"],
            block_sizes,
            use_full_kv_inputs=True,
            update_kv_cache=False,
        )
        self.assertEqual(result[6], 0)
        self.assertEqual(result[7], 0)
        self.assertEqual(result[8], block_sizes)

        metadata_10_chunks = _build_pcp_interleaved_metadata(
            seq_lens=[1280],
            pcp_size=8,
            interleave_size=16,
            pcp_rank=0,
            page_size=16,
        )
        result = _maybe_coalesce_pcp_pseudo_sequences(
            jnp.zeros((160, 4, 128), dtype=jnp.bfloat16),
            jnp.zeros((1280, 1, 128), dtype=jnp.bfloat16),
            metadata_10_chunks["kv_lens"],
            metadata_10_chunks["page_indices"],
            metadata_10_chunks["cu_q_lens"],
            metadata_10_chunks["distribution"],
            metadata_10_chunks["q_start_offsets"],
            metadata_10_chunks["cu_k_lens"],
            block_sizes,
            use_full_kv_inputs=True,
            update_kv_cache=False,
        )
        self.assertEqual(result[6], 0)
        self.assertEqual(result[7], 0)
        self.assertEqual(result[8], block_sizes)

    def test_coalesce_pcp_metadata_skips_when_q_lens_are_not_concrete(self):
        metadata = _build_pcp_interleaved_metadata(
            seq_lens=[882],
            pcp_size=8,
            interleave_size=16,
            pcp_rank=7,
            page_size=16,
        )
        block_sizes = configs.BlockSizes(
            bq_sz=256,
            bq_c_sz=64,
            bkv_sz=256,
            batch_size=1,
            n_buffer=2,
        )

        with patch.object(jax,
                          "device_get",
                          side_effect=TypeError("traced metadata")):
            result = _maybe_coalesce_pcp_pseudo_sequences(
                jnp.zeros((128, 4, 128), dtype=jnp.bfloat16),
                jnp.zeros((1024, 1, 128), dtype=jnp.bfloat16),
                metadata["kv_lens"],
                metadata["page_indices"],
                metadata["cu_q_lens"],
                metadata["distribution"],
                metadata["q_start_offsets"],
                metadata["cu_k_lens"],
                block_sizes,
                use_full_kv_inputs=True,
                update_kv_cache=False,
            )

        self.assertEqual(result[6], 0)
        self.assertEqual(result[7], 0)
        self.assertEqual(result[8], block_sizes)
        self.assertArraysEqual(result[0], metadata["kv_lens"])
        self.assertArraysEqual(result[2], metadata["cu_q_lens"])
        self.assertArraysEqual(result[4], metadata["q_start_offsets"])

    def _run_local_q_full_kv_case(self, q_lens, kv_lens, q_start_offsets,
                                  dtype):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(1234)
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        max_num_seqs = 8
        num_seqs = len(q_lens)

        total_q = sum(q_lens)
        total_kv = sum(kv_lens)
        q = jnp.array(rng.normal(size=(total_q, num_q_heads, head_dim)),
                      dtype=dtype)
        k = jnp.array(rng.normal(size=(total_kv, num_kv_heads, head_dim)),
                      dtype=dtype)
        v = jnp.array(rng.normal(size=(total_kv, num_kv_heads, head_dim)),
                      dtype=dtype)

        cu_q_lens = jnp.array([0] + list(np.cumsum(q_lens)), dtype=jnp.int32)
        cu_k_lens = jnp.array([0] + list(np.cumsum(kv_lens)), dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
        cu_k_lens = jnp.pad(cu_k_lens,
                            (0, max_num_seqs + 1 - cu_k_lens.shape[0]))
        kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)
        kv_lens_arr = jnp.pad(kv_lens_arr,
                              (0, max_num_seqs - kv_lens_arr.shape[0]))
        q_start_offsets_arr = jnp.array(q_start_offsets, dtype=jnp.int32)
        q_start_offsets_arr = jnp.pad(
            q_start_offsets_arr,
            (0, max_num_seqs - q_start_offsets_arr.shape[0]))

        pages_per_seq = max(
            (kv_len + page_size - 1) // page_size for kv_len in kv_lens)
        page_indices = []
        next_page = 0
        for kv_len in kv_lens:
            num_pages = (kv_len + page_size - 1) // page_size
            indices = np.arange(next_page,
                                next_page + num_pages,
                                dtype=np.int32)
            next_page += num_pages
            indices = np.pad(indices, (0, pages_per_seq - num_pages))
            page_indices.append(indices)
        while len(page_indices) < max_num_seqs:
            page_indices.append(np.zeros((pages_per_seq, ), dtype=np.int32))
        page_indices = jnp.array(np.stack(page_indices).reshape(-1),
                                 dtype=jnp.int32)
        distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

        kv_cache_shape = get_kv_cache_shape(
            total_num_pages=max(next_page, 1),
            page_size=page_size,
            actual_num_kv_heads=num_kv_heads,
            actual_head_dim=head_dim,
            kv_dtype=dtype,
        )
        kv_cache = jnp.zeros(kv_cache_shape, dtype=dtype)

        block_sizes = configs.BlockSizes(
            bq_sz=32,
            bq_c_sz=16,
            bkv_sz=64,
            batch_size=1,
            n_buffer=2,
        )
        expected = _ref_local_q_full_kv_attention(q, k, v, cu_q_lens,
                                                  cu_k_lens, kv_lens_arr,
                                                  q_start_offsets_arr,
                                                  num_seqs).astype(dtype)
        actual, _ = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens_arr,
            page_indices,
            cu_q_lens,
            distribution,
            prefill_block_sizes=block_sizes,
            decode_block_sizes=block_sizes,
            out_dtype=dtype,
            update_kv_cache=False,
            q_start_offsets=q_start_offsets_arr,
            cu_k_lens=cu_k_lens,
        )

        self._assert_rpa_close(actual, expected, dtype)

    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def test_local_q_full_kv_single_sequence(self, dtype):
        self._run_local_q_full_kv_case(
            q_lens=[64],
            kv_lens=[192],
            q_start_offsets=[64],
            dtype=dtype,
        )

    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def test_local_q_full_kv_multiple_sequences(self, dtype):
        self._run_local_q_full_kv_case(
            q_lens=[32, 64],
            kv_lens=[128, 192],
            q_start_offsets=[32, 96],
            dtype=dtype,
        )

    def test_return_lse_local_q_full_kv_single_sequence(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(1234)
        dtype = jnp.float32
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        max_num_seqs = 8
        q_lens = [32]
        kv_lens = [96]
        q_start_offsets = [32]

        q = jnp.array(rng.normal(size=(32, num_q_heads, head_dim)),
                      dtype=dtype)
        k = jnp.array(rng.normal(size=(96, num_kv_heads, head_dim)),
                      dtype=dtype)
        v = jnp.array(rng.normal(size=(96, num_kv_heads, head_dim)),
                      dtype=dtype)

        cu_q_lens = jnp.array([0, 32], dtype=jnp.int32)
        cu_k_lens = jnp.array([0, 96], dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
        cu_k_lens = jnp.pad(cu_k_lens,
                            (0, max_num_seqs + 1 - cu_k_lens.shape[0]))
        kv_lens_arr = jnp.pad(jnp.array(kv_lens, dtype=jnp.int32),
                              (0, max_num_seqs - len(kv_lens)))
        q_start_offsets_arr = jnp.pad(
            jnp.array(q_start_offsets, dtype=jnp.int32),
            (0, max_num_seqs - len(q_start_offsets)))

        pages_per_seq = 6
        page_indices = np.zeros((max_num_seqs, pages_per_seq), dtype=np.int32)
        page_indices[0] = np.arange(pages_per_seq, dtype=np.int32)
        page_indices = jnp.array(page_indices.reshape(-1), dtype=jnp.int32)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
        kv_cache = jnp.zeros(
            get_kv_cache_shape(
                total_num_pages=pages_per_seq,
                page_size=page_size,
                actual_num_kv_heads=num_kv_heads,
                actual_head_dim=head_dim,
                kv_dtype=dtype,
            ),
            dtype=dtype,
        )
        block_sizes = configs.BlockSizes(
            bq_sz=32,
            bq_c_sz=16,
            bkv_sz=64,
            batch_size=1,
            n_buffer=2,
        )
        expected, expected_lse = _ref_local_q_full_kv_attention_and_lse(
            q, k, v, cu_q_lens, cu_k_lens, kv_lens_arr, q_start_offsets_arr, 1)

        (actual, actual_lse), _ = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens_arr,
            page_indices,
            cu_q_lens,
            distribution,
            prefill_block_sizes=block_sizes,
            decode_block_sizes=block_sizes,
            out_dtype=dtype,
            update_kv_cache=False,
            q_start_offsets=q_start_offsets_arr,
            cu_k_lens=cu_k_lens,
            return_lse=True,
        )

        self._assert_rpa_close(actual, expected, dtype)
        self.assertAllClose(actual_lse, expected_lse, atol=1e-2, rtol=1e-2)

    def test_return_lse_empty_local_kv_shard(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(1234)
        dtype = jnp.float32
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        max_num_seqs = 8

        q = jnp.array(rng.normal(size=(1, num_q_heads, head_dim)), dtype=dtype)
        k = jnp.zeros((1, num_kv_heads, head_dim), dtype=dtype)
        v = jnp.zeros((1, num_kv_heads, head_dim), dtype=dtype)
        kv_cache = jnp.zeros(
            get_kv_cache_shape(
                total_num_pages=1,
                page_size=page_size,
                actual_num_kv_heads=num_kv_heads,
                actual_head_dim=head_dim,
                kv_dtype=dtype,
            ),
            dtype=dtype,
        )
        kv_lens = jnp.zeros((max_num_seqs, ), dtype=jnp.int32)
        page_indices = jnp.zeros((max_num_seqs, ), dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 1] + [1] * (max_num_seqs - 1),
                              dtype=jnp.int32)
        distribution = jnp.array([1, 1, 1], dtype=jnp.int32)

        (actual, actual_lse), _ = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            prefill_block_sizes=configs.BlockSizes(
                bq_sz=16,
                bq_c_sz=16,
                bkv_sz=16,
                batch_size=1,
                n_buffer=2,
            ),
            decode_block_sizes=configs.BlockSizes(
                bq_sz=16,
                bq_c_sz=16,
                bkv_sz=16,
                batch_size=1,
                n_buffer=2,
            ),
            out_dtype=dtype,
            update_kv_cache=False,
            q_start_offsets=jnp.zeros((max_num_seqs, ), dtype=jnp.int32),
            return_lse=True,
        )

        self.assertAllClose(actual, jnp.zeros_like(actual), atol=0, rtol=0)
        self.assertArraysEqual(actual_lse, jnp.full_like(actual_lse, -jnp.inf))

    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def test_local_q_full_kv_pseudo_sequences_share_full_kv(self, dtype):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(1234)
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        max_num_seqs = 8
        q_lens = [16, 16]
        kv_lens = [64, 64]
        q_start_offsets = [0, 32]
        num_seqs = len(q_lens)

        k = jnp.array(rng.normal(size=(64, num_kv_heads, head_dim)),
                      dtype=dtype)
        v = jnp.array(rng.normal(size=(64, num_kv_heads, head_dim)),
                      dtype=dtype)
        q_full = jnp.array(rng.normal(size=(64, num_q_heads, head_dim)),
                           dtype=dtype)
        q = jnp.concatenate([q_full[0:16], q_full[32:48]], axis=0)

        cu_q_lens = jnp.array([0] + list(np.cumsum(q_lens)), dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
        cu_k_lens = jnp.array([0, 0, 64], dtype=jnp.int32)
        cu_k_lens = jnp.pad(cu_k_lens,
                            (0, max_num_seqs + 1 - cu_k_lens.shape[0]))
        kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)
        kv_lens_arr = jnp.pad(kv_lens_arr,
                              (0, max_num_seqs - kv_lens_arr.shape[0]))
        q_start_offsets_arr = jnp.array(q_start_offsets, dtype=jnp.int32)
        q_start_offsets_arr = jnp.pad(
            q_start_offsets_arr,
            (0, max_num_seqs - q_start_offsets_arr.shape[0]))

        pages_per_seq = 4
        page_indices = np.zeros((max_num_seqs, pages_per_seq), dtype=np.int32)
        page_indices[0] = np.arange(pages_per_seq, dtype=np.int32)
        page_indices[1] = np.arange(pages_per_seq, dtype=np.int32)
        page_indices = jnp.array(page_indices.reshape(-1), dtype=jnp.int32)
        distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

        kv_cache = jnp.zeros(
            get_kv_cache_shape(
                total_num_pages=pages_per_seq,
                page_size=page_size,
                actual_num_kv_heads=num_kv_heads,
                actual_head_dim=head_dim,
                kv_dtype=dtype,
            ),
            dtype=dtype,
        )
        block_sizes = configs.BlockSizes(
            bq_sz=32,
            bq_c_sz=16,
            bkv_sz=64,
            batch_size=1,
            n_buffer=2,
        )
        expected = _ref_local_q_full_kv_attention(q, k, v, cu_q_lens,
                                                  cu_k_lens, kv_lens_arr,
                                                  q_start_offsets_arr,
                                                  num_seqs).astype(dtype)
        actual, _ = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens_arr,
            page_indices,
            cu_q_lens,
            distribution,
            prefill_block_sizes=block_sizes,
            decode_block_sizes=block_sizes,
            out_dtype=dtype,
            update_kv_cache=False,
            q_start_offsets=q_start_offsets_arr,
            cu_k_lens=cu_k_lens,
        )

        self._assert_rpa_close(actual, expected, dtype)

    def _run_pcp_interleaved_local_q_full_kv_case(self, *, seq_lens, pcp_size,
                                                  interleave_size, pcp_rank,
                                                  dtype, block_sizes):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(1234)
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        total_kv = sum(seq_lens)

        q_full = jnp.asarray(rng.normal(size=(total_kv, num_q_heads,
                                              head_dim)),
                             dtype=dtype)
        k = jnp.asarray(rng.normal(size=(total_kv, num_kv_heads, head_dim)),
                        dtype=dtype)
        v = jnp.asarray(rng.normal(size=(total_kv, num_kv_heads, head_dim)),
                        dtype=dtype)
        q = _local_pcp_query(q_full, seq_lens, pcp_size, interleave_size,
                             pcp_rank)
        metadata = _build_pcp_interleaved_metadata(seq_lens, pcp_size,
                                                   interleave_size, pcp_rank,
                                                   page_size)

        kv_cache = jnp.zeros(
            get_kv_cache_shape(
                total_num_pages=metadata["total_pages"],
                page_size=page_size,
                actual_num_kv_heads=num_kv_heads,
                actual_head_dim=head_dim,
                kv_dtype=dtype,
            ),
            dtype=dtype,
        )
        num_pseudo_seqs = int(np.asarray(metadata["distribution"])[2])
        expected = _ref_local_q_full_kv_attention(
            q,
            k,
            v,
            metadata["cu_q_lens"],
            metadata["cu_k_lens"],
            metadata["kv_lens"],
            metadata["q_start_offsets"],
            num_pseudo_seqs,
        ).astype(dtype)

        actual, _ = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            metadata["kv_lens"],
            metadata["page_indices"],
            metadata["cu_q_lens"],
            metadata["distribution"],
            prefill_block_sizes=block_sizes,
            decode_block_sizes=block_sizes,
            out_dtype=dtype,
            update_kv_cache=False,
            q_start_offsets=metadata["q_start_offsets"],
            cu_k_lens=metadata["cu_k_lens"],
        )

        self._assert_rpa_close(actual, expected, dtype)

    def test_pcp_coalesced_interleave16_matches_reference_bfloat16(self):
        self._run_pcp_interleaved_local_q_full_kv_case(
            seq_lens=[1024],
            pcp_size=8,
            interleave_size=16,
            pcp_rank=3,
            dtype=jnp.bfloat16,
            block_sizes=configs.BlockSizes(
                bq_sz=256,
                bq_c_sz=64,
                bkv_sz=256,
                batch_size=1,
                n_buffer=2,
            ),
        )

    def test_pcp_coalesced_multi_sequence_nonzero_rank_matches_reference(self):
        self._run_pcp_interleaved_local_q_full_kv_case(
            seq_lens=[256, 256],
            pcp_size=4,
            interleave_size=8,
            pcp_rank=2,
            dtype=jnp.float32,
            block_sizes=configs.BlockSizes(
                bq_sz=64,
                bq_c_sz=32,
                bkv_sz=64,
                batch_size=1,
                n_buffer=2,
            ),
        )

    def test_suffix_q_existing_interface(self):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(1234)
        dtype = jnp.float32
        q_len = 64
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 128
        page_size = 16
        max_num_seqs = 8

        q = jnp.array(rng.normal(size=(q_len, num_q_heads, head_dim)),
                      dtype=dtype)
        k = jnp.array(rng.normal(size=(q_len, num_kv_heads, head_dim)),
                      dtype=dtype)
        v = jnp.array(rng.normal(size=(q_len, num_kv_heads, head_dim)),
                      dtype=dtype)
        cu_q_lens = jnp.array([0, q_len], dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.array([q_len], dtype=jnp.int32)
        kv_lens = jnp.pad(kv_lens, (0, max_num_seqs - kv_lens.shape[0]))

        pages_per_seq = q_len // page_size
        page_indices = np.zeros((max_num_seqs, pages_per_seq), dtype=np.int32)
        page_indices[0] = np.arange(pages_per_seq, dtype=np.int32)
        page_indices = jnp.array(page_indices.reshape(-1), dtype=jnp.int32)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
        kv_cache = jnp.zeros(
            get_kv_cache_shape(
                total_num_pages=pages_per_seq,
                page_size=page_size,
                actual_num_kv_heads=num_kv_heads,
                actual_head_dim=head_dim,
                kv_dtype=dtype,
            ),
            dtype=dtype,
        )
        expected = _ref_local_q_full_kv_attention(
            q, k, v, cu_q_lens, cu_q_lens, kv_lens,
            jnp.zeros((max_num_seqs, ), dtype=jnp.int32), 1)
        block_sizes = configs.BlockSizes(
            bq_sz=32,
            bq_c_sz=16,
            bkv_sz=64,
            batch_size=1,
            n_buffer=2,
        )

        actual, _ = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            prefill_block_sizes=block_sizes,
            decode_block_sizes=block_sizes,
            out_dtype=dtype,
        )

        self._assert_rpa_close(actual, expected, dtype)


if __name__ == "__main__":
    absltest.main()
