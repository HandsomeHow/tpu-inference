# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from tpu_inference.layers.common.attention_interface import (
    _make_pcp_interleaved_metadata, _make_pcp_interleaved_token_indices,
    _pcp_lse_merge_weight, _update_local_paged_kv_cache, attention,
    compute_pcp_local_mapping, materialize_pcp_kv_for_decode, mla_attention,
    pcp_lse_merge, sharded_ragged_paged_attention)
from tpu_inference.layers.common.attention_metadata import (AttentionMetadata,
                                                            PcpMode)
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.runner.kv_cache import get_kv_cache_shape_with_mesh

# ---- Test Configuration & Constants ----

# Total number of tokens across all sequences in the batch
TOTAL_TOKENS = 10
# Number of sequences in the batch
NUM_SEQS = 2
# Padded maximum number of sequences
MAX_NUM_SEQS = 4
# Number of attention heads (Query)
NUM_HEADS = 8
# Number of attention heads (Key/Value) - for Grouped-Query Attention
NUM_KV_HEADS = 4
# Total number of blocks in the KV cache
NUM_BLOCKS = 32
# Number of tokens per block
BLOCK_SIZE = 16
# Maximum number of blocks a single sequence can occupy
MAX_BLOCKS_PER_SEQ = 8


def _reference_pcp_lse_merge(partial_out, partial_lse):
    max_lse = jnp.max(partial_lse, axis=0)
    valid = max_lse != -jnp.inf
    safe_diffs = jnp.where(valid[None], partial_lse - max_lse[None], 0.0)
    weights = jnp.exp(safe_diffs)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=0), 1e-30)
    weights = jnp.where(valid[None], weights, 0.0)
    return jnp.sum(partial_out.astype(jnp.float32) * weights[..., None],
                   axis=0)


def test_pcp_lse_merge_weight_handles_empty_and_padding_rows():
    all_lses = jnp.array([
        [[0.0, -jnp.inf], [1.0, 2.0], [-jnp.inf, -jnp.inf]],
        [[jnp.log(3.0), 0.0], [-jnp.inf, 3.0], [-jnp.inf, -jnp.inf]],
    ],
                         dtype=jnp.float32)

    rank0_weight = _pcp_lse_merge_weight(all_lses[0], all_lses)
    rank1_weight = _pcp_lse_merge_weight(all_lses[1], all_lses)

    expected_rank0 = jnp.array([[0.25, 0.0], [1.0, 1.0 /
                                              (1.0 + jnp.e)], [0.0, 0.0]],
                               dtype=jnp.float32)
    expected_rank1 = jnp.array([[0.75, 1.0], [0.0, jnp.e /
                                              (1.0 + jnp.e)], [0.0, 0.0]],
                               dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(rank0_weight),
                               np.asarray(expected_rank0),
                               rtol=1e-6,
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(rank1_weight),
                               np.asarray(expected_rank1),
                               rtol=1e-6,
                               atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pcp_lse_merge_matches_reference_with_real_collectives(dtype):
    pcp_size = 2
    if len(jax.local_devices()) < pcp_size:
        pytest.skip(f"requires at least {pcp_size} local devices")

    partial_lse = jnp.array([
        [[0.0, -jnp.inf], [1.0, 2.0], [-jnp.inf, -jnp.inf]],
        [[jnp.log(3.0), 0.0], [-jnp.inf, 3.0], [-jnp.inf, -jnp.inf]],
    ],
                            dtype=jnp.float32)
    partial_out = jnp.arange(pcp_size * 3 * 2 * 4,
                             dtype=jnp.float32).reshape(pcp_size, 3, 2,
                                                        4).astype(dtype)
    expected = _reference_pcp_lse_merge(partial_out, partial_lse).astype(dtype)

    mesh = Mesh(np.array(jax.local_devices()[:pcp_size]), ("pcp", ))

    def merge_one_rank(out_shard, lse_shard):
        return pcp_lse_merge(out_shard[0], lse_shard[0], "pcp")

    merged = jax.jit(
        jax.shard_map(
            merge_one_rank,
            mesh=mesh,
            in_specs=(P("pcp", None, None, None), P("pcp", None, None)),
            out_specs=P(None, None, None),
            check_vma=False,
        ))(partial_out, partial_lse)

    np.testing.assert_allclose(np.asarray(merged),
                               np.asarray(expected),
                               rtol=2e-2 if dtype == jnp.bfloat16 else 1e-6,
                               atol=2e-2 if dtype == jnp.bfloat16 else 1e-6)
    assert not np.isnan(np.asarray(merged)).any()
    np.testing.assert_array_equal(
        np.asarray(merged[2]), np.zeros((2, 4),
                                        dtype=np.asarray(merged).dtype))


def _reference_vllm_cp_slot_mapping(positions, token_req_indices, block_tables,
                                    block_size, cp_size, cp_rank,
                                    interleave_size):
    virtual_block_size = block_size * cp_size
    block_indices = positions // virtual_block_size
    block_numbers = block_tables[token_req_indices, block_indices]
    virtual_offsets = positions - block_indices * virtual_block_size
    is_local = ((virtual_offsets // interleave_size) % cp_size) == cp_rank
    local_offsets = ((virtual_offsets //
                      (cp_size * interleave_size)) * interleave_size +
                     (virtual_offsets % interleave_size))
    slot_ids = block_numbers * block_size + local_offsets
    return is_local, np.where(is_local, slot_ids, -1).astype(np.int32)


def test_compute_pcp_local_mapping_matches_vllm_cp_semantics_batch_requests():
    block_size = 4
    cp_size = 3
    interleave_size = 2
    local_num_blocks = 16
    block_tables = np.array(
        [
            [5, 7, 9],
            [1, 3, 4],
            [10, 11, 12],
        ],
        dtype=np.int32,
    )
    positions = np.array(
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 23, 0, 5, 12, 17, 24, 25],
        dtype=np.int32,
    )
    token_req_indices = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
        dtype=np.int32,
    )

    for cp_rank in range(cp_size):
        actual_is_local, actual_slot_ids = compute_pcp_local_mapping(
            jnp.asarray(positions),
            jnp.asarray(token_req_indices),
            jnp.asarray(block_tables),
            block_size=block_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            interleave_size=interleave_size,
        )
        expected_is_local, expected_slot_ids = _reference_vllm_cp_slot_mapping(
            positions,
            token_req_indices,
            block_tables,
            block_size,
            cp_size,
            cp_rank,
            interleave_size,
        )
        np.testing.assert_array_equal(np.asarray(actual_is_local),
                                      expected_is_local)
        np.testing.assert_array_equal(np.asarray(actual_slot_ids),
                                      expected_slot_ids)
        local_slot_ids = np.asarray(actual_slot_ids)[expected_is_local]
        assert np.all(local_slot_ids < local_num_blocks * block_size)

    _, req0_slot = compute_pcp_local_mapping(
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.asarray(block_tables),
        block_size=block_size,
        cp_size=cp_size,
        cp_rank=0,
        interleave_size=interleave_size,
    )
    _, req1_slot = compute_pcp_local_mapping(
        jnp.array([0], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.asarray(block_tables),
        block_size=block_size,
        cp_size=cp_size,
        cp_rank=0,
        interleave_size=interleave_size,
    )
    assert int(req0_slot[0]) == 5 * block_size
    assert int(req1_slot[0]) == 1 * block_size


@pytest.mark.parametrize(
    ("block_size", "cp_size", "cp_rank", "interleave_size"),
    [(0, 2, 0, 1), (4, 0, 0, 1), (4, 2, 2, 1), (4, 2, 0, 0)],
)
def test_compute_pcp_local_mapping_rejects_invalid_static_args(
        block_size, cp_size, cp_rank, interleave_size):
    with pytest.raises(ValueError):
        compute_pcp_local_mapping(
            jnp.array([0], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([[0]], dtype=jnp.int32),
            block_size=block_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            interleave_size=interleave_size,
        )


def _pcp_decode_source_location(pos, page_size, pcp_size, interleave_size):
    virtual_block_size = page_size * pcp_size
    virtual_block = pos // virtual_block_size
    virtual_offset = pos % virtual_block_size
    src_rank = (virtual_offset // interleave_size) % pcp_size
    src_offset = ((virtual_offset //
                   (pcp_size * interleave_size)) * interleave_size +
                  (virtual_offset % interleave_size))
    return src_rank, virtual_block, src_offset


def _fake_compact_pcp_all_gather(gathered, source_block_tables):
    source_pages = source_block_tables.reshape(-1)

    def fake_all_gather(x, axis_name, axis, tiled):
        assert axis_name == "pcp"
        assert axis == 0
        assert tiled is False
        assert x.shape[0] == source_pages.size
        assert x.shape[0] < gathered.shape[1]
        return jnp.asarray(gathered[:, source_pages])

    return fake_all_gather


def test_materialize_pcp_kv_for_decode_repacks_tokens(monkeypatch):
    page_size = 4
    pcp_size = 2
    interleave_size = 2
    local_pages = 10
    source_block_tables = np.array(
        [
            [7, 3],
            [5, 9],
            [0, 0],
        ],
        dtype=np.int32,
    )
    kv_lens = np.array([5, 10, 0], dtype=np.int32)
    gathered = np.zeros((pcp_size, local_pages, page_size, 2),
                        dtype=np.int32)
    expected_tokens = {}

    for req_idx, seq_len in enumerate(kv_lens):
        for pos in range(int(seq_len)):
            src_rank, virtual_block, src_offset = (
                _pcp_decode_source_location(pos, page_size, pcp_size,
                                            interleave_size))
            src_page = source_block_tables[req_idx, virtual_block]
            value = 1000 * (req_idx + 1) + pos
            gathered[src_rank, src_page, src_offset] = [value, -value]
            expected_tokens[(req_idx, pos)] = [value, -value]

    monkeypatch.setattr(
        "jax.lax.all_gather",
        _fake_compact_pcp_all_gather(gathered, source_block_tables),
    )
    full_kv_cache, full_kv_lens, full_page_indices = (
        materialize_pcp_kv_for_decode(
            jnp.zeros((local_pages, page_size, 2), dtype=jnp.int32),
            jnp.asarray(kv_lens),
            jnp.asarray(source_block_tables),
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            pcp_axis_name="pcp",
        ))

    standard_pages_per_req = source_block_tables.shape[1] * pcp_size
    full_kv_cache_np = np.asarray(full_kv_cache)
    for (req_idx, pos), expected in expected_tokens.items():
        dst_page = req_idx * standard_pages_per_req + pos // page_size
        dst_offset = pos % page_size
        np.testing.assert_array_equal(full_kv_cache_np[dst_page, dst_offset],
                                      np.array(expected, dtype=np.int32))

    np.testing.assert_array_equal(
        full_kv_cache_np[1, 1:],
        np.zeros((3, 2), dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(full_kv_lens), kv_lens)
    np.testing.assert_array_equal(
        np.asarray(full_page_indices),
        np.arange(source_block_tables.shape[0] * standard_pages_per_req,
                  dtype=np.int32),
    )


def test_materialize_pcp_kv_for_decode_repacks_pcp8_virtual_blocks(
        monkeypatch):
    page_size = 16
    pcp_size = 8
    interleave_size = 4
    local_pages = 32
    source_block_tables = np.array(
        [
            [17, 3],
            [5, 29],
            [11, 23],
            [0, 0],
        ],
        dtype=np.int32,
    )
    kv_lens = np.array([6, 31, 130, 0], dtype=np.int32)
    gathered = np.full((pcp_size, local_pages, page_size, 3),
                       -999,
                       dtype=np.int32)
    expected_tokens = {}
    source_ranks_seen = set()

    for req_idx, seq_len in enumerate(kv_lens):
        for pos in range(int(seq_len)):
            src_rank, virtual_block, src_offset = (
                _pcp_decode_source_location(pos, page_size, pcp_size,
                                            interleave_size))
            source_ranks_seen.add(src_rank)
            src_page = source_block_tables[req_idx, virtual_block]
            value = 10000 * (req_idx + 1) + pos
            gathered[src_rank, src_page, src_offset] = [
                value,
                src_rank,
                src_page,
            ]
            expected_tokens[(req_idx, pos)] = [value, src_rank, src_page]

    assert source_ranks_seen == set(range(pcp_size))
    last_rank, _, _ = _pcp_decode_source_location(kv_lens[0] - 1, page_size,
                                                  pcp_size, interleave_size)
    assert last_rank != 0

    monkeypatch.setattr(
        "jax.lax.all_gather",
        _fake_compact_pcp_all_gather(gathered, source_block_tables),
    )
    full_kv_cache, full_kv_lens, full_page_indices = (
        materialize_pcp_kv_for_decode(
            jnp.zeros((local_pages, page_size, 3), dtype=jnp.int32),
            jnp.asarray(kv_lens),
            jnp.asarray(source_block_tables),
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            pcp_axis_name="pcp",
        ))

    standard_pages_per_req = source_block_tables.shape[1] * pcp_size
    full_kv_cache_np = np.asarray(full_kv_cache)
    for (req_idx, pos), expected in expected_tokens.items():
        dst_page = req_idx * standard_pages_per_req + pos // page_size
        dst_offset = pos % page_size
        np.testing.assert_array_equal(full_kv_cache_np[dst_page, dst_offset],
                                      np.array(expected, dtype=np.int32))

    req0_padding_page = 0
    np.testing.assert_array_equal(
        full_kv_cache_np[req0_padding_page, kv_lens[0]:],
        np.zeros((page_size - kv_lens[0], 3), dtype=np.int32),
    )
    req3_start_page = 3 * standard_pages_per_req
    np.testing.assert_array_equal(
        full_kv_cache_np[req3_start_page:req3_start_page +
                         standard_pages_per_req],
        np.zeros((standard_pages_per_req, page_size, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(full_kv_lens), kv_lens)
    np.testing.assert_array_equal(
        np.asarray(full_page_indices),
        np.arange(source_block_tables.shape[0] * standard_pages_per_req,
                  dtype=np.int32),
    )


def test_pcp_decode_update_then_materialize_multiple_steps(monkeypatch):
    page_size = 16
    pcp_size = 8
    interleave_size = 4
    local_pages = 4
    source_page = 2
    source_block_tables = jnp.array([[source_page]], dtype=jnp.int32)
    gathered = np.zeros((pcp_size, local_pages, page_size, 1, 1, 1),
                        dtype=np.float32)

    for pos in range(4):
        src_rank, _, src_offset = _pcp_decode_source_location(
            pos, page_size, pcp_size, interleave_size)
        gathered[src_rank, source_page, src_offset, 0, 0, 0] = 1000 + pos

    def fake_prepare_inputs(q, k, v, q_dtype, kv_dtype):
        del v, q_dtype, kv_dtype
        return jnp.zeros_like(q), k.reshape(k.shape[0], 1, 1, 1)

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.batched_rpa_wrapper.prepare_inputs",
        fake_prepare_inputs,
    )
    monkeypatch.setattr(
        "jax.lax.all_gather",
        _fake_compact_pcp_all_gather(gathered, np.asarray(source_block_tables)),
    )

    for pos in range(4, 7):
        src_rank, _, src_offset = _pcp_decode_source_location(
            pos, page_size, pcp_size, interleave_size)
        assert src_rank == 1
        slot_id = source_page * page_size + src_offset
        updated_rank_cache = _update_local_paged_kv_cache(
            jnp.asarray(gathered[src_rank]),
            jnp.array([[[1000 + pos]]], dtype=jnp.float32),
            jnp.zeros((1, 1, 1), dtype=jnp.float32),
            jnp.array([slot_id], dtype=jnp.int32),
        )
        gathered[src_rank] = np.asarray(updated_rank_cache)

        full_kv_cache, full_kv_lens, _ = materialize_pcp_kv_for_decode(
            jnp.asarray(gathered[0]),
            jnp.array([pos + 1], dtype=jnp.int32),
            source_block_tables,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            pcp_axis_name="pcp",
        )

        expected = np.arange(1000, 1000 + pos + 1, dtype=np.float32)
        np.testing.assert_array_equal(
            np.asarray(full_kv_cache)[0, :pos + 1, 0, 0, 0],
            expected,
        )
        np.testing.assert_array_equal(np.asarray(full_kv_lens),
                                      np.array([pos + 1], dtype=np.int32))


@pytest.mark.parametrize(
    ("page_size", "pcp_size", "interleave_size", "match"),
    [
        (5, 2, 2, "page_size % interleave_size"),
        (4, 1, 2, "pcp_size > 1"),
        (4, 2, 0, "interleave_size > 0"),
    ],
)
def test_materialize_pcp_kv_for_decode_rejects_invalid_args(
        monkeypatch, page_size, pcp_size, interleave_size, match):

    def fake_all_gather(x, axis_name, axis, tiled):
        return jnp.zeros((2, 1, 4, 1), dtype=jnp.int32)

    monkeypatch.setattr("jax.lax.all_gather", fake_all_gather)
    with pytest.raises(ValueError, match=match):
        materialize_pcp_kv_for_decode(
            jnp.zeros((1, 4, 1), dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([[0]], dtype=jnp.int32),
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            pcp_axis_name="pcp",
        )


def test_materialize_pcp_kv_for_decode_rejects_gathered_pcp_size_mismatch(
        monkeypatch):

    def fake_all_gather(x, axis_name, axis, tiled):
        return jnp.zeros((3, 1, 4, 1), dtype=jnp.int32)

    monkeypatch.setattr("jax.lax.all_gather", fake_all_gather)
    with pytest.raises(ValueError, match="match pcp_size"):
        materialize_pcp_kv_for_decode(
            jnp.zeros((1, 4, 1), dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            jnp.array([[0]], dtype=jnp.int32),
            page_size=4,
            pcp_size=2,
            interleave_size=2,
            pcp_axis_name="pcp",
        )


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    # Create a mesh with available devices, useful for running on CPU/GPU/TPU
    # For this test, it will likely be a single CPU device.
    devices = np.array(jax.local_devices()[:1])
    if not devices.any():
        # Add a mock device if no devices are present (e.g., in a CI environment)
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


# ---- Test for `attention` ----


def _test_attention(monkeypatch, mesh, head_dim, use_sinks=False):
    """
    Tests the main `attention` function.

    Verifies that:
    1. It calls the `sharded_ragged_paged_attention` kernel with correct metadata.
    2. The final outputs (kv_cache and attention output) have the correct shapes.
    """
    # 1. Arrange

    # Create input tensors
    q_dtype = jnp.float32
    kv_dtype = jnp.float32
    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim), dtype=q_dtype)
    k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, head_dim), dtype=kv_dtype)
    v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, head_dim), dtype=kv_dtype)
    sinks = jnp.ones((NUM_HEADS, ), dtype=jnp.float32) if use_sinks else None

    kv_cache_shape = get_kv_cache_shape_with_mesh(
        mesh,
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        head_dim,
        kv_dtype,
    )
    kv_cache = jnp.zeros(kv_cache_shape, dtype=kv_dtype)

    # Mock ragged_paged_attention to return a tensor of the correct shape
    mock_paged_attn_kernel = MagicMock(return_value=(jnp.ones(
        (TOTAL_TOKENS, NUM_HEADS, head_dim)), kv_cache), )

    if head_dim == 64:
        monkeypatch.setattr(
            "tpu_inference.layers.common.attention_interface.ragged_paged_attention_hd64",
            mock_paged_attn_kernel,
        )
    else:
        monkeypatch.setattr(
            "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
            mock_paged_attn_kernel,
        )

    # Create AttentionMetadata
    attention_metadata = AttentionMetadata(
        input_positions=jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        block_tables=jnp.zeros((MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32),
    )

    # 2. Act
    final_kv_cache, output = attention(
        kv_cache=kv_cache,
        q=q,
        k=k,
        v=v,
        attention_metadata=attention_metadata,
        mesh=mesh,
        head_dim_original=head_dim,
        sinks=sinks,
    )

    # 3. Assert
    # Check that both mocked kernels were called
    mock_paged_attn_kernel.assert_called_once()

    # Check output shapes
    assert final_kv_cache.shape == kv_cache.shape
    assert output.shape == q.shape

    # Check that the output is the one from our mock
    assert jnp.all(output == 1.0)


def test_attention(monkeypatch, mesh):
    _test_attention(monkeypatch, mesh, 128)


def test_attention_hd64(monkeypatch, mesh):
    _test_attention(monkeypatch, mesh, 64)


def test_attention_sink(monkeypatch, mesh):
    _test_attention(monkeypatch, mesh, 64, True)


def test_attention_sink_no_64_raises_error(monkeypatch, mesh):
    with pytest.raises(
            NotImplementedError,
            match="Attention sink support is only available when head_dim==64"
    ):
        _test_attention(monkeypatch, mesh, 128, True)


# ---- Tests for `sharded_ragged_paged_attention` ----


@pytest.fixture
def gqa_mesh():
    """Provides a mock JAX mesh for GQA testing with tensor parallelism."""
    # This mesh has 8 devices for tensor parallelism over heads.
    # We create a 1x8 mesh for ('attn_data', 'attn_head')
    try:
        devices = np.array(jax.local_devices()[:1] * 4)
        if devices.size == 0:
            raise IndexError
    except IndexError:
        # Fails in environments with no devices
        devices = np.array([jax.devices("cpu")[0]] * 4)

    return Mesh(
        devices.reshape((1, 4)),
        (
            ShardingAxisName.ATTN_DATA,
            ShardingAxisName.ATTN_HEAD,
        ),
    )


def test_sharded_ragged_paged_attention_gqa_replication(monkeypatch, gqa_mesh):
    """
    Tests that K and V heads are correctly replicated for GQA in
    `sharded_ragged_paged_attention`.
    """
    # 1. Arrange
    tp_size = gqa_mesh.shape[ShardingAxisName.ATTN_HEAD]
    assert tp_size == 4
    num_kv_heads = 2  # num_kv_heads < tp_size and tp_size % num_kv_heads == 0
    head_dim = 128
    factor = tp_size // num_kv_heads

    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim))
    # Create K and V with values that can be checked after repeating
    k_content = jnp.arange(TOTAL_TOKENS * num_kv_heads * head_dim).reshape(
        (TOTAL_TOKENS, num_kv_heads, head_dim))
    v_content = -k_content
    k = k_content
    v = v_content

    # The actual shape of kv_cache does not matter as much since we mock the call
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim))

    # Other metadata, can be zero/empty for this test's purpose
    kv_lens = jnp.zeros((MAX_NUM_SEQS, ), dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                             dtype=jnp.int32)
    cu_q_lens = jnp.zeros((MAX_NUM_SEQS + 1, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)
    sm_scale = 1.0

    # Mock jax.shard_map to capture the arguments passed to its mapped function
    mock_shard_map_callable = MagicMock(return_value=(jnp.ones_like(q),
                                                      kv_cache))
    mock_shard_map = MagicMock(return_value=mock_shard_map_callable)
    monkeypatch.setattr("jax.shard_map", mock_shard_map)

    # 2. Act
    sharded_ragged_paged_attention(
        mesh=gqa_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=sm_scale,
    )

    # 3. Assert
    # Check that shard_map was called
    mock_shard_map.assert_called_once()
    # Check that the function returned by shard_map was called with arguments
    mock_shard_map_callable.assert_called_once()

    # Get the arguments passed to the jitted function inside shard_map
    call_args = mock_shard_map_callable.call_args[0]
    replicated_k = call_args[1]
    replicated_v = call_args[2]

    # Check shapes
    assert replicated_k.shape[1] == tp_size
    assert replicated_v.shape[1] == tp_size
    assert replicated_k.shape[1] == k.shape[1] * factor
    assert replicated_v.shape[1] == v.shape[1] * factor

    # Check content of replicated K
    expected_k = jnp.repeat(k_content, factor, axis=1)
    assert jnp.array_equal(replicated_k, expected_k)

    # Check content of replicated V
    expected_v = jnp.repeat(v_content, factor, axis=1)
    assert jnp.array_equal(replicated_v, expected_v)


def test_sharded_rpa_can_replicate_normal_attention_over_pcp_axis(monkeypatch):
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    devices = np.array(jax.local_devices()[:1] * 2).reshape((1, 1, 1, 1, 1, 2))
    pcp_mesh = Mesh(
        devices,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "pcp"))
    q = jnp.ones((16, NUM_HEADS, 128), dtype=jnp.float32)
    k = jnp.ones((16, NUM_KV_HEADS, 128), dtype=jnp.float32)
    v = jnp.ones((16, NUM_KV_HEADS, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((NUM_KV_HEADS, NUM_BLOCKS, BLOCK_SIZE, 128),
                         dtype=jnp.float32)
    kv_lens = jnp.zeros((8, ), dtype=jnp.int32)
    page_indices = jnp.zeros((8, MAX_BLOCKS_PER_SEQ), dtype=jnp.int32)
    cu_q_lens = jnp.zeros((9, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)

    mock_shard_map_callable = MagicMock(return_value=(jnp.ones_like(q),
                                                      kv_cache))
    mock_shard_map = MagicMock(return_value=mock_shard_map_callable)
    monkeypatch.setattr("jax.shard_map", mock_shard_map)

    sharded_ragged_paged_attention(
        mesh=pcp_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        shard_pcp_axis=False,
    )

    in_specs = mock_shard_map.call_args.kwargs["in_specs"]
    assert in_specs[0] == P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD,
                            None)
    assert in_specs[3] == P(ShardingAxisName.BATCH, None,
                            ShardingAxisName.KV_CACHE_HEAD, None, None)
    assert in_specs[6] == P(ShardingAxisName.BATCH)
    assert in_specs[7] == P(ShardingAxisName.BATCH)


def test_sharded_ragged_paged_attention_gqa_incompatible_raises_error(
    gqa_mesh, ):
    """
    Tests that a ValueError is raised for GQA when tp_size is not divisible
    by num_kv_heads.
    """
    # 1. Arrange
    tp_size = gqa_mesh.shape[ShardingAxisName.ATTN_HEAD]
    assert tp_size == 4
    num_kv_heads = 3  # Incompatible with tp_size=4
    head_dim = 128

    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim))
    k = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    v = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim))
    # Other metadata
    kv_lens = jnp.zeros((MAX_NUM_SEQS, ), dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                             dtype=jnp.int32)
    cu_q_lens = jnp.zeros((MAX_NUM_SEQS + 1, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)
    sm_scale = 1.0

    # 2. Act & Assert
    with pytest.raises(
            ValueError,
            match=(f"For GQA/MQA, tp_size {tp_size} must be divisible by "
                   f"num_kv_heads {num_kv_heads}"),
    ):
        sharded_ragged_paged_attention(
            mesh=gqa_mesh,
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=None,
            sm_scale=sm_scale,
        )


def _run_sharded_rpa_capturing_kwargs(monkeypatch, gqa_mesh, update_kv_cache):
    """Helper: run `sharded_ragged_paged_attention` with a stubbed
    `ragged_paged_attention` (the module-level binding) and a passthrough
    `jax.shard_map`. Returns the kwargs forwarded by the closure to the
    underlying kernel.
    """
    head_dim = 128  # non-hd64
    num_kv_heads = 4
    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim))
    k = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    v = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim))
    kv_lens = jnp.zeros((MAX_NUM_SEQS, ), dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                             dtype=jnp.int32)
    cu_q_lens = jnp.zeros((MAX_NUM_SEQS + 1, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)

    captured = {}

    def fake_kernel(*_args, **kwargs):
        captured.update(kwargs)
        return jnp.ones_like(q), kv_cache

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        fake_kernel,
    )

    # Passthrough shard_map so the closure actually executes.
    def passthrough_shard_map(inner_fn, **kwargs):
        captured["in_specs"] = kwargs["in_specs"]
        return inner_fn

    monkeypatch.setattr("jax.shard_map", passthrough_shard_map)

    sharded_ragged_paged_attention(
        mesh=gqa_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        update_kv_cache=update_kv_cache,
    )
    return captured


def test_sharded_rpa_forwards_update_kv_cache_when_not_hd64(
        monkeypatch, gqa_mesh):
    """`sharded_ragged_paged_attention` must forward `update_kv_cache`
    to the underlying kernel on the non-hd64 path. Both kernels (v3 and
    batched) accept the kwarg after this fix; the wrapper forwards it
    unconditionally for non-hd64 head sizes."""
    captured = _run_sharded_rpa_capturing_kwargs(monkeypatch,
                                                 gqa_mesh,
                                                 update_kv_cache=False)

    assert captured.get("update_kv_cache") is False, (
        f"non-hd64 path must forward update_kv_cache=False; got {captured!r}")


def test_sharded_rpa_default_path_does_not_forward_pcp_metadata(
        monkeypatch, gqa_mesh):
    captured = _run_sharded_rpa_capturing_kwargs(monkeypatch,
                                                 gqa_mesh,
                                                 update_kv_cache=True)

    assert "q_start_offsets" not in captured
    assert "cu_k_lens" not in captured


def test_make_pcp_interleaved_metadata_for_middle_rank(monkeypatch):
    monkeypatch.setattr("jax.lax.axis_index", lambda axis_name: 1)

    (expanded_kv_lens, expanded_page_indices, local_cu_q_lens, _,
     q_start_offsets, cu_k_lens) = _make_pcp_interleaved_metadata(
         jnp.array([0, 6, 12], dtype=jnp.int32),
         jnp.array([6, 6], dtype=jnp.int32),
         jnp.array([0, 1], dtype=jnp.int32),
         local_num_tokens=6,
         interleave_size=3,
         pcp_size=2,
         axis_name="pcp",
     )

    np.testing.assert_array_equal(local_cu_q_lens, np.array([0, 3, 3, 6, 6]))
    np.testing.assert_array_equal(q_start_offsets, np.array([3, 0, 3, 0]))
    np.testing.assert_array_equal(expanded_kv_lens, np.array([6, 0, 6, 0]))
    np.testing.assert_array_equal(cu_k_lens, np.array([0, 0, 6, 0, 12]))
    np.testing.assert_array_equal(expanded_page_indices, np.array([0, 0, 1,
                                                                   1]))


def test_make_pcp_interleaved_metadata_for_split_sequence(monkeypatch):
    monkeypatch.setattr("jax.lax.axis_index", lambda axis_name: 1)

    (expanded_kv_lens, _, local_cu_q_lens, _, q_start_offsets,
     cu_k_lens) = _make_pcp_interleaved_metadata(
         jnp.array([0, 8], dtype=jnp.int32),
         jnp.array([8], dtype=jnp.int32),
         jnp.array([0], dtype=jnp.int32),
         local_num_tokens=4,
         interleave_size=2,
         pcp_size=2,
         axis_name="pcp",
     )

    np.testing.assert_array_equal(local_cu_q_lens, np.array([0, 2, 4]))
    np.testing.assert_array_equal(q_start_offsets, np.array([2, 6]))
    np.testing.assert_array_equal(expanded_kv_lens, np.array([8, 8]))
    np.testing.assert_array_equal(cu_k_lens, np.array([0, 0, 8]))


def test_make_pcp_interleaved_token_indices():
    indices = _make_pcp_interleaved_token_indices(
        jnp.array([8], dtype=jnp.int32),
        local_num_tokens=4,
        interleave_size=2,
        pcp_size=2,
    )

    np.testing.assert_array_equal(indices, np.array([0, 1, 4, 5, 2, 3, 6, 7]))


def test_update_local_paged_kv_cache_writes_local_slots_only(monkeypatch):
    packed_kv = jnp.arange(6, dtype=jnp.float32).reshape(6, 1, 1, 1)

    def fake_prepare_inputs(q, k, v, q_dtype, kv_dtype):
        del k, v, q_dtype, kv_dtype
        return jnp.zeros_like(q), packed_kv

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.batched_rpa_wrapper.prepare_inputs",
        fake_prepare_inputs,
    )

    kv_cache = jnp.full((3, 4, 1, 1, 1), -1.0, dtype=jnp.float32)
    k = jnp.zeros((6, 1, 1), dtype=jnp.float32)
    v = jnp.zeros((6, 1, 1), dtype=jnp.float32)
    slot_ids = jnp.array([0, 1, -1, 7, 8, -1], dtype=jnp.int32)

    updated = _update_local_paged_kv_cache(kv_cache, k, v, slot_ids)

    expected = np.full((3, 4, 1, 1, 1), -1.0, dtype=np.float32)
    expected[0, 0, 0, 0, 0] = 0
    expected[0, 1, 0, 0, 0] = 1
    expected[1, 3, 0, 0, 0] = 3
    expected[2, 0, 0, 0, 0] = 4
    np.testing.assert_array_equal(np.asarray(updated), expected)


def test_sharded_rpa_pcp_path_gathers_kv_on_pcp_axis_and_forwards_metadata(
        monkeypatch, gqa_mesh):
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    devices = np.array(jax.local_devices()[:1] * 2).reshape((1, 1, 1, 1, 1, 2))
    pcp_mesh = Mesh(
        devices,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "pcp"))

    head_dim = 128
    num_kv_heads = 4
    q = jnp.ones((4, NUM_HEADS, head_dim), dtype=jnp.float32)
    k = jnp.ones((4, num_kv_heads, head_dim), dtype=jnp.float32)
    v = jnp.full((4, num_kv_heads, head_dim), 2.0, dtype=jnp.float32)
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim),
                         dtype=jnp.float32)
    kv_lens = jnp.array([8], dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_BLOCKS_PER_SEQ, ), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 8], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    captured = {"all_gather_axes": []}

    def fake_kernel(q_arg, k_arg, v_arg, kv_cache_arg, *_args, **kwargs):
        captured["q_shape"] = q_arg.shape
        captured["k_shape"] = k_arg.shape
        captured["v_shape"] = v_arg.shape
        captured["kv_lens"] = _args[0]
        captured["page_indices"] = _args[1]
        captured["cu_q_lens"] = _args[2]
        captured["distribution"] = _args[3]
        captured["q_start_offsets"] = kwargs["q_start_offsets"]
        captured["cu_k_lens"] = kwargs["cu_k_lens"]
        captured["update_kv_cache"] = kwargs["update_kv_cache"]
        return jnp.ones_like(q_arg), kv_cache_arg

    def fake_all_gather(x, axis_name, axis, tiled):
        captured["all_gather_axes"].append(axis_name)
        return jnp.concatenate([x, x], axis=axis)

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        fake_kernel,
    )
    monkeypatch.setattr("jax.lax.axis_index", lambda axis_name: 1)
    monkeypatch.setattr("jax.lax.all_gather", fake_all_gather)

    def passthrough_shard_map(inner_fn, **kwargs):
        captured["in_specs"] = kwargs["in_specs"]
        return inner_fn

    monkeypatch.setattr("jax.shard_map", passthrough_shard_map)

    out, new_cache = sharded_ragged_paged_attention(
        mesh=pcp_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        update_kv_cache=False,
        pcp_mode=PcpMode.PREFILL_LOCAL_Q_FULL_KV,
        cp_kv_cache_interleave_size=2,
    )

    assert out.shape == q.shape
    assert new_cache.shape == kv_cache.shape
    assert captured["in_specs"][3] == P(ShardingAxisName.KV_CACHE_BLOCK, None,
                                        ShardingAxisName.KV_CACHE_HEAD, None,
                                        None)
    assert captured["all_gather_axes"] == ["pcp", "pcp"]
    assert captured["q_shape"] == q.shape
    assert captured["k_shape"] == (8, num_kv_heads, head_dim)
    assert captured["v_shape"] == (8, num_kv_heads, head_dim)
    np.testing.assert_array_equal(captured["kv_lens"], np.array([8, 8]))
    np.testing.assert_array_equal(captured["page_indices"],
                                  np.zeros((16, ), dtype=np.int32))
    np.testing.assert_array_equal(captured["cu_q_lens"], np.array([0, 2, 4]))
    np.testing.assert_array_equal(captured["distribution"], np.array([0, 0,
                                                                      2]))
    np.testing.assert_array_equal(captured["q_start_offsets"], np.array([2,
                                                                         6]))
    np.testing.assert_array_equal(captured["cu_k_lens"], np.array([0, 0, 8]))
    assert captured["update_kv_cache"] is False


def test_sharded_rpa_pcp_path_consumes_precomputed_metadata(monkeypatch):
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    devices = np.array(jax.local_devices()[:1] * 2).reshape((1, 1, 1, 1, 1, 2))
    pcp_mesh = Mesh(
        devices,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "pcp"))

    head_dim = 128
    q = jnp.ones((4, 1, head_dim), dtype=jnp.float32)
    row_values = jnp.arange(4, dtype=jnp.float32)[:, None, None]
    k = jnp.broadcast_to(row_values, (4, 1, head_dim))
    v = k + 100
    kv_cache = jnp.zeros((1, NUM_BLOCKS, BLOCK_SIZE, head_dim),
                         dtype=jnp.float32)
    kv_lens = jnp.array([8], dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_BLOCKS_PER_SEQ, ), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 8], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    captured = {}

    def fail_if_called(*_, **__):
        raise AssertionError("PCP metadata should be supplied by the runner.")

    def fake_cache_update(kv_cache_arg, k_arg, v_arg, slot_ids_arg):
        captured["cache_update_k_rows"] = k_arg[:, 0, 0]
        captured["cache_update_v_rows"] = v_arg[:, 0, 0]
        captured["cache_update_slot_ids"] = slot_ids_arg
        return kv_cache_arg + 3

    def fake_kernel(q_arg, k_arg, v_arg, kv_cache_arg, *_args, **kwargs):
        captured["k_rows"] = k_arg[:, 0, 0]
        captured["v_rows"] = v_arg[:, 0, 0]
        captured["kv_cache"] = kv_cache_arg
        captured["kv_lens"] = _args[0]
        captured["page_indices"] = _args[1]
        captured["cu_q_lens"] = _args[2]
        captured["distribution"] = _args[3]
        captured["q_start_offsets"] = kwargs["q_start_offsets"]
        captured["cu_k_lens"] = kwargs["cu_k_lens"]
        captured["update_kv_cache"] = kwargs["update_kv_cache"]
        return jnp.ones_like(q_arg), kv_cache_arg

    def fake_all_gather(x, axis_name, axis, tiled):
        assert axis_name == "pcp"
        assert axis == 0
        assert tiled is True
        return jnp.concatenate([x, x + 10], axis=axis)

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface._make_pcp_interleaved_metadata",
        fail_if_called,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        fake_kernel,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface._update_local_paged_kv_cache",
        fake_cache_update,
    )
    monkeypatch.setattr("jax.lax.all_gather", fake_all_gather)
    monkeypatch.setattr("jax.shard_map", lambda inner_fn, **_: inner_fn)

    sharded_ragged_paged_attention(
        mesh=pcp_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        pcp_mode=PcpMode.PREFILL_LOCAL_Q_FULL_KV,
        cp_kv_cache_interleave_size=2,
        pcp_kv_lens=jnp.array([8, 8], dtype=jnp.int32),
        pcp_page_indices=jnp.arange(16, dtype=jnp.int32),
        pcp_query_start_loc=jnp.array([0, 2, 4], dtype=jnp.int32),
        pcp_request_distribution=jnp.array([0, 0, 2], dtype=jnp.int32),
        pcp_q_start_offsets=jnp.array([2, 6], dtype=jnp.int32),
        pcp_cu_k_lens=jnp.array([0, 0, 8], dtype=jnp.int32),
        pcp_slot_ids=jnp.array([7, 8, 9, 10], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(captured["cache_update_k_rows"],
                                  np.array([0, 1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        captured["cache_update_v_rows"],
        np.array([100, 101, 102, 103], dtype=np.float32))
    np.testing.assert_array_equal(captured["cache_update_slot_ids"],
                                  np.array([7, 8, 9, 10], dtype=np.int32))
    np.testing.assert_array_equal(
        captured["k_rows"],
        np.array([0, 1, 10, 11, 2, 3, 12, 13], dtype=np.float32))
    np.testing.assert_array_equal(
        captured["v_rows"],
        np.array([100, 101, 110, 111, 102, 103, 112, 113], dtype=np.float32))
    np.testing.assert_array_equal(captured["kv_lens"], np.array([8, 8]))
    np.testing.assert_array_equal(captured["page_indices"], np.arange(16))
    np.testing.assert_array_equal(captured["cu_q_lens"], np.array([0, 2, 4]))
    np.testing.assert_array_equal(captured["distribution"], np.array([0, 0,
                                                                      2]))
    np.testing.assert_array_equal(captured["q_start_offsets"], np.array([2,
                                                                         6]))
    np.testing.assert_array_equal(captured["cu_k_lens"], np.array([0, 0, 8]))
    assert captured["update_kv_cache"] is False
    np.testing.assert_array_equal(captured["kv_cache"],
                                  np.asarray(kv_cache) + 3)


def test_sharded_rpa_pcp_streaming_path_uses_packed_kernel(monkeypatch):
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    monkeypatch.setenv("USE_PCP_STREAMING_RPA_KERNEL", "1")
    monkeypatch.setenv("PCP_STREAMING_RPA_Q_BLOCK_SIZE", "2")
    monkeypatch.setenv("PCP_STREAMING_RPA_KV_BLOCK_SIZE", "8")
    devices = np.array(jax.local_devices()[:1] * 2).reshape((1, 1, 1, 1, 1, 2))
    pcp_mesh = Mesh(
        devices,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "pcp"))

    q = jnp.ones((4, 2, 128), dtype=jnp.float32)
    k = jnp.ones((4, 1, 128), dtype=jnp.float32)
    v = jnp.full((4, 1, 128), 2.0, dtype=jnp.float32)
    kv_cache = jnp.zeros((2, 4, 2, 1, 128), dtype=jnp.float32)
    streaming_schedule = jnp.zeros((1, 2, 2, 1, 128), dtype=jnp.int32)
    captured = {}

    def fake_cache_update(kv_cache_arg, k_arg, v_arg, slot_ids_arg):
        captured["slot_ids"] = slot_ids_arg
        return kv_cache_arg + 5

    def fake_streaming_kernel(q_arg, kv_cache_arg, schedule_arg,
                              active_groups_arg, **kwargs):
        captured["q_shape"] = q_arg.shape
        captured["kv_cache"] = kv_cache_arg
        captured["schedule"] = schedule_arg
        captured["active_groups"] = active_groups_arg
        captured["pcp_size"] = kwargs["pcp_size"]
        captured["q_block_size"] = kwargs["q_block_size"]
        captured["kv_pages_per_block"] = kwargs["kv_pages_per_block"]
        captured["sm_scale"] = kwargs["sm_scale"]
        return jnp.full_like(q_arg, 7.0)

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface._update_local_paged_kv_cache",
        fake_cache_update,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.pcp_streaming_attention_page_groups_packed_local",
        fake_streaming_kernel,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        lambda *_, **__: pytest.fail("old PCP RPA path must not run"),
    )
    monkeypatch.setattr(
        "jax.lax.all_gather",
        lambda *_, **__: pytest.fail("streaming path must not all_gather KV"),
    )
    monkeypatch.setattr("jax.shard_map", lambda inner_fn, **_: inner_fn)

    out, new_cache = sharded_ragged_paged_attention(
        mesh=pcp_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.array([8], dtype=jnp.int32),
        page_indices=jnp.zeros((2, ), dtype=jnp.int32),
        cu_q_lens=jnp.array([0, 4], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        attention_sink=None,
        sm_scale=0.25,
        pcp_mode=PcpMode.PREFILL_LOCAL_Q_FULL_KV,
        cp_kv_cache_interleave_size=4,
        pcp_kv_lens=jnp.array([8, 8], dtype=jnp.int32),
        pcp_page_indices=jnp.arange(4, dtype=jnp.int32),
        pcp_query_start_loc=jnp.array([0, 2, 4], dtype=jnp.int32),
        pcp_request_distribution=jnp.array([0, 0, 2], dtype=jnp.int32),
        pcp_q_start_offsets=jnp.array([0, 4], dtype=jnp.int32),
        pcp_cu_k_lens=jnp.array([0, 0, 8], dtype=jnp.int32),
        pcp_slot_ids=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        pcp_streaming_schedule=streaming_schedule,
        pcp_streaming_active_page_groups=jnp.array([1], dtype=jnp.int32),
    )

    assert out.shape == q.shape
    np.testing.assert_array_equal(np.asarray(out), np.full(q.shape, 7.0))
    np.testing.assert_array_equal(np.asarray(new_cache),
                                  np.asarray(kv_cache) + 5)
    assert captured["q_shape"] == (4, 1, 2, 128)
    np.testing.assert_array_equal(captured["slot_ids"],
                                  np.array([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(captured["schedule"],
                                  np.zeros((2, 2, 1, 128), dtype=np.int32))
    np.testing.assert_array_equal(captured["active_groups"],
                                  np.array([1], dtype=np.int32))
    assert captured["pcp_size"] == 2
    assert captured["q_block_size"] == 2
    assert captured["kv_pages_per_block"] == 2
    assert captured["sm_scale"] == 0.25


def test_sharded_rpa_legacy_pcp_flags_select_explicit_modes(monkeypatch, mesh):
    q = jnp.ones((4, 1, 128), dtype=jnp.float32)
    k = jnp.ones((4, 1, 128), dtype=jnp.float32)
    v = jnp.ones((4, 1, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((1, NUM_BLOCKS, BLOCK_SIZE, 128), dtype=jnp.float32)
    kv_lens = jnp.array([4], dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_BLOCKS_PER_SEQ, ), dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 4], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
    captured = []

    def fake_prefill(**kwargs):
        captured.append("prefill")
        return jnp.ones_like(kwargs["q"]), kwargs["kv_cache"]

    def fake_decode(**kwargs):
        captured.append("decode")
        return jnp.full_like(kwargs["q"], 2.0), kwargs["kv_cache"]

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.sharded_pcp_ragged_paged_attention",
        fake_prefill,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.sharded_pcp_decode_ragged_paged_attention",
        fake_decode,
    )

    sharded_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        use_pcp=True,
    )
    sharded_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        use_pcp_decode=True,
    )

    assert captured == ["prefill", "decode"]


def test_sharded_rpa_rejects_conflicting_legacy_pcp_flags(mesh):
    q = jnp.ones((4, 1, 128), dtype=jnp.float32)
    k = jnp.ones((4, 1, 128), dtype=jnp.float32)
    v = jnp.ones((4, 1, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((1, NUM_BLOCKS, BLOCK_SIZE, 128), dtype=jnp.float32)
    kwargs = dict(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.array([4], dtype=jnp.int32),
        page_indices=jnp.zeros((MAX_BLOCKS_PER_SEQ, ), dtype=jnp.int32),
        cu_q_lens=jnp.array([0, 4], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        attention_sink=None,
        sm_scale=1.0,
    )

    with pytest.raises(ValueError, match="Conflicting PCP mode"):
        sharded_ragged_paged_attention(
            **kwargs,
            pcp_mode=PcpMode.DECODE_SHARDED_KV,
            use_pcp=True,
        )
    with pytest.raises(ValueError, match="Conflicting PCP mode"):
        sharded_ragged_paged_attention(
            **kwargs,
            pcp_mode=PcpMode.PREFILL_LOCAL_Q_FULL_KV,
            use_pcp_decode=True,
        )


def test_sharded_rpa_pcp_decode_path_materializes_kv(monkeypatch):
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    devices = np.array(jax.local_devices()[:1] * 2).reshape((1, 1, 1, 1, 1, 2))
    pcp_mesh = Mesh(
        devices,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "pcp"))

    head_dim = 128
    q = jnp.ones((4, 1, head_dim), dtype=jnp.float32)
    row_values = jnp.arange(4, dtype=jnp.float32)[:, None, None]
    k = jnp.broadcast_to(row_values, (4, 1, head_dim))
    v = k + 100
    kv_cache = jnp.zeros((NUM_BLOCKS, BLOCK_SIZE, 1, 1, head_dim),
                         dtype=jnp.float32)
    cu_q_lens = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    distribution = jnp.array([4, 4, 4], dtype=jnp.int32)
    global_kv_lens = jnp.array([5, 4, 0, 0], dtype=jnp.int32)
    source_block_tables = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    pcp_slot_ids = jnp.array([30, -1, -1, -1], dtype=jnp.int32)
    captured = {}

    def fake_cache_update(kv_cache_arg, k_arg, v_arg, slot_ids_arg):
        captured["cache_update_k_rows"] = k_arg[:, 0, 0]
        captured["cache_update_v_rows"] = v_arg[:, 0, 0]
        captured["cache_update_slot_ids"] = slot_ids_arg
        return kv_cache_arg + 7

    def fake_materialize(kv_cache_arg, kv_lens_arg, source_block_tables_arg,
                         page_size_arg, pcp_size_arg, interleave_size_arg,
                         pcp_axis_name_arg):
        captured["materialize_kv_cache"] = kv_cache_arg
        captured["materialize_kv_lens"] = kv_lens_arg
        captured["materialize_source_block_tables"] = source_block_tables_arg
        captured["materialize_page_size"] = page_size_arg
        captured["materialize_pcp_size"] = pcp_size_arg
        captured["materialize_interleave_size"] = interleave_size_arg
        captured["materialize_axis"] = pcp_axis_name_arg
        return kv_cache_arg + 11, kv_lens_arg, jnp.arange(16, dtype=jnp.int32)

    def fake_kernel(q_arg, k_arg, v_arg, kv_cache_arg, *_args, **kwargs):
        captured["kernel_q_shape"] = q_arg.shape
        captured["kernel_kv_cache"] = kv_cache_arg
        captured["kernel_kv_lens"] = _args[0]
        captured["kernel_page_indices"] = _args[1]
        captured["kernel_cu_q_lens"] = _args[2]
        captured["kernel_distribution"] = _args[3]
        captured["kernel_update_kv_cache"] = kwargs["update_kv_cache"]
        captured["kernel_return_lse"] = kwargs.get("return_lse")
        return jnp.full_like(q_arg, 9.0), kv_cache_arg + 3

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface._update_local_paged_kv_cache",
        fake_cache_update,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.materialize_pcp_kv_for_decode",
        fake_materialize,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.ragged_paged_attention",
        fake_kernel,
    )
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.pcp_lse_merge",
        lambda *args, **kwargs: pytest.fail("pcp_lse_merge must not be called"),
    )

    def passthrough_shard_map(inner_fn, **kwargs):
        captured["in_specs"] = kwargs["in_specs"]
        return inner_fn

    monkeypatch.setattr("jax.shard_map", passthrough_shard_map)

    out, new_cache = sharded_ragged_paged_attention(
        mesh=pcp_mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=global_kv_lens,
        page_indices=jnp.zeros((MAX_BLOCKS_PER_SEQ * 4, ), dtype=jnp.int32),
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        attention_sink=None,
        sm_scale=1.0,
        update_kv_cache=True,
        pcp_mode=PcpMode.DECODE_SHARDED_KV,
        cp_kv_cache_interleave_size=2,
        pcp_slot_ids=pcp_slot_ids,
        pcp_source_block_tables=source_block_tables,
    )

    assert captured["in_specs"][0] == P(ShardingAxisName.BATCH,
                                        ShardingAxisName.KV_CACHE_HEAD, None)
    assert captured["in_specs"][3] == P(ShardingAxisName.KV_CACHE_BLOCK, None,
                                        ShardingAxisName.KV_CACHE_HEAD, None,
                                        None)
    assert captured["in_specs"][6] == P(ShardingAxisName.BATCH)
    assert captured["in_specs"][7] == P(ShardingAxisName.BATCH, None)
    assert captured["in_specs"][8] == P(ShardingAxisName.ATTN_DATA)
    np.testing.assert_array_equal(captured["cache_update_k_rows"],
                                  np.array([0, 1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        captured["cache_update_v_rows"],
        np.array([100, 101, 102, 103], dtype=np.float32))
    np.testing.assert_array_equal(captured["cache_update_slot_ids"],
                                  np.array([30, -1, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(captured["materialize_kv_cache"]),
                                  np.asarray(kv_cache) + 7)
    np.testing.assert_array_equal(captured["materialize_kv_lens"],
                                  np.array([5, 4, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        captured["materialize_source_block_tables"],
        np.arange(16, dtype=np.int32).reshape(4, 4))
    assert captured["materialize_page_size"] == BLOCK_SIZE
    assert captured["materialize_pcp_size"] == 2
    assert captured["materialize_interleave_size"] == 2
    assert captured["materialize_axis"] == "pcp"
    np.testing.assert_array_equal(captured["kernel_kv_cache"],
                                  np.asarray(kv_cache) + 18)
    np.testing.assert_array_equal(captured["kernel_kv_lens"],
                                  np.array([5, 4, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(captured["kernel_page_indices"],
                                  np.arange(16))
    assert captured["kernel_update_kv_cache"] is False
    assert captured["kernel_return_lse"] is None
    np.testing.assert_array_equal(np.asarray(out), np.full_like(q, 9.0))
    np.testing.assert_array_equal(np.asarray(new_cache),
                                  np.asarray(kv_cache) + 7)


def test_sharded_rpa_pcp_decode_rejects_sliding_window(monkeypatch):
    monkeypatch.setattr(ShardingAxisName, "_cls", ShardingAxisNameBase)
    devices = np.array(jax.local_devices()[:1] * 2).reshape((1, 1, 1, 1, 1, 2))
    pcp_mesh = Mesh(
        devices,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "pcp"))

    with pytest.raises(NotImplementedError, match="full attention only"):
        sharded_ragged_paged_attention(
            mesh=pcp_mesh,
            q=jnp.ones((1, 1, 128), dtype=jnp.float32),
            k=jnp.ones((1, 1, 128), dtype=jnp.float32),
            v=jnp.ones((1, 1, 128), dtype=jnp.float32),
            kv_cache=jnp.zeros((NUM_BLOCKS, BLOCK_SIZE, 1, 1, 128),
                               dtype=jnp.float32),
            kv_lens=jnp.array([1], dtype=jnp.int32),
            page_indices=jnp.zeros((MAX_BLOCKS_PER_SEQ, ), dtype=jnp.int32),
            cu_q_lens=jnp.array([0, 1], dtype=jnp.int32),
            distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
            attention_sink=None,
            sm_scale=1.0,
            attention_chunk_size=16,
            pcp_mode=PcpMode.DECODE_SHARDED_KV,
            cp_kv_cache_interleave_size=2,
            pcp_slot_ids=jnp.array([0], dtype=jnp.int32),
            pcp_source_block_tables=jnp.array([[0]], dtype=jnp.int32),
        )


def test_attention_forwards_precomputed_pcp_metadata(monkeypatch, mesh):
    captured = {}
    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, 128), dtype=jnp.float32)
    k = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, 128), dtype=jnp.float32)
    v = jnp.ones((TOTAL_TOKENS, NUM_KV_HEADS, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((NUM_KV_HEADS, NUM_BLOCKS, BLOCK_SIZE, 128),
                         dtype=jnp.float32)
    pcp_metadata = {
        "pcp_kv_lens": jnp.array([5, 5], dtype=jnp.int32),
        "pcp_page_indices": jnp.arange(16, dtype=jnp.int32),
        "pcp_query_start_loc": jnp.array([0, 5, 10], dtype=jnp.int32),
        "pcp_request_distribution": jnp.array([0, 0, 2], dtype=jnp.int32),
        "pcp_q_start_offsets": jnp.array([0, 0], dtype=jnp.int32),
        "pcp_cu_k_lens": jnp.array([0, 5, 10], dtype=jnp.int32),
        "pcp_slot_ids": jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        "pcp_source_block_tables": jnp.arange(16,
                                              dtype=jnp.int32).reshape(4, 4),
        "pcp_streaming_schedule": jnp.zeros((1, 2, 2, 1, 128),
                                            dtype=jnp.int32),
    }
    attention_metadata = AttentionMetadata(
        input_positions=jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        block_tables=jnp.zeros((MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32),
        **pcp_metadata,
    )

    def fake_sharded_rpa(*args, **kwargs):
        captured.update(kwargs)
        return jnp.ones_like(q), kv_cache

    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.sharded_ragged_paged_attention",
        fake_sharded_rpa,
    )

    attention(
        kv_cache=kv_cache,
        q=q,
        k=k,
        v=v,
        attention_metadata=attention_metadata,
        mesh=mesh,
        pcp_mode=PcpMode.PREFILL_LOCAL_Q_FULL_KV,
        cp_kv_cache_interleave_size=2,
    )

    for name, value in pcp_metadata.items():
        assert captured[name] is value


def test_batched_rpa_wrapper_accepts_update_kv_cache():
    """Direct signature check that catches the original #2601 crash:
    the batched RPA wrapper's `ragged_paged_attention` must declare an
    `update_kv_cache` keyword parameter. `sharded_ragged_paged_attention`
    always forwards the kwarg on the non-hd64 path; without this
    signature, that forwarding crashed at trace time with
    `TypeError: ragged_paged_attention() got an unexpected keyword
    argument 'update_kv_cache'`."""
    import inspect

    from tpu_inference.kernels.experimental.batched_rpa import wrapper
    params = inspect.signature(wrapper.ragged_paged_attention).parameters
    assert "update_kv_cache" in params, (
        f"batched RPA wrapper must accept update_kv_cache as a kwarg; "
        f"got params: {list(params.keys())}")
    assert params["update_kv_cache"].default is True, (
        f"update_kv_cache should default to True (no-op for non-KV-share "
        f"callers); got default={params['update_kv_cache'].default!r}")


def test_sharded_rpa_rejects_update_kv_cache_false_on_hd64(gqa_mesh):
    """The hd64 RPA kernel doesn't support KV-share; passing
    update_kv_cache=False must raise rather than silently writing to
    cache. (Currently no model uses head_dim=64 + KV-share, but the
    guard is cheap insurance.)"""
    head_dim = 64
    num_kv_heads = 4
    q = jnp.ones((TOTAL_TOKENS, NUM_HEADS, head_dim))
    k = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    v = jnp.ones((TOTAL_TOKENS, num_kv_heads, head_dim))
    kv_cache = jnp.zeros((num_kv_heads, NUM_BLOCKS, BLOCK_SIZE, head_dim))
    kv_lens = jnp.zeros((MAX_NUM_SEQS, ), dtype=jnp.int32)
    page_indices = jnp.zeros((MAX_NUM_SEQS, MAX_BLOCKS_PER_SEQ),
                             dtype=jnp.int32)
    cu_q_lens = jnp.zeros((MAX_NUM_SEQS + 1, ), dtype=jnp.int32)
    distribution = jnp.zeros((3, ), dtype=jnp.int32)

    with pytest.raises(NotImplementedError, match="head_dim==64"):
        sharded_ragged_paged_attention(
            mesh=gqa_mesh,
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            attention_sink=None,
            sm_scale=1.0,
            update_kv_cache=False,
        )


def test_mla_attention(monkeypatch, mesh):
    """
    Tests the `mla_attention` function.

    Verifies that:
    1. It correctly calculates block sizes using `get_tuned_block_sizes`
    2. It calls `mla_ragged_paged_attention` with the correct arguments
    3. It returns the expected output and updated KV cache
    """
    qk_nope_dim = 32
    qk_rope_dim = 16
    q_lora_rank = 64
    kv_lora_rank = 64

    q_NTA = jnp.ones((NUM_HEADS, TOTAL_TOKENS, q_lora_rank))
    q_rope_TNH = jnp.ones((TOTAL_TOKENS, NUM_HEADS, qk_rope_dim))
    k_SA = jnp.ones((TOTAL_TOKENS, kv_lora_rank))
    k_rope_SH = jnp.ones((TOTAL_TOKENS, qk_rope_dim))

    # Arbitrary cache shape just for testing
    kv_cache_shape = (1, NUM_BLOCKS, BLOCK_SIZE, kv_lora_rank)
    kv_cache = jnp.zeros(kv_cache_shape)

    metadata = AttentionMetadata(
        input_positions=jnp.arange(TOTAL_TOKENS, dtype=jnp.int32),
        block_tables=jnp.zeros((MAX_NUM_SEQS * MAX_BLOCKS_PER_SEQ, ),
                               dtype=jnp.int32),
        seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32),
    )

    expected_output = jnp.full(q_NTA.shape, 0.5)
    expected_new_cache = jnp.full(kv_cache_shape, 0.1)

    mock_mla_kernel = MagicMock(return_value=(expected_output,
                                              expected_new_cache))
    monkeypatch.setattr(
        "tpu_inference.layers.common.attention_interface.mla_ragged_paged_attention",
        mock_mla_kernel)

    final_kv_cache, output = mla_attention(
        q_NTA=q_NTA,
        q_rope_TNH=q_rope_TNH,
        k_SA=k_SA,
        k_rope_SH=k_rope_SH,
        kv_cache=kv_cache,
        md=metadata,
        mesh=mesh,
        num_attention_heads=NUM_HEADS,
        qk_nope_head_dim=qk_nope_dim,
        sm_scale=0.1,
    )

    mock_mla_kernel.assert_called_once()

    # Verify output correctness
    assert jnp.array_equal(output, expected_output)
    assert jnp.array_equal(final_kv_cache, expected_new_cache)

    _, kernel_kwargs = mock_mla_kernel.call_args
    assert kernel_kwargs["num_kv_pages_per_block"] == (3, 1, 1)
    assert kernel_kwargs["num_queries_per_block"] == (1, 16, 16)
    assert kernel_kwargs["sm_scale"] == 0.1
