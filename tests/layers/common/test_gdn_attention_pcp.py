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
"""Tests for GDN attention PCP prefill path correctness."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.layers.common.gdn_attention import (
    GdnAttentionConfig,
    RaggedGatedDeltaRuleImpl,
    run_jax_gdn_attention,
    run_jax_gdn_attention_local,
    run_jax_gdn_attention_pcp_tp_prefill,
)
from tpu_inference.layers.common.sharding import (
    MESH_AXIS_NAMES,
    ShardingAxisNameBase,
)
from tpu_inference.layers.common.utils import (
    reorder_concatenated_tensor_for_sharding,
)
from tpu_inference.runner.tpu_runner import (
    _build_pcp_rank_major_token_order,
    _pcp_local_token_counts,
)


def _make_pcp_mesh(pcp_size: int) -> Mesh:
    """Create a mesh with the given pcp_size using available devices."""
    devices = np.array(jax.devices())
    num_devices = len(devices)
    assert num_devices >= pcp_size, (
        f"Need at least {pcp_size} devices, got {num_devices}")
    # Use pcp_size devices, all other axes = 1
    mesh_shape = [1] * len(MESH_AXIS_NAMES)
    pcp_idx = MESH_AXIS_NAMES.index("pcp")
    mesh_shape[pcp_idx] = pcp_size
    total_needed = int(np.prod(mesh_shape))
    return Mesh(devices[:total_needed].reshape(mesh_shape), MESH_AXIS_NAMES)


class TestGdnAttentionPcpPrefill:
    """Test PCP prefill path produces same results as non-PCP baseline."""

    @pytest.fixture(params=[
        pytest.param({"pcp_size": 2, "interleave_size": 16,
                      "lengths": [128, 64]}, id="pcp2_2reqs"),
        pytest.param({"pcp_size": 2, "interleave_size": 16,
                      "lengths": [37, 19, 51]}, id="pcp2_partial_chunks"),
        pytest.param({"pcp_size": 2, "interleave_size": 16,
                      "lengths": [256]}, id="pcp2_1req"),
        pytest.param({"pcp_size": 4, "interleave_size": 8,
                      "lengths": [128, 64, 32]}, id="pcp4_3reqs"),
    ])
    def test_params(self, request):
        return request.param

    def test_pcp_prefill_matches_baseline(self, test_params):
        """PCP prefill path must produce identical output to baseline."""
        pcp_size = test_params["pcp_size"]
        interleave_size = test_params["interleave_size"]
        lengths = test_params["lengths"]

        num_devices = len(jax.devices())
        if num_devices < pcp_size:
            pytest.skip(f"Need {pcp_size} devices, have {num_devices}")

        # Model hyperparameters (small for fast testing)
        n_kq = 4
        n_v = 4
        d_k = 64
        d_v = 64
        kernel_size = 4
        max_reqs = len(lengths)
        num_tokens = sum(lengths)
        num_blocks = max_reqs + 1
        dim = n_kq * d_k + n_kq * d_k + n_v * d_v  # QKV concatenated

        # Build test data
        rng = jax.random.key(42)
        keys = jax.random.split(rng, 10)

        mixed_qkv = jax.random.normal(keys[0], (num_tokens, dim),
                                      dtype=jnp.bfloat16)
        b = jax.random.normal(keys[1], (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        a = jax.random.normal(keys[2], (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        conv_state = jnp.zeros((num_blocks, kernel_size - 1, dim),
                               dtype=jnp.bfloat16)
        recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v),
                                    dtype=jnp.float32)
        conv_weight = jax.random.normal(keys[3], (dim, 1, kernel_size),
                                        dtype=jnp.bfloat16)
        conv_bias = jax.random.normal(keys[4], (dim,), dtype=jnp.bfloat16)
        A_log = jax.random.normal(keys[5], (n_v,), dtype=jnp.float32)
        dt_bias = jax.random.normal(keys[6], (n_v,), dtype=jnp.float32)

        # Build metadata for a pure-prefill batch
        query_start_loc = np.zeros(max_reqs + 1, dtype=np.int32)
        np.cumsum(lengths, out=query_start_loc[1:])
        query_start_loc = jnp.array(query_start_loc)

        state_indices = jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
        # Pure prefill: all tokens are prefill, no decode
        distribution = jnp.array([0, max_reqs, max_reqs], dtype=jnp.int32)
        # seq_lens == query_lens for fresh prefill
        seq_lens = jnp.array(lengths, dtype=jnp.int32)

        # ============ Run baseline (no PCP) ============
        config = GdnAttentionConfig(
            ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF)

        (ref_conv, ref_rec), ref_output = jax.jit(
            run_jax_gdn_attention_local,
            static_argnames=["n_kq", "n_v", "d_k", "d_v",
                             "kernel_size", "config"],
        )(
            mixed_qkv, b, a, conv_state, recurrent_state,
            conv_weight, conv_bias, A_log, dt_bias,
            query_start_loc, state_indices, distribution, seq_lens,
            n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
            kernel_size=kernel_size, config=config,
        )

        # ============ Run PCP prefill path ============
        mesh = _make_pcp_mesh(pcp_size)

        # Pad to the runner's PCP capacity. Partial chunks can make per-rank
        # token counts imbalanced, so total-token padding is not sufficient.
        padded_num_tokens = int(_pcp_local_token_counts(
            lengths, pcp_size, interleave_size).max()) * pcp_size

        # Build reorder indices using the same function the runner uses
        token_order, _ = _build_pcp_rank_major_token_order(
            np.array(lengths),
            pcp_size,
            interleave_size,
            padded_num_tokens,
        )
        reorder_indices = jnp.array(token_order.astype(np.int32))

        # Pad token arrays to padded_num_tokens
        pad_tokens = padded_num_tokens - num_tokens
        mixed_qkv_padded = jnp.pad(
            mixed_qkv, ((0, pad_tokens), (0, 0)))
        b_padded = jnp.pad(b, ((0, pad_tokens), (0, 0)))
        a_padded = jnp.pad(a, ((0, pad_tokens), (0, 0)))

        # Reorder padded tokens into rank-major order (as the runner does)
        valid = token_order >= 0
        packed_qkv = jnp.zeros_like(mixed_qkv_padded)
        packed_b = jnp.zeros_like(b_padded)
        packed_a = jnp.zeros_like(a_padded)

        valid_indices = np.where(valid)[0]
        src_indices = token_order[valid]
        packed_qkv = packed_qkv.at[valid_indices].set(
            mixed_qkv_padded[src_indices])
        packed_b = packed_b.at[valid_indices].set(b_padded[src_indices])
        packed_a = packed_a.at[valid_indices].set(a_padded[src_indices])

        effective_tp = pcp_size
        packed_qkv = reorder_concatenated_tensor_for_sharding(
            packed_qkv, [n_kq * d_k, n_kq * d_k, n_v * d_v],
            effective_tp, -1)
        conv_weight_pcp = reorder_concatenated_tensor_for_sharding(
            conv_weight, [n_kq * d_k, n_kq * d_k, n_v * d_v],
            effective_tp, 0)
        conv_bias_pcp = reorder_concatenated_tensor_for_sharding(
            conv_bias, [n_kq * d_k, n_kq * d_k, n_v * d_v],
            effective_tp, 0)

        # Apply shardings matching what the model forward would produce.
        # Use ShardingAxisNameBase directly (multi-axis mode with pcp).
        token_sharding = NamedSharding(
            mesh, P(ShardingAxisNameBase.ATTN_DATA))
        state_sharding = NamedSharding(
            mesh, P(ShardingAxisNameBase.BATCH))
        packed_qkv_dev = jax.device_put(
            packed_qkv,
            NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_DATA,
                                  ShardingAxisNameBase.ATTN_HEAD)))
        packed_b_dev = jax.device_put(
            packed_b,
            NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_DATA,
                                  ShardingAxisNameBase.ATTN_HEAD)))
        packed_a_dev = jax.device_put(
            packed_a,
            NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_DATA,
                                  ShardingAxisNameBase.ATTN_HEAD)))
        conv_state_dev = jax.device_put(
            conv_state,
            NamedSharding(mesh, P(ShardingAxisNameBase.BATCH, None,
                                  ShardingAxisNameBase.ATTN_HEAD)))
        rec_state_dev = jax.device_put(
            recurrent_state,
            NamedSharding(mesh, P(ShardingAxisNameBase.BATCH,
                                  ShardingAxisNameBase.ATTN_HEAD,
                                  None, None)))
        conv_weight_dev = jax.device_put(
            conv_weight_pcp,
            NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD,
                                  None, None)))
        conv_bias_dev = jax.device_put(
            conv_bias_pcp,
            NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD)))
        A_log_dev = jax.device_put(
            A_log, NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD)))
        dt_bias_dev = jax.device_put(
            dt_bias, NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD)))
        qsl_dev = jax.device_put(query_start_loc, state_sharding)
        si_dev = jax.device_put(state_indices, state_sharding)
        dist_dev = jax.device_put(distribution, state_sharding)
        sl_dev = jax.device_put(seq_lens, state_sharding)
        reorder_dev = jax.device_put(reorder_indices, token_sharding)

        # Temporarily override ShardingAxisName to use base (multi-axis)
        from tpu_inference.layers.common.sharding import ShardingAxisName
        old_cls = ShardingAxisName._cls
        ShardingAxisName._cls = ShardingAxisNameBase

        try:
            (pcp_conv, pcp_rec), pcp_output = jax.jit(
                run_jax_gdn_attention_pcp_tp_prefill,
                static_argnames=["n_kq", "n_v", "d_k", "d_v",
                                 "kernel_size", "pcp_size", "mesh",
                                 "config"],
            )(
                packed_qkv_dev, packed_b_dev, packed_a_dev,
                conv_state_dev, rec_state_dev,
                conv_weight_dev, conv_bias_dev,
                A_log_dev, dt_bias_dev,
                si_dev, qsl_dev, dist_dev, sl_dev,
                reorder_dev,
                n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                kernel_size=kernel_size, pcp_size=pcp_size,
                mesh=mesh, config=config,
            )
        finally:
            ShardingAxisName._cls = old_cls

        # The PCP output is in packed rank-major order. Extract valid tokens
        # and reorder back to sequential for comparison.
        pcp_output_np = np.array(pcp_output)
        # pcp_output has shape (padded_num_tokens, n_v * d_v)
        # Reconstruct sequential output from packed output
        pcp_output_seq = np.zeros((padded_num_tokens, pcp_output_np.shape[1]),
                                  dtype=pcp_output_np.dtype)
        pcp_output_seq[token_order[valid]] = pcp_output_np[valid]

        # Compare output (only valid tokens)
        ref_output_np = np.array(ref_output)
        np.testing.assert_allclose(
            pcp_output_seq[:num_tokens],
            ref_output_np,
            rtol=5e-2, atol=5e-2,
            err_msg="PCP prefill output differs from baseline")

        # Compare states (should be identical since all ranks do same work)
        np.testing.assert_allclose(
            np.array(pcp_conv), np.array(ref_conv),
            rtol=5e-2, atol=5e-2,
            err_msg="PCP conv_state differs from baseline")
        np.testing.assert_allclose(
            np.array(pcp_rec), np.array(ref_rec),
            rtol=5e-2, atol=5e-2,
            err_msg="PCP recurrent_state differs from baseline")


class TestGdnAttentionPcpDecode:
    """Test PCP decode path keeps GDN on BATCH sharding."""

    def test_decode_only_uses_batch_sharding_under_pcp(self):
        pcp_size = 2
        if len(jax.devices()) < pcp_size:
            pytest.skip(f"Need {pcp_size} devices, have {len(jax.devices())}")

        n_kq = 2
        n_v = 4
        d_k = 32
        d_v = 32
        kernel_size = 4
        max_reqs = 4
        num_tokens = max_reqs
        num_blocks = max_reqs + 1
        dim = n_kq * d_k + n_kq * d_k + n_v * d_v

        rng = jax.random.key(123)
        keys = jax.random.split(rng, 10)
        mixed_qkv = jax.random.normal(keys[0], (num_tokens, dim),
                                      dtype=jnp.bfloat16)
        b = jax.random.normal(keys[1], (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        a = jax.random.normal(keys[2], (num_tokens, n_v),
                              dtype=jnp.bfloat16)
        conv_state = jax.random.normal(keys[3],
                                       (num_blocks, kernel_size - 1, dim),
                                       dtype=jnp.bfloat16)
        recurrent_state = jax.random.normal(keys[4],
                                            (num_blocks, n_v, d_k, d_v),
                                            dtype=jnp.float32)
        conv_weight = jax.random.normal(keys[5], (dim, 1, kernel_size),
                                        dtype=jnp.bfloat16)
        conv_bias = jax.random.normal(keys[6], (dim,), dtype=jnp.bfloat16)
        A_log = jax.random.normal(keys[7], (n_v,), dtype=jnp.float32)
        dt_bias = jax.random.normal(keys[8], (n_v,), dtype=jnp.float32)

        query_start_loc = jnp.arange(max_reqs + 1, dtype=jnp.int32)
        state_indices = jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
        distribution = jnp.array([max_reqs, max_reqs, max_reqs],
                                 dtype=jnp.int32)
        # Decode tokens continue from existing context, so GDN must consume the
        # replicated recurrent state rather than zeroing it like fresh prefill.
        seq_lens = jnp.array([8, 16, 24, 32], dtype=jnp.int32)
        config = GdnAttentionConfig(
            ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl.REF)

        (ref_conv, ref_rec), ref_output = jax.jit(
            run_jax_gdn_attention_local,
            static_argnames=["n_kq", "n_v", "d_k", "d_v",
                             "kernel_size", "config"],
        )(
            mixed_qkv, b, a, conv_state, recurrent_state,
            conv_weight, conv_bias, A_log, dt_bias,
            query_start_loc, state_indices, distribution, seq_lens,
            n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
            kernel_size=kernel_size, config=config,
        )

        mesh = _make_pcp_mesh(pcp_size)
        batch_head = NamedSharding(
            mesh, P(ShardingAxisNameBase.BATCH, ShardingAxisNameBase.ATTN_HEAD))
        batch = NamedSharding(mesh, P(ShardingAxisNameBase.BATCH))

        from tpu_inference.layers.common.sharding import ShardingAxisName
        old_cls = ShardingAxisName._cls
        ShardingAxisName._cls = ShardingAxisNameBase
        try:
            (pcp_conv, pcp_rec), pcp_output = jax.jit(
                run_jax_gdn_attention,
                static_argnames=["n_kq", "n_v", "d_k", "d_v",
                                 "kernel_size", "mesh", "config"],
            )(
                jax.device_put(mixed_qkv, batch_head),
                jax.device_put(b, batch_head),
                jax.device_put(a, batch_head),
                jax.device_put(
                    conv_state,
                    NamedSharding(mesh, P(ShardingAxisNameBase.BATCH, None,
                                          ShardingAxisNameBase.ATTN_HEAD))),
                jax.device_put(
                    recurrent_state,
                    NamedSharding(mesh, P(ShardingAxisNameBase.BATCH,
                                          ShardingAxisNameBase.ATTN_HEAD,
                                          None, None))),
                jax.device_put(
                    conv_weight,
                    NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD,
                                          None, None))),
                jax.device_put(
                    conv_bias,
                    NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD))),
                jax.device_put(
                    A_log,
                    NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD))),
                jax.device_put(
                    dt_bias,
                    NamedSharding(mesh, P(ShardingAxisNameBase.ATTN_HEAD))),
                jax.device_put(state_indices, batch),
                jax.device_put(query_start_loc, batch),
                jax.device_put(distribution, batch),
                jax.device_put(seq_lens, batch),
                n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                kernel_size=kernel_size, mesh=mesh, config=config,
            )
        finally:
            ShardingAxisName._cls = old_cls

        np.testing.assert_allclose(
            np.array(pcp_output), np.array(ref_output),
            rtol=5e-2, atol=5e-2,
            err_msg="PCP decode output differs from BATCH baseline")
        np.testing.assert_allclose(
            np.array(pcp_conv), np.array(ref_conv),
            rtol=5e-2, atol=5e-2,
            err_msg="PCP decode conv_state differs from BATCH baseline")
        np.testing.assert_allclose(
            np.array(pcp_rec), np.array(ref_rec),
            rtol=5e-2, atol=5e-2,
            err_msg="PCP decode recurrent_state differs from BATCH baseline")
