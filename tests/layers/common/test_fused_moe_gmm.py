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

import types

from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common import fused_moe_gmm


def test_remove_sharding_axis():
    assert fused_moe_gmm._remove_sharding_axis(("data", "pcp"),
                                               "pcp") == ("data", )
    assert fused_moe_gmm._remove_sharding_axis("pcp", "pcp") is None
    assert fused_moe_gmm._remove_sharding_axis(None, "pcp") is None
    assert fused_moe_gmm._remove_sharding_axis(("data", ), "pcp") == (
        "data", )


def test_moe_data_axes_uses_full_mlp_data_without_pcp(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "MLP_DATA",
                        ("data", "pcp"),
                        raising=False)
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "PREFILL_CONTEXT",
                        "pcp",
                        raising=False)
    mesh = types.SimpleNamespace(shape={"data": 1, "pcp": 1})

    assert fused_moe_gmm._moe_data_axes(mesh) == ("data", "pcp")


def test_moe_data_axes_replicates_across_pcp(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "MLP_DATA",
                        ("data", "pcp"),
                        raising=False)
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "PREFILL_CONTEXT",
                        "pcp",
                        raising=False)
    mesh = types.SimpleNamespace(shape={"data": 1, "pcp": 2})

    assert fused_moe_gmm._moe_data_axes(mesh) == ("data", )


def test_attention_data_parallelism_ignores_pcp_only(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "ATTN_DATA",
                        ("data", "attn_dp", "pcp"),
                        raising=False)
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "MLP_DATA",
                        ("data", "pcp"),
                        raising=False)
    mesh = types.SimpleNamespace(shape={"data": 1, "attn_dp": 1, "pcp": 2})

    assert not fused_moe_gmm._has_attention_data_parallelism(mesh)


def test_attention_data_parallelism_detects_attention_dp(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "ATTN_DATA",
                        ("data", "attn_dp", "pcp"),
                        raising=False)
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "MLP_DATA",
                        ("data", "pcp"),
                        raising=False)
    mesh = types.SimpleNamespace(shape={"data": 1, "attn_dp": 2, "pcp": 2})

    assert fused_moe_gmm._has_attention_data_parallelism(mesh)


def test_token_partition_spec():
    assert fused_moe_gmm._token_partition_spec(("data", ), 2) == P(
        ("data", ), None)
    assert fused_moe_gmm._token_partition_spec(None, 1) == P(None)


def test_gather_pcp_sharded_tokens_noops_without_pcp(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm.ShardingAxisName,
                        "PREFILL_CONTEXT",
                        "pcp",
                        raising=False)
    mesh = types.SimpleNamespace(shape={"pcp": 1})
    value = object()

    assert fused_moe_gmm._gather_pcp_sharded_tokens(value, mesh) is value
