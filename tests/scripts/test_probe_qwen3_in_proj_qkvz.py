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

import numpy as np
import torch

from scripts.pcp.probe_qwen3_in_proj_qkvz import (
    build_synthetic_tensors,
    diff_stats,
    load_tensor_npy,
    parse_output_sizes,
    parse_torch_dtype,
    split_qkvz,
)


def test_parse_output_sizes():
    assert parse_output_sizes("2,4,6") == [2, 4, 6]


def test_parse_torch_dtype_accepts_bfloat16_aliases():
    assert parse_torch_dtype("bf16") is torch.bfloat16
    assert parse_torch_dtype("bfloat16") is torch.bfloat16


def test_build_synthetic_tensors_shapes_and_dtype():
    hidden_states, weight = build_synthetic_tensors(
        tokens=3,
        hidden_size=5,
        output_sizes=[2, 3],
        dtype=torch.bfloat16,
        seed=0,
        input_scale=0.1,
        weight_scale=0.2,
    )

    assert hidden_states.shape == (3, 5)
    assert weight.shape == (5, 5)
    assert hidden_states.dtype is torch.bfloat16
    assert weight.dtype is torch.bfloat16


def test_load_tensor_npy_casts_to_requested_dtype(tmp_path):
    path = tmp_path / "tensor.npy"
    np.save(path, np.array([[1.25, 2.5]], dtype=np.float32))

    tensor = load_tensor_npy(path, torch.bfloat16)

    assert tensor.shape == (1, 2)
    assert tensor.dtype is torch.bfloat16


def test_split_qkvz_returns_qkv_and_z_views():
    output = np.arange(2 * 12, dtype=np.float32).reshape(2, 12)

    parts = split_qkvz(output, [2, 2, 4, 4], head_v_dim=2)

    assert parts["mixed_qkv"].shape == (2, 8)
    assert parts["z_flat"].shape == (2, 4)
    assert parts["z"].shape == (2, 2, 2)
    np.testing.assert_array_equal(parts["q"], output[:, :2])
    np.testing.assert_array_equal(parts["z_flat"], output[:, 8:12])


def test_diff_stats_reports_max_and_top_diffs():
    baseline = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    candidate = np.array([[1.0, 2.5], [2.0, 4.0]], dtype=np.float32)

    stats = diff_stats(baseline, candidate, topk=1)

    assert stats["shape"] == [2, 2]
    assert stats["nonzero_count"] == 2
    assert stats["max_abs"] == 1.0
    assert stats["max_index"] == [1, 0]
    assert stats["top"] == [{
        "index": [1, 0],
        "baseline": 3.0,
        "candidate": 2.0,
        "abs_diff": 1.0,
    }]
