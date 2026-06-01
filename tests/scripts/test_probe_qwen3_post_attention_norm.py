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

from scripts.pcp.probe_qwen3_post_attention_norm import compare_norm_outputs


def _outputs(raw_residual, cast_residual):
    zeros = np.zeros((1, 2), dtype=np.float32)
    return {
        "input_hidden_states": zeros,
        "input_residual": zeros,
        "raw_hidden_states": zeros,
        "raw_residual": np.asarray(raw_residual, dtype=np.float32),
        "cast_hidden_states": zeros,
        "cast_residual": np.asarray(cast_residual, dtype=np.float32),
    }


def test_compare_norm_outputs_reports_current_engine_asymmetric_cast():
    baseline = _outputs([[1.0, 2.0]], [[1.0, 2.0]])
    candidate = _outputs([[1.0, 2.0]], [[1.0, 2.5]])

    diffs = compare_norm_outputs(baseline, candidate, topk=1)

    assert diffs["raw_outputs"]["residual"]["max_abs"] == 0.0
    assert diffs["as_current_engine"]["residual"]["max_abs"] == 0.5
    assert diffs["cast_effect_candidate"]["residual"]["max_abs"] == 0.5
