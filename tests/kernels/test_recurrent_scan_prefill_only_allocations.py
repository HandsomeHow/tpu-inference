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
"""Static allocation tests for recurrent_scan_v2 kernel variants."""

import jax.numpy as jnp

from tpu_inference.kernels.gdn.v2 import recurrent_scan_v2


class _FakeRef:

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeSemaphoreType:

    @staticmethod
    def DMA(shape):
        return ("DMA", tuple(shape))


def _call_fused_kernel(prefill_only):
    n_kq = 2
    n_v = 8
    d_k = 128
    d_v = 128
    C = 32
    BT = 32
    d = 2 * n_kq * d_k + n_v * d_v

    recurrent_scan_v2.fused_kernel(
        _FakeRef((C * 2, d), jnp.bfloat16),
        _FakeRef((4, n_v, d_k, d_v), jnp.float32),
        _FakeRef((4, ), jnp.int32),
        _FakeRef((4, ), jnp.int32),
        _FakeRef((C * 2, 128), jnp.bfloat16),
        _FakeRef((C * 2, 128), jnp.bfloat16),
        _FakeRef((n_v, ), jnp.float32),
        _FakeRef((n_v, ), jnp.float32),
        _FakeRef((4, 59), jnp.int32),
        [0],
        [1],
        _FakeRef((4, n_v, d_k, d_v), jnp.float32),
        _FakeRef((C * 2, n_v * d_v), jnp.bfloat16),
        C=C,
        BT=BT,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        use_qk_norm_in_gdn=True,
        sublanesize=16,
        prefill_only=prefill_only,
    )


def test_prefill_only_kernel_does_not_allocate_decode_scratch(monkeypatch):
    captured = {}

    def fake_vmem(shape, dtype):
        return ("VMEM", tuple(shape), dtype)

    def fake_run_scoped(fn, *resources):
        captured["fn_name"] = fn.__name__
        captured["resources"] = resources
        fn(*resources)

    def fake_emit_pipeline(body, grid, in_specs, out_specs):
        del body, grid
        captured["in_specs_len"] = len(in_specs)
        captured["out_specs_len"] = len(out_specs)

        def fake_pipeline(*args, scratches):
            del scratches
            captured["pipeline_arg_count"] = len(args)

        return fake_pipeline

    monkeypatch.setattr(recurrent_scan_v2.pltpu, "VMEM", fake_vmem)
    monkeypatch.setattr(recurrent_scan_v2.pltpu, "SemaphoreType",
                        _FakeSemaphoreType)
    monkeypatch.setattr(recurrent_scan_v2.pl, "run_scoped", fake_run_scoped)
    monkeypatch.setattr(recurrent_scan_v2.pltpu, "emit_pipeline",
                        fake_emit_pipeline)

    _call_fused_kernel(prefill_only=True)

    assert captured["fn_name"] == "_run_prefill_only_with_scratch"
    vmem_shapes = [
        resource[1] for resource in captured["resources"]
        if resource[0] == "VMEM"
    ]
    assert vmem_shapes == [
        (2, 8, 128, 128),  # prefill_scratch
        (1, 8, 128, 128),  # state_commit_scratch
    ]
    assert captured["in_specs_len"] == 5
    assert captured["out_specs_len"] == 1
    assert captured["pipeline_arg_count"] == 6


def test_default_kernel_keeps_decode_scratch(monkeypatch):
    captured = {}

    def fake_vmem(shape, dtype):
        return ("VMEM", tuple(shape), dtype)

    def fake_run_scoped(fn, *resources):
        captured["fn_name"] = fn.__name__
        captured["resources"] = resources

    monkeypatch.setattr(recurrent_scan_v2.pltpu, "VMEM", fake_vmem)
    monkeypatch.setattr(recurrent_scan_v2.pltpu, "SemaphoreType",
                        _FakeSemaphoreType)
    monkeypatch.setattr(recurrent_scan_v2.pl, "run_scoped", fake_run_scoped)

    _call_fused_kernel(prefill_only=False)

    assert captured["fn_name"] == "_run_with_scratch"
    vmem_shapes = [
        resource[1] for resource in captured["resources"]
        if resource[0] == "VMEM"
    ]
    assert vmem_shapes == [
        (2, 8, 128, 128),  # prefill_scratch
        (1, 8, 128, 128),  # decode_state_scratch
        (1, 8, 128, 128),  # state_commit_scratch
        (32, 1024),  # decode_output_scratch
    ]
