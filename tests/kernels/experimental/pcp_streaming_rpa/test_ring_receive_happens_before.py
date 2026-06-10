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

import ast
import inspect
import textwrap

import pytest

from tpu_inference.kernels.experimental.pcp_streaming_rpa import kernel

PCP_SIZE = 4
CURR_SLOT = 0
NEXT_SLOT = 1
STALE_VALUE = "stale"


def _initial_slots():
    return [[f"payload-from-rank-{rank}", STALE_VALUE]
            for rank in range(PCP_SIZE)]


def _complete_copy(slots, src_rank):
    dst_rank = (src_rank + 1) % PCP_SIZE
    slots[dst_rank][NEXT_SLOT] = slots[src_rank][CURR_SLOT]


def _consume_after_outgoing_wait_only(delayed_src_rank):
    slots = _initial_slots()

    for src_rank in range(PCP_SIZE):
        if src_rank != delayed_src_rank:
            _complete_copy(slots, src_rank)

    # This mirrors the old kernel contract: rank 0 only knows that its own
    # outgoing copy to rank 1 has completed. It has no happens-before edge from
    # rank 3's outgoing copy into rank 0's NEXT_SLOT.
    assert delayed_src_rank == PCP_SIZE - 1
    return slots[0][NEXT_SLOT]


def _consume_after_round_receive_barrier(delayed_src_rank):
    slots = _initial_slots()

    for src_rank in range(PCP_SIZE):
        if src_rank != delayed_src_rank:
            _complete_copy(slots, src_rank)

    _complete_copy(slots, delayed_src_rank)

    # The added ring barrier is the happens-before edge that prevents rank 0
    # from consuming NEXT_SLOT until every rank's outgoing copy has completed.
    return slots[0][NEXT_SLOT]


def test_outgoing_wait_only_can_consume_stale_incoming_slot():
    assert _consume_after_outgoing_wait_only(
        delayed_src_rank=PCP_SIZE - 1) == STALE_VALUE


def test_round_receive_barrier_orders_incoming_slot_before_consume():
    assert _consume_after_round_receive_barrier(
        delayed_src_rank=PCP_SIZE - 1) == "payload-from-rank-3"


def _contains_remote_op_wait(node):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "wait"
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "remote_op"
        for child in ast.walk(node)
    )


def _contains_local_barrier(node):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "local_barrier"
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "util"
        for child in ast.walk(node)
    )


def _iter_round_loops(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "round_idx"):
            yield node


def _assert_remote_waits_are_followed_by_ring_barriers(fn):
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    wait_sites = 0

    for loop in _iter_round_loops(tree):
        for stmt in loop.body:
            body = getattr(stmt, "body", ())
            for idx, child in enumerate(body):
                if not _contains_remote_op_wait(child):
                    continue
                wait_sites += 1
                if not any(_contains_local_barrier(later)
                           for later in body[idx + 1:]):
                    raise AssertionError(
                        "PCP ring remote_op.wait() is not followed by "
                        "util.local_barrier().")

    assert wait_sites > 0, "No PCP ring remote wait site found."


def test_kernel_contract_catches_outgoing_wait_without_receive_barrier():
    def old_outgoing_wait_only_ring_loop():
        for round_idx in range(pcp_size):
            if round_idx < pcp_size - 1:
                remote_op.wait()

    with pytest.raises(AssertionError, match="not followed"):
        _assert_remote_waits_are_followed_by_ring_barriers(
            old_outgoing_wait_only_ring_loop)


def test_pcp_streaming_kernels_wait_for_incoming_ring_copy_before_next_round():
    _assert_remote_waits_are_followed_by_ring_barriers(
        kernel._pcp_streaming_attention_page_groups_kernel)
    _assert_remote_waits_are_followed_by_ring_barriers(
        kernel._pcp_streaming_attention_page_groups_multi_head_kernel)
