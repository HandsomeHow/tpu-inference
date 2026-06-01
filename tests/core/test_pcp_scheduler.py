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

from types import SimpleNamespace

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler

from tpu_inference.core.sched.pcp_scheduler import (
    PcpAwareScheduler, get_base_scheduler_cls,
    update_vllm_config_for_pcp_scheduler)


def _make_config(pcp_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size),
        scheduler_config=SimpleNamespace(scheduler_cls=Scheduler),
    )


def test_pcp_aware_scheduler_pauses_new_admissions_while_running(monkeypatch):
    scheduler = object.__new__(PcpAwareScheduler)
    scheduler.running = [object()]
    scheduler._pause_state = PauseState.UNPAUSED
    seen = {}

    def fake_schedule(self):
        seen["pause_state"] = self._pause_state
        return "scheduled"

    monkeypatch.setattr(Scheduler, "schedule", fake_schedule)

    assert scheduler.schedule() == "scheduled"
    assert seen["pause_state"] == PauseState.PAUSED_NEW
    assert scheduler._pause_state == PauseState.UNPAUSED


def test_pcp_aware_scheduler_leaves_empty_running_queue_unpaused(monkeypatch):
    scheduler = object.__new__(PcpAwareScheduler)
    scheduler.running = []
    scheduler._pause_state = PauseState.UNPAUSED
    seen = {}

    def fake_schedule(self):
        seen["pause_state"] = self._pause_state
        return "scheduled"

    monkeypatch.setattr(Scheduler, "schedule", fake_schedule)

    assert scheduler.schedule() == "scheduled"
    assert seen["pause_state"] == PauseState.UNPAUSED


def test_pcp_aware_scheduler_preserves_existing_pause_state(monkeypatch):
    scheduler = object.__new__(PcpAwareScheduler)
    scheduler.running = [object()]
    scheduler._pause_state = PauseState.PAUSED_ALL
    seen = {}

    def fake_schedule(self):
        seen["pause_state"] = self._pause_state
        return "scheduled"

    monkeypatch.setattr(Scheduler, "schedule", fake_schedule)

    assert scheduler.schedule() == "scheduled"
    assert seen["pause_state"] == PauseState.PAUSED_ALL
    assert scheduler._pause_state == PauseState.PAUSED_ALL


def test_update_vllm_config_for_pcp_scheduler_sets_scheduler_class():
    config = _make_config(pcp_size=2)

    update_vllm_config_for_pcp_scheduler(config)

    assert config.scheduler_config.scheduler_cls == PcpAwareScheduler


def test_pcp_scheduler_helpers_leave_non_pcp_config_unchanged():
    config = _make_config(pcp_size=1)

    assert get_base_scheduler_cls(config) == Scheduler
    update_vllm_config_for_pcp_scheduler(config)
    assert config.scheduler_config.scheduler_cls == Scheduler
