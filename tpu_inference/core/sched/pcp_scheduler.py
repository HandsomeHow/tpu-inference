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

from typing import Any

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler


def pcp_enabled(vllm_config: Any) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    return isinstance(pcp_size, int) and pcp_size > 1


class PcpAwareScheduler(Scheduler):
    """Scheduler wrapper that avoids PCP mixed prefill/decode batches.

    TPU PCP currently has separate attention paths for initial prefill
    (local-Q/full-KV) and decode (replicated-Q/local-KV + LSE merge). vLLM's
    base scheduler can schedule running decode requests and newly admitted
    prefill requests in the same step. Until TPU PCP has a true mixed-mode
    attention path, prevent new admissions while any request is already
    running; existing running work can still make progress normally.
    """

    def schedule(self):
        if self.running and self._pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
            try:
                return super().schedule()
            finally:
                self._pause_state = PauseState.UNPAUSED
        return super().schedule()


def get_base_scheduler_cls(vllm_config: Any) -> type[Scheduler]:
    return PcpAwareScheduler if pcp_enabled(vllm_config) else Scheduler


def update_vllm_config_for_pcp_scheduler(vllm_config: Any) -> None:
    if pcp_enabled(vllm_config):
        vllm_config.scheduler_config.scheduler_cls = PcpAwareScheduler
