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

import torchax
from jax.sharding import Mesh, PartitionSpec
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)

from tpu_inference.layers.common.process_weights.linear_weights import \
    get_model_matmul_fusion_assignment
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  get_moe_expert_shard_axis)
from tpu_inference.utils import TPU_SECOND_LAST_MINOR, get_mesh_shape_product

# yapf: enable

P = PartitionSpec

logger = init_logger(__name__)


class VllmQuantLinearConfig(QuantLinearConfig):

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh, layer: LinearBase):
        assert isinstance(layer, LinearBase)

        super().__init__(
            enable_sp=vllm_config.compilation_config.pass_config.enable_sp,
            output_sizes=[layer.output_size])
        self.mesh = mesh
        self.tp_size = get_mesh_shape_product(self.mesh,
                                              ShardingAxisName.MLP_TENSOR)

        self.n_shards = get_mesh_shape_product(self.mesh,
                                               self.weight_sharding[0])

        if isinstance(layer, RowParallelLinear):
            self.weight_sharding = P(None, ShardingAxisName.ATTN_HEAD)
            if self.enable_sp:
                self.output_sharding = P(ShardingAxisName.MLP_TENSOR, None)
        elif isinstance(layer, ColumnParallelLinear):
            self.weight_sharding = P(ShardingAxisName.ATTN_HEAD, None)

            if self.enable_sp:
                self.input_sharding = P(ShardingAxisName.MLP_TENSOR, None)

            if isinstance(layer, MergedColumnParallelLinear) or isinstance(
                    layer, QKVParallelLinear):
                self.output_sizes = layer.output_sizes

            self.fuse_matmuls = get_model_matmul_fusion_assignment(
                vllm_config.model_config.model,
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.parallel_config.tensor_parallel_size,
                layer._get_name())
        elif isinstance(layer, ReplicatedLinear):
            self.weight_sharding = P(None, None)
        else:
            logger.warning(
                "Unsupported linear layer type of %s. Can potentially yield "
                " bad performance.", type(layer))

        if isinstance(layer, QKVParallelLinear):
            self.num_proj = 3
        elif isinstance(layer, MergedColumnParallelLinear):
            self.num_proj = len(layer.output_sizes)
        else:
            self.num_proj = 1

        self.bias_sharding = P(self.weight_sharding[0])
        self.n_shards = get_mesh_shape_product(self.mesh,
                                               self.weight_sharding[0])

    def get_input_sharding(self, x: torchax.tensor.Tensor):
        if not self.enable_sp:
            return None
        token_num = x.shape[0]
        # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
        if token_num // self.tp_size < TPU_SECOND_LAST_MINOR:
            return None
        return self.input_sharding

    def get_output_sharding(self, x: torchax.tensor.Tensor):
        if self.enable_sp:
            token_num = x.shape[0]
            # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
            if token_num // self.tp_size < TPU_SECOND_LAST_MINOR:
                return None
        return self.output_sharding


class VllmQuantConfig:
    vllm_config: VllmConfig
    mesh: Mesh

    @classmethod
    def set_configs(cls, vllm_config: VllmConfig, mesh: Mesh):
        cls.vllm_config = vllm_config
        cls.mesh = mesh

    def get_linear_config(self, layer: LinearBase) -> VllmQuantLinearConfig:
        assert isinstance(layer, LinearBase)
        return VllmQuantLinearConfig(self.vllm_config, self.mesh, layer)

    def get_moe_config(self, layer: FusedMoE) -> FusedMoEConfig:
        assert isinstance(layer, FusedMoE)
        moe_config = layer.moe_config
        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        parallel_config = moe_config.moe_parallel_config
        parallel_config.use_ep = use_ep
        if use_ep:
            expert_axis = get_moe_expert_shard_axis(self.mesh)
            mesh_ep_size = get_mesh_shape_product(self.mesh, expert_axis)
            if parallel_config.ep_size != 1:
                logger.info_once(
                    "Disabling vLLM-side MoE expert slicing; TPU MoE EP "
                    "will shard full expert weights over axis %s with size "
                    "%s.", expert_axis, mesh_ep_size)
            parallel_config.tp_size = 1
            parallel_config.tp_rank = 0
            parallel_config.ep_size = 1
            parallel_config.ep_rank = 0

            layer.moe_parallel_config = parallel_config
            layer.expert_map_manager.moe_parallel_config = parallel_config
            layer.expert_map_manager.global_num_experts = layer.global_num_experts
            layer.expert_map_manager._placement_strategy = (
                layer.expert_map_manager._determine_placement_strategy(
                    layer.expert_placement_strategy))
            layer.expert_map_manager._calculate_expert_maps()
            layer.expert_map_manager._routing_tables = (
                layer.expert_map_manager._init_routing_tables())
            layer.expert_map_manager._init_aiter_shared_experts_topK_buffer()
            layer.update_expert_map_info()
            moe_config.num_local_experts = layer.local_num_experts
            moe_config.moe_parallel_config = parallel_config
        return moe_config
