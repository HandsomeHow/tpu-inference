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

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.config.multimodal import BaseDummyOptions
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField, unpack_pcp_streaming_schedule_field)
from tpu_inference.models.common.interface import (ModelInterface,
                                                   MultiModalInterface)
from tpu_inference.runner.tpu_runner import (TPUModelRunner,
                                             _attention_metadata_uses_pcp,
                                             _apply_pcp_rank_major_token_order,
                                             _batch_uses_pcp_decode,
                                             _batch_uses_pcp_prefill,
                                             _build_pcp_decode_attention_metadata,
                                             _build_pcp_attention_metadata,
                                             _build_pcp_logits_indices,
                                             _build_pcp_rank_major_token_order,
                                             _kv_cache_group_supports_pcp_attention_metadata,
                                             _logits_indices_require_global_gather,
                                             _pcp_local_token_counts)


class TestPCPTokenPacking:

    def test_local_token_counts_balanced_for_interleaved_chunks(self):
        counts = _pcp_local_token_counts([8], pcp_size=2, interleave_size=2)

        np.testing.assert_array_equal(counts, np.array([4, 4], dtype=np.int32))

    def test_local_token_counts_handles_multiple_partial_requests(self):
        counts = _pcp_local_token_counts([5, 7],
                                         pcp_size=3,
                                         interleave_size=2)

        np.testing.assert_array_equal(counts,
                                      np.array([5, 4, 3], dtype=np.int32))

    def test_local_token_counts_use_global_start_offsets(self):
        counts = _pcp_local_token_counts(
            [4],
            pcp_size=2,
            interleave_size=2,
            token_start_offsets_per_req=[6],
        )

        np.testing.assert_array_equal(counts,
                                      np.array([2, 2], dtype=np.int32))

    def test_rank_major_token_order_single_request(self):
        order, inverse = _build_pcp_rank_major_token_order(
            [8],
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=8,
        )

        np.testing.assert_array_equal(order, np.array([0, 1, 4, 5, 2, 3, 6,
                                                       7]))
        np.testing.assert_array_equal(inverse,
                                      np.array([0, 1, 4, 5, 2, 3, 6, 7]))

    def test_rank_major_token_order_multiple_requests_with_padding(self):
        order, inverse = _build_pcp_rank_major_token_order(
            [5, 3],
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=12,
        )

        np.testing.assert_array_equal(
            order, np.array([0, 1, 4, 5, 6, -1, 2, 3, 7, -1, -1, -1]))
        np.testing.assert_array_equal(inverse,
                                      np.array([0, 1, 6, 7, 2, 3, 4, 8]))

    def test_rank_major_token_order_uses_global_start_offsets(self):
        order, inverse = _build_pcp_rank_major_token_order(
            [4],
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=4,
            token_start_offsets_per_req=[6],
        )

        np.testing.assert_array_equal(order, np.array([2, 3, 0, 1]))
        np.testing.assert_array_equal(inverse, np.array([2, 3, 0, 1]))

    def test_rank_major_token_order_rejects_non_divisible_padding(self):
        with pytest.raises(ValueError, match="must be divisible"):
            _build_pcp_rank_major_token_order(
                [8],
                pcp_size=2,
                interleave_size=2,
                padded_num_tokens=9,
            )

    def test_apply_rank_major_token_order_reorders_inputs_positions_and_mrope(
            self):
        input_ids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0],
                             dtype=np.int32)
        positions = np.array([0, 1, 2, 3, 4, 0, 1, 2, 0, 0, 0, 0],
                             dtype=np.int32)
        mrope = np.stack([positions, positions + 100, positions + 200])

        inverse = _apply_pcp_rank_major_token_order(
            input_ids,
            positions,
            [5, 3],
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=12,
            mrope_positions_cpu=mrope,
        )

        np.testing.assert_array_equal(
            input_ids,
            np.array([10, 11, 14, 15, 16, 0, 12, 13, 17, 0, 0, 0],
                     dtype=np.int32))
        np.testing.assert_array_equal(
            positions,
            np.array([0, 1, 4, 0, 1, 0, 2, 3, 2, 0, 0, 0], dtype=np.int32))
        np.testing.assert_array_equal(
            mrope[1],
            np.array([100, 101, 104, 100, 101, 0, 102, 103, 102, 0, 0, 0]))
        np.testing.assert_array_equal(inverse,
                                      np.array([0, 1, 6, 7, 2, 3, 4, 8]))

    def test_build_logits_indices_adds_dp_token_offset(self):
        logits_indices = _build_pcp_logits_indices(
            [5, 3],
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=12,
            token_offset=24,
        )

        np.testing.assert_array_equal(logits_indices, np.array([26, 32]))

    def test_build_attention_metadata_single_request(self):
        metadata = _build_pcp_attention_metadata(
            num_scheduled_tokens_per_req=[8],
            seq_lens_per_req=[8],
            block_tables=np.array([[7, 8]], dtype=np.int32),
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=8,
            max_num_reqs_per_dp_rank=1,
            block_size=4,
        )

        np.testing.assert_array_equal(
            metadata.slot_ids,
            np.array([28, 29, 30, 31, 28, 29, 30, 31], dtype=np.int32))
        assert metadata.streaming_schedule is None
        assert metadata.streaming_active_page_groups is None

    def test_build_attention_metadata_adds_streaming_schedule(self):
        metadata = _build_pcp_attention_metadata(
            num_scheduled_tokens_per_req=[8],
            seq_lens_per_req=[8],
            block_tables=np.array([[7, 8]], dtype=np.int32),
            pcp_size=2,
            interleave_size=4,
            padded_num_tokens=8,
            max_num_reqs_per_dp_rank=1,
            block_size=4,
            build_streaming_schedule=True,
            streaming_num_lanes=1,
            streaming_q_block_size=2,
        )

        schedule = metadata.streaming_schedule
        assert schedule is not None
        assert schedule.shape == (4, 2, 1, ScheduleField.PACKED_NUM_FIELDS)
        np.testing.assert_array_equal(metadata.streaming_active_page_groups,
                                      np.array([2], dtype=np.int32))
        req_id = unpack_pcp_streaming_schedule_field(schedule,
                                                     ScheduleField.REQ_ID)
        kv_page_idx = unpack_pcp_streaming_schedule_field(
            schedule, ScheduleField.KV_PAGE_IDX)
        assert np.any(req_id != -1)
        # Streaming schedule consumes the local virtual PCP block table.
        np.testing.assert_array_equal(np.unique(kv_page_idx[req_id != -1]),
                                      np.array([7], dtype=np.int32))

    def test_build_attention_metadata_pads_streaming_schedule_to_stable_shape(
            self):
        block_tables = np.zeros((8, 2064), dtype=np.int32)
        first_chunk = _build_pcp_attention_metadata(
            num_scheduled_tokens_per_req=[4096],
            seq_lens_per_req=[4096],
            block_tables=block_tables,
            pcp_size=8,
            interleave_size=32,
            padded_num_tokens=4096,
            max_num_reqs_per_dp_rank=8,
            block_size=32,
            build_streaming_schedule=True,
            streaming_num_lanes=1,
            streaming_q_block_size=32,
        )
        second_chunk = _build_pcp_attention_metadata(
            num_scheduled_tokens_per_req=[4096],
            seq_lens_per_req=[8192],
            block_tables=block_tables,
            pcp_size=8,
            interleave_size=32,
            padded_num_tokens=4096,
            max_num_reqs_per_dp_rank=8,
            block_size=32,
            build_streaming_schedule=True,
            streaming_num_lanes=1,
            streaming_q_block_size=32,
        )

        assert first_chunk.streaming_schedule is not None
        assert second_chunk.streaming_schedule is not None
        assert (first_chunk.streaming_schedule.shape ==
                second_chunk.streaming_schedule.shape)
        assert (first_chunk.streaming_active_page_groups[0] <
                second_chunk.streaming_active_page_groups[0])
        assert (second_chunk.streaming_active_page_groups[0] <
                second_chunk.streaming_schedule.shape[0] // 8)

    def test_build_attention_metadata_chunked_prefill_continuation(self):
        metadata = _build_pcp_attention_metadata(
            num_scheduled_tokens_per_req=[4],
            seq_lens_per_req=[12],
            block_tables=np.array([[7, 8, 9]], dtype=np.int32),
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=4,
            max_num_reqs_per_dp_rank=1,
            block_size=4,
        )

        np.testing.assert_array_equal(metadata.slot_ids,
                                      np.array([32, 33, 32, 33],
                                               dtype=np.int32))

    def test_build_attention_metadata_pads_empty_pcp_rank_slot_ids(self):
        metadata = _build_pcp_attention_metadata(
            num_scheduled_tokens_per_req=[4],
            seq_lens_per_req=[12],
            block_tables=np.array([[7, 8, 9]], dtype=np.int32),
            pcp_size=2,
            interleave_size=4,
            padded_num_tokens=16,
            max_num_reqs_per_dp_rank=1,
            block_size=8,
        )

        local_padded_tokens = 8
        np.testing.assert_array_equal(
            metadata.slot_ids[:local_padded_tokens],
            np.array([60, 61, 62, 63, -1, -1, -1, -1], dtype=np.int32))
        np.testing.assert_array_equal(
            metadata.slot_ids[local_padded_tokens:],
            np.full(local_padded_tokens, -1, dtype=np.int32))

    def test_build_attention_metadata_rejects_invalid_seq_lens(self):
        with pytest.raises(ValueError, match="seq_lens_per_req"):
            _build_pcp_attention_metadata(
                num_scheduled_tokens_per_req=[4],
                seq_lens_per_req=[3],
                block_tables=np.array([[3]], dtype=np.int32),
                pcp_size=2,
                interleave_size=2,
                padded_num_tokens=4,
                max_num_reqs_per_dp_rank=1,
                block_size=4,
            )

    def test_build_pcp_decode_attention_metadata(self):
        metadata = _build_pcp_decode_attention_metadata(
            seq_lens_per_req=[5, 4],
            block_tables=np.array([[7, 8], [3, 4]], dtype=np.int32),
            block_size=4,
            pcp_size=2,
            interleave_size=2,
            padded_num_tokens=4,
            max_num_reqs_per_dp_rank=2,
        )

        np.testing.assert_array_equal(
            metadata["source_block_tables"],
            np.array([[7], [3]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            metadata["slot_ids"],
            np.array([30, -1, -1, -1, -1, 13, -1, -1], dtype=np.int32),
        )

    def test_build_pcp_decode_attention_metadata_multi_token_continuation(self):
        metadata = _build_pcp_decode_attention_metadata(
            seq_lens_per_req=[12],
            block_tables=np.array([[7, 8, 9]], dtype=np.int32),
            block_size=8,
            pcp_size=2,
            interleave_size=4,
            padded_num_tokens=8,
            max_num_reqs_per_dp_rank=1,
            num_scheduled_tokens_per_req=[4],
        )

        np.testing.assert_array_equal(
            metadata["source_block_tables"],
            np.array([[7, 8]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            metadata["slot_ids"],
            np.array([60, 61, 62, 63, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1],
                     dtype=np.int32),
        )

    def test_build_pcp_decode_attention_metadata_pcp8_nonzero_owner_rank(self):
        block_tables = np.array(
            [
                [17, 3, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                 111, 112, 113, 114],
                [5, 29, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
                 211, 212, 213, 214],
                [11, 23, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
                 311, 312, 313, 314],
                [41, 43, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
                 411, 412, 413, 414],
            ],
            dtype=np.int32,
        )

        metadata = _build_pcp_decode_attention_metadata(
            seq_lens_per_req=[6, 31, 130],
            block_tables=block_tables,
            block_size=16,
            pcp_size=8,
            interleave_size=4,
            padded_num_tokens=8,
            max_num_reqs_per_dp_rank=4,
        )

        np.testing.assert_array_equal(
            metadata["source_block_tables"],
            np.array(
                [
                    [17, 3],
                    [5, 29],
                    [11, 23],
                    [41, 43],
                ],
                dtype=np.int32,
            ),
        )
        expected_slot_ids = np.full(64, -1, dtype=np.int32)
        expected_slot_ids[2] = 23 * 16 + 1
        expected_slot_ids[8] = 17 * 16 + 1
        expected_slot_ids[57] = 5 * 16 + 2
        np.testing.assert_array_equal(metadata["slot_ids"],
                                      expected_slot_ids)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"block_size": 0}, "block_size > 0"),
            ({"pcp_size": 1}, "pcp_size > 1"),
            ({"interleave_size": 0}, "interleave_size > 0"),
            ({"interleave_size": 3}, "block_size % interleave_size"),
            ({"seq_lens_per_req": [33]}, "block table capacity"),
        ],
    )
    def test_build_pcp_decode_attention_metadata_rejects_invalid_args(
            self, kwargs, match):
        args = {
            "seq_lens_per_req": [5],
            "block_tables": np.array([[7, 8]], dtype=np.int32),
            "block_size": 16,
            "pcp_size": 2,
            "interleave_size": 4,
            "padded_num_tokens": 4,
            "max_num_reqs_per_dp_rank": 1,
        }
        args.update(kwargs)

        with pytest.raises(ValueError, match=match):
            _build_pcp_decode_attention_metadata(**args)

    def test_pcp_attention_metadata_filter_skips_mamba_groups(self):
        attention_group = MagicMock()
        attention_group.kv_cache_spec = FullAttentionSpec(block_size=16,
                                                          num_kv_heads=1,
                                                          head_size=64,
                                                          dtype=torch.bfloat16)
        mamba_group = MagicMock()
        mamba_group.kv_cache_spec = MambaSpec(block_size=16,
                                             shapes=((1, ), ),
                                             dtypes=(torch.float32, ))

        assert _kv_cache_group_supports_pcp_attention_metadata(attention_group)
        assert not _kv_cache_group_supports_pcp_attention_metadata(mamba_group)


class TestPCPBatchSelection:

    @staticmethod
    def _vllm_config(pcp_size=2, interleave_size=2):
        vllm_config = MagicMock()
        vllm_config.parallel_config.prefill_context_parallel_size = pcp_size
        vllm_config.parallel_config.cp_kv_cache_interleave_size = interleave_size
        return vllm_config

    @staticmethod
    def _input_batch(req_ids, computed_tokens, prompt_tokens=None):
        input_batch = MagicMock()
        input_batch.req_ids = req_ids
        input_batch.req_id_to_index = {
            req_id: idx for idx, req_id in enumerate(req_ids)
        }
        input_batch.num_computed_tokens_cpu = np.array(computed_tokens,
                                                       dtype=np.int32)
        if prompt_tokens is None:
            prompt_tokens = computed_tokens
        input_batch.num_prompt_tokens = np.array(prompt_tokens, dtype=np.int32)
        return input_batch

    @staticmethod
    def _scheduler(num_scheduled_tokens):
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = num_scheduled_tokens
        return scheduler_output

    def test_initial_prefill_batch_uses_pcp(self):
        assert _batch_uses_pcp_prefill(
            self._vllm_config(),
            self._input_batch(["req1", "req2"], [0, 0], [8, 4]),
            self._scheduler({
                "req1": 8,
                "req2": 4
            }),
            num_reqs=2,
        )

    def test_single_token_prompt_continuation_uses_pcp_decode(self):
        input_batch = self._input_batch(["req1"], [4096], [4097])
        scheduler_output = self._scheduler({"req1": 1})

        assert not _batch_uses_pcp_prefill(
            self._vllm_config(),
            input_batch,
            scheduler_output,
            num_reqs=1,
        )
        assert _batch_uses_pcp_decode(
            self._vllm_config(),
            input_batch,
            scheduler_output,
            num_reqs=1,
        )

    def test_chunked_prompt_continuation_multiple_tokens_uses_pcp_prefill(
            self):
        assert not _batch_uses_pcp_prefill(
            self._vllm_config(pcp_size=1),
            self._input_batch(["req1"], [4096], [4100]),
            self._scheduler({"req1": 4}),
            num_reqs=1,
        )
        input_batch = self._input_batch(["req1"], [4096], [4100])
        scheduler_output = self._scheduler({"req1": 4})
        assert _batch_uses_pcp_prefill(
            self._vllm_config(),
            input_batch,
            scheduler_output,
            num_reqs=1,
        )
        assert not _batch_uses_pcp_decode(
            self._vllm_config(),
            input_batch,
            scheduler_output,
            num_reqs=1,
        )

    def test_chunked_prefill_late_64k_chunk_uses_pcp_prefill(self):
        input_batch = self._input_batch(["req1"], [57344], [65536])
        scheduler_output = self._scheduler({"req1": 4096})

        assert _batch_uses_pcp_prefill(
            self._vllm_config(),
            input_batch,
            scheduler_output,
            num_reqs=1,
        )
        assert not _batch_uses_pcp_decode(
            self._vllm_config(),
            input_batch,
            scheduler_output,
            num_reqs=1,
        )

    def test_pcp_disabled_in_config_does_not_use_pcp(self):
        assert not _batch_uses_pcp_prefill(
            self._vllm_config(pcp_size=1),
            self._input_batch(["req1"], [0], [8]),
            self._scheduler({"req1": 8}),
            num_reqs=1,
        )

    def test_decode_batch_uses_pcp_decode(self):
        assert _batch_uses_pcp_decode(
            self._vllm_config(),
            self._input_batch(["req1", "req2"], [8, 10], [8, 10]),
            self._scheduler({
                "req1": 1,
                "req2": 1
            }),
            num_reqs=2,
        )
        assert not _batch_uses_pcp_decode(
            self._vllm_config(),
            self._input_batch(["req1", "req2"], [0, 0], [8, 1]),
            self._scheduler({
                "req1": 8,
                "req2": 1
            }),
            num_reqs=2,
        )

    def test_attention_metadata_uses_pcp_checks_actual_metadata(self):
        normal_md = AttentionMetadata(input_positions=jnp.array([0]))
        pcp_md = AttentionMetadata(input_positions=jnp.array([0]),
                                   pcp_slot_ids=jnp.array([0]))
        pcp_decode_md = AttentionMetadata(input_positions=jnp.array([0]),
                                          pcp_slot_ids=jnp.array([0]))

        assert not _attention_metadata_uses_pcp(normal_md)
        assert _attention_metadata_uses_pcp(pcp_md)
        assert _attention_metadata_uses_pcp(pcp_decode_md)
        assert _attention_metadata_uses_pcp({"layer.0": normal_md,
                                             "layer.1": pcp_decode_md})

    def test_pcp_config_uses_global_logits_indices_for_mixed_batches(self):
        normal_md = AttentionMetadata(input_positions=jnp.array([0]))

        assert _logits_indices_require_global_gather(
            self._vllm_config(pcp_size=2), normal_md)
        assert not _logits_indices_require_global_gather(
            self._vllm_config(pcp_size=1), normal_md)


class TestTPUJaxRunner:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=MagicMock()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            speculative_config = SpeculativeConfig(
                model='ngram',
                num_speculative_tokens=5,
                prompt_lookup_max=4,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=speculative_config,
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)

    def test_get_supported_tasks_runner(self):
        """Test get_supported_tasks for generate runner type."""
        supported_tasks = self.runner.get_supported_tasks()
        assert supported_tasks == ("generate", )

    def test_get_input_ids_embeds(self):
        """Tests _get_input_ids_embeds for both multimodal and text-only models."""
        # 1. ===== Setup =====
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = [jnp.ones((10, 128))]
        dummy_is_mm_embed = jnp.array([False, True, True], dtype=jnp.bool_)
        dummy_final_embeds = jnp.ones((3, 128))

        # Mock the embedding function
        self.mock_get_input_embed_fn = MagicMock()
        self.runner.embed_input_ids_fn = self.mock_get_input_embed_fn
        self.mock_get_input_embed_fn.return_value = dummy_final_embeds
        self.runner.state_leaves = MagicMock()

        # 2. ===== Act & Assert (Multimodal) =====
        self.runner.is_multimodal_model = True

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds, dummy_is_mm_embed)

        assert input_ids_res is None
        np.testing.assert_array_equal(np.asarray(inputs_embeds_res),
                                      np.asarray(dummy_final_embeds))
        self.mock_get_input_embed_fn.assert_called_once_with(
            self.runner.state_leaves,
            dummy_input_ids,
            dummy_mm_embeds,
            is_multimodal=dummy_is_mm_embed)

        # 3. ===== Act & Assert (Multimodal w/o mm embeds) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = True

        # Without mm_embeds in the current scheduled tokens
        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, None, None)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

        # 4. ===== Act & Assert (Text-only) =====
        self.mock_get_input_embed_fn.reset_mock()
        self.runner.is_multimodal_model = False

        input_ids_res, inputs_embeds_res = self.runner._get_input_ids_embeds(
            dummy_input_ids, dummy_mm_embeds, dummy_is_mm_embed)

        assert inputs_embeds_res is None
        np.testing.assert_array_equal(np.asarray(input_ids_res),
                                      np.asarray(dummy_input_ids))
        self.mock_get_input_embed_fn.assert_not_called()

    @patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
    def test_prepare_inputs_hybrid_kvcache(self, mock_sampling_metadata):
        # create hybrid kv cache config
        # 20 layers, 10 full attn + 10 sw attn
        self._create_mock_hybrid_kv_cache_config()

        # Mock scheduler output.
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10
        scheduler_output.num_scheduled_tokens = {'req1': 10}
        scheduler_output.scheduled_spec_decode_tokens = {}
        scheduler_output.grammar_bitmask = None

        # Mock input_batch
        self.runner.input_batch = MagicMock()
        self.runner.input_batch.num_reqs = 1
        self.runner.input_batch.req_ids = ['req1']
        self.runner.input_batch.req_id_to_index = {'req1': 0}
        self.runner.input_batch.num_computed_tokens_cpu = np.array([10])
        self.runner.input_batch.token_ids_cpu = np.random.randint(
            0, 1000, (8, 64), dtype=np.int32)
        # Concrete numpy array so `.copy()` returns a real ndarray
        # (otherwise the surrounding `device_array` introspection on
        # `MagicMock` recurses on every `dtype` access).
        self.runner.input_batch.mamba_state_indices_cpu = np.zeros(
            self.runner.max_num_reqs, dtype=np.int32)

        # Mock block tables
        # there will be 2 block tables since there are 2 kv cache groups
        mock_block_table = MagicMock()
        mock_block_table.max_num_blocks_per_req = 8
        mock_block_table.get_cpu_tensor.return_value = np.zeros((1, 8),
                                                                dtype=np.int32)
        self.runner.input_batch.block_table = [
            mock_block_table, mock_block_table
        ]

        mock_sampling_instance = MagicMock()
        mock_sampling_metadata.from_input_batch.return_value = mock_sampling_instance

        output = self.runner._prepare_inputs(scheduler_output)
        assert len(output) == 10
        input_ids, positions, attention_metadata, sampling_metadata, logits_indices, spec_decode_metadata, logits_indices_selector, padded_num_reqs, req_ids_dp, padded_num_scheduled_tokens_per_dp_rank = output
        # assert it will create attention metadata for each layer.
        assert isinstance(attention_metadata, dict)
        assert len(attention_metadata) == 20

    def _create_mock_hybrid_kv_cache_config(self):
        mock_kv_cache_config = MagicMock()
        mock_kv_cache_group1 = MagicMock()
        mock_kv_cache_group1.layer_names = [f'layer.{i}' for i in range(10)]
        mock_kv_cache_group2 = MagicMock()
        mock_kv_cache_group2.layer_names = [
            f'layer.{i}' for i in range(10, 20)
        ]
        mock_kv_cache_config.kv_cache_groups = [
            mock_kv_cache_group1, mock_kv_cache_group2
        ]
        mock_kv_cache_config.has_mamba_layers = False
        self.runner.kv_cache_config = mock_kv_cache_config
        self.runner.use_hybrid_kvcache = True


class TestTPUJaxRunnerMultimodalModelLoadedForTextOnly:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        # Setup the runner with the model_config.is_multimodal_model set to True but get_model returning None for embed_multimodal_fn and embed_input_ids_fn.
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=self._model_get_model()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh), \
             patch('jax.device_put', side_effect=lambda x, *args, **kwargs: x):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            # Set multimodal_config to not None, such that the is_multimodal_model property of model_config is True.
            model_config.multimodal_config = MagicMock()

            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)
            self.runner.load_model()

    def _model_get_model(self):
        mock_multimodal_fns = MultiModalInterface(
            precompile_vision_encoder_fn=None,
            embed_multimodal_fn=None,
            embed_input_ids_fn=None,
            get_mrope_input_positions_fn=None)
        return ModelInterface(
            model_fn=MagicMock(),
            compute_logits_fn=MagicMock(),
            pooler_fn=MagicMock(),
            combine_hidden_states_fn=MagicMock(),
            multimodal_fns=mock_multimodal_fns,
            state=MagicMock(),
            state_leaves=MagicMock(),
            lora_manager=None,
            model=None,
        )

    def test_is_multimodal_model(self):
        # Precondition: make sure the model_config claims the model supports MM.
        assert self.runner.model_config.is_multimodal_model

        # Precondition: load the model and returns embed_multimodal_fn as None.
        assert self.runner.embed_multimodal_fn is None

        assert not self.runner.is_multimodal_model

        self.runner.embed_input_ids_fn = MagicMock()
        dummy_input_ids = jnp.array([1, 2, 3])
        dummy_mm_embeds = [jnp.ones((10, 128))]
        dummy_is_mm_embed = jnp.array([False, True, True], dtype=jnp.bool_)
        _ = self.runner._get_input_ids_embeds(dummy_input_ids, dummy_mm_embeds,
                                              dummy_is_mm_embed)
        self.runner.embed_input_ids_fn.assert_not_called()


class TestTPUJaxRunnerDisableMM:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(4)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))

    def _model_get_model(self):
        mock_multimodal_fns = MultiModalInterface(
            precompile_vision_encoder_fn=None,
            embed_multimodal_fn=MagicMock(),
            embed_input_ids_fn=MagicMock(),
            get_mrope_input_positions_fn=None)
        return ModelInterface(
            model_fn=MagicMock(),
            compute_logits_fn=MagicMock(),
            pooler_fn=MagicMock(),
            combine_hidden_states_fn=MagicMock(),
            multimodal_fns=mock_multimodal_fns,
            state=MagicMock(),
            state_leaves=MagicMock(),
            lora_manager=None,
            model=None,
        )

    @pytest.mark.parametrize(
        "limit_per_prompt, still_mm_after_loading_model",
        [
            ({
                "image": BaseDummyOptions(count=0),
                "video": BaseDummyOptions(count=0)
            }, False),
            ({
                "video": BaseDummyOptions(count=0)
            }, False),
            ({
                "image": BaseDummyOptions(count=0),
                "video": BaseDummyOptions(count=1)
            }, True),
            (
                {
                    # Empty limit means no limit, which should not disable MM.
                },
                True)
        ])
    def test_multimodal_model_loading_with_limits(
            self, limit_per_prompt, still_mm_after_loading_model):
        """Test that "--limit-mm-per-prompt" config can disable multi-modality for a multi-modal model.

        If an user *explicitly* sets the limit for all modalities to 0, then we can safely disable multi-modality even if the model itself claims to be multimodal.
        """
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.nnx.Rngs', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=self._model_get_model()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh), \
             patch('jax.device_put', side_effect=lambda x, *args, **kwargs: x):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            model_config.multimodal_config = MagicMock()
            model_config.multimodal_config.limit_per_prompt = limit_per_prompt

            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=None,
                observability_config={},
                additional_config={},
            )

            runner = TPUModelRunner(vllm_config, devices=self.mock_devices)
            # Precondition: make sure the model_config claims the model supports MM.
            assert runner.model_config.is_multimodal_model
            runner.load_model()

            assert runner.is_multimodal_model == still_mm_after_loading_model
