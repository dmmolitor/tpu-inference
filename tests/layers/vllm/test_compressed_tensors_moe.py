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

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchax
from compressed_tensors.compressors.quantized_compressors.pack_quantized import \
    pack_to_int32
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import PartitionSpec
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import FusedMoE

# yapf: disable
from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe import (
    VllmCompressedTensorsW4A8Fp8MoEMethod,
    VllmCompressedTensorsW8A8Fp8MoEMethod)

# yapf: enable

P = PartitionSpec

MODEL = 'BCCard/Qwen3-30B-A3B-FP8-Dynamic'


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.distributed.jax_parallel_state.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # RowParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )

    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)


def initialize_layer_weights(layer: torch.nn.Module):
    torch.manual_seed(42)
    assert isinstance(layer, FusedMoE)

    e = layer.num_experts
    h = layer.hidden_size
    i = layer.intermediate_size_per_partition

    # 1. Initialize w13 (gate and up projections) -> Shape: (E, 2*I, H)
    w13_bf16 = torch.rand((e, 2 * i, h), dtype=torch.bfloat16) / 10
    w13_q, w13_s = test_utils.ref_quantize_fp8(w13_bf16,
                                               torch.float8_e4m3fn,
                                               axis=2)

    assert layer.w13_weight.data.shape == w13_q.shape
    assert layer.w13_weight_scale.data.shape == w13_s.shape

    layer.w13_weight.data = w13_q
    layer.w13_weight_scale.data = w13_s

    # 2. Initialize w2 (down_proj) -> Shape: (E, H, I)
    w2_bf16 = torch.rand((e, h, i), dtype=torch.bfloat16) / 10
    w2_q, w2_s = test_utils.ref_quantize_fp8(w2_bf16,
                                             torch.float8_e4m3fn,
                                             axis=2)

    assert layer.w2_weight.data.shape == w2_q.shape
    assert layer.w2_weight_scale.data.shape == w2_s.shape

    layer.w2_weight.data = w2_q
    layer.w2_weight_scale.data = w2_s

    # Handle optional MoE biases
    if hasattr(layer, 'w13_bias') and layer.w13_bias is not None:
        layer.w13_bias.data = torch.rand_like(layer.w13_bias.data)
    if hasattr(layer, 'w2_bias') and layer.w2_bias is not None:
        layer.w2_bias.data = torch.rand_like(layer.w2_bias.data)


@pytest.mark.parametrize(
    "mesh", [test_utils.get_spmd_mesh(1),
             test_utils.get_spmd_mesh(2)])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("use_ep", [True, False])
def test_fused_moe_method(mesh, num_tokens, intermediate_size, hidden_size,
                          num_experts, topk, use_ep):
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = False

    # Call tpu_inference code
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(num_experts=num_experts,
                         top_k=topk,
                         hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         quant_config=quant_config)
    weight_quant = quant_config.target_scheme_map['Linear']['weights']
    input_quant = quant_config.target_scheme_map['Linear']['input_activations']
    moe = quant_config.get_moe_config(layer)
    method = VllmCompressedTensorsW8A8Fp8MoEMethod(weight_quant, input_quant,
                                                   moe, mesh)
    method.create_weights(layer,
                          num_experts,
                          hidden_size,
                          intermediate_size,
                          params_dtype=torch.float8_e4m3fn)

    initialize_layer_weights(layer)
    method.process_weights_after_loading(layer)

    def unquantize_weight_for_ref(weight, scale):
        return (weight.to(torch.float32) * scale.squeeze(2)).transpose(
            1, 2).cpu()

    seqlen = num_tokens
    with torchax.default_env():
        x = torch.ones((seqlen, hidden_size), dtype=torch.bfloat16).to('jax')
        router_logits = torch.randn((seqlen, num_experts),
                                    dtype=torch.bfloat16).to('jax')
        result = method.apply_monolithic(layer, x, router_logits)
        expected = test_utils.ref_moe(
            x.to(torch.float32).cpu(),
            router_logits.to(torch.float32).cpu(),
            unquantize_weight_for_ref(layer.w13_weight,
                                      layer.w13_weight_scale),
            unquantize_weight_for_ref(layer.w2_weight, layer.w2_weight_scale),
            w1_bias=None,
            w2_bias=None,
            top_k=topk,
            renormalize=True,
            activation="silu")
        assert np.allclose(result, expected, atol=0.05, rtol=0.05)


def quantize_to_int4_grouped(
        weight: torch.Tensor,
        group_size: int,
        axis: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a weight tensor to int4 using group-wise quantization along a specific axis.
    """
    original_shape = weight.shape
    axis = axis % weight.ndim  # Support negative indexing (e.g., -1)

    target_dim_size = original_shape[axis]
    assert target_dim_size % group_size == 0, (
        f"Dimension {axis} size ({target_dim_size}) must be divisible by group_size ({group_size})."
    )

    num_groups = target_dim_size // group_size

    # 1. Split the target axis: (..., target_dim_size, ...) -> (..., num_groups, group_size, ...)
    new_shape = list(original_shape)
    new_shape[axis:axis + 1] = [num_groups, group_size]
    reshaped_w = weight.reshape(new_shape)

    # Because we inserted a dimension, the group_size elements are now at axis + 1
    reduce_dim = axis + 1

    # Uses symmetric quantization with a fixed quantization range of [-8, 7] for int4
    q_min, q_max = -8, 7

    abs_max = reshaped_w.abs().amax(dim=reduce_dim, keepdim=True)
    scale = abs_max / q_max
    scale = scale.clamp(min=1e-7)

    w_q = torch.round(reshaped_w / scale).clamp(q_min, q_max)
    w_q = w_q.reshape(original_shape).to(torch.int8)

    scale = scale.squeeze(reduce_dim)

    return w_q, scale


def dequantize_int4_grouped(w_q: torch.Tensor,
                            scale: torch.Tensor,
                            group_size: int,
                            axis: int = -1) -> torch.Tensor:
    """
    Dequantizes an int4 tensor (stored in int8/uint8) back to floating point.

    Args:
        w_q: The quantized weights.
        scale: The scaling factors for each group.
        group_size: The number of elements grouped together during quantization.
        axis: The dimension where grouping was applied.

    Returns:
        The dequantized floating-point tensor.
    """
    original_shape = w_q.shape
    axis = axis % w_q.ndim  # Support negative indexing
    num_groups = original_shape[axis] // group_size

    # 1. Reshape w_q to isolate the group dimension: (..., num_groups, group_size, ...)
    new_shape = list(original_shape)
    new_shape[axis:axis + 1] = [num_groups, group_size]
    reshaped_w_q = w_q.reshape(new_shape)

    # 2. Unsqueeze scale and zero_point so they broadcast across the group_size
    reduce_dim = axis + 1
    broadcast_scale = scale.unsqueeze(reduce_dim)

    # 3. Symmetric dequantize: w_q * scale
    dequantized = reshaped_w_q.to(broadcast_scale.dtype) * broadcast_scale

    # 4. Collapse the group dimension back into the original shape
    return dequantized.reshape(original_shape)


def initialize_int4_layer_weights(layer: torch.nn.Module,
                                  weight_quant: QuantizationArgs,
                                  hidden_size: int):
    torch.manual_seed(42)
    assert isinstance(layer, FusedMoE)

    group_size = weight_quant.group_size
    experts = layer.global_num_experts
    intermediate_size = layer.intermediate_size_per_partition

    def generate_moe_expert_weights(
        expert_shape: tuple[int, int], ) -> tuple[torch.Tensor, torch.Tensor]:
        """
      Generates, quantizes, and packs a weight.
      """
        q_per_expert = []
        s_per_expert = []

        for _ in range(experts):
            w_block = (torch.rand(expert_shape, dtype=torch.bfloat16) -
                       0.5) / 10
            w_q, w_s = quantize_to_int4_grouped(w_block, group_size, axis=-1)
            q_per_expert.append(w_q)
            s_per_expert.append(w_s)

        # Pack the quantized bits into int32 containers
        q_packed = torch.stack([
            pack_to_int32(q, num_bits=weight_quant.num_bits, packed_dim=1)
            for q in q_per_expert
        ])

        return q_packed, torch.stack(s_per_expert)

    # 1. Initialize w13 (gate and up projections) -> Shape: (E, 2*I, H)
    w13_q_packed, w13_s = generate_moe_expert_weights(
        expert_shape=(2 * intermediate_size, hidden_size))
    assert layer.w13_weight_packed.data.shape == w13_q_packed.shape
    assert layer.w13_weight_scale.data.shape == w13_s.shape

    layer.w13_weight_packed.data = w13_q_packed
    layer.w13_weight_scale.data = w13_s

    # 2. Initialize w2 (down_proj) -> Shape: (E, H, I)
    w2_q_packed, w2_s = generate_moe_expert_weights(
        expert_shape=(hidden_size, intermediate_size))
    assert layer.w2_weight_packed.data.shape == w2_q_packed.shape
    assert layer.w2_weight_scale.data.shape == w2_s.shape

    layer.w2_weight_packed.data = w2_q_packed
    layer.w2_weight_scale.data = w2_s

    # Handle optional MoE biases
    if hasattr(layer, 'w13_bias') and layer.w13_bias is not None:
        layer.w13_bias.data = torch.rand_like(layer.w13_bias.data)
    if hasattr(layer, 'w2_bias') and layer.w2_bias is not None:
        layer.w2_bias.data = torch.rand_like(layer.w2_bias.data)


@pytest.mark.parametrize(
    "mesh", [test_utils.get_spmd_mesh(1),
             test_utils.get_spmd_mesh(2)])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("use_ep", [True, False])
def test_fused_moe_method_w4a8fp8(mesh, num_tokens, intermediate_size,
                                  hidden_size, num_experts, topk, use_ep):
    engine_args = EngineArgs(
        model='nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16',
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = False

    # Call tpu_inference code
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    weight_quant = quant_config.target_scheme_map['Linear']['weights']
    input_quant = quant_config.target_scheme_map['Linear']['input_activations']

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(num_experts=num_experts,
                         top_k=topk,
                         hidden_size=hidden_size,
                         intermediate_size=intermediate_size)
        moe = quant_config.get_moe_config(layer)
        method = VllmCompressedTensorsW4A8Fp8MoEMethod(weight_quant,
                                                       input_quant, moe, mesh)
    method.create_weights(layer,
                          num_experts,
                          hidden_size,
                          intermediate_size,
                          params_dtype=torch.bfloat16)

    initialize_int4_layer_weights(layer, weight_quant, hidden_size=128)
    method.process_weights_after_loading(layer)

    seqlen = num_tokens
    with torchax.default_env():
        x = torch.ones((seqlen, hidden_size), dtype=torch.bfloat16).to('jax')
        router_logits = torch.randn((seqlen, num_experts),
                                    dtype=torch.bfloat16).to('jax')
        result = method.apply_monolithic(layer, x, router_logits)
        w13_weight_dequant = dequantize_int4_grouped(
            layer.w13_weight,
            layer.w13_weight_scale.squeeze(2),
            group_size=weight_quant.group_size,
            axis=1)
        w2_weight_dequant = dequantize_int4_grouped(
            layer.w2_weight,
            layer.w2_weight_scale.squeeze(2),
            group_size=weight_quant.group_size,
            axis=1)
        expected = test_utils.ref_moe(x.to(torch.float32).cpu(),
                                      router_logits.to(torch.float32).cpu(),
                                      w13_weight_dequant.transpose(1, 2).cpu(),
                                      w2_weight_dequant.transpose(1, 2).cpu(),
                                      w1_bias=None,
                                      w2_bias=None,
                                      top_k=topk,
                                      renormalize=True,
                                      activation="silu")
        assert np.allclose(result, expected, atol=0.2, rtol=0.05)
