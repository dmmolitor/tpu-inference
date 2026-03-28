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

import jax
import jax.numpy as jnp
import torch
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import Mesh
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod, CompressedTensorsW4A8Fp8MoEMethod,
    CompressedTensorsW8A8Fp8MoEMethod)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.scalar_type import ScalarType, scalar_types

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j

logger = init_logger(__name__)


# TODO(dmmolitor): Consider using compressed-tensors' unpack_from_int32 once PR
# #609: [Bugfix] Support N-dimensional tensors in pack_to_int32 and
# unpack_from_int32 is available.
def jax_unpack_quantized_values_into_int32(
    w_q: jax.Array,
    wtype: ScalarType,
    packed_dim: int = 0,
) -> jax.Array:
    """JAX implementation of vLLM's unpack_quantized_values_into_int32."""
    # move dim to pack to the end
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    w_q_perm = jnp.transpose(w_q, perm)

    pack_factor = 32 // wtype.size_bits
    mask = (1 << wtype.size_bits) - 1

    new_shape_perm = list(w_q_perm.shape)
    new_shape_perm[-1] *= pack_factor

    unpacked_pieces = []
    for i in range(pack_factor):
        piece = (w_q_perm >> (wtype.size_bits * i)) & mask
        unpacked_pieces.append(piece)

    # Interleave: res[..., i::pack_factor]
    # In JAX/NumPy, stacking and reshaping handles this correctly for the last dim
    res = jnp.stack(unpacked_pieces,
                    axis=-1)  # [..., old_last_dim, pack_factor]
    res = res.reshape(new_shape_perm)
    return jnp.transpose(res, inv_perm)


class VllmCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):

    @staticmethod
    def get_moe_method(
        quant_config: "VllmCompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> CompressedTensorsMoEMethod:
        assert isinstance(layer, FusedMoE)

        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError("All MoE projections need to have same "
                             "quantization scheme but found multiple")

        if scheme_dict is None:
            return VllmUnquantizedFusedMoEMethod(layer.moe_config,
                                                 quant_config.mesh)

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        if quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return VllmCompressedTensorsW8A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config, quant_config.mesh)
        elif quant_config._is_fp8_w4a8(weight_quant, input_quant):
            return VllmCompressedTensorsW4A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config, quant_config.mesh)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class VllmCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod,
                                            VllmQuantConfig):

    def __init__(self,
                 weight_quant: QuantizationArgs,
                 input_quant: QuantizationArgs,
                 moe: FusedMoEConfig,
                 mesh: Mesh,
                 ep_axis_name: str = "model"):
        super().__init__(weight_quant, input_quant, moe)

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Processes and formats the raw model weights after they are loaded from the checkpoint.

        :param self: The method for the layer responsible for processing the weights.
        :param layer: The source PyTorch layer containing the raw, un-sharded weights from the loaded checkpoint.
        :type layer: torch.nn.Module

        Steps:
        1. Read weights from layer object and convert to jax arrays
        2. Interleave concat w13 weights
        3. Shard weights for tp (rowwise w13, colwise w2)
        4. Initialize Params as torch.nn.Parameter
            a. w13_weight - float8_e4m3fn shape: (num_experts, 2 x intermediate_size, input_size)
            b. w2_weight - float8_e4m3fn shape: (num_experts, output_size, intermediate_size)
            c. w13_weight_scale - FP32 shape: (num_experts, 2 x intermediate_size, 1)
            d. w2_weight_scale - FP32shape: (num_experts, output_size, 1)
        """
        assert isinstance(layer, FusedMoE)

        # N.B
        # layer.w13_weight: [num_experts, 2*moe_intermediate_size, hidden_size]
        # layer.w13_weight_scale: [num_experts, 2*moe_intermediate_size, 1]
        # layer.w2_weight: [num_experts, hidden_size, moe_intermediate_size]
        # layer.w2_weight_scale: [num_experts, hidden_size, 1]
        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        @jax.jit
        def process_fp8_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                weights=FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_weight_scale,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_weight_scale,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_fp8_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_bias,
            w2_weight,
            w2_weight_scale,
            w2_bias,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)


class VllmCompressedTensorsW4A8Fp8MoEMethod(CompressedTensorsW4A8Fp8MoEMethod,
                                            VllmQuantConfig):
    """
    MoE method for int4xfp8 (INT4 weights, FP8/16 activations) compressed-tensors.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        mesh: jax.sharding.Mesh,
        ep_axis_name: str = "model",
    ):
        super().__init__(weight_quant, input_quant, moe)

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )
        self.wtype = scalar_types.uint4

        self.weight_quant = weight_quant
        self.input_quant = input_quant

        self.group_size = self.weight_quant.group_size
        self.num_bits = self.weight_quant.num_bits
        self.packed_factor = 32 // self.num_bits

        assert self.weight_quant.symmetric, (
            "Only symmetric quantization is supported for W4A8 MoE")
        assert self.weight_quant.actorder != "group"

        self.disable_expert_map = False

        # from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        # from vllm.model_executor.layers.quantization.utils.quant_utils import (
        #     GroupShape,
        # )

        # self.quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

        # self.ep_axis_name = ep_axis_name

    @property
    def is_monolithic(self) -> bool:
        """Indicates if the MoE operation is monolithic."""
        return True

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **kwargs):
        """
        Initializes the weights and scales for the FusedMoE layer.
        This is called during model initialization to allocate tensors.
        Handles packed int4 weights and grouped/channelwise scales.

        This method differs from the parent class's create_weights in that it
        does not require that the hidden_size and intermediate_size be divisible
        by 256 and instead only requires them to be divisible by the packed
        factor.

        :param layer: The FusedMoE layer to initialize.
        :param num_experts: Total number of experts.
        :param hidden_size: Hidden dimension size.
        :param intermediate_size: Intermediate dimension size.
        :param params_dtype: Data type for parameters like scale and bias.
        :param kwargs: Additional arguments like weight_loader.
        """

        weight_loader = kwargs.get("weight_loader")

        assert hidden_size % self.packed_factor == 0, (
            f"Hidden size ({hidden_size}) must be divisible by packed factor "
            f"({self.packed_factor}).")
        assert intermediate_size % self.packed_factor == 0, (
            f"Intermediate size ({intermediate_size}) must be divisible by "
            f"packed factor ({self.packed_factor}).")

        # W13 weight: [num_experts, 2 * intermediate_size, hidden_size]
        # Contracting dim is hidden_size (dim 2)
        layer.register_parameter(
            "w13_weight_packed",
            PackedvLLMParameter(
                data=torch.empty(num_experts,
                                 2 * intermediate_size,
                                 hidden_size // self.packed_factor,
                                 dtype=torch.int32),
                input_dim=2,
                output_dim=1,
                packed_dim=2,
                packed_factor=self.packed_factor,
                weight_loader=weight_loader,
            ))

        # W2 weight: [num_experts, hidden_size, intermediate_size]
        # Contracting dim is intermediate_size (dim 2)
        layer.register_parameter(
            "w2_weight_packed",
            PackedvLLMParameter(
                data=torch.empty(num_experts,
                                 hidden_size,
                                 intermediate_size // self.packed_factor,
                                 dtype=torch.int32),
                input_dim=1,
                output_dim=2,
                packed_dim=2,
                packed_factor=self.packed_factor,
                weight_loader=weight_loader,
            ))

        # SHAPE PARAMETERS (logical shapes)
        layer.register_parameter(
            "w13_weight_shape",
            BasevLLMParameter(data=torch.tensor(
                [num_experts, 2 * intermediate_size, hidden_size],
                dtype=torch.int64),
                              weight_loader=weight_loader))
        layer.register_parameter(
            "w2_weight_shape",
            BasevLLMParameter(data=torch.tensor(
                [num_experts, hidden_size, intermediate_size],
                dtype=torch.int64),
                              weight_loader=weight_loader))

        # SCALES
        # Determine if we use grouped or channelwise scales
        use_group_scales = (self.weight_quant.strategy == "group"
                            and self.group_size != -1)

        w13_scale_args = {
            "data":
            torch.empty(
                num_experts,
                2 * intermediate_size,
                (hidden_size // self.group_size if use_group_scales else 1),
                dtype=params_dtype),
            "weight_loader":
            weight_loader,
        }
        if use_group_scales:
            layer.register_parameter(
                "w13_weight_scale",
                GroupQuantScaleParameter(output_dim=1,
                                         input_dim=2,
                                         **w13_scale_args))
        else:
            layer.register_parameter(
                "w13_weight_scale",
                ChannelQuantScaleParameter(output_dim=1, **w13_scale_args))

        # w2_weight_scale
        w2_scale_args = {
            "data":
            torch.empty(num_experts,
                        hidden_size,
                        (intermediate_size //
                         self.group_size if use_group_scales else 1),
                        dtype=params_dtype),
            "weight_loader":
            weight_loader,
        }
        if use_group_scales:
            layer.register_parameter(
                "w2_weight_scale",
                GroupQuantScaleParameter(output_dim=1,
                                         input_dim=2,
                                         **w2_scale_args))
        else:
            layer.register_parameter(
                "w2_weight_scale",
                ChannelQuantScaleParameter(output_dim=1, **w2_scale_args))

        # Optional BIAS
        if self.moe.has_bias:
            layer.register_parameter(
                "w13_bias",
                BasevLLMParameter(data=torch.empty(num_experts,
                                                   2 * intermediate_size,
                                                   dtype=params_dtype),
                                  weight_loader=weight_loader))
            layer.register_parameter(
                "w2_bias",
                BasevLLMParameter(data=torch.empty(num_experts,
                                                   hidden_size,
                                                   dtype=params_dtype),
                                  weight_loader=weight_loader))

        # We don't use input scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Processes and shards MoE weights after loading.

        :param self: The method for the layer responsible for processing the weights.
        :param layer: The source PyTorch layer containing the raw, un-sharded weights from the loaded checkpoint.
        :type layer: torch.nn.Module

        Steps:
        For W4A8, we unpack INT4 weights and upcast them to FP8 to reuse
        the standard W8A8 logic and kernels.
        """
        assert isinstance(layer, FusedMoE)

        @jax.jit
        def unpack_and_upcast_single(packed_weight: jax.Array) -> jax.Array:
            unpacked = jax_unpack_quantized_values_into_int32(packed_weight,
                                                              self.wtype,
                                                              packed_dim=2)

            # compressed-tensors uint4 is offset by 8 (0-15 -> -8 to 7)
            return (unpacked - 8).astype(jnp.float8_e4m3fn)

        # N.B
        # layer.w13_weight: [num_experts, 2*moe_intermediate_size, hidden_size]
        # layer.w13_weight_scale: [num_experts, 2*moe_intermediate_size, 1]
        # layer.w2_weight: [num_experts, hidden_size, moe_intermediate_size]
        # layer.w2_weight_scale: [num_experts, hidden_size, 1]
        w13_weight = unpack_and_upcast_single(t2j(
            layer.w13_weight_packed.data))
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w2_weight = unpack_and_upcast_single(t2j(layer.w2_weight_packed.data))
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

        if self.moe.has_bias:
            w13_bias = t2j(layer.w13_bias, use_dlpack=False)
            w2_bias = t2j(layer.w2_bias, use_dlpack=False)
        else:
            w13_bias = w2_bias = None

        @jax.jit
        def process_fp8_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                weights=FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_weight_scale,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_weight_scale,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_fp8_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_bias,
            w2_weight,
            w2_weight_scale,
            w2_bias,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

        # Clean up packed parameters and shape metadata
        if hasattr(layer, "w13_weight_packed"):
            delattr(layer, "w13_weight_packed")
        if hasattr(layer, "w2_weight_packed"):
            delattr(layer, "w2_weight_packed")
        if hasattr(layer, "w13_weight_shape"):
            delattr(layer, "w13_weight_shape")
        if hasattr(layer, "w2_weight_shape"):
            delattr(layer, "w2_weight_shape")

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight).astype(jnp.int4),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight).astype(jnp.int4),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )
        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
