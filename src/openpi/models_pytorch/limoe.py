"""Inference-only PyTorch port of ForceVLA's LIMoE fusion block."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BatchOneLIMoE(nn.Module):
    """Exact deployment path for the fixed BS=1 ForceVLA token layout.

    With 817 input tokens and max_group_size=16, the JAX implementation forms
    817 one-token routing groups.  Capacity can therefore never overflow and
    top-1 routing reduces to a weighted per-token expert evaluation.
    """

    def __init__(self, input_dim: int = 2048, output_dim: int = 1024, num_heads: int = 8, num_experts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.num_experts = num_experts

        self.norm1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.attn_out = nn.Linear(input_dim, input_dim)
        self.norm2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.ffn_in = nn.Linear(input_dim, input_dim)
        self.ffn_out = nn.Linear(input_dim, input_dim)

        self.router = nn.Linear(input_dim, num_experts)
        self.expert_in = nn.Parameter(torch.empty(num_experts, input_dim, input_dim))
        self.expert_in_bias = nn.Parameter(torch.empty(num_experts, input_dim))
        self.expert_out = nn.Parameter(torch.empty(num_experts, input_dim, input_dim))
        self.expert_out_bias = nn.Parameter(torch.empty(num_experts, input_dim))
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != 1:
            raise ValueError("BatchOneLIMoE currently reproduces only the deployment BS=1 routing path")
        # The current ForceVLA prefix plus force token is 817. This is prime and
        # therefore yields one-token groups in the original `_num_groups` logic.
        if value.shape[1] != 817:
            raise ValueError(f"Expected the trained 817-token fusion layout, got {value.shape[1]}")

        residual = value
        normed = self.norm1(value)
        batch, length, _ = normed.shape
        q = self.q_proj(normed).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attention = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch, length, self.input_dim)
        value = residual + self.attn_out(attended)
        value = value + self.ffn_out(F.gelu(self.ffn_in(self.norm2(value)), approximate="tanh"))

        # Flaxformer's RouterWeights is explicitly bfloat16 even though the
        # enclosing LIMoE block is float32. Preserve that quantization boundary;
        # it can affect top-1 choices when two expert logits are nearly tied.
        router_value = value.to(torch.bfloat16)
        router_logits = F.linear(
            router_value,
            self.router.weight.to(torch.bfloat16),
            self.router.bias.to(torch.bfloat16),
        )
        probabilities = torch.softmax(router_logits, dim=-1).to(value.dtype)
        weights, indices = probabilities.max(dim=-1)
        expert_values = torch.empty_like(value)
        for expert in range(self.num_experts):
            selected = indices == expert
            if selected.any():
                x = value[selected]
                x = F.relu(x @ self.expert_in[expert] + self.expert_in_bias[expert])
                expert_values[selected] = x @ self.expert_out[expert] + self.expert_out_bias[expert]
        value = value + expert_values * weights.unsqueeze(-1)
        return self.output(value)


def _copy_linear(module: nn.Linear, params: Mapping) -> None:
    module.weight.data.copy_(torch.from_numpy(np.asarray(params["kernel"])).T)
    module.bias.data.copy_(torch.from_numpy(np.asarray(params["bias"])))


def load_jax_params(module: BatchOneLIMoE, params: Mapping) -> BatchOneLIMoE:
    encoder = params["encoderblock"]
    module.norm1.weight.data.copy_(torch.from_numpy(np.asarray(encoder["encoder_in_norm"]["scale"])))
    module.norm1.bias.data.copy_(torch.from_numpy(np.asarray(encoder["encoder_in_norm"]["bias"])))
    module.norm2.weight.data.copy_(torch.from_numpy(np.asarray(encoder["encoder_out_norm"]["scale"])))
    module.norm2.bias.data.copy_(torch.from_numpy(np.asarray(encoder["encoder_out_norm"]["bias"])))

    attention = encoder["MultiHeadDotProductAttention_0"]
    for target, source_name in ((module.q_proj, "query"), (module.k_proj, "key"), (module.v_proj, "value")):
        source = attention[source_name]
        target.weight.data.copy_(torch.from_numpy(np.asarray(source["kernel"]).reshape(module.input_dim, -1)).T)
        target.bias.data.copy_(torch.from_numpy(np.asarray(source["bias"]).reshape(-1)))
    out_kernel = np.asarray(attention["out"]["kernel"]).reshape(module.input_dim, module.input_dim)
    module.attn_out.weight.data.copy_(torch.from_numpy(out_kernel).T)
    module.attn_out.bias.data.copy_(torch.from_numpy(np.asarray(attention["out"]["bias"])))
    _copy_linear(module.ffn_in, encoder["MlpBlock_0"]["Dense_0"])
    _copy_linear(module.ffn_out, encoder["MlpBlock_0"]["Dense_1"])
    _copy_linear(module.router, params["RouterWeights_0"]["w"])
    module.expert_in.data.copy_(torch.from_numpy(np.asarray(params["MlpBlock_0"]["wi"]["kernel"])))
    module.expert_in_bias.data.copy_(torch.from_numpy(np.asarray(params["MlpBlock_0"]["wi"]["bias"])))
    module.expert_out.data.copy_(torch.from_numpy(np.asarray(params["MlpBlock_0"]["wo"]["kernel"])))
    module.expert_out_bias.data.copy_(torch.from_numpy(np.asarray(params["MlpBlock_0"]["wo"]["bias"])))
    _copy_linear(module.output, params["Dense_0"])
    return module
