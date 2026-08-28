"""PyTorch equivalent of the JAX temporal force encoder.

The conversion functions consume the pure parameter dictionary restored from an
Orbax ForceVLA checkpoint.  They transpose Flax linear/Conv1D kernels into the
layouts expected by PyTorch without changing the trained function.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from openpi.models.force_encoder import ForceEncoderConfig


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: ForceEncoderConfig, dilation: int):
        super().__init__()
        self.left_padding = (config.kernel_size - 1) * dilation
        self.activation = config.activation
        self.conv1 = nn.Conv1d(in_dim, out_dim, config.kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(out_dim, eps=1e-6)
        self.conv2 = nn.Conv1d(out_dim, out_dim, config.kernel_size, dilation=dilation)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value) if self.activation == "silu" else F.gelu(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Public layout stays [B,N,C], matching the JAX encoder and dataset.
        residual = value if self.residual is None else self.residual(value)
        value = F.pad(value.transpose(1, 2), (self.left_padding, 0))
        value = self.conv1(value).transpose(1, 2)
        value = self._activate(self.norm1(value))
        value = F.pad(value.transpose(1, 2), (self.left_padding, 0))
        value = self.conv2(value).transpose(1, 2)
        value = self._activate(self.norm2(value))
        return value + residual


class TemporalTCNForceEncoder(nn.Module):
    def __init__(self, config: ForceEncoderConfig, output_dim: int):
        super().__init__()
        self.config = config
        self.stem = nn.Linear(config.input_dim, config.hidden_dims[0])
        input_dims = (config.hidden_dims[0], *config.hidden_dims[:-1])
        self.blocks = nn.ModuleList(
            CausalTemporalBlock(in_dim, out_dim, config, dilation)
            for in_dim, out_dim, dilation in zip(input_dims, config.hidden_dims, config.dilations, strict=True)
        )
        self.output_proj = nn.Linear(config.hidden_dims[-1], output_dim)

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = self.stem(history * mask.unsqueeze(-1).to(history.dtype))
        for block in self.blocks:
            value = block(value)
        if self.config.aggregation == "last":
            pooled = value[:, -1]
        else:
            weights = mask.unsqueeze(-1).to(value.dtype)
            pooled = (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output_proj(pooled)


def _copy_linear(module: nn.Linear, params: Mapping) -> None:
    module.weight.data.copy_(torch.from_numpy(np.asarray(params["kernel"])).T)
    module.bias.data.copy_(torch.from_numpy(np.asarray(params["bias"])))


def _copy_conv(module: nn.Conv1d, params: Mapping) -> None:
    # Flax Conv1D [kernel,in,out] -> torch [out,in,kernel].
    kernel = np.asarray(params["kernel"]).transpose(2, 1, 0).copy()
    module.weight.data.copy_(torch.from_numpy(kernel))
    module.bias.data.copy_(torch.from_numpy(np.asarray(params["bias"])))


def load_jax_params(module: TemporalTCNForceEncoder, params: Mapping) -> TemporalTCNForceEncoder:
    """Load one restored ``temporal_force_encoder`` subtree in-place."""
    _copy_linear(module.stem, params["stem"])
    _copy_linear(module.output_proj, params["output_proj"])
    blocks = params["blocks"]
    for index, block in enumerate(module.blocks):
        source = blocks[index] if index in blocks else blocks[str(index)]
        _copy_conv(block.conv1, source["conv1"])
        _copy_conv(block.conv2, source["conv2"])
        block.norm1.weight.data.copy_(torch.from_numpy(np.asarray(source["norm1"]["scale"])))
        block.norm1.bias.data.copy_(torch.from_numpy(np.asarray(source["norm1"]["bias"])))
        block.norm2.weight.data.copy_(torch.from_numpy(np.asarray(source["norm2"]["scale"])))
        block.norm2.bias.data.copy_(torch.from_numpy(np.asarray(source["norm2"]["bias"])))
        if block.residual is not None:
            _copy_linear(block.residual, source["residual"])
    return module
