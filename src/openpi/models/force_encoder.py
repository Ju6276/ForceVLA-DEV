"""Force front-ends for ForceVLA.

All encoders return one token in the original ForceVLA force-fusion width.
"""

import dataclasses
import math

import flax.nnx as nnx
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ForceEncoderConfig:
    type: str = "instantaneous"
    input_dim: int = 6
    hidden_dims: tuple[int, ...] = (1024, 1024, 1024, 1024)
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout_rate: float = 0.1
    activation: str = "silu"
    aggregation: str = "last"
    sampling_rate_hz: float = 200.0
    window_ms: float = 100.0
    history_source: str = "timestamp_stream"

    def __post_init__(self):
        if self.type not in {"instantaneous", "avg_pool", "max_pool", "tcn"}:
            raise ValueError(f"Unknown force encoder type: {self.type}")
        if self.aggregation not in {"last", "mean"}:
            raise ValueError(f"Unknown temporal aggregation: {self.aggregation}")
        if self.activation not in {"silu", "gelu"}:
            raise ValueError(f"Unknown TCN activation: {self.activation}")
        if self.history_source not in {"timestamp_stream", "aligned_state"}:
            raise ValueError(f"Unknown force history source: {self.history_source}")
        if self.input_dim != 6:
            raise ValueError("ForceVLA currently expects a 6D wrench")
        if self.type == "tcn" and len(self.hidden_dims) != len(self.dilations):
            raise ValueError("TCN hidden_dims must provide one hidden width per dilated block")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        if self.kernel_size < 1 or self.sampling_rate_hz <= 0 or self.window_ms <= 0:
            raise ValueError("kernel size, sampling rate, and window duration must be positive")

    @property
    def max_history_samples(self) -> int:
        return max(1, math.ceil(self.sampling_rate_hz * self.window_ms / 1000.0))


class CausalTemporalBlock(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        dilation: int,
        activation: str,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.conv1 = nnx.Conv(
            in_dim,
            out_dim,
            kernel_size=(kernel_size,),
            kernel_dilation=(dilation,),
            padding="VALID",
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(out_dim, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_dim,
            out_dim,
            kernel_size=(kernel_size,),
            kernel_dilation=(dilation,),
            padding="VALID",
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(out_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.residual = nnx.Linear(in_dim, out_dim, rngs=rngs) if in_dim != out_dim else None

    def _activate(self, x):
        return nnx.silu(x) if self.activation == "silu" else nnx.gelu(x)

    def __call__(self, x, *, train: bool = False):
        # NNX Conv uses [batch, spatial, channels]. Left-only padding makes the
        # convolution strictly causal even for dilation > 1.
        left_pad = (self.kernel_size - 1) * self.dilation
        y = self.conv1(jnp.pad(x, ((0, 0), (left_pad, 0), (0, 0))))
        y = self.dropout(self._activate(self.norm1(y)), deterministic=not train)
        y = self.conv2(jnp.pad(y, ((0, 0), (left_pad, 0), (0, 0))))
        y = self.dropout(self._activate(self.norm2(y)), deterministic=not train)
        skip = self.residual(x) if self.residual is not None else x
        return y + skip


class TemporalTCNForceEncoder(nnx.Module):
    def __init__(self, config: ForceEncoderConfig, output_dim: int, *, rngs: nnx.Rngs):
        self.config = config
        self.stem = nnx.Linear(config.input_dim, config.hidden_dims[0], rngs=rngs)
        block_input_dims = (config.hidden_dims[0], *config.hidden_dims[:-1])
        self.blocks = [
            CausalTemporalBlock(
                in_dim,
                out_dim,
                config.kernel_size,
                dilation,
                config.activation,
                config.dropout_rate,
                rngs=rngs,
            )
            for in_dim, out_dim, dilation in zip(
                block_input_dims, config.hidden_dims, config.dilations, strict=True
            )
        ]
        self.output_proj = nnx.Linear(config.hidden_dims[-1], output_dim, rngs=rngs)

    def __call__(self, history, mask, *, train: bool = False):
        # Invalid left padding is kept at zero even after dataset normalization.
        x = self.stem(history * mask[..., None].astype(history.dtype))
        for block in self.blocks:
            x = block(x, train=train)
        if self.config.aggregation == "last":
            # Histories are left padded, so the final element is the current sample.
            pooled = x[:, -1]
        else:
            weights = mask[..., None].astype(x.dtype)
            pooled = jnp.sum(x * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
        return self.output_proj(pooled)


def pool_force_history(history, mask, mode: str):
    """Mask-aware non-learned temporal baselines."""
    if mode == "avg_pool":
        weights = mask[..., None].astype(history.dtype)
        return jnp.sum(history * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
    if mode == "max_pool":
        masked = jnp.where(mask[..., None], history, -jnp.inf)
        pooled = jnp.max(masked, axis=1)
        return jnp.where(jnp.isfinite(pooled), pooled, 0)
    raise ValueError(f"Unsupported pooling mode: {mode}")
