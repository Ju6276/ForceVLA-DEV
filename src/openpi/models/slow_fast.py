"""Teacher-guided slow/fast student components.

The Slow student owns a nominal action chunk.  At each fast-loop tick, the
force-conditioned expert consumes the *current* interpolated reference and
predicts one pose residual.  It intentionally does not predict another action
chunk: future residuals would otherwise be conditioned on force/state that have
not happened yet.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.force_encoder as _force_encoder


@dataclasses.dataclass(frozen=True)
class FastResidualConfig:
    """Configuration for the lightweight force residual student."""

    reference_dim: int = 7
    state_dim: int = 7
    pose_dims: int = 6
    time_feature_dim: int = 2
    intent_dim: int = 1024
    width: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    dropout_rate: float = 0.0
    predict_gate: bool = False
    force_encoder: _force_encoder.ForceEncoderConfig = dataclasses.field(
        default_factory=lambda: _force_encoder.ForceEncoderConfig(
            type="tcn",
            sampling_rate_hz=100,
            window_ms=100,
            max_sample_age_ms=12,
            history_source="timestamp_stream",
        )
    )

    def __post_init__(self) -> None:
        if (
            min(
                self.reference_dim,
                self.state_dim,
                self.pose_dims,
                self.time_feature_dim,
                self.intent_dim,
                self.width,
                self.mlp_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
            <= 0
        ):
            raise ValueError("Fast residual dimensions must be positive")
        if self.pose_dims > self.reference_dim:
            raise ValueError("pose_dims cannot exceed reference_dim")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")
        if self.force_encoder.type != "tcn":
            raise ValueError("Fast residual student requires a causal TCN force encoder")
        if self.force_encoder.hidden_dims[-1] != self.width:
            raise ValueError("The final TCN hidden width must match the fast expert width")


class TemporalTCNBackbone(nnx.Module):
    """TCN stem/blocks matching the Teacher, without its 1024->2048 projection."""

    def __init__(self, config: _force_encoder.ForceEncoderConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.stem = nnx.Linear(config.input_dim, config.hidden_dims[0], rngs=rngs)
        block_input_dims = (config.hidden_dims[0], *config.hidden_dims[:-1])
        self.blocks = [
            _force_encoder.CausalTemporalBlock(
                in_dim,
                out_dim,
                config.kernel_size,
                dilation,
                config.activation,
                config.dropout_rate,
                rngs=rngs,
            )
            for in_dim, out_dim, dilation in zip(block_input_dims, config.hidden_dims, config.dilations, strict=True)
        ]

    def __call__(self, history, mask, *, train: bool = False):
        if history.ndim != 3 or history.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected force history [B,N,{self.config.input_dim}], got {history.shape}")
        if mask.shape != history.shape[:-1]:
            raise ValueError(f"Expected force mask {history.shape[:-1]}, got {mask.shape}")
        x = self.stem(history * mask[..., None].astype(history.dtype))
        for block in self.blocks:
            x = block(x, train=train)
        if self.config.aggregation == "last":
            return x[:, -1]
        weights = mask[..., None].astype(x.dtype)
        return jnp.sum(x * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)


class IntentProjector(nnx.Module):
    """Compress slow-VLA hidden states into a small cache of intent tokens."""

    def __init__(self, input_dim: int, output_dim: int, num_tokens: int = 2, *, rngs: nnx.Rngs):
        if min(input_dim, output_dim, num_tokens) <= 0:
            raise ValueError("Intent projector dimensions must be positive")
        self.input_proj = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.queries = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (num_tokens, output_dim), jnp.float32)
        )
        self.output_norm = nnx.RMSNorm(output_dim, rngs=rngs)

    def __call__(self, hidden, mask):
        if hidden.ndim != 3 or mask.shape != hidden.shape[:-1]:
            raise ValueError(f"Expected hidden [B,S,D] and mask [B,S], got {hidden.shape} and {mask.shape}")
        projected = self.input_proj(hidden)
        scores = jnp.einsum("bsh,kh->bks", projected, self.queries.value) * projected.shape[-1] ** -0.5
        scores = jnp.where(mask[:, None, :], scores, -1e30)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(projected.dtype)
        # An all-padding context produces a deterministic zero cache rather than NaNs.
        any_valid = jnp.any(mask, axis=-1, keepdims=True)
        intent = jnp.einsum("bks,bsh->bkh", weights, projected)
        intent = jnp.where(any_valid[:, :, None], intent, 0)
        return self.output_norm(intent)


class SlowNominalStudentAdapter(nnx.Module):
    """Package an independently trained force-free Slow student's outputs.

    Stage-2 paired full/null inference supplies offline targets; it is not the
    deployed Slow network.  This adapter receives the standalone Slow action
    chunk and its vision-language context, then creates the cached intent sent
    to the fast loop.  It never receives force or a learned null-force token.
    """

    def __init__(
        self,
        context_dim: int = 2048,
        intent_dim: int = 1024,
        num_intent_tokens: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.intent_projector = IntentProjector(context_dim, intent_dim, num_tokens=num_intent_tokens, rngs=rngs)

    def __call__(self, nominal_actions, prefix_context, prefix_mask):
        if nominal_actions.ndim != 3:
            raise ValueError(f"Expected nominal action chunk [B,H,D], got {nominal_actions.shape}")
        if prefix_context.shape[0] != nominal_actions.shape[0]:
            raise ValueError("Nominal actions and prefix context must have matching batches")
        return nominal_actions, self.intent_projector(prefix_context, prefix_mask)


def _apply_rope(x, positions):
    if x.shape[-1] % 2:
        raise ValueError("RoPE head_dim must be even")
    exponent = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = 10_000**exponent
    radians = positions[..., None, None] / timescale[None, None, None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class GemmaStyleDecoderLayer(nnx.Module):
    """One pre-norm Gemma-style GQA block used by the residual draft head."""

    def __init__(self, config: FastResidualConfig, *, rngs: nnx.Rngs):
        self.config = config
        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        self.input_norm = nnx.RMSNorm(config.width, rngs=rngs)
        self.q_proj = nnx.Linear(config.width, q_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.width, kv_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.width, kv_dim, use_bias=False, rngs=rngs)
        self.attn_out = nnx.Linear(q_dim, config.width, use_bias=False, rngs=rngs)
        self.post_attention_norm = nnx.RMSNorm(config.width, rngs=rngs)
        self.gate_proj = nnx.Linear(config.width, config.mlp_dim, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.width, config.mlp_dim, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.mlp_dim, config.width, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, tokens, valid_mask, *, context_length: int, train: bool = False):
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:-1]:
            raise ValueError(f"Expected tokens [B,S,D] and mask [B,S], got {tokens.shape} and {valid_mask.shape}")
        if not 0 < context_length <= tokens.shape[1]:
            raise ValueError("context_length must identify a non-empty prefix")
        b, sequence_length, _ = tokens.shape
        x = self.input_norm(tokens)
        q = self.q_proj(x).reshape(b, sequence_length, self.config.num_heads, self.config.head_dim)
        k = self.k_proj(x).reshape(b, sequence_length, self.config.num_kv_heads, self.config.head_dim)
        v = self.v_proj(x).reshape(b, sequence_length, self.config.num_kv_heads, self.config.head_dim)
        positions = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (b, sequence_length))
        q = _apply_rope(q, positions)
        k = _apply_rope(k, positions)
        repeats = self.config.num_heads // self.config.num_kv_heads
        k = jnp.repeat(k, repeats, axis=2)
        v = jnp.repeat(v, repeats, axis=2)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
        logits *= self.config.head_dim**-0.5

        # A standard causal decoder mask is enough now that there is exactly
        # one residual query and it is the final token.  The query can inspect
        # every condition, while no condition can inspect the output query.
        structural = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=jnp.bool_))
        attention_mask = structural[None, None, :, :] & valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        logits = jnp.where(attention_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(logits, axis=-1).astype(tokens.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probs, v)
        attended = attended.reshape(b, sequence_length, self.config.num_heads * self.config.head_dim)
        attended = self.attn_out(attended)
        tokens = tokens + self.dropout(attended, deterministic=not train)

        x = self.post_attention_norm(tokens)
        mlp = jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x)
        mlp = self.down_proj(mlp)
        return tokens + self.dropout(mlp, deterministic=not train)


class FastForceResidualStudent(nnx.Module):
    """One-block expert that predicts one force-conditioned pose correction."""

    def __init__(self, config: FastResidualConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.force_encoder = TemporalTCNBackbone(config.force_encoder, rngs=rngs)
        self.intent_proj = (
            nnx.Linear(config.intent_dim, config.width, rngs=rngs) if config.intent_dim != config.width else None
        )
        self.state_proj = nnx.Linear(config.state_dim, config.width, rngs=rngs)
        self.reference_proj = nnx.Linear(config.reference_dim, config.width, rngs=rngs)
        self.time_proj = nnx.Linear(config.time_feature_dim, config.width, rngs=rngs)
        self.token_types = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (5, config.width), jnp.float32)
        )
        self.residual_query = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (config.width,), jnp.float32)
        )
        self.decoder = GemmaStyleDecoderLayer(config, rngs=rngs)
        output_dim = config.pose_dims + int(config.predict_gate)
        self.residual_head = nnx.Linear(
            config.width,
            output_dim,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        force_history,
        force_history_mask,
        intent_tokens,
        state,
        reference_action,
        time_features,
        *,
        intent_mask=None,
        train: bool = False,
    ):
        config = self.config
        if intent_tokens.ndim != 3 or intent_tokens.shape[-1] != config.intent_dim:
            raise ValueError(f"Expected intent [B,K,{config.intent_dim}], got {intent_tokens.shape}")
        if state.shape != (intent_tokens.shape[0], config.state_dim):
            raise ValueError(f"Expected state [B,{config.state_dim}], got {state.shape}")
        expected_reference = (intent_tokens.shape[0], config.reference_dim)
        if reference_action.shape != expected_reference:
            raise ValueError(f"Expected current reference action {expected_reference}, got {reference_action.shape}")
        expected_time = (intent_tokens.shape[0], config.time_feature_dim)
        if time_features.shape != expected_time:
            raise ValueError(f"Expected normalized time features {expected_time}, got {time_features.shape}")

        b, intent_length, _ = intent_tokens.shape
        if intent_mask is None:
            intent_mask = jnp.ones((b, intent_length), dtype=jnp.bool_)
        if intent_mask.shape != (b, intent_length):
            raise ValueError(f"Expected intent mask {(b, intent_length)}, got {intent_mask.shape}")

        intent = self.intent_proj(intent_tokens) if self.intent_proj is not None else intent_tokens
        intent = intent + self.token_types.value[0]
        force = self.force_encoder(force_history, force_history_mask, train=train)[:, None, :]
        force = force + self.token_types.value[1]
        state_token = self.state_proj(state)[:, None, :] + self.token_types.value[2]
        reference = self.reference_proj(reference_action)[:, None, :] + self.token_types.value[3]
        time_token = self.time_proj(time_features)[:, None, :] + self.token_types.value[4]
        context = jnp.concatenate([intent, force, state_token, reference, time_token], axis=1)
        context_mask = jnp.concatenate(
            [
                intent_mask,
                jnp.any(force_history_mask, axis=1, keepdims=True),
                jnp.ones((b, 3), dtype=jnp.bool_),
            ],
            axis=1,
        )
        query = jnp.broadcast_to(self.residual_query.value[None, None, :], (b, 1, config.width))
        tokens = jnp.concatenate([context, query], axis=1)
        valid_mask = jnp.concatenate([context_mask, jnp.ones((b, 1), dtype=jnp.bool_)], axis=1)
        hidden = self.decoder(tokens, valid_mask, context_length=context.shape[1], train=train)
        output = self.residual_head(hidden[:, -1]).astype(jnp.float32)
        residual = output[..., : config.pose_dims]
        gate = jax.nn.sigmoid(output[..., config.pose_dims]) if config.predict_gate else jnp.ones((b,))
        return residual, gate


class FastStudentWithIntentProjector(nnx.Module):
    """Trainable Stage-5 head on top of a frozen Slow vision-language context."""

    def __init__(
        self,
        config: FastResidualConfig,
        *,
        slow_context_dim: int = 2048,
        num_intent_tokens: int = 2,
        rngs: nnx.Rngs,
    ):
        self.intent_projector = IntentProjector(
            slow_context_dim,
            config.intent_dim,
            num_tokens=num_intent_tokens,
            rngs=rngs,
        )
        self.fast_student = FastForceResidualStudent(config, rngs=rngs)

    def __call__(
        self,
        slow_context,
        slow_context_mask,
        force_history,
        force_history_mask,
        state,
        reference_action,
        time_features,
        *,
        train: bool = False,
    ):
        intent_tokens = self.intent_projector(slow_context, slow_context_mask)
        residual, gate = self.fast_student(
            force_history,
            force_history_mask,
            intent_tokens,
            state,
            reference_action,
            time_features,
            train=train,
        )
        return residual, gate, intent_tokens
