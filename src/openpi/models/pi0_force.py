import dataclasses
import logging
import typing

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.force_encoder as _force_encoder
import openpi.models.gemma as _gemma
import openpi.models.limoe_simple as _limoe
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0_GuidanceConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    siglip_variant: str = "So400m/14"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    force_encoder: _force_encoder.ForceEncoderConfig = dataclasses.field(
        default_factory=_force_encoder.ForceEncoderConfig
    )
    # Stage 1 keeps this disabled and uses the full force token. Stage 2
    # enables it for the learned missing-force condition.
    enable_null_force_token: bool = False
    force_condition: typing.Literal["full", "null"] = "full"
    # The selected Stage 2 adds a small null-only low-rank adapter to the action
    # vector field. The full-force path bypasses it, preserving Stage 1.
    enable_nominal_adapter: bool = False
    nominal_adapter_rank: int = 32
    nominal_adapter_pose_dims: int = 6

    def __post_init__(self):
        if self.force_condition == "null" and not self.enable_null_force_token:
            raise ValueError("force_condition='null' requires enable_null_force_token=True")
        if self.enable_nominal_adapter and not self.enable_null_force_token:
            raise ValueError("enable_nominal_adapter=True requires enable_null_force_token=True")
        if self.nominal_adapter_rank <= 0:
            raise ValueError("nominal_adapter_rank must be positive")
        if not 0 < self.nominal_adapter_pose_dims <= self.action_dim:
            raise ValueError("nominal_adapter_pose_dims must be in [1, action_dim]")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0_Guidance":
        return Pi0_Guidance(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            history_spec = None
            history_mask_spec = None
            if self.force_encoder.type != "instantaneous":
                history_spec = jax.ShapeDtypeStruct(
                    [batch_size, self.force_encoder.max_history_samples, self.force_encoder.input_dim], jnp.float32
                )
                history_mask_spec = jax.ShapeDtypeStruct(
                    [batch_size, self.force_encoder.max_history_samples], jnp.bool_
                )
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                force_history=history_spec,
                force_history_mask=history_mask_spec,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0_Guidance(_model.BaseModel):
    def __init__(self, config: Pi0_GuidanceConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=config.siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        # self.guidance_proj = nnx.Linear(config.action_dim, 3 * paligemma_config.width, rngs=rngs) ###
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # self.action_state_attention = nnx.MultiHeadAttention(
        #     in_features=action_expert_config.width,
        #     num_heads=8,
        #     rngs=rngs
        # )
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.force_in_proj = nnx.Linear(6, paligemma_config.width, rngs=rngs)
        self.force_encoder_config = config.force_encoder
        self.temporal_force_encoder = (
            _force_encoder.TemporalTCNForceEncoder(config.force_encoder, paligemma_config.width, rngs=rngs)
            if config.force_encoder.type == "tcn"
            else None
        )
        self.null_force_token = (
            nnx.Param(jnp.zeros((paligemma_config.width,), dtype=jnp.float32))
            if config.enable_null_force_token
            else None
        )
        self.nominal_adapter_in = (
            nnx.Linear(action_expert_config.width, config.nominal_adapter_rank, use_bias=False, rngs=rngs)
            if config.enable_nominal_adapter
            else None
        )
        self.nominal_adapter_out = (
            nnx.Linear(
                config.nominal_adapter_rank,
                config.nominal_adapter_pose_dims,
                use_bias=False,
                kernel_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
            if config.enable_nominal_adapter
            else None
        )
        self.nominal_adapter_pose_dims = config.nominal_adapter_pose_dims
        self.force_condition = config.force_condition
        print("paligemma_config.width: ", paligemma_config.width)
        self.limoe = nnx_bridge.ToNNX(
            _limoe.LIMoEBlock(
                mlp_dim=paligemma_config.width,
                num_experts=4,
                num_top_k=1,
                num_heads=paligemma_config.num_heads,
                out_dim=action_expert_config.width,
            )
        )
        self.limoe.lazy_init(jnp.zeros((32, 200, paligemma_config.width)), True, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
        force_condition: typing.Literal["full", "null"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s action_emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b 1 force_emb"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        # obs.state is shape [b, 13] (13 = 7 prio + 6 force, ee pose: xyz+rpy, gripper)
        observations = jnp.zeros_like(obs.state)
        observations = observations.at[:, :7].set(obs.state[:, :7]) ## robot state, xyz + rpy + gripper
        state_token = self.state_proj(observations)[:, None, :] # [b, 1, d]
        # state_token = self.state_proj(obs.state)[:, None, :] # [b, 1, d]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        force_tokens = self.encode_force(obs, train=train, force_condition=force_condition)[:, None, :]
        return tokens, input_mask, ar_mask, force_tokens

    def encode_force(
        self,
        obs: _model.Observation,
        *,
        train: bool = False,
        force_condition: typing.Literal["full", "null"] | None = None,
    ):
        """Return the single force token consumed by the unchanged LIMoE interface."""
        condition = self.force_condition if force_condition is None else force_condition
        if condition == "null":
            if self.null_force_token is None:
                raise ValueError("The null force condition requires a learned null_force_token")
            return jnp.broadcast_to(self.null_force_token.value, (obs.state.shape[0], self.null_force_token.shape[0]))

        encoder_type = self.force_encoder_config.type
        if encoder_type == "instantaneous":
            # Preserve the original parameter name and exact computation.
            return self.force_in_proj(obs.state[:, 7:13])
        if obs.force_history is None or obs.force_history_mask is None:
            raise ValueError(f"force_history and force_history_mask are required for {encoder_type!r} mode")
        assert self.temporal_force_encoder is not None
        return self.temporal_force_encoder(obs.force_history, obs.force_history_mask, train=train)

    def _project_action_velocity(
        self,
        hidden: at.Float[at.Array, "b ah action_emb"],
        force_condition: typing.Literal["full", "null"],
    ) -> _model.Actions:
        velocity = self.action_out_proj(hidden)
        if force_condition == "null" and self.nominal_adapter_in is not None:
            assert self.nominal_adapter_out is not None
            pose_correction = self.nominal_adapter_out(nnx.silu(self.nominal_adapter_in(hidden)))
            velocity = velocity.at[..., : self.nominal_adapter_pose_dims].add(pose_correction)
        return velocity

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        return self.compute_loss_for_force_condition(
            rng,
            observation,
            actions,
            force_condition=self.force_condition,
            train=train,
        )

    def compute_loss_for_force_condition(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        force_condition: typing.Literal["full", "null"],
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        v_t, u_t = self.compute_flow_velocity_for_force_condition(
            rng,
            observation,
            actions,
            force_condition=force_condition,
            train=train,
        )
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def compute_flow_velocity_for_force_condition(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        force_condition: typing.Literal["full", "null"],
        train: bool = False,
    ) -> tuple[_model.Actions, _model.Actions]:
        """Return predicted and target flow velocities for one force condition."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, force_tokens = self.embed_suffix(
            observation, x_t, time, train=train, force_condition=force_condition
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        limoe_out = self.limoe(jnp.concatenate([prefix_out, force_tokens], axis=1)) ## prefix_out is vlm
        hidden = limoe_out[0][:, -self.action_horizon :] + suffix_out[:, -self.action_horizon :]
        v_t = self._project_action_velocity(hidden, force_condition)
        return v_t, u_t

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        return self.sample_actions_for_force_condition(
            rng,
            observation,
            force_condition=self.force_condition,
            num_steps=num_steps,
        )

    def sample_actions_for_force_condition(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        force_condition: typing.Literal["full", "null"],
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """Sample one explicit force condition for paired Teacher distillation."""
        if force_condition == "null" and self.null_force_token is None:
            raise ValueError("Null-force sampling requires a learned null_force_token")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_out_fix, kv_cache = self._prepare_action_prefix(observation)
        return self._sample_actions_with_prefix(
            rng,
            observation,
            force_condition=force_condition,
            num_steps=num_steps,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_out_fix=prefix_out_fix,
            kv_cache=kv_cache,
        )

    def sample_nominal_actions_and_context(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ):
        """Run the null path once and expose its contextualized prefix for intent projection."""
        if self.null_force_token is None:
            raise ValueError("Nominal context sampling requires a learned null_force_token")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_out_fix, kv_cache = self._prepare_action_prefix(observation)
        actions = self._sample_actions_with_prefix(
            rng,
            observation,
            force_condition="null",
            num_steps=num_steps,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_out_fix=prefix_out_fix,
            kv_cache=kv_cache,
        )
        return actions, prefix_out_fix, prefix_mask

    def sample_paired_actions_and_context(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ):
        """Efficiently generate matched full/null actions and one shared Slow context."""
        if self.null_force_token is None:
            raise ValueError("Paired sampling requires a learned null_force_token")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_out_fix, kv_cache = self._prepare_action_prefix(observation)
        kwargs = {
            "num_steps": num_steps,
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
            "prefix_out_fix": prefix_out_fix,
            "kv_cache": kv_cache,
        }
        # The identical RNG gives both flow samplers exactly the same initial
        # noise; force condition is the only difference between their outputs.
        full = self._sample_actions_with_prefix(
            rng, observation, force_condition="full", **kwargs
        )
        null = self._sample_actions_with_prefix(
            rng, observation, force_condition="null", **kwargs
        )
        return full, null, prefix_out_fix, prefix_mask

    def _prepare_action_prefix(self, observation: _model.Observation):
        """Encode vision/language once for both action sampling and cached intent."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out_fix, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return prefix_tokens, prefix_mask, prefix_out_fix, kv_cache

    def _sample_actions_with_prefix(
        self,
        rng,
        observation,
        *,
        force_condition,
        num_steps,
        prefix_tokens,
        prefix_mask,
        prefix_out_fix,
        kv_cache,
    ):
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, force_tokens = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                force_condition=force_condition,
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None

            # The cached suffix pass returns no prefix tensor; reuse the fixed
            # prefix output computed before the denoising loop.
            limoe_out = self.limoe(jnp.concatenate([prefix_out_fix, force_tokens], axis=1)) ## prefix_out is vlm
            hidden = limoe_out[0][:, -self.action_horizon :] + suffix_out[:, -self.action_horizon :]
            v_t = self._project_action_velocity(hidden, force_condition)
            # v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
