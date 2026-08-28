"""PyTorch inference model for the unified Temporal ForceVLA teacher."""

from __future__ import annotations

import torch

from openpi.models_pytorch.force_encoder import TemporalTCNForceEncoder
from openpi.models_pytorch.limoe import BatchOneLIMoE
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


class ForceVLAPytorch(PI0Pytorch):
    """Stage-1 full-force inference path with the original 32D padded action representation."""

    def __init__(self, config, force_encoder_config):
        super().__init__(config)
        self.temporal_force_encoder = TemporalTCNForceEncoder(force_encoder_config, output_dim=2048)
        self.limoe = BatchOneLIMoE(input_dim=2048, output_dim=1024, num_heads=8, num_experts=4)

    def embed_suffix(self, state, noisy_actions, timestep):
        # ForceVLA's robotics state token contains only xyz + rotation-6D +
        # gripper. The remaining padded coordinates, including the old wrench
        # slots, are explicitly zeroed exactly as in the JAX model.
        state_without_force = torch.zeros_like(state)
        state_without_force[:, :10] = state[:, :10]
        return super().embed_suffix(state_without_force, noisy_actions, timestep)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        if observation.force_history is None or observation.force_history_mask is None:
            raise ValueError("Temporal ForceVLA requires force_history and force_history_mask")
        if observation.state.shape[0] != 1:
            raise ValueError("The converted deployment path currently supports BS=1")

        batch_size = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((batch_size, self.config.action_horizon, self.config.action_dim), device)
        images, image_masks, language, language_mask, state = self._preprocess_observation(observation, train=False)
        prefix, prefix_mask, prefix_ar = self.embed_prefix(images, image_masks, language, language_mask)
        prefix_attention = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_mask, prefix_ar))
        prefix_positions = torch.cumsum(prefix_mask, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        prefix_outputs, cache = self.paligemma_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        prefix_context = prefix_outputs[0].float()
        self.last_prefix_context = prefix_context.detach()
        self.last_force_token = force_token = self.temporal_force_encoder(
            observation.force_history.to(device=device, dtype=torch.float32),
            observation.force_history_mask.to(device=device),
        ).float()

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        actions = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(state, actions, time.expand(batch_size))
            suffix_length = suffix_mask.shape[1]
            prefix_to_suffix = prefix_mask[:, None, :].expand(batch_size, suffix_length, prefix_mask.shape[1])
            suffix_attention = make_att_2d_masks(suffix_mask, suffix_ar)
            full_attention = self._prepare_attention_masks_4d(
                torch.cat([prefix_to_suffix, suffix_attention], dim=-1)
            ).to(
                dtype=self.paligemma_with_expert.gemma_expert.model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
            )
            positions = torch.sum(prefix_mask, dim=-1)[:, None] + torch.cumsum(suffix_mask, dim=1) - 1
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=full_attention,
                position_ids=positions,
                past_key_values=cache,
                inputs_embeds=[None, suffix],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
            suffix_output = outputs[1][:, -self.config.action_horizon :].float()
            fusion = self.limoe(torch.cat([prefix_context, force_token[:, None, :]], dim=1))
            hidden = fusion[:, -self.config.action_horizon :] + suffix_output
            velocity = self.action_out_proj(hidden)
            actions = actions + dt * velocity
            time = time + dt
        return actions
