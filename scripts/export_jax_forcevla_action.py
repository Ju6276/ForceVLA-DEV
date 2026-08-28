"""Export the JAX Teacher action for the same deterministic Button smoke row."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader, weight_loaders
from evaluate_forcevla_checkpoint import _checkpoint_params_path, _load_batch, _sample_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--debug-features", type=pathlib.Path)
    args = parser.parse_args()
    config = dataclasses.replace(config_lib.get_config("forcevla_button_temporal_100hz_val"), batch_size=1, num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    observation, _, _ = _load_batch(dataset, _sample_transform(data_config), np.asarray([0]))
    shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference = nnx.state(shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.checkpoint)), missing_regex=r"a^"
    ).load(reference)
    model = config.model.load(params, remove_extra_params=False)
    if args.debug_features is not None:
        from openpi.models import model as model_lib
        processed_observation = model_lib.preprocess_observation(None, jax.device_put(observation), train=False)
        _, _, prefix_context, _ = model._prepare_action_prefix(processed_observation)
        force_token = model.encode_force(processed_observation, force_condition="full")
        np.savez(
            args.debug_features,
            prefix_context=np.asarray(jax.block_until_ready(prefix_context.astype(jnp.float32))),
            force_token=np.asarray(jax.block_until_ready(force_token.astype(jnp.float32))),
        )
    sample = nnx_utils.module_jit(model.sample_actions)
    noise = jnp.zeros((1, config.model.action_horizon, config.model.action_dim), dtype=jnp.float32)
    actions = sample(jax.random.key(0), jax.device_put(observation), num_steps=args.flow_steps, noise=noise)
    np.save(args.output, np.asarray(jax.block_until_ready(actions)))


if __name__ == "__main__":
    main()
