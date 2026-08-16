"""One-step ForceVLA training smoke test on a released USB insertion episode."""

import argparse
from io import BytesIO
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
import polars as pl

from openpi.models import model as model_lib
from openpi.models.force_encoder import ForceEncoderConfig
from openpi.models.pi0_force import Pi0_GuidanceConfig


def _decode_image(cell) -> np.ndarray:
    image = Image.open(BytesIO(cell["bytes"])).convert("RGB").resize((224, 224))
    return np.asarray(image, dtype=np.float32) / 127.5 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("instantaneous", "tcn"), required=True)
    parser.add_argument("--episode", type=pathlib.Path, required=True)
    args = parser.parse_args()

    frame, history_length, horizon, batch_size = 100, 3, 8, 4
    table = pl.read_parquet(
        args.episode,
        columns=["action", "observation.state", "observation.image", "observation.wrist_image"],
    ).slice(frame - history_length + 1, history_length + horizon - 1)
    states = np.stack(table["observation.state"].to_list()).astype(np.float32)
    actions = np.stack(table["action"].to_list())[history_length - 1 : history_length - 1 + horizon].astype(
        np.float32
    )
    base_image = _decode_image(table["observation.image"][history_length - 1])
    wrist_image = _decode_image(table["observation.wrist_image"][history_length - 1])
    state = np.pad(states[history_length - 1], (0, 19))
    actions = np.pad(actions, ((0, 0), (0, 25)))

    def repeat(value):
        return jnp.repeat(jnp.asarray(value)[None], batch_size, axis=0)

    use_history = args.mode == "tcn"
    observation = model_lib.Observation(
        images={
            "base_0_rgb": repeat(base_image),
            "left_wrist_0_rgb": repeat(wrist_image),
            "right_wrist_0_rgb": jnp.zeros_like(repeat(base_image)),
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), bool),
            "left_wrist_0_rgb": jnp.ones((batch_size,), bool),
            "right_wrist_0_rgb": jnp.zeros((batch_size,), bool),
        },
        state=repeat(state),
        force_history=repeat(states[:history_length, 7:13]) if use_history else None,
        force_history_mask=jnp.ones((batch_size, history_length), bool) if use_history else None,
        tokenized_prompt=jnp.zeros((batch_size, 16), jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 16), bool),
    )
    target_actions = repeat(actions)
    force_config = ForceEncoderConfig(
        type=args.mode,
        sampling_rate_hz=30,
        window_ms=100,
        history_source="aligned_state",
    )
    # Dummy language/action widths and the smallest SigLIP variant keep this a
    # quick architecture smoke test. The ForceVLA loss and module path are real.
    config = Pi0_GuidanceConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        siglip_variant="mu/14",
        action_horizon=horizon,
        max_token_len=16,
        force_encoder=force_config,
    )
    model = config.create(jax.random.key(0))

    def loss_fn(module):
        return jnp.mean(module.compute_loss(jax.random.key(1), observation, target_actions, train=False))

    loss_before, gradients = nnx.value_and_grad(loss_fn)(model)
    parameters = nnx.state(model, nnx.Param)
    optimizer = optax.sgd(1e-4)
    updates, _ = optimizer.update(gradients, optimizer.init(parameters), parameters)
    nnx.update(model, optax.apply_updates(parameters, updates))
    loss_after = loss_fn(model)

    print(f"mode={args.mode}")
    print(f"batch_size={batch_size}")
    print(f"force_history_shape={None if observation.force_history is None else observation.force_history.shape}")
    print(f"loss_before={float(loss_before):.8f}")
    print(f"loss_after={float(loss_after):.8f}")
    print(f"grad_norm={float(optax.global_norm(gradients)):.8f}")
    print(f"finite={bool(jnp.isfinite(loss_before) & jnp.isfinite(loss_after))}")
    print(f"updated={bool(jnp.abs(loss_after - loss_before) > 0)}")


if __name__ == "__main__":
    main()
