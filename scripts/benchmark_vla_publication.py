"""Measure BS=1 VLA latency and realized packet publication rate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

from flax import nnx
import jax
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader, weight_loaders
from evaluate_forcevla_checkpoint import _checkpoint_params_path, _load_batch, _sample_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", default="forcevla_button_temporal_100hz_val")
    parser.add_argument("--path-name", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--mode", choices=("full", "null"), default="full")
    args = parser.parse_args()

    config = dataclasses.replace(config_lib.get_config(args.config), batch_size=1, num_workers=0)
    source = config_lib.get_config(args.data_config)
    data_config = source.data.create(source.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    observations = []
    noises = []
    for index in range(32):
        observation, _, _ = _load_batch(dataset, _sample_transform(data_config), np.asarray([index % len(dataset)]))
        observations.append(jax.device_put(observation))
        noises.append(jax.device_put(model_lib.row_keyed_noise(
            index, np.asarray([index]), action_horizon=config.model.action_horizon, action_dim=config.model.action_dim
        )))
    shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference = nnx.state(shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.checkpoint)), missing_regex=r"a^"
    ).load(reference)
    model = config.model.load(params, remove_extra_params=False)
    if args.mode == "null":
        infer = nnx_utils.module_jit(model.sample_nominal_actions_and_context)
    else:
        infer = nnx_utils.module_jit(model.sample_actions)
    jax.block_until_ready(infer(jax.random.key(0), observations[0], num_steps=args.flow_steps, noise=noises[0]))

    period = 1.0 / args.request_rate
    started = time.monotonic()
    deadline = started + args.duration
    next_request = started
    latencies = []
    packets = 0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_request:
            time.sleep(next_request - now)
        index = packets % len(observations)
        began = time.monotonic()
        result = infer(jax.random.key(packets + 1), observations[index], num_steps=args.flow_steps, noise=noises[index])
        jax.block_until_ready(result)
        completed = time.monotonic()
        if completed <= deadline:
            packets += 1
            latencies.append((completed - began) * 1000.0)
        next_request = max(next_request + period, completed)
    elapsed = time.monotonic() - started
    values = np.asarray(latencies)
    print(json.dumps({
        "path": args.path_name,
        "batch_size": 1,
        "measurement_duration_s": elapsed,
        "published_packets": packets,
        "mean_latency_ms": float(values.mean()),
        "std_latency_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "realized_publication_rate_hz": packets / elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
