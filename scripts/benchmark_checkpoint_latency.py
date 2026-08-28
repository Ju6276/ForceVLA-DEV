"""Batch-one checkpoint latency benchmark on real Button observations.

This is deliberately separate from the architecture-only benchmark: every path
loads its trained checkpoint, receives a real held-out input, and is synchronized
after every forward so JAX dispatch time is not mistaken for inference latency.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
import types

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import fast_dataset
from openpi.training import weight_loaders

from evaluate_fast_residual import _load_model as load_fast_model
from evaluate_fast_residual import _run_metadata
from evaluate_forcevla_checkpoint import _checkpoint_params_path, _load_batch, _sample_transform


def _measure(fn, *, warmup: int, repeats: int) -> np.ndarray:
    for index in range(warmup):
        jax.block_until_ready(fn(index))
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        started = time.perf_counter_ns()
        jax.block_until_ready(fn(index + warmup))
        samples[index] = (time.perf_counter_ns() - started) / 1e6
    return samples


def _report(path: str, samples: np.ndarray, *, checkpoint: pathlib.Path, batch_size: int) -> None:
    mean_ms = float(np.mean(samples))
    std_ms = float(np.std(samples, ddof=1))
    payload = {
        "path": path,
        "checkpoint": str(checkpoint.resolve()),
        "device": str(jax.devices()[0]),
        "batch_size": batch_size,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "packet_rate_hz": 1000.0 / mean_ms,
        "repeats": len(samples),
    }
    print(json.dumps(payload, indent=2))


def benchmark_forcevla(args: argparse.Namespace) -> None:
    config = dataclasses.replace(config_lib.get_config(args.config), batch_size=args.batch_size, num_workers=0)
    source_config = config_lib.get_config(args.data_config)
    data_config = source_config.data.create(source_config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    observation, _, _ = _load_batch(
        dataset, _sample_transform(data_config), np.arange(args.batch_size, dtype=np.int64)
    )
    observation = jax.device_put(observation)

    shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference = nnx.state(shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.checkpoint)), missing_regex=r"a^"
    ).load(reference)
    model = config.model.load(params, remove_extra_params=False)
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    noises = [
        jax.device_put(
            model_lib.row_keyed_noise(
                args.seed + index,
                np.arange(index, index + args.batch_size, dtype=np.int64),
                action_horizon=config.model.action_horizon,
                action_dim=config.model.action_dim,
            )
        )
        for index in range(max(args.warmup + args.repeats, 2))
    ]

    if args.single_camera is None:
        def infer(index: int):
            return sample_actions(
                jax.random.key(args.seed + index),
                observation,
                num_steps=args.flow_steps,
                noise=noises[index],
            )
    else:
        if args.single_camera not in observation.images:
            raise ValueError(f"Unknown camera {args.single_camera}; choose from {tuple(observation.images)}")

        def sample_one_camera_impl(module, rng, obs, noise):
            obs = model_lib.preprocess_observation(None, obs, train=False)
            obs = obs.replace(
                images={args.single_camera: obs.images[args.single_camera]},
                image_masks={args.single_camera: obs.image_masks[args.single_camera]},
            )
            prefix_tokens, prefix_mask, prefix_out, kv_cache = module._prepare_action_prefix(obs)
            return module._sample_actions_with_prefix(
                rng,
                obs,
                force_condition=module.force_condition,
                num_steps=args.flow_steps,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                prefix_out_fix=prefix_out,
                kv_cache=kv_cache,
                noise=noise,
            )

        sample_one_camera = nnx_utils.module_jit(types.MethodType(sample_one_camera_impl, model))

        def infer(index: int):
            return sample_one_camera(jax.random.key(args.seed + index), observation, noises[index])

    samples = _measure(infer, warmup=args.warmup, repeats=args.repeats)
    _report(args.path_name, samples, checkpoint=args.checkpoint, batch_size=args.batch_size)


def benchmark_pi0(args: argparse.Namespace) -> None:
    model_config = pi0.Pi0Config()
    source_config = config_lib.get_config(args.data_config)
    data_config = source_config.data.create(source_config.assets_dirs, model_config)
    dataset = data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    observation, _, _ = _load_batch(
        dataset, _sample_transform(data_config), np.arange(args.batch_size, dtype=np.int64)
    )
    observation = jax.device_put(observation)

    shape = nnx.eval_shape(model_config.create, jax.random.key(0))
    reference = nnx.state(shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.checkpoint)), missing_regex=r"a^"
    ).load(reference)
    model = model_config.load(params, remove_extra_params=False)
    sample_actions = nnx_utils.module_jit(model.sample_actions)

    def infer(index: int):
        return sample_actions(jax.random.key(args.seed + index), observation, num_steps=args.flow_steps)

    samples = _measure(infer, warmup=args.warmup, repeats=args.repeats)
    _report(args.path_name, samples, checkpoint=args.checkpoint, batch_size=args.batch_size)


def benchmark_fast(args: argparse.Namespace) -> None:
    cache = fast_dataset.load_slow_cache(args.slow_cache)
    arrays = fast_dataset.load_stage3_fast_arrays(args.targets, chunk_steps=cache.chunk_steps)
    checkpoint = args.checkpoint.resolve()
    metadata = _run_metadata(checkpoint)
    model, _ = load_fast_model(
        checkpoint,
        chunk_steps=cache.chunk_steps,
        force_blind_staleness=bool(metadata.get("config", {}).get("force_blind_staleness", True)),
    )
    positions = np.flatnonzero(cache.row_key_positions >= 0)[: args.batch_size]
    key_positions = cache.row_key_positions[positions]
    inputs = (
        jax.device_put(jnp.asarray(cache.context_tokens[key_positions], dtype=jnp.float32)),
        jax.device_put(jnp.asarray(cache.context_mask[key_positions], dtype=jnp.bool_)),
        jax.device_put(jnp.asarray(arrays.force_history[positions], dtype=jnp.float32)),
        jax.device_put(jnp.asarray(arrays.force_history_mask[positions], dtype=jnp.bool_)),
        jax.device_put(jnp.asarray(arrays.state[positions], dtype=jnp.float32)),
        jax.device_put(jnp.asarray(cache.reference_actions[positions, 0], dtype=jnp.float32)),
        jax.device_put(jnp.asarray(cache.time_features[positions], dtype=jnp.float32)),
    )

    @nnx.jit
    def infer(module, *values):
        residual, staleness, _, _ = module(*values, train=False)
        return residual, staleness

    def one_step(_index: int):
        return infer(model, *inputs)

    samples = _measure(one_step, warmup=args.warmup, repeats=args.repeats)
    _report(args.path_name, samples, checkpoint=checkpoint, batch_size=args.batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", choices=("forcevla", "pi0", "fast"))
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--path-name", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--single-camera", default=None)
    parser.add_argument("--config")
    parser.add_argument("--data-config")
    parser.add_argument("--targets", type=pathlib.Path)
    parser.add_argument("--slow-cache", type=pathlib.Path)
    args = parser.parse_args()
    if args.path == "forcevla":
        if not args.config or not args.data_config:
            parser.error("forcevla requires --config and --data-config")
        benchmark_forcevla(args)
    elif args.path == "pi0":
        if not args.data_config:
            parser.error("pi0 requires --data-config")
        benchmark_pi0(args)
    else:
        if args.targets is None or args.slow_cache is None:
            parser.error("fast requires --targets and --slow-cache")
        benchmark_fast(args)


if __name__ == "__main__":
    main()
