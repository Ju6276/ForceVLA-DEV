"""Measure independently published Slow/Fast packets under shared-GPU contention.

The replay clocks match the Button data: RGB/state advance at 30 Hz and the Fast
request clock advances at 100 Hz.  Each worker loads its trained checkpoint and
records completion timestamps; publication rate is counted, not inferred from a
separately measured latency. Both workers use deployment batch size one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import threading
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import fast_dataset
from openpi.training import weight_loaders

from evaluate_fast_residual import _load_model as load_fast_model
from evaluate_fast_residual import _run_metadata
from evaluate_forcevla_checkpoint import _checkpoint_params_path, _load_batch, _sample_transform


def _load_slow(args):
    config = dataclasses.replace(config_lib.get_config(args.slow_config), batch_size=1, num_workers=0)
    source = config_lib.get_config(args.data_config)
    data_config = source.data.create(source.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transform = _sample_transform(data_config)
    observations = []
    for offset in range(args.replay_observations):
        indices = np.asarray([offset % len(dataset)], dtype=np.int64)
        observation, _, _ = _load_batch(dataset, transform, indices)
        observations.append(jax.device_put(observation))

    shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference = nnx.state(shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.slow_checkpoint)), missing_regex=r"a^"
    ).load(reference)
    model = config.model.load(params, remove_extra_params=False)
    infer = nnx_utils.module_jit(model.sample_nominal_actions_and_context)
    noises = [
        jax.device_put(
            model_lib.row_keyed_noise(
                args.seed + index,
                np.asarray([index], dtype=np.int64),
                action_horizon=config.model.action_horizon,
                action_dim=config.model.action_dim,
            )
        )
        for index in range(args.replay_observations)
    ]
    return infer, observations, noises


def _load_fast(args):
    cache = fast_dataset.load_slow_cache(args.slow_cache)
    arrays = fast_dataset.load_stage3_fast_arrays(args.targets, chunk_steps=cache.chunk_steps)
    checkpoint = args.fast_checkpoint.resolve()
    metadata = _run_metadata(checkpoint)
    model, _ = load_fast_model(
        checkpoint,
        chunk_steps=cache.chunk_steps,
        force_blind_staleness=bool(metadata.get("config", {}).get("force_blind_staleness", True)),
    )
    positions = np.flatnonzero(cache.row_key_positions >= 0)[: args.replay_observations]
    inputs = []
    for position in positions:
        key = int(cache.row_key_positions[position])
        inputs.append(tuple(jax.device_put(value) for value in (
            jnp.asarray(cache.context_tokens[key : key + 1], dtype=jnp.float32),
            jnp.asarray(cache.context_mask[key : key + 1], dtype=jnp.bool_),
            jnp.asarray(arrays.force_history[position : position + 1], dtype=jnp.float32),
            jnp.asarray(arrays.force_history_mask[position : position + 1], dtype=jnp.bool_),
            jnp.asarray(arrays.state[position : position + 1], dtype=jnp.float32),
            jnp.asarray(cache.reference_actions[position : position + 1, 0], dtype=jnp.float32),
            jnp.asarray(cache.time_features[position : position + 1], dtype=jnp.float32),
        )))

    @nnx.jit
    def infer(module, *values):
        residual, staleness, _, _ = module(*values, train=False)
        return residual, staleness

    return model, infer, inputs


def _summary(name: str, publishes: list[float], latencies: list[float], duration: float, requests: int) -> dict:
    values = np.asarray(latencies, dtype=np.float64) * 1000.0
    return {
        "path": name,
        "requests": requests,
        "published_packets": len(publishes),
        "realized_publication_rate_hz": len(publishes) / duration,
        "completion_ratio": len(publishes) / max(requests, 1),
        "mean_completion_latency_ms": float(np.mean(values)) if len(values) else None,
        "std_completion_latency_ms": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--fast-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--slow-config", default="forcevla_button_temporal_stage2_null_bc")
    parser.add_argument("--data-config", default="forcevla_button_temporal_100hz_val")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--slow-period", type=float, default=0.1)
    parser.add_argument("--fast-period", type=float, default=0.01)
    parser.add_argument("--replay-observations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    slow_infer, slow_observations, slow_noises = _load_slow(args)
    fast_model, fast_infer, fast_inputs = _load_fast(args)

    # Compile and warm both paths before the publication window begins.
    jax.block_until_ready(slow_infer(
        jax.random.key(args.seed), slow_observations[0], num_steps=10, noise=slow_noises[0]
    ))
    jax.block_until_ready(fast_infer(fast_model, *fast_inputs[0]))

    start_event = threading.Event()
    stop_event = threading.Event()
    slow_publish: list[float] = []
    fast_publish: list[float] = []
    slow_latency: list[float] = []
    fast_latency: list[float] = []
    request_counts = {"slow": 0, "fast": 0}
    measurement_start = 0.0

    def slow_worker():
        start_event.wait()
        next_request = measurement_start
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_request:
                stop_event.wait(next_request - now)
                continue
            elapsed = now - measurement_start
            index = int(elapsed * 30.0) % len(slow_observations)
            request_counts["slow"] += 1
            began = time.monotonic()
            result = slow_infer(
                jax.random.key(args.seed + request_counts["slow"]),
                slow_observations[index],
                num_steps=10,
                noise=slow_noises[index],
            )
            jax.block_until_ready(result)
            completed = time.monotonic()
            if not stop_event.is_set():
                slow_latency.append(completed - began)
                slow_publish.append(completed)
            next_request = max(next_request + args.slow_period, completed)

    def fast_worker():
        start_event.wait()
        next_request = measurement_start
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_request:
                stop_event.wait(next_request - now)
                continue
            # Button state/action anchors are 30 Hz.  The 100 Hz request clock reads
            # the latest causal force-history/state tensor (ZOH between anchors).
            elapsed = now - measurement_start
            index = int(elapsed * 30.0) % len(fast_inputs)
            request_counts["fast"] += 1
            began = time.monotonic()
            result = fast_infer(fast_model, *fast_inputs[index])
            jax.block_until_ready(result)
            completed = time.monotonic()
            if not stop_event.is_set():
                fast_latency.append(completed - began)
                fast_publish.append(completed)
            next_request += args.fast_period
            # Do not burst old requests after a long GPU stall; count them as missed.
            if next_request < completed:
                next_request = completed

    threads = [threading.Thread(target=slow_worker), threading.Thread(target=fast_worker)]
    measurement_start = time.monotonic() + 0.1
    for thread in threads:
        thread.start()
    start_event.set()
    time.sleep(args.duration)
    stop_event.set()
    for thread in threads:
        thread.join(timeout=10.0)
    measured = time.monotonic() - measurement_start

    payload = {
        "device": str(jax.devices()[0]),
        "requested_replay_rates_hz": {"rgb_state": 30.0, "force_fast_tick": 100.0},
        "slow_batch_size": 1,
        "fast_batch_size": 1,
        "measurement_duration_s": measured,
        "reference": _summary("ForceDelta reference", slow_publish, slow_latency, measured, request_counts["slow"]),
        "residual": _summary("ForceDelta residual", fast_publish, fast_latency, measured, request_counts["fast"]),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
