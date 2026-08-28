"""Export one deterministic transformed Button observation for cross-framework checks."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np

from openpi.training import config as config_lib
from openpi.training import data_loader
from evaluate_forcevla_checkpoint import _load_batch, _sample_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    config = dataclasses.replace(config_lib.get_config("forcevla_button_temporal_100hz_val"), batch_size=1, num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    observation, _, _ = _load_batch(dataset, _sample_transform(data_config), np.asarray([args.row]))
    arrays = {
        "state": np.asarray(observation.state),
        "force_history": np.asarray(observation.force_history),
        "force_history_mask": np.asarray(observation.force_history_mask),
        "tokenized_prompt": np.asarray(observation.tokenized_prompt),
        "tokenized_prompt_mask": np.asarray(observation.tokenized_prompt_mask),
    }
    for name, image in observation.images.items():
        arrays[f"image::{name}"] = np.asarray(image)
        arrays[f"image_mask::{name}"] = np.asarray(observation.image_masks[name])
    np.savez(args.output, **arrays)


if __name__ == "__main__":
    main()
