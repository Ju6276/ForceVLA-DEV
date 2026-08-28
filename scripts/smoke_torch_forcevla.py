"""Load the converted Temporal ForceVLA and run a real Button observation."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from types import SimpleNamespace

import numpy as np
from safetensors.torch import load_model
import torch

from openpi.models import force_encoder
from openpi.models_pytorch.forcevla_pytorch import ForceVLAPytorch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--observation", type=pathlib.Path, required=True)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--debug-features", type=pathlib.Path)
    args = parser.parse_args()
    config = SimpleNamespace(
        pi05=False,
        action_dim=32,
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        dtype="bfloat16",
        compile_model=False,
    )
    encoder_config = force_encoder.ForceEncoderConfig(
        type="tcn", sampling_rate_hz=100, window_ms=100, max_sample_age_ms=12
    )
    model = ForceVLAPytorch(config, encoder_config)
    load_model(model, str(args.checkpoint / "model.safetensors"), strict=True)
    device = torch.device("cuda")
    model.eval().to(device)

    data = np.load(args.observation)
    image_names = sorted(key.split("::", 1)[1] for key in data.files if key.startswith("image::"))
    tensor = lambda name, dtype=None: torch.as_tensor(data[name], device=device, dtype=dtype)
    observation = SimpleNamespace(
        images={
            name: tensor(f"image::{name}", torch.float32).permute(0, 3, 1, 2).contiguous()
            for name in image_names
        },
        image_masks={name: tensor(f"image_mask::{name}", torch.bool) for name in image_names},
        state=tensor("state", torch.float32),
        force_history=tensor("force_history", torch.float32),
        force_history_mask=tensor("force_history_mask", torch.bool),
        tokenized_prompt=tensor("tokenized_prompt", torch.long),
        tokenized_prompt_mask=tensor("tokenized_prompt_mask", torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )
    noise = torch.zeros((1, 50, 32), device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        actions = model.sample_actions(device, observation, noise=noise, num_steps=args.flow_steps)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    result = {
        "shape": list(actions.shape),
        "finite": bool(torch.isfinite(actions).all()),
        "latency_ms_including_first_forward": elapsed,
        "valid_action_abs_mean": float(actions[..., :10].abs().mean()),
    }
    if args.output is not None:
        np.save(args.output, actions.float().cpu().numpy())
    if args.debug_features is not None:
        np.savez(
            args.debug_features,
            prefix_context=model.last_prefix_context.float().cpu().numpy(),
            force_token=model.last_force_token.float().cpu().numpy(),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
