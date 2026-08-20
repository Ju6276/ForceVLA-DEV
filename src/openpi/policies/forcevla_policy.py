import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_forcevla_example() -> dict:
    """Creates a random input example compatible with Flexiv config."""
    return {
        "state": np.ones((14,)),  # observation.state, 7 ee pose, 1 gripper, 6 force
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Forcevla_inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0
    use_force_history: bool = False
    force_history_from_state: bool = False
    # Restrict the state pathway to robot proprioception for a force-free
    # student. For example, 7 keeps xyz, rpy, and gripper while replacing the
    # wrench dimensions with padding that is independent of force.
    robot_state_dims: int | None = None

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        raw_state = np.asarray(data["state"])
        current_state = raw_state[-1] if self.force_history_from_state else raw_state
        if self.robot_state_dims is not None:
            if self.robot_state_dims <= 0 or self.robot_state_dims > current_state.shape[-1]:
                raise ValueError(
                    f"robot_state_dims must be in [1, {current_state.shape[-1]}], got {self.robot_state_dims}"
                )
            model_state = current_state[..., : self.robot_state_dims]
        else:
            model_state = current_state
        state = transforms.pad_to_dim(model_state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["image"])
        left_wrist_image = _parse_image(data["wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        if self.use_force_history:
            if self.force_history_from_state:
                if raw_state.ndim != 2 or raw_state.shape[-1] < 13:
                    raise ValueError(f"Expected aligned state history [N, >=13], got {raw_state.shape}")
                history = np.asarray(raw_state[:, 7:13], dtype=np.float32)
                mask = np.ones(history.shape[:-1], dtype=np.bool_)
            else:
                if "force_history" not in data or "force_history_mask" not in data:
                    raise ValueError("Temporal force mode requires force_history and force_history_mask")
                history = np.asarray(data["force_history"], dtype=np.float32)
                mask = np.asarray(data["force_history_mask"], dtype=np.bool_)
            if history.ndim != 2 or history.shape[-1] != 6 or mask.shape != history.shape[:-1]:
                raise ValueError(
                    f"Expected force_history [N, 6] and mask [N], got {history.shape} and {mask.shape}"
                )
            inputs["force_history"] = history
            inputs["force_history_mask"] = mask

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class Forcevla_outputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For forcevla, we only return the first 7 actions (since the rest is padding), xyz  + RPY + gripper
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
