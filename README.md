# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
## 
ForceVLA is based on the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA)； Both training and inference are based on π₀.

> **Teacher branch scope.** This branch builds a unified Temporal ForceVLA teacher by replacing
> only the instantaneous force front-end with a configurable causal temporal encoder. FVLMoE, the
> Action Expert, action representation, and flow-matching objective remain unchanged. Teacher-student
> distillation, force-masked teacher passes, nominal/residual students, and slow-fast execution are
> intentionally not implemented at this stage.

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.


## dataset
https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data

## Installation

When cloning this repo, make sure to update submodules:

```bash
conda create -n forcevla python=3.11 -y

```

```bash
python -m pip install --upgrade pip setuptools wheel
conda install -c nvidia cuda-toolkit=12.8
```

```bash
cd lerobot/
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e .
```

```bash
cd ./openpi
pip install -e .
```

```bash
cd dlimp/
pip install -e .
```

```bash
cd packages/
cd openpi-client/
pip install -e .
```

```bash
cd flaxformer/
pip install -e .
```
## train policy
```bash
export HF_LEROBOT_HOME="xxxxxx"
python scripts/compute_norm_stats.py --config-name forcevla_lora 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  python scripts/train.py forcevla_lora --exp-name=my_experiment --overwrite  --batch_size 32 --save_interval 2000 --keep_period 10000
```

### Temporal force front-end

ForceVLA retains its checkpoint-compatible instantaneous front-end by default. The model's
`force_encoder.type` can be set to `instantaneous`, `avg_pool`, `max_pool`, or `tcn`. For example:

```python
Pi0_GuidanceConfig(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    force_encoder=ForceEncoderConfig(
        type="tcn",
        sampling_rate_hz=200,
        window_ms=100,
        hidden_dims=(1024, 1024, 1024, 1024),
        dilations=(1, 2, 4, 8),
        dropout_rate=0.1,
        aggregation="last",  # or "mean"
        history_source="timestamp_stream",  # or "aligned_state" for released 30 Hz ForceVLA data
    ),
)
```

Temporal modes expect each raw dataset item to retain `observation.force` as `[N, 6]`,
`observation.force_timestamps` as `[N]`, and the VLA `timestamp` as a scalar (all timestamps in
seconds). `LeRobotForcevlaDataConfig` exposes these key names. It extracts only samples in
`(timestamp - window_ms, timestamp]`, keeps native-rate samples, and returns a left-padded history
plus validity mask. FVLMoE, the Action Expert, action target, and training objective are unchanged.

The released ForceVLA LeRobot data stores one wrench inside `observation.state` per 30 Hz frame.
For a low-rate compatibility experiment, set `history_source="aligned_state"` and
`sampling_rate_hz=30`; the loader requests causal negative frame offsets and never labels this as
native-rate force.

The named training configs make the mode explicit:

```text
forcevla_lora                    = original instantaneous 6D force
forcevla_temporal_lora_aligned   = RGB-aligned force history through the causal TCN (30 Hz default)
```

To train the continuous 30 Hz history variant on the released ForceVLA-style data:

```bash
python scripts/compute_norm_stats.py --config-name forcevla_temporal_lora_aligned
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    forcevla_temporal_lora_aligned \
    --exp-name=forcevla_temporal_30hz \
    --overwrite \
    --batch_size=4
```

Change the `repo_id` in `forcevla_temporal_lora_aligned` to select another ForceVLA-format task. Do
not launch `forcevla_lora` when intending to use history: that named config deliberately preserves
the original instantaneous baseline. The aligned config defaults to the released dataset's 30 Hz.
For a dataset where RGB, state, and force are all 20 Hz, set `sampling_rate_hz=20`; the same 100 ms
physical window then contains 2 force frames. The configured sampling rate must match the dataset
FPS because the model input shape is static at initialization.

For future native-rate datasets, the loader can keep LeRobot RGB/state/action rows at their original
rate and join one timestamped NPZ force sidecar per episode. Each sidecar must use the same
episode-relative clock as the LeRobot `timestamp` and contain `force: [M, 6]` plus
`timestamps: [M]`, for example `force/episode_000000.npz`. Configure it with:

```python
LeRobotForcevlaDataConfig(
    repo_id="your/lerobot_dataset",
    native_force_sidecar=NativeForceSidecarConfig(data_dir="/data/force"),
)
```

The corresponding model config must use the native timestamp stream:

```python
Pi0_GuidanceConfig(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    force_encoder=ForceEncoderConfig(
        type="tcn",
        sampling_rate_hz=200,
        window_ms=100,
        history_source="timestamp_stream",
    ),
)
```

At 200 Hz with a 100 ms window, each 30 Hz VLA/RGB row retrieves the 20 wrench samples in
`(timestamp - 100 ms, timestamp]`. The sidecar is cached per episode, future samples are excluded,
and neighboring RGB rows may correctly use overlapping force windows.
