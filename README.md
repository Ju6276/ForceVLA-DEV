# Temporal ForceVLA Teacher

本分支基于原始 [ForceVLA](https://github.com/THUDaDa/ForceVLA)，首先把 instantaneous 6D wrench
前端替换为严格因果、按时间戳对齐的 temporal force encoder；在已经训练好的统一 Teacher 上，再为
Teacher-guided slow/fast distillation 准备 full/null targets、force-free Slow VLA 和轻量 Fast residual
student。

## 当前状态（2026-08-20）

当前选用的 Button Press 路线只有这一条：

```text
Stage 1  Temporal ForceVLA Teacher
                  ↓ freeze
Stage 2  同一 Teacher 增加 null-force 条件能力
                  ↓ freeze
Stage 3  离线提取 matched full/null targets
                  ↓
Stage 4  独立、无 Force 前端的 Slow nominal VLA
                  ↓ freeze
Stage 5  Intent Projector + Fast force residual student
```

| 项目 | 当前状态 | 本地产物 |
| --- | --- | --- |
| USB instantaneous baseline | 已训练 | step `39999` |
| USB 30 Hz Temporal Teacher | 已训练 | step `40000` |
| Button 100 Hz-force Temporal Teacher（Stage 1） | 已训练 | step `39999` |
| Button null-force adaptation（Stage 2） | 已训练 | step `9999` |
| Button matched target extraction（Stage 3） | 已完成 | train 30,675 rows / 56 episodes；val 6,429 rows / 10 episodes |
| Force-free Slow VLA（Stage 4） | 正式 10k 训练进行中，W&B online | step 5k / final 将保存 |
| Fast residual student（Stage 5） | 正式 cache、训练、checkpoint、validation 入口和端到端 smoke 已验证；等待 Slow 完成 | 无正式 checkpoint |

checkpoint、W&B 本地目录和二进制 targets 均被 Git 忽略，不会随代码推送。

已删除的旧方案包括：low-pass nominal target、固定力阈值/free-space loss、USB Stage 2A/2B、
`avg_pool/max_pool` force encoder、相关旧评估脚本和 checkpoints。当前代码只保留 instantaneous、
Temporal TCN、Button null-BC 和选定的 Slow/Fast 路线。

## Stage 1：Temporal ForceVLA Teacher

### 不变的 ForceVLA 主体

Stage 1 只替换 force front-end。以下部分保持原始 ForceVLA 设计：

- vision/language backbone；
- robot-state pathway；
- FVLMoE / force-aware fusion；
- Action Expert；
- action chunk 表示；
- flow-matching objective；
- 数据集原有的 RGB/state/action 行频率。

### 两种 force encoder

`ForceEncoderConfig.type` 只保留当前使用的两种模式：

- `instantaneous`：原始 ForceVLA，当前 6D wrench 经过原始 `6 → 2048` 投影；
- `tcn`：当前 temporal Teacher。

TCN 使用 FAVLA-style 宽度配置，但不声称逐层复现 FAVLA：

```text
input: B × N × 6
per-sample stem: 6 → 1024
4 causal residual TCN blocks: 1024 → 1024
dilations: 1, 2, 4, 8
aggregation: last causal grid state
output projection: 1024 → 2048
output: one 2048D force token
```

每个 temporal block 包含两层 `kernel_size=3` Conv1D、LayerNorm、SiLU、dropout 和残差连接。
Conv1D 只做左侧 padding，因此不会读取未来 force。最终仍只向原始 FVLMoE 接口提供一个 2048D
force token，窗口变长不会增加 token 数量。

示例：

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
        aggregation="last",
        history_source="timestamp_stream",
    ),
)
```

### 高频力的 timestamp alignment

RGB/state/action 与 force 保留各自的时间戳，不要求 frame 数一致。LeRobot dataset 仍以原始 VLA 行
索引；每个 episode 的独立高频力流存为 sidecar：

```text
force/episode_000000.npz
  force:      [M, 6]
  timestamps: [M]
```

Button named configs 默认从 `data/panda_button_press_100hz_causal/raw_sidecars` 读取；可以把该目录做成
指向实际数据位置的软链接，或在配置中改为自己的路径。`data/` 被 Git 忽略。

对每个 VLA timestamp `t_k`，代码只使用 `(t_k - window, t_k]` 中已到达的 force。固定采样率模式
生成因果时间网格，每个 grid point 使用不晚于该点的最新原始 sample（causal zero-order hold）；
过旧或缺失值填 0，并在 `force_history_mask` 中标为 invalid。

Button Press 当前使用 `100 Hz × 100 ms`：

```text
t_k - 90 ms, t_k - 80 ms, ..., t_k
            10 个 6D wrench slots
```

这里的 100 Hz 只表示一个 VLA row 内部的 force-history 分辨率。RGB、state 和 action label 仍保持
原数据频率，并没有被伪造为 100 Hz。相同物理窗口下：

```text
200 Hz × 100 ms → 20 × 6
300 Hz × 100 ms → 30 × 6
```

窗口长度是物理时间，不是写死的 frame 数；future sample 永远不会被选中。

### 30 Hz aligned compatibility mode

原始 ForceVLA 发布数据将一帧 wrench 放在 `observation.state`，与 RGB row 对齐。兼容实验使用：

```python
ForceEncoderConfig(
    type="tcn",
    sampling_rate_hz=30,
    window_ms=100,
    history_source="aligned_state",
)
```

该模式得到约 3 帧历史，但不能称为 native-rate 高频力。

## 当前配置

| config name | 用途 |
| --- | --- |
| `forcevla_lora` | 原始 instantaneous ForceVLA |
| `forcevla_temporal_lora_aligned` | 发布数据的 30 Hz temporal compatibility baseline |
| `forcevla_usb_lora` | USB instantaneous 40k baseline |
| `forcevla_usb_temporal_lora_aligned` | USB 30 Hz Temporal Teacher |
| `forcevla_button_temporal_100hz` | Button 100 Hz-force Temporal Teacher |
| `forcevla_button_temporal_100hz_val` | Button held-out episode loader |
| `forcevla_button_temporal_stage2_null_bc` | 当前选用的 Button Stage 2 |
| `forcevla_button_slow_lora` | Stage 4 Slow VLA train split |
| `forcevla_button_slow_lora_val` | Stage 4 Slow VLA held-out split/cache extraction |

这些 LoRA config 沿用 OpenPI/ForceVLA 的 freeze filter：Gemma 主权重冻结，LoRA 参数可训练；视觉编码器
和部分 robotics projections 仍可训练。因此它是仓库的 low-memory LoRA recipe，不是严格意义上的
“只训练 LoRA matrices”。

## Stage 2：同一 Teacher 的 null-force adaptation

当前只保留 `forcevla_button_temporal_stage2_null_bc`。它从 Stage 1 step `39999` 初始化：

- full-force 路径和全部 Stage 1 参数冻结；
- 新增一个 2048D learned `null_force_token`；
- 新增 rank-32、仅 null 条件启用的 Action-Expert adapter；
- null 路径仍使用原始 expert action 和原始 flow-matching objective；
- full 路径绕过 null token 和 null-only adapter，因此 Stage 1 full-force 函数按结构保留；
- 不使用 low-pass、固定 force threshold 或 free-space mask。

Stage 2 让同一个 Teacher 在缺少 force 时仍能输出动作。它本身不能保证 `A_full - A_null` 已经是
物理上最优或因果可识别的 residual；这个差值只称为 Teacher 的 force-induced deviation。

## Stage 3：matched full/null target extraction

冻结 Stage 2 Teacher，在同一样本、同一 flow noise 和同一采样设置下计算：

```text
A_full = Teacher(V, L, S, F_history)
A_null = Teacher(V, L, S, null_force)

A_nom_target = A_null
delta_A_target = A_full[..., :6] - A_null[..., :6]
```

Button targets 已完整保存到：

```text
artifacts/button_stage3_paired_targets/train
artifacts/button_stage3_paired_targets/val
```

targets 位于 ForceVLA normalized action space，并按 split-local dataset row 顺序保存。训练/验证以
episode 划分，互不重叠。`artifacts/` 已加入 `.gitignore`，465 MB 二进制 targets 不会上传 GitHub。

重新提取示例：

```bash
python scripts/extract_forcevla_paired_targets.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_stage2_null_bc \
    --checkpoint=checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999 \
    --output-dir=artifacts/button_stage3_paired_targets/train
```

## Stage 4：Slow nominal VLA

`forcevla_button_slow_lora` 是独立、无 Force 前端的标准 `Pi0Config`，不是带 null token 的 Teacher：

```text
Vision + Language → VLM context
Robot State(7D) → state/action pathway
                         ↓
                  A_ref action chunk
```

Slow 只读取 7D robot state（xyz、rpy、gripper），不读取 instantaneous wrench 或 force history。
训练 target 是 Stage 3 的 `normalized_null_actions`。它从 Stage 2 checkpoint 中加载结构兼容的 VLA/
Action weights，但部署模型本身没有 ForceVLA force front-end、TCN 或 null token。

正式 10k 训练命令：

```bash
unset WANDB_MODE
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_slow_lora \
    --exp-name=button_press_slow_null_distill \
    --overwrite
```

配置：batch size 4、10k steps、500-step warmup、peak LR `2.5e-5`、final LR `2.5e-6`，
step 5k 和 final 保存；W&B project 为 `forcevla`。

## Stage 5：Fast residual student

当前选定结构为：

```text
Force history: 10 × 6 @ 100 Hz × 100 ms
        ↓
Causal TCN: 6 → 1024, 4 × 1024 blocks, dilation 1/2/4/8
        ↓
1 × 1024D force token
                                ┐
Latest robot state: 7D ────────→ state token
Cached Slow V-L context ───────→ 2 intent tokens
Current interpolated A_ref: 7D ─→ reference token
[chunk phase, context age] ─────→ time token
Learned residual query ─────────→ query token
                                ┘
                  1 × Gemma-style decoder layer
                                ↓
                     single-step 6D delta_A
```

Fast 的 7D 输入是最新 robot state，不是 force。force 始终通过独立的 `10 × 6` causal TCN 路径进入。
Fast TCN 不做 Teacher 的 `1024 → 2048` 输出投影，因为 Fast decoder 的 width 本身就是 1024。

```text
pi_fast(F[t-100ms:t], S_t, Z_intent, A_ref(t), phase/age) → delta_A[:6]

A_cmd[:6] = A_ref[:6] + gate * delta_A
A_cmd[6]  = A_ref[6]       # gripper 由 Slow 负责
```

已实现：

- `src/openpi/models/slow_fast.py`：Intent Projector、Teacher-style TCN、1-layer Gemma-style
  single-step residual expert；
- `src/openpi/training/slow_fast_distillation.py`：paired targets、Slow loss、residual + reconstruction loss；
- `src/openpi/serving/slow_fast_runtime.py`：原子 Slow packet cache、按 timestamp 插值 `A_ref`、
  phase/age 和 `A_ref + delta_A` composition；
- `scripts/extract_slow_cache.py`：以 10 Hz 因果采样 Slow，缓存预测 chunk 与压缩后的 V-L context，
  并为每个数据时间戳构造插值后的 `A_ref`；
- `scripts/train_fast_residual.py`：正式 Fast optimizer、W&B、5k/final checkpoint 和 held-out validation。

Fast 主实验只优化 `MSE(delta_A_fast, A_full-A_null)`。`A_ref+delta_A_fast` 对 `A_full` 的
reconstruction MSE 只作为 validation metric；默认不把它加入训练 loss，避免 Fast 学习与 force 无关的
Slow prediction/interpolation error。脚本保留 `--reconstruction-weight` 作为显式 ablation，默认值为 0。

Slow 完成后，先提取 train/val cache：

```bash
python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_slow_lora \
    --stage3-dir=artifacts/button_stage3_paired_targets/train \
    --checkpoint=checkpoints/forcevla_button_slow_lora/button_press_slow_null_distill/9999 \
    --output=artifacts/button_slow_cache/train.npz

python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_slow_lora_val \
    --stage3-dir=artifacts/button_stage3_paired_targets/val \
    --checkpoint=checkpoints/forcevla_button_slow_lora/button_press_slow_null_distill/9999 \
    --output=artifacts/button_slow_cache/val.npz
```

然后正式训练 Fast：

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_fast_residual.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --output-dir=checkpoints/button_press_fast_residual \
    --steps=10000 --batch-size=64
```

仍未完成的是：真实机器人上的异步 slow/fast control loop。离线训练与 held-out evaluation 路径已经接通。

## 安装

仓库在 Ubuntu 22.04、Python 3.11 和 NVIDIA CUDA 环境下开发：

```bash
git clone --recurse-submodules <repo-url>
cd ForceVLA-DEV

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e lerobot
pip install -e packages/openpi-client
```

原始 ForceVLA 数据：<https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data>

## Stage 1 训练示例

```bash
export HF_LEROBOT_HOME=/path/to/lerobot/root

python scripts/compute_norm_stats.py --config-name forcevla_usb_lora
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_usb_lora \
    --exp-name=forcevla_usb_instantaneous_40k \
    --overwrite \
    --batch_size=4
```

USB temporal compatibility baseline：

```bash
python scripts/compute_norm_stats.py --config-name forcevla_usb_temporal_lora_aligned
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_usb_temporal_lora_aligned \
    --exp-name=forcevla_usb_temporal \
    --overwrite \
    --batch_size=4
```

`forcevla_lora`/`forcevla_usb_lora` 始终是 instantaneous baseline；训练 history model 时必须显式选择
temporal config。所有本分支 ForceVLA config 的 W&B project 均为 `forcevla`。

## 当前执行顺序

1. 完成 Stage 4 Slow 10k；
2. 提取 train/held-out Slow packets，并报告 `A_ref` 对 `A_null` 的误差；
3. 完成 Stage 5 Fast 10k；
4. 比较 held-out residual MSE 与 zero-residual baseline，并检查 `A_ref + delta_A` 对 `A_full` 的重建误差；
5. 离线结果通过后，再实现真实机器异步执行。

当前仓库不宣称 Temporal 一定优于 Instantaneous，也不宣称 non-zero `A_full - A_null` 已经证明了
有效 slow/fast 控制分解；最终结论必须来自 held-out 和在线机器人实验。
