# Temporal ForceVLA Teacher

本分支基于原始 [ForceVLA](https://github.com/THUDaDa/ForceVLA)，首先把 instantaneous 6D wrench
前端替换为严格因果、按时间戳对齐的 temporal force encoder；在已经训练好的统一 Teacher 上，再为
Teacher-guided slow/fast distillation 准备 full/null targets、force-free Slow VLA 和轻量 Fast residual
student。

## 当前状态（2026-08-21）

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
| Button Stage 1–5 | **全部产物已删除，待重跑** | 无 |

Button 的 Stage 1–5 checkpoint、Stage-3 targets、Slow cache 和评估结果已于 2026-08-20 全部删除。
原因是发现了 rpy delta 缺少角度折叠的数据缺陷（见下文），归一化把 roll 的尺度放大了 196 倍，
导致此前所有 Button 训练都没有真正学习 roll。代码和 norm stats 已修复，需要从 Stage 1 重跑。

当前没有训练进程。checkpoint、W&B 本地目录和二进制 targets 均被 Git 忽略，不会随代码推送。

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

正式 Slow W&B run：<https://wandb.ai/ju-dong6276-technical-university-of-munich/forcevla/runs/1iactyn8>。
10 Hz cache 中，interpolated `A_ref` 对 Stage-3 `A_null` 的 normalized pose MSE 为：train `0.03421`，
held-out `0.03127`。

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

A_cmd[:6] = A_ref[:6] + delta_A
A_cmd[6]  = A_ref[6]       # gripper 由 Slow 负责
```

正式训练配置 `predict_gate=false`，因此没有学习 gate；代码中的 gate 接口仅作为可选扩展保留，当前值恒为 1。

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

正式 Fast W&B run：<https://wandb.ai/ju-dong6276-technical-university-of-munich/forcevla/runs/dnivkphb>。

全 held-out 6,429 rows 的结果：

| 条件 | normalized residual MSE |
| --- | ---: |
| zero-residual baseline | `0.005374` |
| Fast，正确 force history | `0.000611` |
| Fast，force history 全零 | `0.009372` |
| Fast，force history 随机打乱 | `0.010801` |

正确 force 相对 zero-residual baseline 降低 `88.63%`；置零或打乱 force 后误差分别是正确 force 的
`15.34×` 和 `17.67×`，说明 Fast 不是只依赖 Slow context/state/reference 猜 residual。组合指标中，
`A_ref + delta_A_fast` 对 `A_full` 的 MSE 为 `0.03058`，仅 `A_ref` 为 `0.03526`，改善 `13.26%`；
当前最终控制误差仍主要受 Slow reference error 限制。

### 对 ground-truth expert 的物理单位评估

以上都以 Teacher 的 `A_full` 为参考。`scripts/evaluate_fast_residual.py` 现在同时报告所有变体对
**真实 expert action** 的误差，单位为米和弧度，并按接触状态分层。接触判据是相对 episode 静止 wrench
的力偏差，而不是绝对幅值——原始 wrench 含约 5.5 N 的工具重力/传感器偏置，绝对阈值会把自由空间
误判为接触。旋转维度的差值按 `(-pi, pi]` 折叠。

held-out 6,429 rows（接触 2,446 / 自由空间 3,983，阈值 5 N）：

| 分层 | 变体 | translation RMSE (m) | rotation RMSE (rad) |
| --- | --- | ---: | ---: |
| 接触 | Slow `A_ref` only | `0.02365` | `0.02680` |
| 接触 | Slow + Fast residual | `0.02295` | `0.02874` |
| 接触 | Teacher `A_null` | `0.02290` | `0.02233` |
| 接触 | Teacher `A_full` | `0.02186` | `0.02149` |
| 自由空间 | Slow `A_ref` only | `0.01164` | `0.02649` |
| 自由空间 | Slow + Fast residual | `0.01110` | `0.02698` |

结论分两部分：

- **平移方向 Fast 是正收益。** 接触段 translation RMSE 从 `23.65 mm` 降到 `22.95 mm`（`+2.94%`），
  自由空间 `+4.68%`。Teacher 自身的 force gain 在接触段为 translation `+4.55%`，所以 Fast 复现了
  Teacher 力增益的约三分之二。
- **旋转方向所有模型都没有学到。** roll RMSE 在 Slow、Teacher-null、Teacher-full、Fast 之间几乎不变
  （`0.021`–`0.031` rad），Fast 甚至让接触段 roll 变差 `7.22%`。原因见下节的 roll wrapping 问题：
  roll 在 normalized loss 中的权重被压低了约两个数量级，训练信号几乎为零。

因为 roll 的平方误差与 z 方向相当，未加权的总 MSE 被这个未训练的维度主导，`fast vs slow-only` 的
总 MSE 增益因此是负的（接触段 `-5.88%`）。在 roll 修好之前，总 MSE 不是有意义的模型选择指标；
应当看分维度或分平移/旋转的数字。

### 已知未修复的数据缺陷：欧拉角落在分支切点上

上表中 rotation 的数字受一个已定位但**尚未修复**的缺陷影响。数据集的末端姿态用欧拉角记录
（`observation.state` 的字段名即 `x, y, z, roll, pitch, yaw, gripper_width, force_*`），而 Button Press
的末端静止在 roll ≈ ±π，正好落在欧拉角的分支切点上。同一个物理姿态因此被随机记成 `+3.1416`
或 `-3.1416`。全量扫描的结果：

| 数据集 | 帧数 | roll | pitch | yaw |
| --- | ---: | --- | --- | --- |
| Button Press（100 ep） | 56910 | 96.1% 在 +π 侧、3.9% 在 −π 侧，34 次相邻帧 2π 跳变 | 干净 | 干净 |
| USB Insert（50 ep） | 34478 | 4 次 2π 跳变 | 最大值离 π 仅 0.12° | 4 次 2π 跳变 |

这带来两个后果。一是 `transforms.DeltaActions` 逐维相减构造 delta action 时，这些帧产生 ±2π 的假
delta：

```text
action roll = -3.1415, state roll = +3.1297
naive delta = -6.2712      # 整整一圈
真实姿态变化 = +0.0120
```

二是 norm stats 被污染。Button 的 state roll 以圆均值重新绕分支后真实 std 只有 `0.0067`，而记录值
是 `1.2461`，**虚高 187 倍**。z-score 之后 roll 的真实变化被压缩到应有尺度的 1/187，这影响全部
56910 帧而不只是那 34 次跳变，等于 Teacher / Slow / Fast 三级都几乎没有学过 roll。

上游 pi0 不会踩到这个问题：ALOHA 和 DROID 用关节空间，Libero 用 axis-angle，这些表示下直接相减都
是安全的。ForceVLA 是唯一对**绝对欧拉角**做减法的配置。为了与上游 ForceVLA / pi0 的管线保持一致，
仓库目前**不打补丁**，保留原始行为。

**所有 rotation 相关的既有 checkpoint 和评估结论都不成立**，上面那张 held-out 表格保留在此仅作
基线参考。修复需要改变旋转表示（例如把姿态表达为相对某个标定参考系的偏差，或改用 6D 连续表示），
并重算 norm stats、重跑 Stage 1–5。

运行方式：

```bash
python scripts/evaluate_fast_residual.py \
    --targets=artifacts/button_stage3_paired_targets/val \
    --slow-cache=artifacts/button_slow_cache/val.npz \
    --checkpoint=checkpoints/button_press_fast_residual/step-10000/params \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --contact-threshold-n=5.0 \
    --output=artifacts/button_fast_evaluation/val_expert_and_contact.json
```

省略 `--norm-stats-dir` 时退化为原来的 normalized-space、不分层报告。

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

重新加载 final checkpoint 并运行 force 消融：

```bash
python scripts/evaluate_fast_residual.py \
    --targets=artifacts/button_stage3_paired_targets/val \
    --slow-cache=artifacts/button_slow_cache/val.npz \
    --checkpoint=checkpoints/button_press_fast_residual/step-10000/params \
    --output=artifacts/button_fast_evaluation/val_force_ablation.json
```

仍未完成的是：真实机器人上的异步 slow/fast control loop。离线训练与 held-out evaluation 路径已经接通。

另外，Stage 3 的 summary 显示 Teacher 的力条件对拟合 expert action 的帮助有限：normalized 全维
`full_vs_expert` 为 train `0.03361` / val `0.04381`，`null_vs_expert` 为 train `0.03536` / val `0.04704`，
相对改善只有 4.9% / 6.9%。考虑到 Stage 2 的 null 路径只有 1 个 token 加 rank-32 adapter 可训练，
这个差距里还混着容量不对称的成分。在 roll wrapping 修好并重新评估之前，不应把 `A_full - A_null`
当作已被验证的力修正量。

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

Stage 4/5 离线训练和 held-out force ablation 已完成。下一步按优先级是：

1. 修复 roll 的 2π wrapping，重算 norm stats，重跑 Stage 1–5。在此之前旋转维度的所有结论无效；
2. 单独降低 Slow `A_ref` 对 `A_null` 的误差，不让 Fast 补偿 force-agnostic Slow error。先用不同
   `--slow-rate-hz` 重跑 cache 提取，把误差分解为 Slow 模型误差与 chunk 插值/外推误差；
3. 对齐训练与部署的时间语义：`extract_slow_cache.py` 的 `--context-age-scale-ms` 默认 100 ms，
   而 `slow_fast_runtime.reference_time_features` 默认 250 ms，两者必须一致；训练数据还应注入
   真实的 Slow 推理延迟，当前允许 context age 为 0，这在真机上不可实现；
4. 按 timestamp 实现真实机器 10 Hz Slow / 高频 Fast 的异步执行；
5. 在线测试安全门控、pose residual 限幅、gripper 由 Slow 独占，以及接触事件指标。

当前仓库不宣称 Temporal 一定优于 Instantaneous，也不宣称 non-zero `A_full - A_null` 已经证明了
有效 slow/fast 控制分解；最终结论必须来自 held-out 和在线机器人实验。
