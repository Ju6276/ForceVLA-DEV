# Temporal ForceVLA

基于 [ForceVLA](https://github.com/THUDaDa/ForceVLA) 的分支。两处改动：把 instantaneous 6D wrench 前端
换成严格因果、按时间戳对齐的 temporal force encoder；在冻结的 Teacher 之上训练一个轻量 Fast residual
student，由 Teacher 的 null-force 路径充当 Slow 参考。

```text
Stage 1  Temporal ForceVLA Teacher
              ↓ freeze
Stage 2  同一 Teacher 增加 null-force 条件能力
              ↓ freeze
Stage 3  离线提取 matched full/null targets
              ↓  （从 Teacher 的 null 路径提取 Slow cache，不另训模型）
Stage 4  Intent Projector + Fast force residual student
```

## 状态

**没有任何已训练产物。** checkpoint、norm stats、Stage-3 targets、Slow cache 全部已删除，USB 与 Button
两条线都要从 Stage 1 重跑。末端姿态改为 6D 连续旋转后归一化统计必须重算，旧 checkpoint 的输入输出布局
也不再兼容，保留它们只会误导。

Slow/Fast 这条线在 2026-08-21 改过 6D 旋转、时序随机化、K 步 chunk 和部署契约。这些改动只有单元测试和
合成数据 smoke 验证过，**没有跑过真实训练或真机**。

`assets/`、`checkpoints/`、`artifacts/`、`wandb/`、`data/` 均被 Git 忽略，仓库里只有代码。

## 安装

Ubuntu 22.04 + Python 3.11 + CUDA：

```bash
git clone --recurse-submodules <repo-url>
cd ForceVLA-DEV
git submodule update --init lerobot dlimp   # 忘了 --recurse-submodules 时补这一步
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e . && pip install -e lerobot && pip install -e packages/openpi-client
```

`lerobot` 与 `dlimp` 是 submodule，pin 在 `.gitmodules`；`flaxformer` 直接 vendored 在仓库里，不用装，但
要加进 `PYTHONPATH`。`third_party/aloha` 和 `third_party/libero` 只有跑那两个 benchmark 才需要。

跑测试：

```bash
PYTHONPATH=$PWD/src:$PWD/flaxformer python -m pytest src/openpi -q
PYTHONPATH=$PWD/src:$PWD/flaxformer python scripts/smoke_fast_pipeline.py
```

## 数据布局

RGB/state/action 走 LeRobot dataset，高频力单独存 sidecar，两者各自保留时间戳，不要求帧数一致：

```text
data/panda_button_press_100hz_causal/raw_sidecars/episode_000000.npz
    force:      [M, 6]
    timestamps: [M]
```

数据集本身不在仓库里，需要单独拿。LeRobot 按 `repo_id` 在 `~/.cache/huggingface/lerobot` 下找它，缺失
时会尝试联网下载并失败，所以要软链到本地副本：

```bash
ln -sfn <数据集路径> ~/.cache/huggingface/lerobot/panda_button_press
```

Button 数据集是 66 个 episode、37104 帧，编号 0–65 连续，力采样率全部 ≥100 Hz。`config.py` 里
`_BUTTON_PRESS_TRAIN_EPISODES`（56 条）和 `_BUTTON_PRESS_VAL_EPISODES`（10 条）的索引就是按这个编号写
死的，两个 split 在力采样率和按钮位置上都匹配（均值 196 Hz vs 196 Hz）。

要换力采样率门槛就用 `scripts/filter_dataset_by_force_rate.py`（`--dry-run` 只看选中结果）。它按整段平均
速率 `(n-1)/(t_last-t_first)` 筛选，而不是中位间隔——中位数在丢包的流上仍然虚高。它会重排 episode 编号和
dataset 级 frame index，所以**必须同步重映射上面那两串索引**，否则 train/val 会静默错位；也不要按位置切
（取末尾若干条），因为采集场次不同，尾部的力率和按钮位置都有系统偏移。

## Button 完整重跑流程

### 1. norm stats

必须用当前 6D 代码重算。`compute_norm_stats.py` 写到 `assets/<config_name>/<asset_id>`，Stage 1 和
Stage 2 各一份；val config 显式指向 Stage 1 的目录，不需要自己的。**不要**对 `..._val` 跑这个脚本
——那会用 held-out episode 算统计量。

```bash
python scripts/compute_norm_stats.py --config-name forcevla_button_temporal_100hz

SRC=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56
mkdir -p assets/forcevla_button_temporal_stage2_null_bc
cp -r "$SRC" assets/forcevla_button_temporal_stage2_null_bc/
```

### 2. Stage 1：Temporal Teacher

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_temporal_100hz \
    --exp-name=button_press_temporal_100hz --overwrite --batch_size=4
```

### 3. Stage 2：null-force adaptation

从 Stage 1 初始化，冻结 full 路径，只训一个 2048D `null_force_token` 和 rank-32 null-only adapter。

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_temporal_stage2_null_bc \
    --exp-name=button_press_stage2_null_bc --overwrite --batch_size=4
```

### 4. Stage 3：matched full/null targets

冻结 Stage 2 Teacher，同一样本、同一 flow noise 下算 `A_full` 与 `A_null`，
`delta_A_target = A_full[..., :9] - A_null[..., :9]`。

```bash
CKPT=checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999

python scripts/extract_forcevla_paired_targets.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_stage2_null_bc \
    --checkpoint=$CKPT --output-dir=artifacts/button_stage3_paired_targets/train

python scripts/extract_forcevla_paired_targets.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_100hz_val \
    --checkpoint=$CKPT --output-dir=artifacts/button_stage3_paired_targets/val
```

**先看这一步的 summary 再往下走。** `full_vs_expert` 与 `null_vs_expert` 的相对差就是力条件带来的全部
收益，也是 Fast 的信号上限。这个差只有个位数百分比的话，后面 Stage 4 能拿到的收益同样有限，值得先停下
来查为什么力条件没起作用。

### 5. Slow cache

Slow 就是 Stage-2 Teacher 的 null 路径，不另训模型。只要 `--seed` 与 Stage 3 一致，两边 flow noise 都由
`row_keyed_noise` 按 dataset row 生成，`A_ref` 与 `A_null` 在每个 Slow 观测行上逐位相同，剩下的参考误差
只有更新间隔内的陈旧度。

**train 必须 `--slow-rate-hz 0` 提全速率**，训练时才能重抽时序；val 保留一个固定实现作为不参与训练的
时序。

```bash
python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_stage2_null_bc \
    --stage3-dir=artifacts/button_stage3_paired_targets/train \
    --checkpoint=$CKPT --slow-rate-hz 0 \
    --output=artifacts/button_slow_cache/train.npz

python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_100hz_val \
    --stage3-dir=artifacts/button_stage3_paired_targets/val \
    --checkpoint=$CKPT \
    --output=artifacts/button_slow_cache/val.npz
```

检查 val summary 的 `time_features`：`saturated_age_fraction` 应为 0，`alpha_interior_fraction` 应明显
大于 0，`age_alpha_correlation` 不应接近 ±1。任一项不满足说明时间 token 退化了。

### 6. Stage 4：训练 Fast residual

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_fast_residual.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --output-dir=checkpoints/button_press_fast_residual \
    --steps=10000 --batch-size=64 \
    --chunk-steps=5 --timing-resample-interval=500
```

W&B 的 `timing/*` 记录每次重抽后的 `ready_row_fraction`、`mean_normalized_age` 和
`saturated_age_fraction`；后者不为 0 说明 `--context-age-scale-ms` 太小。训练用的时序带写进
`metadata.json` 的 `trained_timing`，部署时要读它。

### 7. held-out 评估

```bash
python scripts/evaluate_fast_residual.py \
    --targets=artifacts/button_stage3_paired_targets/val \
    --slow-cache=artifacts/button_slow_cache/val.npz \
    --checkpoint=checkpoints/button_press_fast_residual/step-10000/params \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --contact-threshold-n=5.0 \
    --output=artifacts/button_fast_evaluation/val.json
```

同一条命令也会跑 force 消融（force history 置零 / 打乱），用来确认 Fast 不是只靠 Slow
context/state/reference 猜 residual。省略 `--norm-stats-dir` 退化为 normalized-space、不分层的报告。

重点看三项：

- `deployment_error_decomposition`：`(A_ref + delta) - A_full = (A_ref - A_null) + (delta - (A_full - A_null))`
  精确拆成一个与力无关的 Slow 项和唯一被训练的 Fast 项。两项都含 `A_null`，**不能相加**，只比量级。
  如果 Slow 项比整个力修正信号还大，组合改善就会被分母锁死，此时该去降 Slow 误差而不是调 Fast。
- `per_step_residual_mse_normalized`：chunk 后几步是否也学到了。
- 分平移 / 测地线旋转的物理误差。各模型的旋转误差是否拉开差距，是 6D 改造是否奏效的判据。

## 设计要点

### 6D 连续旋转

数据集用欧拉角记录末端姿态，而 Button Press 静止在 roll ≈ ±π，正好是分支切点：同一物理姿态被随机记成
`+3.1416` 或 `-3.1416`。后果是 `DeltaActions` 逐维相减产生 ±2π 的假 delta——train 集实测 **5.37% 的样本**
如此，roll 的 delta std 因而是 1.451（pitch/yaw 只有 0.013 量级，即虚高约 108 倍）。z-score 之后 roll 的
真实变化被压到百分之一量级，Teacher/Slow/Fast 三级都几乎没学过它。改成 6D 后最大 |delta| 从 6.28 降到
0.058，2π 跳变消失。

`forcevla_policy.py` 现在把 xyz+rpy 改写成 xyz+6D（Zhou et al.，旋转矩阵前两列），模型输入 16D state /
10D action，输出再转回 rpy 给机器人。磁盘上的 parquet 仍是欧拉角，公共 `transforms.DeltaActions` 不打
补丁。核对轨迹连续性：

```bash
python scripts/convert_rpy_to_6d.py --input-root <dataset> --check-only
```

### Temporal force encoder

```text
input [B, N, 6] → 6→1024 stem → 4 个因果残差 TCN block (dilation 1/2/4/8) → 取最后一个因果格 → 1024→2048
```

Conv1D 只做左侧 padding，永远不会读到未来的力。窗口是物理时间不是帧数：对每个 VLA timestamp `t_k`
只用 `(t_k - window, t_k]` 内已到达的力，按因果零阶保持重采样到固定网格，过旧或缺失的格子填 0 并在
`force_history_mask` 里标为 invalid。Button 用 `100 Hz × 100 ms` = 10 个 6D slot。换力传感器频率不需要
改代码。

`ForceEncoderConfig.type` 只保留 `instantaneous`（原始 ForceVLA）和 `tcn`（当前 Teacher）。

### Fast residual student

```text
Force history 10×6 @100 Hz ─→ causal TCN ─→ 1 force token
最新 robot state 10D       ─→ state token
缓存的 Slow V-L context    ─→ 2 intent tokens          ┐
当前插值 A_ref 10D         ─→ reference token          ├→ 1 层 Gemma-style decoder
[context age, interp alpha]─→ time token               │
learned residual query     ─→ query token              ┘
                                    ↓
                    5 × xyz+6D residual，间隔一个 action period
```

```text
A_cmd[k, :9] = A_ref(t + k·action_period)[:9] + delta_A[k]
A_cmd[k, 9]  = A_ref(t + k·action_period)[9]      # gripper 由 Slow 独占
```

**输出短 chunk 而不是单步**，所以 Fast 的输出速率与调用速率解耦：跟得上就只执行 `k=0`，落后了就往后
多消费几步。后面几步只在落后时才执行，因此 loss 按 `--step-decay 0.5` 几何降权。

**两个时间特征必须互不共线。** 旧的 `[phase, age]` 是同一个 age 除以两个常数，实测相关系数 1.0，等于
只有一维。现在是 `[age/scale, alpha]`，alpha 是两个 Teacher waypoint 之间的插值位置。

**Fast 不接图像。** `sample_paired_actions_and_context` 里 full 和 null 共用同一份 `prefix_out_fix` 和
同一个 flow noise，所以 `A_full − A_null` 里视觉的贡献精确抵消，标签是力条件的纯函数。喂新鲜图像无法
降低这个目标的损失。要让 Fast 应对 Slow 更新之间的场景变化，得改目标定义，不是加输入。

主实验只优化 `MSE(delta_A_fast, A_full - A_null)`；`--reconstruction-weight` 默认 0，避免 Fast 去学与力
无关的 Slow 预测/插值误差。`predict_gate=false`，gate 接口保留但恒为 1。

### Slow 时序按区间随机化

按单一实测值提取的 cache 只描述那一台机器。提取时对每个更新间隔和每个 packet 独立抽样：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--slow-rate-hz 5 15` | 5–15 Hz | 一份 cache 覆盖一段频率带 |
| `--slow-latency-ms 50 300` | 50–300 ms | packet 在 `t_key + latency` 才可用 |
| `--update-jitter-ms 10` | 10 ms | 把更新时刻移出 30 Hz 网格；不抖动则 `alpha` 恒为 0 |
| `--context-age-scale-ms 600` | 600 ms | 必须覆盖整个「延迟 + 周期」带，否则 age 截断到 1.0 |

`--action-rate-hz` 是唯一**不**随机化的时间量：它是 Teacher chunk 的步距，由训练数据决定，不是部署旋钮。
训练时默认每 500 步用 `resample_slow_cache` 重画一次时序，所以 train cache 必须是全速率提取的。

### 部署

**真机不跑 `extract_slow_cache.py`，也不加载那个 npz。** 离线 npz 是训练产物；真机上 `SlowWorker` 在
自己的线程里实时跑 Teacher 的 null 路径，把结果塞进运行时的 `SlowReferenceCache`（单槽双缓冲）。离线
cache 存在的唯一理由，是让 Fast 训练时看到的 packet 结构和运行时一致。

`slow_fast_deploy` 固化了两边必须一致的东西：动作截到 `ROBOT_DIMS`、prefix pool 成相同 bin 数、state
必须先转成 6D。

```python
from openpi.serving import slow_fast_deploy, slow_fast_loop, slow_fast_runtime

contract = slow_fast_deploy.load_contract(
    "artifacts/button_slow_cache/train.npz",
    fast_run="checkpoints/button_press_fast_residual",  # 时序带来自训练 run，不是 cache
)
build_packet = slow_fast_deploy.SlowPacketBuilder(contract)

def observe():
    timestamp, raw_state, images = read_robot()
    return timestamp, images, slow_fast_deploy.convert_robot_state(raw_state)

def infer(observation):
    return build_packet(*teacher.sample_nominal_actions_and_context(rng, observation))

worker = slow_fast_loop.SlowWorker(
    observe=observe, infer=infer, cache=slow_fast_runtime.SlowReferenceCache(),
    config=contract.slow_fast_config(residual_limit=0.02),
    context_age_scale_s=contract.context_age_scale_s, period_s=0.1,
)
```

形状从 cache 的 summary 读，**时序带必须从 Fast 训练 run 的 `metadata.json` 读**——推荐的 train cache 是
全速率提取的，它自己记录的带是退化的 `[0, 0]`。上机前用
`contract.check_measured_timing(slow_rate_hz=..., slow_latency_s=...)` 核对实测时序；超出带外要加宽带、
同步加大 `--context-age-scale-ms`，重提 cache 并重训 Fast（Stage 1–3 不受影响）。

还需要：安全门控、pose residual 限幅、确认 gripper 由 Slow 独占。

## 配置

| config name | 用途 |
| --- | --- |
| `forcevla_lora` | 原始 instantaneous ForceVLA |
| `forcevla_temporal_lora_aligned` | 发布数据的 30 Hz temporal compatibility baseline |
| `forcevla_usb_lora` | USB instantaneous 40k baseline |
| `forcevla_usb_temporal_lora_aligned` | USB 30 Hz Temporal Teacher |
| `forcevla_button_temporal_100hz` | Button Stage 1，100 Hz 力 |
| `forcevla_button_temporal_100hz_val` | Button held-out loader |
| `forcevla_button_temporal_stage2_null_bc` | Button Stage 2 |

`forcevla_lora` / `forcevla_usb_lora` 始终是 instantaneous baseline，训练 temporal 模型必须显式选 temporal
config。这些 LoRA config 沿用 OpenPI/ForceVLA 的 freeze filter：Gemma 主权重冻结，LoRA 参数、视觉编码器
和部分 robotics projection 可训练——是 low-memory recipe，不是严格的「只训 LoRA」。所有 config 的 W&B
project 均为 `forcevla`。

## 尚未证明的事

本仓库不宣称 Temporal 优于 Instantaneous，也不宣称非零的 `A_full - A_null` 已经证明了有效的 slow/fast
控制分解。Slow/Fast 这条线目前只有单元测试和合成数据 smoke，没有真实训练结果。结论必须来自 held-out
与在线机器人实验。
