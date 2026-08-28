# Teacher-Guided Slow–Fast ForceVLA

基于 [ForceVLA](https://github.com/THUDaDa/ForceVLA) 的分支。当前主线是在统一 force-aware Teacher 上构造
matched full/null targets，再训练一个轻量 Fast residual student；Teacher 的 null-force 路径直接充当 Slow，
不另训 Slow Student。Temporal force encoder 是 Teacher 的前端组件，不作为论文的独立创新点。

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

Button 的 Stage 1 Teacher（40k）、Stage 2（10k）、Stage-3 targets、Slow cache 和 Fast Student（10k）
已经完成一次真实离线流程；尚未做真机验证。instantaneous/native-instantaneous 仅保留为可选的 Teacher
前端 sanity ablation，不是运行主流程的前置条件。旧的 Euler-angle checkpoint 不兼容当前 6D 连续旋转布局，
不应复用。

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

# 每个新终端都执行；后续命令默认从仓库根目录运行。
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/flaxformer${PYTHONPATH:+:$PYTHONPATH}"
```

`lerobot` 与 `dlimp` 是 submodule，pin 在 `.gitmodules`；`flaxformer` 直接 vendored 在仓库里，不用装，但
要加进 `PYTHONPATH`。`third_party/aloha` 和 `third_party/libero` 只有跑那两个 benchmark 才需要。

跑测试：

```bash
python -m pytest -q
python scripts/smoke_fast_pipeline.py
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

只有运行可选 instantaneous 消融时，才需要额外执行：

```bash
python scripts/compute_norm_stats.py --config-name forcevla_button_instantaneous
```

`forcevla_button_native_instantaneous_100hz` 故意直接复用 Temporal 的 train56 norm stats，保证它和 TCN
看到的 sidecar wrench 使用完全相同的归一化；不要为它单独重算统计量。

### 2. Stage 1：Unified Force-Aware Teacher

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_temporal_100hz \
    --exp-name=button_press_temporal_100hz --overwrite --batch-size=4
```

`--overwrite` 会重建同名实验目录；要继续已有训练应改用 `--resume`，两者不要同时传。

### 2b. 可选 Teacher 前端消融：instantaneous ForceVLA

这是原 ForceVLA 的当前 6D wrench 前端 `state[10:16] → Linear(6, 2048)`，不读取高频 sidecar。除 force
encoder 外，它与上面的 Temporal Teacher 使用相同 56/10 split、Pi0 base 权重、6D action、LoRA/freeze
filter、batch size、40k steps 和 20k 保存间隔。

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_instantaneous \
    --exp-name=button_press_instantaneous_40k --overwrite --batch-size=4
```

### 2c. 可选同源消融：native instantaneous（已配置，按需启动）

这个配置与 Temporal 使用同一份100 Hz timestamp sidecar、同一个 `(t-100 ms, t]` 的 `10×6` 因果窗口、
12 ms freshness mask 和同一份 train56 norm stats，但只读取窗口最后的当前槽位：

```text
latest causal 6D wrench at t -> original force_in_proj Linear(6, 2048) -> one force token
```

其余9个历史槽位不进入网络，也不创建 TCN。这样 `native_instantaneous` 与 Temporal TCN 的差异只剩
“当前单点”还是“100 ms历史编码”。以下命令只供准备好后启动，目前不需要运行：

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_native_instantaneous_100hz \
    --exp-name=button_press_native_instantaneous_100hz_40k --overwrite --batch-size=4
```

训练结束后的同口径评估命令是：

```bash
python scripts/evaluate_forcevla_checkpoint.py \
    --config-name=forcevla_button_native_instantaneous_100hz \
    --data-config-name=forcevla_button_native_instantaneous_100hz_val \
    --checkpoint=checkpoints/forcevla_button_native_instantaneous_100hz/button_press_native_instantaneous_100hz_40k/39999 \
    --output-dir=artifacts/button_stage1_evaluation/native_instantaneous \
    --contact-sidecar-dir=data/panda_button_press_100hz_causal/raw_sidecars
```

三组结果应按下面的顺序解释：

- 原版30 Hz instantaneous vs native instantaneous：力数据源、采样与时间戳对齐的影响。
- native instantaneous vs Temporal TCN：在相同高频数据源下，历史编码本身的影响。

训练完成后，用同一验证 split 和按 dataset row 固定的 flow noise 分别评估，避免两次采样噪声不同：

```bash
python scripts/evaluate_forcevla_checkpoint.py \
    --config-name=forcevla_button_instantaneous \
    --data-config-name=forcevla_button_instantaneous_val \
    --checkpoint=checkpoints/forcevla_button_instantaneous/button_press_instantaneous_40k/39999 \
    --output-dir=artifacts/button_stage1_evaluation/instantaneous \
    --contact-sidecar-dir=data/panda_button_press_100hz_causal/raw_sidecars

python scripts/evaluate_forcevla_checkpoint.py \
    --config-name=forcevla_button_temporal_100hz \
    --data-config-name=forcevla_button_temporal_100hz_val \
    --checkpoint=checkpoints/forcevla_button_temporal_100hz/button_press_temporal_100hz/39999 \
    --output-dir=artifacts/button_stage1_evaluation/temporal \
    --contact-sidecar-dir=data/panda_button_press_100hz_causal/raw_sidecars
```

评估器同时报告 normalized action MSE、物理平移 RMSE（米）、SO(3) 测地线旋转 RMSE（弧度）和 gripper
误差，并按同一高频 sidecar 划分 contact / free-space。它不把米、弧度和 6D 坐标混成一个 pose MSE。

### 3. Stage 2：null-force adaptation

从 Stage 1 初始化，冻结 full 路径，只训一个 2048D `null_force_token` 和 rank-32 null-only adapter。

```bash
WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_button_temporal_stage2_null_bc \
    --exp-name=button_press_stage2_null_bc --overwrite --batch-size=4
```

### 4. Stage 3：matched full/null targets

冻结 Stage 2 Teacher，同一样本、同一 flow noise 下算 `A_full` 与 `A_null`，
`delta_A_target = A_full[..., :9] - A_null[..., :9]`。

```bash
CKPT=checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999

python scripts/extract_forcevla_paired_targets.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_stage2_null_bc \
    --checkpoint=$CKPT --output-dir=artifacts/button_stage3_paired_targets/train \
    --seed=0

python scripts/extract_forcevla_paired_targets.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_100hz_val \
    --checkpoint=$CKPT --output-dir=artifacts/button_stage3_paired_targets/val \
    --seed=0
```

**先看这一步的 summary 再往下走。** `normalized_full_vs_expert_mse` 与 `normalized_null_vs_expert_mse`
的相对差用于确认 full/null 路径是否产生了非退化、接触相关的教师信号；它不是 Fast 性能的严格上界，也不
等同于真实控制收益。如果两条路径几乎重合，应先检查 Stage 2 和 force conditioning，再训练 Fast。

两个指标都按 `all` / `xyz` / `rotation_6d` / `gripper` 分组给出，且只统计前 10 个真实机器人维度——模型
输出宽 32 维，其余 22 维是 padding，把它们平均进去会把误差和收益一起拉向 0。分组读法：力条件主要该体现
在 `xyz` 和 `rotation_6d` 上；如果收益全落在 `gripper`，那不是接触控制的证据。

### 5. Slow cache

Slow 就是 Stage-2 Teacher 的 null 路径，不另训模型。只要 `--seed` 与 Stage 3 一致，两边 flow noise 都由
`row_keyed_noise` 按 dataset row 生成，`A_ref` 与 `A_null` 在每个 Slow 观测行上逐位相同，剩下的参考误差
只有更新间隔内的陈旧度。

**train 必须 `--slow-rate-hz 0` 提全速率**，训练时才能重抽时序；val 保留一个固定实现作为不参与训练的
时序。

train cache 上的 `--update-jitter-ms` 不用管：cache 会同时存下未抖动的网格时间戳，训练每次重抽时序都从
网格重新抽一次抖动，不会在 cache 已有的抖动上再叠一次。

```bash
CKPT=checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999

python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_stage2_null_bc \
    --stage3-dir=artifacts/button_stage3_paired_targets/train \
    --checkpoint=$CKPT --slow-rate-hz 0 \
    --output=artifacts/button_slow_cache/train.npz --seed=0

python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_100hz_val \
    --stage3-dir=artifacts/button_stage3_paired_targets/val \
    --checkpoint=$CKPT \
    --output=artifacts/button_slow_cache/val.npz --seed=0
```

检查 val summary 的 `time_features`：`saturated_age_fraction` 应为 0，`alpha_interior_fraction` 应明显
大于 0，`age_alpha_correlation` 不应接近 ±1。任一项不满足说明时间 token 退化了。

### 6. Stage 4：训练 Fast residual

`train_fast_residual.py` 会拒绝写入非空目录，防止覆盖旧 checkpoint。每次正式重跑都给 `FAST_RUN` 一个新的
目录名；下面的训练、评估和部署必须使用同一个目录。

```bash
FAST_RUN=checkpoints/fast_residual_twohead_repro

WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_fast_residual.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --output-dir=$FAST_RUN \
    --steps=10000 --batch-size=64 \
    --chunk-steps=5 --timing-resample-interval=500 \
    --slow-rate-hz 5 15 --slow-latency-ms 50 300 \
    --staleness-weight=1.0 --seed=0 \
    --wandb-project=forcevla --wandb-name=button_press_fast_twohead_repro
```

W&B 的 `timing/*` 记录每次重抽后的 `ready_row_fraction`、`mean_normalized_age` 和
`saturated_age_fraction`；后者不为 0 说明 `--context-age-scale-ms` 太小。训练用的时序带写进
`metadata.json` 的 `trained_timing`，部署时要读它。

### 7. held-out 评估

```bash
# 若这是新终端，设成训练时使用的同一目录。
FAST_RUN=checkpoints/fast_residual_twohead_repro

python scripts/evaluate_fast_residual.py \
    --targets=artifacts/button_stage3_paired_targets/val \
    --slow-cache=artifacts/button_slow_cache/val.npz \
    --checkpoint=$FAST_RUN/step-10000/params \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --contact-threshold-n=5.0 \
    --output=artifacts/button_fast_evaluation/val.json
```

同一条命令也会跑 force 消融（force history 置零 / 打乱），用来确认力头不是只靠 Slow
context/state/reference 猜 residual。消融只改变力头的输出：主配置下 staleness head 的输出在三种情形下
必须逐位相同，`staleness_force_blindness_max_deviation` 报告最大偏差，不为 0 说明 force token
泄漏进了陈旧头。**反过来，对 `--force-sighted-staleness` 训练的消融 run 该值必须非 0**；那种 run 上读到 0
说明 checkpoint 被装进了错误的架构。

head 布局和架构开关都从 run 目录的 `metadata.json` 读，不用手动指定：单头的旧 checkpoint 仍可直接评估，
只是不报 staleness 项。`force_blind_staleness` 和 `analytic_rebase` 改变的是前向的含义而不是形状，
漏读不会报错只会静默出错，所以它们必须随 checkpoint 一起走。

`--norm-stats-dir` 是必需的，不再可省。所有 pose 量都是相对 state 的 delta，而 Slow reference 的基准是它
所属 packet 那一行的 state、Teacher 与 expert 的基准是当前行的 state；要把它们放在一起比，必须先各自反归
一化再加回自己的物理 state，还原成绝对位姿。缺了 norm stats 做不到这一步，直接比 delta 就是拿两个不同原
点的向量相减。测地线旋转误差同样只在绝对位姿上有意义：delta 的 6D 列模长接近 0，Gram–Schmidt 会把它正交
化成一个与真实姿态无关的旋转矩阵，且不报错。

重点看三项：

- `deployment_error_decomposition`：Slow 与总组合误差分别报告物理平移 RMSE（米）和 SO(3) 测地线旋转
  RMSE（弧度）；两个 Fast 项报告各自优化的 normalized MSE。单位不同，不能相加，只用于定位误差来自 Slow
  reference、力残差、陈旧修正，还是最终组合。现在每一项都有对应的头和监督，不再有"只报告、不优化"的项。
- `staleness_vs_reference_drift`：陈旧头相对"输出零"（即单头学生的等效行为）的增益。
- `per_step_residual_mse_normalized`：chunk 后几步是否也学到了。
- 分平移 / 测地线旋转的物理误差。报告不再给出把 xyz 米和无量纲 6D 坐标混在一起的总体 pose MSE；
  `rotation_6d_coordinate_rmse` 只作表示诊断，模型排序看 `translation_rmse_m` 和
  `rotation_geodesic_rmse_rad`。

### 8. 可选消融与离线时序扫描

`flash_gemma` 保持相同的 TCN、Slow cache、双头 targets 和训练超参数，只替换 Fast 的融合/读出模块：
原始配置使用一个无位置编码的 Gemma-style set block 和单个展平 chunk query；该配置使用 OpenPI 原生
Gemma block（Gemma RMSNorm、RoPE、GQA、gated MLP）以及每个 residual step 一个 learned query。这对齐
Realtime-VLA FLASH 的 draft-head 结构原则，但不是其 speculative verification 系统，也不加载完整 18 层
Gemma。为隔离架构变量，第一轮从零初始化，并写入独立目录：

```bash
FLASH_GEMMA_RUN=checkpoints/fast_residual_flash_gemma

WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_fast_residual.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --output-dir=$FLASH_GEMMA_RUN \
    --model-profile=flash_gemma \
    --steps=10000 --batch-size=64 --chunk-steps=5 \
    --timing-resample-interval=500 \
    --slow-rate-hz 5 15 --slow-latency-ms 50 300 \
    --staleness-weight=1.0 --seed=0 \
    --wandb-project=forcevla --wandb-name=button_press_fast_flash_gemma
```

评估仍使用第 7 节的命令；loader 从 run 的 `metadata.json` 自动恢复 `decoder_type`，不要把
`flash_gemma` checkpoint 强行装入原始 `selected` 结构。

Teacher-initialized 消融保持上述架构与训练设置不变，只改变初始化。它复制生成 Stage 3 paired
targets 的 ForceVLA Teacher 的 TCN stem/4 blocks，以及第 0 个 1024D Action Expert Gemma block；
Teacher 的 LoRA 增量会先合并进稠密权重。Teacher 独有的 TCN `1024->2048` 投影不会复制，Fast
独有的 intent/state/reference/time projections、per-step queries 和双输出 heads 仍按通常方式初始化：

```bash
TEACHER_INIT_RUN=checkpoints/fast_residual_flash_gemma_teacher_init
TEACHER_CKPT=checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999

WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_fast_residual.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --output-dir=$TEACHER_INIT_RUN \
    --model-profile=flash_gemma_teacher_init \
    --teacher-init-checkpoint=$TEACHER_CKPT --teacher-init-layer=0 \
    --steps=10000 --batch-size=64 --chunk-steps=5 \
    --timing-resample-interval=500 \
    --slow-rate-hz 5 15 --slow-latency-ms 50 300 \
    --staleness-weight=1.0 --seed=0 \
    --wandb-project=forcevla --wandb-name=button_press_fast_flash_gemma_teacher_init
```

`metadata.json/teacher_initialization` 会记录来源、Teacher layer、复制数组数、合并 LoRA 数和实际复制
参数量，避免把“请求了初始化”和“初始化确实发生了”混为一谈。

`force_only` 使用同一训练入口，只关闭 staleness head；必须写到另一个空目录：

```bash
FORCE_ONLY_RUN=checkpoints/fast_residual_force_only_repro

WANDB_MODE=online XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_fast_residual.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --output-dir=$FORCE_ONLY_RUN \
    --steps=10000 --batch-size=64 --chunk-steps=5 \
    --timing-resample-interval=500 \
    --slow-rate-hz 5 15 --slow-latency-ms 50 300 \
    --no-staleness-head --seed=0 \
    --wandb-project=forcevla --wandb-name=button_press_fast_force_only_repro
```

另外两个消融开关同样复用这个入口，各写到自己的空目录：

| 开关 | 作用 | 状态 |
|---|---|---|
| `--force-sighted-staleness` | staleness head 改从共享前向读出，不再屏蔽 force token。decoder 只跑一遍，fast path 成本减半 | 已测，主配置不采用 |
| `--analytic-rebase` | staleness 目标去掉基准修正项，头只学 drift，闭式项**仅在离线评估合成时**加回 | **offline ablation only**，见下 |
| `--single-head-total-target` | 关闭第二个 head，让单个 head 直接学习 `delta_force + delta_stale`；必须保持 `--reconstruction-weight=0` | 已测，见第 10 节 |

`--analytic-rebase` 训练出的 checkpoint **不可部署**：只有 `scripts/evaluate_fast_residual.py` 会把闭式项
加回来以便同口径打分，`src/openpi/serving/` 里没有对应路径。新训练的 run 会把
`analytic_rebase_is_offline_ablation_only` 写进 `metadata.json`；`abl_analytic_rebase{,_s1,_s2}`
三个既有产物是事后补写的，这些目录不在 Git 里，转移或重建时要确认该字段仍在。

陈旧目标的线性可读性基线（不需要 GPU，train 拟合 val 打分）：

```bash
python scripts/probe_staleness_linearity.py \
    --train-targets=artifacts/button_stage3_paired_targets/train \
    --val-targets=artifacts/button_stage3_paired_targets/val \
    --train-slow-cache=artifacts/button_slow_cache/train.npz \
    --val-slow-cache=artifacts/button_slow_cache/val.npz \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --output=artifacts/button_fast_evaluation/staleness_linear_probe.json
```

它测的是线性可读性，不是信息上限：低值同样可能是量存在但被非线性编码。特征分四组，`tabular` 故意
**不含**缓存的 Slow context，`tabular_plus_slow_context` 才覆盖力盲陈旧头的全部可见输入。
ridge 在 train 内部划出的 dev 上选，从不看 val；`--ridge 0` 用于验证加入 `S_k` 后两个目标残差重合的退化关系。

时序扫描需要 held-out 的全速率 Slow cache；普通的 `val.npz` 是一个固定低速实现，不能向上重采样：

```bash
CKPT=checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999
FAST_RUN=checkpoints/fast_residual_twohead_repro

python scripts/extract_slow_cache.py \
    --config-name=forcevla_button_temporal_stage2_null_bc \
    --data-config-name=forcevla_button_temporal_100hz_val \
    --stage3-dir=artifacts/button_stage3_paired_targets/val \
    --checkpoint=$CKPT --slow-rate-hz 0 \
    --output=artifacts/button_slow_cache/val_fullrate.npz --seed=0

python scripts/sweep_fast_latency.py \
    --targets=artifacts/button_stage3_paired_targets/val \
    --slow-cache=artifacts/button_slow_cache/val_fullrate.npz \
    --checkpoint=$FAST_RUN/step-10000/params \
    --fast-run=$FAST_RUN \
    --norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56 \
    --slow-rate-hz 3 5 10 15 20 30 \
    --slow-latency-ms 50 150 300 450 600 800 \
    --contact-threshold-n=5.0 \
    --output=artifacts/button_fast_evaluation/latency_sweep_repro.json

python scripts/benchmark_fast_path_latency.py --budget-hz=100 --repeats=200
```

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

主流程使用 `tcn`；`instantaneous`（原始 ForceVLA）和 `native_instantaneous`（同源最新单点）只用于可选
Teacher 前端消融。

### Fast residual student

```text
Force history 10×6 @100 Hz ─→ causal TCN ─→ 1 force token
最新 robot state 10D       ─→ state token
缓存的 Slow V-L context    ─→ 2 intent tokens          ┐
当前插值 A_ref 10D         ─→ reference token          ├→ 1 层 Gemma-style decoder
[context age, interp alpha]─→ time token               │
learned residual query     ─→ query token              ┘
                                    ↓
              force head:     5 × xyz+6D residual    （见到 force token）
              staleness head: 5 × xyz+6D correction  （force token 被屏蔽）
```

```text
A_cmd[k, :9] = A_ref(t + k·action_period)[:9] + delta_force[k] + delta_stale[k]
A_cmd[k, 9]  = A_ref(t + k·action_period)[9]      # gripper 由 Slow 独占
```

**两个头，两个目标。** 部署时要补的总量是 `A_full(t) − A_ref(t)`，它恰好分解成两部分：

```text
delta_force 目标 = A_full(t) − A_null(t)      # 力改变了什么
delta_stale 目标 = A_null(t) − A_ref(t) + (σ_state/σ_action)·(S_t − S_k)
                                              # 语义缓存变旧期间参考漂移了多少
```

**陈旧目标里那一项基准修正不是可选的。** ForceVLA 的动作是相对 state 的 delta，而 `A_ref` 是相对
Slow 观测那一行的 state `S_k` 的 delta，`A_null(t)` 是相对当前行 `S_t` 的 delta。两者原点不同，直接相减
等于把两个不同原点的向量相减，漏掉的就是原点位移，换算到动作归一化单位就是 `(σ_state/σ_action)·(S_t − S_k)`。
漏掉它会让物理旋转误差相对单头基线倒退 74.85%——旋转坐标在这两行之间的变化量和参考漂移本身同量级。
`S_k` 在部署时来自 Slow packet 的 `state_at_observation`，`to_absolute_command` 已经在用它还原绝对指令。

单头直接训练这个和是可学习的，但当前 Button 单 seed 对照不如分头监督。`single_head_total` 的 5k 最佳已保存
checkpoint 在 held-out 数据上的 step-0 总修正 MSE 为 0.02369（相对零修正改善 13.94%）；主双头 10k 为
0.02230（改善 18.97%）。单头继续到 10k 后退化到 0.02605，说明它还更容易过拟合。此前
`--reconstruction-weight=0.5` 的实验不是这个对照：它同时要求一个 head 拟合 force target 和总重建，确实会让
力残差保真度从 +91.8% 掉到 −18.6%，但不能用来否定直接 sum-target。以上仍是单 seed 离线结果，不能写成
普遍机制结论。

陈旧项与力基本无关，因此 staleness head 走第二次 decoder 前向，**force token 在 attention mask 里被屏蔽**，
力无关性由结构保证而不是靠损失函数自觉。评估脚本会在力消融下比较两次 staleness 输出，主配置下
`staleness_force_blindness_max_deviation` 必须为 0。代价是 decoder 要跑两遍，fast path 的计算量约翻倍。

这个代价是实测值得的，但结论有边界。`--force-sighted-staleness`（共享单次前向）三 seed 对照下，
陈旧头增益由 +0.038 转为 −0.095，即比直接输出零还差，说明它已不在预测陈旧漂移；对 Teacher 的合成平移
RMSE 也从 0.00865 退到 0.01114。但同一组 run 在**对 expert** 的口径上反而更好，所以只能说力盲更好地保持了
Teacher 定义的职责分解，不能说它是整体控制性能更好的系统——后者要等真机。

`--no-staleness-head` 退回单头的旧参数结构。

**输出短 chunk 而不是单步**，所以 Fast 的输出速率与调用速率解耦：跟得上就只执行 `k=0`，落后了就往后
多消费几步。后面几步只在落后时才执行，因此 loss 按 `--step-decay 0.5` 几何降权。

**两个时间特征必须互不共线。** 旧的 `[phase, age]` 是同一个 age 除以两个常数，实测相关系数 1.0，等于
只有一维。现在是 `[age/scale, alpha]`，alpha 是两个 Teacher waypoint 之间的插值位置。

**Fast 不接新图像。** `sample_paired_actions_and_context` 里 full 和 null 共用同一份 `prefix_out_fix` 和
同一个 flow noise，因此差分没有视觉输入或采样噪声不匹配；但
`A_full − A_null` 仍是以视觉语言上下文、state 和任务为条件的 Teacher force-induced deviation，不是只依赖
force 的纯函数。Fast 通过缓存的 Slow context 接收这部分条件。如果要响应两次 Slow 更新之间出现的新视觉
变化，才需要改变目标和输入定义。

主实验优化两项：`MSE(delta_force, A_full − A_null)` 和 `MSE(delta_stale, 上式的基准修正后陈旧目标)`。两者都
精确时它们的和就等于同一基准下的 `A_full − A_ref`，所以监督执行量的 `--reconstruction-weight` 在最优点冗余，默认 0；调高它是让两个
头互相让渡误差。`predict_gate=false`，gate 接口保留但恒为 1，且**只作用于力残差**——gate 表达的是"对力读数
不确信"，不该顺带决定要不要按一份过期的计划行动。

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
必须先转成 6D、state 与 force history 都要按训练 norm stats 归一化。

下面是机器人侧的**接口接线模板，不是可直接执行的 launcher**。仓库目前没有 Button 真机的一键启动脚本；
`read_robot`、`read_latest_robot_state`、`teacher`、`student`、`force_buffer` 和 `send_robot_command` 必须由
具体机器人驱动提供。离线训练与评估应使用上面的命令，不要把该模板误当成已完成的硬件部署入口。

```python
import time

from openpi.serving import slow_fast_deploy, slow_fast_loop, slow_fast_runtime

contract = slow_fast_deploy.load_contract(
    "artifacts/button_slow_cache/train.npz",
    fast_run="checkpoints/fast_residual_twohead_repro",  # 必须与 Stage 4 的 FAST_RUN 一致
)
build_packet = slow_fast_deploy.SlowPacketBuilder(contract)
# state 和 force history 都归一化；力传感器的原始牛顿值不能直接喂 Fast。
normalizers = slow_fast_deploy.load_normalizers(data_config.norm_stats)

def observe_slow():
    timestamp, raw_state, teacher_observation = read_robot()
    # timestamp 必须和 time.monotonic() 使用同一个时钟域。
    return timestamp, teacher_observation, slow_fast_deploy.convert_robot_state(raw_state)

def infer(observation):
    return build_packet(*teacher.sample_nominal_actions_and_context(rng, observation))

cache = slow_fast_runtime.SlowReferenceCache()
worker = slow_fast_loop.SlowWorker(
    observe=observe_slow, infer=infer, cache=cache,
    config=contract.slow_fast_config(residual_limit=0.02),
    context_age_scale_s=contract.context_age_scale_s, period_s=0.1,
)
# predict_residual 返回 (力残差, 陈旧修正, gate)；陈旧修正为 None 表示单头学生。
# 两个限幅必须分开给：陈旧修正通常比力残差大几倍，共用 residual_limit 会把它削掉。
controller = slow_fast_loop.SlowFastController(
    cache=cache, force_buffer=force_buffer, predict_residual=student,
    unnormalize_action=normalizers["unnormalize_action"],
    config=contract.slow_fast_config(residual_limit=0.02, staleness_limit=0.10),
    **{key: normalizers[key] for key in ("normalize_state", "normalize_force_history")},
)
assert contract.predicts_staleness, "两头学生才需要 staleness_limit；单头 run 这里是 False"

# Fast inference 和 actuator 不在同一线程。Fast 完成时发布整个短 chunk；
# actuator 按当前 timestamp 自动跳过推理期间已经过期的前几步。
command_cache = slow_fast_loop.FastCommandCache()

def observe_fast():
    timestamp, raw_state = read_latest_robot_state()
    return timestamp, slow_fast_deploy.convert_robot_state(raw_state)

fast_worker = slow_fast_loop.FastWorker(
    observe=observe_fast, controller=controller, command_cache=command_cache,
    period_s=0.02,  # 请求频率；真实完成频率仍由 Fast inference latency 决定
)

worker.start()
fast_worker.start()

# 独立的 actuator tick：sample() 根据 timestamp 选择 k，而不是永远执行 k=0。
try:
    send_robot_command(command_cache.sample(time.monotonic())["command"])
except (slow_fast_loop.MissingFastCommandError, slow_fast_loop.StaleFastCommandError):
    hold_or_abort_safely()
    # Fast 后台线程若已因致命错误退出，只有这里能拿到原因；不查就只能看到
    # chunk 一直过期，看不出是推理挂了还是 Slow 没跟上。
    fast_worker.raise_if_failed()
```

`normalize_force_history` 是必填参数，没有默认值：Fast 是在归一化后的力上训的，直接喂原始牛顿值在偏置和
量级上都错，但不会触发任何形状检查。归一化作用于整个零填充窗口；无效 slot 归一化后是 `-mean/std`，但
两个力编码器都在 stem 之前乘 `force_history_mask`，所以模型看到的仍是 0——在 controller 里额外清零是等价
的，不清零也不会失配。`controller.step()` 的 state 则严格要求已经通过 `convert_robot_state()` 转成 10D，
原始 7D state 会在模型调用前直接报错。

actuator 侧只需捕获两类异常：`MissingFastCommandError`（Fast 还没发布首个 chunk）和
`StaleFastCommandError`（整个 chunk 已过期）。`sample()` 容忍亚周期的时钟读取竞态——actuator 先读时钟、
worker 随后装入 packet 是正常的读序，不是错误——只在两个时钟真正相差超过一个 command period 时抛
`ValueError`。Fast worker 对时间戳不前进的观测按 transient 跳过（`StaleFastObservationError`），一次驱动
抖动不会让它永久停机。

形状从 cache 的 summary 读，**时序带必须从 Fast 训练 run 的 `metadata.json` 读**——推荐的 train cache 是
全速率提取的，它自己记录的带是退化的 `[0, 0]`。上机前用
`contract.check_measured_timing(slow_rate_hz=..., slow_latency_s=...)` 核对实测时序；超出带外要加宽带、
同步加大 `--context-age-scale-ms`，重提 cache 并重训 Fast（Stage 1–3 不受影响）。

`metadata.json` 的 `format_version` 为 2 时带 staleness head，`predict_staleness` 记录了这一点，
`load_contract` 会读进 `contract.predicts_staleness`。用 version 1 的旧 checkpoint 时 `predict_residual`
只返回两个值，控制器会当场抛错而不是静默少加一项。

还需要：安全门控、力残差与陈旧修正各自的限幅、确认 gripper 由 Slow 独占。

## 配置

| config name | 用途 |
| --- | --- |
| `forcevla_lora` | 原始 instantaneous ForceVLA |
| `forcevla_temporal_lora_aligned` | 发布数据的 30 Hz temporal compatibility baseline |
| `forcevla_usb_lora` | USB instantaneous 40k baseline |
| `forcevla_usb_temporal_lora_aligned` | USB 30 Hz Temporal Teacher |
| `forcevla_button_instantaneous` | Button instantaneous 40k 公平基线 |
| `forcevla_button_instantaneous_val` | Button instantaneous held-out loader |
| `forcevla_button_native_instantaneous_100hz` | Button 同源100 Hz最新单点力40k对照 |
| `forcevla_button_native_instantaneous_100hz_val` | Button 同源最新单点力 held-out loader |
| `forcevla_button_temporal_100hz` | Button Stage 1，100 Hz 力 |
| `forcevla_button_temporal_100hz_val` | Button held-out loader |
| `forcevla_button_temporal_stage2_null_bc` | Button Stage 2 |

`forcevla_lora` / `forcevla_usb_lora` 始终是 instantaneous baseline，训练 temporal 模型必须显式选 temporal
config。这些 LoRA config 沿用 OpenPI/ForceVLA 的 freeze filter：Gemma 主权重冻结，LoRA 参数、视觉编码器
和部分 robotics projection 可训练——是 low-memory recipe，不是严格的「只训 LoRA」。所有 config 的 W&B
project 均为 `forcevla`。

## 尚未证明的事

Temporal 与 Instantaneous 的优劣不是当前论文主张，相关配置只保留为可选 Teacher 前端消融。非零的
`A_full - A_null` 本身也不等于已证明有效控制分解。当前 Slow/Fast 有真实离线 held-out 结果，但最终结论仍
需要真机实验。
