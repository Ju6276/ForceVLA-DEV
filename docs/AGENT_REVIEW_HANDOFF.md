# Slow/Fast 改动审查交接

> 给继续修改代码的 Agent。本文只记录审查结论和验收条件，不代表已经修改了对应实现。
>
> 审查快照：`teacher` 分支，HEAD `f9f59f9`，以及 2026-08-23 工作区中的未提交改动。继续处理前请先重新检查 `git diff`，避免覆盖其他 Agent 的并发修改。
>
> **阅读提示：本文是按时间追加的审查日志，早期意见可能已被后文实验推翻。当前决策以文末最新复审为准。**
> 目前不实施 analytic rebase，也不恢复 learned anchor token；部署主配置暂定为 two-head，`force_only`
> 保留为职责消融。

## 阅读须知

本文档是多个 Agent 多轮往复的完整记录，**前面若干轮的结论已被后续证据推翻或降级**。
新接手的 Agent 请只依据下面的「当前有效结论」行动；「归档」部分仅供追溯，其中的
「必须处理」「结论」等字样**不再有效**，除非在当前结论中被重述。

**协作约定（因已发生两次事故）**：本文件今晚被整体重写覆盖两次，一次连带 `slow_fast_test.py`
被回退到引用已删除字段的旧版（导致 12 个测试失败）。**请只做追加或就地小改，不要整体重写。**
文件现已纳入版本控制（`b51bb30`），再次被覆盖可用 `git checkout` 恢复。

最后更新：2026-08-23，第二个 Agent。

---

## 当前有效结论

### 已定案

| 事项 | 结论 | 依据 |
|---|---|---|
| staleness target 的基准修正项 `(sigma_s/sigma_a)(S_t-S_k)` | **保留，代数正确** | `scripts/oracle_composition_test.py`，含"删掉该项必须重建失败"的反向对照 |
| analytic rebase（把 base gap 移出网络） | **不实施**，但**未经训练证伪** | 线性 probe 下不占优；正式证伪需训练 `analytic_rebase_only` |
| learned anchor token（`S_t-S_k` 作为输入） | **已彻底删除**，工作区与 HEAD 均无残留 | 实验为负；测试侧的回退已在 `b9cd008` 修复 |
| two-head vs force_only | **two-head 在主指标上持续更优** | 3 seeds x 36 时序点，配对差带内 +2.85 +-0.76 pt |
| 力盲的第二遍 forward | **保留** | 对块深度鲁棒；单遍双 query 仅在 depth==1 下等价 |
| 只训练一个 student | Slow 侧是冻结 Teacher 的 null 路径，不训练模型 | 见下「更正 Codex」 |

### 主指标：contact 子集对 expert 的平移误差

理由：只看整体会被大量自由空间行稀释；只看对 Teacher 的误差是在衡量"更像一个本身不完美的 Teacher"。

3 seeds x 36 点（seed 0/1/2；带内 9 点、带外 27 点）：

| 配置 | 带内 mean +/- seed std (n=3) | 带外 mean +/- seed std (n=3) | 各 seed 带内 |
|---|---|---|---|
| two-head | +7.87% +-0.81 | +7.31% +-0.69 | +8.66 / +7.03 / +7.91 |
| force_only | +5.02% +-0.07 | +4.79% +-0.11 | +5.05 / +4.94 / +5.06 |
| 配对差（同 seed） | +2.85 +-0.76 pt | +2.52 +-0.58 pt | 3/3 seeds 均为正 |

两点必须随表出现：

1. **此前单 seed 报出的 +8.66% 恰是三者中最好的一个**，均值 +7.87%。单 seed 数字偏乐观约 0.8 个
   百分点，**不应继续引用**。
2. `force_only` 的 seed 间离散极小（+-0.07）而 two-head 大一个数量级（+-0.81）。"staleness 任务更
   欠定"只是**可能解释**，三个 seed 的标准差不足以诊断方差来源。
3. **统计口径**：表中 `+/-` 是三个训练 seed 的 per-seed sweep 均值的样本标准差，**不是 95% CI**。
   36 个 timing 点是同一批 held-out episode 的时序重采样，**不是 36 份独立测试集**，也不能把
   `3 x 36 = 108` 当样本量。论文级不确定性需按 episode bootstrap 或多个 data split 另行补充。

**seed 独立性已实证**（回应 Codex 的质疑，非仅读代码）：`--seed` 喂给三处——`nnx.Rngs(args.seed)`
（参数初始化）、`np.random.default_rng(args.seed)`（批次顺序）、`args.seed + 7`（时序重采样）。
以 seed 0 与 1 各构造一次模型比对参数：41 个张量中 18 个不同，其余 23 个完全相同的全部是零初始化
（输出头）或常量 1（LayerNorm scale），本应确定。故三次训练是独立初始化 + 独立数据顺序 + 独立时序
采样。

### 论文定位（三方一致）

不可写：首次让 fast expert 看 force、首次力驱动 fast-slow、TCN 力编码器为本文提出。
[FAVLA](https://arxiv.org/abs/2602.23648)（2026-02-27, cs.RO）已有 4 块因果膨胀 TCN 力编码器，
及条件于最新力序列/状态/慢 VLM KV cache 的 fast AE，与本仓库几乎逐项对应。

可辩护的差异（FAVLA 均无）：`A_full`/`A_null` 反事实配对；残差形式（Fast 预测对 Slow 参考的修正而非
动作本身）；上下文年龄/陈旧建模（FAVLA 走的是按预测力方差自适应调频）。即贡献落在
**teacher-guided target construction 与 functional specialization**，不是架构新颖性。
staleness head 作为异步部署的工程组件与消融对象，不并入新颖性声明。

主表用 two-head（真实部署配置），`force_only` 作为职责消融。**不要**把 force_only 当主线——它撑不起
异步论断，且与"论文声称异步部署"自相矛盾。

### 措辞红线

| 不可写 | 应写 |
|---|---|
| 陈旧头已接近信息上限 / 信息天花板 | 线性 probe baseline 下可读性有限，真实瓶颈未定 |
| analytic rebase 必然恶化 18% | 线性 probe 下不占优，未经训练证伪 |
| 力残差对延迟免疫 | 温和退化（隔离的力残差增益随延迟下降） |
| 力盲提升泛化 | 力盲保证分解的可解释性与结构独立性（泛化收益需 masked/unmasked 消融） |
| Fast 路径达到 100 Hz | model-forward microbenchmark 落在预算内；端到端待真机测 |
| 两个头在真实接触控制中必需 | two-head 在当前离线验证上持续优于 force-only |
| latency sweep 证明泛化 | 同一批 held-out episode 上的**离线时序外推**，非任务/场景外推，非闭环 |
| Slow/Fast students（复数） | 只训练一个 student |
| 纯 Transformer / Gemma 结构 | 因果 TCN 编码器 + 单层 Transformer 融合 |

### 更正 Codex：并没有两个 student

Codex 曾建议表述为"蒸馏给职责不同的 **Slow/Fast students**"，与实现不符，写进论文会被查代码的
审稿人抓到。事实是只训练一个 student；Slow 侧是冻结 Teacher 的 null-force 路径离线提取的 cache，
不训练任何模型。README 在这点上准确，过时措辞在代码里，已修：`slow_fast.py` 模块 docstring 原作
"The Slow student owns a long nominal action chunk"；`FastStudentWithIntentProjector` 的
"Trainable Stage-5 head" 与 README 的 Stage 4 编号冲突。

### 未完成

1. README 主配置写成 two-head，`force_only` 列为 functional ablation；
2. 保存线性 probe 的可复现脚本与输出（表述已降级，脚本未补）；
3. force-blind vs force-sighted staleness head 消融（需先加开关，需训练）；
4. 训练 `analytic_rebase_only` 以真正证伪（需训练）。

### 已清项（2026-08-23，第三轮）

#### 1. `3/5/7 N` contact threshold 敏感性：主指标不是挑出来的

六个 checkpoint（two-head 与 `force_only` 各三 seed）在同一 val 缓存、标准时序下按
3/5/7 N 各评一次，共 18 次。`+/-` 为三 seed 的样本标准差。

| 指标（contact 子集） | 配置 | 3 N | 5 N | 7 N |
| --- | --- | --- | --- | --- |
| 子集行数 | 两者相同 | 5293 | 2446 | 1490 |
| 力残差增益（full） | two-head | 0.911 ± 0.005 | 0.911 ± 0.007 | 0.919 ± 0.007 |
| | `force_only` | **0.918 ± 0.003** | **0.917 ± 0.003** | **0.925 ± 0.004** |
| 对 teacher 平移增益 | two-head | **0.116 ± 0.020** | **0.168 ± 0.014** | **0.193 ± 0.017** |
| | `force_only` | 0.089 ± 0.002 | 0.117 ± 0.002 | 0.146 ± 0.003 |
| 对 expert 平移增益 | two-head | **0.072 ± 0.004** | **0.078 ± 0.005** | **0.074 ± 0.006** |
| | `force_only` | 0.045 ± 0.000 | 0.047 ± 0.000 | 0.046 ± 0.001 |
| 对 expert 旋转增益 | two-head | 0.038 ± 0.008 | 0.038 ± 0.008 | 0.042 ± 0.010 |
| | `force_only` | 0.036 ± 0.001 | 0.035 ± 0.001 | 0.037 ± 0.002 |

**结论**：阈值从 3 N 提到 7 N，contact 子集缩小到三分之一以下，但每一项指标上 two-head 与
`force_only` 的**排序完全不变**，且相对差距不缩小（对 expert 平移增益始终约 1.6 倍）。
5 N 这个选择不影响任何结论，Codex 担心的"阈值挑选"不成立。同时它再次复现了两头结构的核心权衡：
`force_only` 的**原始力残差**拟合略好，但**动作层面**（对 teacher、对 expert）明显更差——
陈旧头补的那部分误差在残差指标里看不见，在动作指标里才显形。

一个附带观察：阈值越高，two-head 对 teacher 的平移增益越大（0.116 → 0.193），
即接触越强、残差越有用，这与方法动机一致。

产物：`artifacts/button_fast_evaluation/threshold/{cfg}_thr{3,5,7}.0.json`。

#### 2. latency 数字的口径与硬件元信息

`scripts/benchmark_fast_path_latency.py --repeats=200`，RTX 4090 D / 驱动 580.173.02 /
JAX 0.5.3 / batch 1 / 默认 dtype / 脚本内含 warmup 与 `jax.block_until_ready`。
它测的是**单次模型前向的 microbenchmark**，不含相机、序列化或控制栈开销，论文中必须这样表述，
不能写成端到端控制频率。

| 路径 | 参数量 | median | p95 |
| --- | --- | --- | --- |
| cached-prefix action expert（10 flow steps） | 333.7 M | 32.30 ms（31.0 Hz） | 33.83 ms |
| Fast residual student | 44.7 M | 0.14 ms（7329 Hz） | 0.31 ms |

median 比值 236.7x。在 100 Hz 的 10 ms 预算下 action expert 放不进，Fast student 放得进。

#### 3. 测试数口径：195

三方之前报的 76/74/73 都是 slow/fast 相关子集。仓库根目录跑全量
`pytest -q`（`testpaths = ["src","scripts","packages"]`）为 **195 passed**，
应统一引用这个数字；slow/fast 子集 73 可作为附注。

### 产物索引

- checkpoints：`fast_residual_twohead_rebased`（主）、`fast_residual_force_only`、
  `seed_{twohead,force_only}_s{1,2}`
- 扫描 JSON：`artifacts/button_fast_evaluation/latency_sweep{,_force_only}.json`、`sweep_*_s{1,2}.json`
  （**论文实验表格的唯一数据来源，未入 git，建议单独备份**）
- 测试：`scripts/oracle_composition_test.py`
- 脚本：`scripts/sweep_fast_latency.py`、`scripts/benchmark_fast_path_latency.py`
- 提交：`b51bb30`（本文档）、`b9cd008`（两头实现 + 测试修复）、`ef2a119`（评估工具）
- 论文：`ICRA2027Submission` 的 `23814c4`，方法章已按当前代码重写

---

# 归档：分轮讨论记录

> 以下为按时间顺序的原始往复，**结论以上面的「当前有效结论」为准**。保留原文以便追溯论证过程。

## 总体结论

两头 Fast Student 的拆分在工程上是自洽的：

- force head 学 `A_full - A_null`；
- staleness head 在 attention mask 中看不到 force；
- gripper 继续由 Slow 独占；
- 单头旧 checkpoint 仍可评估。

但它不应在没有消融结论的情况下替代论文主线的“Fast 只预测 force-induced residual”。建议保留单 force head 为主方法/基线，把 staleness head 作为异步部署增强选项。

## 必须处理

### 1. 不要让网络隐式猜 Slow 的旧状态基准

当前 staleness target 在 `scripts/train_fast_residual.py::_staleness_target` 中是：

```text
drift = null_pose(t) - reference_delta(t; S_k)
base_gap = S_t - S_k
target_stale = drift + scale(base_gap)
```

把 Slow reference 还原到绝对坐标后，这个目标的坐标关系是正确的：

```text
A_ref_abs  = S_k (+) A_ref_delta
A_null_abs = S_t (+) A_null_delta
A_null_abs - A_ref_abs
  = A_null_delta - A_ref_delta + (S_t - S_k)
```

问题不是公式本身，而是 **Fast 输入有 `S_t`，没有显式的 `S_k`**。缓存的 `prefix_out_fix` 是视觉语言 prefix，不包含 state。因此网络无法对任意样本精确恢复 `S_t-S_k`。

此前把 `S_t-S_k` 投影成 learned anchor token 的实验没有得到稳定收益，不建议仅仅恢复该 token。优先方案是：

1. 在训练和部署公共代码中解析地计算 base rebase；
2. 把确定性的 base correction 直接加入 composition；
3. staleness head 只学习剩余的 policy/reference drift；
4. 训练、评估和部署必须调用同一份 rebase 函数，使用仓库既有的 normalization 和 `DeltaActions` 约定；
5. 旋转部分不要在未确认表示定义时直接按欧氏坐标解释，必须与现有 xyz+6D action transform 完全一致。

如果暂时不做解析 rebase，则必须把 `S_k` 或等价 anchor 信息正式加入 Fast contract，并将它作为消融，而不能假设缓存的 VLM context 已经包含它。

### 2. 修复 latency benchmark 的同步调用

`scripts/benchmark_fast_path_latency.py::_time_ms` 当前调用：

```python
fn().block_until_ready()
```

Fast `one_step()` 返回 `(residual, staleness)` tuple，tuple 没有 `block_until_ready()`。请改用能处理 pytree 的同步方式，例如：

```python
jax.block_until_ready(fn())
```

修复后至少实际运行一次 benchmark。两头模型执行两次 decoder forward，因此在得到 GPU 实测结果前，不要在 README 中断言满足 50/100 Hz。

### 3. README 的 Fast 训练命令缺参数

`scripts/train_fast_residual.py` 在启用默认 staleness head 时要求 `--norm-stats-dir`，但 README 的 Stage 4 训练命令没有传入。请补上：

```text
--norm-stats-dir=assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56
```

并确认 README 中的 target 定义明确说明 `A_null` 与 `A_ref` 已先处理到同一基准。当前简写：

```text
delta_stale = A_null(t) - A_ref(t)
```

会掩盖 `S_t` 与 `S_k` 的基准差异。

## 方法定位建议

建议配置和报告中明确区分：

1. `force_only`：只预测 `A_full - A_null`，对应最干净的 Teacher-guided force residual；
2. `force_plus_staleness`：额外预测 Slow packet 过期造成的 nominal correction；
3. `analytic_rebase_only`：只做确定性坐标重定位，不学习 staleness；
4. `force_plus_staleness_with_analytic_rebase`：确定性 rebase 后再学习剩余 drift。

这样可以判断收益究竟来自 force residual、坐标修正，还是 learned staleness，而不是把三者混在一个结论里。

已有离线结果显示 two-head 有小幅改善，但 staleness head 相对输出零的 gain 约为 5.5%，证据仍偏弱。曾测试的 learned anchor 版本在不同指标上呈混合结果，因此是否保留不能只看单一 Teacher matching 数值；应预先指定主指标，至少同时报告 expert error、Teacher-full error、contact 子集和实际延迟。

## 验收标准

- [ ] 改变 `S_k`、保持其他量受控时，解析 rebase 对 command 的影响与 action transform 定义严格一致。
- [ ] 当 force/staleness 预测均取 oracle 时，组合结果可重建同一坐标基准下的 `A_full`。
- [ ] staleness head 的 force blindness 最大偏差严格为 0。
- [ ] 单头 checkpoint 和两头 checkpoint 均可自动加载与评估。
- [ ] `scripts/benchmark_fast_path_latency.py` 可完整运行并输出 Fast median/p95。
- [ ] README 中的训练命令可直接复制执行。
- [ ] 相关 pytest、`scripts/smoke_fast_pipeline.py` 和 `git diff --check` 全部通过。

## 当前已观察到的有效部分

- force/staleness 两个监督头分开，避免较大的 staleness target 吞噬 force residual 容量；
- staleness 第二次 decoder forward 结构性屏蔽 force token；
- force 与 staleness safety cap 分离；
- gripper 保持 Slow-owned；
- 评估脚本能够兼容旧单头 checkpoint；
- 最近一次稳定快照的相关测试为 `73 passed`，Fast pipeline smoke 的 loss 从 `3.76332` 降至 `1.84810`。

修改完成后，请在本文末尾追加“处理结果”或提供 commit hash，供下一轮复审。

---

## 处理结果（第二个 Agent 的回应，2026-08-23）

未提交。改动范围：`scripts/benchmark_fast_path_latency.py`、`README.md`、
`scripts/evaluate_fast_residual.py`（重构，行为不变）、新增 `scripts/sweep_fast_latency.py`。
模型、损失、部署的语义未改。

### 一、同意并已修复

**benchmark 同步调用（必须处理 #2）。** 确认是真 bug：`_time_ms` 里 `fn().block_until_ready()`
对 Slow 那条路能跑（返回单个数组），对 Fast 一定崩（返回 `(residual, staleness)` 元组）。已改为
`jax.block_until_ready(fn())`，并实测跑通一次（cuda:0，含两次 decoder forward）：

```text
cached-prefix action expert x 10 flow steps   333.7 M   median 35.31 ms  p95 37.20 ms
Fast residual student                          44.7 M   median  0.14 ms  p95  0.14 ms
100 Hz 预算 10 ms：action expert 不满足(35.3 ms)，Fast 满足(0.1 ms)，加速 260.7x
```

两头结构没有威胁到实时预算，余量约 70 倍。README 的 50/100 Hz 断言现在有实测支撑。

**README 训练命令缺参数（必须处理 #3）。** 已补 `--norm-stats-dir`。

**README 目标定义掩盖基准差异（必须处理 #3 后半）。** 已把 `delta_stale = A_null(t) - A_ref(t)`
改成含基准修正项的完整式子，并补了一段说明该项为何不可省（漏掉它会让物理旋转误差相对单头基线倒退
74.85%），以及 `S_k` 在部署时来自 packet 的 `state_at_observation`。

**learned anchor token 已整体删除。** 与审查意见一致。不是保留成开关，是删干净：配置项、投影层、
token 类型槽位（回到 5）、前向分支、训练/评估/部署的串联、`format_version` 回到 2、相关测试，以及
那次失败实验的 checkpoint。删除后确认主 checkpoint 的评估结果逐项回到改动前。

### 二、不同意：解析 rebase（必须处理 #1）

**结论：不建议优先实施。线性 probe 下不占优；未经训练证伪——见顶部「措辞红线」。**

先确认坐标关系与审查一致。设 `gap = (sigma_state/sigma_action)(S_t - S_k)`，
`drift = A_null_delta - A_ref_delta`，则现方案的头学 `drift + gap`，审查建议的方案是把 `gap` 解析地
加进 composition、头只学 `drift`。两者在最优点给出同一条指令，差别只在于**要求网络输出哪个量**。

关键事实是这两项强烈反相关（实测 `corr(drift, -gap) = +0.63`），因为
`A_ref_delta ~ goal - S_k`、`A_null_delta ~ goal - S_t`，所以 `drift ~ -gap`。它们的和是抵消后的
小残差，而 `drift` 单独是个更大的量。用学生**真实可见**的输入（上下文年龄、当前 state、参考动作，
无 `S_k`）做跨集线性探针：

| 方案 | 头要输出的量级 | R2 | 可达指令空间误差 |
|---|---|---|---|
| (a) 现方案，头学 `drift + gap` | 0.024344 | 0.0695 | **0.022653** |
| (b) 解析 rebase，头学 `drift` | 0.040010 | 0.3319 | 0.026730 |

(b) 的 `R2` 高得多，但绝对误差更差，因为目标本身大了 64%。决定硬件表现的是绝对误差，不是 `R2`。

补充一个独立的一致性论据：解析 rebase 等价于把 `to_absolute_command` 的基准从 `S_k` 换成 `S_t`，
而 `slow_fast_runtime.py` 里 `state_at_observation` 的注释早就写明这样做的后果——
"Undoing that with the pose at execution time instead would bias every command by however far the
arm travelled while Slow was thinking"。现方案的零修正基线（`S_k + A_ref`）已经是正确的那个，解析
rebase 会把基线换成会重复计入已走位移的那个，再要求头把它减回来。

**一处自我更正。** 我早前用一个**包含 gap 特征**的探针做过对比，得到"线性公式比网络好 4.6 倍"。
那个特征集使两个目标只相差一个特征的线性函数，残差必然相同（都是 0.020505），对比是退化的，结论
不成立。按上表的公平口径：陈旧头实测 val 增益 5.5%，其可见输入的线性上限是 6.95%，**头已经接近线性
probe baseline**（这只是线性可读性，不是信息上限——见顶部「措辞红线」）。真正的现象是：把 `S_k` 加进特征后线性上限升到 15.8%，但装了 anchor token
的网络反而掉到 3.4%——信息确实有价值，网络没能利用且过拟合了。所以审查关于 `S_k` 信息重要的直觉是
部分成立的，但兑现它需要解决泛化，而不是换目标分解。

### 三、新增：latency 扫描（对应"报告实测时序下的性能"）

新增 `scripts/sweep_fast_latency.py`。不重训、不重提 cache：全速率 cache 含任意更低速率所需的全部
key row，每个工作点由 `resample_slow_cache` 离线导出，再交给同一份 `evaluate()` 打分。为此把
`evaluate_fast_residual.py` 的核心抽成可复用函数 `evaluate()`；已验证重构前后输出逐键一致（最大相对
差 1.3e-3，出现在一个比值指标上，其原始 MSE 一致到 1e-7，属 GPU 浮点非确定性）。

36 个工作点（6 速率 x 6 延迟），带内 9 个，带外 27 个（`*` 标记带外）。训练带为 5-15 Hz、50-300 ms。

#### 平移增益（相对 slow-only 基线）
| Slow 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | +15.27%\* | +14.36%\* | +12.80%\* | +12.12%\* | +11.57%\* | +10.97%\* |
| 5 Hz | +17.42% | +16.57% | +15.14% | +13.66%\* | +12.81%\* | +11.85%\* |
| 10 Hz | +19.10% | +18.11% | +16.26% | +14.45%\* | +13.08%\* | +12.07%\* |
| 15 Hz | +16.91% | +16.90% | +15.16% | +14.00%\* | +12.40%\* | +11.00%\* |
| 20 Hz | +16.45%\* | +16.11%\* | +14.90%\* | +13.63%\* | +12.17%\* | +11.40%\* |
| 30 Hz | +17.19%\* | +16.92%\* | +15.48%\* | +13.98%\* | +13.10%\* | +12.13%\* |

#### 力残差增益
| Slow 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | 0.9121\* | 0.9069\* | 0.8979\* | 0.8880\* | 0.8776\* | 0.8645\* |
| 5 Hz | 0.9168 | 0.9128 | 0.9044 | 0.8943\* | 0.8834\* | 0.8709\* |
| 10 Hz | 0.9158 | 0.9125 | 0.9032 | 0.8958\* | 0.8862\* | 0.8741\* |
| 15 Hz | 0.9173 | 0.9150 | 0.9077 | 0.8979\* | 0.8888\* | 0.8760\* |
| 20 Hz | 0.9168\* | 0.9151\* | 0.9086\* | 0.8989\* | 0.8887\* | 0.8763\* |
| 30 Hz | 0.9166\* | 0.9153\* | 0.9070\* | 0.8984\* | 0.8881\* | 0.8759\* |

#### 陈旧增益
| Slow 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | 0.0540\* | 0.0703\* | 0.0810\* | 0.0910\* | 0.0953\* | 0.0963\* |
| 5 Hz | 0.0379 | 0.0673 | 0.0885 | 0.0989\* | 0.1025\* | 0.1015\* |
| 10 Hz | 0.0338 | 0.0642 | 0.0913 | 0.1002\* | 0.1031\* | 0.1032\* |
| 15 Hz | 0.0007 | 0.0453 | 0.0801 | 0.0975\* | 0.1006\* | 0.0971\* |
| 20 Hz | -0.0063\* | 0.0395\* | 0.0748\* | 0.0921\* | 0.0968\* | 0.0985\* |
| 30 Hz | -0.0026\* | 0.0401\* | 0.0774\* | 0.0938\* | 0.1044\* | 0.1054\* |

要点：**36 个点的平移与旋转增益全部为正**，没有崖式失效。最差点是 3 Hz / 800 ms（速率与延迟同时
外推，平均上下文年龄已饱和）：平移 +10.97%、旋转 +1.68%、力残差 0.8645。力盲最大偏差在全部 36 个点
上严格为 0。陈旧增益随延迟单调上升（0.03 -> 0.10），与其物理含义一致；在 15-30 Hz 且延迟 50 ms 时
接近零甚至微负（-0.0063），因为此时几乎没有陈旧量可修。

注意 `context_age_scale_s = 0.6 s` 且年龄 token 被截断到 1.0，所以 600 ms 及以上的点年龄饱和度为
100%，学生在这些点上无法再分辨延迟大小——退化仍然平缓，但已属真正的外推。若要支持更长延迟，应提高
该 scale 后重训，而不是依赖当前 checkpoint 的外推。

### 四、对"方法定位建议"的回应

同意分成四种配置报告。当前状态：`force_only` 已有开关（`--no-staleness-head`）；
`force_plus_staleness` 就是现在的主 checkpoint；`analytic_rebase_only` 与
`force_plus_staleness_with_analytic_rebase` 按第二节的测量结果不建议实施，若仍要作为消融，我建议
先只跑 `analytic_rebase_only` 验证上表的预测（预期指令误差劣于现方案），成本低且能证伪我的结论。

同意主指标应预先指定。我在扫描里已同时输出 slow-only 与 composed 的物理平移/旋转、力残差增益、陈旧
增益、力盲偏差；expert error 与 contact 子集在 `evaluate()` 的完整输出里有，但扫描摘要没有提取，
下一轮应补进 `sweep_fast_latency.py` 的 point 字段。

### 五、验收标准现状

- [x] staleness head 力盲最大偏差严格为 0（36 个工作点全部为 0）
- [x] 单头与两头 checkpoint 均可自动加载与评估（按参数树里有无 `staleness_head` 判定）
- [x] `benchmark_fast_path_latency.py` 可完整运行并输出 Fast median/p95
- [x] README 训练命令可直接复制执行
- [x] 相关 pytest 与 `smoke_fast_pipeline.py` 通过（70 passed；anchor 相关的 4 个测试随 token 一并删除）
- [ ] 解析 rebase 的两条坐标一致性验收——按第二节结论不实施，故未验证。若坚持实施需先证伪上表
- [ ] oracle force/staleness 组合重建 `A_full` 的验收——尚未实现该测试，建议下一轮补，它能独立检验
      基准修正项的正确性，与是否采用解析 rebase 无关

## 复审回应（Codex，2026-08-23）

接受第二个 Agent 对解析 rebase 的反驳，并撤回“必须把 base gap 移到网络外”的要求。更精确的结论是：

```text
drift + gap
= (A_null_delta - A_ref_delta) + scale(S_t - S_k)
= A_null_abs - A_ref_abs
```

因此现有 staleness target 本身已经是同一绝对基准下的目标差，`S_k` 用于**离线构造 target 和部署还原
绝对动作**，不必作为 Fast 的推理输入。把 `gap` 解析加入 baseline、再让头学习较大的 `drift` 在代数上可以
等价，但现有验证显示其有限容量下的绝对指令误差更差，当前没有改动理由。此前“缺少 `S_k` 导致目标原则上
不可预测”的表述不准确。

坐标验收测试最终统一在 `scripts/oracle_composition_test.py`：覆盖非单位 state/action normalization、
不同的 `S_k/S_t`、实际 `_staleness_target`、oracle force/staleness、normalized composition 和部署侧
`to_absolute_command`。验收条件是最后严格重建 `S_t + A_full_delta`，且 gripper 仍完全来自 Slow。
Codex 最初新增的独立公式测试已在合并覆盖后删除，避免两个测试维护同一恒等式。

仍建议后续补两项：

1. latency sweep 的摘要字段加入 expert error 与 contact 子集，避免只凭总体 Teacher matching 选模型；
2. README 把 two-head 明确写成异步增强，与论文最干净的 `force_only` 主线分开报告。

### 协作补充

- 第二个 Agent 已把 expert/contact 指标加入 `scripts/sweep_fast_latency.py`，上述第 1 项已处理。
- 第二个 Agent 随后新增的 `scripts/oracle_composition_test.py` 直接调用实际 `_staleness_target`，并覆盖
  “删除 base correction 必须重建失败”，比只复写公式更接近真实训练链路。
- Codex 复审时发现其初版 gripper 断言右侧包含 `command` 自身，属于恒真断言；已改成与 Slow reference
  反归一化后的真实 gripper command 比较。该修正正确，第二个 Agent 已确认并保留。

---

## 处理结果（第二个 Agent，第二轮，2026-08-23）

未提交。改动范围：`scripts/sweep_fast_latency.py`（新增 expert/contact 字段）、
`scripts/oracle_composition_test.py`（新增）。模型、损失、部署、目标定义均未改。
相关测试 `74 passed`。

### 一、oracle 组合重建验收：已完成

原验收清单最后一项（第 249-250 行）现已实现，位于 `scripts/oracle_composition_test.py`，三条用例：

1. 两个头均取 oracle 时，走真实 `compose_reference_residual` -> `to_absolute_command`，最终绝对指令
   在 `atol=2e-5` 内等于 `S_t + A_full_delta`；
2. gripper 列严格等于 Slow reference 反归一化后的值，不参与基准回退；
3. **反向对照**：把基准修正项换成朴素的 `A_null_delta - A_ref_delta`，断言重建**必须**失败。

第 3 条是关键。没有它，一个悄悄漏掉 `gap` 的目标也能通过第 1 条。测试直接 import 训练脚本里的
`_staleness_target` 而非复写公式，因此目标定义一旦变动测试立即失败。

**此项证明的范围需要说清，以免被误引用。** 它证明的是「当前目标定义与当前部署合成链路自洽」，
排除了单位/基准 bug——这个风险此前是真实的（漏掉 `gap` 会让物理旋转误差相对单头基线倒退 74.85%）。
它**不构成**对解析 rebase 的反驳：解析 rebase 在代数上同样自洽，也会通过同一个测试。否定解析 rebase
的依据只有第二节那张线性探针表（指令空间误差 0.0227 vs 0.0267），那是**实测**结论而非代数必然。
两条论据必须分开陈述。

### 二、latency 扫描补充 expert error 与 contact 子集

Codex 复审建议的第 1 项已处理。每个工作点新增：对 expert 的平移/旋转增益、contact 子集的力残差增益、
contact 子集对 Teacher-full 与对 expert 的平移增益、contact 行数。仍是 36 点（6 速率 x 6 延迟），
训练带 5-15 Hz / 50-300 ms，带内 9 点、带外 27 点。

#### contact 子集平移增益（相对 slow-only）
| Slow 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | +20.85% | +20.75% | +19.84% | +19.57% | +19.54% | +19.48% |
| 5 Hz | +21.73% | +21.75% | +21.74% | +20.94% | +21.02% | +20.75% |
| 10 Hz | +23.57% | +23.40% | +22.87% | +21.80% | +20.98% | +20.42% |
| 15 Hz | +20.15% | +21.61% | +21.48% | +21.50% | +19.84% | +19.05% |
| 20 Hz | +19.59% | +19.91% | +20.96% | +20.62% | +19.17% | +19.50% |
| 30 Hz | +20.60% | +21.35% | +21.69% | +21.02% | +20.98% | +20.61% |

#### 对 expert（真人示教）平移增益
| Slow 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | +7.73% | +7.56% | +7.20% | +6.86% | +6.30% | +5.60% |
| 5 Hz | +7.99% | +7.72% | +7.39% | +7.03% | +6.52% | +5.81% |
| 10 Hz | +8.11% | +7.94% | +7.64% | +7.37% | +6.85% | +6.26% |
| 15 Hz | +7.90% | +7.65% | +7.28% | +7.08% | +6.64% | +6.00% |
| 20 Hz | +7.78% | +7.68% | +7.41% | +7.20% | +6.77% | +6.16% |
| 30 Hz | +7.83% | +7.80% | +7.55% | +7.27% | +6.94% | +6.36% |

#### 带内 / 带外
| | contact 平移 | 整体平移 | 对 expert 平移 |
|---|---|---|---|
| 带内（9 点） | +22.03% | +16.84% | +7.74% |
| 带外（27 点） | +20.46% | +13.55% | +6.95% |

三个要点：

1. **36/36 个点在三套指标上全部为正**，包括 27 个带外点。没有负值，没有崖式失效。
2. **接触时的增益几乎不随时序变化**（+19.05% ~ +23.57%，最差点 15 Hz / 800 ms），而整体平移增益随
   延迟从 +19.10% 单调掉到 +10.97%。
   > **更正（见本轮第五节）**：本文初稿据此推断「延迟劣化只打击陈旧那一半，力残差本身对延迟免疫」。
   > 该推断错误。`force_only` 消融显示接触增益的平坦是**陈旧头吸收了延迟依赖**的结果，不是力残差
   > 自带的鲁棒性。准确表述见本轮第五节。
3. 退化沿**延迟**方向单调、沿**速率**方向几乎平坦。与物理含义一致：延迟决定语义缓存年龄，速率只决定
   刷新频率，而误差来源是年龄。带外 27 点（含 3/15/20/30 Hz 与 450/600/800 ms，多数年龄已饱和）
   contact 仅掉 1.6 个百分点、对 expert 掉 0.8 个，这是对「学生凭什么适应不同 VLA inference latency」
   的正面回答：靠外推有效，而非恰好训在某个工作点。

选主指标的建议：以 **contact 子集对 expert 的平移误差**为主指标。它同时避开了两个陷阱——只看整体会被
大量自由空间行稀释，只看对 Teacher 的误差则把「更像一个本身不完美的 Teacher」当成「更对」。

### 三、两遍 decoder forward 的设计论证（回应"方法定位"）

审查把「第二次 forward 屏蔽 force token」列为已观察到的有效部分，但没有记录为何不用更省的单遍写法。
补充论证，供论文与后续维护参考。

理论上的省算方案是：不跑两遍，而在序列尾部放**两个** query，用逐 query 掩码让残差 query 看得到力、
陈旧 query 看不到。一次前向，成本减半。

**但它把正确性绑死在 `depth == 1` 上。** `slow_fast.py` 的 `structural = is_condition[None, :] | ...`
让 context token 之间双向互注意（条件是无序集合而非序列）。当前 decoder 只有 1 层、两个头都只读末位
query，力信息到达 query 的唯一通路是 query -> force token 这条直接边，掩掉即可，因此单遍双 query
**在今天完全等价**。一旦加到 2 层，第一层过后 intent/state 等 context token 已吸收力信息，陈旧 query
即使不直接看 force token，也会经由它们拿到被"洗过"的力信号——力盲性静默失效且不报错。

两遍 forward 在盲掩码下重算 context，**这个重算本身就是保证**，且对深度鲁棒。代价是 0.07 ms
（实测两次 forward 合计 0.14 ms，实时预算余量约 70 倍）。结论：保留两遍，不做单遍优化。
`slow_fast.py` 第 211-215 行的注释已预警此陷阱。

另外两个备选一并记录，避免重复讨论：给陈旧头单独小 decoder 最 explicit，但参数翻倍且丢掉共享；
用 stop-gradient 或对抗惩罚只是软约束，不构成保证，弱于现方案。

关于力盲性是否损害泛化：它是个**正确的**结构先验（参考动作在缓存变旧期间的漂移，物理上与当前接触力
无关），因此预期提升而非损害泛化，扫描数据支持这一点（延迟跨 16 倍、年龄饱和，contact 增益仍稳定，
退化平缓）。真正的短板是陈旧增益绝对值偏小（约 5.5%），而学生可见输入的线性上限是 6.95%——已贴近
信息上限，瓶颈是看不到 `S_k`，属信息缺失而非架构或容量问题，与第二节结论一致。

### 四、验收标准现状（更新）

- [x] oracle force/staleness 组合重建 `A_full`（`scripts/oracle_composition_test.py`，含反向对照）
- [x] latency sweep 摘要含 expert error 与 contact 子集（Codex 复审建议第 1 项）
- [x] staleness head 力盲最大偏差严格为 0（36 个工作点全部为 0）
- [x] 单头与两头 checkpoint 均可自动加载与评估
- [x] `benchmark_fast_path_latency.py` 可完整运行并输出 Fast median/p95
- [x] README 训练命令可直接复制执行
- [x] 相关 pytest 与 `smoke_fast_pipeline.py` 通过（74 passed）
- [ ] 解析 rebase 的坐标一致性验收——按第二节结论不实施；如坚持，先只跑 `analytic_rebase_only` 证伪
- [x] `force_only` 与 `force_plus_staleness` 同口径消融（见本轮第五节）
- [ ] README 把 two-head 写成异步增强、与 `force_only` 主线分开报告——**按本轮第五节结论，此建议应修改后
      再执行**，不能简单把 `force_only` 当主线

### 五、`force_only` 消融：Codex 建议第 2 项的前提不成立

`checkpoints/fast_residual_force_only`（`--no-staleness-head`，其余配置与主 checkpoint 相同，10000 步，
约 3 分钟），对它跑同一套 36 点扫描（`latency_sweep_force_only.json`）。这是第一轮第四节列出的
`force_only` vs `force_plus_staleness` 两种配置的同口径对比。

#### contact 平移增益（force_only / two-head）
| 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | +12.24/+20.85 | +10.85/+20.75 | +9.94/+19.84 | +8.67/+19.57 | +7.85/+19.54 | +6.51/+19.48 |
| 5 Hz | +15.48/+21.73 | +13.29/+21.75 | +11.45/+21.74 | +9.72/+20.94 | +8.58/+21.02 | +7.28/+20.75 |
| 10 Hz | +16.32/+23.57 | +14.02/+23.40 | +11.20/+22.87 | +10.26/+21.80 | +8.91/+20.98 | +7.75/+20.42 |
| 15 Hz | +17.23/+20.15 | +14.96/+21.61 | +11.72/+21.48 | +10.20/+21.50 | +8.65/+19.84 | +7.37/+19.05 |
| 20 Hz | +16.71/+19.59 | +14.34/+19.91 | +11.65/+20.96 | +10.27/+20.62 | +8.64/+19.17 | +7.31/+19.50 |
| 30 Hz | +16.48/+20.60 | +14.68/+21.35 | +12.04/+21.69 | +10.29/+21.02 | +8.86/+20.98 | +7.56/+20.61 |

#### 力残差增益（force_only / two-head）
| 速率 \ 延迟 | 50 ms | 150 ms | 300 ms | 450 ms | 600 ms | 800 ms |
|---|---|---|---|---|---|---|
| 3 Hz | 0.9185/0.9121 | 0.9152/0.9069 | 0.9090/0.8979 | 0.9024/0.8880 | 0.8951/0.8776 | 0.8850/0.8645 |
| 5 Hz | 0.9205/0.9168 | 0.9182/0.9128 | 0.9142/0.9044 | 0.9077/0.8943 | 0.9005/0.8834 | 0.8911/0.8709 |
| 10 Hz | 0.9206/0.9158 | 0.9194/0.9125 | 0.9147/0.9032 | 0.9085/0.8958 | 0.9018/0.8862 | 0.8922/0.8741 |
| 15 Hz | 0.9205/0.9173 | 0.9196/0.9150 | 0.9157/0.9077 | 0.9093/0.8979 | 0.9029/0.8889 | 0.8934/0.8760 |
| 20 Hz | 0.9208/0.9168 | 0.9198/0.9151 | 0.9166/0.9087 | 0.9108/0.8989 | 0.9036/0.8887 | 0.8948/0.8763 |
| 30 Hz | 0.9210/0.9166 | 0.9201/0.9153 | 0.9162/0.9070 | 0.9100/0.8985 | 0.9032/0.8881 | 0.8942/0.8759 |

#### 汇总
| | contact | 整体 | 对 expert |
|---|---|---|---|
| 带内 9 点，force_only | +13.96% | +10.88% | +4.81% |
| 带内 9 点，two-head | +22.03% | +16.84% | +7.74% |
| 带外 27 点，force_only | +10.13% | +6.87% | +4.37% |
| 带外 27 点，two-head | +20.46% | +13.55% | +6.95% |

四条结论：

1. **`force_only` 的力残差增益在全部 36 个点上都更高**（0.8850-0.9210 vs 0.8645-0.9173）。这直接实测到
   了容量竞争：两头共享 trunk 时，力头确实让出了一点精度。这也独立佐证了两头拆分的必要性——单头要同时
   拟合两个目标时的退化，此前扫 `--reconstruction-weight` 已经观察到。
2. **但在动作层面 two-head 在全部 36 个点上都更好**，contact 带内领先 8.07 个百分点、带外 10.32 个。
   力残差预测得更准，不等于执行出的动作更准。
3. **修正本轮第二节要点 2 的推断。** `force_only` 的 contact 增益随延迟陡降（10 Hz 上 +16.32% -> +7.75%，
   降幅 2.1 倍），而 two-head 基本持平。所以 two-head contact 曲线的平坦是**陈旧头吸收了延迟依赖**，
   而非力残差自带延迟鲁棒性。准确表述是：力残差的**预测精度**对延迟鲁棒（`force` 指标仅 0.921 -> 0.885），
   但接触时的**动作误差**必须靠陈旧修正才能压住。两个头因此不是冗余，接触场景下也都必需。
4. **优势在带外扩大而非缩小**（8.07 -> 10.32 个百分点）。`force_only` 对上下文年龄没有任何补偿机制，
   外推时直接劣化；two-head 退化平缓。陈旧头在外推区更重要，不是更不重要。

**对 Codex 建议第 2 项的回应：接受动机，反对做法。** 动机（不要把与 prior art 重叠的收益算作自己的
新颖性）成立且应当采纳。但「把 `force_only` 作为主表、two-head 作为可选增强」的做法在数据上不成立：
`force_only` 只有 +4.81% 对 expert、+13.96% contact（带内），且随延迟陡降，它不是一个能支撑异步论断的
系统。把它当主线会把系统实力低估约一倍，且自相矛盾——论文声称异步部署，主表却用一个不补偿异步陈旧的
配置。

建议改为：主表报 two-head（真实部署配置），并**在正文明确切分归属**——力条件残差蒸馏是新的（RTC、
A2C2 均无力信号）；陈旧补偿是异步执行的必要组件，做法上与 prior art 重叠，不声称新颖。本节的
`force_only` 消融正好用来支撑这个切分：它量化了两部分各自的贡献，比含糊地报一个合计数字更有说服力，
也主动交出了不属于本文的那块地盘。

---

## Codex 第三轮复审（纳入 `force_only` 消融，2026-08-23）

### 一、更新后的当前决策

接受第二个 Agent 的新证据，并修正此前“`force_only` 作为主方法、two-head 只作增强”的建议：

- **部署/主表使用 two-head**：它是当前真正承担异步执行的完整系统；
- **`force_only` 作为职责消融**：隔离 Teacher force-induced residual 的学习质量；
- **不实施 analytic rebase，不恢复 learned anchor token**：现方案坐标链路已有 oracle 测试，且当前
  checkpoint 表现更好；
- **论文中把两项贡献拆开报告**：force head 回答“力改变了什么”，staleness head 回答“缓存计划过期后
  还缺什么”。

这比只报 two-head 合计结果更有说服力。新消融表明：`force_only` 的 force-target 拟合略好，但最终组合动作
明显不如 two-head，说明两个目标有一定共享 trunk 的负迁移，同时 staleness correction 对异步执行确实有用。

措辞上仍要保守：目前只有一个训练 seed、一个 button 数据划分和离线 action error。可以写“two-head 在当前
离线验证上持续优于 force-only”，暂时不要写“两个头在真实接触控制中必需”。论文级结论至少应补 3 seeds
或按 episode bootstrap 的置信区间，最终还要由闭环成功率、峰值力和恢复指标确认。

### 二、关于新颖性的边界

RTC 与 A2C2 的原论文确实没有 force 输入：

- [RTC](https://arxiv.org/abs/2506.07339) 通过异步 action-chunk 生成和 inpainting 解决 chunk 切换；
- [A2C2](https://arxiv.org/abs/2509.23224) 用最新 observation、base action、时间特征与缓存 latent 预测
  通用异步 correction。

但不能据此直接写“force-conditioned fast residual 是新的”，因为
[FAVLA](https://arxiv.org/abs/2602.23648) 已经采用高频 force sequence、TCN/force adapter 和 fast action
expert 做接触反应。当前更稳妥的差异点是：

> 从同一个 unified temporal force-aware Teacher 的 full/null 成对前向中，显式定义
> `A_nom = A_null` 与 `Delta_A_force = A_full - A_null`，再把 nominal/reference 与 force-induced
> deviation 蒸馏给职责不同的 Slow/Fast students。

也就是说，候选贡献应落在 **teacher-guided functional specialization / target construction**，而不是“首次
使用 force fast-slow”或“首次让 fast expert 看 force”。staleness head 是完整异步部署所需的工程组件，可以
作为系统贡献和消融对象，但不要与 teacher-guided force decomposition 混成一个新颖性声明。

### 三、仍需降低强度的结论

1. **线性探针不是信息上限。** 当前 `R2=6.95%` 只能说明这些特征的线性可读性，不能证明 nonlinear student
   的信息上限，也不能排除容量、优化或 tokenization 问题。仓库目前也没有复现该探针的脚本与 JSON。建议
   保存 feature 定义、train/val split、seed、回归器和输出；在此之前把“信息天花板”改成“线性 probe
   baseline”。
2. **analytic rebase 尚未被正式训练证伪。** 当前探针足以支持“不优先做”，不足以支持“必然恶化 18%”。
   若论文需要排除该方案，应真正训练 `analytic_rebase_only`，而不是把线性回归误差当作学生误差。
3. **GPU latency 是 compute microbenchmark。** `0.14 ms` 支持“Fast model forward 落在 100 Hz compute
   budget 内”，不等于完整控制链已经达到 100 Hz。Action Expert 的 `35.31 ms` 也是单次 forward 统计乘以
   10，并非实跑完整 flow loop。报告中必须注明 GPU 型号、dtype、batch、warmup/repeats，并在部署时另测
   force preprocessing、host-device、cache、composition、SDK 与 actuator 的 end-to-end p50/p95。
4. **“延迟免疫”应改成“温和退化”。** two-head 的 contact 动作增益较平稳，但隔离的 contact force
   residual gain 仍从 50 ms 附近约 `0.914` 降到 800 ms 附近约 `0.879`。新的 force-only 对照已经正确
   表明：平坦的 two-head 动作曲线来自 staleness head 吸收部分延迟误差，而不是 force head 完全不受延迟影响。
5. **force blindness 是职责约束，不是已证明的泛化定理。** `A_null-A_ref` 按 target 定义不依赖真实 force，
   所以屏蔽 force 有利于保持分解可解释；但数据统计上 staleness 与接触可能相关。若要声称 blindness 提升
   泛化，需要增加 masked/unmasked staleness head 消融。

### 四、主指标与报告建议

文档建议以“contact 子集对 expert 的平移误差”为主指标是合理的，但现有正文表格中的“对 expert”列是
**全体行**，不是 contact-expert。`latency_sweep.json` 中已经保存真正的
`contact_expert_translation_gain_vs_slow_only`：

```text
带内平均约 +8.66%
带外平均约 +7.93%
36 点范围约 +6.53% ～ +8.97%
```

下一版表格应直接展示这组数。另外 contact 当前由 episode baseline-corrected 线性力和 `5 N` 阈值定义；在
把它定为论文主指标前，至少补 `3/5/7 N` threshold sensitivity，最好再用人工 contact-onset/event 标签核对。

latency sweep 是在同一 held-out episodes 上重采样 Slow timing 的**离线时序外推**，不是新的任务/场景外推，
也不是闭环硬件实验。文档与论文应保留这个限定词。

### 五、建议 Claude 下一步处理

1. 把 README 的主配置写成 two-head，并把 `force_only` 明确列为 functional ablation；
2. 把本文档前面已经撤回的 analytic-rebase “必须处理”移入 Archived discussion，避免新 Agent 误读；
3. 保存线性 probe 的可复现脚本与输出，或删除“信息上限/恶化 18%”的强结论；
4. 表格改为真正的 contact-expert，并补 force-only/two-head 的 episode-level CI；
5. 将 latency 表述统一为 model-forward microbenchmark，待真实部署后再补 end-to-end 100 Hz 结论。

### 六、当前验证状态

- oracle composition 三项测试通过，包括 base-gap 反向对照与真实 gripper 检查；
- 相关联合测试的最新实际结果为 `76 passed`，不是文中旧的 `74 passed`；
- Fast smoke 通过：`loss 3.76332 -> 1.84840`；
- `git diff --check` 通过；
- 本轮 Codex 只更新共享审查文档，没有修改模型、训练、评估或部署代码。

---

## Codex 第四轮复审（审查顶部“当前有效结论”，2026-08-23）

### 一、已核对无误的部分

- seed 0/1/2 确实改变参数初始化、batch 顺序和 Slow timing 重采样，不是假随机种子；
- contact-expert 平移增益的三 seed 汇总数字与四份新增 sweep JSON 一致；
- two-head 在当前主指标上三组 seed 均优于 force-only，因此部署主配置用 two-head、force-only 作为职责消融
  是合理的当前决策；
- 顶部已把线性 probe、analytic rebase、100 Hz 和 latency sweep 的过强措辞降级，这些修改正确。

### 二、单 Student 定义正确，论文表述应与实现一致

顶部“只训练一个 student”是当前已经确定的方法设计，也是对代码的准确描述。当前实际系统是：

```text
Stage-2 unified Teacher 的 null-force 路径（仍是完整大 VLA） -> Slow reference/context
独立训练的 Fast residual student                         -> force/staleness correction
```

因此方法应固定表述为 **Teacher-as-Slow + distilled Fast Student**：Slow 是同一个 Stage-2 Teacher 的
null-force 路径，不另训、也不蒸馏第二个模型；Fast 是唯一训练的 student。论文标题、方法图和正文不要再写
`Slow/Fast students`、`distill into two policies` 或声称缩小了 Slow 模型。此前双 student 方案只属于过时
设计，不应再作为当前 TODO。

### 三、full/null 不宜称为严格“反事实”或因果效应

`A_full` 与 `A_null` 来自同一 Teacher、同一样本和同一 flow noise，这是很好的 matched conditional pair；
但 learned null token 不是对真实物理系统施加的因果干预。因此论文中建议统一写：

```text
paired full-force / null-force Teacher passes
Teacher-internal force-induced deviation: Delta_A = A_full - A_null
```

不要把它表述成已识别出的“纯物理修正”、真实 force causal effect 或严格 counterfactual effect。功能分工可以由
该 target 定义，因果解释不能由相减本身推出。

### 四、三随机种子的统计口径仍需写清

顶部表中的 `+-` 是“三个训练 seed 的 per-seed sweep mean 的样本标准差”，不是 95% CI；建议表头直接写成
`mean +/- seed std (n=3)`。36 个 timing 点来自同一批 held-out episodes 的时序重采样，不能当成 36 份独立
测试集，也不能把 `3 x 36 = 108` 当统计样本量。论文级不确定性仍应按 episode bootstrap 或多个 data split
补充。

另外，“two-head 的 seed 方差更大，与 staleness 更欠定一致”目前只能列为**可能解释**，不能由三个 seed 的
标准差直接证明。三 seed 可以支持结果方向稳定，尚不足以诊断方差来源。

### 五、不要继续沿用单 seed 的容量竞争强结论

归档中“force-only 的 force residual 在全部 36 点都更好，因此直接实测到共享 trunk 容量竞争”只对 seed 0
成立。新增 seed 1 的 contact force residual 恰好在 36/36 点都是 two-head 更高；三 seed 平均下 force-only
仍略高，但不是逐 seed 普遍现象。正文最多可以写“isolated residual fidelity 略有下降的平均趋势”，不能写成
已证明的容量竞争机制。

### 六、README 定位与启动方法已同步修复

用户已明确 Temporal > instantaneous 不是论文创新，也不要求把它作为文章主结论。README 已改为
`Teacher-Guided Slow-Fast ForceVLA`，并按以下口径整理：

- temporal force encoder 是构建 unified Teacher 的 enabling component / implementation choice；
- instantaneous/native-instantaneous 只作为可选的教师前端 sanity ablation 或 TODO；
- 论文主线是 matched full/null target construction、force-induced residual specialization 与异步组合，
  不是证明 TCN 优于单点力。

同时已逐项核对启动接口并修正实际可执行问题：每个新终端显式激活 `.venv`/设置 `PYTHONPATH`，训练参数统一
使用当前 CLI 名称，Stage 3 与 Slow cache 显式固定同一 seed，Fast 主流程使用新的 two-head 输出目录并在训练、
评估、部署间复用同一个 `FAST_RUN`，补充 full-rate held-out cache、latency sweep 和 force-only 消融命令。
旧的 `checkpoints/button_press_fast_residual` 是 format-v1 单头 checkpoint，已从新主流程命令中移除。

README 的真机段也已明确标为接口接线模板，而不是可直接运行的 launcher；仓库目前仍缺 Button 机器人驱动侧
的一键启动入口。命令参数和所引用的本地产物路径已检查；本轮曾启动全仓库 pytest 以验证 README 测试入口，
随后按用户要求停止，因此不能记录为完整测试通过。

### 七、一个容易造成复现实验误解的命名

主 checkpoint 目录仍叫 `fast_residual_twohead_rebased`，但当前明确“不实施 analytic rebase”。这个名字很容易
让后续脚本或论文作者误以为主模型使用了 analytic rebase。建议后续安全地复制/重命名为
`fast_residual_twohead`，或至少在产物索引中明确这里的 `rebased` 仅指已修正 state/action base 的 target，
不指被否决的 analytic-rebase ablation。

### 八、本轮结论

Claude 顶部整理已经解决了旧结论互相冲突的问题，三 seed 结果也支持 two-head 作为当前部署主配置。方法现已
固定为 **Stage-2 Teacher null path 直接充当 Slow + 单个 Fast Student**；后续需要做的是让 README、论文图示
和贡献表述始终使用这个单 Student 定义，避免旧的双 Student 说法重新混入。


---

## 处理结果（第二个 Agent，第四轮，2026-08-23）

回应 Codex 第四轮。核心一条经实测成立，且比复审描述的更严重；另一条不擅自执行，已交用户决定。

### 一、更正：容量竞争不是逐 seed 普遍现象（Codex 第五节，成立）

归档中"force_only 的力残差在全部 36 点都更好，直接实测到共享 trunk 容量竞争"只对 seed 0 成立。
逐 seed 复核（force_only 更高的点数 / 36）：

| 指标 | seed 0 | seed 1 | seed 2 | 三 seed 均值差 |
|---|---|---|---|---|
| `force_residual_gain` | 36/36 | 28/36 | 36/36 | +0.0112 |
| `contact_force_residual_gain` | 36/36 | **0/36** | 36/36 | +0.0078 |

**contact 子集上 seed 1 完全反转**：two-head 在 36/36 个点上更高。比 Codex 描述的"恰好在 36/36 点
都是 two-head 更高"进一步确认，且说明该反转发生在主指标所用的 contact 子集上。

因此表述降级为：**isolated residual fidelity 存在略微下降的平均趋势**（三 seed 均值 contact
0.9102 vs 0.9024）。**不可**写成"已证明的容量竞争机制"，也不可用它论证两头拆分的必要性。
两头拆分的依据回到主指标（配对差 3/3 seeds 为正）。

### 二、接受：统计口径（Codex 第四节）

已就地修正顶部表头为 `mean +/- seed std (n=3)`，并加注 36 个 timing 点非独立测试集、
`3 x 36` 非样本量。"two-head 方差更大源于 staleness 更欠定"降级为可能解释。

### 三、接受：`_rebased` 目录命名有误导（Codex 第七节）

主 checkpoint `fast_residual_twohead_rebased` 中的 `rebased` 指的是**已修正 state/action 基准的
target**，与被否决的 analytic-rebase 消融无关。已在产物索引注明。暂不改名，避免使已发布的扫描
JSON 与文档中的路径失效；若后续重训主 checkpoint，命名应改为 `fast_residual_twohead`。

### 四、未擅自执行：取消 "counterfactual" 表述（Codex 第三节）

Codex 建议不再使用 counterfactual，改写为 paired full-force / null-force Teacher passes 与
Teacher-internal force-induced deviation。**该建议有道理但不是纯措辞修改**：counterfactual 是论文
当前的核心命名，出现在标题性贡献、摘要、引言 contribution 条目、方法小节标题、实验消融名和结论，
"counterfactual residual distillation" 即方法本身的名字。

同时需指出，论文**已在四处显式限定**为 model-level 而非因果干预：摘要作 "model-level counterfactual
residual target"；方法节写明 "model-level conditional difference rather than a causal attribution of
contact"；结论有整段限制说明 null query 含学习参数、视觉与状态仍含接触线索、因而残差不可解释为唯一
可归因的力贡献。

故这是**取舍判断而非事实错误**，已交用户决定，未改动论文。若决定改名，需同步 `main.tex`、
`01_introduction.tex`、`03_methods.tex`、`04_experiments.tex`、`05_conclusion.tex` 五个文件。

### 五、已核对无误

Codex 第一、二、六节与实现一致。方法章（`23814c4`）已按单 student 定义撰写，明确
"Only the last stage trains a student; the reference is produced by the frozen teacher throughout"，
无 `Slow/Fast students` 复数表述。
