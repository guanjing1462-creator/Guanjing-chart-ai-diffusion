# 4K 谱面扩散模型训练报告

**任务**：在纯 CPU（8 核 / 可用内存约 2GB / 无 GPU）上把 `tools/train-diffusion.py` 的一维 U-Net DDPM 训练到验证损失明显下降，产出可用权重。

**结论**：已完成。24 个 epoch、总用时 2153.6s（35.9 分钟），验证损失 **0.3582 → 0.2155（下降 39.8%）**，最优权重落在 `out/models/diffusion4k.pt`，已用 `tools/sample-diffusion.py` 实测可直接 strict 加载并出谱。

> 注意：数据文件 `out/diff-data/dataset.npz` 只读未改动；`tools/paint-gj.py` 未触碰。

---

## 1. 最终配置

| 项 | 值 |
| --- | --- |
| base（U-Net 基础通道） | **48** |
| batch size | **64** |
| timesteps（DDPM 步数） | **200**（线性 schedule，beta 1e-4→0.02） |
| 窗口长度 | **128 步 × 4 键**（1/16 网格二值钢琴卷，未降分辨率） |
| 参数量 | **0.407M（407,012）** |
| torch 线程数 | **1**（关键，见第 2 节） |
| 每 epoch 窗口数 | 6000（从中 5700 训练 / 300 验证，按 `--seed 0` 随机抽取） |
| 优化器 | AdamW，lr 2e-4，weight_decay 0.01，余弦退火到 0（T_max=24） |
| 梯度裁剪 | 1.0 |
| 预测目标 | eps（噪声），loss = MSE |
| 验证集 | 300 窗口，**固定 t 与噪声**（同一把尺子量，曲线可比） |

**复现命令**：

```bash
cd /root/work/yamero-ai/proj
python3 tools/train-diffusion.py \
  --data out/diff-data/dataset.npz \
  --max-windows 6000 --batch 64 --base 48 --timesteps 200 \
  --threads 1 --lr 2e-4 --val-frac 0.05 --seed 0 \
  --epochs 24 --out out/models/diffusion4k.pt \
  --log out/diffusion/train-log.jsonl
```

---

## 2. 测速与精简决策

### 2.1 最重要的发现：CPU 被抢占时，线程越少越快

机器上同时跑着多个子智能体，8 核被瓜分。同一模型、同 batch=64，`torch.set_num_threads(N)` 实测每 step 耗时：

| 线程数 | base=32 | base=48 | 相对 NT=1 |
| --- | --- | --- | --- |
| 1 | **617 ms** | **955 ms** | 1.0× |
| 2 | 833 ms | 1180 ms | 1.4× |
| 4 | 1370 ms | 1339 ms | 1.9× |
| 8 | 2197 ms | 2802 ms | **3.2×** |

**8 线程比单线程慢 3.2 倍**——不是算力不够，是线程互相打断（上下文切换 + 缓存抖动）。把线程降到 1 就白拿了 3 倍加速，比改模型结构收益大得多。

### 2.2 原始配置为什么跑不完

任务给定：30551 个窗口跑 1 epoch 在 900s 内跑不完。按上面的实测（base=48、NT=8 约 2.8s/step），477 batch 需要约 1330s，确实超。

### 2.3 取舍

| 手段 | 是否采用 | 理由 |
| --- | --- | --- |
| `torch.set_num_threads(1)` | **采用** | 3.2× 加速，零代价 |
| 每 epoch 用 6000 窗口（而非全集 30551） | **采用** | 一个 epoch 从 ~1330s 降到 ~90s；批内梯度仍来自 5700 个不同窗口，统计上够用 |
| 降 base 到 32 | 未采用 | 只省约 30% 单步时间，但参数从 0.41M 掉到 0.21M，且线程优化后已不缺时间 |
| 减少层数 / 输入降到 64 行 | 未采用 | **会破坏下游**：`tools/sample-diffusion.py` 里复制了一份 `UNet1D` 来加载权重，改结构会导致 strict load 失败 |
| timesteps 200 → 100 | 未采用 | **本代码里 timesteps 对训练算力零影响**（每 batch 仍只做一次前向）。而 100 步时 abar_T = exp(-1.005) ≈ 0.366，末端仍残留 37% 信号，采样从纯噪声起步会跑出训练分布；200 步 abar_T ≈ 0.132 更合理，故保留 200 |
| 混合精度 | 未采用 | CPU 上收益有限，如题所述 |

---

## 3. 损失曲线

`train` 为该 epoch 全部 batch 的加权平均 MSE；`val` 为 300 个验证窗口在**固定 t/噪声**下的 eps-MSE。

| epoch | train | val | 用时(s) |
| --- | --- | --- | --- |
| 1 | 0.4816 | 0.3582 | 99 |
| 2 | 0.3548 | 0.3407 | 96 |
| 3 | 0.3431 | 0.3341 | 95 |
| 4 | 0.3370 | 0.3249 | 94 |
| 5 | 0.3282 | 0.3155 | 138 |
| 6 | 0.3230 | 0.3064 | 203 |
| 7 | 0.3111 | 0.3017 | 101 |
| 8 | 0.2952 | 0.2830 | 101 |
| 9 | 0.2823 | 0.2722 | 93 |
| 10 | 0.2684 | 0.2641 | 82 |
| 11 | 0.2566 | 0.2537 | 81 |
| 12 | 0.2528 | 0.2461 | 83 |
| 13 | 0.2460 | 0.2397 | 81 |
| 14 | 0.2414 | 0.2337 | 82 |
| 15 | 0.2315 | 0.2286 | 82 |
| 16 | 0.2318 | 0.2250 | 84 |
| 17 | 0.2249 | 0.2221 | 81 |
| 18 | 0.2262 | 0.2201 | 77 |
| 19 | 0.2168 | 0.2188 | 68 |
| 20 | 0.2239 | 0.2170 | 67 |
| 21 | 0.2161 | 0.2162 | 66 |
| 22 | 0.2158 | 0.2158 | 66 |
| 23 | 0.2183 | 0.2156 | 66 |
| 24 | 0.2147 | **0.2155** | 66 |

训练/验证损失几乎重合（无过拟合迹象：6000 窗口 × 24 轮、0.41M 参数，容量与数据量匹配）。

**参考基线**：eps 预测任务里"永远输出 0"的平凡预测器 MSE 恰为 1.0。验证损失 0.2155 ≈ 解释了 78.5% 的噪声方差。

---

## 4. 单 epoch 耗时

| 指标 | 值 |
| --- | --- |
| 每 epoch batch 数 | 90（5700 / 64） |
| 平均单 epoch | **89.7 s** |
| 中位数 | **82 s** |
| 最快 / 最慢 | 66 s / **203 s**（epoch 6） |
| 稳态单 batch | 约 0.73–1.1 s |
| ≤180s 的 epoch | 23 / 24 |

**关于 epoch 6 的 203s**：那一轮首个 batch 耗时 9723ms（正常 700–1500ms），是别的子智能体在同一时刻抢 CPU 的突发抢占，不是配置问题；前后轮次都在 101s 以内。稳态下 1 epoch 约 66–100s，满足 ≤180s 的要求。

---

## 5. 收敛判断

- **仍在下降，但已进入平台**：
  - epoch 1→18：0.3582 → 0.2201（降 38.6%，主要收益在这里）；
  - epoch 18→24：0.2201 → 0.2155（降 2.1%）。
  - 前 6 轮降幅 0.0565，最后 6 轮降幅仅 0.0045，δ 已收敛到噪声中。
- **注意**：后半段平坦有相当一部分是**余弦退火把 lr 降到 0** 的结果（epoch 19 起 lr 已低于 2.1e-5，epoch 22 起低于 3e-6），不是模型到顶。
- 训练损失与验证损失同步下降且几乎贴合，未见震荡、发散或过拟合。
- 判断：**收敛正常**。若给更长预算（同配置 60–90 轮，或放开 lr 不提前退火），验证损失有望继续下探，但边际收益递减。

---

## 6. 权重文件

**路径**：`out/models/diffusion4k.pt`（1.65 MB，epoch 24，val 0.2155）

加载方式（`torch.load` 返回 dict）：

```python
import torch
ck = torch.load('out/models/diffusion4k.pt', map_location='cpu', weights_only=False)
# ck 结构：
#   'state'          : dict  模型 state_dict（76 个张量，407,012 参数）
#   'base'           : int   48      U-Net 基础通道
#   'temb_dim'       : int   128     时间嵌入维度
#   'timesteps'      : int   200     DDPM 步数（必须与采样 schedule 一致）
#   'val'            : float 0.21554 该权重对应的验证损失
#   'epoch'          : int   24
#   'batch'          : int   64
#   'rows'           : int   128     窗口长度
#   'windows'        : int   6000    训练所用窗口数
#   'params'         : int   407012
#   'train_seconds'  : float 2155.1  累计训练秒数
```

模型必须用 `UNet1D(cin=4, base=ck['base'], temb_dim=ck['temb_dim'])` 重建后 `load_state_dict(ck['state'])`。

**端到端验证**（用成品权重实跑）：

```bash
python3 tools/sample-diffusion.py --ckpt out/models/diffusion4k.pt \
  --num 2 --threads 1 --quiet --outdir /tmp/final-samp
```

结果：strict 加载成功（无缺键/多键警告），2 个窗口采出占位率 0.106、每窗口 71/38 个音符、约 3.9 nps、单窗口 4.95s（T=200），产物 `rolls-s0d1-seed0.npz` / `sample-meta-s0d1-seed0.json` 正常落盘。

---

## 7. 对 `tools/train-diffusion.py` 的改动

只改了训练脚本，模型结构与默认行为保持向后兼容：

1. **新增 `--threads`（默认 1）**：设置 `torch.set_num_threads` 与 interop 线程数。这是本次提速的主要来源。
2. **修复验证损失加权 bug**：原代码用 `sl.stop - sl.start` 当末块真实长度，当 `len(te) < 512` 时会把验证损失放大 `512/len(te)` 倍（实测显示成 11.92，真值 0.745）。已改为按真实块长加权。
3. **验证集固定 t 与噪声（`--val-fixed`，默认开）**：原来每轮重新抽 t 和噪声，验证损失抖动到看不出趋势。
4. **子集改为按 `--seed` 随机抽取**：原 `X[:n]` 取前 N 个窗口，会按歌曲顺序偏置风格/难度分布。
5. **新增 `--val-frac` / `--seed` / `--min-epochs` / `--time-budget` / `--log`**：时间预算避免超时；`--log` 逐 epoch 追加 jsonl。
6. **checkpoint 增补元信息**：`temb_dim / epoch / batch / rows / windows / params / train_seconds`，`out/models/diffusion4k.pt` 现在自解释。原有 `state / base / timesteps / val` 键保持不动，`sample-diffusion.py` 无需修改。

产物：
- `out/models/diffusion4k.pt` — 最优权重
- `out/diffusion/train-log.jsonl` — 逐 epoch 结构化日志
- `out/diffusion/train.log` — 原始 stdout
- `out/diffusion/REPORT.md` — 本文件

---

## 8. 下一步建议

1. **要更好的模型**：同样配置跑 60–90 轮（约 1.5–2 小时），并把余弦退火展开到实际轮数，别让 lr 提前归零；这是当前唯一确定的收益来源。
2. **要更快**：任何后续 torch 任务一律 `set_num_threads(1)`（本次实测 3.2× 加速），比换模型结构划算。
3. **要注意**：loss 下降不等于出谱质量提升（仓库里 `tools/eval-diffusion.sh` 的注释就记录过 loss 0.108→0.10 而踩音率没动）。下一步应把 `diffusion4k.pt` 接进采样 + 踩音对齐评测，用踩音率而不是 loss 判断好坏。
