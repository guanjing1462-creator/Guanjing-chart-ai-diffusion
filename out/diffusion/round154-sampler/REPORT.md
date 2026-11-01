# 第 154 轮：S1 采样器（多尺度全分辨率 + mel 通量条件）端到端验收

> **一句话结论**：新采样器 `tools/sample-diffusion32align.py` 已能把 `r153-S1-multi.pt`
> 的 `cond_net.projs.*` **逐键严格加载**（删任一键即报错，strict=False 不可用），并用
> `--cond-mel` 喂 32 带 mel onset 通量条件跑通「音频 → .osu」。端到端证据显示
> **模型确实在使用条件**（平移条件让输出明显改变：与配对输出 IoU 0.48–0.73、
> 概率相关 0.73–0.91；平移后「对给定条件」的相关高于「对真条件」），
> 但 **Heart Fraction 上的跟随幅度弱**（r250 池化 0.026–0.096），
> **±1 格输出相关没有稳定下降** → 只能说「采音生效、相位分辨率在机制上是 1 格」，
> 还不能说「整曲级 ±1 格跟随已达标」。

---

## 1. 改动清单（旧文件一字未改）

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `tools/sample-diffusion32align.py` | **新增**（从 `sample-diffusion32c.py` 复制） | 内联 `MultiScaleCondNet`；`inject=multi` 四层注入；元数据驱动构模型；`--cond-mel`；严格加载 |
| `tools/make-mel-cond.py` | **新增** | 单首歌 → 32 带 mel onset 通量条件（与 `build-mel-cond.py` 同一套函数） |
| `tools/check-align-sampler.py` | **新增** | 端到端验收：占位率、到最近强起音距离、250/1000ms 密度相关、平移对照 |
| `tools/sample-diffusion32c.py`、`tools/train-diffusion32-align.py`、`tools/build-mel-cond.py`、`tools/paint-gj.py`、旧 npz | 未改 | 采样器是复制后改副本；mel 口径靠 import 复用，不重写 |

架构一致性（与 `train-diffusion32-align.py` 逐字段）：
`net` = Conv1d(cin→32,k3) → SiLU → Conv1d(32→32,k3) → SiLU，
`projs[i]` = 1×1 Conv(32→level_ch[i])，`level_pool=(1,2,4,4)`、`level_ch=(base,2base,2base,2base)`
= `(48,96,96,96)`；`forward` 里 `e0+=c0, e1+=c1, e2+=c2, m+=c3`。
实测总参数 **456,820**，与 checkpoint 元数据 `params` 完全相同。

## 2. 严格加载的校验方式

1. 结构推断**只从权重形状 + 元数据**：`cond_net.projs.*` 存在 ⇒ `inject='multi'`，
   `cond_dim` 取 projs 的输入通道、`cond_channels` 取 `net.0` 的输入通道、`projs` 输出通道
   必须等于 `(base,2base,2base,2base)`。
2. 元数据交叉校验（任一不符即退出）：`inject`（必须 == 推断值，S1 必须是 `multi`）、
   `cond_dim / cond_channels / cond_stride`、`inject_scales`（multi 必须是 `[256,128,64,64]`）、
   `rows=256 / grid=1/32`、`pred`；**`cond_key` 必须存在且等于 `--cond-key`（默认 `M`）**。
3. **逐键比对**：先算 `模型键集合 − checkpoint 键集合`，缺键/多键当场报错并列出键名，
   再 `load_state_dict(strict=True)`。不用 strict=False。
   实测（故意删 `cond_net.projs.3.bias`）：
   `权重键不匹配，拒绝 strict=False 静默丢条件分支。缺键 ['cond_net.projs.3.bias']`；
   用 `--cond-key C` → `checkpoint 的 cond_key='M' 与给定条件 'C' 不一致`。
4. 条件路径按 `cond_channels` 分派：32 ⇒ 只收 `--cond-mel`（给 `--cond` 直接报错）；
   4 ⇒ 只收 `--cond`（4 带 `feat`，形状 `[T,4]` 严格校验）；两条路径都不补零、不静默重复。
   同代码也支持 `inject=mid4`（老 `AudioCondNet`）：实测 `r153-S4-mid4.pt` 以
   `mid4 / cond_ch=32 / stride=4` 严格加载通过。仓库里更老的 4 带 `diff-cond-*.pt`
   缺少 `freq` 缓冲区，与旧采样器一样**拒绝加载**（不允许随机补 Fourier 基底）。

## 3. mel 条件生成口径（与 `tools/build-mel-cond.py` 同代码）

`load_mono`（`build-audio-cond-fixed.py`，22050 单声道）→
`melspectrogram(n_mels=32, n_fft=1024, hop_length=256, sr=22050, fmin=0, fmax=8000, center=True)`
→ `log1p` → **正向时间差分 `max(Δ,0)`（onset 通量，不是能量）** →
按 1/32 格（`60000/bpm/8` ms，起点 `--offset`，默认 0 = 第一条红线的位置）取格内**均值** →
逐通道 P99 归一到 [0,1]。`tools/make-mel-cond.py` 里 `onset_flux / cell_mean` 是与
`build-mel-cond.py` 逐行相同的实现，`make-mel-cond` 只是把「多谱数据集」换成「单首歌」。

Heart Fraction（136.3 BPM，196.4s）产出：

| 项 | 值 |
| --- | --- |
| 1/32 格 | 55.03 ms |
| 格数 | 3570（有效 3570） |
| 条件形状 | `[3570,32]` float16，均值 0.1204，>0 占 92.4% |
| P99 尺度中位 | 0.2524 |
| 自证 | 条件总强度 vs `onset_strength` 包络 r=**+0.373**（同格聚合后仍同向）；lag1 r=+0.553 |

（对照：训练集 `M` 均值 0.1687 / >0 0.859；本曲条件略弱但同量级。）

## 4. 端到端生成（6 窗 × 256 格 = 1536 格 = 84.5s，seed 0，style 0）

`.osu`：`tools/roll-to-osu32.py <6 个 npy> --concat --bpm 136.3`（从 0 起连续排，
与条件的格起点一致）。统计脚本：`tools/check-align-sampler.py`（强起音 =
`onset_strength` + `peak_pick(paint-gj 同参)` 后取强度前 30%，全曲 353 个 = 1.80/s）。

### 4.1 `--diff 0.35`（占位率偏高那一档）

| 运行 | 占位率 | 音符 | 起音距离中位 | r250(真C) 池化 | r250(给定C) 池化 | r1000(真C) | Koishi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 配对 | 0.1683 | 1034 | 157.4 ms | +0.096 | +0.096 | +0.087 | 16.19 Emik Zeta，mid/high jack |
| +1 格 | 0.1602 | 984 | 155.5 ms | +0.078 | +0.094 | +0.098 | 15.79，mid/low jack |
| +4 格 | 0.1725 | 1060 | 166.5 ms | +0.045 | +0.045 | +0.052 | 16.30 Emik Zeta，high jack |

### 4.2 `--diff 0.15`（本报告主结果，占位率更贴近可玩谱面）

| 运行 | 占位率 | 音符 | 起音距离中位 | r250(真C) 池化 | r250(给定C) 池化 | r1000(真C) | Koishi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 配对 | 0.0871 | 535 | 149.9 ms | +0.026 | +0.026 | −0.021 | diff 无值（`< Alpha Low`，speed） |
| +1 格 | 0.0895 | 550 | 140.6 ms | +0.061 | +0.084 | +0.048 | 同上 |
| +4 格 | 0.0840 | 516 | 169.2 ms | −0.027 | −0.020 | −0.057 | 同上 |

平移对照（同一 seed，只换条件）：+1 格与配对输出 IoU **0.480**、概率相关 **0.735**；
+4 格 IoU **0.647**、概率相关 **0.888**。即条件平移**确实改变了输出**（不是随机权重那种
“完全变”，也不是条件无效那种“完全不变”）。

### 4.3 域内单窗对照（补充证据，定位机制）

一窗训练集 val 窗口（`dataset.npz` 第 2303 窗，BPM 263.9，`--diff 0.06` 以对齐该窗占位率 0.093），
条件分别取原样 / +1 格 / +4 格，同一 seed：

| 条件 | 占位率 | r250 vs 真条件 | r1000 vs 真条件 |
| --- | --- | --- | --- |
| 原样 | 0.0605 | **+0.532** | **+0.505** |
| +1 格 | 0.0576 | +0.389 | +0.281 |
| +4 格 | 0.0615 | +0.595 | +0.734 |

**同一套代码、同一权重**在域内单窗上 r250 = 0.53（远高于本轮 Heart Fraction 的 0.03–0.10），
说明采样器喂条件的方式没问题；+1 格退化（0.532→0.389，r1000 0.505→0.281）而 +4 格不退化，
与本轮 153 的「分辨率边界在 1 格」一致（单窗样本太薄，只能作为方向性证据）。

## 5. 结论：S1 是否在用条件？

**是，但幅度不足以直接出谱。** 支持「在用」的证据：
1. 加载与注入结构逐字段正确（参数 456820 对上，`projs.*` 严格加载）；
2. 换条件（±1/±4 格）输出显著改变（IoU 0.48 / 0.65，概率相关 0.73 / 0.89）；
3. 条件平移后「对给定条件」的相关高于「对真条件」（+1 格 0.084 vs 0.061；+4 格概率相关同向），
   即输出跟着被喂进去的条件走，不是不随条件变化的噪声；
4. 域内单窗 r250 0.53，且 +1 格退化。

不能宣称的部分：
- Heart Fraction 的 6 窗端到端 r250 池化只有 0.026–0.096，±1 格**没有**稳定下降
  （0.026→0.061→−0.027），因此本轮**不能**在整曲域上复现「±1 格明显下降」；
- 原因是多方面的：该曲不在 14 首训练歌里（域外）、只采了 6 窗、`--diff` 没做标定
  （0.35 时占位 0.168 太密，稀释跟随；0.15 时占位 0.087，Koishi 判 `< Alpha Low` 不给难度值）、
  强起音 1.8/s 太稀，±1 格 = 55ms 远小于到强起音的中位距离 150ms，距离指标本身分辨不了 1 格。

## 6. 剩余限制

1. **只有 6 窗 / 84.5s**：没有整曲拼接，没有 `--offset` 对齐到该曲真实红线（本轮按 0 起），
   也没有段落一致性检查。
2. **单一 seed / style / diff**：没有做多 seed 方差，也没有做 diff→占位率的标定曲线；
   0.15 档的 Koishi 难度落在 `< Alpha Low`，说明该档还太稀，不能算「可玩」。
3. **域外泛化**：训练只用 14 首，Heart Fraction 的 mel 通量分布均值 0.120 vs 训练集 0.169，
   跟随幅度掉了一个量级；需要更多歌（或该曲）重训/微调才能谈整曲出谱。
4. **强起音定义**沿用 `paint-gj` 单路 `peak_pick` + 前 30%，1.80/s 偏稀；换多频带合并检测
   会改变「起音距离中位」的绝对值（口径敏感，横向比较只在同一脚本内有效）。
5. `--cond-mel` 只接受 `[256,32] / [N,256,32] / [T,32]`，不做重采样；BPM 必须与生成
   条件时一致（1/32 格时长不同会整曲相位错开）。

## 7. 复现命令

```bash
# 1) 生成条件（P99 归一，口径同 build-mel-cond.py）
python3 tools/make-mel-cond.py --audio "ref/multisong/Heart Fraction.ogg" \
    --bpm 136.3 --out out/diffusion/round154-sampler/Heart-Fraction-mel.npz

# 2) 采样 6 窗（--start-cell 0 = 从歌曲 0 格起，窗口 i 取 start+i*256 起）
python3 tools/sample-diffusion32align.py --ckpt out/models/r153-S1-multi.pt \
    --cond-mel out/diffusion/round154-sampler/Heart-Fraction-mel.npz \
    --num 6 --start-cell 0 --style 0 --diff 0.15 --seed 0 \
    --outdir out/diffusion/round154-sampler/d15-paired

# 3) 转 .osu + Koishi
python3 tools/roll-to-osu32.py out/diffusion/round154-sampler/d15-paired/roll-*.npy \
    --concat --bpm 136.3 -o out/diffusion/round154-sampler/d15-paired.osu
node tools/gj/batch-koishi.mjs out/diffusion/round154-sampler/d15-paired.osu

# 4) 验收统计
python3 tools/check-align-sampler.py --audio "ref/multisong/Heart Fraction.ogg" \
    --bpm 136.3 --true-cond out/diffusion/round154-sampler/Heart-Fraction-mel.npz \
    --run paired=out/diffusion/round154-sampler/d15-paired \
    --run shift1=out/diffusion/round154-sampler/d15-shift1 \
    --run shift4=out/diffusion/round154-sampler/d15-shift4 \
    --out out/diffusion/round154-sampler/acceptance-d15.json
```

产物：`Heart-Fraction-mel.npz`（条件）、`paired.osu` / `shift1.osu` / `shift4.osu`（diff 0.35）、
`d15-*.osu`（diff 0.15）、`acceptance.json` / `acceptance-d15.json`、
`d15-*/sample-meta-*.json`（含严格加载信息与条件定位）。
