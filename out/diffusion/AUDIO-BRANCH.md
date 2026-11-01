# 音频条件分支（audio conditioning）接线与有效性验证

> 目标：把 Mug-Diffusion 那条「mel 频谱当主条件」的分支，在本仓自己的数据上做出等价物，
> 并**用实验证明这个条件真的能改变生成结果**。
>
> 结论先说：**分支接上了，条件有效**——固定噪声、只换条件，生成密度能因果地移动
> **+12~15%**（无条件模型完全不动）。但**验证损失只降 0.2%**，因为这个指标对条件不敏感；
> 而且条件带来的主要是「密度先验」，不是音符级的对齐。下面把数字和证据摆清楚。

---

## 0. 结论速览

| 项 | 结果 |
| --- | --- |
| 接口 | `--audio-cond <npz> --cond-dim 32`（训练）、`--cond <npz> --cond-index N`（采样） |
| 结构 | `[B,128,4] → Conv1d(4→32)×2 → [B,32,128] → Conv1d(32→96,k=4,s=4) → [B,96,32]`，**加到 U-Net 中间层** |
| 参数量 | 0.4070M → 0.4229M（+15,904，+3.9%） |
| 验证损失（3 seed 均值） | 无条件 0.29767 → 有条件 **0.29710**（−0.19%，3/3 seed 都更低） |
| 多噪声配对（10 组 × 3 seed） | 有条件更低的噪声组 **25/30**，平均差 +0.00069 |
| 条件有效性（因果干预） | 固定 `x_t`，条件从最弱换到最强：预测占位率 **0.3993→0.4464（+11.8%）**；无条件模型**逐位不变** |
| 采样（真 DDPM） | 同一种子换条件：**5.1% 的格翻面**；生成密度与真实密度的相关 **r=+0.452**（置换条件 r=+0.093） |
| 条件强度 → 密度 | 最弱 4 段条件 → 122 音符/窗口；最强 4 段 → **186 音符/窗口** |
| 向后兼容 | 旧权重 `diffusion4k*.pt` strict 加载；新旧采样脚本输出**逐位相同**；无 `--audio-cond` 时前向**逐位相同** |
| ⚠ 关键坑 | 条件分支的**投影层初始化**决定它学不学得动（零初始化 144 步内等于没接，见 §3.4） |

---

## 1. 接口

### 1.1 训练（默认关闭，不加参数与旧行为完全一致）

```bash
python3 tools/train-diffusion.py \
    --data out/diff-data/audio-cond-labels.npz \
    --audio-cond out/diff-data/cond-windows.npz \
    --cond-dim 32 \
    --epochs 12 --batch 16 --val-frac 0.1 --threads 1 \
    --seed 0 --init-seed 0 \
    --out out/models/diff-cond-seed0.pt \
    --log out/diffusion/logs/cond-seed0.jsonl
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--audio-cond PATH` | `''`（关） | 条件文件；`''` = 不建分支，模型结构与旧权重逐字节一致 |
| `--cond-dim N` | `32` | 逐位置特征的通道数（`--audio-cond` 时生效） |
| `--cond-drop p` | `0.0` | 训练时按概率把条件置零（classifier-free guidance 用） |
| `--cond-root DIR` | `out/multisong` | cond-windows 配对时单歌 `<歌名>-cond.npz` 所在目录 |
| `--init-seed N` | `-1` | `>=0` 时先 `torch.manual_seed(N)`；A/B 配对实验用（默认 `-1` = 不动全局 RNG，旧行为） |

`--audio-cond` 支持三种文件：

1. `{'cond': [N,128,4]}` —— 已按 `--data` 行序排好（`out/diff-data/audio-cond-labels.npz` 就带这个键）；
2. `{'X','song','start','songs'}` —— `cond-windows.npz` 格式，按 **(歌名, 起始格)** 与 `--data` 的行配对；
3. `{'feat': [T,4]}` —— 单首歌条件文件；训练会明确报错（训练需要整批对齐）。

**没有条件时自动喂零**：`--data` 换成没有音频的全量 `dataset.npz` 时，199 行的条件对不上
30551 行数据，脚本会打印警告并把条件全设为零（模型照常学无条件去噪）：

```
[警告] 条件窗口 199 行 ≠ 数据 30551 行，且 --data 没有 song/start 可配对
       → 全部用全零条件（等价于无条件训练，模型仍保留无条件能力）
  条件覆盖：0.0% 的窗口有条件（其余全零）；通道均值 [0.0, 0.0, 0.0, 0.0]
```

### 1.2 采样

```bash
# 单首歌的条件文件 + 起始格
python3 tools/sample-diffusion.py --ckpt out/models/diff-cond-seed0.pt \
    --cond out/multisong/Heart_Fraction-cond.npz --cond-index 896 --num 2 --threads 1

# cond-windows.npz + 窗口序号（自动找歌名与起始格）
python3 tools/sample-diffusion.py --ckpt out/models/diff-cond-seed0.pt \
    --cond out/diff-data/cond-windows.npz --cond-index 7 --num 2

# 强制零条件（和无条件采样对比）
python3 tools/sample-diffusion.py --ckpt out/models/diff-cond-seed0.pt --cond-zero --num 2
```

- `cond_dim` **从 ckpt 里读**：旧 ckpt 没有这个键 → `cond_dim=0` → 结构与以前完全一致；
- 带 `cond_dim>0` 的 ckpt 不给 `--cond` → 自动喂全零，打印提示；
- 给 `--cond` 但 ckpt 是 `cond_dim=0` → 明确报错退出（exit 2），不会静默忽略；
- 输出文件名带条件标记：`roll-s0d1-seed0-c896-00.npy`、`...-czero-00.npy`；`rolls-*.npz` 里多存 `C=[128,4]`。

### 1.3 评估

```bash
# 完整评估（验证损失 + 单步条件敏感 + 因果干预 + 真采样对比）
python3 tools/eval-audio-cond.py \
    --cond-ckpt out/models/diff-cond-seed0.pt \
    --nocond-ckpt out/models/diff-nocond-seed0.pt \
    --threads 1 --out out/diffusion/eval-audio-cond-seed0.json

# 一键复跑 3 seed 的 A/B 训练（本报告用的配置）
bash tools/ab-audio-cond.sh

# 给 199 个窗口补 style/diff 标签 + 预切条件
python3 tools/build-audio-cond-labels.py
```

---

## 2. 结构设计

### 2.1 形状流转

```
音频条件 [B,128,4]（每格 4 个 onset 强度：低频/中频/高频/总）
  └─ transpose → [B,4,128]
     └─ Conv1d(4→32, k=3, pad=1) → SiLU
        └─ Conv1d(32→32, k=3, pad=1) → SiLU      = 逐位置特征 [B,32,128]
           └─ Conv1d(32→96, k=4, stride=4)       ← 同时下采样 + 投影
              └─ [B,96,32]
                 └─ 加到 U-Net 中间层： m = mid(e2, te) + cond_mid      ← 通道相加
```

- U-Net 内部分辨率 `128 → 64 → 32`（两次 `MaxPool1d(2)`），所以中间层是 `[B, 2*base, 32] = [B,96,32]`，
  `kernel=stride=4` 的卷积正好把 128 格压到 32 格，通道对齐到 96；
- 加法注入（不是拼接）：形状不用改、后面所有 Block 不变、旧权重不受影响；
- `cond=None` 时喂全零：`[B,128,4]` 全零张量，模型保留无条件能力；
- 参数量：`4·32·3+32 + 32·32·3+32 + 96·32·4+96 = 15,904`，
  总参数 **0.4070M → 0.4229M（+3.9%）**。

### 2.2 为什么注入在中间层（1/4 分辨率）

- 条件的逐位置特征保留全分辨率 `[B,32,128]`，只在**注入点**降到 1/4（4 个格 = 1 拍），
  这样每拍的低/中/高频 onset 可以调节那一拍附近的写法密度；
- 代价：条件对 1/16 格的**精确**时间控制有限（见 §6.5 的负面结果）。

### 2.3 训练侧改动

- `Diffusion.loss(..., mode='eps', cond=None)` 增参，默认值保证旧调用不变；
- 训练循环：`cb = C[idx]`，`--cond-drop` 时按概率置零；`diff.loss(..., cond=cb)`；
- 验证：固定 t/噪声的 `eval_loss()` 也带上条件（同一批 val 窗口）；
- ckpt 新增元数据：`cond_dim / cond_channels / audio_cond / cond_drop / data / val_idx / val_frac / seed / init_seed`
  （`val_idx` 让评估脚本能用**完全相同的验证切分**，不用猜随机数）。

### 2.4 采样侧同步

`tools/sample-diffusion.py` 里那份 UNet1D 副本按同样的字段加了 `AudioCondNet` 与 `cond_dim`，
`load_model()` 优先读 ckpt 的 `cond_dim`，读不到再从 `cond_net.proj.weight` 的形状反推。
**副本与训练脚本的模块名逐字段一致**（`cond_net.net.*` / `cond_net.proj.*`），所以权重可以互相 strict 加载。

### 2.5 ⭐ 一个必须记住的坑：投影层初始化

第一版照 ControlNet 的习惯把 `proj` **零初始化**（想让条件分支从 0 开始长）。结果：

| `proj` 初始化 | 训练后 `proj` |w| | 条件贡献 / 中间层幅度 | 换条件时输出变化 | 因果干预（占位率 弱→强） |
| --- | --- | --- | --- | --- |
| 零初始化 | 0.0026 | 2.0% | 0.79% | 0.3989 → 0.4095（**+2.7%**） |
| 默认 `std≈0.051` | 0.0444 | 5.6% | 0.74% | 0.3991 → 0.4099（+2.7%） |
| **`std=0.3`（终版）** | 0.2383 | 21.8% | 3.62% | 0.3761 → 0.4231（**+12.5%**） |

只有 199 个窗口 / 12 轮 = **144 步**，Adam 每步最多把权重挪 ~lr；
零初始化的投影层在这点步数里根本长不起来，条件等于白插。
把 `proj` 的初始化标准差提到 `0.3`（初始贡献 ~28%）之后，条件才真的参与计算。
**这个差别在验证损失上几乎看不出来**（0.29714 vs 0.29710），但条件的作用差 5 倍 ——
所以只看 loss 会误判「分支没用」。

> 实现：`nn.init.normal_(self.proj.weight, std=0.3)`，bias 置零。
> 不新增参数名，所以零初始化/默认初始化训出来的 ckpt 也能被同一份采样代码 strict 加载。

---

## 3. 数据与配对

### 3.1 窗口与条件切片

`out/diff-data/cond-windows.npz`：`X [199,4,128] uint8`（钢琴卷）、`song int16`、`start int32`、`songs`（6 首）。

配对规则（`tools/build-audio-cond-labels.py` 与训练脚本同一套）：

```
cond[i] = feat[song[i]][ start[i] : start[i]+128 ]     # [128,4]，feat 来自 out/multisong/<歌名>-cond.npz
```

199 个窗口全部在范围内（`start+128 ≤ steps`，逐歌检查过，0 越界）。
`start` 是**逐首**的格号（不是拼接后的全局格号），6 首的 `steps` 分别是 5224/1782/1374/1466/2530/1236。

### 3.2 标签（已知局限）

这 6 首歌不在 `out/packs/{tech,jack,speed,stamina}/` 里，没有风格标签；`cond-windows.npz` 本身也没有 style/diff。
按 `tools/diff-data.py` 的规则补：

| 歌 | 行数 | 时长 | nps | diff 档 |
| --- | --- | --- | --- | --- |
| Frums_SuperMix | 6688 | 607.6s | 11.01 | 1 |
| Heart_Fraction | 2228 | 188.4s | 11.83 | 1 |
| Plum_FallenEra | 1674 | 163.2s | 10.26 | 1 |
| REDALiCE_x_t_pazolite___Xterfusion | 1871 | 165.3s | 11.32 | 1 |
| _______________Flash_Me_Back | 3145 | 304.7s | 10.32 | 1 |
| seatrus___purifikation | 1533 | 148.0s | 10.36 | 1 |

- **style 全 0**（没有标签）、**diff 全 1**（6 首 nps 都在 10~12）→
  两个嵌入在本次实验里都是**常量**，所以 A/B 的差异只可能来自音频条件（这反而让实验更干净）；
- 产物 `out/diff-data/audio-cond-labels.npz`：`X / style / diff / song / start / cond[199,128,4] / nps / songs`。

### 3.3 对齐自检

- 逐格 `corr(总 onset, 该格有音符)`：均值 **0.267**、中位 0.240、**88.2% 的窗口为正** → 配对的格号没有错位；
- 条件的**逐格**判别力：总 onset 分 5 箱，格内平均音符数 1.095 → 1.720 → 1.940 → 1.980 → 1.977（后三箱饱和）；
- 条件的**逐窗口**判别力：平均 onset vs 窗口占位率 **r=0.671**；
- 条件的**逐行**判别力：行剖面相关均值只有 **0.149** ——
  这正是「条件主要给密度、不给音符位置」的根源，后面所有结果都能对上。

### 3.4 条件信息量（决定上限）

用最简单的非神经预测器估上限（同一批验证窗口、同一把尺子）：

| 预测器 | eps-MSE |
| --- | --- |
| 全局平均谱面 | 5.558 |
| **用 cond 预测每窗密度** × 全局形状 | 5.333（−4%） |
| 真值（oracle） | 0 |
| 训练 12 轮的无条件模型 | 0.3017 |
| 训练 12 轮的条件模型 | 0.3016 |

模型从 `x_t` 本身就能恢复绝大部分结构（0.30 vs 常数预测 5.5），
条件能补的只有「这个窗口有多密」这一层先验（常数预测器上值 4%），
落到最终验证损失上只剩千分之几。**这不是分支没接上，是指标本性。**

---

## 4. A/B 实验设置

| 项 | 值 |
| --- | --- |
| 数据 | `out/diff-data/audio-cond-labels.npz`（199 窗口：179 训练 / 20 验证，`val-frac 0.1`） |
| 模型 | `base=48`、`T=200`、`pred=eps`、AdamW lr 2e-4 / wd 0.01、cosine |
| 训练 | `batch=16`、**12 轮**、`--threads 1`、每轮 12 个 batch（144 步） |
| 配对 | `--seed s --init-seed s` → A/B **共享同一套初始权重、同一 batch 顺序、同一 t/噪声流**；验证切分与固定验证噪声也完全相同 |
| A | 加 `--audio-cond cond-windows.npz --cond-dim 32`（0.4229M） |
| B | 同配置，不加 `--audio-cond`（0.4070M） |
| seed | 0 / 1 / 2（各 2 个模型，共 6 次训练） |
| 耗时 | 单次 42~98s（受机器负载影响），6 次合计约 **5.5 分钟**；含消融变体共约 13 分钟 |

> 说明：A/B 的初始权重相同（`--init-seed`），但 A 多了条件分支，所以从第 1 步起输出就不同 —— 这是必须的，
> 否则条件分支永远得不到梯度。

---

## 5. 损失对比

### 5.1 训练日志里的验证损失（固定 t/噪声，同一把尺子）

| seed | B 无条件 | A 零初始化 | A 默认初始化 | **A 终版（std=0.3）** |
| --- | --- | --- | --- | --- |
| 0 | 0.30165 | 0.30163 | 0.30163 | **0.30160** |
| 1 | 0.29151 | 0.29046 | 0.29068 | **0.29073** |
| 2 | 0.29984 | 0.29932 | 0.29962 | **0.29898** |
| 均值 | 0.29767 | 0.29714 | 0.29731 | **0.29710** |

差值（B − A 终版）：+0.00005 / +0.00078 / +0.00086，**3/3 seed 都是 A 更好，平均 +0.00056（−0.19%）**。

### 5.2 多噪声配对（更稳的估计）

每个 ckpt 在 20 个验证窗口上换 10 组固定 `(t, 噪声)`，配对比较：

| seed | A | B | 配对差（B−A） | A 更低的噪声组 |
| --- | --- | --- | --- | --- |
| 0 | 0.41892 | 0.41941 | +0.00049 ± 0.00070 | 6/10 |
| 1 | 0.40391 | 0.40430 | +0.00038 ± 0.00033 | 9/10 |
| 2 | 0.41863 | 0.41984 | +0.00121 ± 0.00075 | **10/10** |
| 合计 | — | — | **+0.00069** | **25/30** |

结论：**方向一致但幅度很小**（0.17%）。0.0007 的差距在单次验证上会被噪声淹没，
只有配对 + 多噪声才能看出来；seed 0 的 6/10 说明它并不稳健。

### 5.3 为什么损失几乎不动

1. 模型主要靠 `x_t` 恢复结构，条件只补「密度先验」这一小块（§3.4）；
2. 参数只看 199 个窗口 × 144 步，条件分支能学到的映射很粗；
3. 逐格 `x0` 预测的 F1 在单步反推里几乎是平的（匹配 0.7055 vs 置换 0.7044），
   说明条件没有改善**给定噪声谱面时的重建**，它改善的是**从纯噪声生成时该有多密**。

**所以：不能只用验证损失判断条件是否有效。**

---

## 6. 条件是否有效 —— 四组证据

### 6.1 因果干预（最硬的证据）

固定同一批 `x_t`（验证窗口 @ t=100，噪声固定），把所有窗口的条件**统一换成**最弱/最强的几段
（199 个窗口里条件均值最小/最大的 4 段取平均），看模型预测的 `x0` 占位率：

| seed | A：最弱条件 | A：最强条件 | 相对变化 | B（无条件，负对照） |
| --- | --- | --- | --- | --- |
| 0 | 0.3993 | 0.4464 | **+11.8%** | 0.4229 → 0.4229（差 0） |
| 1 | 0.3378 | 0.3884 | **+15.0%** | 0.3720 → 0.3720（差 0） |
| 2 | 0.3980 | 0.4581 | **+15.1%** | 0.4304 → 0.4304（差 0） |

- `x_t` 完全相同 → 差异**只可能**来自条件；
- 无条件模型的两次输出**逐位相同**（差值精确为 0）→ 排除「代码路径本身有随机性」；
- 另一个噪声抽样（`eval-audio-cond.py` [2b]）：0.3889 → 0.4351（+11.9%），结论一致。

### 6.2 真采样：换掉条件，生成谱面会变

8 个验证窗口、`T=200` DDPM、同一种子（同初始噪声），匹配条件 vs 置换条件（按条件强度排序后错开一半）：

| 指标 | A（有条件） | B（无条件，负对照） |
| --- | --- | --- |
| 两次生成平均 \|ΔP\| | 0.0542（最大 0.288） | **0.0** |
| 阈值化后翻面的格子 | **5.1%** | 0 |
| 生成密度 vs 真实密度 r | **+0.452** | — |
| 同样相关，置换条件 | +0.093 | — |
| 密度绝对误差 \|生成−真实\| | 0.1763（置换 0.1760） | — |
| 每窗口耗时 | ~5s | ~5s |

- **匹配条件的生成密度与真实密度的排序相关 +0.452，置换条件只有 +0.093** → 条件换对了，密度更像真实谱面；
- 但**绝对误差没改善**（0.1763 vs 0.1760）：模型整体**低估密度**
  （生成 0.22~0.36，真实 0.42~0.54），这是小样本 + 12 轮的欠拟合，不是条件的问题。

### 6.3 条件强度 → 生成密度（更直观）

从 199 个窗口里取条件均值最小/最大的各 4 段，各自配 4 个生成（同一初始噪声）：

| 组 | 条件均值 | 生成占位率 | 每窗口音符数 |
| --- | --- | --- | --- |
| 最弱 4 段 | 0.082 | 0.2393 | 130 / 99 / 123 / 138（均 122） |
| 最强 4 段 | 0.560 | 0.3633 | 193 / 165 / 187 / 199（均 186） |

**条件越强 → 生成越密（+52% 音符）**，噪声种子完全相同。

### 6.4 同一首歌的两个不同 start

`Heart_Fraction` 起始格 896（条件均值 0.079）vs 1408（0.550），同一种子各生成一次：

- 两次生成平均 `|ΔP| = 0.4999`（几乎完全不同）；
- 音符数 111 vs 133。

### 6.5 反面对照：条件**没有**做到的事

- 单步反推 `x0`（t=100）时，匹配条件的 F1 **0.7055** vs 置换 **0.7044**（20 个窗口里只有 7 个匹配更好）→
  条件**不改善**「已有噪声谱面时的重建」；
- 真采样里匹配 vs 置换的 F1 0.3455 vs 0.3451 → 音符级对齐没有可测改善；
- 与 §3.3 对得上：条件与音符的**逐行**相关只有 0.149，能给的只是密度先验。

**所以准确的结论是：条件对「生成密度」有明确、可复现、因果的作用；对「音符位置/重建精度」几乎没有帮助。**

---

## 7. 向后兼容验证

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 旧权重 strict 加载 + 采样 | `python3 tools/sample-diffusion.py --ckpt out/models/diffusion4k-x0.pt --num 1 --threads 1 --outdir /tmp/compat` | `加载方式: strict`、`cond_dim: 0`、`pred: x0`、12.03s/窗口 ✅ |
| 新旧采样脚本输出一致 | 同一 ckpt/seed 分别跑新脚本与备份脚本 | 阈值卷**逐位相同**、概率卷**逐位相同**（max\|ΔP\|=0） ✅ |
| 无 `--audio-cond` 时前向等价 | 旧模块 vs 新模块（`cond_dim=0`，权重拷贝后同一输入） | state_dict 键 76=76、参数 407012=407012、输出**逐位相同**、同种子 loss 完全相同 ✅ |
| 无音频数据集 | `--data dataset.npz --audio-cond cond-windows.npz` | 打印警告 → 条件全零 → 正常训练（val 0.6531@1 轮） ✅ |
| 旧权重 + `--cond` | `--ckpt out/models/diffusion4k.pt --cond ...` | 明确报错 exit 2，不静默忽略 ✅ |

`out/models/diffusion4k.pt` / `diffusion4k-x0.pt` **没有被改动**；`tools/paint-gj.py`、`out/diff-data/dataset.npz` 也没有动。

---

## 8. 代码改动清单

| 文件 | 改动 |
| --- | --- |
| `tools/train-diffusion.py` | 新增 `AudioCondNet`（L58）、`UNet1D(cond_dim=)` 与 `forward(..., cond=None)`（L94/L124）、`load_audio_cond` / `_cond_seg`（L162/L227）、5 个新参数（L262-273）、条件加载与零填充（L288-301）、训练循环带条件与 `--cond-drop`（L363）、验证带条件、ckpt 元数据（L389-399） |
| `tools/sample-diffusion.py` | 同步 `AudioCondNet` / `UNet1D(cond_dim=)`（L99/L131/L149/L161）、`ddpm_sample(..., cond=)`、`load_model` 读 `cond_dim`（L330-334）、`load_cond_segment`（L373）、`--cond/--cond-index/--cond-root/--cond-zero/--cond-dim` 参数与条件采样状态打印、输出名带条件标记 |
| `tools/build-audio-cond-labels.py`（新） | 给 199 窗口补 style/diff 标签、预切 `cond`、对齐自检 |
| `tools/eval-audio-cond.py`（新） | 多噪声验证损失 + 单步条件敏感 + 因果干预 + 真采样对比，输出 json |
| `tools/ab-audio-cond.sh`（新） | 3 seed × {A,B} 的一键复跑脚本 |
| `out/diffusion/_bak/*.orig` | 改动前的两个脚本备份 |

产物：

```
out/models/diff-cond-seed{0,1,2}.pt           A 终版（0.4229M）
out/models/diff-nocond-seed{0,1,2}.pt         B（0.4070M）
out/models/diff-cond-seed{0,1,2}-zeroinit.pt  消融：零初始化
out/models/diff-cond-seed{0,1,2}-init0051.pt  消融：默认初始化
out/diffusion/logs/*.jsonl                    逐轮日志
out/diffusion/eval-audio-cond-seed0.json      完整评估结果
out/diffusion/eval-audio-cond-3seeds.json     3 seed 的验证损失 + 因果干预
out/diff-data/audio-cond-labels.npz           带标签 + 预切条件的 199 窗口
```

---

## 9. 局限

1. **样本量**：只有 6 首歌 / 199 个窗口（179 训练 / 20 验证），12 轮 = 144 步。
   验证损失差 0.0007 完全可能是 20 个验证窗口的偶然，只有配对统计（25/30）才站得住。
2. **条件调节标签退化**：style 全 0、diff 全 1，两个嵌入是常量；
   这次实验干净但**没有验证「音频条件 × 风格/难度」的交互**。
3. **条件本身很粗**：只有 4 个 onset 频带（不是 Mug 的完整 mel），
   逐行相关 0.149 → 只能表达密度，不能表达音高/位置。
4. **注入点在 1/4 分辨率**（32 步 ≈ 每拍一个条件向量），细粒度时间控制有限。
   想进一步提升音符级对齐，应该在 `u2`（全分辨率、base 通道）再加一个注入点。
5. **生成密度整体偏低**（0.22~0.36 vs 真实 0.42~0.54），说明模型还没训够；
   条件带来的 12~15% 相对移动是在这个偏低的基础上测的。
6. **评估口径**：验证损失用的是 eps-MSE；采样评估只有 8 个窗口、单次采样；
   F1 这类音符级指标对噪声谱面不友好。**没有做人类可玩性评估**。
7. **没和 Mug-Diffusion 比**：这里只证明「等价接口在自己的数据上有效」，没有对比两者生成质量。

---

## 10. 复现

```bash
cd /root/work/yamero-ai/proj

# 1) 数据：补标签 + 预切条件
python3 tools/build-audio-cond-labels.py

# 2) A/B 训练（3 seed × 2 模型，12 轮，各 ~1 分钟）
bash tools/ab-audio-cond.sh

# 3) 完整评估（约 2.5 分钟）
python3 tools/eval-audio-cond.py --threads 1

# 4) 采样
python3 tools/sample-diffusion.py --ckpt out/models/diff-cond-seed0.pt \
    --cond out/multisong/Heart_Fraction-cond.npz --cond-index 896 --num 2 --threads 1
```

---

## 11. 顺带发现的既有问题（没改，只报告）

`tools/train-diffusion.py` 的训练循环是：

```python
loss = diff.loss(model, X[idx], S[idx], D[idx], cond=cb)   # ← 没有传 a.pred
```

`diff.loss(..., mode='eps')` 的 `mode` 默认是 `'eps'`，所以 **`--pred x0` / `--pred v` 时训练目标仍然是 eps**，
但 ckpt 里记录的 `pred` 是 `x0`/`v`，采样脚本会按 `x0`/`v` 去解码 → **训练目标与解码方式不一致**。
`diffusion4k-x0.pt` 就是这种状态。修法是把 `a.pred` 作为第 5 个位置参数传进 `diff.loss`：

```python
loss = diff.loss(model, X[idx], S[idx], D[idx], a.pred, cond=cb)
```

为了不改动现有行为（任务要求向后兼容），**本次没有修**；本报告所有实验都用默认的 `eps`，不受影响。
