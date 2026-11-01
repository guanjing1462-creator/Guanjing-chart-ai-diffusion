# 音频条件数据（audio conditioning for the diffusion model）

给扩散模型补一个「音频条件」分支：Mug-Diffusion 那边用的是 mel 频谱，这里在自己数据上做对应物 ——
**每首歌按分析文件的 1/16 网格切格，每格 4 个 onset 强度（低频/中频/高频/总），归一化到 0–1。**

> ⚠ **这是小样本，不是全量训练数据。**
> 本地只有 **6 首**歌有音频（`ref/multisong/`），所以条件特征只覆盖这 6 首。
> 它的用途是**把音频条件分支的接口跑通、验证形状与对齐**；全量训练要么补齐音频，
> 要么用 `out/diff-data/cond-windows.npz` 这份配套小样本（见第 5 节）。

---

## 1. 结论速览

| 产物 | 内容 | 结论 |
| --- | --- | --- |
| `out/multisong/<歌名>-cond.npz` | 6 个文件，`feat [T,4] float16` | 共 13612 格，值域 0–1，对齐 1/16 网格 |
| `out/diff-data/cond-index.npz` | `idx / song / start`，把 dataset 窗口对到 (歌, 起始格) | **0 / 30551** 个窗口能对上（原因见第 4 节，不是脚本坏了） |
| `out/diff-data/cond-windows.npz` | 配套小样本：直接从这 6 首切窗，`X / song / start / songs` | **199 个窗口**，可与 cond 直接配对，音频条件分支今天就能验证 |

**为什么 `cond-index.npz` 是空的：** `out/diff-data/dataset.npz` 的 30551 个窗口全部来自
`out/packs/` 里**别人的曲子**（人类谱面语料），而这 6 首有音频的歌在本地**没有任何 1/16 量化的谱面**
（`out/multisong/charts/*.osu` 和 `out/all/*.osu` 是本仓 GJ 流程生成的、而且是 1/32 量化）。
没有谱面 = 没有窗口 = 没有可条件化的样本。第 4 节给了三条独立证据。

---

## 2. 特征定义

每格 4 个值（`feat[:, j]`）：

| 通道 | 名称 | 频带 | 含义 |
| --- | --- | --- | --- |
| `ch0` | `low_0_200` | 0–200 Hz | 低频 onset 强度（底鼓、贝斯击打） |
| `ch1` | `mid_200_2000` | 200–2000 Hz | 中频 onset 强度（军鼓、人声、主旋律起音） |
| `ch2` | `high_2000_8000` | 2000–8000 Hz | 高频 onset 强度（踩镲、镲片、齿音） |
| `ch3` | `onset_total` | 0–8000 Hz | 总 onset 强度 |

算法（`tools/audio-cond-data.py`，单声道 22.05 kHz）：

1. `|STFT|`：`n_fft=512 / hop=128`（5.805 ms/帧）—— 和 Mug 的 mel 条件（`[128, 32768]`，5.805 ms/帧）
   与本仓 `build-audio-features.py` 的帧率完全一致，将来把 4 维特征换成 mel 分支时**帧率都不用改**；
2. 128 带 mel 滤波（`fmin=0, fmax=8000`）→ `log1p` → 沿时间**正向差分取正**（谱通量，只保留上升沿）；
3. 按上表三个频带把 mel 带上的通量求和；总通道 = 三个带之和（原始通量上满足 `total = low+mid+high`）；
4. 落到网格：格 `k` 覆盖 `[offsetMs + k·stepMs, offsetMs + (k+1)·stepMs)`，取**格内最大值**（保住瞬态）；
   帧→格的时刻用「帧中心」`((c+1)·hop + n_fft/2)`，合成脉冲轨验证：**100% 落在正确的格**（没有 off-by-one）；
5. 归一化：**每首、每通道**按 P99 缩放并截断到 [0,1]（系数记在 `cond-meta.json`）。

> 归一化是**逐通道独立**的，所以归一化后 `total ≠ low+mid+high`（各自缩放不同）。
> 这样做是为了让弱的高频带也有足够动态；代价是**通道之间的绝对能量不可比**（跨歌比较也一样）。
> 要还原绝对量：`raw = feat * p99_scale`。

网格起点/步长直接取分析文件：`offsetMs` / `stepMs`（`stepMs = 60000/bpm/4`），所以和
`out/multisong/*-fx.json`、GJ 谱面用的是同一套时钟。

---

## 3. 每首歌的网格长度

| 歌 | 分析文件 | T（格） | 时长 (s) | offsetMs | stepMs | BPM | 补零格 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Heart_Fraction | `Heart_Fraction-fx.json` | 1782 | 196.4 | 162.05 | 110.05 | 136.30 | 0 |
| seatrus___purifikation | `seatrus___purifikation-fx.json` | 1236 | 150.5 | 426.88 | 121.62 | 123.33 | 1 |
| Plum_FallenEra | `Plum_FallenEra-fx.json` | 1374 | 171.3 | −22.00 | 124.99 | 120.01 | 3 |
| REDALiCE_x_t_pazolite___Xterfusion | `REDALiCE_x_t_pazolite___Xterfusion-fx.json` | 1466 | 170.0 | 64.00 | 115.72 | 129.62 | 0 |
| _______________Flash_Me_Back | `_______________Flash_Me_Back-fx.json` | 2530 | 322.9 | 457.04 | 128.35 | 116.87 | 18 |
| Frums_SuperMix | `Frums_SuperMix-fx.json` | 5224 | 621.5 | 153.36 | 119.36 | 125.67 | 18 |
| **合计** | | **13612** | 1632.7 | | | | 40 |

- 「补零格」= 网格比音频长出来的尾巴（`steps·stepMs` 略微超过 `durationSec`），这些格的特征为 0，共 40 格（0.3%）。
- 每次只处理一首，STFT 分块（4096 帧 ≈ 24 s）计算，内存峰值不随曲长增长（最长那首 621 s 也照常）。

交叉检查（**只是旁证，不是指标**）：与 `fx.json` 里既有 `onset` 包络的相关 r = 0.24–0.57
（`Frums 0.247 / Heart 0.332 / Plum 0.456 / Xterfusion 0.571 / Flash 0.242 / purifikation 0.255`）。
两者定义不同（他们用 2048/256 的 mean 聚合，这里用 512/128 的分带通量），r 不高属预期；
网格是否对得上靠的是第 2 节第 4 步的合成脉冲验证，以及下面第 4 节的频谱/能量证据。

---

## 4. `dataset.npz` 的匹配结果：0 / 30551

### 4.1 先把「每个窗口来自哪儿」找回来

`dataset.npz` 只有 `X / style / diff`，没有出处。`tools/build-cond-index.py` 的做法是**重放**：
按 `tools/diff-data.py` 的同一套规则（`out/packs/{tech,jack,speed,stamina}/*.osu`、相邻间隔中位数当步长、
`WIN=128`、步长 64 的滑窗、`≥32` 行、`≥12` 命中）重新切一遍，得到 88787 个候选窗口，
再把每行的内容 md5 与 dataset 的 30551 行比对。

**结果：30551 / 30551 全部找回（来源无法还原 0 个）** —— 说明 dataset.npz 确实是由这套规则从 `out/packs`
生成的，溯源可靠。若将来切片规则被改动，脚本会明确报「来源无法还原 N 个」，不会瞎猜。

### 4.2 匹配门槛（三条都要过）

| 门槛 | 规则 | 为什么 |
| --- | --- | --- |
| 元数据 | 谱面 `Title/TitleUnicode/Artist/ArtistUnicode/AudioFilename/Version` 归一化后含歌名 | 先确认「是这首歌」 |
| 步长 | 谱面 1/16 步长（相邻音符间隔中位数）与分析 `stepMs` 相对误差 ≤ 3% | 128 格窗口必须正好覆盖 128 个音频格 |
| 相位 | 音符落在网格线（±15% 格）的比例 ≥ 0.6 | 谱面得是对着**这条**音频做的 |

「音符所在格的 onset 特征 / 随机基线」也算了，但**只作诊断不作门槛**：谱面一密这个比值天然接近 1，
门槛太弱会放进假阳性（实测会误收 4 个 120 BPM 巧合的谱）。

### 4.3 结果

全库 **2608** 张谱面 × 6 首：

- **步长 + 相位同时吻合：6 个 (谱, 歌) 组合** —— 全是速度巧合（120/127 BPM 之类，例如
  `[236bpm] boot` 127.00 ms ≈ Plum 的 124.99 ms），元数据一条都对不上，是别人的曲子；
- **元数据命中：1 个** —— `out/packs/jack/5798118.osu`（"Xterfusion (low)"，Guanjing's Eta Minijack
  Practice Pack）。名字是这首歌，但它是**变速练习包**：谱面 1/16 步长 59.00 ms，我们的音频 115.72 ms，
  差 49%（正好 1/32 vs 1/16）。再把音符时刻对到我们的音频上取短时能量跃升：**≈ 0（甚至还偏低）**，
  随机时刻反而是 +0.007 —— 音符没踩在这条音频的击打上，确认是另一个变速版本；
- **最终接受：0**。

还有一个容易误会的点：`out/multisong/charts/Heart_Fraction.osu` 的 `Title` 是 `INTERNET YAMERO`
（`paint-gj.py` 里 `--title` 的默认值，所有 GJ 谱面都带这个模板名），而 `out/packs` 里正好有 2 张
`INTERNET YAMERO` 练习谱（1.55x / 0.8x）。即使假设它们是同一首歌，这两张谱的 1/16 步长是
**97.0 ms / 101.0 ms**，与 Heart 的 110.05 ms 差 8–12%，同样过不了步长门槛。所以 0 这个结论
不依赖于「哪首歌是哪首歌」的判定。

另外注意：本仓 GJ 生成的谱面（`out/multisong/charts/`、`out/all/`）是 **1/32 量化**（34–36% 的音符在半格上），
即使把它们放进 `out/packs`，也会因为「128 格窗口只覆盖一半时长」被步长门槛拒掉 —— 这是对的，
除非条件网格也改成 1/32。

> 所以：**0 个窗口可条件化**，原因是数据面的（没有这 6 首的 1/16 谱面），不是实现面的。

### 4.4 为什么相信这个 0

1. 溯源全中（30551/30551），不存在「窗口来源不明所以匹配不上」；
2. 元数据 + 步长 + 相位三个门槛分别独立否掉了所有候选（6 个网格巧合、1 个变速练习包）；
3. 正例自测：把一首 1/16 量化、元数据正确的合成谱放进临时 pack 目录，配 23 个窗口的迷你 dataset，
   工具 **23/23 全中，`start` 与真值逐个完全一致**（`song=1` = Heart_Fraction，`start=4,74,139,…`）——
   机制能命中，不是只会返回 0。

---

## 5. 配套小样本 `cond-windows.npz`（推荐的接口验证路径）

既然 dataset.npz 里的窗口没法条件化，就直接从这 6 首的谱面在**分析网格**上切窗：

- 来源：`out/multisong/charts/<歌名>.osu`（GJ 流程生成）；
- 网格：`offsetMs + k·stepMs`，窗口 128 格，步长 64 格（50% 重叠）；
- 过滤：窗口内 ≥ 32 个音符时刻、≥ 12 个命中格（与 `diff-data.py` 的密度门槛一致）；
- 结果：**199 个窗口**（Frums 79 / Heart 26 / Plum 20 / Xterfusion 21 / Flash 35 / purifikation 18）；
- 注意：这些谱是 1/32 量化，切到 1/16 网格时有 34–36% 的音符被吸到最近格（丢的是半格细节，
  窗口覆盖的时间跨度不变）。要让谱面细节也 1:1，需要 1/32 的条件网格。

自检：窗口音符密度与 cond 各通道的相关 r ≈ +0.13–0.15（jack 谱占位率 0.44，密度已饱和，弱正相关属预期）。

---

## 6. 文件清单与格式

| 路径 | 键 | dtype / shape | 说明 |
| --- | --- | --- | --- |
| `out/multisong/Heart_Fraction-cond.npz` 等 6 个 | `feat` | `float16 [T,4]` | 音频条件；`T` = 该歌格数（见第 3 节表），列顺序见第 2 节 |
| `out/multisong/cond-meta.json` | — | JSON | 每首的网格参数、P99 系数、补零格、交叉检查值、特征定义说明 |
| `out/diff-data/cond-index.npz` | `idx` / `song` / `start` | `int32[n]` / `int16[n]` / `int32[n]` | dataset 第 `idx` 行 → 第 `song` 首（id 见下）的起始格 `start`；窗口占 `[start, start+128)`；当前 `n=0` |
| `out/diff-data/cond-index.json` | — | JSON | 统计、`song` id ↔ 歌名、2608 张谱的判定诊断、配套小样本信息 |
| `out/diff-data/cond-windows.npz` | `X` / `song` / `start` / `songs` | `uint8 [W,4,128]` / `int16[W]` / `int32[W]` / `<U…[6]` | 配套小样本，`W=199`；`X` 与 `dataset.npz` 的 `X` 同编码（4 键 × 128 格二值） |

`song` id 顺序（两处一致）：`0 Frums_SuperMix, 1 Heart_Fraction, 2 Plum_FallenEra,
3 REDALiCE_x_t_pazolite___Xterfusion, 4 _______________Flash_Me_Back, 5 seatrus___purifikation`。

复现：

```bash
python3 tools/audio-cond-data.py            # 6 首 → out/multisong/*-cond.npz（一次一首）
python3 tools/build-cond-index.py           # 重放溯源 → cond-index.npz + cond-windows.npz
# 单首重跑：--song Heart --force；只重建索引不动配套样本：--labeled ''
```

---

## 7. 怎么接进模型

条件张量与谱面窗口同长度（128 格），所以最省事的接法是沿时间维拼：

```python
import numpy as np

z = np.load('out/diff-data/cond-windows.npz')
X, song, start, names = z['X'], z['song'], z['start'], z['songs']

i = 0
c = np.load(f'out/multisong/{names[song[i]]}-cond.npz')['feat']   # [T,4] float16
cond = c[start[i]:start[i] + 128].astype(np.float32)               # [128,4]

# 条件分支：4 → C（与谱面 4 键同宽，方便直接 concat），再和谱面一起进 UNet
# cond_t = torch.from_numpy(cond).T[None]          # [1,4,128]
# h = cond_encoder(cond_t)                          # Conv1d(4, C, k=3) → [1,C,128]
# h = h + timestep_emb[:, :, None]                  # Mug 里 mel 条件是这么加的
# model_input = torch.cat([noisy_x, h], dim=1)      # [1, 4+C, 128]
```

对应关系：Mug 的音频条件是 mel `[128, 32768]` → `MelspectrogramScaleEncoder1D`（10 级下采样）→
**逐时间位置通道拼接**（`MUG-DIFFUSION-剖析.md` §3）。这里 4 维分带 onset 就是它的降维等价物，
形状 `[128,4]` 与谱面窗口等长，所以「拼到通道」这一套接口可以直接照搬。

如果要上 classifier-free guidance，训练时按 10–20% 概率把 `cond` 置零（Mug 本身没搬 LDM 的
`cond_dropout`，所以这步是可选的）；先只接 `ch3`（语义最接近 `fx.onset`）做最小验证也够。

**一句话下一步：** 先用 `cond-windows.npz` 的 199 个窗口 + `-cond.npz` 把 4→C 的条件分支跑通
（形状 `[128,4]`、与谱面同长度、置零可训），再决定要不要为这 6 首补 1/16 量化谱面 / 1/32 条件网格。

---

## 8. 已知限制

1. **小样本**：只有 6 首、13612 格、199 个配套窗口。只能验证接口，不能当训练集；
2. **通道独立归一**：`total ≠ low+mid+high`（见第 2 节），跨通道/跨歌的绝对能量不可比；
3. **1/16 分辨率**：条件最小时间粒度是 `stepMs`（110–128 ms）；谱面若用 1/32 量化，条件跟不上半格细节；
4. **溯源依赖切片规则**：`cond-index.npz` 的前提是 dataset.npz 由 `tools/diff-data.py` 当前规则生成；
   规则一改就要重跑（脚本会用「来源无法还原」明确告警）；
5. **音频版本敏感**：变速/变调版的谱面（练习包常见）会被步长门槛正确拒绝，但那也意味着这些窗口无法条件化；
6. **与 `fx.onset` 的 r 偏低**（0.24–0.57）是两个包络定义不同，不代表网格不对（网格用合成脉冲验证过）；
7. **未改动**：`tools/train-diffusion.py`、`tools/paint-gj.py`、`out/diff-data/dataset.npz` 都只读没写。
