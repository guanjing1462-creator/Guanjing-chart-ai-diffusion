# round143 · 音频条件（踩音）训练数据

> 给 GJ 扩散模型补「音频 → 落点」的条件监督。本轮**不训练**，只做数据：用 osu 镜像找到这 6 首歌
> 的真实 mania 谱面 → 逐条验证匹配 → 切成 1/32 窗口 → 与已算好的音频条件特征按**绝对时间**对齐 →
> 抽查 3 个窗口的时间对齐。
>
> **结论先说**：6 首歌里只有 **2 首**能找到可验证的 mania 4K 谱面。Xterfusion 3 张（2 张 gameplay
> 去重后）、Flash Me Back 2 张，共 **4 张谱 / 165 个窗口**（candidate 里另有 7K 谱、音频超差谱、
> 以及 9 个下载 503 的 set，全部记拒绝）。Xterfusion 的对齐证据是决定性的（全曲位移扫描峰在 0 格）；
> Flash Me Back 只有弱证据，原因是**源条件特征**在该歌上分辨率不足（本地检测 BPM 是真实值的 2/3），
> 不是切片错位。

## 0. 数字速览

| 项 | 结果 |
| --- | --- |
| 搜索 | 6 首歌 × 1–3 条查询（`Xterfusion`/`Flash Me Back`/`seatrus`/`purifikation`/`FallenEra`/`Frums`/`SuperMix`/`fraction`…），命中 mania set 候选 ~90 个，人工比对标题+艺术家后精选 16 个 set 尝试下载 |
| 实际下载 | **7 个 .osz，116,982,814 B（111.6 MiB）**（限额 200MB 内）；**9 个 set 端点 503 失败**（重试 2–3 次无效） |
| 得到的 .osu | 19 张（含同 set 多难度）+ 9 个 set 下载失败无法取得 |
| **接受** | **4 张**（tier A 2 张、tier B 2 张），全部入集；另有 1 张 `[sv]` 与已接受谱 gameplay 完全相同 → 去重记拒绝 |
| **拒绝** | **20 张/次**：7K 谱 2、音频时长超差 8、gameplay 重复 1、下载失败 9 |
| 入集歌曲 | 2 首（Xterfusion、Flash Me Back） |
| 数据集 | `out/diff-data-audio/dataset.npz` — **165 窗**，`X`[165,4,256] uint8 / `C`[165,256,4] float16 |
| 窗口 | 256 格 @ 1/32，stride 128，最少 20 占位格 |
| 占位率 | **13.08%** |
| p132 | **0.1120**（相邻占用格间隔 = 1 格的比例，池化 14548 个间隔） |
| 条件覆盖率 | **99.940%** |
| 平均音符/窗口 | 133.9 |
| split（按谱分组、无跨集） | train 79 / val 30 / test 56 窗 |

## 1. 搜索与下载清单

镜像 `https://osu.direct/api/search`（`ChildrenBeatmaps[].Mode==3` 即 mania），下载 `https://osu.direct/api/d/<SetID>`。

| 歌（本地文件） | 查询词 | 命中情况 | 结论 |
| --- | --- | --- | --- |
| Heart Fraction | `Heart Fraction` / `fraction` / `HeartFraction` | 大量 "Heart"（TOOBOE 等）与 "Red Fraction" 同名歌，无一首是本地这首；`HeartFraction` 零结果 | **拒绝：无 mania 谱** |
| seatrus - purifikation | `purifikation` / `seatrus` / `seatrus purifikation` | 全是 "Purification"（OkameP / Noah / Daniel Deluxe…）或 seatrus 的其它曲 | **拒绝：无 mania 谱** |
| Plum - FallenEra | `FallenEra` | SetID **2630531** `Plum - FallenEra (2022 Remake)` mania=4 BPM 180；另有 `PLUM MEGAMIX 2`（692 s 混音，非本曲） | **拒绝：下载 503**（3 次重试），拿不到音频无法做时长/BPM 验证 |
| REDALiCE x t+pazolite - Xterfusion | `Xterfusion` | 16 个 mania set（BPM 128 / 256 两派），标题艺术家一致 | **接受 3 张 → 去重后 2 张谱** |
| かめりあ - Flash Me Back | `Flash Me Back` | 40+ 个 mania set，标题艺术家一致，BPM 175 | **接受 2 张**；7K 2 张、音频超差 8 张拒绝 |
| Frums - SuperMix | `Frums` / `SuperMix` | Frums 单曲一堆，但**没有 621.5 s 的 SuperMix**（本地 10:21） | **拒绝：无 mania 谱** |

下载失败的 9 个 set（全部返回 `{"code":503,"message":"failed to fetch beatmap set"}`，串行/换 UA/间隔重试
均无效，判断为镜像侧缺缓存）：**2630531, 2501284, 2503929, 1826508, 1831601, 1832304, 1955928,
1856132, 1836984**。下载字节合计 **116,982,814 B（111.6 MiB）**，未超 200 MB；`_dl/` 里另有解出的
.osu 与音频（约 51 MB，用于量时长，可删）。

## 2. 匹配验证（逐张）

判据：① 标题/艺术家与本地歌一致（人工确认）；② **音频时长差 ≤ 3 s**（`ffprobe` 量 .osz 内音频
与本地 `ref/multisong/*`）；③ BPM 与 `-fx.json` 的 `bpm` 差 ≤2%。谱面 BPM 取 `[TimingPoints]` 中
uninherited=1 红线里**时间覆盖最长**的那条（变速谱 = 主段 BPM）。

### 2.1 接受

| 歌 | SetID | .osu | 键 | 谱面音频(s) | 时长差(s) | 谱面BPM | 本地BPM | BPM差 | tier | tempo倍数 | 倍数归一后 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REDALiCE x t+pazolite - Xterfusion | 2614015 | REDALiCE x t+pazolite - Xterfusion (dsyFiredragon) [NO SV].osu | 4K | 170.040 | 0.030 | 128.00 | 129.62 | 1.27% | A | — | 1.266% |
| REDALiCE x t+pazolite - Xterfusion | 2628774 | REDALiCEt+pazolite - Xterfusion (KiLi-sens) [x x xxx].osu | 4K | 167.919 | 2.091 | 128.00 | 129.62 | 1.27% | A | — | 1.266% |
| かめりあ - Flash Me Back | 1835578 | Camellia - Flash Me Back (ZirconEscalus) [Temporal Anomaly].osu | 4K | 322.952 | 0.082 | 175.00 | 116.87 | 33.22% | B | ×1.5 | 0.174% |
| かめりあ - Flash Me Back | 2538176 | Camellia - Flash Me Back (DaveClicky) [like the good ol days or somethin].osu | 4K | 322.952 | 0.082 | 175.00 | 116.87 | 33.22% | B | ×1.5 | 0.174% |

- **tier A**：三项全过。Xterfusion 2614015 的 .osz 内 `audio.mp3` 与本地
  `ref/multisong/REDALiCE x t+pazolite - Xterfusion.mp3` **md5 完全相同**
  (`c204bd96dca8124dca0e191971a9b03c`) → 同一录音，0.03 s 是容器/解码差。
- **tier B（tempo-倍数）**：Flash Me Back 谱面 **175 BPM**，本地 `-fx.json` 检测 **116.87 BPM**；
  175 ÷ (116.87×3/2) = 1.0017 → 本地检测落在真实拍级的 **2/3**（一个 175 BPM 的 1/16 格里正好
  落 3 个 1/32 格）。音频时长差仅 **0.082 s**，同一录音无疑；但**原始 BPM 差 33.2%**，
  所以单列 tier B，不冒充 tier A。

### 2.2 拒绝

| 歌 | SetID | .osu | 音频(s) | 拒绝原因 |
| --- | --- | --- | --- | --- |
| Plum - FallenEra | 2630531 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| REDALiCE x t+pazolite - Xterfusion | 2501284 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| REDALiCE x t+pazolite - Xterfusion | 2614015 | REDALiCE x t+pazolite - Xterfusion (dsyFiredragon) [sv].os | 170.04 | gameplay 内容与 REDALiCE x t+pazolite - Xterfusion (dsyFiredragon) [NO SV].osu 完全相同（只差 SV/视觉），去重 |
| かめりあ - Flash Me Back | 1826508 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 1828871 | Camellia - Flash Me Back (GeoAlienXD) [FLASHY AF (New Vers | 328.777 | 音频时长差 5.91s > 3.0s |
| かめりあ - Flash Me Back | 1828871 | Camellia - Flash Me Back (GeoAlienXD) [FLASHY AF (Old Vers | 328.777 | 音频时长差 5.91s > 3.0s |
| かめりあ - Flash Me Back | 1828871 | Camellia - Flash Me Back (GeoAlienXD) [Flashless].osu | 328.777 | 音频时长差 5.91s > 3.0s |
| かめりあ - Flash Me Back | 1828871 | Camellia - Flash Me Back (GeoAlienXD) [Flashy].osu | 328.777 | 音频时长差 5.91s > 3.0s |
| かめりあ - Flash Me Back | 1831601 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 1832304 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 1836984 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 1856132 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 1955928 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 2187330 | Camellia - Flash Me Back (Aiyoori) [A while ago].osu | 327.144 | 音频时长差 4.27s > 3.0s |
| かめりあ - Flash Me Back | 2187330 | Camellia - Flash Me Back (Aiyoori) [Flash Me Back].osu | 327.144 | 音频时长差 4.27s > 3.0s |
| かめりあ - Flash Me Back | 2187330 | Camellia - Flash Me Back (Aiyoori) [Flashback].osu | 327.144 | 音频时长差 4.27s > 3.0s |
| かめりあ - Flash Me Back | 2187330 | Camellia - Flash Me Back (Aiyoori) [Rewind].osu | 327.144 | 音频时长差 4.27s > 3.0s |
| かめりあ - Flash Me Back | 2503929 | —（未下载成功） | — | 下载端点 503，无本地文件 → 无法验证时长/BPM（镜像侧缺缓存） |
| かめりあ - Flash Me Back | 2625633 | Camellia - Flash Me Back (Gndsklgb) [Lasting memories].osu | 322.952 | parser_keys_not_4; keys=7（非 4K，X 是 [N,4,256]） |
| かめりあ - Flash Me Back | 2625633 | Camellia - Flash Me Back (Gndsklgb) [Tears of Joy].osu | 322.952 | parser_keys_not_4; keys=7（非 4K，X 是 [N,4,256]） |

## 3. 数据集

`out/diff-data-audio/dataset.npz`

| 键 | 形状/类型 | 含义 |
| --- | --- | --- |
| `X` | [165,4,256] uint8 | 二值钢琴卷（1/32 格，4 键） |
| `C` | [165,256,4] float16 | 音频条件切片（低/中/高频 + 总 onset） |
| `song` | [165] int8 | 歌索引，指向 `songs`（6 首本地歌顺序固定） |
| `chart_id` | [165] U24 | `gja_` + gameplay content sha256 前 20 位 |
| `split` | [165] int8 | 0=train 1=val 2=test，**按谱面整张分配** |
| `start_cell` | [165] int32 | 窗口起点在该谱绝对 1/32 格坐标 |
| 溯源 | `songs` U48[6] / `bpm` f32（窗口起点红线 BPM）/ `t0_ms` f32（窗口起点毫秒）/ `cond_coverage` f32 / `window_cells` / `stride_cells` / `grid_division` | 审计用 |

**网格**：`cell(t)=round(8·beat(t))`，`beat(t)` 由该谱真实红线分段线性给出——与
`tools/diff-data32-grouped.py` 同一份解析器（`tools/gj-data-manifest.py`）和同一公式。
4 张谱全部是多红线变速谱（distinct BPM 13–17），**按分段处理，不跳过**：格坐标始终是拍坐标，
窗口内格长随红线变化，与全量 1/32 数据同口径。
**条件**：`k = floor((t_cell_ms − cond_offset_ms) / cond_step_ms)`，一个 1/16 格覆盖它时间范围内的
所有 1/32 格。两边只共享「绝对时间」，因此**谱面 BPM 与本地检测 BPM 不成整数倍也不会相位错**。
**去重**：2614015 的 `[NO SV]` 与 `[sv]` 只差视觉位移、gameplay sha256 相同 → 只入集一张，
否则同谱会同时进 train/val（泄漏）。

### 3.1 每张谱

| 歌 | SetID | .osu | split | tier | 音符 | 窗口 | distinct BPM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| _______________Flash_Me_Ba | 1835578 | Camellia - Flash Me Back (ZirconEscalus) [Temporal | train | B | 3836 | 56 | 13 |
| _______________Flash_Me_Ba | 2538176 | Camellia - Flash Me Back (DaveClicky) [like the go | test | B | 3406 | 56 | 13 |
| REDALiCE_x_t_pazolite___Xt | 2614015 | REDALiCE x t+pazolite - Xterfusion (dsyFiredragon) | train | A | 1795 | 23 | 16 |
| REDALiCE_x_t_pazolite___Xt | 2628774 | REDALiCEt+pazolite - Xterfusion (KiLi-sens) [x x x | val | A | 2105 | 30 | 17 |

| 歌 | 谱面 | 窗口 |
| --- | --- | --- |
| REDALiCE x t+pazolite - Xterfusion | 2 | 53 |
| かめりあ - Flash Me Back | 2 | 112 |

split 规则：每首歌按内容哈希排序，**第一张必进 train**（保证两首歌都在训练集），其余按（歌, 哈希）
轮流进 val/test。结果 **test 只来自 Flash Me Back（56 窗）、val 只来自 Xterfusion（30 窗）**——
无同谱跨集泄漏，但跨歌不均衡（见 §5）。

## 4. 抽查：条件切片是否时间对齐

三个口径分开查，避免自证：

- **A** 窗口时间范围 vs 该范围内音频 onset 峰值时刻（任务要求的打印项）；
- **B** `C.onset` 与 `X` 占位的相对位移扫描（±8 个 1/32 格，看峰是否在 0）；
- **C** 全曲独立验证：谱面音符占用（按谱面绝对时间投到 `fx.json` 的 1/16 网格）与 `fx.json` 的
  `onset` 做 ±12 格位移扫描。`fx.json` 是**另一条管线**（`detect-bpm.py` + 音频分析）的产物，
  与本次切片代码无关。

**窗口 #61 — かめりあ - Flash Me Back / Camellia - Flash Me Back (DaveClicky) [like the good ol days or somethin].osu（test，gja_aaa36c333148a4a119c9）**

- 窗口 `start_cell=512`，时间 **27467.9 – 38439.3 ms**（10.97 s），窗口 BPM 175.0，1/32 格 42.857 ms，条件覆盖 1.0
- **该时间段内音频 onset 峰值时刻**（ms，`-cond.npz` 1/16 网格 ch3 局部极大，取前 6 个）：`[29078.6, 32287.3, 34597.5, 36779.4, 37421.2, 30105.4]`
- 峰值 → 最近谱面音符时间距离（ms）：`[103.4, 20.3, 69.5, 54.4, 11.2, 66.4]`，中位 **60.4** ms
- B. C.onset vs X 占位 位移扫描（±8 个 1/32 格）：best = **-2 格**，corr(0)=0.0236，corr(best)=0.3394
- C. 全曲独立验证（谱面音符占用 vs `fx.json` onset，±12 个 1/16 格）：best = **-3 格**，corr(0)=**0.0902**，corr(best)=0.1302，±12 格控制 0.0253/0.0869（3406 音符）

**窗口 #112 — REDALiCE x t+pazolite - Xterfusion / REDALiCE x t+pazolite - Xterfusion (dsyFiredragon) [NO SV].osu（train，gja_14d1d7248effd6e33363）**

- 窗口 `start_cell=0`，时间 **0.0 – 15000.0 ms**（15.0 s），窗口 BPM 128.0，1/32 格 58.594 ms，条件覆盖 0.9922
- **该时间段内音频 onset 峰值时刻**（ms，`-cond.npz` 1/16 网格 ch3 局部极大，取前 6 个）：`[411.2, 1915.6, 1221.2, 2262.7, 1568.4, 642.6]`
- 峰值 → 最近谱面音符时间距离（ms）：`[56.8, 40.6, 67.8, 80.3, 71.6, 60.4]`，中位 **64.1** ms
- B. C.onset vs X 占位 位移扫描（±8 个 1/32 格）：best = **0 格**，corr(0)=0.2019，corr(best)=0.2019
- C. 全曲独立验证（谱面音符占用 vs `fx.json` onset，±12 个 1/16 格）：best = **0 格**，corr(0)=**0.4184**，corr(best)=0.4184，±12 格控制 0.157/0.1436（1795 音符）

**窗口 #44 — かめりあ - Flash Me Back / Camellia - Flash Me Back (ZirconEscalus) [Temporal Anomaly].osu（train，gja_49bb8824d511d19bc2e8）**

- 窗口 `start_cell=5504`，时间 **241410.7 – 252382.1 ms**（10.97 s），窗口 BPM 175.0，1/32 格 42.857 ms，条件覆盖 1.0
- **该时间段内音频 onset 峰值时刻**（ms，`-cond.npz` 1/16 网格 ch3 局部极大，取前 6 个）：`[249579.9, 246114.5, 242649.2, 246756.3, 245729.5, 246499.6]`
- 峰值 → 最近谱面音符时间距离（ms）：`[59.1, 10.5, 39.2, 31.3, 33.5, 32.6]`，中位 **33.0** ms
- B. C.onset vs X 占位 位移扫描（±8 个 1/32 格）：best = **7 格**，corr(0)=-0.0793，corr(best)=0.1271
- C. 全曲独立验证（谱面音符占用 vs `fx.json` onset，±12 个 1/16 格）：best = **-3 格**，corr(0)=**0.0944**，corr(best)=0.1155，±12 格控制 0.048/0.0738（3836 音符）

### 4.1 结论

- **Xterfusion（tier A）：对齐成立，无整体偏移。** 1795 个音符的全曲扫描峰**正好在 0 格**
  （corr **0.4184**），±3 格控制掉到 **−0.144 / −0.139**、±12 格 0.157 / 0.144；
  音符处的条件 onset 均值 **0.4625**，随机时刻 0.3979，
  ±0.5 格控制 0.4127 / 0.4286。
  落点确实压在高 onset 上 → 这一首可以做「细粒度踩音」的验证歌。
- **Flash Me Back（tier B）：只有弱证据，且短板在源特征。** 音符处 onset 均值 0.4115
  高于随机 0.3588（+14.7%），
  但全曲扫描峰在 **−3 格**（0.1155 vs 0 格 0.0944），
  逐窗口 C↔X 相关均值仅 0.012。查源数据：该歌 `out/multisong/cond-meta.json` 本来就记着
  `corr_with_fx_onset=0.2424`、`corr_best_shift_cells=3`、`grid_onset_align=0.2857`（vs 半格控制
  0.2782，几乎没余量）——本地分析网格在这首歌上是 116.87 BPM（真实 175 的 2/3），1/16 格 = 3 个
  1/32 格，同一个起音落在格内 3 个位置无法区分，**条件信号时间锐度本来就不够**。切片按绝对时间
  查表，offset 为 0，不是本次构建的错位。
- 对齐链路上游已固定：`C` 每格由该格绝对时间查表，`X` 由红线绝对时间量化；Xterfusion 的扫描峰
  落在 0 格直接证明这条链路没有整体偏移。

## 5. 剩余限制

1. **只有 2/6 首拿到谱面**。Heart Fraction、seatrus - purifikation、Frums - SuperMix 在镜像上没有
   对应 mania 谱；Plum - FallenEra 有 SetID 2630531 但下载 503，拿不到音频 → 无法验证 → 记拒绝。
   （其中 Heart Fraction / purifikation 这两首在本地是 `.ogg`，音频特征质量也一般。）
2. **数据量很小**：165 窗 / 4 谱 / 2 歌，占全量 30551 窗的 **0.54%**。够验证「条件 → 落点」接口、
   做小规模微调；**不足以**训通用踩音模型。且 Xterfusion 的 2 张谱来自 2 个 set、Flash Me Back 的
   2 张来自 2 个 set —— 谱面多样性几乎为零。
3. **Flash Me Back 条件锐度弱**（源网格 116.87 BPM = 175×2/3，1/16 格盖 3 个 1/32 格）。要做细粒度
   监督就应按 **175 BPM 重算该歌 `-cond.npz`**（1/16 格 85.7 ms），而不是沿用现有网格。
4. **split 跨歌不均衡**：test 只有 Flash Me Back、val 只有 Xterfusion；按谱分组无泄漏，但 val/test
   音源不同，「验证损失」不可直接横比。
5. **tier B 的 BPM 口径**：若评审要求 BPM 严格 ≤2%，Flash Me Back 的 112 窗应整体剔除，数据集只剩
   53 窗（Xterfusion）。报告里两个口径都给了，别混用。
6. 2628774（Xterfusion）的 .osz 音频比本地短 **2.091 s**（167.919 vs 170.040，ogg vs mp3）：时长差在
   3 s 内故接受，但本地最后约 2 s 音频没有对应谱面内容。

## 6. 复现

```bash
# 搜索（看 ChildrenBeatmaps[].Mode == 3）+ 下载，总量 117MB
curl -s "https://osu.direct/api/search?q=Xterfusion&ps=20"
curl -sL "https://osu.direct/api/d/2614015" -o out/diff-data-audio/_dl/t_2614015.osz

# 验证 + 建库（纯 CPU，numpy，约 8 秒）
python3 tools/build-audio-cond-windows.py \
    --osz-root out/diff-data-audio/_dl --cond-root out/multisong \
    --out out/diff-data-audio/dataset.npz --report out/diff-data-audio/report.json

# 抽查时间对齐（3 个窗口）
python3 tools/spotcheck-audio-cond-align.py
```

产物：`dataset.npz`、`report.json`（逐 set 验证表 + 统计）、`spotcheck.json`、`align-probe.json`、
`_dl/`（.osz 原文与解出的 .osu/音频）。

## 7. 下一步（训练侧）

1. **接入方式**：`dataset.npz` 里 `C` 与 `X` 行一一对应，可直接当 `--audio-cond` 用；但
   `tools/train-diffusion.py` 的「按 (歌名, 起始格) 配对」分支会**撞格**（两首歌 1/32 格长不同，
   起始格会重复）——应改成按 `chart_id + start_cell` 配对，或直接走行对齐路径。
2. **用 Xterfusion 53 窗做踩音验证集**（tier A、音频 md5 同一录音、扫描峰在 0 格），
   Flash Me Back 的 112 窗只当「密度先验」；不要用 FMB 评估「音符是否压 onset」。
3. **数据规模是硬约束**：下一步应优先扩歌（有音频 + 有 mania 4K 谱），而不是在这 165 窗上堆 epoch；
   扩歌时沿用本报告的验证脚本，BPM 与时长两条判据都别省。
