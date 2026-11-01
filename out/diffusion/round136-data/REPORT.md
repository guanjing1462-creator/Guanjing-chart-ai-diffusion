# Round136 · GJ 数据可信划分与验收

## 实际清点与去重

- 原始 .osu：2617；接收文件：2601；拒收：16。
- 字节级重复：19；额外玩法内容重复：11；共去重：30；唯一谱图：2571。
- 同源分组：175（可划分 175）；身份不明隔离谱：0。
- 固定 seed=136，按组数分 train/val/test，val=10.0%、test=10.0%、其余train；不拆大组凑图数。
- 玩法SHA256包括红线时间/拍长/拍号、每键起止时刻（含LN尾），忽略标签、文件名、音效和排版；所有原文件路径及字节SHA256仍保留。

| 划分 | 唯一谱图 | 同源组 | 原文件（含副本） |
|---|---:|---:|---:|
| train | 2299 | 141 | 2323 |
| val | 148 | 17 | 148 |
| test | 124 | 17 | 130 |
| quarantine | 0 | 0 | 0 |

### 风格计数

多style重复谱同时计入各style membership，因此下面四行不能直接相加；manifest另给可加总的 primary_style 固定顺序统计，不擅自裁决标签冲突。

| style | 原始文件 | 接收文件 | 唯一内容membership | train | val | test | 隔离 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tech | 538 | 527 | 526 | 478 | 40 | 8 | 0 |
| jack | 1051 | 1048 | 1045 | 940 | 32 | 73 | 0 |
| speed | 484 | 483 | 480 | 386 | 51 | 43 | 0 |
| stamina | 544 | 543 | 540 | 509 | 25 | 6 | 0 |

## 无交集断言

在已观测身份的传递闭包内，按同set、标题+艺人（含Unicode）、音频字节hash、非通用音频引用名、重复玩法hash合组。以下均为本次实际计算；不是未标识音频的绝对无同曲证明。

| 对比 | group_id | chart_id | 玩法hash | 原文件hash | 身份key |
|---|---:|---:|---:|---:|---:|
| train/val | 0 | 0 | 0 | 0 | 0 |
| train/test | 0 | 0 | 0 | 0 | 0 |
| val/test | 0 | 0 | 0 | 0 | 0 |

- 所有交集为零：True。每张谱恰好属于一组：True。
- 下游先固定组划分再切窗，所有同图窗口继承同一split；不能再调用窗口randperm划分。

## 实证问题与旧NPZ验收边界

- **window_random_split**：窗口 randperm 后切 train/val，没有 chart/song 分组；同曲、重叠窗口可被分到两侧。没有旧窗口来源，不能证明某次历史切分的实际同曲交集为零。 证据：`tools/train-diffusion.py:302-309; tools/train-diffusion32c.py:340-354`。
- **timing_section_and_tempo**：timing() 扫全文件而非 TimingPoints 节，只取首个正 beatLength；HitObjects 可被误读为 timing。变速检测读最后字段 effects 而非第7字段 uninherited，未支持分段变速；量化忽略红线 offset，span 又按首尾差计算，可能截断。 证据：`tools/diff-data-true.py:19-30,69-90`。
- **estimated_grid**：前两版以相邻行间隔中位数猜网格，未解析 TimingPoints，不能当真实 1/16 或 1/32 拍点。 证据：`tools/diff-data.py:53-79; tools/diff-data32.py:154-181`。
- **row_rate_not_nps**：len(rows)/秒是 RPS（同行和弦合并后行率），不是按各键起点计数的 NPS。密度/占位率桶也不是经过校准的真实难度。 证据：`tools/diff-data.py:68-70; tools/diff-data32.py:166-167; tools/diff-data-true.py:81-82`。
- **ln_duration_lost**：type&128 仅留下 LN 起点，结束时间及 hold/release 均未保存；旧二值卷无法恢复 LN 时值。 证据：`tools/diff-data.py:16-43; tools/diff-data32.py:57-84; tools/diff-data-true.py:34-49,86-91`。
- **partial_provenance_not_a_certificate**：cond-index 是6首音频的内容重放候选索引；歧义时仍取 picks[0]，不是全库唯一 source_id。unresolved_provenance=0 不代表每行得到唯一曲目身份。 证据：`tools/build-cond-index.py:326-360; out/diff-data/cond-index.json:2-6`。

本次原谱实数（重复文件也计数，避免与唯一谱统计混淆）：

- `accepted_charts_with_ln` = 1980
- `accepted_charts_with_tempo_changes` = 277
- `accepted_distinct_onset_rows` = 5745584
- `accepted_ln_heads` = 187259
- `accepted_note_heads` = 9940615
- `legacy_red_count_differs_from_actual_red_count` = 2567
- `legacy_timing_first_candidate_outside_timingpoints` = 378
- `legacy_variable_bpm_skip_predicate` = 168

旧NPZ仅以标准库读取ZIP/NPY头（不加载torch、不训练），不是来源重建：

- `out/diff-data/dataset.npz`：`cannot_prove_no_leakage`；X:[30551, 4, 128] |u1; diff:[30551] |u1; style:[30551] |u1。
  缺少 `group_id`, `source_id`, `split`, `start_cell`，**不能证明旧NPZ无同曲/重叠窗泄漏**。
- `out/diff-data32/dataset.npz`：`cannot_prove_no_leakage`；X:[88787, 4, 256] |u1; diff:[88787] |u1; style:[88787] |u1。
  缺少 `group_id`, `source_id`, `split`, `start_cell`，**不能证明旧NPZ无同曲/重叠窗泄漏**。
- `out/diff-data-true/dataset.npz`：`cannot_prove_no_leakage`；X:[23553, 4, 256] |u1; diff:[23553] |u1; style:[23553] |u1。
  缺少 `group_id`, `source_id`, `split`, `start_cell`，**不能证明旧NPZ无同曲/重叠窗泄漏**。
- `out/diff-data/cond-index.json`：`partial_audio_index_not_full_source_certificate`；{"dataset_windows": 30551, "matched_windows": 0, "unresolved_provenance": 0, "ambiguous_multi_source": 0, "no_leakage_proven": false}。

## 冲突、缺失与拒收

冲突记录 194 条；详见manifest.conflicts，绝不静默选择其中一个身份。多set/多音频/多标题并不自动判错（练习包可能合法），统一保守合组。

- 缺失 `Artist_or_ArtistUnicode`：11 个接收原文件。
- 缺失 `audio_file_unavailable`：2601 个接收原文件。
- 缺失 `positive_BeatmapSetID`：4 个接收原文件。
- 拒收 `out/packs/tech/3556066.osu`：conflicting_red_timing:2103。
- 拒收 `out/packs/tech/3556067.osu`：conflicting_red_timing:1931。
- 拒收 `out/packs/tech/3556068.osu`：conflicting_red_timing:1785。
- 拒收 `out/packs/tech/3556069.osu`：conflicting_red_timing:1661, conflicting_red_timing:1661。
- 拒收 `out/packs/tech/3556070.osu`：conflicting_red_timing:1553, conflicting_red_timing:1553。
- 拒收 `out/packs/tech/3556071.osu`：conflicting_red_timing:1458。
- 拒收 `out/packs/tech/3556072.osu`：conflicting_red_timing:1375, conflicting_red_timing:1375。
- 拒收 `out/packs/tech/3556073.osu`：conflicting_red_timing:1301, conflicting_red_timing:1301。
- 拒收 `out/packs/tech/3556074.osu`：conflicting_red_timing:1234。
- 拒收 `out/packs/tech/3556075.osu`：conflicting_red_timing:2269。
- 拒收 `out/packs/tech/5793616.osu`：conflicting_red_timing:57436。
- 拒收 `out/packs/jack/5629889.osu`：conflicting_red_timing:0。
- 拒收 `out/packs/jack/5629890.osu`：conflicting_red_timing:0。
- 拒收 `out/packs/jack/5630000.osu`：conflicting_red_timing:0。
- 拒收 `out/packs/speed/5931366.osu`：conflicting_red_timing:87502.36, conflicting_red_timing:106933.421。
- 拒收 `out/packs/stamina/5183464.osu`：conflicting_red_timing:12191, conflicting_red_timing:16364, conflicting_red_timing:16883, conflicting_red_timing:17792, conflicting_red_timing:18311, conflicting_red_timing:18907, conflicting_red_timing:19282。

## 给后续数据构建器的最小来源接口

- `source_id`：每个窗口的 chart_id（本 manifest 的内容寻址 ID），Unicode/string [N]；不要用行号替代。
- `group_id`：每个窗口继承 chart.group_id，同曲/同源/重复连通分量的 ID，string [N]。
- `split`：每个窗口原样继承 chart.split，train/val/test；quarantine 和 rejected 不切窗。禁止再做窗口随机切分。
- `start_cell`：int64 [N]，在该 chart 固定 timing 原点上的绝对起始格，不得每窗把首音符归零。
- `window_cells`：int，窗口长；区间 [start_cell,start_cell+window_cells)，同时保存 grid_division=8（每拍8格=1/32）。
- `timing`：chart.timing_segments 保存红线 offset_ms/beat_length_ms/beat_start/meter；cell(t)=round(8*beat_at_ms(t))，支持变速。
- `ln`：至少保存 onset/hold/release 三通道，或另存 lane,start_ms,end_ms；跨窗 LN 必须有 carry-in/carry-out，不可只截起点。
- `density`：分别叫 row_rate_rps、note_head_rate_nps、occupancy；密度条件不能标注为真实难度。归一化参数只能在 train 拟合。
- `audit`：保存 manifest_sha256、原始 file_sha256、content_sha256、parser/grid 版本；按 group_id/source_id/content hash 验证集间无交集，按 source_id 和半开格区间检查重叠窗。旧 NPZ 需重新从原谱构建来源，不能按相似窗口猜唯一来源。

## 限制与复现

- 此清单尚未接入任何训练器；旧窗口随机切分的验证数值不能据此升级为无泄漏指标。
- 分组保证覆盖已观测身份的传递闭包，不证明未标识、改名、转码或缺音频文件实际不是同曲；元数据无法提供音频指纹真值。
- 练习包共用 set ID/标题时整包保守合组，可能很大，图数/风格比例不保证80/10/10；组不能拆开补比例。
- 同内容多style保留全部标签并记录冲突；primary_style仅为可加总统计的固定顺序标签，不是真值裁决。
- 本工具只生成manifest和报告，不生成窗口/谱包、不修改原始.osu/旧NPZ、不训练或评估模型。
- 运行 `python3 tools/gj-data-manifest.py --out-dir <新的空目录>` 可复用；默认种子136。输出文件存在即拒绝覆盖。
- 快测试：`python3 -m unittest discover -s tests -p 'test_gj_data_manifest.py' -v`，全部为临时目录内合成.osu，CPU/标准库，无网络。
- 输入清单指纹：`90098b3c8f7232579eed01156f0f58b619241414791abfa592f925ba85f35fd9`。
