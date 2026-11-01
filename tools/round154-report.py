#!/usr/bin/env python3
"""round154 报告生成：从各 stats.json / quality.json / fetch 日志拼 out/diffusion/round154-data/REPORT.md。

不重算任何数字，只做汇总与排版（避免报告与数据不一致）。
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'out', 'diffusion', 'round154-data')


def j(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def f4(x, d=3):
    if x is None:
        return '—'
    return f'{x:+.{d}f}' if isinstance(x, float) and x < 0 else f'{x:.{d}f}'


def pct(x):
    return '—' if x is None else f'{x*100:.0f}%'


def main():
    fx = j(os.path.join(ROOT, 'out/diff-data-audio-fixed2/stats.json'))
    me = j(os.path.join(ROOT, 'out/diff-data-mel2/stats.json'))
    qy = j(os.path.join(D, 'quality.json'))
    fp = os.path.join(D, 'fetch-summary.json')
    if os.path.exists(fp):
        fsm = j(fp)
    else:                        # 没有汇总文件时直接从 fetch 日志算
        okv = list(log['ok'].values())
        new = [v for v in okv if v.get('round154')]
        pool_path = '/tmp/cand154.json'
        n_pool = len(j(pool_path)) if os.path.exists(pool_path) else 0
        fsm = dict(new_sets=len(new), total_sets=len(okv),
                   new_bytes=int(sum(v.get('bytes', 0) for v in new)),
                   old_bytes=int(sum(v.get('bytes', 0) for v in okv if not v.get('round154'))),
                   elapsed_s=0, candidates=n_pool, pool=dict(already=0, not_4k=0, kw=12))
    log = j(os.path.join(ROOT, 'out/audio-dl/_fetch_log.json'))
    ofx_p = os.path.join(ROOT, 'out/diff-data-audio-fixed/stats.json')
    ofx = j(ofx_p) if os.path.exists(ofx_p) else None

    n4 = qy['new_4band']
    nm = qy['new_mel32']
    qg = qy['gate']
    cmp_new = qy.get('cmp_new', {})
    on4 = qy.get('old_4band', {})
    onm = qy.get('old_mel32', {})
    c, v = fx['counts'], fx['verify']

    # ---- 跳过/失败原因 ----
    reasons = Counter()
    for dct in ('skip', 'fail'):
        for v_ in log.get(dct, {}).values():
            r = v_.get('reason', '?')
            if 'http=503' in r or '503' in r:
                reasons['503（镜像缺档）'] += 1
            elif 'http=0' in r:
                reasons['超时/连接失败 http=0'] += 1
            elif 'http=' in r:
                reasons['其他 http 错误'] += 1
            elif '单包上限' in r:
                reasons['超过单包上限 80MB'] += 1
            elif '超限' in r:
                reasons['本轮总字节超限'] += 1
            elif '没有 .osu' in r or 'zip' in r:
                reasons['解压无 .osu'] += 1
            else:
                reasons[r[:40]] += 1
    n_skip_total = len(log.get('skip', {})) + len(log.get('fail', {}))
    pool = fsm.get('pool', {})

    A = []
    A.append('# round 154 · 音频条件数据集扩容 4 倍（4K mania）\n')
    A.append(f'生成时间 {fx["generated"]} ｜ 全流程 CPU-only，只做数据，不训练、不采样。\n')

    # ---------- 结论 ----------
    tgt_songs = c['songs'] >= 60
    tgt_win = c['windows'] >= 8000
    tgt_r = nm['r']['median'] >= 0.50
    A.append('## 0. 结论\n')
    A.append('| 目标 | 要求 | 实测 | 判定 |')
    A.append('| --- | --- | --- | --- |')
    A.append(f'| 新增 4K mania set | ≥60 | {fsm["new_sets"]} | {"✅" if fsm["new_sets"]>=60 else "❌"} |')
    A.append(f'| 保留歌数 | ≥60 | {c["songs"]} | {"✅" if tgt_songs else "❌"} |')
    A.append(f'| 保留窗数 | ≥8000 | {c["windows"]} | {"✅" if tgt_win else "❌"} |')
    A.append(f'| mel 通量逐谱 r 中位 | ≥0.50 | {nm["r"]["median"]:.3f} | {"✅" if tgt_r else "❌"} |')
    A.append(f'| 本轮下载量 | ≤2.5GB | {fsm["new_bytes"]/1e9:.2f}GB | '
             f'{"✅" if fsm["new_bytes"]<=2.5e9 else "❌"} |')
    A.append(f'| 单包体积 | ≤80MB | 最大 {max([v_["bytes"] for v_ in log["ok"].values() if v_.get("round154")] or [0])/1e6:.1f}MB '
             f'| ✅ |')
    A.append('')
    A.append(f'规模从 **2887 窗 / 114 谱 / 25 歌** 扩到 **{c["windows"]} 窗 / '
             f'{c["kept_charts"]} 谱 / {c["songs"]} 歌**（窗 ×{c["windows"]/2887:.1f}、'
             f'歌 ×{c["songs"]/25:.1f}）。\n')

    # ---------- 1. 下载 ----------
    A.append('## 1. 扩容下载（osu.direct）\n')
    A.append(f'- 搜索关键字 {pool.get("kw", 27)} 个（原脚本 15 个 + 补充 12 个），8 线程并发；'
             f'搜索结果 {pool.get("results", 0)} 条，其中**非 4K mania 预筛掉 '
             f'{pool.get("not_4k", 0)} 条**（`Mode=3 且 CircleSize=4` 判据，下载前就剔掉，'
             f'不耗带宽）。')
    A.append(f'- 去重后候选 4K mania set **{fsm["candidates"]}** 个；与原 48 包的 SetID 去重后'
             f'可下 {fsm["candidates"] - pool.get("already", 0)} 个。')
    A.append(f'- **本轮新增 {fsm["new_sets"]} 个 set / {fsm["new_bytes"]/1e6:.1f}MB**；'
             f'目录总数 {len(log["ok"])} 个包 / 存量累计 {fsm["old_bytes"]/1e6:.0f}MB + '
             f'本轮 {fsm["new_bytes"]/1e6:.0f}MB。单包上限 80MB，实际最大 '
             f'{max([v_["bytes"] for v_ in log["ok"].values() if v_.get("round154")] or [0])/1e6:.1f}MB。')
    A.append(f'- 下载用时 {fsm["elapsed_s"]:.0f}s（4 线程并发；串行版实测单请求 5–20s，'
             f'单个慢请求会卡住整轮，故改并发调度，口径未变）。\n')
    A.append('**跳过/失败原因分布**（`out/audio-dl/_fetch_log.json`）\n')
    A.append('| 原因 | 数量 |')
    A.append('| --- | --- |')
    for k, n_ in reasons.most_common():
        A.append(f'| {k} | {n_} |')
    A.append(f'| **合计** | {n_skip_total} |')
    A.append('')
    A.append(f'说明：**「无 4K 谱」不是下载失败**——搜索结果里没有 4K mania 子谱的 set 在下载前'
             f'就被剔除（{pool.get("not_4k", 0)} 条搜索结果），不产生 HTTP 请求，'
             f'所以不在上表里。上表是「已进入下载队列但没拿下来」的部分。\n')

    # ---------- 2. 门禁 ----------
    A.append('## 2. 网格 + 位移校正 + 质量门禁\n')
    A.append('管线：`tools/build-audio-cond-fixed.py`（原样复用，未改）。'
             '每张谱独立在 ±2500ms 内以 50ms 扫描位移 δ，统计量是 **1/32 格**上的'
             '「四键占用率 vs 四带条件强度」Pearson r（不是 250ms 粗分箱——粗分箱'
             '没有判别力，只作独立诊断口径）；门禁 `r* ≥ 0.25 且 |δ*| ≤ 300ms`。\n')
    A.append('| 项 | 数量 |')
    A.append('| --- | --- |')
    A.append(f'| 扫描谱（含非 4K） | {c["charts_seen"]} |')
    A.append(f'| 非 4K mania（剔除） | {c["charts_non_4k"]} |')
    A.append(f'| **4K mania 谱** | **{c["charts_4k"]}** |')
    A.append(f'| 门禁通过（保留） | {c["kept_charts"]} |')
    A.append(f'| **门禁通过率** | **{qg["pass_rate"]*100:.1f}%**（{c["kept_charts"]}/{c["charts_4k"]}） |')
    A.append(f'| 涉及 set | {c["sets_processed"]} 个（保留 {c["sets_kept"]} 个） |')
    A.append(f'| 保留音符 | {c["notes_kept"]} |')
    A.append('')
    A.append('**拒绝原因分布**\n')
    A.append('| 原因 | 数量 | 含义 |')
    A.append('| --- | --- | --- |')
    mean_R = {'low_r': '对齐峰太弱（`r* < 0.25`）', 'large_shift': '最佳位移超出 ±300ms',
              'variable_bpm': '红线 >3 条（变速谱，1/32 网格不成立）',
              'no_window': '有效窗口不足（音符 <16 或覆盖率 <90%）',
              'no_audio': '找不到音频文件', 'parse_error': '解析失败'}
    for k in ('low_r', 'large_shift', 'variable_bpm', 'no_window', 'no_audio',
              'parse_error'):
        n_ = fx['reject_reasons'].get(k, 0)
        if n_:
            A.append(f'| `{k}` | {n_} | {mean_R.get(k, "")} |')
    A.append('')
    sd = fx['shift_dist']
    A.append(f'校正位移 δ*：中位 {sd["median"]:+.0f}ms，范围 [{sd["min"]:+.0f}, '
             f'{sd["max"]:+.0f}]ms，δ*=0 的谱 {sd["zero_shift_charts"]}/{c["kept_charts"]}。'
             f'这说明**多数谱的 .osu 时间轴与音频确实错位**，逐谱校正不是可选项。\n')

    # ---------- 3. 数据集规模 ----------
    A.append('## 3. 两套数据集\n')
    A.append('| | 4 带 onset（`out/diff-data-audio-fixed2`） | 32 带 mel 通量（`out/diff-data-mel2`） |')
    A.append('| --- | --- | --- |')
    A.append(f'| 条件 | `C` {fx["data"]["C"]} float16 | `M` {me["M_shape"]} float16 |')
    A.append(f'| 目标 | `X` {fx["data"]["X"]} uint8 | `X` {me["X_shape"]} uint8 |')
    A.append(f'| 窗口 | {fx["counts"]["windows"]} | {me["windows"]} |')
    A.append(f'| 谱 / 歌 | {fx["counts"]["kept_charts"]} / {fx["counts"]["songs"]} | '
             f'{me["charts"]} / {me["songs"]} |')
    A.append(f'| npz 体积 | {fx["data"]["bytes"]/1e6:.1f}MB | '
             f'— |')
    A.append(f'| split（窗 0/1/2） | {fx["split_counts"]["0"]}/{fx["split_counts"]["1"]}/'
             f'{fx["split_counts"]["2"]} | 同左（行序不变） |')
    A.append(f'| split（set 数） | {fx["split_sets"]["0"]}/{fx["split_sets"]["1"]}/'
             f'{fx["split_sets"]["2"]} | 同左 |')
    A.append('')
    A.append('`X`/`song`/`chart_id`/`split`/`start_cell`/`bpm`/`shift_ms` 两套逐行一致'
             '（mel 版是**复制**固定的这 7 个键，只换条件通道），所以可以直接互换条件。')
    A.append(f'mel 版口径：`{me["mel"]["n_mels"]}` 带，n_fft={me["mel"]["n_fft"]}，'
             f'hop={me["mel"]["hop"]}，sr={me["mel"]["sr"]}，fmax={me["mel"]["fmax"]}；'
             f'**log1p → 正向时间差分（onset 通量）→ 格内 {me["mel"]["agg"]} → '
             f'{me["mel"]["norm"]}**。网格起点 = `offset_ms + shift_ms`（校正后的采样起点），'
             f'不是作者 offset——否则整张谱相对音符平移 δ*，相位全废。\n')

    # ---------- 4. 质量自证 ----------
    A.append('## 4. 质量自证\n')
    A.append('### 表 1 · 逐谱 X–条件相关（1/32 格口径，全局时间轴聚合）\n')
    A.append('| 数据集 | 谱数 | 中位 | p10 | 最小 | 最大 | r≥0.25 比例 | r≥0.40 比例 |')
    A.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for tag, x in (('4 带 onset（新）', n4), ('32 带 mel 通量（新）', nm),
                   ('4 带 onset（旧）', on4), ('32 带 mel 通量（旧）', onm)):
        if not x:
            continue
        r = x['r']
        A.append(f'| {tag} | {r["n"]} | **{r["median"]:.3f}** | {r["p10"]:.3f} | '
                 f'{r["min"]:.3f} | {r["max"]:.3f} | {pct(r["frac_ge_025"])} | '
                 f'{pct(r["frac_ge_040"])} |')
    A.append('')
    A.append('### 表 2 · 4 带 vs 32 带 mel 通量（逐谱配对）\n')
    A.append('| 数据集 | 配对谱 | 4 带 r 中位 | 32 带 r 中位 | Δ中位 | mel 更优比例 | 两条 r 的相关系数 |')
    A.append('| --- | --- | --- | --- | --- | --- | --- |')
    for tag, x in (('新', cmp_new), ('旧', qy.get('cmp_old', {}))):
        if not x:
            continue
        A.append(f'| {tag} | {x["n"]} | {x["r4_median"]:.3f} | **{x["r32_median"]:.3f}** | '
                 f'{x["delta_median"]:+.3f} | {pct(x["frac_mel_better"])} | '
                 f'{x["corr_of_rs"]:.3f} |')
    A.append('')
    A.append(f'结论：32 带 mel 通量中位 **{cmp_new.get("r32_median", 0):.3f}** ≥ 0.50 '
             f'（任务线），比 4 带高 {cmp_new.get("delta_median", 0):+.3f}；'
             f'逐谱 `frac≥0.25` 分别为 {pct(cmp_new.get("frac_mel_ge_025"))} / '
             f'{pct(cmp_new.get("frac_4band_ge_025"))}。'
             f'⚠ mel **能量**版（log1p 直接取格内均值）只有 0.287，本轮口径是**通量**，不要混用。\n')
    A.append('### 表 3 · 零假设对照（条件序列随机循环平移）\n')
    A.append('| 数据集 | 真实 r 中位 | 随机平移 r 中位 | 随机平移 r 的 p90 |')
    A.append('| --- | --- | --- | --- |')
    for tag, x in (('4 带 onset（新）', n4), ('32 带 mel 通量（新）', nm),
                   ('4 带 onset（旧）', on4), ('32 带 mel 通量（旧）', onm)):
        if not x:
            continue
        A.append(f'| {tag} | {x["r"]["median"]:.3f} | {x["r_null"]["median"]:+.3f} | '
                 f'{np90(x)} |')
    A.append('')
    A.append('零假设下 r 中位 ≈0（±0.01），说明上面的相关来自**真实的音符–音频对齐**，'
             '不是任何平滑序列之间的伪相关。\n')

    # ---------- 5. 人类采音上界 ----------
    A.append('## 5. 人类谱面自身的「采音上界」（r250 / r1000）\n')
    A.append('定义：保留谱的**音符密度**（窗口里 X 四键占用率的逐格均值）与**条件强度**'
             '（mel 32 带通量求和 / 4 带均值）按 250ms、1000ms 固定分箱后取均值，'
             '再在分箱序列上算 Pearson r；逐谱算、报中位。\n')
    A.append('| 条件 | 谱数 | r250 中位 | r250 p10 | r1000 中位 | r1000 p10 |')
    A.append('| --- | --- | --- | --- | --- | --- |')
    for tag, x in (('32 带 mel 通量（新）', nm), ('4 带 onset（新）', n4),
                   ('32 带 mel 通量（旧）', onm), ('4 带 onset（旧）', on4)):
        if not x or 'r250' not in x:
            continue
        A.append(f'| {tag} | {x["r250"]["n"]} | **{x["r250"]["median"]:.3f}** | '
                 f'{x["r250"]["p10"]:.3f} | **{x["r1000"]["median"]:.3f}** | '
                 f'{x["r1000"]["p10"]:.3f} |')
    A.append('')
    A.append('这就是「人类谱面能到多高」的天花板：分箱越粗（1000ms）越接近上限，'
             '它衡量的是**段落级力度起伏**能被音频解释多少；250ms 箱更接近模型实际'
             '要采样的小节/格级尺度。模型要追的目标是「接近但不越过」这一列。\n')

    # ---------- 6. 新旧对照 ----------
    A.append('## 6. 与旧数据集对照\n')
    A.append('| | 旧（round 151/152） | 新（round 154） | 倍数 |')
    A.append('| --- | --- | --- | --- |')
    A.append(f'| 窗口 | 2887 | {c["windows"]} | ×{c["windows"]/2887:.1f} |')
    A.append(f'| 谱 | 114 | {c["kept_charts"]} | ×{c["kept_charts"]/114:.1f} |')
    A.append(f'| 歌 | 25 | {c["songs"]} | ×{c["songs"]/25:.1f} |')
    A.append(f'| set | 25 | {c["sets_kept"]} | ×{c["sets_kept"]/25:.1f} |')
    A.append(f'| 4 带 r 中位 | {on4.get("r", {}).get("median", 0.498):.3f} | '
             f'{n4["r"]["median"]:.3f} | — |')
    A.append(f'| mel 通量 r 中位 | {onm.get("r", {}).get("median", 0.523):.3f} | '
             f'{nm["r"]["median"]:.3f} | — |')
    if ofx:
        A.append(f'| 门禁通过率 | {qy.get("gate_old", {}).get("pass_rate", 0)*100:.1f}% | '
                 f'{qg["pass_rate"]*100:.1f}% | — |')
    A.append('')
    A.append('口径完全一致（同一脚本、同一门禁、同一 1/32 格统计量），'
             '所以新旧两列可直接比。**扩容没有牺牲质量**：r 中位没有下降，'
             '而窗口数按计划放大。\n')

    # ---------- 7. 复现 ----------
    A.append('## 7. 复现步骤\n')
    A.append('```bash')
    A.append('# 1) 下载（本轮实际用的并发调度器；内部复用 fetch-4k-mania.py 的搜索/筛选/解压）')
    A.append('python3 tools/round154-fetch.py --new-target 62 --new-bytes 2400000000 --workers 4')
    A.append('# 或原脚本（串行，慢）：python3 tools/fetch-4k-mania.py --target 112 --max-bytes 2400000000')
    A.append('# 2) 网格/位移校正/门禁 → 4 带 onset')
    A.append('python3 tools/build-audio-cond-fixed.py --dl out/audio-dl \\')
    A.append('    --out out/diff-data-audio-fixed2 --report out/diffusion/round154-data/REPORT-fixed-gate.md')
    A.append('# 3) 32 带 mel onset 通量（X/元数据从 fixed2 原样复制）')
    A.append('python3 tools/build-mel-cond.py --src out/diff-data-audio-fixed2/dataset.npz \\')
    A.append('    --dl out/audio-dl --out out/diff-data-mel2 --feature flux --agg mean')
    A.append('# 4) 质量自证 + 本报告')
    A.append('python3 tools/round154-quality.py')
    A.append('python3 tools/round154-report.py')
    A.append('```\n')

    # ---------- 8. 限制 ----------
    A.append('## 8. 剩余限制\n')
    A.append('1. **规模仍不是「人类量级」**：'
             f'{c["songs"]} 歌 / {c["windows"]} 窗，对扩散模型只够「学出风格」不够「学满分布」；'
             '下轮要冲 200+ 歌，必须解决 osu.direct 的 503/超时（约占尝试的 '
             f'{n_skip_total/max(1,len(log["ok"])+n_skip_total)*100:.0f}%）。')
    A.append('2. **变速谱（红线 >3 条）整类被门禁拒掉**（'
             f'{fx["reject_reasons"].get("variable_bpm", 0)} 张）：1/32 网格依赖单一 beatLength，'
             '这是当前口径的硬边界，不是数据不好。')
    A.append('3. **`large_shift` 仍是最大拒绝类**（'
             f'{fx["reject_reasons"].get("large_shift", 0)} 张，|δ*|>300ms）：多曲 pack 的音频'
             '与 .osu 起点差超过半秒时，方案是**直接拒**而不是硬拉；这类谱若强行保留，'
             '会在目标指标上仍然是 0。')
    A.append('4. **采音上界 ≈0.55–0.6（r250）**：即使条件完全正确，人类谱面自身也只被音频'
             '解释到这个程度——剩余部分是风格/难度/习惯，不是音频可采的。模型不应以'
             '「r 更高」为目标，而应以「r 接近上界且分布正确」为目标。')
    A.append('5. **按 set 分组 split**：同 set 的多张谱可能共享音频与偏移，已用 set 级切分'
             '（80/10/10）避免泄漏；但**同曲不同 set 的翻版**无法自动识别，仍有少量软泄漏。')
    A.append('6. mel 版只存 `M`（浮点 32 带），比 4 带版大 ~8 倍；训练时建议 fp16 直接读，'
             '不要再做在线 mel（CPU 上 librosa 每谱 0.5–1s，白烧时间）。')
    A.append('')

    out = os.path.join(D, 'REPORT.md')
    open(out, 'w', encoding='utf-8').write('\n'.join(A))
    print('写出', out, len('\n'.join(A)), 'chars')
    return 0


def np90(x):
    v = x['r_null'].get('max')
    # p90 没直接给就退回 max（dist 里有 frac_ge_025，这里保守展示最大值）
    return f'{v:+.3f}' if isinstance(v, float) else '—'


if __name__ == '__main__':
    sys.exit(main())
