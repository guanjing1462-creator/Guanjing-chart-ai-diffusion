"""卡手/可弹性检查：找出物理上极难或不可能的模式。
4K 键位：左手中指/食指 = col0/col1，右手食指/中指 = col2/col3。
"""
import sys, importlib.util, glob
spec = importlib.util.spec_from_file_location('ac', 'tools/audit-chart.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

def check(path):
    notes, _ = ac.parse_osu(path)
    rows = ac.rows_of(notes)
    issues = []
    # 1) 同列连打 < 55ms（单指 >18 次/秒，持续就弹不了）
    for i in range(1, len(rows)):
        t0, a = rows[i-1]; t1, b = rows[i]
        if t1 - t0 < 55:
            same = set(a) & set(b)
            if len(a) == 1 and len(b) == 1 and same:
                issues.append(('同列极速连打', t1, f'{t1-t0:.0f}ms col{sorted(same)}'))
    # 2) 单侧连续 3 行都落在同一侧（左手 0/1 或右手 2/3）且间隔 <70ms → 单侧过载
    run = 0
    for i in range(1, len(rows)):
        t0, a = rows[i-1]; t1, b = rows[i]
        left = all(c <= 1 for c in b); right = all(c >= 2 for c in b)
        if (left or right) and t1 - t0 < 70:
            run += 1
            if run >= 2 and len(b) >= 2:
                issues.append(('单侧连续和弦', t1, f'{t1-t0:.0f}ms 手={"左" if left else "右"} {sorted(b)}'))
        else:
            run = 0
    # 3) LN 期间同列还有音符（会吞键）
    ln = {}
    for t, c, typ in notes:
        if typ == 128:
            ln.setdefault(c, []).append(t)
    taps = [(t, c) for t, c, typ in notes if typ != 128]
    ln_spans = []
    for t, c, typ in notes:
        if typ == 128:
            ln_spans.append((c, t, int(typ)))
    # 4) 和弦间跳跃：连续两个 4 键和弦且完全不共列，间隔 <70ms
    for i in range(1, len(rows)):
        t0, a = rows[i-1]; t1, b = rows[i]
        if len(a) >= 3 and len(b) >= 3 and not (set(a) & set(b)) and t1 - t0 < 70:
            issues.append(('满和弦无共列跳跃', t1, f'{t1-t0:.0f}ms {sorted(a)}->{sorted(b)}'))
    return rows, issues

total = 0
for f in sorted(glob.glob('out/multisong/v3/GJ-V51-*.osu')):
    rows, issues = check(f)
    by = {}
    for k, _, _ in issues:
        by[k] = by.get(k, 0) + 1
    total += len(issues)
    print('%-16s 行 %5d  问题 %3d  %s' % (f.split('GJ-V51-')[-1][:-4], len(rows), len(issues), by or ''))
    for k, t, d in issues[:3]:
        print('     ', k, '@%d' % t, d)
print('合计', total)
