#!/usr/bin/env python3
"""数据集自动衔接器：等下载完成 → 补特征库 → 补张量 → 报告最终盘点。

    python3 tools/dataset-pipeline.py        # 前台守护（也可 nohup 后台）

下载状态来自 out/fetch-mug-status.json（fetch-mug-list.mjs 写）。
feature-db / mug_tensors 都是增量续传，重复跑无副作用。
"""
import json
import os
import subprocess
import sys
import time

ROOT = '/workspace/dsh000/06-chart-ai'
STATUS = os.path.join(ROOT, 'out/fetch-mug-status.json')
os.chdir(ROOT)


def sh(cmd):
    print('+', cmd, flush=True)
    return subprocess.call(cmd, shell=True)


def main():
    waited = 0
    while True:
        d = json.load(open(STATUS))
        if d.get('done'):
            print(f'下载完成：ok={d["ok"]} fail={d["fail"]}（非4K {d.get("notMania")}）', flush=True)
            break
        if waited and waited % 600 == 0:
            print(f'[等待] have {d["have"]}/{d["total"]}  rate {d.get("rate")} /s  ETA {d.get("etaMin")} min', flush=True)
        time.sleep(30)
        waited += 30

    print('== 补 21 维控制特征库 ==', flush=True)
    sh('python3 tools/build-feature-db.py out/charts -o out/feature-db.jsonl --workers 8')
    print('== 补 Mug 格式张量 ==', flush=True)
    sh('python3 tools/build-mug-tensors.py out/charts -o out/mug_tensors')

    n_charts = len([f for f in os.listdir('out/charts') if f.endswith('.osu')])
    n_feat = sum(1 for _ in open('out/feature-db.jsonl'))
    n_tensors = len([f for f in os.listdir('out/mug_tensors') if f.endswith('.npz')])
    report = {
        'charts': n_charts, 'features': n_feat, 'tensors': n_tensors,
        'audioSets': len([d for d in os.listdir('/tmp/osz_x')]) if os.path.isdir('/tmp/osz_x') else 0,
        'done': True,
    }
    json.dump(report, open('out/dataset-final.json', 'w'), indent=1)
    print('最终盘点：', report, flush=True)


if __name__ == '__main__':
    main()
