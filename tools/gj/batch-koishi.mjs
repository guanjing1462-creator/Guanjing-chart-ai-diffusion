// 批量跑观景台 Koishi 分析：输出 难度值 + 键型分类
import fs from 'node:fs';
import path from 'node:path';

const src = fs.readFileSync(new URL('./gj_worker.js', import.meta.url), 'utf8');
const code = new Function(src + '\nreturn globalThis.KOISHI_WORKER_SOURCE;')();
for (const dep of ['dep_list-balance.js', 'dep_beatmap-parser.js']) {
  const depSrc = fs.readFileSync(new URL('./' + dep, import.meta.url), 'utf8');
  new Function('self', depSrc)({});
}

const files = process.argv.slice(2);
const results = [];
for (const f of files) {
  const text = fs.readFileSync(f, 'utf8');
  let out = null;
  const self = {
    onmessage: null,
    postMessage(msg) {
      if (msg && msg.report) out = msg.report;
      else if (msg && msg.error) out = { __error: msg.error };
    },
    addEventListener() {}, removeEventListener() {},
    location: { href: 'https://guanjingscene.xyz/' },
    navigator: { userAgent: 'node' },
    performance: { now: () => Date.now() },
    setTimeout, clearTimeout, crypto, importScripts() {},
  };
  const factory = new Function('self', code + '\n;return self;');
  factory(self);
  self.onmessage({ data: { action: 'analyze', text, options: {} } });
  const t0 = Date.now();
  while (!out && Date.now() - t0 < 400000) {
    await new Promise(r => setTimeout(r, 100));
  }
  const r = out?.result;
  results.push({
    file: path.basename(f),
    diff: (r && Number.isFinite(r.numericDifficulty)) ? Number(r.numericDifficulty.toFixed(2)) : null,
    tier: r ? r.estDiff : null,
    pattern: r ? r.primaryPattern : null,
    probs: r ? Object.fromEntries(Object.entries(r.probabilities || {})
      .map(([k, v]) => [k, Number(v.toFixed(3))])) : null,
    error: out?.__error ?? (out ? null : 'timeout'),
  });
  const dd = (r && Number.isFinite(r.numericDifficulty)) ? r.numericDifficulty.toFixed(2) : '—';
  console.error(`${path.basename(f)} → ${dd} ${r?.estDiff ?? ''} [${r?.primaryPattern ?? '?'}]`);
}
console.log(JSON.stringify(results, null, 1));
