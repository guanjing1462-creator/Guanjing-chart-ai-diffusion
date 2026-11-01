// 在 Node 里跑观景台的 Koishi 分析 worker（与站点同源实现）
import fs from 'node:fs';

const workerFile = new URL('./gj_worker.js', import.meta.url);
const src = fs.readFileSync(workerFile, 'utf8');
// 该文件给 globalThis 赋一个巨大的源码字符串；直接执行它再取回来（避免对大字符串用正则）
const code = new Function(src + '\nreturn globalThis.KOISHI_WORKER_SOURCE;')();
if (!code) { console.error('找不到 KOISHI_WORKER_SOURCE'); process.exit(1); }

const chartPath = process.argv[2];
if (!chartPath) { console.error('用法: node run-koishi.mjs chart.osu'); process.exit(1); }
const text = fs.readFileSync(chartPath, 'utf8');

let resolved = false;
const self = {
  onmessage: null,
  postMessage(msg) {
    if (msg && msg.report) {
      resolved = true;
      console.log(JSON.stringify(msg.report, null, 1));
    } else if (msg && msg.error) {
      console.error('worker error:', msg.error);
      process.exit(2);
    }
  },
  addEventListener() {}, removeEventListener() {},
  location: { href: 'https://guanjingscene.xyz/' },
  navigator: { userAgent: 'node' },
  performance: { now: () => Date.now() },
  setTimeout, clearTimeout, crypto,
  importScripts() {},
};
self.self = self;
self.globalThis = self;

// 先注入 worker 依赖的全局模块
// 依赖模块用真实 globalThis 执行：它们给 globalThis.X 赋值，裸引用才能解析到
for (const dep of ['dep_list-balance.js', 'dep_beatmap-parser.js']) {
  const depSrc = fs.readFileSync(new URL('./' + dep, import.meta.url), 'utf8');
  new Function('self', depSrc)(self);
}

const factory = new Function('self', code + '\n;return self;');
try {
  factory(self);
} catch (e) {
  console.error('worker 初始化失败:', e.message);
  process.exit(3);
}
if (typeof self.onmessage !== 'function') {
  console.error('worker 没有 onmessage');
  process.exit(4);
}
self.onmessage({ data: { action: 'analyze', text, options: {} } });
// 允许微任务/异步完成
const t0 = Date.now();
while (!resolved && Date.now() - t0 < 120000) {
  await new Promise(r => setTimeout(r, 200));
}
if (!resolved) { console.error('超时：没有收到报告'); process.exit(5); }
