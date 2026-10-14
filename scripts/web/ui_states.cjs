// Screenshot every interface state at three viewport sizes, without running the GPU pipeline.
// node scripts/web/ui_states.cjs <base-url> <out-dir> [label]
// Completed, failed, cancelled and interrupted states come from real runs in the server's history.
// "empty" and "running" are MOCKED by rewriting /api/jobs in the browser (file names say so).
// Screenshots contain dataset imagery: keep them outside Git. Writes ui_states.json with overflow and
// console-error results.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');

const [BASE = 'http://127.0.0.1:8765', OUT = 'D:/TAORG_runs/web-qa/ui', LABEL = 'after'] = process.argv.slice(2);
const SIZES = [['desktop', 1440, 900], ['laptop', 1280, 720], ['narrow', 390, 844]];

(async () => {
  fs.mkdirSync(OUT, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const jobs = await (await fetch(BASE + '/api/jobs')).json();
  const pick = s => jobs.find(j => j.status === s && (s !== 'completed' || (j.summary && j.summary.steps > 10)));
  const results = [];
  try {
    for (const [size, w, h] of SIZES) {
      for (const state of ['empty', 'upload', 'running', 'failed', 'cancelled', 'interrupted', 'completed']) {
        const page = await browser.newPage({viewport: {width: w, height: h}});
        const errors = [];
        page.on('pageerror', e => errors.push(e.message));
        page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
        let target = null;
        if (state === 'empty') await page.route('**/api/jobs', r => r.fulfill({json: []}));
        if (state === 'running') {
          const base = pick('completed');
          const now = Date.now() / 1000;
          const fake = {...base, id: base.id, status: 'running', stage: 'sgg', completed_stages: ['perception'],
            summary: null, created: now - 80, error: null,
            stage_times: {perception: [now - 78, now - 20], sgg: [now - 20, null]}};
          await page.route('**/api/jobs', r => r.fulfill({json: [fake, ...jobs.filter(j => j.id !== base.id)]}));
          target = base.id;
        } else if (!['empty', 'upload'].includes(state)) target = (pick(state) || {}).id;
        await page.goto(BASE);
        await page.waitForSelector('#history .historyrow, #history p', {timeout: 15000});
        if (target) {
          const more = page.locator('#historyMore');
          if (await more.count() && await more.isVisible() && !(await page.locator(`.historyrow[data-id="${target}"]`).count())) await more.click();
          await page.locator(`.historyrow[data-id="${target}"]`).click();
          await page.waitForTimeout(1200);
        }
        if (state === 'upload') {
          const sample = 'D:/TAORG_runs/web-samples/MOT17-09_sample.mp4';
          if (fs.existsSync(sample)) await page.locator('#video').setInputFiles(sample);
          await page.waitForTimeout(1500);
        }
        if (state === 'completed') {
          await page.locator('#vSlider').fill('10');
          await page.waitForTimeout(800);
        }
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
        const file = `${LABEL}_${size}_${state}${['empty', 'running'].includes(state) ? '_MOCKED' : ''}.png`;
        await page.screenshot({path: path.join(OUT, file), fullPage: true});
        results.push({size, state, file, overflow, errors});
        console.log(size, state, 'overflow', overflow, errors.length ? JSON.stringify(errors) : '');
        await page.close();
      }
    }
  } finally {
    fs.writeFileSync(path.join(OUT, `${LABEL}_ui_states.json`), JSON.stringify(results, null, 1));
    await browser.close();
  }
})();
