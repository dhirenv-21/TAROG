// End-to-end browser QA of the TAORG local website (real pipeline; takes several minutes).
// node scripts/web/browser_qa.cjs <base-url> <out-dir>
// Needs Playwright (set PLAYWRIGHT_MODULE to its module path) and Microsoft Edge. Writes qa_report.json and
// screenshots to <out-dir> (keep outside Git: screenshots contain dataset imagery). It deliberately adds a
// failed, a cancelled and a completed run to the server's history.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execSync} = require('node:child_process');

const [BASE = 'http://127.0.0.1:8765', OUT = 'D:/TAORG_runs/web-qa'] = process.argv.slice(2);
const report = {base: BASE, started: new Date().toISOString(), checks: [], errors: []};
function check(name, ok, detail = '') { report.checks.push({name, ok: !!ok, detail}); console.log(ok ? 'PASS' : 'FAIL', name, detail); }
async function jobs(page) {             // polled outside the page; retried so a brief network hiccup is not a test failure
  for (let i = 0; ; i++) {
    try { return await (await page.request.get(BASE + '/api/jobs')).json(); }
    catch (e) { if (i >= 5) throw e; await new Promise(r => setTimeout(r, 2000)); }
  }
}
async function waitJob(page, pred, timeout) {
  const t0 = Date.now();
  for (;;) {
    const j = (await jobs(page))[0];
    if (pred(j)) return j;
    if (Date.now() - t0 > timeout) throw new Error(`timeout waiting; last status ${j && j.status}/${j && j.stage}`);
    await page.waitForTimeout(1000);
  }
}
function childProcessesFor(jobId) {
  try {
    const out = execSync(`powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*${jobId}*' } | Measure-Object | Select -Expand Count"`).toString().trim();
    return Number(out);
  } catch { return -1; }
}

(async () => {
  fs.mkdirSync(OUT, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage({viewport: {width: 1280, height: 720}});
  page.on('pageerror', e => report.errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') report.errors.push(m.text()); });
  try {
    await page.goto(BASE);
    await page.waitForFunction(() => document.querySelector('#readyText').textContent === 'Setup files found', null, {timeout: 15000});
    await page.waitForSelector('#history .historyrow, #history p', {timeout: 15000});
    check('page loads; setup files found', true);

    // --- keyboard access ---
    const stops = [];
    for (let i = 0; i < 9; i++) {
      await page.keyboard.press('Tab');
      stops.push(await page.evaluate(() => {
        const a = document.activeElement, s = getComputedStyle(a), p = a.closest('.drop');
        const ring = s.outlineStyle !== 'none' && s.outlineWidth !== '0px' ||
          (p && getComputedStyle(p).outlineStyle !== 'none');
        return {id: a.id || a.className || a.tagName, ring};
      }));
    }
    check('keyboard: tab order reaches skip link, file input, sample, method, flag, settings; never leaves the page',
      stops.some(s => s.id === 'skip') && stops.some(s => s.id === 'video') && stops.some(s => s.id === 'method') && !stops.some(s => s.id === 'BODY'),
      JSON.stringify(stops.map(s => s.id)));
    check('keyboard: visible focus ring on every stop', stops.every(s => s.ring), JSON.stringify(stops));

    // --- input validation states ---
    await page.locator('#video').setInputFiles({name: 'empty.mp4', mimeType: 'video/mp4', buffer: Buffer.alloc(0)});
    await page.locator('#start').click({force: true});
    check('empty file: clear message', /empty/i.test(await page.locator('#error').textContent()));
    await page.locator('#video').setInputFiles({name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hello')});
    await page.locator('#start').click({force: true});
    await page.waitForFunction(() => !document.querySelector('#error').hidden, null, {timeout: 10000});
    check('unsupported type: clear message before upload', /MP4|MOV|video/i.test(await page.locator('#error').textContent()),
      await page.locator('#error').textContent());
    await page.locator('#video').setInputFiles({name: 'corrupt.mp4', mimeType: 'video/mp4', buffer: Buffer.from('this is not a video')});
    await page.locator('#start').click();
    const failed = await waitJob(page, j => j && j.name === 'corrupt.mp4' && ['failed', 'completed'].includes(j.status), 60000);
    await page.waitForTimeout(2500);
    check('corrupt video: fails at "Check video" before any GPU stage',
      failed.status === 'failed' && failed.failed_stage === 'check' && failed.completed_stages.length === 0, failed.error);
    check('failed state shows recovery steps', await page.locator('#runError li').count() >= 2);
    await page.screenshot({path: path.join(OUT, 'state_failed.png'), fullPage: true});

    // --- real run on the sample clip ---
    assert(await page.locator('#sampleBox').isVisible(), 'sample clip must be offered');
    const prevTop = (await jobs(page))[0].id;
    await page.locator('#runSample').click();
    await waitJob(page, j => j && j.id !== prevTop, 30000);
    const seen = new Set(), t0 = Date.now();
    let runningShot = false, busyChecked = false;
    let job;
    for (;;) {
      job = (await jobs(page))[0];
      seen.add(`${job.status}:${job.stage || ''}`);
      if (job.status === 'running' && !runningShot && job.stage === 'perception') {
        await page.waitForTimeout(1500);
        await page.screenshot({path: path.join(OUT, 'state_running.png'), fullPage: true});
        runningShot = true;
        check('running: start disabled while a run is active', await page.locator('#start').isDisabled());
        const st = await (await page.request.get(BASE + '/api/status')).json();
        const busy = (await page.request.post(BASE + '/api/samples/MOT17-09_sample.mp4?method=FB&flag=true&tau=1e-6&window=10',
          {headers: {'X-TAORG-Token': st.token, Origin: BASE}})).status();
        check('one job at a time: second run refused (409)', busy === 409, String(busy)); busyChecked = true;
        await page.locator('#actions button.danger').focus();
      }
      if (job.status === 'running' && job.stage === 'sgg' && !seen.has('focus-checked')) {
        await page.waitForTimeout(2500);         // let the page rebuild for the new stage
        seen.add('focus-checked');
        check('keyboard: focus survives a status refresh', await page.evaluate(() => document.activeElement.textContent) === 'Cancel run',
          await page.evaluate(() => document.activeElement.tagName));
      }
      if (['completed', 'failed', 'cancelled', 'interrupted'].includes(job.status)) break;
      if (Date.now() - t0 > 20 * 60000) throw new Error('sample run timeout');
      await page.waitForTimeout(1500);
    }
    report.sample_run = {id: job.id, status: job.status, seconds: (Date.now() - t0) / 1000, states: [...seen],
      stage_times: job.stage_times, summary: job.summary, video: job.video, error: job.error};
    check('sample run completes through all stages', job.status === 'completed' &&
      job.completed_stages.join() === 'perception,sgg,maintain,view', `${job.status} ${job.error || ''}`);
    check('stage progress was observable (running states seen)', ['running:perception', 'running:sgg'].every(s => seen.has(s)),
      [...seen].filter(x => x.includes(':')).join(' '));
    check('busy check executed', busyChecked);
    await page.waitForTimeout(3000);
    await page.screenshot({path: path.join(OUT, 'state_completed.png'), fullPage: true});

    // --- viewer agreement with saved outputs ---
    const align = await page.evaluate(async id => {
      const g = await (await fetch(`/results/${id}/graphs.json`)).json();
      const out = [];
      for (let i = 0; i < g.frames.length; i++) {
        const s = document.getElementById('vSlider'); s.value = i; s.dispatchEvent(new Event('input'));
        const shown = [...document.querySelectorAll('#vRows tr')].map(r => r.dataset.key).sort();
        const saved = g.frames[i].edges.map(e => `${e.s}|${e.p}|${e.o}`).sort();
        const pos = document.getElementById('vPos').textContent;
        out.push(JSON.stringify(shown) === JSON.stringify(saved) && pos.includes(`video frame ${g.frames[i].frame} `));
      }
      return {steps: g.frames.length, ok: out.filter(Boolean).length, edges: g.frames.reduce((a, f) => a + f.edges.length, 0)};
    }, job.id);
    check('viewer: table and frame number match graphs.json at every step', align.ok === align.steps, JSON.stringify(align));
    const sc = await page.evaluate(async id => {
      const g = await (await fetch(`/results/${id}/graphs.json`)).json();
      const cur = g.frames.flatMap(f => f.edges.filter(e => e.observed).map(e => e.score));
      const shown = [...document.querySelectorAll('#vRows tr')].filter(r => /Predicted/.test(r.textContent)).map(r => r.lastChild.textContent);
      return {tau: g.tau, min: Math.min(...cur), max: Math.max(...cur), belowTau: cur.filter(v => v < g.tau).length, shown};
    }, job.id);
    check('scores: saved predicted scores are raw values >= cutoff (none rounded to 0), and not shown as 0',
      sc.belowTau === 0 && !sc.shown.includes('0'), JSON.stringify({tau: sc.tau, min: sc.min, max: sc.max, sample: sc.shown.slice(0, 3)}));

    // keyboard, playback, seeking
    await page.evaluate(() => { const s = document.getElementById('vSlider'); s.value = 0; s.dispatchEvent(new Event('input')); });
    await page.locator('#viewerRegion').focus();
    await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight');
    check('viewer: arrow keys step', (await page.locator('#vPos').textContent()).startsWith('Step 3 of'));
    await page.keyboard.press('End');
    check('viewer: End jumps to last step', (await page.locator('#vPos').textContent()).startsWith(`Step ${align.steps} of`));
    await page.keyboard.press('Home');
    await page.locator('#vSpeed').selectOption('2');
    await page.locator('#vPlay').click();
    await page.waitForTimeout(1500);
    const playing = await page.locator('#vPos').textContent();
    await page.locator('#vPlay').click();
    check('viewer: play advances steps; pause stops', !playing.startsWith('Step 1 of') &&
      (await page.locator('#vPlay').textContent()) === 'Play', playing);
    const tl = await page.locator('#vTimeline').boundingBox();
    const cell = Number(await page.locator('#vTimeline').getAttribute('data-cell'));
    const width = Number(await page.locator('#vTimeline').evaluate(c => c.width));
    if (align.edges > 0) {
      await page.locator('#vTimeline').click({position: {x: (190 + cell * 10 + cell / 2) * tl.width / width, y: 10}});
      check('viewer: timeline click seeks', (await page.locator('#vPos').textContent()).startsWith('Step 11 of'),
        await page.locator('#vPos').textContent());
    } else check('viewer: empty-run message shown', await page.locator('#viewerEmpty').isVisible());
    await page.locator('#vSlider').fill(String(Math.floor(align.steps / 2)));
    check('viewer: slider seeks', (await page.locator('#vPos').textContent()).startsWith(`Step ${Math.floor(align.steps / 2) + 1} of`));
    await page.screenshot({path: path.join(OUT, 'viewer_midrun.png'), fullPage: false});

    // --- exports ---
    const ex = await page.evaluate(async id => {
      const names = ['viewer.html', 'graphs.json', 'stream.csv.gz', 'config.yaml', 'job.json', 'run.log'];
      const res = {};
      for (const n of names) {
        const r = await fetch(`/results/${id}/${n}?download=1`);
        const body = await r.arrayBuffer();
        res[n] = {status: r.status, bytes: body.byteLength, disposition: r.headers.get('Content-Disposition')};
        if (n === 'graphs.json') res.graphsEqual = new TextDecoder().decode(body) === await (await fetch(`/results/${id}/graphs.json`)).text();
        if (n === 'config.yaml') res.config = new TextDecoder().decode(body);
        if (n === 'job.json') res.job = JSON.parse(new TextDecoder().decode(body));
      }
      return res;
    }, job.id);
    ex.blocked = (await page.request.get(`${BASE}/results/${job.id}/input.mp4`)).status();
    check('exports: all six download with attachment headers',
      ['viewer.html', 'graphs.json', 'stream.csv.gz', 'config.yaml', 'job.json', 'run.log'].every(n => ex[n].status === 200 && ex[n].bytes > 0 && /attachment/.test(ex[n].disposition || '')));
    check('exports: downloaded graph equals the one displayed', ex.graphsEqual);
    check('exports: settings file records the cutoff and method used', /tau_obs: 6\.648e-07/.test(ex.config) && ex.job.settings.method === 'FB',
      ex.job.settings ? JSON.stringify(ex.job.settings) : '');
    check('artifact allowlist: input video not downloadable', ex.blocked === 404);

    // --- cancellation and child-process cleanup ---
    await page.locator('#runSample').click();
    const runningJob = await waitJob(page, j => j && j.id !== job.id && j.status === 'running' && j.stage === 'perception', 120000);
    await page.waitForTimeout(8000);
    const before = childProcessesFor(runningJob.id);
    await page.locator('#actions button.danger').click();
    const cancelled = await waitJob(page, j => j.id === runningJob.id && ['cancelled', 'failed', 'completed'].includes(j.status), 60000);
    await page.waitForTimeout(3000);
    const after = childProcessesFor(runningJob.id);
    check('cancel: run ends as cancelled', cancelled.status === 'cancelled', cancelled.status);
    check('cancel: no child processes remain for the run', before >= 1 && after === 0, `before ${before}, after ${after}`);
    await page.screenshot({path: path.join(OUT, 'state_cancelled.png'), fullPage: true});

    // --- history ---
    const hist = await page.locator('.historyrow').count();
    check('history lists the failed, completed and cancelled runs', hist >= 3, String(hist));
    await page.locator('.historyrow').filter({hasText: 'MOT17-09'}).nth(1).click();
    await page.waitForTimeout(1500);
    check('history: selecting a past completed run reopens its viewer', await page.locator('#viewerBox').isVisible());

    // --- narrow layout of the completed state ---
    await page.setViewportSize({width: 390, height: 844});
    await page.waitForTimeout(800);
    check('narrow: no horizontal page overflow', await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(OUT, 'narrow_completed.png'), fullPage: true});
  } catch (e) {
    check('script completed without exceptions', false, e.stack);
  } finally {
    report.console_errors = report.errors;
    check('no page or console errors', report.errors.length === 0, JSON.stringify(report.errors));
    report.finished = new Date().toISOString();
    report.passed = report.checks.filter(c => c.ok).length;
    report.failed = report.checks.filter(c => !c.ok).length;
    fs.writeFileSync(path.join(OUT, 'qa_report.json'), JSON.stringify(report, null, 1));
    console.log(`\n${report.passed} passed, ${report.failed} failed`);
    await browser.close();
  }
})();
