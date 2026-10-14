// Quick visual capture of the running website at several viewport sizes (QA helper; screenshots stay on D:).
// node scripts/web/shots.cjs <base-url> <out-dir>
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const [base = 'http://127.0.0.1:8765', out = 'D:/TAORG_runs/web-qa'] = process.argv.slice(2);
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const errors = [];
  try {
    for (const [name, w, h] of [['desktop', 1440, 900], ['laptop', 1280, 720], ['narrow', 390, 844]]) {
      const page = await browser.newPage({viewport: {width: w, height: h}});
      page.on('pageerror', e => errors.push(`${name}: ${e.message}`));
      page.on('console', m => { if (m.type() === 'error') errors.push(`${name} console: ${m.text()}`); });
      await page.goto(base);
      await page.waitForSelector('#history .historyrow', {timeout: 10000}).catch(() => {});
      await page.waitForTimeout(1500);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      await page.screenshot({path: path.join(out, `${name}.png`), fullPage: true});
      console.log(name, 'horizontal overflow px:', overflow);
      await page.close();
    }
  } finally { await browser.close(); }
  console.log('errors:', JSON.stringify(errors));
})();
