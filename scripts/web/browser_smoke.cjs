// Optional browser check against a running local server. Requires Playwright.
// node scripts/web/browser_smoke.cjs <existing-test-video> <screenshot-directory>
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const [video, output] = process.argv.slice(2);
  assert(video && output, 'Supply an existing test video and screenshot directory');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto('http://127.0.0.1:8765');
    await page.waitForFunction(() => document.querySelector('#readyText').textContent === 'Environment ready');
    await page.locator('#video').setInputFiles(video);
    assert(await page.locator('#preview').isVisible());
    await page.locator('#method').selectOption('VH');
    assert(await page.locator('#flag').isDisabled());
    await page.locator('#method').selectOption('FB');
    await page.locator('#flag').check();
    assert(await page.locator('#start').isEnabled());
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    await page.setViewportSize({width: 1440, height: 1050});
    const history = page.locator('.historyrow').filter({hasText: 'website-smoke.mp4'}).first();
    await history.click();
    await page.waitForFunction(() => !document.querySelector('#viewer').hidden);
    const viewer = page.frameLocator('#viewer');
    await viewer.locator('canvas').first().waitFor();
    assert(await viewer.locator('canvas').first().isVisible());
    await page.screenshot({path: path.join(output, 'completed.png'), fullPage: true});
    // Exercise actual UI submission and failure reporting without another GPU run.
    await page.locator('#video').setInputFiles({name: 'invalid-web-test.mp4',
      mimeType: 'video/mp4', buffer: Buffer.from('invalid test video')});
    await page.locator('#start').click();
    await page.waitForFunction(() => document.querySelector('#runName').textContent === 'invalid-web-test.mp4');
    await page.waitForFunction(() => document.querySelector('#badge').textContent === 'failed', null, {timeout: 120000});
    assert(await page.locator('#runError').isVisible());
    assert(await page.locator('#start').isEnabled());
    await page.locator('#logs summary').click();
    await page.waitForFunction(() => document.querySelector('#logText').textContent.includes('perception'));
    await page.screenshot({path: path.join(output, 'failed.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log('PASS: upload selection, method controls, desktop/mobile layout, viewer, failed upload processing, logs, no JS errors');
  } finally {
    await browser.close();
  }
})().catch(e => {console.error(e); process.exitCode = 1;});
