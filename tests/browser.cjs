/* Browser-only fixtures exercise the UI; no fixture data reaches the corpus or server. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const base = process.env.DNB_TEST_URL || 'http://127.0.0.1:8765';
  const output = path.resolve(__dirname, '../results/macbert/browser');
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: process.env.DNB_BROWSER || 'msedge' });
  const failures = [], checks = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1365, height: 900 } });
    page.on('pageerror', error => failures.push(error.message));
    await page.goto(base);
    await page.waitForFunction(() => !document.getElementById('category').disabled);
    await page.waitForFunction(() => document.getElementById('status').textContent !== '正在连接');
    const liveHealth = await (await page.request.get(base + '/api/health')).json();
    if (!liveHealth.ready) {
      assert.equal(await page.locator('#submit').isDisabled(), true);
      const response = await page.request.post(base + '/api/predict', { data: { category: 'sleep', text: '这是一条隔离接口测试文本。' } });
      assert.equal(response.status(), 503);
      assert.equal('scores' in await response.json(), false);
      checks.push('untrained_server_returns_503_without_scores');
    }
    for (const [name, width, height] of [['desktop',1365,900],['mobile',390,844]]) {
      await page.setViewportSize({ width, height });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      assert.equal(await page.locator('#text').isVisible(), true);
      await page.screenshot({ path: path.join(output, name + '.png'), fullPage: true });
      checks.push(name + '_no_horizontal_overflow');
    }
    const schema = await (await page.request.get(base + '/api/schema')).json();
    await page.route('**/api/health', route => route.fulfill({ json: { ready: true, run_id: 'browser-fixture-only' } }));
    await page.route('**/api/predict', async route => {
      const input = route.request().postDataJSON();
      const category = schema.categories.find(c => c.id === input.category);
      const scores = Object.fromEntries(category.outputs.map((m,i) => [m.key, 23+i*7]));
      await route.fulfill({ json: { category: input.category, scores, model_id: 'browser-fixture-only', run_id: 'not-a-trained-model', elapsed_ms: 1 } });
    });
    await page.locator('#refresh').click();
    await page.waitForFunction(() => document.getElementById('status').textContent === '模型已就绪');
    for (const category of schema.categories) {
      await page.locator('#category').selectOption(category.id);
      await page.locator('#text').fill('这是一条仅用于网页控件测试的中文内容。');
      await page.locator('#submit').click();
      await page.waitForFunction(() => document.getElementById('result-status').textContent.startsWith('已完成'));
      assert.equal(await page.locator('.metric').count(), category.outputs.length);
      assert.deepEqual(await page.locator('.metric-label').allTextContents(), category.outputs.map(m => m.label));
      assert.equal(await page.locator('meter:visible').count(), category.outputs.length);
      assert.equal(await page.locator('#json-detail').isVisible(), true);
      checks.push(category.id + '_fixed_fields_and_numeric_results_fixture');
    }
    await page.locator('#text').fill('输入内容改变之后，旧结果必须立即清除。');
    assert.equal(await page.locator('meter:visible').count(), 0);
    assert.equal(await page.locator('#json-detail').isVisible(), false);
    checks.push('editing_invalidates_stale_results');
    await page.unroute('**/api/predict');
    await page.route('**/api/predict', route => route.fulfill({ status: 500, json: { error: '隔离测试错误' } }));
    await page.locator('#submit').click();
    await page.locator('#error').waitFor({ state: 'visible' });
    assert.equal(await page.locator('meter:visible').count(), 0);
    checks.push('failed_request_never_leaves_old_scores');
    await page.locator('#clear').click();
    assert.equal(await page.locator('#text').inputValue(), '');
    assert.equal(await page.locator('#submit').isDisabled(), true);
    checks.push('empty_input_and_clear');
    assert.deepEqual(failures, []);
    await fs.writeFile(path.join(output, 'report.json'), JSON.stringify({ passed: true, live_model_ready: liveHealth.ready, checks, page_errors: failures, fixture_predictions_are_not_model_accuracy: true }, null, 2));
    console.log(JSON.stringify({ passed: true, checks: checks.length, live_model_ready: liveHealth.ready, output }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
