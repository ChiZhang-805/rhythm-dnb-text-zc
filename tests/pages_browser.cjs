/* Check the real static deployment; no model response or network API is mocked. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const base = process.env.DNB_TEST_URL || 'http://127.0.0.1:8766/github-pages/';
  const output = path.resolve(__dirname, '../results/macbert/pages-browser');
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({headless:true, channel:process.env.DNB_BROWSER || 'msedge'});
  const errors = [], apiRequests = [], checks = [];
  try {
    const page = await browser.newPage({viewport:{width:1365,height:900}});
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', request => { if (new URL(request.url()).pathname.includes('/api/')) apiRequests.push(request.url()); });
    const response = await page.goto(base);
    assert.equal(response.status(),200);
    await page.waitForFunction(() => !document.getElementById('category').disabled);
    assert.equal(await page.locator('#status').textContent(),'模型服务未连接');
    assert.equal(await page.locator('#deployment-mode').textContent(),'GitHub Pages');
    const schema = await (await page.request.get(new URL('schema.json',base).href)).json();
    const config = await (await page.request.get(new URL('config.json',base).href)).json();
    assert.equal(config.mode,'static');
    assert.equal(config.api_base_url,null);
    checks.push('static_mode_without_unconfigured_api_requests');
    for (const category of schema.categories) {
      await page.locator('#category').selectOption(category.id);
      await page.locator('#text').fill('这是一条仅用于静态网页检查的中文文本。');
      assert.equal(await page.locator('#submit').isDisabled(),true);
      assert.deepEqual(await page.locator('.metric-label').allTextContents(),category.outputs.map(x=>x.label));
      assert.equal(await page.locator('meter:visible').count(),0);
      checks.push(category.id+'_all_output_names_no_fake_scores');
    }
    await page.locator('#category').selectOption('emotion');
    await page.locator('#clear').click();
    for (const [name,width,height] of [['desktop',1365,900],['mobile',390,844]]) {
      await page.setViewportSize({width,height});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),true);
      await page.screenshot({path:path.join(output,name+'.png'),fullPage:true});
      checks.push(name+'_no_horizontal_overflow');
    }
    await page.locator('#refresh').click();
    assert.equal(await page.locator('#status').textContent(),'模型服务未连接');
    assert.deepEqual(apiRequests,[]);
    assert.deepEqual(errors,[]);
    const result = {passed:true,url:base,checks,api_requests:apiRequests,page_errors:errors,model_ready:false};
    await fs.writeFile(path.join(output,'report.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1);});
