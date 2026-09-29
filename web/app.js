const $ = id => document.getElementById(id);
let contract = [], ready = false, busy = false, revision = 0, latest = null;
let apiBase = null;

function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function controls() { $('submit').disabled = !ready || busy || !$('text').value.trim(); }
function clearResult() {
  latest = null;
  $('json-detail').hidden = true;
  $('json').textContent = '';
  $('model-info').textContent = '';
  $('result-status').textContent = '等待评分';
  renderMetrics();
}
function renderMetrics(values = {}) {
  const task = contract.find(c => c.id === $('category').value);
  $('metrics').replaceChildren();
  for (const metric of task?.outputs || []) {
    const row = document.createElement('div'); row.className = 'metric';
    const label = document.createElement('span'); label.className = 'metric-label'; label.textContent = metric.label;
    const value = document.createElement('span'); value.className = 'metric-value'; value.textContent = values[metric.key] ?? '—';
    const bar = document.createElement('meter'); bar.min = 0; bar.max = 100;
    bar.value = values[metric.key] ?? 0; bar.setAttribute('aria-label', metric.label);
    bar.hidden = values[metric.key] === undefined;
    row.append(label, value, bar); $('metrics').append(row);
  }
}
async function get(path, options) {
  const response = await fetch(path, {credentials:'omit', cache:'no-store', ...options});
  if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('服务没有返回有效JSON，请检查接口配置。');
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `请求失败（${response.status}）`);
  return value;
}
function api(path, options) {
  if (!apiBase) throw new Error('模型服务未连接。');
  return get(new URL(`api/${path}`, apiBase), options);
}
async function health() {
  if (!apiBase) {
    ready = false; $('status').textContent = '模型服务未连接';
    $('status').classList.remove('ready'); controls(); return;
  }
  try {
    const value = await api('health');
    ready = value.ready === true;
    error();
    $('status').textContent = ready ? '模型已就绪' : '尚未完成正式训练';
    $('status').classList.toggle('ready', ready);
  } catch (e) {
    ready = false; $('status').textContent = '服务连接失败'; error(e.message);
  }
  controls();
}
function changed() { revision++; clearResult(); error(); controls(); }
$('text').addEventListener('input', () => { $('count').textContent = `${$('text').value.length} 字符`; changed(); });
$('category').addEventListener('change', changed);
$('clear').addEventListener('click', () => { $('text').value = ''; $('count').textContent = '0 字符'; changed(); $('text').focus(); });
$('refresh').addEventListener('click', health);
$('scoring-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || !ready || !$('text').value.trim()) return;
  const requestRevision = revision;
  busy = true; controls(); error(); clearResult(); $('result-status').textContent = '正在评分';
  try {
    const value = await api('predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({category:$('category').value, text:$('text').value.trim()})});
    if (requestRevision !== revision) return;
    const expected = contract.find(c => c.id === $('category').value).outputs.map(m => m.key);
    if (value.category !== $('category').value || !value.scores || Object.keys(value.scores).length !== expected.length ||
        expected.some(key => !Number.isInteger(value.scores[key]) || value.scores[key] < 0 || value.scores[key] > 100)) {
      throw new Error('模型返回的字段或分数不符合当前类别，未显示结果。');
    }
    latest = value; renderMetrics(value.scores);
    $('result-status').textContent = `已完成 · ${value.elapsed_ms} ms`;
    $('model-info').textContent = `${value.model_id} · ${value.run_id}`;
    $('json').textContent = JSON.stringify(value, null, 2); $('json-detail').hidden = false;
  } catch (e) {
    if (requestRevision === revision) { clearResult(); $('result-status').textContent = '评分失败'; error(e.message); }
  } finally { busy = false; controls(); }
});
$('copy').addEventListener('click', async () => {
  if (!latest) return;
  try { await navigator.clipboard.writeText(JSON.stringify(latest, null, 2)); $('copy').textContent = '已复制'; setTimeout(() => { $('copy').textContent = '复制'; }, 1500); }
  catch { error('复制失败，请选中JSON内容复制。'); }
});
(async () => {
  try {
    const config = await get('./config.json');
    if (config.mode === 'local') {
      apiBase = new URL('./', location.href);
    } else if (config.mode === 'static') {
      $('deployment-mode').textContent = 'GitHub Pages';
      if (config.api_base_url) {
        const url = new URL(config.api_base_url);
        if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) throw new Error('线上模型服务必须是无凭据的HTTPS地址。');
        if (!url.pathname.endsWith('/')) url.pathname += '/';
        apiBase = url;
      }
    } else throw new Error('未知部署模式。');
    contract = (await get('./schema.json')).categories;
    $('category').replaceChildren(...contract.map(c => {
      const option = document.createElement('option'); option.value = c.id;
      option.textContent = c.label + (c.auxiliary ? '（辅助任务）' : ''); return option;
    }));
    $('category').disabled = false; renderMetrics(); await health();
  } catch (e) { error(e.message); $('status').textContent = '初始化失败'; }
})();
