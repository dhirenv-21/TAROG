'use strict';
const $ = id => document.getElementById(id);
let token = '', ready = false, selected = null, jobs = [], uploading = false, previewURL = null;
let maxUpload = 2048 * 1024**2;
const activeStates = ['uploading', 'queued', 'running', 'cancelling'];
const stages = [['perception', 'Track people'], ['sgg', 'Predict relations'], ['maintain', 'Maintain graph'], ['view', 'Build viewer']];
const names = {FB: 'Fixed buffer', NM: 'No memory', B2: 'GatedEMA + hysteresis', VH: 'V-HPED'};
function error(message) {$('error').textContent = message; $('error').hidden = !message;}
async function api(path, options = {}) {
  const res = await fetch(path, {...options, headers: {'X-TAORG-Token': token, ...options.headers}});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
function buttonState() {$('start').disabled = !ready || uploading || jobs.some(j => activeStates.includes(j.status));}
function chooseFile() {
  const file = $('video').files[0];
  if (previewURL) URL.revokeObjectURL(previewURL);
  $('preview').hidden = !file;
  $('filename').textContent = file ? file.name : 'Choose a video';
  $('filemeta').textContent = file ? `${(file.size / 1024**2).toFixed(1)} MB · ready to upload` : 'or drop it here';
  if (file) {previewURL = URL.createObjectURL(file); $('preview').src = previewURL;}
}
$('video').addEventListener('change', chooseFile);
$('preview').addEventListener('error', () => {
  $('filemeta').textContent = 'Browser preview unavailable for this codec. Analysis can still run.';
});
for (const name of ['dragenter', 'dragover']) $('drop').addEventListener(name, e => {e.preventDefault(); $('drop').classList.add('over');});
for (const name of ['dragleave', 'drop']) $('drop').addEventListener(name, e => {e.preventDefault(); $('drop').classList.remove('over');});
$('drop').addEventListener('drop', e => {if (e.dataTransfer.files.length) {$('video').files = e.dataTransfer.files; chooseFile();}});
$('method').onchange = () => {
  const m = $('method').value, supported = ['FB', 'B2'].includes(m);
  $('flag').disabled = !supported;
  if (!supported) $('flag').checked = false;
  $('window').disabled = m !== 'FB';
  $('methodHint').textContent = {FB: 'Keeps recent relationships for a short window.', NM: 'Shows predictions without retaining relationships.', B2: 'Experimental: raw EGTR scores are small; this method may produce an empty graph.', VH: 'Experimental visibility filter with certified lazy maintenance. Output quality is not validated.'}[m];
};
function action(label, href, download = false) {
  const a = document.createElement('a'); a.textContent = label; a.href = href;
  if (download) a.download = ''; else {a.target = '_blank'; a.rel = 'noopener';}
  $('actions').append(a);
}
function renderDetail() {
  const job = jobs.find(j => j.id === selected);
  $('empty').hidden = !!job; $('runDetail').hidden = !job;
  if (!job) return;
  $('runName').textContent = job.name;
  $('badge').textContent = job.status; $('badge').className = `badge ${job.status}`;
  $('runInfo').textContent = `${names[job.options.method]}${job.options.flag ? ' + takeover flag' : ''} · ${new Date(job.created * 1000).toLocaleString()}${job.summary ? ` · ${job.summary.steps} steps · ${job.summary.edges} relationship records` : ''}`;
  $('stages').replaceChildren();
  for (const [key, text] of stages) {
    const li = document.createElement('li'); li.textContent = text;
    li.className = job.completed_stages.includes(key) ? 'done' : job.stage === key ? 'current' : '';
    $('stages').append(li);
  }
  $('runError').hidden = !job.error; $('runError').textContent = job.error || '';
  $('actions').replaceChildren();
  if (['queued', 'running'].includes(job.status)) {
    const b = document.createElement('button'); b.textContent = 'Cancel run';
    b.onclick = async () => {b.disabled = true; try {await api(`/api/jobs/${job.id}/cancel`, {method: 'POST'}); await refresh();} catch(e) {error(e.message);}};
    $('actions').append(b);
  }
  $('viewer').hidden = job.status !== 'completed';
  if (job.status === 'completed') {
    const base = `/results/${job.id}/`;
    if ($('viewer').getAttribute('src') !== base + 'viewer.html') $('viewer').src = base + 'viewer.html';
    action('Open full viewer ↗', base + 'viewer.html');
    action('Download viewer', base + 'viewer.html?download=1', true);
    action('Graph JSON', base + 'graphs.json?download=1', true);
    action('Observation stream', base + 'stream.csv.gz?download=1', true);
    action('Run settings', base + 'config.yaml?download=1', true);
  }
  action('Download log', `/results/${job.id}/run.log?download=1`, true);
}
async function refresh() {
  jobs = await api('/api/jobs');
  if (!selected && jobs.length) selected = jobs[0].id;
  $('runCount').textContent = `${jobs.length} run${jobs.length === 1 ? '' : 's'}`;
  $('history').replaceChildren();
  if (!jobs.length) {const p = document.createElement('p'); p.className = 'muted'; p.textContent = 'Your runs will appear here.'; $('history').append(p);}
  for (const job of jobs) {
    const row = document.createElement('button'); row.className = 'historyrow' + (selected === job.id ? ' selected' : '');
    const left = document.createElement('span'), title = document.createElement('strong'), meta = document.createElement('small'), badge = document.createElement('span');
    title.textContent = job.name; meta.textContent = `${names[job.options.method]} · ${new Date(job.created * 1000).toLocaleString()}`;
    badge.textContent = job.status; badge.className = `badge ${job.status}`; left.append(title, meta); row.append(left, badge);
    row.onclick = () => {selected = job.id; refresh().catch(e => error(e.message));}; $('history').append(row);
  }
  renderDetail(); buttonState();
  if (selected && $('logs').open) {const id = selected, log = await api(`/api/jobs/${id}/log`); if (selected === id) $('logText').textContent = log.text;}
}
$('logs').addEventListener('toggle', () => {if ($('logs').open) refresh().catch(e => error(e.message));});
$('runForm').onsubmit = async e => {
  e.preventDefault(); error('');
  const file = $('video').files[0]; if (!file) return;
  if (file.size > maxUpload) {error(`Choose a video under ${Math.round(maxUpload / 1024**2)} MB.`); return;}
  const query = new URLSearchParams({name: file.name, method: $('method').value, flag: String($('flag').checked), tau: $('tau').value, window: $('window').value});
  uploading = true; buttonState();
  try {
    const response = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest(); xhr.open('POST', '/api/jobs?' + query);
      xhr.setRequestHeader('X-TAORG-Token', token); xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.upload.onprogress = p => {$('uploadStatus').textContent = p.lengthComputable ? `Uploading to your local workspace · ${Math.round(p.loaded / p.total * 100)}%` : 'Uploading…';};
      xhr.onerror = () => reject(new Error('Cannot reach the local server. Check that it is running.'));
      xhr.onload = () => {try {const d = JSON.parse(xhr.responseText); if (xhr.status >= 400) reject(new Error(d.error || 'Upload failed')); else resolve(d);} catch {reject(new Error('The server rejected this upload. Refresh the page and retry.'));}};
      xhr.send(file);
    });
    selected = response.id; $('uploadStatus').textContent = 'Uploaded. Processing may take several minutes.';
    await refresh();
  } catch(e) {error(e.message); $('uploadStatus').textContent = '';}
  finally {uploading = false; buttonState();}
};
async function init() {
  try {
    const state = await api('/api/status'); token = state.token; ready = state.ready;
    maxUpload = state.max_upload;
    $('uploadLimit').textContent = `MP4, MOV, AVI, MKV, WebM · up to ${Math.round(maxUpload / 1024**2)} MB`;
    $('tau').value = state.default_tau;
    $('readyDot').textContent = ready ? '●' : '○'; $('readyText').textContent = ready ? 'Environment ready' : 'Setup needs attention';
    for (const check of state.checks) {const li = document.createElement('li'); li.textContent = `${check.ok ? '✓' : 'Missing:'} ${check.name}`; li.title = check.path; $('checks').append(li);}
    if (!ready) document.querySelector('.environment').open = true;
    await refresh();
  } catch(e) {error('Cannot connect to TAORG. Start the local server, then refresh this page.');}
  const poll = async () => {try {await refresh();} catch(e) {error('Connection lost. Your saved runs remain on disk. Refresh after restarting the server.');} setTimeout(poll, 2000);};
  setTimeout(poll, 2000);
}
init();
