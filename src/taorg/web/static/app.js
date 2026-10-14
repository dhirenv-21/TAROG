'use strict';
// TAORG local workspace. Talks only to the local server; renders saved outputs as they are.
const $ = id => document.getElementById(id);
const ACTIVE = ['uploading', 'queued', 'running', 'cancelling'];
const STAGES = [['check', 'Check video'], ['perception', 'Track people'], ['sgg', 'Predict relationships'],
  ['maintain', 'Maintain graph'], ['view', 'Build viewer']];
const STAGE_LABEL = Object.fromEntries(STAGES);
const METHOD = {FB: 'Fixed buffer', NM: 'No memory (raw predictions)', B2: 'GatedEMA + hysteresis (experimental)',
  VH: 'V-HPED filter (experimental)'};
const METHOD_HINT = {
  FB: 'Recommended. Shows each predicted relationship and keeps it for a short window after the model stops predicting it.',
  NM: 'Shows only what the model predicts at each step; nothing is retained.',
  B2: 'Experimental. It expects scores between 0 and 1, but raw EGTR scores are tiny, so this method usually produces an empty graph.',
  VH: 'Experimental visibility-aware filter (with certified lazy maintenance). Same score-scale problem as GatedEMA; output quality is not validated.'};
const REL = {behind: 'is behind', in_front_of: 'is in front of', next_to: 'is next to', watch: 'is watching'};
let token = '', ready = false, selected = null, jobs = [], uploading = false, previewURL = null;
let maxUpload = 2048 * 1024 ** 2, maxSeconds = 300, serverStarted = 0, samples = [];

// ---------- helpers ----------
function showError(msg) { $('error').textContent = msg; $('error').hidden = !msg; }
async function api(path, options = {}) {
  const res = await fetch(path, {...options, headers: {'X-TAORG-Token': token, ...options.headers}});
  let data = {};
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error(data.error || `The server answered ${res.status}. Refresh the page and try again.`);
  return data;
}
function el(tag, props = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v; else e.setAttribute(k, v);
  }
  e.append(...children.filter(c => c !== null && c !== undefined));
  return e;
}
const fmtTime = s => s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`;
const fmtDate = t => new Date(t * 1000).toLocaleString();
// Older exports rounded scores to 4 decimals, turning raw EGTR scores (~1e-6) into 0; never show those as 0.
const score = v => (viewer.rounded && v < 0.0001 ? '< 0.0001 (rounded in this older run)' :
  v === 0 ? '0' : v < 0.001 ? v.toExponential(2) : v.toFixed(3));
const person = id => `Person ${id}`;
function settingsQuery() {
  return new URLSearchParams({method: $('method').value, flag: String($('flag').checked),
    tau: $('tau').value, window: $('window').value});
}
function validateSettings() {
  const tau = Number($('tau').value), win = Number($('window').value);
  if (!(tau > 0 && tau <= 1)) return 'The score cutoff must be greater than 0 and at most 1.';
  if (!(Number.isInteger(win) && win >= 1 && win <= 300)) return 'The buffer length must be a whole number from 1 to 300.';
  return '';
}
function buttonState() {
  const busy = uploading || jobs.some(j => ACTIVE.includes(j.status));
  $('start').disabled = !ready || busy || !$('video').files[0];
  $('runSample').disabled = !ready || busy;
  $('selfcheck').disabled = busy;
  $('start').textContent = busy ? 'A run is in progress…' : 'Start analysis';
}

// ---------- form ----------
const VIDEO_SUFFIXES = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'];
function fileProblem(file) {
  if (file.size === 0) return 'This file is empty. Choose another video.';
  if (!VIDEO_SUFFIXES.some(x => file.name.toLowerCase().endsWith(x))) return 'Choose an MP4, MOV, AVI, MKV, WebM or M4V video.';
  if (file.size > maxUpload) return `This file is larger than the ${Math.round(maxUpload / 1024 ** 2)} MB upload limit.`;
  return '';
}
function chooseFile() {
  const file = $('video').files[0];
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $('previewNote').hidden = true;
  $('preview').hidden = !file;
  $('filename').textContent = file ? file.name : 'Choose a video';
  const problem = file && fileProblem(file);
  $('filemeta').textContent = file ? `${(file.size / 1024 ** 2).toFixed(1)} MB · ${problem ? 'cannot be used' : 'ready to start'}` : 'or drop a file here';
  showError('');
  if (problem) showError(problem);
  if (file && !problem) { previewURL = URL.createObjectURL(file); $('preview').src = previewURL; }
  else $('preview').hidden = true;
  buttonState();
}
$('video').addEventListener('change', chooseFile);
$('preview').addEventListener('error', () => {
  if (!$('video').files[0]) return;
  $('preview').hidden = true;
  $('previewNote').hidden = false;
  $('previewNote').textContent = 'Your browser cannot preview this video format. That is fine: the analysis decodes it separately.';
});
$('preview').addEventListener('loadedmetadata', () => {
  const d = $('preview').duration;
  if (Number.isFinite(d)) {
    $('filemeta').textContent = `${$('filemeta').textContent.split(' · ')[0]} · ${d.toFixed(1)} s`;
    if (d > maxSeconds) showError(`This video is ${Math.round(d)} s long; the limit is ${Math.round(maxSeconds)} s. Trim it to a short clip.`);
  }
});
for (const n of ['dragenter', 'dragover']) $('drop').addEventListener(n, e => { e.preventDefault(); $('drop').classList.add('over'); });
for (const n of ['dragleave', 'drop']) $('drop').addEventListener(n, e => { e.preventDefault(); $('drop').classList.remove('over'); });
$('drop').addEventListener('drop', e => { if (e.dataTransfer.files.length) { $('video').files = e.dataTransfer.files; chooseFile(); } });
function methodChanged() {
  const m = $('method').value, flagOK = ['FB', 'B2'].includes(m);
  $('flag').disabled = !flagOK;
  if (!flagOK) $('flag').checked = false;
  $('window').disabled = m !== 'FB';
  $('methodHint').textContent = METHOD_HINT[m];
}
$('method').addEventListener('change', methodChanged);

$('runForm').addEventListener('submit', async e => {
  e.preventDefault();
  showError('');
  const file = $('video').files[0];
  if (!file) { showError('Choose a video first.'); return; }
  const problem = fileProblem(file); if (problem) { showError(problem); return; }
  const bad = validateSettings(); if (bad) { showError(bad); return; }
  const query = settingsQuery(); query.set('name', file.name);
  uploading = true; buttonState();
  try {
    const response = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/jobs?' + query);
      xhr.setRequestHeader('X-TAORG-Token', token);
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.upload.onprogress = p => {
        $('uploadStatus').textContent = p.lengthComputable ?
          `Copying the video into the local workspace: ${Math.round(p.loaded / p.total * 100)}% sent` : 'Copying the video…';
      };
      xhr.onerror = () => reject(new Error('Cannot reach the local server. Check that the TAORG window is still open.'));
      xhr.onload = () => {
        let d = {}; try { d = JSON.parse(xhr.responseText); } catch { /* ignore */ }
        if (xhr.status >= 400) reject(new Error(d.error || 'The server rejected this upload. Refresh the page and retry.'));
        else resolve(d);
      };
      xhr.send(file);
    });
    selected = response.id;
    $('uploadStatus').textContent = 'Video received. Processing has started; it usually takes a few minutes.';
    await refresh();
    $('workspace').focus();
  } catch (err) { showError(err.message); $('uploadStatus').textContent = ''; }
  finally { uploading = false; buttonState(); }
});

$('runSample').addEventListener('click', async () => {
  showError('');
  const bad = validateSettings(); if (bad) { showError(bad); return; }
  if (!samples.length) return;
  try {
    const r = await api(`/api/samples/${encodeURIComponent(samples[0].name)}?` + settingsQuery(), {method: 'POST'});
    selected = r.id; $('uploadStatus').textContent = 'Sample clip started. Processing usually takes a few minutes.';
    await refresh(); $('workspace').focus();
  } catch (err) { showError(err.message); }
});

$('selfcheck').addEventListener('click', async () => {
  $('selfcheck').disabled = true; $('selfcheckResults').replaceChildren();
  $('selfcheckText').textContent = 'Checking both environments… this can take up to a minute.';
  try {
    const r = await api('/api/selfcheck', {method: 'POST'});
    for (const c of r.checks) {
      $('selfcheckResults').append(el('li', {class: c.ok ? 'ok' : 'bad'}, `${c.name}: ${c.detail || (c.ok ? 'OK' : 'failed')} (${c.seconds} s)`));
    }
    const ok = r.checks.every(c => c.ok);
    $('selfcheckText').textContent = ok ? 'Both environments load and see the GPU.' :
      'A check failed. Follow docs/SETUP.md, then run the check again.';
  } catch (err) { $('selfcheckText').textContent = err.message; }
  finally { buttonState(); }
});

// ---------- history and run detail ----------
function statusText(job) {
  const st = job.stage_times || {};
  switch (job.status) {
    case 'uploading': return 'Copying the video into the local workspace…';
    case 'queued': return 'Queued: processing starts in a moment.';
    case 'running': {
      const label = STAGE_LABEL[job.stage] || 'Processing';
      const since = st[job.stage] ? ` for ${fmtTime(Date.now() / 1000 - st[job.stage][0])}` : '';
      return `Running: ${label}${since}. Each stage loads its models first; the whole run usually takes a few minutes.`;
    }
    case 'cancelling': return 'Stopping the current stage…';
    case 'cancelled': return 'Cancelled. Outputs of completed stages are kept for inspection.';
    case 'interrupted': return 'Interrupted: the server stopped during this run. Start a new run to analyse the video again.';
    case 'failed': return `Failed${job.failed_stage ? ` at “${STAGE_LABEL[job.failed_stage]}”` : ''}.`;
    case 'completed': {
      const s = job.summary || {};
      const total = Object.values(st).filter(t => t && t[1]).reduce((a, t) => a + (t[1] - t[0]), 0);
      const parts = [];
      if (s.steps !== undefined) parts.push(`${s.steps} relationship steps`);
      if (s.tracks !== undefined) parts.push(`${s.tracks} tracked people`);
      if (s.edges !== undefined) parts.push(`${s.edges} relationship records` +
        (s.current !== undefined ? ` (${s.current} predicted, ${s.retained} retained)` : ''));
      return `Completed${total ? ` in ${fmtTime(total)}` : ''}${parts.length ? ': ' + parts.join(', ') : ''}.`;
    }
    default: return job.status;
  }
}
function recovery(job) {
  if (job.status !== 'failed' && job.status !== 'interrupted') return null;
  const tips = job.failed_stage === 'check' ?
    ['Use a short clip (10–60 s works best) saved as MP4 (H.264).', 'Choose the file again and press Start analysis.'] :
    ['Open the processing log below; the last lines name the problem.', 'If a model or environment is missing, the setup panel on the left shows it; see docs/SETUP.md.',
      'Start a new run after fixing it. Partial outputs of this run stay on disk.'];
  return el('ul', {}, ...tips.map(t => el('li', {text: t})));
}
function stageClass(job, key) {
  if (key === 'check') {
    if (job.failed_stage === 'check') return 'failed';
    if (job.video || job.completed_stages.length) return 'done';
    return job.stage === 'check' ? 'current' : '';
  }
  if (job.completed_stages.includes(key)) return 'done';
  if (job.stage === key) return 'current';
  if (job.failed_stage === key) return 'failed';
  return '';
}
function action(label, href, download = false) {
  const a = el('a', {href, text: label});
  if (download) a.setAttribute('download', ''); else { a.target = '_blank'; a.rel = 'noopener'; }
  $('actions').append(a);
}
function renderSettings(job) {
  const s = job.settings || {}, v = job.video, o = job.options;
  const rows = [
    ['Method', METHOD[o.method]],
    ['Takeover flag', o.flag ? 'On (frozen appearance model)' : 'Off'],
    ['Score cutoff', String(o.tau)],
    ['Fixed-buffer length', o.method === 'FB' ? `${o.window} steps` : 'not used by this method'],
    ['Relationship step', `every ${s.relationship_stride || 3}rd video frame`],
    ['Video', v ? `${v.width}×${v.height}, ${v.fps} fps${v.duration_s ? `, ${v.duration_s} s` : ''}` : 'not checked'],
    ['Input SHA-256', job.input_sha256 ? job.input_sha256.slice(0, 16) + '…' : '—'],
    ['Settings file SHA-256', job.config_sha256 ? job.config_sha256.slice(0, 16) + '…' : '—'],
    ['Code version', job.git_commit ? job.git_commit.slice(0, 10) + (job.git_dirty_src ? ' + local changes' : '') : '—'],
  ];
  if (job.sample) rows.push(['Sample source', `${job.sample.source} · ${job.sample.licence}`]);
  $('settingsList').replaceChildren(...rows.flatMap(([k, val]) => [el('dt', {text: k}), el('dd', {text: val})]));
}
function renderDetail() {
  const job = jobs.find(j => j.id === selected);
  $('empty').hidden = !!job; $('runDetail').hidden = !job;
  if (!job) return;
  const saved = job.status === 'completed' && job.created < serverStarted;
  $('runKind').textContent = (job.origin === 'sample' ? 'Sample clip · ' : '') + (saved ? 'Saved result from an earlier session' : 'Run');
  $('runName').textContent = job.name;
  $('runWhen').textContent = `Started ${fmtDate(job.created)} · ${METHOD[job.options.method]}${job.options.flag ? ' + takeover flag' : ''}`;
  $('badge').textContent = job.status; $('badge').className = `badge ${job.status}`;
  $('statusLine').textContent = statusText(job);
  $('stages').replaceChildren(...STAGES.map(([key, label]) => {
    const t = (job.stage_times || {})[key];
    const li = el('li', {class: stageClass(job, key)}, el('b', {text: label}),
      t && t[1] ? `${fmtTime(t[1] - t[0])}` : '');
    return li;
  }));
  $('runError').hidden = !job.error;
  $('runError').replaceChildren(...(job.error ? [el('p', {text: job.error, style: 'margin:0'}), recovery(job)] : []));
  $('actions').replaceChildren();
  if (['queued', 'running'].includes(job.status)) {
    const b = el('button', {type: 'button', class: 'danger', text: 'Cancel run'});
    b.addEventListener('click', async () => {
      b.disabled = true;
      try { await api(`/api/jobs/${job.id}/cancel`, {method: 'POST'}); await refresh(); } catch (err) { showError(err.message); }
    });
    $('actions').append(b);
  }
  if (job.status === 'completed') {
    const base = `/results/${job.id}/`;
    action('Open offline viewer ↗', base + 'viewer.html');
    action('Download offline viewer', base + 'viewer.html?download=1', true);
    action('Graph (JSON)', base + 'graphs.json?download=1', true);
    action('Predictions (CSV)', base + 'stream.csv.gz?download=1', true);
    action('Settings (YAML)', base + 'config.yaml?download=1', true);
    action('Run record (JSON)', base + 'job.json?download=1', true);
  }
  if (!['uploading', 'queued'].includes(job.status)) action('Processing log', `/results/${job.id}/run.log?download=1`, true);
  renderSettings(job);
  if (job.status === 'completed') loadViewer(job); else { $('viewerBox').hidden = true; viewer.jobId = null; stopPlay(); }
}
let lastSig = '';
async function refresh(force = false) {
  jobs = await api('/api/jobs');
  if (!selected && jobs.length) selected = jobs[0].id;
  // Rebuild the DOM only when something changed, so keyboard focus is not lost during polling.
  const sig = selected + JSON.stringify(jobs.map(j => [j.id, j.status, j.stage, j.completed_stages, j.summary,
    j.error, j.video, j.stage_times]));
  if (sig === lastSig && !force) {
    const job = jobs.find(j => j.id === selected);
    if (job) $('statusLine').textContent = statusText(job);
    buttonState();
    if (selected && $('logs').open && job && ACTIVE.includes(job.status)) {
      const log = await api(`/api/jobs/${job.id}/log`); if (selected === job.id) $('logText').textContent = log.text;
    }
    return;
  }
  lastSig = sig;
  const active = document.activeElement;
  const focusedRow = active && active.dataset ? active.dataset.id : null;
  const focusedAction = active && $('actions').contains(active) ? active.textContent : null;
  $('runCount').textContent = `${jobs.length} saved run${jobs.length === 1 ? '' : 's'}`;
  if (!jobs.length) $('history').replaceChildren(el('p', {class: 'muted', text: 'Your runs will appear here.'}));
  else $('history').replaceChildren(...jobs.map(job => {
    const s = job.summary;
    const meta = `${fmtDate(job.created)} · ${METHOD[job.options.method].split(' (')[0]}${job.options.flag ? ' + flag' : ''}` +
      (s ? ` · ${s.steps} steps · ${s.edges} records` : '');
    const row = el('button', {type: 'button', class: 'historyrow' + (selected === job.id ? ' selected' : ''),
      'aria-pressed': String(selected === job.id), 'data-id': job.id},
    el('span', {}, el('strong', {text: (job.origin === 'sample' ? 'Sample · ' : '') + job.name}), el('small', {text: meta})),
    el('span', {class: `badge ${job.status}`, text: job.status}));
    row.addEventListener('click', () => { selected = job.id; refresh(true).catch(err => showError(err.message)); });
    return row;
  }));
  if (focusedRow) { const r = $('history').querySelector(`[data-id="${focusedRow}"]`); if (r) r.focus(); }
  renderDetail(); buttonState();
  if (focusedAction) { const a = [...$('actions').children].find(x => x.textContent === focusedAction); if (a) a.focus(); }
  if (selected && $('logs').open) {
    const id = selected, log = await api(`/api/jobs/${id}/log`);
    if (selected === id) $('logText').textContent = log.text;
  }
}
$('logs').addEventListener('toggle', () => { if ($('logs').open) refresh(true).catch(err => showError(err.message)); });

// ---------- viewer ----------
const viewer = {jobId: null, data: null, i: 0, playing: false, timer: null, images: new Map(), keys: [], rowOf: new Map()};
function edgeKey(e) { return `${e.s}|${e.p}|${e.o}`; }
async function loadViewer(job) {
  $('viewerBox').hidden = false;
  if (viewer.jobId === job.id) return;
  stopPlay();
  viewer.jobId = job.id; viewer.data = null; viewer.i = 0; viewer.images = new Map();
  $('vPos').textContent = 'Loading saved graph…';
  try {
    const res = await fetch(`/results/${job.id}/graphs.json`);
    if (!res.ok) throw new Error('graphs.json is missing for this run.');
    const data = await res.json();
    if (viewer.jobId !== job.id) return;
    viewer.data = data;
    viewer.rounded = data.frames.some(f => f.edges.some(e => e.observed && e.score < data.tau));
    const keys = new Map();
    data.frames.forEach(f => f.edges.forEach(e => keys.set(edgeKey(e), e)));
    viewer.keys = [...keys.keys()].sort((a, b) => {
      const [as, ap, ao] = a.split('|'), [bs, bp, bo] = b.split('|');
      return (+as - +bs) || (+ao - +bo) || ap.localeCompare(bp);
    });
    viewer.rowOf = new Map(viewer.keys.map((k, r) => [k, r]));
    $('vSlider').max = Math.max(0, data.frames.length - 1); $('vSlider').value = 0;
    const total = data.frames.reduce((a, f) => a + f.edges.length, 0);
    $('viewerEmpty').hidden = total > 0;
    $('viewerEmpty').textContent = total ? '' : 'No relationships were predicted in this run at the saved score cutoff. ' +
      'This is the real output: either nobody was in one of the four supported relationships, or the model missed them. ' +
      'Tracked people are still shown on the frames.';
    const canvas = $('vCanvas');
    canvas.width = 960; canvas.height = Math.round(960 * data.height / data.width);
    drawTimeline(); show(0);
  } catch (err) { $('vPos').textContent = `Cannot load this run's graph: ${err.message}`; }
}
function frameImage(f) {
  if (!viewer.images.has(f)) {
    const img = new Image();
    img.src = `/results/${viewer.jobId}/frames/${String(f).padStart(6, '0')}.jpg`;
    img.onload = () => { if (viewer.data && viewer.data.frames[viewer.i].frame === f) drawFrame(); };
    img.onerror = () => { img.failed = true; if (viewer.data && viewer.data.frames[viewer.i].frame === f) drawFrame(); };
    viewer.images.set(f, img);
  }
  return viewer.images.get(f);
}
function show(i) {
  const d = viewer.data; if (!d) return;
  viewer.i = Math.max(0, Math.min(d.frames.length - 1, i));
  $('vSlider').value = viewer.i;
  const f = d.frames[viewer.i];
  const cur = f.edges.filter(e => e.observed).length, ret = f.edges.length - cur;
  const pos = `Step ${viewer.i + 1} of ${d.frames.length} · video frame ${f.frame} · ${f.t.toFixed(2)} s · ` +
    `${f.edges.length} relationship${f.edges.length === 1 ? '' : 's'} (${cur} predicted, ${ret} retained)` +
    (f.flagged.length ? ` · takeover flag on ${f.flagged.map(person).join(', ')}` : '');
  $('vPos').textContent = pos;
  $('vSlider').setAttribute('aria-valuetext', pos);
  frameImage(f.frame);
  if (viewer.i + 1 < d.frames.length) frameImage(d.frames[viewer.i + 1].frame);   // prefetch
  drawFrame(); renderRows(f); drawTimeline();
}
function renderRows(f) {
  const rows = [...f.edges].sort((a, b) => (b.observed - a.observed) || (a.s - b.s) || (a.o - b.o));
  $('vRows').replaceChildren(...rows.map(e => el('tr', {'data-key': edgeKey(e)},
    el('td', {text: person(e.s)}), el('td', {text: REL[e.p] || e.p}), el('td', {text: person(e.o)}),
    el('td', {class: e.observed ? 'st-cur' : 'st-ret', text: e.observed ? 'Predicted now' : 'Retained'}),
    el('td', {class: 'num', text: score(e.score)}))));
  $('vNone').hidden = rows.length > 0;
}
function drawFrame() {
  const d = viewer.data, f = d.frames[viewer.i], c = $('vCanvas'), g = c.getContext('2d');
  const sx = c.width / d.width, sy = c.height / d.height, img = frameImage(f.frame);
  g.fillStyle = '#20262a'; g.fillRect(0, 0, c.width, c.height);
  if (img.complete && !img.failed && img.naturalWidth) g.drawImage(img, 0, 0, c.width, c.height);
  else { g.fillStyle = '#c9d1cd'; g.font = '18px system-ui'; g.fillText(img.failed ? 'Frame image not available for this run' : 'Loading frame…', 20, 34); }
  const centre = b => [(b[0] + b[2]) / 2 * sx, (b[1] + b[3]) / 2 * sy];
  g.lineWidth = 2.5; g.font = 'bold 15px system-ui';
  for (const [t, b] of Object.entries(f.tracks)) {
    const flagged = f.flagged.includes(+t);
    g.strokeStyle = flagged ? '#e05aa8' : 'rgba(255,255,255,.85)';
    g.setLineDash(flagged ? [3, 3] : []);
    g.strokeRect(b[0] * sx, b[1] * sy, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy);
    g.setLineDash([]);
    const label = `P${t}`, w = g.measureText(label).width + 10;
    g.fillStyle = flagged ? '#b0307d' : 'rgba(23,43,53,.9)';
    g.fillRect(b[0] * sx, b[1] * sy - 22, w, 22);
    g.fillStyle = '#fff'; g.fillText(label, b[0] * sx + 5, b[1] * sy - 6);
  }
  for (const e of f.edges) {
    const a = f.tracks[e.s], b = f.tracks[e.o];
    if (!a || !b) continue;
    const [ax, ay] = centre(a), [bx, by] = centre(b);
    const colour = e.observed ? '#4aa3ff' : '#ffb340';
    g.strokeStyle = colour; g.fillStyle = colour; g.lineWidth = 3; g.setLineDash(e.observed ? [] : [9, 6]);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke(); g.setLineDash([]);
    const ang = Math.atan2(by - ay, bx - ax);
    g.beginPath(); g.moveTo(bx, by); g.lineTo(bx - 14 * Math.cos(ang - 0.4), by - 14 * Math.sin(ang - 0.4));
    g.lineTo(bx - 14 * Math.cos(ang + 0.4), by - 14 * Math.sin(ang + 0.4)); g.closePath(); g.fill();
    const text = (REL[e.p] || e.p).replace('is ', ''), mx = (ax + bx) / 2, my = (ay + by) / 2;
    g.font = 'bold 14px system-ui'; const tw = g.measureText(text).width + 10;
    g.fillStyle = 'rgba(16,22,25,.85)'; g.fillRect(mx - tw / 2, my - 20, tw, 20);
    g.fillStyle = colour; g.fillText(text, mx - tw / 2 + 5, my - 5);
  }
}
const TL = {row: 18, gutter: 190, minCell: 4};
function drawTimeline() {
  const d = viewer.data, c = $('vTimeline'); if (!d) return;
  const n = d.frames.length, cell = Math.max(TL.minCell, Math.floor((c.parentElement.clientWidth - TL.gutter - 4) / Math.max(1, n)));
  c.width = TL.gutter + cell * n + 4;
  c.height = Math.max(40, viewer.keys.length * TL.row + 8);
  const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height);
  g.font = '12px system-ui'; g.textBaseline = 'middle';
  if (!viewer.keys.length) { g.fillStyle = '#4f5d63'; g.fillText('No relationships in this run.', 8, 20); return; }
  viewer.keys.forEach((k, r) => {
    const [s, p, o] = k.split('|');
    g.fillStyle = r % 2 ? '#f6f8f5' : '#fff'; g.fillRect(0, 4 + r * TL.row, c.width, TL.row);
    g.fillStyle = '#172b35'; g.fillText(`P${s} ${(REL[p] || p).replace('is ', '')} P${o}`, 6, 4 + r * TL.row + TL.row / 2);
  });
  d.frames.forEach((f, i) => f.edges.forEach(e => {
    const r = viewer.rowOf.get(edgeKey(e));
    g.fillStyle = e.observed ? '#1f63a8' : '#d48a1c';
    g.fillRect(TL.gutter + i * cell, 6 + r * TL.row, Math.max(1, cell - 1), TL.row - 4);
  }));
  g.fillStyle = '#b0307d';
  g.fillRect(TL.gutter + viewer.i * cell + cell / 2 - 1, 0, 2, c.height);
  c.dataset.cell = cell;
  c.setAttribute('aria-label', `Timeline of ${viewer.keys.length} person-pair relationships over ${n} steps`);
}
$('vTimeline').addEventListener('click', e => {
  const c = $('vTimeline'), rect = c.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (c.width / rect.width) - TL.gutter;
  if (x >= 0) { stopPlay(); show(Math.floor(x / Number(c.dataset.cell || 1))); }
});
function stopPlay() { viewer.playing = false; clearTimeout(viewer.timer); $('vPlay').textContent = 'Play'; }
function play() {
  const d = viewer.data; if (!d) return;
  if (viewer.i >= d.frames.length - 1) show(0);
  viewer.playing = true; $('vPlay').textContent = 'Pause';
  const step = () => {
    if (!viewer.playing) return;
    if (viewer.i >= d.frames.length - 1) { stopPlay(); return; }
    show(viewer.i + 1);
    viewer.timer = setTimeout(step, 1000 * (d.stride / d.fps) / Number($('vSpeed').value));
  };
  viewer.timer = setTimeout(step, 1000 * (d.stride / d.fps) / Number($('vSpeed').value));
}
$('vPlay').addEventListener('click', () => (viewer.playing ? stopPlay() : play()));
$('vPrev').addEventListener('click', () => { stopPlay(); show(viewer.i - 1); });
$('vNext').addEventListener('click', () => { stopPlay(); show(viewer.i + 1); });
$('vFirst').addEventListener('click', () => { stopPlay(); show(0); });
$('vSlider').addEventListener('input', () => { stopPlay(); show(Number($('vSlider').value)); });
$('viewerRegion').addEventListener('keydown', e => {
  if (!viewer.data || ['INPUT', 'SELECT'].includes(e.target.tagName) && e.key !== ' ') return;
  const map = {ArrowRight: () => show(viewer.i + 1), ArrowLeft: () => show(viewer.i - 1), Home: () => show(0),
    End: () => show(viewer.data.frames.length - 1)};
  if (e.key === ' ' && e.target.tagName !== 'BUTTON') { e.preventDefault(); viewer.playing ? stopPlay() : play(); }
  else if (map[e.key]) { e.preventDefault(); stopPlay(); map[e.key](); }
});
window.addEventListener('resize', () => { if (viewer.data) drawTimeline(); });

// ---------- start-up ----------
async function init() {
  methodChanged();
  try {
    const state = await api('/api/status');
    token = state.token; ready = state.ready; maxUpload = state.max_upload; maxSeconds = state.max_seconds;
    serverStarted = state.server_started;
    $('uploadLimit').textContent = `MP4, MOV, AVI, MKV, WebM · up to ${Math.round(maxUpload / 1024 ** 2)} MB and ${Math.round(maxSeconds)} s`;
    $('tau').value = state.default_tau; $('window').value = state.default_window;
    $('readyDot').textContent = ready ? '●' : '○';
    $('readyText').textContent = ready ? 'Setup files found' : 'Setup needs attention';
    $('checks').replaceChildren(...state.checks.map(c => el('li', {class: c.ok ? 'ok' : 'bad', title: c.path},
      c.ok ? c.name : `Missing: ${c.name} (${c.path})`)));
    if (!ready) showError('Some required files are missing (see the setup panel). Analysis is disabled until they are installed.');
    samples = await api('/api/samples');
    if (samples.length) {
      const s = samples[0];
      $('sampleBox').hidden = false;
      $('sampleText').textContent = `${s.title}. Source: ${s.source}. Licence: ${s.licence}. Pedestrian footage without relationship labels, so predictions on it cannot be checked.`;
    }
    await refresh();
  } catch (err) { showError('Cannot connect to TAORG. Start it with start_web.cmd, then refresh this page.'); }
  const poll = async () => {
    try { await refresh(); } catch (err) { showError('Connection to the local server was lost. Saved runs remain on disk; restart the server and refresh.'); }
    setTimeout(poll, jobs.some(j => ACTIVE.includes(j.status)) ? 1500 : 4000);
  };
  setTimeout(poll, 2000);
}
init();
