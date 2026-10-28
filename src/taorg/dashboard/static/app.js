/* TAORG dashboard: hash-routed pages over the read-only JSON API. */
(function () {
  "use strict";
  const $main = document.getElementById("main");
  const { lineChart } = window.Charts;

  // ---------------------------------------------------------------- helpers
  async function api(path, opts) {
    let r;
    try { r = await fetch(path, opts); } catch (e) { throw new Error("The dashboard server is not reachable. Is scripts/run_dashboard.py running?"); }
    if (!r.ok) {
      let msg = r.statusText;
      try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) { /* keep statusText */ }
      throw new Error(`${msg} (HTTP ${r.status})`);
    }
    return r.json();
  }
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const f3 = v => (v == null || !isFinite(v) ? "—" : (+v).toFixed(3));
  const f2 = v => (v == null || !isFinite(v) ? "—" : (+v).toFixed(2));
  const signed = (v, d = 3) => (v == null ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const ci = (c, d = 3) => (c ? `[${signed(c[0], d)}, ${signed(c[1], d)}]` : "");

  const METHOD_LABELS = {
    no_memory: "No memory", fixed_window: "Fixed window", ema: "EMA", gated_ema: "Vis-gated EMA",
    hmm: "HMM", vhped_handset: "V-HPED (hand-set)", vt_hmm_fit: "VT-HMM (fit)",
    vhped_novis_fit: "V-HPED no-visibility (fit)", vhped_self_fit: "V-HPED self-only (fit)",
    vhped_full_fit: "V-HPED (fit)", vhped_lazyexact_fit: "V-HPED lazy-exact (fit)",
    vhped_full_fit_prior05: "V-HPED (fit, prior 0.5)", vhped_fit_rhounk05: "V-HPED (fit, ρ_unk = 0.5)",
    vhped_fit_rhounk1: "V-HPED (fit, ρ_unk = 1)", ia_full_fit: "IA-V-HPED (fit)",
    ia_lazyexact_fit: "IA-V-HPED lazy-exact (fit)", ia_nobirth_fit: "IA-V-HPED no-birth (fit)",
    ia_nopost_fit: "IA-V-HPED no-posterior (fit)",
  };
  const COND_LABELS = {
    occlusion_flicker: "Occlusion + flicker", occlusion_flicker_idswitch: "Occlusion + flicker + ID switches",
    swaps0: "No swaps", swaps2_k0: "2 swaps, κ uninformative", swaps2_k05: "2 swaps, κ partly informative",
    swaps2_k1: "2 swaps, κ informative", swaps2_k1_jitter: "2 swaps, κ informative + time jitter",
    swaps2_k1_visflip: "2 swaps, κ informative + 10% visibility flips", swaps4_k0: "4 swaps, κ uninformative",
    swaps4_k1: "4 swaps, κ informative",
  };
  const mLabel = m => METHOD_LABELS[m] || m;
  const cLabel = c => COND_LABELS[c] || c;
  const PALETTE = ["#2563eb", "#d97706", "#0f766e", "#7c3aed", "#dc2626", "#64748b", "#db2777", "#65a30d", "#0891b2", "#a16207", "#4338ca"];
  // Fixed colours for the methods that appear most often, so they read the same on every page.
  const METHOD_COLORS = {
    gated_ema: "#d97706", vt_hmm_fit: "#0f766e", hmm: "#7c3aed", vhped_handset: "#64748b",
    no_memory: "#9ca3af", fixed_window: "#a16207", ema: "#db2777", vhped_lazyexact_fit: "#0891b2",
    vhped_novis_fit: "#dc2626", vhped_self_fit: "#65a30d",
  };
  const colorFor = (list, m, ref) => {
    if (m === ref) return PALETTE[0];
    if (METHOD_COLORS[m]) return METHOD_COLORS[m];
    const free = PALETTE.slice(1).filter(c => !list.some(x => x !== ref && METHOD_COLORS[x] === c));
    const others = list.filter(x => x !== ref && !METHOD_COLORS[x]);
    return free[others.indexOf(m) % free.length] || PALETTE[1];
  };

  const METRICS = {
    frame_f1: { label: "Frame F1", better: "high" },
    persistence_f1: { label: "Persistence F1", better: "high" },
    termination_delay_mean: { label: "Termination delay (frames)", short: "Term. delay", better: "low", digits: 2 },
    false_persistence_rate: { label: "False persistence", better: "low" },
    churn: { label: "Churn", better: "low" },
    ece: { label: "ECE", better: "low" },
    brier: { label: "Brier", better: "low" },
    idswitch_spurious_rate: { label: "ID-switch spurious rate", short: "ID-switch spur.", better: "low" },
    selected_threshold: { label: "Threshold (val.)", short: "Threshold", better: null, digits: 2 },
  };

  function runName(r) {
    const cmd = r.command || "";
    const m = cmd.match(/--config\s+\S*?([\w-]+)\.ya?ml/);
    return m ? m[1] : (r.kind === "matrix" ? "ablation_matrix_synthetic (default)" : r.label);
  }
  function fmtTime(s) {
    if (!s) return "—";
    const iso = /^\d{8}T\d{6}Z$/.test(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}T${s.slice(9, 11)}:${s.slice(11, 13)}:${s.slice(13, 15)}Z` : s;
    const d = new Date(iso);
    return isNaN(d) ? s : d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }
  const statusPill = s => `<span class="pill ${s === "completed" ? "good" : s === "interrupted" || s === "stopped" ? "warn" : "bad"}">${esc(s)}</span>`;

  const loading = (msg = "Loading…") => `<div class="state"><span class="spinner"></span>${esc(msg)}</div>`;
  const errorBox = e => `<div class="notice error" role="alert"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg><div>${esc(e.message || e)}</div></div>`;
  const synthNote = (extra = "") => `<div class="notice"><svg viewBox="0 0 24 24"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></svg><div><b>Synthetic evidence only.</b> Every number here comes from saved runs on generated videos; no real-video result exists yet. ${extra}</div></div>`;

  let runsCache = null;
  async function getRuns(force) {
    if (!runsCache || force) runsCache = await api("/api/runs");
    return runsCache;
  }
  const matrixCache = {};
  async function getMatrix(id) {
    if (!matrixCache[id]) matrixCache[id] = await api(`/api/matrix/${encodeURIComponent(id)}`);
    return matrixCache[id];
  }

  /** Sortable table. columns: [{key, label, num, render(row), value(row), best: "high"|"low"}] */
  function table(container, { columns, rows, sortKey, sortDir = "desc", rowClass }) {
    let key = sortKey, dir = sortDir;
    const draw = () => {
      const col = columns.find(c => c.key === key);
      const sorted = col ? [...rows].sort((a, b) => {
        const va = col.value ? col.value(a) : a[key], vb = col.value ? col.value(b) : b[key];
        const cmp = typeof va === "string" ? va.localeCompare(vb) : (va ?? -Infinity) - (vb ?? -Infinity);
        return dir === "asc" ? cmp : -cmp;
      }) : rows;
      const best = {};
      for (const c of columns) {
        if (!c.best) continue;
        const vals = rows.map(r => (c.value ? c.value(r) : r[c.key])).filter(v => v != null && isFinite(v));
        if (vals.length) best[c.key] = c.best === "high" ? Math.max(...vals) : Math.min(...vals);
      }
      const head = columns.map(c => {
        const sortable = c.sortable !== false;
        const aria = sortable ? ` aria-sort="${c.key === key ? (dir === "asc" ? "ascending" : "descending") : "none"}"` : "";
        return `<th class="${c.num ? "num" : ""}"${aria} scope="col">${sortable ? `<button class="sort" data-k="${c.key}">${esc(c.label)}</button>` : esc(c.label)}</th>`;
      }).join("");
      const body = sorted.map(r => `<tr class="${rowClass ? rowClass(r) : ""}">${columns.map(c => {
        const v = c.value ? c.value(r) : r[c.key];
        const isBest = c.best && v != null && Math.abs(v - best[c.key]) < 1e-12;
        return `<td class="${c.num ? "num" : ""} ${isBest ? "best" : ""}">${c.render ? c.render(r) : esc(v)}</td>`;
      }).join("")}</tr>`).join("");
      container.innerHTML = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${columns.length}" class="state">No rows match.</td></tr>`}</tbody></table></div>`;
      container.querySelectorAll("button.sort").forEach(b => b.addEventListener("click", () => {
        const k = b.dataset.k;
        if (k === key) dir = dir === "asc" ? "desc" : "asc"; else { key = k; dir = columns.find(c => c.key === k).best === "low" ? "asc" : "desc"; }
        draw();
        container.querySelector(`button.sort[data-k="${k}"]`).focus();
      }));
    };
    draw();
  }

  function seg(container, options, value, onChange) {
    container.className = "seg";
    container.setAttribute("role", "group");
    container.innerHTML = options.map(o => `<button type="button" data-v="${esc(o.value)}" aria-pressed="${o.value === value}">${esc(o.label)}</button>`).join("");
    container.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      container.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      onChange(b.dataset.v);
    }));
  }

  // ---------------------------------------------------------------- pages
  const pages = {};

  pages.overview = {
    title: "Overview",
    sub: "Occluded or ended? Visibility-thinned Hawkes filtering for video scene-graph relationships",
    async render(root) {
      const o = await api("/api/overview");
      if (!o.headline) {
        root.innerHTML = `<div class="card"><div class="state">No completed ablation run found under <code>${esc(o.runs_root)}</code>. Run <code>scripts/run_ablation_matrix.py</code> first.</div></div>`;
        return;
      }
      const hl = o.headline, ref = hl.reference_method;
      const a = hl.rows[0];
      const rv = (row, k) => row && row[k] ? row[k][0] : null;
      const p = a.paired || {};
      const cx = o.complexity;
      root.innerHTML = `
        <div class="stack fade-in">
          ${synthNote(`Headline numbers are from run <a href="#/results?run=${encodeURIComponent(hl.run_id)}"><code>${esc(hl.run_id)}</code></a> (5 seeds × 20 test videos).`)}
          <div class="grid grid-4">
            <div class="card stat"><div class="stat-label">Frame F1 · ${esc(cLabel(a.condition))}</div>
              <div class="stat-value">${f3(rv(a.reference, "frame_f1"))}</div>
              <div class="stat-note">${esc(mLabel(ref))} · gated EMA ${f3(rv(a.gated_ema, "frame_f1"))}<br><span class="ci">Δ ${signed(p.frame_f1_video_diff_mean)} ${ci(p.frame_f1_video_ci95)}</span></div></div>
            <div class="card stat"><div class="stat-label">Termination delay (frames)</div>
              <div class="stat-value">${f2(rv(a.reference, "termination_delay_mean"))}</div>
              <div class="stat-note">gated EMA ${f2(rv(a.gated_ema, "termination_delay_mean"))}<br><span class="ci">Δ ${signed(p.termination_delay_mean_video_diff_mean, 2)} ${ci(p.termination_delay_mean_video_ci95, 2)}</span></div></div>
            <div class="card stat"><div class="stat-label">Calibration error (ECE)</div>
              <div class="stat-value">${f3(rv(a.reference, "ece"))}</div>
              <div class="stat-note">gated EMA ${f3(rv(a.gated_ema, "ece"))}<br><span class="ci">EMA scores are not probabilities</span></div></div>
            <div class="card stat"><div class="stat-label">Exact lazy maintenance</div>
              <div class="stat-value">${cx ? f2(cx.max_speedup) + "×" : "—"}</div>
              <div class="stat-note">${cx ? `max speedup · min ${f2(cx.min_speedup)}×<br><span class="ci">${cx.mismatches} decision mismatches over ${cx.grid_points} grid points</span>` : "No complexity run saved"}</div></div>
          </div>
          <div class="grid grid-3-1">
            <div class="card">
              <div class="card-head"><div><h2>Result by condition</h2><p>Test frame F1, mean over seeds; threshold chosen on validation</p></div><span class="spacer"></span><a class="btn" href="#/results?run=${encodeURIComponent(hl.run_id)}">Open results</a></div>
              <div class="card-body flush" id="ovTable"></div>
            </div>
            <div class="card">
              <div class="card-head"><h2>What the project does</h2></div>
              <div class="card-body stack" style="gap:12px">
                <p class="muted">TAORG post-processes per-frame scene-graph detections and decides, for every relationship edge, whether it is <b>temporarily occluded</b> or has <b>really ended</b>, giving a calibrated probability that it is still alive.</p>
                <ul class="plain small">
                  <li><b>V-HPED</b>: visibility-thinned Hawkes detection model with a latent alive/dead chain and exact likelihood.</li>
                  <li><b>Exact lazy maintenance</b>: untouched pairs are updated in closed form and woken by a heap only when they may cross the threshold.</li>
                  <li><b>Identity-aware extension</b>: models tracker ID switches (IA-v1 evaluated, negative; IA-v2 not evaluated).</li>
                </ul>
                <div class="row"><a class="btn primary" href="#/explorer">Explore a video</a><a class="btn" href="#/runs">Run history</a></div>
              </div>
            </div>
          </div>
          <div class="grid grid-2">
            <div class="card"><div class="card-head"><h2>Saved runs</h2><span class="spacer"></span><a href="#/runs" class="small">View all</a></div>
              <div class="card-body row">${Object.entries(o.counts).map(([k, v]) => `${statusPill(k)}<span class="muted small" style="margin-right:10px">${v} run${v === 1 ? "" : "s"}</span>`).join("")}</div></div>
            <div class="card"><div class="card-head"><h2>Parameter identifiability</h2><span class="spacer"></span><a href="#/identifiability" class="small">Details</a></div>
              <div class="card-body small muted" id="ovRec"></div></div>
          </div>
        </div>`;
      table(root.querySelector("#ovTable"), {
        rows: hl.rows, sortKey: null,
        columns: [
          { key: "condition", label: "Condition", sortable: false, render: r => esc(cLabel(r.condition)) },
          { key: "ref", label: mLabel(ref), num: true, sortable: false, render: r => f3(rv(r.reference, "frame_f1")) },
          { key: "g", label: "Vis-gated EMA", num: true, sortable: false, render: r => f3(rv(r.gated_ema, "frame_f1")) },
          { key: "best", label: "Best method", sortable: false, render: r => `${esc(mLabel(r.best_frame_f1.method))} <span class="muted">${f3(r.best_frame_f1.frame_f1[0])}</span>` },
          { key: "v", label: "V-HPED vs gated EMA", sortable: false, render: r => {
            const d = r.paired && r.paired.frame_f1_video_ci95;
            if (!d) return "—";
            if (d[0] > 0) return `<span class="pill good">better ${ci(d)}</span>`;
            if (d[1] < 0) return `<span class="pill bad">worse ${ci(d)}</span>`;
            return `<span class="pill">no clear difference ${ci(d)}</span>`;
          } },
        ],
      });
      const rec = o.recovery;
      const recEl = root.querySelector("#ovRec");
      if (rec) {
        const s = rec.summary;
        const within = Object.entries(s).filter(([, v]) => Math.abs(v.fitted_mean - v.true) <= 0.1 * Math.abs(v.true));
        recEl.innerHTML = `${within.length} of ${Object.keys(s).length} parameters are fitted within 10% of the true value (${within.map(([k]) => `<code>${esc(k)}</code>`).join(", ") || "none"}). The others trade off against each other, so fitted values should not be read as interpretable.`;
      } else recEl.textContent = "No parameter-recovery run saved.";
    },
  };

  pages.explorer = {
    title: "Filter explorer",
    sub: "Replay a saved synthetic test video through the real filters with the run's fitted parameters",
    async render(root, params) {
      const runs = (await getRuns()).filter(r => r.kind === "matrix" && r.status === "completed");
      if (!runs.length) { root.innerHTML = `<div class="card"><div class="state">No completed ablation run to replay.</div></div>`; return; }
      const primary = (await api("/api/overview")).headline?.run_id;
      root.innerHTML = `
        <div class="stack fade-in">
          <div class="notice info"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></svg>
            <div>This regenerates one <b>synthetic</b> test video exactly as the saved run did and runs the selected filters on it, using that run's fitted parameters and validation thresholds. Nothing is retrained and no run is written. Each metric row is checked against the per-video result the run saved.</div></div>
          <form class="card" id="exForm" novalidate>
            <div class="card-body stack" style="gap:14px">
              <div class="controls">
                <label class="field">Run<select id="exRun">${runs.map(r => `<option value="${esc(r.id)}">${esc(runName(r))} · ${esc(r.id.slice(0, 15))}</option>`).join("")}</select></label>
                <label class="field">Condition<select id="exCond"></select></label>
                <label class="field">Seed<select id="exSeed"></select></label>
                <label class="field">Test video<select id="exVideo"></select></label>
              </div>
              <div><div class="small muted" style="margin-bottom:6px" id="exMethodsLabel">Methods (up to 6)</div><div class="checks" id="exMethods" role="group" aria-labelledby="exMethodsLabel"></div></div>
              <div class="row"><button class="btn primary" type="submit" id="exGo"><svg viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg>Run filters</button><span class="small muted" id="exHint"></span></div>
            </div>
          </form>
          <div id="exOut"></div>
        </div>`;
      const $ = s => root.querySelector(s);
      const runSel = $("#exRun");
      runSel.value = params.run && runs.some(r => r.id === params.run) ? params.run : (primary || runs[0].id);
      let opts = null;

      async function loadOptions() {
        $("#exOut").innerHTML = "";
        $("#exGo").disabled = true;
        try { opts = await api(`/api/explorer/${encodeURIComponent(runSel.value)}/options`); }
        catch (e) { $("#exOut").innerHTML = errorBox(e); return; }
        $("#exCond").innerHTML = opts.conditions.map(c => `<option value="${esc(c)}">${esc(cLabel(c))}</option>`).join("");
        $("#exSeed").innerHTML = opts.seeds.map(s => `<option>${s}</option>`).join("");
        $("#exVideo").innerHTML = Array.from({ length: opts.num_test_videos }, (_, i) => `<option value="${i}">#${i + 1}</option>`).join("");
        const defaults = [opts.reference_method, "gated_ema", "vt_hmm_fit", "ia_full_fit"].filter(m => opts.methods.includes(m)).slice(0, 3);
        $("#exMethods").innerHTML = opts.methods.map(m => `<label class="check"><input type="checkbox" value="${esc(m)}" ${defaults.includes(m) ? "checked" : ""}><span class="swatch" style="background:${colorFor(opts.methods, m, opts.reference_method)}"></span>${esc(mLabel(m))}</label>`).join("");
        if (params.condition && opts.conditions.includes(params.condition)) $("#exCond").value = params.condition;
        $("#exGo").disabled = false;
        validate();
      }
      function chosen() { return [...root.querySelectorAll("#exMethods input:checked")].map(i => i.value); }
      function validate() {
        const n = chosen().length;
        const ok = n >= 1 && n <= 6;
        $("#exGo").disabled = !ok;
        $("#exHint").textContent = n === 0 ? "Select at least one method." : n > 6 ? "Select at most 6 methods." : `${n} selected`;
        return ok;
      }
      runSel.addEventListener("change", loadOptions);
      $("#exMethods").addEventListener("change", validate);
      $("#exForm").addEventListener("submit", async ev => {
        ev.preventDefault();
        if (!validate()) return;
        const body = { run_id: runSel.value, condition: $("#exCond").value, seed: +$("#exSeed").value, video_index: +$("#exVideo").value, methods: chosen() };
        const out = $("#exOut");
        out.innerHTML = `<div class="card">${loading("Running filters on the video…")}</div>`;
        $("#exGo").disabled = true;
        try { renderSim(out, await api("/api/explorer/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }), opts); }
        catch (e) { out.innerHTML = errorBox(e); }
        finally { validate(); }
      });
      await loadOptions();
    },
  };

  function renderSim(out, sim, opts) {
    const methods = sim.results.map(r => r.method);
    const color = m => colorFor(opts.methods, m, opts.reference_method);
    const allMatch = sim.results.every(r => r.matches_saved);
    const edges = sim.edges;
    const pick = edges.find(e => e.terminates_at != null && e.occluded_frames > 0) || edges.find(e => e.in_gt) || edges[0];
    const edgeLabel = e => `Pair ${e.subject}→${e.object} · predicate ${e.predicate}` +
      (e.terminates_at != null ? ` · ends @${e.terminates_at}` : e.in_gt ? " · alive to end" : " · not in ground truth") +
      (e.occluded_frames ? ` · ${e.occluded_frames} occluded frames` : "");
    out.innerHTML = `
      <div class="stack fade-in">
        <div class="card">
          <div class="card-head"><div><h2>Per-video metrics · <span class="mono">${esc(sim.video_id)}</span></h2><p>${esc(cLabel(sim.condition))} · seed ${sim.seed} · ${sim.num_frames} frames at ${sim.fps} fps${sim.id_switches.length ? ` · ${sim.id_switches.length} ID switches` : ""}</p></div>
            <span class="spacer"></span>${allMatch ? `<span class="pill good">✓ matches saved run</span>` : `<span class="pill bad">differs from saved run</span>`}</div>
          <div class="card-body flush" id="simTable"></div>
        </div>
        <div class="card">
          <div class="card-head"><div><h2>Edge timeline</h2><p>Score per frame (P(alive) for HMM/V-HPED variants; EMA-type scores are not probabilities). Dashed lines: each method's threshold.</p></div><span class="spacer"></span>
            <label class="field" style="min-width:min(100%,340px)"><span class="sr">Edge</span><select id="simEdge">${edges.map(e => `<option value="${esc(e.key)}" ${e === pick ? "selected" : ""}>${esc(edgeLabel(e))}</option>`).join("")}</select></label></div>
          <div class="card-body"><div id="simChart"></div></div>
        </div>
      </div>`;
    table(out.querySelector("#simTable"), {
      rows: sim.results, sortKey: "frame_f1",
      columns: [
        { key: "method", label: "Method", render: r => `<span class="swatch" style="background:${color(r.method)};margin-right:6px"></span>${esc(mLabel(r.method))}`, value: r => mLabel(r.method) },
        { key: "frame_f1", label: "Frame F1", num: true, best: "high", render: r => f3(r.frame_f1) },
        { key: "persistence_f1", label: "Persistence F1", num: true, best: "high", render: r => f3(r.persistence_f1) },
        { key: "termination_delay_mean", label: "Term. delay", num: true, best: "low", render: r => f2(r.termination_delay_mean) },
        { key: "false_persistence_rate", label: "False persist.", num: true, best: "low", render: r => f3(r.false_persistence_rate) },
        { key: "churn", label: "Churn", num: true, best: "low", render: r => f3(r.churn) },
        ...(sim.id_switches.length ? [{ key: "idswitch_spurious", label: "ID-switch spurious", num: true, render: r => `${r.idswitch_spurious} / ${r.idswitch_predicted}` }] : []),
        { key: "threshold", label: "Threshold", num: true, render: r => f2(r.threshold) },
        { key: "latency_ms_per_frame", label: "ms / frame", num: true, render: r => f3(r.latency_ms_per_frame) },
        { key: "matches_saved", label: "Saved F1", num: true, sortable: false, render: r => `${f3(r.saved_frame_f1)} ${r.matches_saved ? '<span class="pos" title="matches">✓</span>' : '<span class="neg" title="differs">✗</span>'}` },
      ],
    });
    const drawEdge = key => {
      const e = edges.find(x => x.key === key);
      const xs = Array.from({ length: sim.num_frames }, (_, i) => i);
      const series = methods.map(m => ({ id: m, name: mLabel(m), color: color(m), x: xs, y: (sim.scores[m][key] || xs.map(() => null)) }));
      const ymax = Math.max(1, ...series.flatMap(s => s.y.filter(v => v != null)));
      lineChart(out.querySelector("#simChart"), {
        series, xLabel: "frame", yLabel: "score / P(alive)", xDomain: [0, sim.num_frames - 1], yDomain: [0, ymax],
        height: 260, xFormat: v => String(Math.round(v)), yFormat: v => (+v).toFixed(2), step: true,
        hlines: sim.results.map(r => ({ y: r.threshold, color: color(r.method) })),
        vlines: e.terminates_at != null ? [{ x: e.terminates_at + 0.5, color: "#b91c1c", label: "true end" }] : [],
        bands: sim.id_switches.filter(s => s.tracks.includes(e.subject) || s.tracks.includes(e.object)).map(s => ({ x0: s.start - 0.5, x1: s.end + 0.5, color: "#fecaca" })),
        strips: [
          { label: "GT alive", values: e.gt, color: "#4b5563" },
          { label: "Occluded", values: e.occluded, color: "#93c5fd" },
          { label: "Detected", values: e.observed, color: "#111827", kind: "tick" },
        ],
        marginLeft: 72,
      });
    };
    out.querySelector("#simEdge").addEventListener("change", ev => drawEdge(ev.target.value));
    drawEdge(pick.key);
  }

  pages.results = {
    title: "Ablation results",
    sub: "Saved synthetic ablation matrices: test metrics, threshold curves and paired comparisons",
    async render(root, params) {
      const runs = (await getRuns()).filter(r => r.kind === "matrix" && r.status === "completed");
      if (!runs.length) { root.innerHTML = `<div class="card"><div class="state">No completed ablation run saved.</div></div>`; return; }
      const primary = (await api("/api/overview")).headline?.run_id;
      root.innerHTML = `
        <div class="stack fade-in">
          ${synthNote("Values are mean ± s.d. over seeds; best value per column is highlighted.")}
          <div class="card"><div class="card-body row">
            <label class="field" style="min-width:min(100%,360px)">Run<select id="rsRun">${runs.map(r => `<option value="${esc(r.id)}">${esc(runName(r))} · ${esc(r.id.slice(0, 15))}</option>`).join("")}</select></label>
            <div class="field" style="flex:1;min-width:240px"><span class="small muted" style="font-weight:500">Condition</span><div id="rsCond"></div></div>
          </div></div>
          <div id="rsBody"></div>
        </div>`;
      const runSel = root.querySelector("#rsRun");
      runSel.value = params.run && runs.some(r => r.id === params.run) ? params.run : (primary || runs[0].id);
      let cond = null;
      const body = root.querySelector("#rsBody");
      async function loadRun() {
        body.innerHTML = `<div class="card">${loading()}</div>`;
        let m;
        try { m = await getMatrix(runSel.value); } catch (e) { body.innerHTML = errorBox(e); return; }
        cond = m.conditions.includes(cond) ? cond : m.conditions[0];
        seg(root.querySelector("#rsCond"), m.conditions.map(c => ({ value: c, label: cLabel(c) })), cond, c => { cond = c; draw(m); });
        draw(m);
      }
      function draw(m) {
        const ref = m.reference_method;
        const rows = m.table.filter(t => t.condition === cond);
        const metricKeys = Object.keys(METRICS).filter(k => rows[0] && rows[0][k]);
        body.innerHTML = `
          <div class="stack">
            <div class="card"><div class="card-head"><div><h2>Test metrics</h2><p>${esc(cLabel(cond))} · reference method <b>${esc(mLabel(ref))}</b> (shaded) · click a column to sort</p></div><span class="spacer"></span>
              <a class="btn" href="/api/files/${encodeURIComponent(m.run.id)}/metrics.json" download>Download metrics.json</a></div>
              <div class="card-body flush" id="rsTable"></div></div>
            <div class="card"><div class="card-head"><div><h2>Threshold sweep</h2><p>Seed-mean test curves; click legend entries to hide or show methods</p></div><span class="spacer"></span>
              <label class="field"><span class="sr">Metric</span><select id="rsMetric">${["frame_f1", "persistence_f1", "termination_delay_mean", "false_persistence_rate", "churn"].map(k => `<option value="${k}">${esc(METRICS[k].label)}</option>`).join("")}</select></label></div>
              <div class="card-body"><div id="rsChart"></div></div></div>
            <div class="card"><div class="card-head"><div><h2>Paired comparison against ${esc(mLabel(ref))}</h2><p>Δ = reference − method. Seed-level mean ± s.d.; video-level 95% bootstrap CI (10,000 resamples)</p></div></div>
              <div class="card-body flush" id="rsPaired"></div></div>
          </div>`;
        table(body.querySelector("#rsTable"), {
          rows, sortKey: "frame_f1", rowClass: r => (r.method === ref ? "ref" : ""),
          columns: [
            { key: "method", label: "Method", value: r => mLabel(r.method), render: r => esc(mLabel(r.method)) },
            ...metricKeys.map(k => ({
              key: k, label: METRICS[k].short || METRICS[k].label, num: true, best: METRICS[k].better,
              value: r => r[k][0],
              render: r => `${(METRICS[k].digits === 2 ? f2 : f3)(r[k][0])}${k !== "selected_threshold" ? `<span class="sd">±${(METRICS[k].digits === 2 ? f2 : f3)(r[k][1])}</span>` : ""}`,
            })),
          ],
        });
        const methods = m.methods;
        const metricSel = body.querySelector("#rsMetric");
        const shown = new Set([ref, "gated_ema", "vt_hmm_fit", "hmm", "no_memory", "ia_full_fit", "vhped_handset"]);
        const crowded = methods.length > 6;
        const drawCurve = () => {
          const k = metricSel.value;
          const series = methods.map(name => {
            const c = m.curves[`${cond}|${name}`];
            return c && { id: name, name: mLabel(name), color: colorFor(methods, name, ref), x: c.threshold, y: c[k], width: name === ref ? 2.4 : 1.6, markers: true, hidden: crowded && !shown.has(name) };
          }).filter(Boolean);
          lineChart(body.querySelector("#rsChart"), { series, xLabel: "decision threshold", yLabel: METRICS[k].label, xDomain: [0, 1], height: 300 });
        };
        metricSel.addEventListener("change", drawCurve);
        drawCurve();
        const paired = m.paired.filter(p => p.condition === cond);
        const dcell = (p, k, d = 3) => {
          const v = p[`${k}_seed_diff`];
          return v ? `${signed(v[0], d)}<span class="sd">±${Math.abs(v[1]).toFixed(d)}</span>` : "—";
        };
        const ciCell = (p, k, better, d = 3) => {
          const c = p[`${k}_video_ci95`];
          if (!c) return "—";
          const cls = (better === "high" ? c[0] > 0 : c[1] < 0) ? "pos" : (better === "high" ? c[1] < 0 : c[0] > 0) ? "neg" : "";
          return `<span class="${cls}">${signed(p[`${k}_video_diff_mean`], d)}</span> <span class="ci">${ci(c, d)}</span>`;
        };
        table(body.querySelector("#rsPaired"), {
          rows: paired, sortKey: null,
          columns: [
            { key: "b", label: "Compared with", sortable: false, render: p => esc(mLabel(p.b)) },
            { key: "f", label: "Δ frame F1 (seeds)", num: true, sortable: false, render: p => dcell(p, "frame_f1") },
            { key: "fs", label: "Seeds ref. higher", num: true, sortable: false, render: p => `${p.frame_f1_a_greater_seeds ?? "—"} / ${m.run.seeds?.length ?? 5}` },
            { key: "fc", label: "Δ frame F1 (videos, 95% CI)", num: true, sortable: false, render: p => ciCell(p, "frame_f1", "high") },
            { key: "dc", label: "Δ term. delay (videos, 95% CI)", num: true, sortable: false, render: p => ciCell(p, "termination_delay_mean", "low", 2) },
          ],
        });
      }
      runSel.addEventListener("change", loadRun);
      await loadRun();
    },
  };

  pages.efficiency = {
    title: "Lazy efficiency",
    sub: "Eager vs exact-lazy V-HPED per-frame cost on synthetic touch patterns",
    async render(root) {
      const runs = (await getRuns()).filter(r => r.kind === "complexity");
      if (!runs.length) { root.innerHTML = `<div class="card"><div class="state">No complexity benchmark saved. Run <code>scripts/run_complexity_benchmark.py</code>.</div></div>`; return; }
      const d = await api(`/api/complexity/${runs[0].id}`);
      const rows = d.rows;
      const pairs = [...new Set(rows.map(r => r.num_pairs))].sort((a, b) => a - b);
      const fracs = [...new Set(rows.map(r => r.touch_fraction))].sort((a, b) => a - b);
      const mism = rows.reduce((s, r) => s + r.decision_mismatches, 0);
      const best = rows.reduce((a, b) => (b.speedup > a.speedup ? b : a));
      const worst = rows.reduce((a, b) => (b.speedup < a.speedup ? b : a));
      root.innerHTML = `
        <div class="stack fade-in">
          ${synthNote(`Latencies are pure-Python timings on one machine (${esc(d.machine || "unknown")}); memory was not measured.`)}
          <div class="grid grid-4">
            <div class="card stat"><div class="stat-label">Decision mismatches</div><div class="stat-value">${mism}</div><div class="stat-note">eager vs lazy, ${rows.length} grid points</div></div>
            <div class="card stat"><div class="stat-label">Best speedup</div><div class="stat-value">${f2(best.speedup)}×</div><div class="stat-note">${best.num_pairs} pairs, ${best.touch_fraction * 100}% touched per frame</div></div>
            <div class="card stat"><div class="stat-label">Worst speedup</div><div class="stat-value">${f2(worst.speedup)}×</div><div class="stat-note">${best.num_pairs === worst.num_pairs ? "" : `${worst.num_pairs} pairs, `}${worst.touch_fraction * 100}% touched${worst.speedup < 1 ? ": lazy is slower" : ""}</div></div>
            <div class="card stat"><div class="stat-label">Benchmark</div><div class="stat-value">${d.args.frames} <span class="small muted">frames</span></div><div class="stat-note">${d.args.predicates.join(", ")} predicates · pairs ${pairs.join(", ")}</div></div>
          </div>
          <div class="grid grid-2">
            <div class="card"><div class="card-head"><div><h2>Speedup vs touch fraction</h2><p>Eager ms ÷ lazy ms; dashed line = break-even</p></div></div><div class="card-body"><div id="cxSpeed"></div></div></div>
            <div class="card"><div class="card-head"><div><h2>Per-frame latency</h2><p>Mean ms per frame vs number of pairs</p></div><span class="spacer"></span><div id="cxFrac"></div></div><div class="card-body"><div id="cxLat"></div></div></div>
          </div>
          <div class="card"><div class="card-head"><h2>All grid points</h2><span class="spacer"></span><a class="btn" href="/api/files/${esc(runs[0].id)}/results.json" download>Download results.json</a></div><div class="card-body flush" id="cxTable"></div></div>
        </div>`;
      lineChart(root.querySelector("#cxSpeed"), {
        series: pairs.map((p, i) => { const rs = rows.filter(r => r.num_pairs === p).sort((a, b) => a.touch_fraction - b.touch_fraction); return { id: p, name: `${p} pairs`, color: PALETTE[i], x: rs.map(r => r.touch_fraction), y: rs.map(r => r.speedup), markers: true }; }),
        xLabel: "fraction of pairs touched per frame", yLabel: "speedup (×)", xDomain: [0, 1], hlines: [{ y: 1, color: "#6b7280" }], height: 260,
        xFormat: v => `${Math.round(v * 100)}%`, yFormat: v => (+v).toFixed(1),
      });
      let frac = fracs[0];
      const drawLat = () => {
        const rs = rows.filter(r => r.touch_fraction === frac).sort((a, b) => a.num_pairs - b.num_pairs);
        lineChart(root.querySelector("#cxLat"), {
          series: [
            { id: "eager", name: "Eager", color: PALETTE[1], x: rs.map(r => r.num_pairs), y: rs.map(r => r.eager_ms_mean), markers: true },
            { id: "lazy", name: "Lazy (exact)", color: PALETTE[0], x: rs.map(r => r.num_pairs), y: rs.map(r => r.lazy_ms_mean), markers: true },
          ], xLabel: "number of pairs", yLabel: "ms / frame", height: 260, xFormat: v => String(Math.round(v)), yFormat: v => (+v).toFixed(0),
        });
      };
      seg(root.querySelector("#cxFrac"), fracs.map(f => ({ value: String(f), label: `${f * 100}%` })), String(frac), v => { frac = +v; drawLat(); });
      drawLat();
      table(root.querySelector("#cxTable"), {
        rows, sortKey: "num_pairs", sortDir: "asc",
        columns: [
          { key: "num_pairs", label: "Pairs", num: true, render: r => r.num_pairs },
          { key: "touch_fraction", label: "Touched", num: true, render: r => `${r.touch_fraction * 100}%` },
          { key: "eager_ms_mean", label: "Eager ms", num: true, render: r => f2(r.eager_ms_mean) },
          { key: "lazy_ms_mean", label: "Lazy ms", num: true, render: r => f2(r.lazy_ms_mean) },
          { key: "speedup", label: "Speedup", num: true, best: "high", render: r => `<span class="${r.speedup >= 1 ? "pos" : "neg"}">${f2(r.speedup)}×</span>` },
          { key: "pair_steps", label: "Pair steps", num: true, render: r => `${r.pair_steps.toLocaleString()} / ${r.eager_equivalent_pair_steps.toLocaleString()}` },
          { key: "wakeups", label: "Wake-ups", num: true, render: r => r.wakeups.toLocaleString() },
          { key: "heap_max", label: "Heap max", num: true, value: r => r.heap.max_size, render: r => r.heap.max_size.toLocaleString() },
          { key: "decision_mismatches", label: "Mismatches", num: true, render: r => (r.decision_mismatches === 0 ? `<span class="pill good">0</span>` : `<span class="pill bad">${r.decision_mismatches}</span>`) },
        ],
      });
    },
  };

  pages.identifiability = {
    title: "Identifiability",
    sub: "Can V-HPED's parameters be recovered from data generated by V-HPED itself?",
    async render(root) {
      const runs = (await getRuns()).filter(r => r.kind === "param_recovery");
      if (!runs.length) { root.innerHTML = `<div class="card"><div class="state">No parameter-recovery run saved. Run <code>scripts/run_param_recovery.py</code>.</div></div>`; return; }
      const d = await api(`/api/recovery/${runs[0].id}`);
      const rows = Object.entries(d.summary).map(([k, v]) => ({ param: k, ...v, rel: (v.fitted_mean - v.true) / Math.abs(v.true) }));
      root.innerHTML = `
        <div class="stack fade-in">
          ${synthNote(`Data were simulated from known parameters (${esc(d.args.videos)} videos per seed, seeds ${esc(d.args.seeds)}, ${esc(d.args.iters)} iterations) and the model was re-fitted from the hand-set initialisation.`)}
          <div class="grid grid-3-1">
            <div class="card"><div class="card-head"><div><h2>Recovered parameters</h2><p>Relative error = (fitted mean − true) / |true|. “Within 10%” is a display convention, not a test.</p></div></div><div class="card-body flush" id="prTable"></div></div>
            <div class="card"><div class="card-head"><h2>Reading this</h2></div><div class="card-body small muted stack" style="gap:10px">
              <p>In this saved run, the observation-side parameters (visibility thinning ρ<sub>occ</sub> and the false-positive rate ε) are recovered closely.</p>
              <p>The excitation parameters (μ, α, β) are not: they trade off against each other, so different values explain the data almost equally well. The cause is not established.</p>
              <p>Consequence: fitted V-HPED parameters should not be interpreted one by one.</p></div></div>
          </div>
        </div>`;
      table(root.querySelector("#prTable"), {
        rows, sortKey: "rel_abs", sortDir: "asc",
        columns: [
          { key: "param", label: "Parameter", render: r => `<code>${esc(r.param)}</code>` },
          { key: "true", label: "True", num: true, render: r => f3(r.true) },
          { key: "init", label: "Init", num: true, render: r => f3(r.init) },
          { key: "fitted_mean", label: "Fitted (mean ± s.d.)", num: true, render: r => `${f3(r.fitted_mean)}<span class="sd">±${f3(r.fitted_std)}</span>` },
          { key: "rel_abs", label: "Relative error", num: true, value: r => Math.abs(r.rel), render: r => `<span style="display:inline-flex;align-items:center;gap:8px"><span style="display:inline-block;width:70px;height:6px;background:#eef0f3;border-radius:3px;overflow:hidden"><span style="display:block;height:100%;width:${Math.min(Math.abs(r.rel), 1) * 100}%;background:${Math.abs(r.rel) <= 0.1 ? "#10b981" : "#f59e0b"}"></span></span>${signed(r.rel * 100, 1)}%</span>` },
          { key: "status", label: "", sortable: false, render: r => (Math.abs(r.rel) <= 0.1 ? `<span class="pill good">within 10%</span>` : `<span class="pill warn">not recovered</span>`) },
        ],
      });
    },
  };

  pages.runs = {
    title: "Run history",
    sub: "Every saved run under outputs/runs, including interrupted ones",
    async render(root) {
      const runs = await getRuns(true);
      const kinds = [...new Set(runs.map(r => r.kind))];
      const statuses = [...new Set(runs.map(r => r.status))];
      root.innerHTML = `
        <div class="stack fade-in">
          <div class="card"><div class="card-body row">
            <label class="field" style="flex:1;min-width:200px">Search<input type="search" id="rnQ" placeholder="Run id, config or command"></label>
            <label class="field">Type<select id="rnKind"><option value="">All</option>${kinds.map(k => `<option>${esc(k)}</option>`).join("")}</select></label>
            <label class="field">Status<select id="rnStatus"><option value="">All</option>${statuses.map(k => `<option>${esc(k)}</option>`).join("")}</select></label>
          </div></div>
          <div class="card"><div class="card-head"><h2 id="rnCount"></h2></div><div class="card-body flush" id="rnTable"></div></div>
        </div>`;
      const $ = s => root.querySelector(s);
      const DL = ["metrics.json", "results.json", "metadata.json", "config.resolved.yaml", "INTERRUPTED.txt", "STOPPED.txt"];
      const draw = () => {
        const q = $("#rnQ").value.trim().toLowerCase(), k = $("#rnKind").value, s = $("#rnStatus").value;
        const rows = runs.filter(r => (!k || r.kind === k) && (!s || r.status === s) &&
          (!q || [r.id, r.label, r.command, runName(r)].join(" ").toLowerCase().includes(q)));
        $("#rnCount").textContent = `${rows.length} of ${runs.length} runs`;
        table($("#rnTable"), {
          rows, sortKey: "start", value: null,
          columns: [
            { key: "start", label: "Started", value: r => (r.start_time || r.id.split("/").pop()).replace(/\D/g, "").slice(0, 14), render: r => esc(fmtTime(r.start_time || r.id.split("/").pop().slice(0, 15).replace(/^(\d{8})_(\d{6})$/, "$1T$2Z"))) },
            { key: "id", label: "Run", render: r => `<div class="mono">${esc(r.id)}</div><div class="small muted">${esc(runName(r))}</div>${r.note ? `<div class="small" style="color:var(--warn);white-space:normal;max-width:420px">${esc(r.note)}</div>` : ""}` },
            { key: "kind", label: "Type", render: r => `<span class="pill">${esc(r.kind)}</span>` },
            { key: "status", label: "Status", render: r => statusPill(r.status) },
            { key: "duration_s", label: "Duration", num: true, render: r => (r.duration_s ? `${(r.duration_s / 60).toFixed(1)} min` : "—") },
            { key: "files", label: "Files", sortable: false, render: r => DL.filter(f => r.files.includes(f)).map(f => `<a class="small" href="/api/files/${r.id.split("/").map(encodeURIComponent).join("/")}/${f}" download>${esc(f)}</a>`).join(" · ") || '<span class="muted small">—</span>' },
            { key: "open", label: "", sortable: false, render: r => (r.kind === "matrix" && r.status === "completed" ? `<a class="small" href="#/results?run=${encodeURIComponent(r.id)}">Results</a> · <a class="small" href="#/explorer?run=${encodeURIComponent(r.id)}">Explore</a>` : r.kind === "complexity" ? `<a class="small" href="#/efficiency">View</a>` : r.kind === "param_recovery" ? `<a class="small" href="#/identifiability">View</a>` : "") },
          ],
        });
      };
      ["#rnQ", "#rnKind", "#rnStatus"].forEach(s => $(s).addEventListener("input", draw));
      draw();
    },
  };

  // ---------------------------------------------------------------- router & shell
  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    const [name, query] = raw.split("?");
    return { name: pages[name] ? name : "overview", params: Object.fromEntries(new URLSearchParams(query || "")) };
  }
  let renderToken = 0;
  async function route() {
    const { name, params } = parseHash();
    const page = pages[name];
    const token = ++renderToken;
    document.querySelectorAll("nav a").forEach(a => {
      const on = a.dataset.route === name;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    document.getElementById("pageTitle").textContent = page.title;
    document.getElementById("pageSub").textContent = page.sub;
    document.title = `${page.title} · TAORG Dashboard`;
    closeMenu();
    window.Charts.hideTip();
    $main.innerHTML = loading();
    try {
      const host = document.createElement("div");
      await page.render(host, params);
      if (token === renderToken) { $main.replaceChildren(host); window.scrollTo(0, 0); }
    } catch (e) {
      if (token === renderToken) $main.innerHTML = errorBox(e);
    }
  }

  const sidebar = document.getElementById("sidebar"), scrim = document.getElementById("scrim"), menuBtn = document.getElementById("menuBtn");
  function closeMenu() { sidebar.classList.remove("open"); scrim.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); }
  menuBtn.addEventListener("click", () => { const open = !sidebar.classList.contains("open"); sidebar.classList.toggle("open", open); scrim.hidden = !open; menuBtn.setAttribute("aria-expanded", String(open)); });
  scrim.addEventListener("click", closeMenu);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeMenu(); });

  async function health() {
    const st = document.getElementById("apiStatus");
    try {
      const [hh, runs] = await Promise.all([api("/api/health"), getRuns()]);
      st.innerHTML = hh.runs_root_exists
        ? `<span class="dot dot-good"></span><span>API connected<span class="label-long"> · ${runs.length} saved runs</span></span>`
        : `<span class="dot dot-amber"></span><span>No runs folder</span>`;
    } catch (e) {
      st.innerHTML = `<span class="dot dot-bad"></span><span>API unavailable</span>`;
    }
  }

  window.addEventListener("hashchange", route);
  health();
  route();
})();
