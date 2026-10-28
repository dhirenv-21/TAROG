/* Minimal SVG charts for the TAORG dashboard (no external libraries). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const tooltip = () => document.getElementById("tooltip");

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) { hi = lo + 1; }
    const raw = (hi - lo) / Math.max(count, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function showTip(evt, html) {
    const t = tooltip();
    if (!t) return;
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = x + "px";
    t.style.top = y + "px";
  }
  function hideTip() { const t = tooltip(); if (t) t.hidden = true; }

  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  /**
   * Line chart.
   * opts: { series: [{id, name, color, x[], y[] (null = gap), dash, hidden}],
   *         xLabel, yLabel, xDomain, yDomain, height, xFormat, yFormat,
   *         vlines: [{x, color, label}], hlines: [{y, color, dash}],
   *         strips: [{label, values[0/1], color, kind: "fill"|"tick"}],
   *         bands: [{x0, x1, color}], legend: true, step: false }
   */
  function lineChart(container, opts) {
    container.classList.add("chart");
    const draw = () => render(container, opts);
    draw();
    if (!container._ro) {
      let last = container.clientWidth;
      container._ro = new ResizeObserver(() => {
        if (Math.abs(container.clientWidth - last) > 4) { last = container.clientWidth; draw(); }
      });
      container._ro.observe(container);
    }
  }

  function render(container, o) {
    container.innerHTML = "";
    const W = Math.max(container.clientWidth || 640, 280);
    const strips = o.strips || [];
    const stripH = 14, stripGap = 4;
    const stripBlock = strips.length ? strips.length * (stripH + stripGap) + 10 : 0;
    const narrow = W < 480;
    const m = { l: narrow ? 44 : (o.marginLeft || 56), r: 14, t: 10 + stripBlock, b: 42 };
    const H = (o.height || 280) + stripBlock;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const visible = o.series.filter(s => !s.hidden);

    const xs = [], ys = [];
    for (const s of visible) {
      s.x.forEach((x, i) => { if (s.y[i] != null && isFinite(s.y[i])) { xs.push(x); ys.push(s.y[i]); } });
    }
    const allX = o.series.flatMap(s => s.x);
    let [x0, x1] = o.xDomain || [Math.min(...allX), Math.max(...allX)];
    let [y0, y1] = o.yDomain || [Math.min(0, ...ys), Math.max(...ys, 0) * 1.08 || 1];
    if (!isFinite(x0) || !isFinite(x1)) { x0 = 0; x1 = 1; }
    if (!isFinite(y0) || !isFinite(y1) || y1 === y0) { y0 = 0; y1 = 1; }
    const sx = v => m.l + (v - x0) / (x1 - x0 || 1) * iw;
    const sy = v => m.t + ih - (v - y0) / (y1 - y0 || 1) * ih;
    const xf = o.xFormat || (v => String(+v.toFixed(3)));
    const yf = o.yFormat || (v => String(+v.toFixed(3)));

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": o.ariaLabel || o.yLabel || "chart" }, container);

    // strips
    strips.forEach((st, k) => {
      const y = 6 + k * (stripH + stripGap);
      el("text", { x: m.l - 8, y: y + stripH - 3, "text-anchor": "end" }, svg).textContent = st.label;
      el("rect", { x: m.l, y, width: iw, height: stripH, fill: "#f3f4f6", rx: 2 }, svg);
      const n = st.values.length, cw = iw / Math.max(n - 1, 1);
      if (st.kind === "tick") {
        st.values.forEach((v, t) => { if (v) el("rect", { x: sx(t) - 0.75, y: y + 2, width: 1.5, height: stripH - 4, fill: st.color }, svg); });
      } else {
        let start = null;
        for (let t = 0; t <= n; t++) {
          const on = t < n && st.values[t];
          if (on && start === null) start = t;
          if (!on && start !== null) {
            el("rect", { x: sx(start) - cw / 2, y, width: (t - start) * cw, height: stripH, fill: st.color, rx: 2 }, svg);
            start = null;
          }
        }
      }
    });

    // bands
    for (const b of o.bands || []) {
      el("rect", { x: sx(b.x0), y: m.t - stripBlock, width: Math.max(sx(b.x1) - sx(b.x0), 2), height: ih + stripBlock, fill: b.color, opacity: 0.5 }, svg);
    }

    // grid + axes
    const g = el("g", { class: "grid" }, svg);
    const yt = niceTicks(y0, y1, 5).filter(v => v >= y0 - 1e-9 && v <= y1 + 1e-9);
    for (const v of yt) {
      el("line", { x1: m.l, x2: m.l + iw, y1: sy(v), y2: sy(v) }, g);
      el("text", { x: m.l - 8, y: sy(v) + 4, "text-anchor": "end" }, svg).textContent = yf(v);
    }
    const xt = niceTicks(x0, x1, narrow ? 4 : 8).filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9);
    for (const v of xt) {
      el("text", { x: sx(v), y: m.t + ih + 16, "text-anchor": "middle" }, svg).textContent = xf(v);
    }
    el("line", { class: "domain", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, svg);
    if (o.xLabel) el("text", { class: "axis-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = o.xLabel;
    if (o.yLabel) el("text", { class: "axis-title", x: 12, y: m.t + ih / 2, transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle" }, svg).textContent = o.yLabel;

    for (const h of o.hlines || []) {
      if (h.y < y0 || h.y > y1) continue;
      el("line", { x1: m.l, x2: m.l + iw, y1: sy(h.y), y2: sy(h.y), stroke: h.color, "stroke-dasharray": h.dash || "4 4", "stroke-width": 1, opacity: 0.7 }, svg);
    }
    for (const v of o.vlines || []) {
      el("line", { x1: sx(v.x), x2: sx(v.x), y1: m.t - stripBlock, y2: m.t + ih, stroke: v.color, "stroke-dasharray": "5 4", "stroke-width": 1.2 }, svg);
      if (v.label) el("text", { x: sx(v.x) + 4, y: m.t + 12, fill: v.color, style: `fill:${v.color}` }, svg).textContent = v.label;
    }

    // series
    for (const s of visible) {
      let d = "", pen = false;
      s.x.forEach((x, i) => {
        const y = s.y[i];
        if (y == null || !isFinite(y)) { pen = false; return; }
        if (o.step && pen) d += `H${sx(x).toFixed(1)}V${sy(y).toFixed(1)}`;
        else d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
        pen = true;
      });
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round" }, svg);
      if (s.markers) s.x.forEach((x, i) => { if (s.y[i] != null) el("circle", { cx: sx(x), cy: sy(s.y[i]), r: 3, fill: "#fff", stroke: s.color, "stroke-width": 1.6 }, svg); });
    }

    // hover
    const hover = el("line", { y1: m.t - stripBlock, y2: m.t + ih, stroke: "#9ca3af", "stroke-width": 1, visibility: "hidden" }, svg);
    const ov = el("rect", { x: m.l, y: 0, width: iw, height: m.t + ih, fill: "transparent" }, svg);
    ov.addEventListener("mousemove", evt => {
      const r = svg.getBoundingClientRect();
      const px = (evt.clientX - r.left) * (W / r.width);
      const xv = x0 + (px - m.l) / iw * (x1 - x0);
      let html = `<div style="margin-bottom:3px;color:#d1d5db">${esc(o.xLabel || "x")}: ${esc(xf(nearestX(visible, xv) ?? xv))}</div>`;
      let snapX = null;
      for (const s of visible) {
        const i = nearestIndex(s.x, xv);
        if (i < 0) continue;
        snapX = s.x[i];
        const y = s.y[i];
        html += `<div class="t-row"><span class="swatch" style="background:${s.color}"></span>${esc(s.name)}<b>${y == null ? "—" : esc(yf(y))}</b></div>`;
      }
      for (const st of strips) {
        const i = Math.round(xv);
        if (i >= 0 && i < st.values.length) html += `<div class="t-row" style="color:#d1d5db">${esc(st.label)}<b>${st.values[i] ? "yes" : "no"}</b></div>`;
      }
      if (snapX != null) { hover.setAttribute("x1", sx(snapX)); hover.setAttribute("x2", sx(snapX)); hover.setAttribute("visibility", "visible"); }
      showTip(evt, html);
    });
    ov.addEventListener("mouseleave", () => { hover.setAttribute("visibility", "hidden"); hideTip(); });

    if (o.legend !== false) {
      const lg = document.createElement("div");
      lg.className = "legend";
      for (const s of o.series) {
        const b = document.createElement("button");
        b.type = "button";
        b.setAttribute("aria-pressed", String(!s.hidden));
        b.innerHTML = `<span class="swatch" style="background:${s.color}"></span>${esc(s.name)}`;
        b.addEventListener("click", () => { s.hidden = !s.hidden; render(container, o); });
        lg.appendChild(b);
      }
      for (const st of strips) {
        const sp = document.createElement("span");
        sp.style.cssText = "display:inline-flex;align-items:center;gap:6px";
        sp.innerHTML = `<span class="swatch" style="background:${st.color}"></span>${esc(st.label)}`;
        lg.appendChild(sp);
      }
      container.appendChild(lg);
    }
  }

  function nearestIndex(arr, v) {
    let best = -1, bd = Infinity;
    arr.forEach((x, i) => { const d = Math.abs(x - v); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  function nearestX(series, v) {
    if (!series.length) return null;
    const i = nearestIndex(series[0].x, v);
    return i < 0 ? null : series[0].x[i];
  }

  window.Charts = { lineChart, showTip, hideTip };
})();
