// Phase 5H-1 supervisor deck (6 slides). Every number is copied from a saved artifact; see speaker notes.
// Usage: NODE_PATH=<dir with pptxgenjs> node scripts/phase5h1/build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = path.join("results", "phase5h1", "showcase", "phase5h1_supervisor_update.pptx");
const FIG = (f) => path.join("results", "phase5h1", "showcase", f);

const COL = { ink: "1D2B36", slate: "4A5A66", mist: "EEF2F4", white: "FFFFFF", verm: "D55E00", blue: "0072B2",
  green: "009E73", amber: "E69F00", pink: "CC79A7", grey: "8A949B" };
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "TAORG project";
pres.title = "Phase 5H-1 supervisor update";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "DARK", background: { color: COL.ink },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9.0, h: 0.8,
    fontFace: HEAD, fontSize: 28, bold: true, color: COL.white, align: "left", valign: "top", margin: 0 }, text: "" } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.4, h: 0.3, color: "9AA7B0", fontFace: BODY, fontSize: 10 },
});
pres.defineSlideMaster({
  title: "LIGHT", background: { color: COL.white },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.35, w: 9.0, h: 0.8,
    fontFace: HEAD, fontSize: 28, bold: true, color: COL.ink, align: "left", valign: "top", margin: 0 }, text: "" } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.4, h: 0.3, color: COL.grey, fontFace: BODY, fontSize: 10 },
});

const txt = (slide, text, o) => slide.addText(text, Object.assign({ isTextBox: true, fontFace: BODY, margin: 0 }, o));
const circle = (slide, x, y, d, fill, label, size = 14) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  txt(slide, label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: size, bold: true, color: COL.white });
};
const card = (slide, x, y, w, h, fill) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });

// ---------------------------------------------------------------- 1. Problem and question
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("Do relationships trigger each other in time?", { placeholder: "title" });
  txt(s, "Research question", { x: 0.5, y: 1.35, w: 5.6, h: 0.35, fontSize: 16, bold: true, color: "F2B48C" });
  txt(s, "Can a temporal point-process (Hawkes-type) model learn dependencies between relationship " +
    "events in video — for example “release starts → hold ends” — beyond what simpler explanations " +
    "(how busy an object pair is, how long relations last, which relations are currently on) already capture?",
    { x: 0.5, y: 1.75, w: 5.6, h: 1.9, fontSize: 16, color: COL.white, valign: "top" });
  txt(s, "Why it matters: a genuine event-dependency model could justify a temporal-reasoning contribution; " +
    "if simple models explain the signal, the extra complexity is not justified.",
    { x: 0.5, y: 3.75, w: 5.6, h: 0.9, fontSize: 14, color: "C9D3DA", italic: true, valign: "top" });
  txt(s, "Evidence classes in this project", { x: 6.55, y: 1.35, w: 3.0, h: 0.35, fontSize: 14, bold: true, color: "F2B48C" });
  const rows = [[COL.blue, "Synthetic worlds", "Phases 3–4"], [COL.verm, "Real ground-truth annotations (VidOR)", "Phase 5H"],
    [COL.grey, "Detector outputs", "not used yet"]];
  rows.forEach(([c, a, b], i) => {
    const y = 1.85 + i * 0.95;
    card(s, 6.55, y, 3.0, 0.8, "26394A");
    s.addShape(pres.shapes.OVAL, { x: 6.7, y: y + 0.25, w: 0.3, h: 0.3, fill: { color: c }, line: { color: c } });
    txt(s, a, { x: 7.12, y: y + 0.08, w: 2.35, h: 0.4, fontSize: 14, bold: true, color: COL.white, valign: "middle" });
    txt(s, b, { x: 7.12, y: y + 0.45, w: 2.35, h: 0.3, fontSize: 12, color: "C9D3DA", valign: "middle" });
  });
  txt(s, "TAORG · Phase 5H-1 progress update · 7 Oct 2026", { x: 0.5, y: 5.15, w: 6, h: 0.3, fontSize: 11, color: "9AA7B0" });
  s.addNotes("Opening: the question is whether relationship events (onsets/offsets of predicates such as hold, release, " +
    "next_to) influence each other's timing, beyond simple explanations. Evidence classes: Phases 3-4 used synthetic " +
    "worlds (results/phase3, results/phase4); Phase 5H uses real VidOR ground-truth annotations only " +
    "(data/vidor, annotations, no videos); no detector outputs have been used anywhere in Phase 5H.");
}

// ---------------------------------------------------------------- 2. Pipeline and milestones
{
  const s = pres.addSlide({ masterName: "LIGHT" });
  s.addText("Pipeline and completed milestones", { placeholder: "title" });
  const steps = [
    [COL.blue, "Phase 3", "Synthetic", "Visibility gating helps under controlled corruption", "H1 = PASS"],
    [COL.blue, "Phase 4", "Synthetic", "Complex V-HPED filter not justified over simple baselines", "Method = NO"],
    [COL.slate, "Phase 5H", "Literature", "Hawkes maths exists already; the gap is empirical", "Track = MODIFY"],
    [COL.verm, "Stage 0", "VidOR GT", "Cross-relation timing signal exists but is small", "WEAK-MIXED"],
    [COL.verm, "Phase 5H-1", "VidOR GT", "Pre-registered; GLM engine built and tested", "Milestone"],
  ];
  const w = 1.7, gap = 0.125, y = 1.45;
  steps.forEach(([c, ph, cls, desc, res], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, y, w, 3.0, COL.mist);
    circle(s, x + 0.12, y + 0.15, 0.45, c, String(i + 1), 16);
    txt(s, ph, { x: x + 0.65, y: y + 0.13, w: w - 0.7, h: 0.3, fontSize: 15, bold: true, color: COL.ink });
    txt(s, cls, { x: x + 0.65, y: y + 0.42, w: w - 0.7, h: 0.25, fontSize: 11, color: c === COL.slate ? COL.slate : c });
    txt(s, desc, { x: x + 0.12, y: y + 0.8, w: w - 0.24, h: 1.4, fontSize: 14, color: COL.ink, valign: "top" });
    card(s, x + 0.12, y + 2.35, w - 0.24, 0.48, c);
    txt(s, res, { x: x + 0.12, y: y + 2.35, w: w - 0.24, h: 0.48, fontSize: 14, bold: true, color: COL.white,
      align: "center", valign: "middle" });
  });
  txt(s, "Every experiment was pre-registered before evaluation; failed and negative results are kept in the record.",
    { x: 0.5, y: 4.7, w: 9.0, h: 0.35, fontSize: 14, italic: true, color: COL.slate });
  s.addNotes("Sources: Phase 3 H1 = PASS (research/phase3_exit_report.md, results/phase3/aggregated/h1_decision.json); " +
    "Phase 4 H2 = MIXED, H3 = MIXED, H4 = FAIL, METHOD CONTRIBUTION = NO (research/phase4_exit_report.md, " +
    "results/phase4/aggregated/decisions.json); Phase 5H gate MODIFY (research/phase5h_hawkes_design.md); " +
    "Stage 0 HAWKES SIGNAL = WEAK-MIXED (research/phase5h_stage0_report.md, results/phase5h/stage0/summary.json); " +
    "Phase 5H-1 pre-registration commit 07bf369, GLM engine commit 4d810ce. Phases 3-4 are synthetic only.");
}

// ---------------------------------------------------------------- 3. Stage 0 result
{
  const s = pres.addSlide({ masterName: "LIGHT" });
  s.addText("Stage 0: a real but small timing signal", { placeholder: "title" });
  s.addImage({ path: FIG("fig_stage0_gain_vs_null.png"), x: 1.05, y: 1.15, w: 7.9, h: 2.73,
    altText: "Observed Stage 0 gain with 95% CI versus 19 shuffled-timing null runs and the practical threshold" });
  const stats = [["+0.074", "observed held-out gain, nats/event (95% CI 0.066–0.083)", COL.verm],
    ["0.040", "median gain after shuffling timing — about 54% of the observed gain", COL.grey],
    ["7%", "timing-specific excess relative to within-relation gain (registered bar: 10%)", COL.ink]];
  stats.forEach(([n, l, c], i) => {
    const x = 0.5 + i * 3.05;
    txt(s, n, { x, y: 3.95, w: 2.85, h: 0.6, fontSize: 32, bold: true, color: c, fontFace: HEAD });
    txt(s, l, { x, y: 4.55, w: 2.85, h: 0.65, fontSize: 12, color: COL.slate, valign: "top" });
  });
  s.addNotes("All values from results/phase5h/stage0/summary.json (VidOR ground-truth annotations; TEST = VidOR " +
    "validation, 835 videos, 50,509 events). Observed HK2-HK1 = +0.0742 [0.0660, 0.0827]; 19 circular-shift null " +
    "values 0.0392-0.0431, median 0.0404; excess 0.0338 = 0.070 x within-relation gain 0.4820. Criteria: C1 pass " +
    "(beats null), C2 pass (beats duration/base controls), C3 fail (below 10%), C4 fail (stable cells 404 vs null " +
    "up to 422). Decision rule frozen in research/phase5h_stage0_preregistration.md. Figure: " +
    "results/phase5h1/showcase/fig_stage0_gain_vs_null.png (generated from saved artifacts only).");
}

// ---------------------------------------------------------------- 4. Unresolved
{
  const s = pres.addSlide({ masterName: "LIGHT" });
  s.addText("What Stage 0 leaves unresolved", { placeholder: "title" });
  const items = [
    [COL.verm, "Heterogeneity", "Busy object pairs produce more events of every kind. More than half of the gain survives shuffled timing, so the model may be measuring activity, not triggering."],
    [COL.blue, "Suppression", "Within 0.5 s, some relations start less often than baseline. A positive-only Hawkes model cannot represent that."],
    [COL.green, "Imbalance", "Top 6 predicates = 86% of events; spatial relations = 63.5%. Contact and vehicle relations are rare."],
    [COL.slate, "No untouched data", "Every annotated VidOR video was used in Stage 0, so Phase 5H-1 is exploratory. Confirmation needs another dataset."],
  ];
  items.forEach(([c, h, b], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.35 + Math.floor(i / 2) * 1.85;
    card(s, x, y, 4.4, 1.65, COL.mist);
    circle(s, x + 0.18, y + 0.2, 0.5, c, String(i + 1), 16);
    txt(s, h, { x: x + 0.85, y: y + 0.22, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: COL.ink });
    txt(s, b, { x: x + 0.85, y: y + 0.65, w: 3.4, h: 0.95, fontSize: 13, color: COL.slate, valign: "top" });
  });
  s.addNotes("1: results/phase5h/stage0/summary.json (null median 0.0404 of 0.0742). 2: peri-event table in " +
    "results/phase5h/stage0/stage0_tables.md, e.g. ON(in_front_of)->ON(next_to) at 0-0.5 s: 0.0001 vs baseline " +
    "0.0087 (descriptive, Stage 0 FIT). 3: results/phase5h1/audit/audit.json (Stage 0 FIT only: 5,600 videos, " +
    "356,040 events; top-6 share 0.864; spatial family 226,176 events = 63.5%); figure " +
    "results/phase5h1/showcase/fig_fit_support.png. 4: data-use audit in research/phase5h1_preregistration.md §2.");
}

// ---------------------------------------------------------------- 5. Phase 5H-1 design + new work
{
  const s = pres.addSlide({ masterName: "LIGHT" });
  s.addText("Phase 5H-1: controlled comparison, now built", { placeholder: "title" });
  txt(s, "Model ladder (one family, nested)", { x: 0.5, y: 1.3, w: 4.2, h: 0.35, fontSize: 15, bold: true, color: COL.ink });
  const ladder = [["G0", "base rates"], ["G1", "+ pair heterogeneity"], ["G2", "+ duration (simple control)"],
    ["G3", "+ current state (transition control)"], ["G4", "+ event history (excite / inhibit)"]];
  ladder.forEach(([g, d], i) => {
    const y = 1.72 + i * 0.6;
    const c = i === 4 ? COL.verm : (i >= 2 ? COL.slate : COL.grey);
    card(s, 0.5 + i * 0.12, y, 4.1 - i * 0.12, 0.5, i === 4 ? "FBE3D3" : COL.mist);
    circle(s, 0.58 + i * 0.12, y + 0.06, 0.38, c, g, 11);
    txt(s, d, { x: 1.06 + i * 0.12, y, w: 3.5 - i * 0.12, h: 0.5, fontSize: 14, color: COL.ink, valign: "middle" });
  });
  txt(s, "Completed this session", { x: 5.1, y: 1.3, w: 4.4, h: 0.35, fontSize: 15, bold: true, color: COL.ink });
  s.addText([
    { text: "Primary-source literature check (frailty, point-process GLM, jitter nulls, semi-Markov)", options: { bullet: true, breakLine: true } },
    { text: "Fitting-split audit: 5,600 videos, 356,040 events, 34 onset / 35 offset targets", options: { bullet: true, breakLine: true } },
    { text: "Pre-registration committed before any comparison (07bf369)", options: { bullet: true, breakLine: true } },
    { text: "GLM engine; likelihood matches an exact reference; 234 tests pass (4d810ce)", options: { bullet: true, breakLine: true } },
    { text: "Smoke run on 300 training videos — pipeline check only, not evidence", options: { bullet: true } },
  ], { x: 5.1, y: 1.72, w: 4.4, h: 3.0, fontSize: 14, color: COL.ink, fontFace: BODY, paraSpaceAfter: 6,
    valign: "top", isTextBox: true, margin: 0 });
  s.addNotes("Ladder: research/phase5h1_preregistration.md §4 (point-process GLM with exp link; piecewise-constant " +
    "covariates; exact compensator). Literature: research/phase5h1_literature.md. Audit: " +
    "results/phase5h1/audit/audit.json and predicate_support.csv. Engine: src/taorg/hawkes/glm.py, tests " +
    "tests/unit/test_hawkes_glm.py (eager-reference likelihood, valid intensity under extreme inhibition, " +
    "positive-only bounds, reproducibility, leakage guard). Full suite 234 passed. Smoke: " +
    "results/phase5h1/smoke/smoke.json — labelled SMOKE, not evidence.");
}

// ---------------------------------------------------------------- 6. Next experiment + gate
{
  const s = pres.addSlide({ masterName: "DARK" });
  s.addText("Next experiment and decision gate", { placeholder: "title" });
  txt(s, "Next run (registered, exploratory)", { x: 0.5, y: 1.3, w: 4.3, h: 0.35, fontSize: 15, bold: true, color: "F2B48C" });
  s.addText([
    { text: "Fit G0–G4 on 5,600 training videos; choose penalty on 700 selection videos", options: { bullet: true, breakLine: true } },
    { text: "Evaluate on 1,535 held-out videos (previously used only in aggregate)", options: { bullet: true, breakLine: true } },
    { text: "Two nulls × 19 runs; 5-fold stability; test 4 Stage 0 hypotheses", options: { bullet: true, breakLine: true } },
    { text: "Needs parallel fitting: estimated 30–60 single-thread CPU-hours", options: { bullet: true } },
  ], { x: 0.5, y: 1.72, w: 4.3, h: 2.6, fontSize: 14, color: COL.white, fontFace: BODY, paraSpaceAfter: 6,
    valign: "top", isTextBox: true, margin: 0 });
  const gates = [["GO", COL.green, "All 4 criteria pass → confirm on an independent dataset"],
    ["NARROW", COL.amber, "Signal only in specific relation families → restrict claims"],
    ["STOP", COL.verm, "Simple models explain it → drop Hawkes; evaluation paper remains"]];
  gates.forEach(([g, c, d], i) => {
    const y = 1.35 + i * 1.12;
    card(s, 5.2, y, 4.3, 0.95, "26394A");
    card(s, 5.32, y + 0.2, 1.15, 0.55, c);
    txt(s, g, { x: 5.32, y: y + 0.2, w: 1.15, h: 0.55, fontSize: 15, bold: true, color: COL.white, align: "center", valign: "middle" });
    txt(s, d, { x: 6.6, y: y + 0.1, w: 2.8, h: 0.75, fontSize: 13, color: COL.white, valign: "middle" });
  });
  txt(s, "Decision needed: approve compute for the full run, and an independent annotated dataset for confirmation.",
    { x: 0.5, y: 4.75, w: 9.0, h: 0.4, fontSize: 14, bold: true, color: "F2B48C" });
  s.addNotes("Plan and rules: research/phase5h1_preregistration.md §2 (splits; data-use audit), §6 (nulls), §7 " +
    "(stability), §8 (hypotheses), §9 (GO / NARROW / STOP). Compute estimate from smoke timing " +
    "(results/phase5h1/smoke/smoke.json: about 20 s per rung on 200 videos) extrapolated to 5,600 videos, the " +
    "4-value ridge grid, 38 null refits of 4 rungs and fold fits — an estimate (30-60 h), not a measurement. Even GO only licenses a confirmatory replication on " +
    "independent data (candidates: ImageNet-VidVRD or PVSG, which annotate relation start/end times).");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("written", f));
