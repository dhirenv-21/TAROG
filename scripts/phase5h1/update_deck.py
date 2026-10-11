"""Update the Phase 5H-1 supervisor deck with the registered results (python-pptx).

Keeps slides 1-5 of results/phase5h1/showcase/phase5h1_supervisor_update.pptx (problem, milestones,
Stage 0, open questions, design), replaces the outdated slide 6 ("approve compute") with four results
slides. Every number is formatted from results/phase5h1/full/analysis.json at build time.

Usage:
    python scripts/phase5h1/update_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

DECK = Path("results/phase5h1/showcase/phase5h1_supervisor_update.pptx")
F = Path("results/phase5h1/full")
INK, SLATE, VERM, GREEN, GREY, WHITE = (RGBColor(0x1D, 0x2B, 0x36), RGBColor(0x4A, 0x5A, 0x66),
                                       RGBColor(0xD5, 0x5E, 0x00), RGBColor(0x00, 0x9E, 0x73),
                                       RGBColor(0x8A, 0x94, 0x9B), RGBColor(0xFF, 0xFF, 0xFF))


def text(slide, x, y, w, h, runs, size=13, color=INK, bold=False, font="Calibri"):
    """runs: list of paragraphs; each paragraph is a str or list of (str, {bold, color}) pieces."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(4)
        for piece, opt in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = piece
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.name = font
            r.font.color.rgb = opt.get("color", color)
    return tb


def titled(prs, layout, title):
    s = prs.slides.add_slide(prs.slide_layouts.get_by_name(layout))
    for ph in list(s.placeholders):                      # pptxgenjs masters: draw the title like the master does
        ph._element.getparent().remove(ph._element)
    text(s, 0.5, 0.35, 9.0, 0.8, [title], size=26 if len(title) < 50 else 22, bold=True,
         color=WHITE if layout == "DARK" else INK, font="Cambria")
    return s


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def main() -> None:
    a = json.loads((F / "analysis.json").read_text(encoding="utf-8"))
    pb = json.loads((F / "posthoc_boundary.json").read_text(encoding="utf-8"))
    C, D, S, ST = a["comparisons"], a["decision"], a["sample"], a["stability"]
    reps = a["nulls"]["replicates"]
    circ_inf = 100 * (min(r["n_events"] for r in reps if r["kind"] == "circ") / S["eval_events"] - 1)
    jit_lo = 100 * (1 - max(r["n_events"] for r in reps if r["kind"] == "jit") / S["eval_events"])
    jit_hi = 100 * (1 - min(r["n_events"] for r in reps if r["kind"] == "jit") / S["eval_events"])
    ci = lambda s: f"{s['mean']:+.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"

    prs = Presentation(str(DECK))
    sld = prs.slides._sldIdLst
    for sid in list(sld)[5:]:                           # drop outdated slide(s) 6+
        prs.part.drop_rel(sid.rId)
        sld.remove(sid)

    # 6 — registered run and decision
    s = titled(prs, "LIGHT", f"Registered comparison: decision {D['label']}")
    s.shapes.add_picture(str(F / "fig_main_comparison.png"), Inches(0.4), Inches(1.15), width=Inches(5.9))
    rows = [("K1 history | heterogeneity + duration", ci(C["G4x-G2"]["EVAL"]), D["K1"]),
            ("K2 history | transition model", ci(C["G4-G3"]["EVAL"]), D["K2"]),
            ("K3 practical size", f"{D['time_excess']:.3f} ≥ {D['k3_threshold']:.3f}", D["K3"]),
            ("K4 reproducible interactions", f"{ST['n_discoveries']} discoveries", D["K4"]),
            ("Families passing", ", ".join(D["families_passing"]), True)]
    paras = [[(lab + "\n", {"bold": True, "size": 11}), (val + "   ", {"size": 11}),
              ("PASS" if ok else "FAIL", {"bold": True, "size": 11, "color": GREEN if ok else VERM})] for lab, val, ok in rows]
    text(s, 6.45, 1.2, 3.3, 3.2, paras, size=11)
    text(s, 6.45, 4.45, 3.3, 0.9, [f"Held-out on {int(S['eval_events']):,} events in 1535 videos; parameters frozen "
                                    f"(commit 289319e) before evaluation. EXPLORATORY, annotations only."],
         size=9, color=SLATE)
    notes(s, "Source: results/phase5h1/full/tables.md T3, T5; analysis.json. GO needs K1-K4; K4 failed, so the "
             "registered rule gives NARROW (K1 + at least one family).")

    # 7 — nulls and negative results
    s = titled(prs, "LIGHT", "Above every null; single interactions do not replicate")
    s.shapes.add_picture(str(F / "fig_null_comparison.png"), Inches(0.35), Inches(1.05), width=Inches(6.2))
    s.shapes.add_picture(str(F / "fig_k4_stability.png"), Inches(0.35), Inches(3.1), width=Inches(6.2))
    text(s, 6.75, 1.3, 2.95, 3.8, [
        [("Positive", {"bold": True, "color": GREEN})],
        f"Observed gains are about 5x the largest of 38 nulls (K1 null max {D['null_cross_max']:.3f}, K2 null max {D['null_time_max']:.3f}).",
        [("Negative findings", {"bold": True, "color": VERM})],
        f"K4: {ST['n_discoveries']} discoveries (min FDR {ST['fdr_scan_min']:.2f} > 0.10).",
        f"Null quality: N-circ +{circ_inf:.0f}% events, N-jit −{jit_lo:.1f}% to −{jit_hi:.1f}%.",
        "K4 statistic is inflated by ridge shrinkage in sparse cells (post-hoc)."], size=10)
    notes(s, "Sources: T4 (nulls), T7 (K4), posthoc_foldnull_cells.json. Whole weight-vector Spearman "
             f"{ST['rho_real']:.3f} beats every null (max {max(ST['rho_null']):.3f}); no single cell passes.")

    # 8 — mechanism and annotation caveat
    s = titled(prs, "LIGHT", "Mechanism, and the annotation-timing caveat")
    s.shapes.add_picture(str(F / "fig_ablation_stability.png"), Inches(0.35), Inches(1.1), width=Inches(6.4))
    ab = a["ablations"]
    text(s, 6.95, 1.15, 2.8, 4.2, [
        [("Mechanism", {"bold": True})],
        f"'Another relation starts → this one ends': {ab['G3+H:ON>OFF-G3']['EVAL']['mean']:+.3f}",
        f"Lags ≤ 2 s carry it; 2–10 s adds {ab['G3+H:lag2-G3']['EVAL']['mean']:+.3f}",
        f"Positive-only (classical Hawkes) is worse: {C['G4+-G4']['EVAL']['mean']:+.3f}",
        [("Caveat", {"bold": True, "color": VERM})],
        f"{100 * pb['EVAL']['share']['tie']:.1f}% of events tie exactly with another relation's boundary on the "
        "same pair (spatial relabelling). Predictive association in annotation timing — not causal, not real video."],
         size=11)
    notes(s, "Sources: T8 ablations, T6 family test, posthoc_boundary.json (POST-HOC, descriptive).")

    # 9 — decision and next step
    s = titled(prs, "DARK", "Decision and next step")
    text(s, 0.5, 1.25, 5.6, 3.6, [
        [("NARROW: ", {"bold": True, "color": RGBColor(0xF2, 0xB4, 0x8C), "size": 16}),
         ("a real, consistent predictive gain from cross-relation timing in VidOR annotations, "
          "mainly spatial and motion relabelling within 2 s.", {"color": WHITE, "size": 15})],
        [("No method contribution: ", {"bold": True, "color": RGBColor(0xF2, 0xB4, 0x8C)}),
         ("standard GLM; classical Hawkes is the worse variant; no reproducible interaction.", {"color": WHITE})],
        [("Main paper track: ", {"bold": True, "color": RGBColor(0xF2, 0xB4, 0x8C)}),
         ("robustness/evaluation paper (Phases 3–4); this as an exploratory section.", {"color": WHITE})]],
         size=14, color=WHITE)
    text(s, 6.4, 1.25, 3.2, 3.6, [
        [("Proposed, not started", {"bold": True, "color": RGBColor(0xF2, 0xB4, 0x8C)})],
        "Independent confirmation (VidVRD or PVSG), pre-registered first:",
        "• repaired nulls (reflection jitter, boundary-tie control)",
        "• support-aware stability statistic",
        "• directional hypotheses fixed from this report",
        [("Demo ≠ evidence: ", {"bold": True, "color": RGBColor(0xF2, 0xB4, 0x8C)}),
         ("the video demo is a synthetic illustration with hand-picked settings.", {"color": RGBColor(0xC9, 0xD3, 0xDA)})]],
         size=11, color=WHITE)
    notes(s, "research/phase5h1_exit_report.md §14-15; research/phase5h1_supervisor_update.md.")
    prs.save(str(DECK))
    print("deck written:", DECK, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
