# TAORG next method — failure audit of the strongest baseline and candidate selection

> Written 2026-10-11 on branch `codex/taorg-next-method`, before any registered comparison.
> **Data used:**
> - existing episode-study artifacts (`results/episodes/`, calibration and the already-reported
>   evaluation);
> - new **designated development seeds 7000–7019**, disjoint from every calibration, evaluation
>   and smoke split of earlier phases and of the planned study.
>
> No new evaluation data was used to choose the failure regime. Audit artifacts:
> `results/next_method/audit/` (`A/`). Scripts: `scripts/next_method/audit_dev.py`,
> `scripts/next_method/dev_pilot.py`.
>
> Frozen and unchanged: episode study STOP, Phase 5H-1 NARROW, Phase 4 NO, Stage 0 WEAK-MIXED.

## 0. Relocation check (step 1 of the task)

| Item | Finding | Action |
|---|---|---|
| Workspace | `D:\ADSA project` (copied from `C:\Users\dhire\OneDrive\...\ADSA project`; the C: copy still exists) | none |
| Python package | The editable install still pointed at the C: copy | Re-installed editable from D: (`pip install -e . --no-deps`). `taorg` now imports from `D:\ADSA project\src` |
| Code/config paths | No source, script, config or test references the old path; only stale `__pycache__` bytecode does (harmless) | none |
| Checkpoints | `D:/TAORG_runs/episodes` (already on D:). SHA-256 of all 140 calibration and 550 evaluation parts re-verified: 0 mismatches | none |
| Tests | 302 passed from the D: location | none |
| Git | Branch, history and remote intact; `HEAD` = pushed `codex/taorg-episodes` (PR #3) | none |
| Resources | C: 5.5 GB free; D: 266 GB free; 32 logical CPUs, but only ≈ 1.6 GB of 32 GB RAM free | ≤ 4 workers |

## 1. Starting point

On the episode-study evaluation (`results/episodes/tables/descriptive.md`), GatedEMA + hysteresis
(**B2**) is the strongest practical baseline:
- it never lost to the episode estimator on hidden termination, misleading flags or boundary
  flicker (episode exit report §9);
- it has the best J in C5 (0.886), S0, S1 and S3.

We start from B2 and do not rebuild the failed estimator.

## 2. Where B2 still fails

### 2.1 Observations

**O1 (calibration artifacts, `results/episodes/raw/calibration_summary.csv.gz`).**
- B2's pooled operating point (#0091) has a per-condition J regret against B2's own best sampled
  configuration of:
  - 0.141 in C1 (gaps);
  - 0.056 in C5;
  - 0.020 in C4;
  - 0.015 in C6 and in C7;
  - 0.000 in C2 and in C3.
- No single B2 configuration is good everywhere: the minimax regret over C1–C7 is 0.141, attained
  by the pooled point itself.

**O2 (development seeds, `A/a2_headroom.csv`).** Condition-matched (label-using) B2 tuning gains
over the pooled point:

| C1 | S2 | C5 | S1 | C4 | C6 | C7 | C2 / C3 / S0 / S3 |
|---|---|---|---|---|---|---|---|
| 0.134 | 0.150 | 0.050 | 0.032 | 0.027 | 0.014 | 0.010 | 0.000 |

The matched configurations differ by condition: #0144 (C1, C4, C7), #0091 (C2, C3, S0, S3),
#0051 (C5, C6, S2), #0042 (S1).

**O3 (development seeds, `A/a3_errors.csv`).** Error decomposition of the frozen B2 point
(mean frames per 12-s video):
- **C1:** 82 missed frames in temporary gaps, plus 49 onset-delay frames.
- **S2:** 172 missed frames in temporary gaps, plus 56 in hidden-termination blocks.
- **C2 / C3:** 291 / 409 false-persistence frames inside OCC-flagged post-termination windows.
- **C6:** 82 ghost frames, plus 84 fragmentation frames.
- **Every condition:** 43–75 onset-delay frames (the on-threshold).

**O4 (development seeds, `A/a4_runs.csv`).** Per video, absence runs that end in re-detection
number:
- 10–58 occluded-onset runs;
- 73 UNKNOWN-onset runs (in S2);
- 35–175 visible-onset runs.

The input stream therefore contains tens of observable run outcomes per video.

### 2.2 Classification of each failure (hypotheses, with the evidence for them)

| Failure | Type | Evidence | Fixed by a simple parameter change? |
|---|---|---|---|
| Missed temporary gaps under OCC/UNK flags (C1, S2, S1, S3) | **Calibration / regime mismatch** of the non-visible absence evidence | A condition-matched B2 recovers 0.13–0.15 J (O2). The pooled point holds non-visible absences too briefly | **No single fixed B2 setting** (O1 minimax); a regime-specific one does, but that needs labels |
| False persistence after hidden termination or under misleading flags (C2, C3) | **Missing information** while the OCC flag lasts | B2's matched headroom is 0 there (O2). Every method tested in the episode study had FPT ≥ 10 s in C2 | No; information-limited (Phase 4 §3; episode formulation §3) |
| Onset delay (all conditions) | **Temporal decision rule** (on-threshold); this is the price of flicker protection | Present in every condition (O3); trades against C5 false births | A lower θ_on trades it against flicker (episode-study C5) |
| Ghosts and fragmentation under ID swaps (C6) | **Identity association** | 82 ghost frames ≈ No-Memory floor (episode study L45); matched headroom 0.014 J | No. Small headroom; the identity cue was nearly inert under pooled tuning (episode post-hoc E1) |

### 2.3 Practical importance beyond the synthetic world (hypotheses, not evidence)

- Real visibility and occlusion estimates differ across trackers, datasets and base models, and a
  post-processor tuned in one setting is used in another. A fixed calibration of "how long a
  hidden relation should be held" is then wrong in at least one regime.
- This is the same transfer problem the episode study's S1–S3 conditions emulate.
- **Untested on real data.**
- ID-switch ghosts may matter more or less in practice depending on the tracker (episode Reviewer A).

## 3. Candidates (at most three)

| | K1: survival-calibrated hold for non-visible absences, learned online from re-detection outcomes | K2: memory remapping on detected identity swaps | K3: risk-calibrated two-sided decision rule |
|---|---|---|---|
| Failure addressed | Gap misses under OCC/UNK and regime mismatch (C1, S2, S1, S3) | C6 ghosts and fragmentation | Onset delay vs false births; false persistence |
| Mechanism | Absence runs are a mixture of "gap" (re-detected) and "terminated" (no return within H). The posterior of a gap after l absent bins under a mixture-cure model is q(l) = 1 − S(H)/S(l). Hold while q ≥ τ. S is a Kaplan–Meier estimate from the stream, shrunk to a calibration prior | When two same-class tracks jump to each other's predicted positions, permute the relation memory between them | Choose on/off thresholds by an explicit cost ratio (SPRT-style) |
| Closest established work | Mixture cure models (Farewell 1982; Maller & Zhou 1996); Kaplan–Meier (1958); online detection-profile learning (Mahler, Vo & Vo 2011); online EM for HMMs (Cappé 2011); tracker coasting buffers (SORT `max_age`, ByteTrack `track_buffer`); age-dependent survival in labelled-RFS trackers (arXiv 1611.06011) | Track re-identification and tracklet re-association; MHT/JPDA (Reid 1979; Fortmann et al. 1983) | Wald 1945 SPRT; Page 1954 CUSUM; hysteresis (Canny 1986). Already tested as the episode estimator's CUSUM (STOP) |
| Possibly distinct contribution | A label-free, nonparametric, visibility-stratified re-detection-time posterior used as the termination rule of a relationship post-processor, recalibrated online per stream. Not new statistics | Thin: tracker-level association repair done in the relation layer; the synthetic swaps are clean pairwise exchanges, which would flatter it | None beyond what was already tested |
| Falsifier | Fails to beat B2 in the primary regime; **or** a fixed per-flag coasting buffer (same information, equal tuning) matches it; **or** the online learning adds nothing over its static prior | Ghost reduction under synthetic swaps only, with C7 false alarms | Matches B2 |
| Added tuning / compute | +2 parameters (τ, κ; horizon H fixed at 3 s); O(H) per decision, O(1) bookkeeping | +2–3 parameters | +1–2 parameters |

### 3.1 Selection (explicit criteria)

1. **Audited headroom in the failure it targets:**
   - K1 targets the largest matched-tuning headroom (C1 0.134, S2 0.150);
   - K2 targets 0.014;
   - K3 targets onset delay, which is a deliberate trade-off.
2. **Equal information.** K1 uses only the input stream and B2's own information. K2 needs
   swap-structure assumptions that the synthetic world satisfies too conveniently.
3. **Smallness.** K1 replaces one decision of B2 (termination during non-visible absence) and
   adds 2 parameters.
4. **Falsifiability against an established alternative.** K1 has two concrete established
   comparators with the same information: the fixed per-flag coasting buffer, and the
   geometric adaptive detection-profile learner.
5. **Not already tested.** K3 essentially re-tests the episode estimator's CUSUM idea.

**Selected: K1.** K2 and K3 are not pursued.

## 4. Development pilot (declared; dev seeds 7000–7009; never evaluation)

`A/dev_pilot.csv` (iteration 1) and `A/dev_pilot_v2_visible_b2.csv` (iteration 2). Prototype
snapshot: `scripts/next_method/dev_snapshots/cure_prototype_pilot_v2.py`. Both iterations used
B2's frozen base parameters with a coarse (τ, κ) grid. **The pilot only checks feasibility; the
registered study re-tunes everything on calibration seeds.**

**Iteration 1: the rule replaces *all* of B2's absence decisions, visible included.**
- *Observation:* every variant is worse than B2 in almost every condition (e.g. km τ = 0.6,
  κ = 20: C2 −0.070, C3 −0.093, C4 −0.135).
- *Explanation (hypothesis):* the self-supervised label "re-detected within H" conflates a
  temporary gap with a genuine short termination followed by recurrence. Visible-absence runs
  that end in recurrence teach the rule to hold through real terminations.
- This is a **counterexample** to the assumption that re-detection implies persistence. It is
  carried into the formulation (§6) as a stated limitation.

**Iteration 2: visible absences keep B2's own rule; the hold applies to non-visible absences only.**
- *Observation:* vs B2, the best variants gain:

  | C1 | S2 | C6 | C4 / C5 | C2 | C3 |
  |---|---|---|---|---|---|
  | +0.116 | +0.18 to +0.20 | +0.058 | +0.014 / +0.015 | −0.014 | −0.005 |

- **But "static" (prior only, no online learning) gives the same gain as "km"** (C1 +0.116 for
  both; S2 +0.179 vs +0.190). Pooled and geometric variants are within about 0.02.
- *Explanation (hypothesis):* almost all of the gain comes from the **rule form**: a calibrated,
  duration-based hold for non-visible absences, decoupled from the visible decision. A static
  rule of this form is equivalent to a **per-flag fixed coasting buffer** (formulation §5,
  Property P2), an established tracker mechanism. Online adaptation contributes at most ≈ 0.01
  in-distribution.

**Consequence for registration.** The decisive question is whether the *online* component beats
a tuned fixed per-flag coasting buffer with the same information and budget, especially under
visibility shift. The pilot makes a **NARROW or STOP outcome likely**. This is declared in the
pre-registration (§2) before any registered comparison. We proceed because the study is small
(about 1 CPU-hour) and decisive, and because a negative answer settles whether this direction can
support an algorithm paper.

## 5. Gaps of this audit

- Synthetic world only (the episode-study world).
- Development headroom uses B2's 200 sampled configurations (a finite search).
- Real-data importance is a hypothesis.
- Iteration-2 effects are from 10 development seeds and a coarse grid.

## Sources (literature, retrieved 2026-10-11)

- Farewell 1982 (Biometrics 38:1041–1046, doi:10.2307/2529885); Maller & Zhou 1996, via the
  review https://projecteuclid.org/journals/statistics-surveys/volume-18/issue-none/Mixture-cure-model-methodology-in-survival-analysis--Some-recent/10.1214/24-SS147.pdf
- Kaplan & Meier 1958 (JASA 53:457–481)
- Mahler, Vo & Vo 2011 (IEEE TSP 59(8):3497–3513, doi:10.1109/TSP.2011.2128316):
  https://espace.curtin.edu.au/handle/20.500.11937/6872
- Cappé & Moulines 2009 (JRSS-B 71(3):593–613); Cappé 2011, online EM for HMMs:
  https://arxiv.org/abs/0908.2359
- Wald 1945 (Ann. Math. Stat. 16(2):117–186): https://projecteuclid.org/euclid.aoms/1177731118
- Labelled-RFS visual MOT with age-dependent survival: https://arxiv.org/pdf/1611.06011
- SoDA lost-track thresholds: https://arxiv.org/pdf/2008.07725
- SORT / ByteTrack / Canny / Reid / Fortmann et al. / Page: see `research/taorg_episode_related_work.md`
