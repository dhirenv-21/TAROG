# Stage 2 — Literature Review and Novelty Audit

> Date: 2026-10-28. Method: web search of primary sources (arXiv abstract pages,
> CVF open access, publisher/USGS pages). **Coverage is partial**: about a dozen
> targeted searches, abstracts rather than full texts for most items. Any claim
> below marked *(abstract only)* needs a full-text check before it goes in a paper.
> No claim of novelty in this repository should be treated as verified beyond
> what this audit supports.

## 1. Closest prior work found

| Prior work | Main method | Handles occlusion? | Models termination? | Predicate dependencies? | Handles ID switches? | Difference from V-HPED |
|---|---|---|---|---|---|---|
| **Markov-modulated Hawkes process** — Wang, Bebbington & Harte, *Ann. Inst. Stat. Math.* 2012 ([ism.ac.jp](https://www.ism.ac.jp/editsec/aism/pdf/10463_2010_Article_320.pdf)); Wu et al., arXiv 1903.03223 ([arXiv](https://arxiv.org/pdf/1903.03223)) | Hidden Markov state switches Hawkes background/decay; EM | — (no detection model) | Hidden state can switch to "quiescent" | Multivariate in later work | No | **Same mathematical family as V-HPED's alive/dead × Hawkes core.** V-HPED adds visibility-dependent thinning of the observations, a per-edge birth process, cross-predicate hazard, and the VidSGG application. The core "hidden on/off state modulating a Hawkes process + forward filtering" is **established**. |
| **Thinned / missing-event Hawkes** — Deutsch & Ross, arXiv 2006.09015 ([arXiv](https://arxiv.org/abs/2006.09015)) | Hawkes events observed through a detection function; ABC inference | Missing events ≈ detection failure | No | No | No | Detection thinning of Hawkes events is **established** (per search summary; *full text not checked*). V-HPED's thinning is regime-dependent and enters an exact discrete-time filter rather than ABC. |
| **Thinning** — Lewis & Shedler 1979 | Poisson thinning | — | — | — | — | Standard operation. |
| **Dynamic occupancy models** — MacKenzie et al., *Ecology* 84(8):2200–2207, 2003 ([USGS](https://www.usgs.gov/publications/estimating-site-occupancy-colonization-and-local-extinction-when-a-species-detected)) | Latent occupied/unoccupied state with colonization/extinction; detection probability < 1; covariates on detection | Imperfect detection (the ecological analogue of occlusion) | **Yes — "local extinction"** | No | No | **Very close conceptual analogue**: latent presence + imperfect, covariate-dependent detection + extinction. V-HPED differs by self-exciting (history-dependent) detection rates and continuous-time hazards. "Visibility-dependent detection probability" is a detection covariate in occupancy terms → **established idea**. |
| **Labeled RFS multi-object tracking** — arXiv 1611.06011 ([arXiv](https://arxiv.org/pdf/1611.06011)) | Survival probability delays track termination when occlusion is likely | Yes | Yes (track termination) | — | Implicitly (labels) | Same principle for *object tracks*, not relationship edges. |
| **PDAF / JPDAF** — Bar-Shalom & Tse, *Automatica* 1975 ([pdf](https://lankku.kapsi.fi/Moo/kirjallisuus/bar-shalom-1975.pdf)); review Bar-Shalom, Daum & Huang 2009 | Association-probability-weighted measurement updates | — | No (PDAF does not confirm/delete tracks) | — | Association uncertainty | **Established** principle for weighting evidence by association probability; relevant to Direction A. |
| **World Scene Graph Generation / ActionGenome4D** — Peddi et al., arXiv 2603.13185, Mar 2026 ([arXiv](https://arxiv.org/abs/2603.13185)) | 3D world-frame graph incl. unobserved objects; PWG (object-permanence buffer), MWAE, 4DST | **Yes** (core topic) | Not addressed *(abstract only)* | Not stated | Not addressed *(abstract only)* | Closest VidSGG competitor on occlusion. Learned, 3D, end-to-end-ish; does not model *why* a relation stopped being observed or calibrated termination. Must be cited and compared conceptually. |
| **RTAformer** — Heo, Lim & Choi, arXiv 2610.20091, ACCV 2026 ([arXiv](https://arxiv.org/abs/2610.20091)) | Separate entity persistence vs relation-change streams ("relational semantic inertia") | Not mentioned *(abstract only)* | Relation transitions (not explicit termination) | Implicit | Cross-frame alignment (drift), not ID noise | Addresses the opposite failure (sticking to stale predicates) inside the network; not post-hoc, not probabilistic. |
| **TCDSG** — Ruschel et al., arXiv 2412.02808 ([arXiv](https://arxiv.org/pdf/2412.02808)) | End-to-end temporally consistent dynamic scene graphs, sequence-level matching; MEVA re-annotation with persistent IDs | — | Action tracklets | — | **Yes — identity consistency is the topic**; lists query switches and multi-hypothesis methods as future work | End-to-end identity consistency; no post-hoc probabilistic identity-uncertainty filter. |
| **TEMPURA** — Nag et al., CVPR 2023 ([arXiv](https://arxiv.org/html/2304.00733)) | Uncertainty attenuation (GMM) for noisy labels / temporal fluctuations | — | — | — | No | Uncertainty about labels, not visibility or identity. |
| **FloCoDe** — Khandelwal, CVPRW 2024 ([CVF](https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/papers/Khandelwal_FloCoDe_Unbiased_Dynamic_Scene_Graph_Generation_with_Temporal_Consistency_and_CVPRW_2024_paper.pdf)); STKET, arXiv 2309.13237 | Flow-aware temporal consistency; relation transition priors | — | Transition priors (STKET) | Yes (STKET knowledge) | No | In-network temporal consistency; not visibility-aware. |
| **SceneGraphVLM** — arXiv 2605.13667 ([arXiv](https://arxiv.org/html/2605.13667)) | VLM-based dynamic SGG; claims suppression of spurious / hallucinated relations | — | — | — | — | Different paradigm; temporal-consistency claims only. |
| STTran, DSG-DETR, Action Genome, Neural Hawkes, Know-Evolve, DyRep | (see paper bib) | — | — | — | — | Unchanged from earlier list. |

**Searches that returned nothing directly matching:** "Hawkes process + video visual relation / scene graph"; "scene graph relationship termination with visible absence + probabilistic filtering"; "tracker uncertainty propagated into scene graph edges".
Absence of hits is weak evidence (limited search), not proof of novelty.

## 2. Classification of the project's claims

| Claim | Classification |
|---|---|
| Hidden alive/dead state modulating Hawkes detections, forward filtering, MLE | **Established technique** (Markov-modulated Hawkes; HMM filtering) |
| Visibility-dependent detection probability | **Established idea** (detection covariates in occupancy models; thinning) |
| Exact censoring when ρ = 0 (Prop. 1); evidence = −(λ−ε)Δ (Prop. 2) | **Direct consequences** of the model; correct but not new theory |
| Applying this to VidSGG edge persistence, with per-edge births, regime-dependent trace decay, frozen persistence/termination metrics | **New combination / new application** — no VidSGG paper found that models *why* an edge disappeared with a calibrated alive probability |
| Cross-predicate termination hazard γ | **New combination** (covariate-driven extinction is standard in occupancy models; predicate-to-predicate hazards in VidSGG not found). Empirically small effect (+0.003 F1). |
| Lazy maintenance via closed-form untouched evolution + wake-up heap | Event-driven scheduling with lazy deletion is **established**; the exactness conditions (L1, L2) for this filter are a **small new technical result** |
| Robustness to ID switches | **Not addressed** by V-HPED; measured failure (Stage 1 §7) |

## 3. Three strongest defensible research gaps

**G1 — Identity uncertainty is not separated from visibility in post-hoc relationship filtering.**
TCDSG treats identity consistency end-to-end; WSGG handles unobserved objects
but (per abstract) not tracker identity noise; PDAF weights evidence by
association probability for *object state*, not for relationship existence.
No work found that uses an inference-time identity-continuity signal to (a)
dilute relationship evidence and (b) govern births of relationships on pairs
whose identity just changed. Our own baseline fails exactly here (Stage 1 §7).
*Strength: concrete, measurable, motivated by an observed failure.*

**G2 — Calibrated relationship termination under partial observability in VidSGG.**
RTAformer targets relation transitions; WSGG targets persistence of unobserved
objects; neither (per abstracts) outputs a calibrated probability that a
relationship has ended or separates censored from negative evidence. V-HPED
does, with ECE 0.03 vs 0.29 for gated EMA (synthetic). *Strength: moderate —
the modelling is established, the application and evaluation are not.*

**G3 — Exactness conditions for event-driven maintenance of modulated-point-process filters.**
Lazy scheduling is standard; showing *when* a forward filter admits exact
elapsed-time closed forms (and when it provably does not) is a small, checkable
algorithmic result suitable for an ADSA contribution. *Strength: modest.*

## 4. Consequences for the paper
1. Cite Markov-modulated Hawkes processes and dynamic occupancy models explicitly and
   present V-HPED as an **instance** of that family adapted to VidSGG — not as a new
   point-process model.
2. Cite WSGG/ActionGenome4D and RTAformer as the closest VidSGG work.
3. Novelty should rest on G1 (if the extension works), G2 (application + evaluation),
   and G3 (algorithmic exactness), in that order.

## 5. Limits of this audit
* About a dozen searches; mostly abstract-level reading.
* Not searched: ACM MM / TPAMI / TIP 2024–2026 VidSGG robustness papers in depth;
  "survival analysis + visual relationships"; occupancy-style models in vision.
* Bibliographic details for new references must be checked before submission.
