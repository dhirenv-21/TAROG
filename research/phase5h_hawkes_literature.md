# Phase 5H — Hawkes feasibility: literature verification

> Status: literature check performed 2026-09-30 with web search. **Verification level is stated
> per entry.** "Abstract-verified" means authors, venue and main idea were confirmed from the
> abstract or proceedings page, not from a full-text read. Nothing here is a novelty claim. A
> journal submission requires a full-text read of every entry marked *(abstract)*, plus a broader
> Google Scholar / Semantic Scholar sweep. The web search is not systematic enough to prove
> absence.

## 1. Search log

| Query theme | Result |
|---|---|
| Hawkes + scene graph / dynamic scene graph / visual relationship | **no paper found** applying a Hawkes/TPP intensity model to visual-relationship events in video |
| Hawkes + video understanding / action recognition / anticipation | TPP for action sequences (APP-VAE, Time Perception Machine); no Hawkes over relationships |
| Hawkes + human-object interaction | none found |
| Hawkes / TPP + temporal knowledge graphs | Know-Evolve, GHNN, GHT and later work (text/news event KGs, not vision) |
| Scene-graph forecasting in video | SceneSayer (NeuralODE/SDE), FDSG (neural SDE); no point process |
| Noisy / partially observed / censored / missing-event Hawkes | substantial statistics and ML literature (§3) |
| Inhibitory Hawkes | established nonlinear and exponential-inhibition theory (§4) |
| Marked Hawkes, weighted events, label noise | ETAS tradition; robust deep Hawkes under label noise (§5) |
| Point processes + multi-object tracking | a spatio-temporal TPP used to filter noisy detections (§6) |
| Datasets with relation temporal extents | VidOR, ImageNet-VidVRD, PVSG (dense); Action Genome (sparse) (§7) |

## 2. Graph / relational point processes (closest in structure)

| Work | Problem | Event | Intensity / kernel | Learning | Missing events | Graph | Visual? | Overlap with TAORG | Gap left |
|---|---|---|---|---|---|---|---|---|---|
| **Know-Evolve** — Trivedi, Dai, Wang, Song, ICML 2017 *(abstract)* | reasoning on dynamic KGs | occurrence of a fact (s, r, o, t) | multivariate point process, intensity modulated by a score from learned entity embeddings | MLE, deep embeddings | not addressed (per abstract) | yes | no | fact = relationship event; intensity-based | not vision; no visibility/censoring; fully observed timestamps |
| **GHNN** — Han, Ma, Wang, Günnemann, Tresp, AKBC 2020 *(abstract)* | forecasting on temporal KGs | fact occurrence at future time | neural Hawkes-style intensity over graph event history (generalises the neural Hawkes process to evolving graphs) | neural, MLE | not addressed (per abstract) | yes | no | closest formal ancestor of "multivariate relationship-event Hawkes" | not vision; events assumed fully observed; no identity uncertainty |
| **GHT** — Sun, Geng, Zhong, Hu, He, EMNLP 2022 *(abstract)* | TKG entity + time prediction | fact occurrence | Transformer encoders + relational continuous-time encoding for Hawkes-process feature evolution | neural | not addressed (per abstract) | yes | no | as GHNN | as GHNN |
| Later graph-Hawkes TKG work, e.g. GAttNHP (arXiv 2607.14733), Hawkes scale-free TKG (arXiv 2501.01974) *(title/abstract only)* | TKG extrapolation | facts | neural Hawkes variants | neural | not assessed | yes | no | same family | same |
| **Relational event models** — Butts 2008 and successors (Annual Review of Statistics; relational hyperevent models, JRSS-A) *(abstract/secondary)* | social interaction sequences | dyadic (sender, receiver, type) events | conditional intensity as a function of past history and covariates (often log-linear) | partial likelihood | usually fully observed | dyads | no | a dyadic relational event = (subject, object, predicate) onset; statistics-side ancestor | not vision; no visibility; no identity uncertainty |

## 3. Missing, censored, partially observed and noisy Hawkes

| Work | Missingness model | Method | Overlap | Remaining gap for TAORG |
|---|---|---|---|---|
| **Shelton et al., "Hawkes Process Inference with Missing Data"**, AAAI 2018 *(abstract)* | unobserved events in a multivariate Hawkes process | MCMC with auxiliary variables (parent structure, "virtual events") sampling the posterior over unobserved events | directly covers "event not observed ≠ event did not occur" | observation process not tied to a *visibility* signal |
| **Mei, Qin, Eisner, "Imputing Missing Events in Continuous-Time Event Streams"**, ICML 2019 *(abstract)* | a fraction of events unobserved, several missingness mechanisms | particle smoothing with a bidirectional continuous-time LSTM proposal; complete data modelled by a neural Hawkes process | directly covers imputation under known missingness | visibility-dependent missingness from a tracker is a specific instance of their missingness mechanisms |
| **Xu, Luo, Zha, "Learning Hawkes Processes from Short Doubly-Censored Event Sequences"**, ICML 2017 *(abstract)* | observation window censored at both ends | sampling-stitching data synthesis | window censoring | intermittent (not only boundary) censoring |
| Partially interval-censored MHP (PCMHP) and discretely observed MHP (particle MCMC, arXiv 2503.18351; neural estimation, arXiv 2506.01258) *(secondary summaries)* | only counts per interval in some dimensions | EM / particle MCMC | coarse, frame-sampled annotation | — |
| Partially observed interacting Hawkes systems (arXiv 1902.07062) *(title)* | subset of components observed | statistical inference | unobserved components | — |
| **Trouleau et al., "Learning Hawkes Processes Under Synchronization Noise"**, ICML 2019 *(abstract)*; **"Cumulants of Hawkes Processes are Robust to Observation Noise"**, ICML 2021 *(title/abstract)* | random unknown time shifts per dimension | robust MLE; cumulant-based structure learning | detection latency / timestamp jitter | — |
| **ABC learning of Hawkes with missing or noisy event times** (arXiv 2006.09015) *(summary)* | thinning (random missing), censoring (unobserved regions), superposed ghost points | approximate Bayesian computation | **thinning = missed detections, censoring = occlusion windows, ghost points = false positives / ID-switch ghosts**, all named explicitly | — |
| **Bonnet et al., "Spectral analysis for the inference of noisy Hawkes processes"**, Scand. J. Stat. 2025 *(abstract)* | Hawkes superposed with an independent Poisson noise process | spectral / second-order | false-positive detections | — |

**Conclusion §3:** "Hawkes inference when events are missed, censored or spuriously added" is
**established theory**, including the exact three corruption types TAORG studies (thinning,
censoring, ghost superposition). Visibility-aware censoring in video would be an
*instantiation*, not a new theory.

## 4. Inhibition (mutually exclusive relations)

| Work | Formulation | Relevance |
|---|---|---|
| **Bonnet, Martinez Herrera, Sangnier, "MLE for Hawkes processes with self-excitation or inhibition"**, Stat. Probab. Lett. 2021 *(abstract)* | univariate exponential Hawkes, intensity (·)⁺ (positive part) allowing negative α; exact MLE | valid inhibition |
| **Bonnet, Martinez Herrera, Sangnier, "Inference of multivariate exponential Hawkes processes with inhibition…"**, Statistics and Computing 2023 *(abstract)* | multivariate exponential kernel with positive-part link | valid multivariate inhibition + estimation |
| Nonlinear Hawkes (Brémaud & Massoulié 1996, via secondary sources); variational nonlinear Hawkes with GP self-effects (PMC 8947061); interacting Hawkes with multiplicative inhibition (Stoch. Proc. Appl. 2022) *(titles/secondary)* | λ = φ(μ + Σ h), φ ≥ 0 | general framework |
| **Mei & Eisner, "The Neural Hawkes Process"**, NeurIPS 2017 *(abstract)* | continuous-time LSTM, softplus intensities; events can excite **or inhibit** | neural inhibition |

**Conclusion §4:** negative α must go through a link function, e.g. λ_r = (μ_r + Σ …)⁺ or
softplus. This is established. Inhibition is **not** a contribution.

## 5. Marks, weights, noisy labels

| Work | Content | Relevance to "identity-weighted history" |
|---|---|---|
| **ETAS** (Ogata 1988) and marked-Hawkes literature (e.g. Bayesian nonparametric marked Hawkes, arXiv 2511.22538; "Hawkes Models and Their Applications", Annual Review of Statistics 2024) *(secondary)* | excitation α·ρ(m_j)·g(Δt) with a mark-dependent productivity ρ(m) | **λ = μ + Σ w_j α g(Δt) with w_j = identity confidence is exactly a marked Hawkes process with mark-dependent impact ρ(m_j) = w_j.** Not novel. |
| **Robust Deep Hawkes Process under Label Noise of Both Event and Occurrence** (arXiv 2407.17164, 2024) *(abstract)* | noise-robust training for noisy event types/times | noisy relationship labels |
| **Unmixing Noise from Hawkes Process…** (arXiv 2406.16938) *(abstract)* | marked multivariate Hawkes + variational inference to separate structured events from noise | ghost/false events |

## 6. Vision uses of point processes

| Work | Event | Model | Overlap / gap |
|---|---|---|---|
| **APP-VAE** — Mehrasa, Jyothi, Durand, He, Sigal, Mori, CVPR 2019 *(abstract)* | times and categories of **actions** (MultiTHUMOS, Breakfast) | VAE point process (not Hawkes) | action-level, not object-pair relationships; fully observed labels |
| **Time Perception Machine** (arXiv 1808.04063) *(title)* | when/where/what of activity | TPP | activity level |
| **Spatio-Temporal Point Process for MOT** — Wang et al., IEEE TNNLS 2023 *(abstract)* | problematic **detections** | conv-RNN intensity (not Hawkes) used to filter bad detections before association | point process over perception noise; not relationships |
| **SceneSayer** (Peddi et al., ECCV 2024) *(abstract)*; **FDSG** (Yang et al., arXiv 2506.01487) *(abstract)* | future scene graphs on Action Genome | NeuralODE / NeuralSDE latent dynamics; no intensity model; no occlusion/identity handling (per abstracts) | **the natural baselines for any relationship-forecasting Hawkes claim** |

## 7. Datasets: is there event timing to model?

| Dataset | Relation timing | Suitability for point processes |
|---|---|---|
| **Action Genome** (Ji, Krishna, Fei-Fei, Niebles, CVPR 2020) — 25 relation classes (attention/spatial/contact) | 5 frames uniformly sampled per action; **≈ 35 annotated frames per video** | **poor**: onsets/offsets are only known to within the sampling gap; the GT event stream is interval-censored by design |
| **VidOR** (10k videos, 50 predicates, 80 object classes) | every relation has **start and end frame** | good: explicit onset/offset events |
| **ImageNet-VidVRD** (1k videos, 132 predicates) | relation instances with begin/end frames | good but small |
| **PVSG** (CVPR 2023, 400 videos, 57 predicates, avg 76.5 s) | temporally consistent relation trajectories on ≈ 150k densely labelled frames | good; long videos, egocentric + third-person |

## 8. Answers to the required questions

- **Is there verified prior work applying a genuine (multivariate) Hawkes intensity to visual
  relationship events in video?** None found in this search. The absence is *not proven*; a full
  Scholar/Semantic Scholar sweep is required before any "first" wording, and "first" wording is
  forbidden by the project rules anyway.
- **Is the mathematics new?** No. Multivariate exponential Hawkes, inhibition via link functions,
  missing/censored/thinned/noisy observations, and marked (weighted) excitation are all
  established (§3–§5).
- **What remains open?** An *application and evaluation* gap: whether relationship events from a
  VidSGG pipeline exhibit measurable cross-relation history dependence beyond per-edge exponential
  filtering, and whether censoring-aware inference driven by *tracker visibility* recovers that
  dependence under occlusion and identity noise. The gap is empirical, not theoretical.
