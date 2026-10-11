# TAORG relationship episodes: related work and novelty check

> Written **before** any episode-study code or experiment (2026-10-11). It verifies the closest
> primary literature for the mechanisms used in `research/taorg_episode_formulation.md`.
> Bibliographic details were checked against publisher, arXiv or DOI records found by web search
> on 2026-10-11 (sources listed at the end). Where only secondary sources were reachable, this is
> stated. Nothing here is a result of this project.

## 1. The question and why prior phases motivate it

Phase 4 (`research/phase4_exit_report.md` §20) found that the exact joint (Z, V) filter (V-HPED
M3+) **relapses**: after the first post-termination drop it outputs a terminated relation again
(98–100% of terminations in F1). The frozen first-drop termination-delay metric cannot see this.
It also found that the only robust gain was suppressing identity-switch "ghosts", and that this gain
came from a one-line identity-tempered likelihood. Phase 4 Reviewer B asked for "termination
semantics (absorbing state, hysteresis, or a smoothed decision)". Reviewer C asked for "an
equal-dimension control (e.g. a GatedEMA extended with an identity term)".

This study asks: **can explicit relationship episodes (an episode can end only once; any recurrence
is a new episode) prevent false reactivation and identity ghosts without harming persistence or the
detection of genuine recurrence?** Every mechanism involved has a long history, listed below.

## 2. Closest literature by mechanism

### 2.1 Existence probability with track initiation and termination (closest overall)

- **IPDA**, Mušicki, Evans and Stanković, *IEEE TAC* 39(6):1237–1241, 1994,
  doi:10.1109/9.293185. A recursively updated **probability of track existence** drives track
  initiation (confirmation), maintenance and deletion (false-track discrimination). Our
  within-episode existence filter (formulation §3) is a single-target, measurement-origin-certain
  special case of that idea: the "measurement" is a relation detection, not a position.
- **JIPDA**, Mušicki and Evans, *IEEE TAES* 40(3):1093–1099, 2004,
  doi:10.1109/TAES.2004.1337482. This is the multi-target version, with joint association events
  and per-track existence probabilities.
- **Bernoulli filters**, Ristic, Vo, Vo and Farina, "A tutorial on Bernoulli filters: theory,
  implementation and applications", *IEEE TSP* 61(13):3406–3430, 2013,
  doi:10.1109/TSP.2013.2257765. A Bernoulli random finite set has an existence probability and a
  state density. Its prediction combines a birth probability and a survival probability. Our
  episode estimator differs in one respect only: **within** an episode the birth probability is
  zero (termination is absorbing), and births are handled by a separate sequential test
  (§2.3 below). The Bernoulli filter instead lets births re-enter the same existence variable.
- **Labelled RFS / GLMB**, Vo and Vo, *IEEE TSP* 61(13):3460–3475, 2013,
  doi:10.1109/TSP.2013.2259822. Labelled random finite sets give each object trajectory a unique
  label, so a reborn object gets a **new label** rather than resuming an old one. "Episode" in this
  study is the same distinction applied to relationship instances: the label is (key, episode
  index). We claim no novelty for that.

### 2.2 Practical track management (M-of-N confirmation, coast-then-delete)

- **SORT**, Bewley, Ge, Ott, Ramos and Upcroft, ICIP 2016, doi:10.1109/ICIP.2016.7533003,
  arXiv:1602.00763. The reference implementation (github.com/abewley/sort) confirms a track only
  after `min_hits` associated detections and deletes it after `max_age` frames without one. These
  are deterministic birth-confirmation and termination rules. (Parameter defaults differ between
  forks; seen in code, not in the paper text.)
- **ByteTrack**, Zhang et al., ECCV 2022. Lost tracks are kept for a buffer period (`track_buffer`,
  30 frames in common implementations) before deletion. We checked this through implementation
  documentation, not the paper's text.

These are the engineering ancestors of "confirm births, delay deletions". A GatedEMA with an
on/off threshold pair (our baseline B2) is the relationship-level analogue.

### 2.3 Sequential change detection for births

- **CUSUM**, Page, "Continuous inspection schemes", *Biometrika* 41(1–2):100–115, 1954,
  doi:10.1093/biomet/41.1-2.100. A cumulative log-likelihood-ratio statistic, reset at zero, raises
  an alarm when it crosses a threshold. It followed Wald's sequential probability ratio test (SPRT).
  Our birth rule (formulation §4) is a textbook CUSUM on per-frame detection evidence. The
  refractory term (a threshold that is raised right after a termination and decays back) is a
  time-varying alarm threshold. We do not know of it as a named method, and it is a small
  design choice, not a contribution.
- **Bayesian online changepoint detection**, Adams and MacKay, arXiv:0710.3742, 2007. This method
  maintains a posterior over the **run length** (time since the last change) with a hazard
  function. It is the closest Bayesian alternative to our "CUSUM + absorbing existence filter"
  pair, and it is also a duration-aware (semi-Markov) model.

### 2.4 Duration-aware (semi-Markov) models

- **HSMM review**, S.-Z. Yu, "Hidden semi-Markov models", *Artificial Intelligence*
  174(2):215–243, 2010, doi:10.1016/j.artint.2009.11.011. An HSMM gives each state an explicit,
  non-geometric dwell-time distribution. Our baseline B4 is an explicit-duration HSMM forward filter
  with Weibull dwell times for ON and OFF and the same flag-conditioned emissions as the VA-HMM. It
  is the standard way to make "time since the relationship started or ended" matter. If the
  episode estimator's gains came only from duration modelling, B4 should match it.

### 2.5 Hysteresis decisions

- **Canny**, "A computational approach to edge detection", *IEEE TPAMI* 8(6):679–698, 1986,
  doi:10.1109/TPAMI.1986.4767851. Canny popularised hysteresis thresholding (a high threshold to
  start, a low threshold to continue) as a way to suppress streaking and fragmentation. (The
  mechanism description comes from secondary sources; the full text was not re-read.) Two-threshold
  hysteresis on a smoothed score is our baseline B2.

### 2.6 Uncertain identity association

- **MHT**, Reid, "An algorithm for tracking multiple targets", *IEEE TAC* 24(6):843–854, 1979,
  doi:10.1109/TAC.1979.1102177. This defines hypotheses that a measurement came from a known
  target, a new target, or clutter.
- **JPDA**, Fortmann, Bar-Shalom and Scheffe, "Sonar tracking of multiple targets using joint
  probabilistic data association", *IEEE J. Oceanic Eng.* 8(3):173–184, 1983,
  doi:10.1109/JOE.1983.1145560.
- Our identity-reliability term is far simpler than either. It is a per-track posterior that a
  constant-velocity prediction residual came from an identity switch, and it **tempers** the evidence
  of relations touching that track (likelihood mixture w·L + (1 − w)). It does not re-associate
  identities. It is the Phase 4 M5 mixture likelihood (L24) with a velocity-predicted residual in
  place of a static displacement. Re-association is out of scope.

### 2.7 Relationship detection in video (application domain)

- **VidVRD**, Shang, Ren, Guo, Zhang and Chua, "Video visual relation detection", ACM MM 2017,
  doi:10.1145/3123266.3123380. A relation instance is a triplet plus subject and object
  trajectories over a temporal span, i.e. an **episode**. Segment-level predictions are associated
  into instances. Later work associates relation segments with multiple-hypothesis methods ("Video
  relation detection via multiple hypothesis association", ACM MM 2020; title and venue only, not
  read).
- **STTran**, Cong et al., ICCV 2021, arXiv:2107.12309, and **TEMPURA**, Nag et al., CVPR 2023,
  arXiv:2304.00733. Dynamic scene-graph generators get temporal consistency from learned temporal
  context. Neither post-processes relation outputs with explicit episode semantics; both are
  learned models evaluated with frame-level recall, not episode-level metrics.

## 3. What would and would not be new

| Element | Prior art | Status in this study |
|---|---|---|
| Existence probability, deleted below a threshold | IPDA/JIPDA, Bernoulli filter | **not new** |
| Absorbing termination; rebirth as a new labelled instance | labelled RFS, MHT track labels, SORT/ByteTrack track lifecycle | **not new** |
| CUSUM birth confirmation | Page 1954; M-of-N confirmation | **not new** |
| Raised threshold right after termination (refractory) | time-varying alarm thresholds; OFF-duration modelling in HSMMs | a minor design choice, **not claimed** |
| Visibility-conditioned absence evidence | Phases 3–4 of this project (GatedEMA, VA-HMM); missing-data likelihoods generally | **not new** |
| Identity-reliability tempering | Phase 4 M5 (L24); robust mixture likelihoods | **not new** |
| Transition context (successor relation raises the termination hazard) | Phase 4 M6 exclusivity (L25); HSMM with covariates | **not new** |
| Event-history context (exponential kernel on recent onsets) | Hawkes/point-process GLMs (Phase 5H, L30–L31) | **not new** |
| Combining these for **relationship episodes** in VidSGG post-processing, with relapse-aware episode metrics | No direct precedent found in this non-exhaustive search (VidVRD association is the closest) | **at most an application/evaluation contribution**, and only if the experiment supports it |

**Combining standard mechanisms is not a method novelty claim.** If the study succeeds, the honest
framing is "relationship-level track management with relapse-aware evaluation". The candidate
contribution would be the evaluation protocol (episode metrics that defeat always-on and always-off
output) and evidence about which simple mechanism matters. If GatedEMA with hysteresis (B2) or the
HSMM (B4) matches the episode estimator, the recommendation is to use the simpler method.

## 4. Gaps in this check (stated, not hidden)

- Not exhaustive. No systematic search of VidSGG post-processing or temporal-smoothing papers was
  done beyond the items above.
- The IPDA, Canny, ByteTrack and labelled-RFS details come from abstracts, records or
  implementations, not full-text reading.
- No check of the action-detection literature (temporal action localisation also produces
  segments and uses NMS / hysteresis-like post-processing).

## Sources (retrieved 2026-10-11)

- Yu 2010 HSMM: https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf ; https://en.wikipedia.org/wiki/Hidden_semi-Markov_model
- IPDA: https://034.medsci.ox.ac.uk/research/publications/2281301 ; https://findanexpert.unimelb.edu.au/scholarlywork/1085864-integrated-probabilistic-data-association-finite-resolution
- JIPDA: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5750805/ ; https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc05/pdffiles/papers/2092.pdf
- CUSUM / Page 1954: https://en.wikipedia.org/wiki/CUSUM ; https://www.scinapse.io/papers/2051903196
- Bernoulli filters: https://espace.curtin.edu.au/handle/20.500.11937/6652?show=full ; https://arxiv.org/pdf/2308.08104
- BOCPD: https://arxiv.org/pdf/0710.3742
- SORT: https://arxiv.org/abs/1602.00763v2 ; https://api.openalex.org/works/doi:10.1109%2FICIP.2016.7533003 ; implementation copies e.g. https://huggingface.co/spaces/ECCV2022/bytetrack/blob/refs%2Fpr%2F1/yolox/sort_tracker/sort.py
- ByteTrack buffer: https://supervision.roboflow.com/latest/trackers/ ; https://inference.roboflow.com/workflows/blocks/byte_track_tracker/
- Canny 1986: https://www.deepdyve.com/lp/doc/pGaQyaDXpb ; https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm
- Labelled RFS: https://ieeexplore.ieee.org/document/6507656 ; https://espace.curtin.edu.au/handle/20.500.11937/24008
- MHT / JPDA: https://ieeexplore.ieee.org/document/1102177 ; https://api.crossref.org/works/10.3390%2FS18103193 ; https://arxiv.org/pdf/2308.06326
- VidVRD: https://www-users.cse.umn.edu/~guo00109/assets/publication/mm17-shangxd.pdf ; https://api.openalex.org/works/doi:10.1145%2F3123266.3123380
- STTran: https://arxiv.org/pdf/2107.12309 ; TEMPURA: https://openaccess.thecvf.com/content/CVPR2023/papers/Nag_Unbiased_Scene_Graph_Generation_in_Videos_CVPR_2023_paper.pdf
