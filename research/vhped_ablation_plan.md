# V-HPED ablation plan (Phase 4) — registered with the pre-registration

Every rung must earn its place. The ladder is evaluated with the same calibration seeds, equal
random-search budget (200 samples per rung), condition-matched tuning and held-out evaluation as
H2 (`research/phase4_preregistration.md` §5).

| Step | Comparison | Mechanism tested | Target conditions (where it should help) | Must not hurt | Retention rule |
|---|---|---|---|---|---|
| A0 | M1 vs M0 | probabilistic flag likelihoods vs deterministic gating | all failure conditions | N0–N2 | reported (baseline question: does probabilistic use of visibility alone explain gains?) |
| A1 | M2 vs M1 | continuous time + latent visibility persistence | SH7, SH8 (frame rate); UNK runs | N0–N2 | descriptive; formulation §8 predicts ≈ 0 at 30 fps |
| A2 | M3 vs M2 | flags as noisy sensors (r_ov, r_vo) | F3, F4, F7, N1 | N0, N2 | ΔHV ≥ 0.02, CI lower > 0 in ≥ 1/3 of target cells; no N0–N2 loss |
| A3 | M4 vs M3 | occlusion-onset termination coupling η | F1, F7, F8 | N0–N2 | same rule (a-priori expectation: fails, formulation §6) |
| A4 | M5 vs M4 | identity-tempered likelihoods | F5, F8 (identity frontier) | N0–N2 (pays for jitter false alarms) | same rule |
| A5 | M6 vs M5 | cross-predicate exclusivity evidence | T1, T2 | T3, F2 | = H3 decision |

**Interpretation rules (fixed in advance).**

- If a rung fails its retention rule, the final method is the highest rung whose every added
  mechanism was retained (for example, if A3 fails, M4's η is dropped). Because each rung is
  searched independently, a failed rung is *not* re-tuned or redesigned in Phase 4.
- Random-search noise: rungs are sampled independently, so M(k) vs M(k−1) differences include
  search variance. A negative ΔHV for a strict superset model is therefore evidence of search
  difficulty (curse of dimensionality) and will be reported as such, not hidden.
- The H2 treatment is fixed to M5 **regardless** of the ablation outcome, so that the primary
  hypothesis cannot be steered by choosing the best rung after the fact. The ablation informs the
  *final method definition* for later phases, not the H2 decision.
- Tuning-capacity control: M5 re-selected from a seeded random half of its samples
  (pre-registration §8).
