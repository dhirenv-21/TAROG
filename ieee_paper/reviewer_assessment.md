# Reviewer-Style Assessment (not part of the paper)

*Written as a critical reviewer for a computer-vision / multimedia venue. It is not a
prediction of acceptance.*

## Summary
The paper proposes V-HPED, a post-hoc filter for video scene-graph edges. It combines
a visibility-thinned Hawkes detection model, a latent alive/dead chain and an exact forward
likelihood. It also proves conditions for exact lazy maintenance with a wake-up heap.
Evaluation is purely synthetic.

## Scores (1 = poor, 5 = excellent)
| Criterion | Score | Comment |
|---|---|---|
| Originality | 2.5 | The statistical core is a Markov-modulated Hawkes process with thinning, an established family that the paper correctly acknowledges. Novelty lies in the VidSGG formulation, the censored-vs-negative-evidence framing, and the lazy-exactness conditions. |
| Technical correctness | 4 | Equations match the code, and key identities are tested (NumPy = PyTorch to 1e-9, HMM nesting, eager = lazy). The left-Riemann discretisation is stated. Prop. 5 is argued and checked numerically, not proved formally. |
| Mathematical rigour | 3 | Lemma 1 and Theorem 1 are clean but simple. The complexity bound for H = O(A) is argued, not tested. |
| Experimental validity | 2 | Synthetic only. The generator shares the method's assumptions (visibility labels are given, detections are independent given state), which favours visibility-aware methods. There is no real detector, tracker or dataset. |
| Baseline fairness | 3 | The baselines are hand-set at defaults, while V-HPED variants are fitted. Gated EMA is not tuned on validation beyond its threshold, which may understate it. |
| Reproducibility | 4 | Configs, seeds, manifests and run IDs are saved, and one cell was reproduced bit-exactly. There is no git commit, and AG access is unverified. |
| Clarity | 4 | Claims are scoped and negative results are reported. |
| Related work | 3 | The recent work is mostly cited from abstracts. There is no empirical comparison with any VidSGG network. |
| Strength of evidence | 2 | Effects are statistically clear but small (+0.008 F1), synthetic, and mixed (VT-HMM is more accurate; ID switches break the fitted model). |

## Strongest contribution
The **exact lazy-maintenance result**: an elapsed-time semigroup under two conditions,
with numerically checked necessity, a simple wake-up heap whose decisions matched the
reference in every test, and an honest cost model (faster for sparse touch patterns,
slower when every pair is touched). It is self-contained and verifiable.

## Biggest weakness
**No real-data evidence.** Every accuracy claim comes from a generator built on the
model's own assumptions. Combined with the fact that the most accurate synthetic variant is
the visibility-only HMM, the case for the Hawkes component in practice is unproven.

## Likely reviewer objections
1. "Why should a reader believe this transfers to Action Genome or real trackers?" There are no real experiments.
2. "Is the Hawkes part needed?" VT-HMM has the higher frame F1 (+0.011).
3. "Baselines are weak or untuned." Gated EMA decays are hand-set. Tune them on validation with the same budget as V-HPED.
4. "The identity-switch failure undermines the robustness framing." The title avoids claiming ID robustness, but the paper still reports a failed extension and an unevaluated one.
5. "The synthetic generator encodes visibility perfectly." Only one visibility-error condition (10% flips) was tested, and only in the IA-v1 run.
6. "Fitted parameters are not identifiable", so interpretability claims are limited.
7. "Calibration comparison is unfair to EMA", since EMA scores are not probabilities. The paper notes this.
8. The latency results come from Python on one machine, and memory is not measured.

## Specific improvements, ordered by value
1. Run the **Action Genome pilot**, which is already implemented, and report Recall@K plus the persistence/termination metrics with one base model's predictions.
2. **Tune all baselines** (gated EMA decays, fixed window, HMM parameters) on validation with a comparable search budget.
3. Evaluate **IA-v2** on synthetic data, then on real tracker outputs with scores defined from association quality, or drop the identity section.
4. Add a **visibility-error sweep** (5–30% flips) and a misspecified generator (correlated detections, non-exponential dwell times).
5. Measure **memory** and run the adversarial alternating workload. Consider a compiled implementation so the latency claims are not Python-bound.
6. Turn Prop. 5 into a formal proof, and test the heap-size bound.
7. Search the literature more systematically (ACM MM, TPAMI, TIP 2023–2026) for occlusion- or termination-aware temporal SGG. Read the full texts of WSGG and RTAformer.
8. Report effect sizes relative to between-seed variance, and correct for multiple comparisons across the ablation table.

## Fit for venue
In its current form it is better suited to a workshop, or an algorithms/data-structures
venue (for the lazy-maintenance part), than a main computer-vision track. A main-track
submission would need item 1 at minimum.
