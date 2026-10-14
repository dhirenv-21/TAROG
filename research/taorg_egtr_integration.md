# Phase 6 — Real scene-graph model integration: EGTR

> Date 2026-10-13/14; branch `codex/taorg-completion`.
>
> **Status: the integration is COMPLETE and verified against the source.** The **predictor-level
> evaluation stopped at its registered development gate** (`research/taorg_final_exit_report.md`).

## 1. Selection

| Candidate | Verified | Decision |
|---|---|---|
| **EGTR** (Im et al., CVPR 2024; `naver-ai/egtr`) | **Apache-2.0** (LICENSE and NOTICE); one-stage Deformable-DETR SGG; official VG and OI checkpoints; inference needs no compiled extension (PyTorch fallback for deformable attention) | **Chosen** (user approval 2026-10-13) |
| RelTR (`yrcong/RelTR`) | **No licence file** (404 for every variant), so all rights are reserved by default | Rejected |
| STTran / TEMPURA / DSG-DETR | Action Genome; compiled extensions on old torch | Rejected (scope) |
| PSG models | Heavy mmdet stack | Rejected (scope) |

No local predictions, checkpoints or scene-graph data existed before this phase.

## 2. Provenance and versions

| Item | Value |
|---|---|
| Source | `https://github.com/naver-ai/egtr` @ `7f87450f32758ed8583948847a8186f2ee8b21e3`, cloned to `D:/TAORG_runs/egtr/src` (not vendored) |
| Checkpoint | Official VG model `egtr_vg.tar.gz` (Google Drive; 456 MB; SHA-256 `39e9f9ee…d5d2fe`). Inner `epoch=03-validation_loss=1.71.ckpt`, SHA-256 `8926640f…1f84d5`, verified at load |
| Training data | Visual Genome (RelTR/EGTR split: 57,723 train / 5,000 val / 26,446 test images; 150 object, 50 predicate classes) |
| Prior exposure | VG had never been used in TAORG |
| Overlap risk | VG images partly come from YFCC100M, as do VidOR videos (different media; overlap not checked, not excluded) |
| Environment | Isolated venv `D:/TAORG_runs/egtr/venv` (`environments/egtr-venv-local.txt`): transformers 4.18.0, timm 0.5.4, huggingface-hub 0.10.1, pytorch-lightning 1.6.4 (evaluation-path imports only), pycocotools 2.0.11, setuptools 69.5.1; system torch 2.14.0+cu132 |
| Why a separate env | BoxMOT needs huggingface-hub ≥ 1.7; transformers 4.18 needs < 1.0 |
| Hardware | RTX 4060 Laptop: 0.28 s/frame at batch 1 (640×360–640×480 input resized to the 800 short side); 1.6 GB peak GPU |

## 3. Adapter (`src/taorg/adapters/egtr.py`)

The adapter uses EGTR's own model, configuration and feature extractor, and reproduces the
official single-prediction SGDet post-processing of `train_egtr.evaluate_batch`:

| Step | Implementation |
|---|---|
| object scores and classes | max softmax over 150 classes (no-object excluded) |
| pair score | outer product, diagonal 0 |
| relation scores | clamp(`pred_rel`) × clamp(`pred_connectivity`) |
| ranking | max over predicates of relation score × pair score |
| output | top 100 pairs; boxes cxcywh → xyxy in original pixels |

**Declared deviations** (none changes the computation):
1. The timm backbone is built with `pretrained=False`; every weight is then loaded from the
   checkpoint with `strict=True`, so nothing is downloaded and nothing is left uninitialised.
2. The feature extractor is constructed locally with the `SenseTime/deformable-detr` preprocessing
   parameters (size 800, max 1333, ImageNet normalisation, COCO detection format).
3. The deformable-attention CUDA kernel is not compiled. The adapter calls EGTR's own
   `ms_deform_attn_core_pytorch` directly, instead of reaching it through the source's
   per-call try/except, which prints a message.
4. The evaluator only (not the adapter) replaces the uncompiled Cython `bbox_overlaps` with a NumPy
   port using the same +1-pixel convention.

**Vocabularies** (`configs/egtr/vg_vocab.json`, exported from the official annotation file):
- class index c ↔ VG category id c + 1;
- predicate column p ↔ `rel_categories[p + 1]`.

**Confidence semantics.**
- Object scores are softmax probabilities.
- Relation scores are sigmoid outputs times a sigmoid connectivity.
- Their product is a **ranking score, not a calibrated probability**.

## 4. Verification against the source (`scripts/egtr/reproduce_vg_subset.py`; `results/egtr/reproduction.json`)

**Subset.** A seeded VG **test** subset of 200 images (seed 20261013), range-fetched from the
official image zips (manifest in `results/egtr/vg_subset_manifest.json`).

**Adapter vs official evaluation path** (same images, batch 1): **bit-identical**.
- 200/200 images have identical classes and identical ranked pairs.
- Maximum absolute difference: 0 for boxes, object scores and relation scores.

**Official metrics on the subset vs the authors' full-test results** (shipped with the
checkpoint):

| | R@20 | R@50 | R@100 | mR@50 | AP50 |
|---|---|---|---|---|---|
| Subset (200 images) | 21.6 | 28.1 | 32.7 | 7.2 | 35.7 |
| Authors (26,446 images) | 23.5 | 30.2 | 34.4 | 7.9 | 30.8 |

The differences are consistent with subset sampling variation. This is a **subset sanity check, not
a full reproduction**: the 9.7 + 5.5 GB image archives were not downloaded.

## 5. Video integration (`src/taorg/experiments/final/stream.py`; `scripts/egtr/run_video_sgg.py`)

**Frames.** EGTR runs on every 3rd frame (≈ 10 fps). The internal frame f is `frames/{f:06d}.jpg`,
the same JPEG frames the tracker used. Boxes are in original pixel coordinates.

**Persons.** 13 VG person classes, object score ≥ 0.1, top 30 per frame.

**Predicate mapping** (fixed before scoring):

| VidOR | VG |
|---|---|
| `behind` | `behind` |
| `in_front_of` | `in front of` |
| `next_to` | `near` |
| `watch` | max(`watching`, `looking at`) |

**Identity mapping.** EGTR person queries are assigned one-to-one (Hungarian, IoU ≥ 0.5) to the
tracker's output detection boxes, and keys use tracker IDs.

**Leakage audit.**
- `build_predicted_stream` takes no annotation input. An AST test forbids annotation helpers there.
- Annotations enter only through `scoring_labels` and `oracle_keys`, for scoring.

**Contract use.** The stream separates:
- **observed** (score ≥ τ);
- **evaluated negative** (both endpoints matched to EGTR persons; VISIBLE absence record);
- **not evaluated** (no record).

This is the distinction the observation audit (L57) found missing.

## 6. What happened on VidOR (development only; `results/final/dev/diagnosis.json`)

1. **Identity mapping works:** 76% of tracker outputs are matched to an EGTR person query (14.2
   EGTR person queries vs 3.6 tracker outputs per frame).
2. **The score scale is tiny.** Person–person triplet scores for the mapped predicates have a 99.9th
   percentile of 3e-5 to 9e-4. The registered τ grid (≥ 0.005) admitted almost nothing, and the
   development gate failed (`research/taorg_final_exit_report.md`).
3. **The signal is weak.** As a ranking score for "annotated relation active" among matched pairs,
   AUC is:
   - `in_front_of` 0.53;
   - `behind` 0.60;
   - `watch` 0.66;
   - `next_to` 0.69.
   At a 90th-percentile threshold, the raw stream's frame F1 is 0.15.
4. **The score scale defeats threshold-based temporal methods.** GatedEMA + hysteresis and V-HPED
   expect [0, 1] confidences; raw EGTR triplet scores never reach their ON thresholds. Threshold-free
   methods (NoMemory, FixedWindow) are unaffected.

**Interpretation (hypothesis).** An image-level VG model, applied frame by frame to user video,
gives weak person–person relation evidence for VidOR's mostly spatial and attention predicates.
That makes it a poor substrate for testing temporal maintenance.

## 7. Limitations

- One model, one checkpoint (VG); a subset-only metric check.
- Four mapped predicates.
- Image-level predictions with no temporal context.
- Runtime with the PyTorch fallback (0.28 s/frame) forced stride-3 processing.
