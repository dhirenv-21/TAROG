# Setup

TAORG runs on Python 3.10. The main research code needs one environment. The real-video pipeline
needs two more pinned **stage environments**, because BoxMOT (huggingface-hub ≥ 1.7) and EGTR
(transformers 4.18, hub < 1.0) cannot share one.

Tested configuration: Windows 11, RTX 4060 Laptop (8 GB), CUDA via torch 2.14.0+cu132.

## 1. Main environment (research code, tests, temporal methods, analysis, viewer)

```bash
pip install -e .
pip install torch==2.14.0 torchvision==0.29.0 numpy==2.2.6 scipy==1.15.3 scikit-learn==1.7.2 pandas==2.3.3 matplotlib==3.10.8 PyYAML==6.0.3 opencv-python trackers==2.6.1 supervision==0.30.9
python -m pytest -q
```

The two tests that need BoxMOT are skipped in this environment.

## 2. Perception environment (detector + OSNet + BoT-SORT)

```bash
python -m venv --system-site-packages D:/TAORG_runs/appearance/venv
D:/TAORG_runs/appearance/venv/Scripts/python -m pip install --no-deps -r environments/perception-venv-local.txt
```

- **BoxMOT** 25.0.0 is AGPL-3.0. It is used as an unmodified library and is not vendored.
- **ReID weights:** `osnet_x1_0_msmt17.pt` (torchreid model zoo; SHA-256 `b7d73dc6…1cf2`). Download
  it to `D:/TAORG_runs/appearance/weights/`:

  ```bash
  python -m gdown "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M" -O D:/TAORG_runs/appearance/weights/osnet_x1_0_msmt17.pt
  ```

- **Detector:** the torchvision `fasterrcnn_resnet50_fpn_v2` COCO_V1 weights (SHA-256 `dd69338a…`)
  download automatically into `TORCH_HOME` (set it to a D: path).
- **Frozen takeover flag:** `results/appearance/freeze/frozen_models.pkl` (in git; checksum-verified
  at load).

## 3. EGTR environment (scene-graph model)

```bash
git clone https://github.com/naver-ai/egtr.git D:/TAORG_runs/egtr/src
git -C D:/TAORG_runs/egtr/src checkout 7f87450f32758ed8583948847a8186f2ee8b21e3
python -m venv --system-site-packages D:/TAORG_runs/egtr/venv
D:/TAORG_runs/egtr/venv/Scripts/python -m pip install --no-deps -r environments/egtr-venv-local.txt
D:/TAORG_runs/egtr/venv/Scripts/python -m pip install setuptools==69.5.1
```

- **Checkpoint:** the official EGTR VG checkpoint (Apache-2.0 code; trained on Visual Genome). It is
  `egtr_vg.tar.gz`, 456 MB, SHA-256 `39e9f9ee…`:

  ```bash
  python -m gdown "https://drive.google.com/uc?id=18phcRxbrEI7HqIuM2OLAPuwAF5k3pUC2" -O D:/TAORG_runs/egtr/egtr_vg.tar.gz
  ```

  Extract it into `D:/TAORG_runs/egtr/ckpt/`. The `.ckpt` inside has SHA-256 `8926640f…`, which the
  adapter verifies.
- **No compilation is needed.** The deformable-attention CUDA kernel is not built, so EGTR's own
  PyTorch fallback is used, at about 0.28 s/frame on an RTX 4060.
- **Paths:** set `TAORG_EGTR_SRC` / `TAORG_EGTR_ARTIFACT` if you use other locations.

## 4. Data used by the experiments (not redistributed)

| Data | Location | Source |
|---|---|---|
| MOT17 labels and frames | `D:/TAORG_runs/observation`, `D:/TAORG_runs/appearance/MOT17Det` | `scripts/appearance/fetch_mot17_frames.py` (range fetch; CC BY-NC-SA 3.0) |
| VidOR annotations | `data/vidor/` | HF mirror `shangxd/vidor` |
| VidOR videos | `D:/TAORG_runs/downstream/videos` | `scripts/downstream/fetch_vidor.py --split {training,validation,final}` |
| VG annotations and the 200-image subset | `D:/TAORG_runs/egtr/vg*` | `scripts/egtr/fetch_vg_subset.py` |
