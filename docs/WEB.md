# Local video website

Double-click `start_web.cmd` in the checkout, or run:

```powershell
python -m taorg.web --open
```

Open http://127.0.0.1:8765. Choose a video, select a method, and press
**Start analysis**. The page shows stage progress, processing logs, saved run
history, and the existing interactive graph viewer. Download the viewer to open
it offline, or export graph JSON, observations, and settings.

The server uses the main Python environment. Complete [SETUP.md](SETUP.md) first:
tracking and EGTR still run in their separate pinned environments. No new web
dependencies are required. The environment panel checks local files; it does not
verify that every package imports or download missing resources.

## What it runs

The website wraps the existing pipeline: person detection and tracking,
appearance takeover evidence, EGTR relationship prediction, temporal maintenance,
and the viewer. It currently supports person–person **behind, in front of,
next to, and watch** relationships. It is not a general video question-answering
system. Predictions can be wrong, and temporal quality gains are not established.

Fixed buffer is the default. GatedEMA and V-HPED remain experimental with raw
EGTR scores and may produce empty graphs. The takeover flag is available for
Fixed Buffer and GatedEMA. Changing UI settings makes an exploratory run, not a
registered research evaluation. The frozen models and historical results are
unchanged.

Start with a short clip. The 2 GB limit is an upload limit, not a promise that
long videos fit RAM or disk. Extracted frames and the embedded offline viewer
can be much larger than the original video. One run is allowed at a time;
processing can take several minutes. Browser video preview depends on codec
support; pipeline decoding uses OpenCV.

## Storage and stopping

Runs default to `D:\TAORG_runs\web\<run-id>` outside Git. Each includes the input
video, input checksum, configuration snapshot/checksum, Git commit, stage outputs,
metadata, and logs. No video is sent to a hosting service. Jobs continue if the
browser closes. **Cancel run** stops the pipeline's child process tree and keeps
partial outputs. **Ctrl+C** in the server terminal stops the website and its
current processing job. Keep that terminal open while using the page.

History survives restarts. Interrupted jobs are marked interrupted; they are
not automatically resumed. After a forced crash, check for surviving Python
workers before launching another GPU run. A work-folder lock prevents two
website servers using the same folder. Do not run another inference process
alongside this one on a memory-constrained laptop.

The server listens only on `127.0.0.1` and checks Host, Origin, and a per-launch
request token. It is for a single trusted local user; do not expose it through a
public tunnel or reverse proxy. Downloads are restricted to named run artifacts.

## Options

```powershell
python -m taorg.web --port 8766 --work D:\TAORG_runs\web --max-upload-mb 512 --open
```

`--config` selects a pipeline YAML file. `TAORG_PY_PERCEPTION` and
`TAORG_PY_EGTR` override its interpreter paths. `--torch-home` selects the
existing detector cache (default `D:\TAORG_runs\downstream\torch_home`).
The required cached detector filename is
`hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`.

Failed runs remain in history with their logs. Correct missing resources using
the setup guide. Re-upload after correcting a failure; partial results are kept
for inspection. Source code is pushed to Git, while local videos, weights, and
generated run files stay on D:.
