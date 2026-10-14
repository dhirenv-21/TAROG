# Local video website

The website runs the TAORG pipeline on a short video and shows the result in an
interactive viewer. Everything runs locally on this computer: there is no cloud,
no account and no upload to a hosting service. For a presentation, see
[DEMO_GUIDE.md](DEMO_GUIDE.md). For what was tested, see
[WEB_VALIDATION.md](WEB_VALIDATION.md).

## Start

Double-click `start_web.cmd` in the checkout, or run:

```powershell
python -m taorg.web --open
```

The page opens at http://127.0.0.1:8765. Keep the server window open while you
use it.

The page follows three numbered steps.

1. **Configure a run.** Choose a video, drop one onto the page, or press
   **Analyse sample**. Keep the recommended settings (Fixed buffer with the
   takeover flag), or open **Advanced settings**. Press **Start analysis**.
2. **Monitor the run.** Each stage is shown with an icon and a text state
   (done, running now, failed, not started) and the time it took. The page does
   not estimate percentages or finish times. **Cancel run**, the processing log
   and the settings used are in the same card. Saved runs are listed under
   **Recent runs**, with name, start time, method, result and status. The newest
   12 are shown; **Show all** lists the rest. Runs created before the current
   server started are labelled *Saved result from an earlier session*.
3. **Inspect results.** For a completed run, explore the viewer: Play, step
   with ← and →, use Home and End, drag the slider, or click the timeline. The
   **Downloads** row in the same card holds every export.

### What it does and does not do

The website tracks **people** and predicts four person–person relationships:
**behind, in front of, next to, watching**. It does not recognise other objects
and does not answer questions about the video. Predictions come from EGTR, an
image model applied to every third frame independently. They can be wrong.
Empty graphs are shown as they are.

The research study found **no evidence** that the temporal methods improve real
predicted relationships. Fixed buffer is the default. GatedEMA + hysteresis and
V-HPED are labelled experimental: they expect scores in [0, 1], but raw EGTR
scores are about 10⁻⁶, so they usually produce empty graphs. Changing settings
makes an exploratory run, not a registered evaluation. Frozen models, thresholds
and historical results are unchanged.

### Reading the viewer

- **Predicted at this step** (solid blue): EGTR produced the relationship on this
  frame with a raw score at or above the cutoff.
- **Retained by temporal memory** (dashed orange): not predicted now, but the
  method still holds it.
- **Takeover flag** (dashed magenta box): the frozen appearance check suspects
  the tracker gave this person's ID to someone else.
- **P7** is the tracker's ID 7 for one person in this video.
- **Score** is the raw EGTR ranking score, or the method's own score for retained
  edges. It is **not a probability** and is never rescaled.
- The table lists exactly the edges in the saved `graphs.json` for that step.
- Runs made before 2026-10-14 exported scores rounded to 4 decimals, so tiny raw
  scores were saved as 0. The viewer shows those as
  "< 0.0001 (rounded in this older run)". Newer runs keep 4 significant figures.

### Downloads

| Button | File | Contents |
|---|---|---|
| Open offline viewer / Offline viewer | `viewer.html` | Self-contained viewer that opens without the server |
| Graph JSON | `graphs.json` | Per-step tracks, takeover flags and edges (`observed: true` = predicted now) |
| Predictions CSV | `stream.csv.gz` | Raw thresholded EGTR observations before temporal maintenance |
| Settings YAML | `config.yaml` | Exact pipeline configuration of the run |
| Run record | `job.json` | Status, stage times, settings, input and config SHA-256, Git commit |
| Log | `run.log` | Output of every stage; failed, cancelled and interrupted runs offer it as **Download processing log** |

Only these named files can be downloaded. The input video and other folder
contents cannot.

### Interface design

- **Theme.** A light research-dashboard style: navy text, one teal accent for
  actions, and neutral cards with thin borders. Blue (solid) and amber (dashed)
  are kept only for predicted and retained relationships, as Design.md
  specifies.
- **Typography and icons.** System fonts only (Inter if it is installed,
  otherwise Segoe UI or the system sans-serif) and small inline SVG icons. There
  are no remote fonts, CDN files or frameworks, so the page works offline.
- **State is never colour alone.** Status badges, stages and relationship rows
  always carry an icon or line style plus text. All text/background pairs meet
  WCAG AA contrast (≥ 4.5:1; the lowest computed is 5.36:1).
- **Accessibility.** Keyboard focus is always visible. The viewer accepts
  ←, →, Home, End and Space. Reduced-motion preferences are respected, and the
  page uses no animation.

## Requirements

- Complete [SETUP.md](SETUP.md) first. Tracking and EGTR run in their own pinned
  environments (`D:\TAORG_runs\appearance\venv` and `D:\TAORG_runs\egtr\venv`);
  the server itself uses the main environment and needs no extra packages.
- An NVIDIA GPU with CUDA visible to both environments.
- At least **5 GB free** on the work drive. The environment card warns below this.
  Extracted frames and the offline viewer can be many times larger than the
  video.
- Videos of at most **300 s** (`--max-seconds`) and 2 GB (`--max-upload-mb`).
  Start with clips under a minute.
- Typical time on the development laptop for the 10 s, 960×540 sample: about
  **95 s**, of which tracking takes about 55 s and EGTR about 37 s. Time grows
  roughly with the number of frames.

### Environment card and pre-demo check

The card ticks each required file: both interpreters, the appearance weights,
the takeover model, the EGTR source and checkpoint, the cached detector, and
free disk space. **Run pre-demo check** goes further. It imports torch, BoxMOT
and OpenCV in the tracking environment, and torch, transformers and timm in the
EGTR environment, and confirms that each sees CUDA. It takes under a minute and
downloads nothing. It is refused while a run is active.

## Sample clip (optional)

```powershell
python scripts/web/make_sample_clip.py
```

The script builds `D:\TAORG_runs\web-samples\MOT17-09_sample.mp4`:
- the first 300 frames (10 s) of MOT17-09, resized to 960×540, MPEG-4 Part 2;
- made from the MOT17 frames already on D:, with no download;
- with a `.json` file next to it that records the source, licence, attribution
  and SHA-256.

MOT17 is licensed **CC BY-NC-SA 3.0**: non-commercial, with attribution. The
clip and its frames stay on D: and are **not committed**. MOT17 has no
relationship labels, so predictions on the clip are unverified.

Only clips listed in the samples folder can be started (`--samples` selects the
folder). A client can never supply a path.

## Behaviour and safety

- **One run at a time.** A second start is refused with a message.
- **Video check.** Before any GPU stage, OpenCV must open the file, decode a
  frame, and read a sensible frame rate, at least 3 frames and a duration within
  the limit. Otherwise the run fails at *Check video* with re-export advice.
  Empty files and unsupported extensions are rejected before upload.
- **Browser preview.** Preview depends on the browser's codecs. If the browser
  cannot preview a file, the page says so, and analysis still works.
- **Cancel run** stops the stage's whole process tree and keeps partial outputs.
- **Closing the server.** Ctrl+C in the server window stops the website and the
  current run.
- **Interrupted runs.** If the server is killed or crashes, the next start
  marks the run *interrupted*. It also stops the run's leftover pipeline worker,
  but only a process whose command line runs `taorg.pipeline` on that run's
  folder. Interrupted runs are not resumed.
- **One server per folder.** A lock on the work folder prevents a second server
  using it.
- **Local only.** The server listens only on `127.0.0.1`. It checks Host, Origin
  and `Sec-Fetch-Site`, and changes require a token issued for each launch.
  Do not expose it through a tunnel or reverse proxy.
- **Run folders.** Each run is kept in `D:\TAORG_runs\web\<run-id>` outside Git.
  The folder holds the input video and its SHA-256, a config snapshot and its
  SHA-256, the Git commit (with a flag if `src`/`configs` had uncommitted
  changes), stage outputs, stage times and logs.

## Options

```powershell
python -m taorg.web --port 8766 --work D:\TAORG_runs\web --max-upload-mb 512 --max-seconds 120 --samples D:\TAORG_runs\web-samples --open
```

- `--config` selects a pipeline YAML file.
- `TAORG_PY_PERCEPTION` and `TAORG_PY_EGTR` override its interpreter paths.
- `--torch-home` selects the detector cache (default
  `D:\TAORG_runs\downstream\torch_home`). That cache must contain
  `hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`.

## Troubleshooting

| Symptom | What to do |
|---|---|
| "Setup needs attention" | The crossed-out item names the missing file; follow SETUP.md. Nothing is downloaded automatically. |
| Pre-demo check says CUDA False or fails to import | Re-create that environment per SETUP.md; check `nvidia-smi`. |
| "Another TAORG server is using this folder" | Close the other server window, or use a different `--work`. |
| Port already in use | Start with `--port 8766`. |
| Fails at *Check video* | Re-export as MP4 (H.264), trim to under the duration limit. |
| Fails at *Track people* / *Predict relationships* | Open **Processing log**; usually a missing weight, GPU memory, or another GPU program. Close other GPU programs and retry. |
| "This video cannot be previewed" | Browser codec only; analysis still works. |
| Run shows *interrupted* | The server stopped mid-run; start a new run. |
| Empty graph | A real result: no relationship scored above the cutoff. Try Fixed buffer, or lower the cutoff under Advanced (exploratory). |
| Page shows "Cannot reach the local server" | The server window was closed; start it again. History is kept. |

Run the browser QA (real pipeline, about 4 minutes; it adds failed, completed
and cancelled runs to the history, so point it at a QA work folder):

```powershell
python -m taorg.web --port 8770 --work D:\TAORG_runs\web-qa-work
node scripts/web/browser_qa.cjs http://127.0.0.1:8770 D:/TAORG_runs/web-qa
```
