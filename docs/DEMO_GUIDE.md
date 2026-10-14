# Final demo guide (3–5 minutes)

This guide is for presenting the TAORG website to an examiner. Setup and
troubleshooting are in [WEB.md](WEB.md). What was actually tested is listed in
[WEB_VALIDATION.md](WEB_VALIDATION.md).

## 1. Before the demo (do this 15 minutes ahead)

**Prerequisites.**
- [SETUP.md](SETUP.md) is complete on this laptop. That means the main
  environment, the tracking venv (`D:\TAORG_runs\appearance\venv`), the EGTR
  venv (`D:\TAORG_runs\egtr\venv`), cached weights and an NVIDIA GPU.
- At least 5 GB of free space on D:.
- Optional: the sample clip at `D:\TAORG_runs\web-samples\MOT17-09_sample.mp4`.
  Create it with `python scripts/web/make_sample_clip.py`, using the MOT17
  frames already on D:. Nothing is downloaded.

**Pre-demo check.**
1. Close other GPU programs, then double-click `start_web.cmd`. The browser
   opens at http://127.0.0.1:8765. Keep the black server window open.
2. The **environment** card should read **Setup files found**, with every row
   ticked.
3. Press **Run pre-demo check**. It takes under a minute (7 s on the development laptop) and loads both
   inference environments on the GPU. Both rows must say *ready*. If either
   fails, read its detail line and fix it using WEB.md.
4. Do one full run of the sample clip, so you know how long it takes on this
   machine. Leave the result in **Recent runs**; it is your fallback.
5. Close the server window and start `start_web.cmd` again. Runs created before
   this launch are then labelled **Saved result from an earlier session**.

## 2. Demo sequence

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the header and the intro | "Everything runs locally on this laptop. It tracks the people in a short video and predicts four relationships between them: behind, in front of, next to, and watching. It is not general scene understanding." |
| 0:30 | Point at the five steps, then press **Analyse the sample clip**. Alternatively, choose a short video and press **Start analysis**. | "The sample is 10 seconds of a public pedestrian benchmark, MOT17-09, used non-commercially under its CC BY-NC-SA licence. It has no relationship ground truth, so what you will see are unverified predictions." |
| 1:00 | While it runs, point at the stage list and the elapsed time | "Each stage is a separate process. Tracking and EGTR run in their own pinned environments. The page shows which stage is running and how long it has taken. It does not guess a percentage or a finish time." |
| 1:30 | Open **Settings used for this run** | "These are the exact settings, the config hash and the Git commit, recorded for reproducibility." |
| 2:00 | When it completes, use the viewer: press Play, step with ← and →, and click the timeline | "Solid blue arrows are relationships EGTR predicted at this step. Dashed orange ones were not predicted now but are retained by the temporal memory. The table lists exactly the same edges as the saved `graphs.json`." |
| 3:00 | Hover over the **Score** header | "These are raw EGTR ranking scores, typically around 10⁻⁶. They are not probabilities, and we do not rescale them." |
| 3:30 | Show the download buttons and **Recent runs** | "Every run keeps its graph, its observations, its settings and its log. The offline viewer opens without the server." |
| 4:00 | Close with the limitations card | See section 4. |

If the live run is slow or fails, see section 5.

## 3. What the outputs mean

- **P7** is track ID 7 from the tracker. It is a label for one person in this
  video, not an identity.
- **Predicted at this step** means EGTR output this relationship on this frame,
  with a raw score at or above the cutoff (default `6.648e-7`). That default is
  a development diagnostic quantile, not a validated operating point.
- **Retained by temporal memory** means the edge was not predicted now. The
  method (Fixed buffer by default) still holds it, because it was predicted
  within the last *W* relationship steps.
- **Takeover flag** (dashed magenta box) means the frozen appearance check
  suspects the tracker swapped this ID to another person. With the option on,
  that person's retained relationships are cleared.
- Relationships are predicted on every third frame. A **step** is one of those
  frames.
- An **empty graph** is a real result. It is shown as it is.

## 4. How to explain it

**Algorithm.** TAORG maintains a temporal relationship graph over tracked
people. Per-frame predictions come in as observations. A maintenance policy
decides what to keep when a prediction is missing:
- Fixed buffer;
- GatedEMA + hysteresis;
- V-HPED, a Hawkes-process intensity filter.

Phase 5 implemented lazy maintenance that is exactly equal to the eager
reference.

**Implementation.**
- Detection and tracking: Faster R-CNN + BoxMOT.
- Appearance takeover flag: frozen.
- Relationships: EGTR, an image scene-graph model, run per frame.
- Then temporal maintenance, then the viewer.
- The website is a standard-library Python server that only listens on
  127.0.0.1. It runs one job at a time, as subprocesses.

**Results and limitations (say this plainly).**
- The registered final evaluation stopped at its development gate. Raw EGTR
  person–person scores (~10⁻⁶ to 10⁻³) sit far below the registered thresholds.
- There is no evidence that the temporal methods improve real predicted
  relationships. GatedEMA and V-HPED usually give empty graphs on raw scores,
  which is why they are labelled experimental.
- Narrower earlier findings are recorded in `Memory.md`. In the registered
  downstream study, which used oracle relationship streams on real tracks, the
  fixed buffer and GatedEMA with the frozen takeover flag reduced ghost
  relationship time. The appearance audit decision was NARROW. Neither result
  shows better real predicted relationships.
- EGTR was trained on still images, so its predictions flicker. Accuracy on the
  demo clip is unknown because there are no labels.

## 5. Fallback: saved results (label them as saved)

If the GPU, the environments or time fail during the demo, select the run
you made during the pre-demo check in **Recent runs**. The page labels it
**"Saved result from an earlier session"**, because it was created before the
current server started (step 1.5). Say so out loud: *"This is a saved run from before
the demo, not a fresh one."* Its settings, log and downloads remain available.

If the server itself will not start, open the downloaded `viewer.html` from a
completed run directly in the browser. It is self-contained. State that it is a
prerecorded result.

Never present a saved run as a live one.
