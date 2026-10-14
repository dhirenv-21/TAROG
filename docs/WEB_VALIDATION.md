# Local website validation — 2026-10-14

This is engineering validation of the website wrapper. It is not a scientific
evaluation. No research decision, frozen model, threshold, metric or historical
artifact was changed. Hardware: the development Windows 11 laptop with an NVIDIA
GPU. Browser automation used headless Microsoft Edge through Playwright.

## 1. Real pipeline runs through the website

The sample clip is MOT17-09, frames 1–300, at 960×540 and 30 fps (MPEG-4
Part 2, built by `scripts/web/make_sample_clip.py` from frames already on D:).
Its input SHA-256 is `4a08ece1…5d148b15`. The licence is CC BY-NC-SA 3.0. The
clip is not committed. MOT17 has **no relationship labels**, so the outputs
below are counts, not accuracy.

The final recorded run is `cea775220bc749f2accac0d3b1edb95d`, in the QA work
folder `D:/TAORG_runs/web-qa-work`.
- Settings: Fixed buffer + frozen takeover flag, cutoff `6.648e-7`, buffer 10
  steps, relationships every 3rd frame.
- Config SHA-256: `c5716975…7e882`.
- Git base: `04c3282`, with the uncommitted website changes of this pass
  (`git_dirty_src: true` is recorded honestly).

| Stage | Seconds |
|---|---|
| Check video | < 1 |
| Track people (perception env) | 65.6 |
| Predict relationships (EGTR env) | 40.4 |
| Maintain graph | 0.9 |
| Build viewer | 0.6 |
| **Total wall time seen by the browser** | **107** |

The output had 100 relationship steps and 71 track IDs. There were 1,162 edge
records: 490 predicted at their step and 672 retained by the fixed buffer.
92 steps had at least one edge. All 9 completed runs of this clip during QA
gave identical counts (1,162 edges, 490 predicted). Eight took 92–108 s from
the start of tracking to the end of the viewer stage. One took 321 s: it ran
while the laptop briefly suspended, which also broke one QA attempt with
`ERR_NETWORK_IO_SUSPENDED`. That QA attempt is preserved in the defect history
below; the script now retries transient polling errors.

## 2. Browser QA: `scripts/web/browser_qa.cjs`

The final run against the final code passed **30 of 30 checks**. The report is
at `D:/TAORG_runs/web-qa/qa_report.json`, with screenshots in the same folder.
The script drives the real pipeline and verified the following.

**Keyboard and focus**
- Tab order runs: skip link, brand, file input, sample button, method, flag,
  advanced settings, pre-demo check, run actions.
- Every stop shows a visible focus ring, and focus never leaves the page.
- Focus on **Cancel run** survives the page update when the stage changes.

**Input errors**
- An empty file and a `.txt` file are rejected with clear messages before upload.
- A corrupt `.mp4` fails at *Check video*, with no GPU stage started, and shows
  recovery steps.

**Running state**
- *Start* is disabled while a run is active.
- A second start is refused (HTTP 409).
- The stage list showed perception, EGTR and later stages as they ran.

**Completed state**
- The run passed through all four stages.
- **Viewer alignment:** at each of the 100 steps, the table's relationship set
  equals that step's `graphs.json` edges, and the displayed video frame number
  matches.
- Every saved predicted score is at or above the cutoff (minimum `6.651e-7`),
  so none is rounded to 0.

**Viewer controls**
- ←/→ step, End jumps, Play advances and pauses, the slider seeks, and clicking
  the timeline seeks to the clicked step.

**Exports**
- All six downloads (offline viewer, graph JSON, predictions CSV, settings
  YAML, run record, log) return 200 as attachments.
- The downloaded `graphs.json` is byte-identical to the one the viewer shows.
- `config.yaml` and `job.json` record the method, flag, cutoff and buffer
  actually used.
- The input video is **not** downloadable (404, allowlist).

**Cancellation**
- A run cancelled during tracking ends as *cancelled*.
- Python processes for that run went from 2 before the cancel to 0 after.

**History**
- Failed, completed and cancelled runs are listed.
- Selecting an older completed run reopens its viewer.

**Layout and console**
- At a 390 px width there is no horizontal page overflow.
- There were no page errors or console errors during the whole session.

`scripts/web/shots.cjs` found 0 px horizontal overflow at 1440×900, 1280×720
and 390×844, with no console errors. Screenshots of all states were inspected
by eye. They led to these fixes:
- the timeline is capped at 340 px and scrolls;
- the viewer is stacked below 1,600 px, because the 1440 px side-by-side table
  wrapped person names;
- the table is tightened at 390 px so it fits without clipping;
- an unusable file no longer reads "ready to start".

Screenshots contain dataset imagery and stay on D:. They are not committed.

## 3. Other live checks

- **Pre-demo check (button):** tracking env showed boxmot 25.0.0, torch
  2.14.0+cu132, CUDA True (2.4 s). Relationship env showed transformers 4.18.0,
  CUDA True (4.5 s). Total 7 s, with no console errors.
- **Server crash.** A run was started, and the server process was killed during
  tracking.
  - Two pipeline Python processes survived as orphans.
  - On restart, the run was marked *interrupted* and both orphans were stopped
    (0 left).
  - The error text says the leftover worker was stopped.
  - The test was repeated after the port change, with the same result.
- **Port conflict.** Before the fix, a second server on the same port **started
  silently**, because Windows `SO_REUSEADDR` allowed it. After the fix, it exits
  with "Port 8770 is already in use".
- **Sample clip reproducibility.** Re-running `make_sample_clip.py` after its
  edits reproduced the same SHA-256.

## 4. Defects found and fixed in this pass

1. **Scores exported as 0.** `graphs.json` and `stream.csv.gz` rounded scores to
   4 decimals, so raw EGTR scores (~10⁻⁶) were saved as `0.0` and the viewer
   showed "0".
   - Exports now keep 4 significant figures (`export_score`, with a unit test).
   - The temporal methods always received unrounded scores. Maintenance results
     and metrics are unaffected.
   - For older runs, the viewer detects predicted edges below the cutoff and
     shows "< 0.0001 (rounded in this older run)" instead of 0.
2. **Orphaned GPU workers after a server crash.** These are now stopped on
   restart, but only if the process's command line names `taorg.pipeline` and
   that run's folder.
3. **Two servers could share one port on Windows.** The server now binds
   exclusively.
4. **Focus loss.** Refreshes rebuilt the run's action buttons while focused.
   Focus is now restored.
5. **QA-script problems found along the way:**
   - the script read the previous job instead of the new one;
   - its own PowerShell query counted as a "leftover" process;
   - deliberate 404/409 probes were logged as console errors;
   - Tab ran off the page before the history loaded.

   All four were fixed in the script, not hidden. The failing attempts' logs
   were overwritten by later runs; their outcomes are recorded here.
6. **Earlier in this pass** (summarised in Memory.md):
   - the iframe viewer was replaced by an accessible in-page viewer;
   - early video validation was added;
   - the self-check and sample clip were added;
   - settings are shown for each run;
   - contrast, typography and responsive layout were improved.

## 5. Automated tests

| Check | Result |
|---|---|
| Full main-environment suite | **403 passed, 4 skipped** (86 s) |
| Website tests (`tests/unit/test_web.py`) | 16 passed |
| Pipeline tests (`tests/unit/test_pipeline.py`) | 4 passed |
| Ruff on the website, its tests and the sample script | passed |
| `node --check` on `app.js`, `browser_qa.cjs`, `shots.cjs` | passed |
| `git diff --check` | passed |

The 4 skips are environment-specific: two need the BoxMOT environment and two
need the EGTR environment. Both environments were exercised by the real runs
above, but their own test suites were not rerun.

The website tests cover:
- settings validation; hashes and config snapshots;
- the real-video probe accepting and rejecting files;
- the probe failure stopping before any stage;
- single-worker enforcement; interrupted history;
- leftover-worker cleanup, including an unrelated process being left alone;
- stage order, interpreters, stage failure and summaries;
- Host, Origin and token checks; upload limits; artifact allowlisting;
- frame paths and path traversal;
- the sample allowlist; self-check refusal while busy;
- exclusive port binding; cancellation of a real child process.

`tests/unit/test_pipeline.py` already failed ruff line-length rules before this
pass. Only the added test was kept within the rules.

## 6. Not tested or not claimed

- Accuracy of any relationship. There are no labels for the sample clip, and
  the registered evaluation stopped at its development gate.
- Browsers other than Microsoft Edge (Chromium). Firefox and Safari were not
  tested.
- A screen reader. Only keyboard access, labels, ARIA attributes and focus
  visibility were checked.
- Contrast by tool. Text colours were darkened and inspected by eye, but no
  automated contrast audit was run.
- Uploads near the 2 GB limit, and videos near the 300 s limit. Only the limit
  logic is unit-tested.
- Codecs other than MPEG-4 Part 2, beyond the corrupt-file case. The
  "cannot preview this format" message was seen for the corrupt upload. Preview
  of real H.264 or WebM files was not checked.
- Machines other than the development laptop, and Ctrl+C shutdown. Graceful
  shutdown uses the existing `close()` path; only a hard kill was tested live.
- GatedEMA and V-HPED were not run end to end in this pass. They remain
  experimental and usually produce empty graphs on raw scores.

To reproduce, run a QA server on a separate work folder, then the script.
Playwright must be available to Node (set `PLAYWRIGHT_MODULE` if needed), and
Edge must be installed. The script adds failed, completed and cancelled runs to
that folder's history.

```powershell
python -m taorg.web --port 8770 --work D:\TAORG_runs\web-qa-work
node scripts/web/browser_qa.cjs http://127.0.0.1:8770 D:/TAORG_runs/web-qa
```

## 7. Visual redesign check — 2026-10-15

The UI-only pass changed `index.html`, `style.css` and `app.js` rendering. No
backend, API, saved-run format, pipeline, threshold or score change was made.

**Screenshots.** `scripts/web/ui_states.cjs` captured seven states (empty,
upload, running, failed, cancelled, interrupted, completed) at 1440×900,
1280×720 and 390×844, before and after the redesign. That is 21 captures each,
saved in `D:/TAORG_runs/web-qa/ui`. "Empty" and "running" are **mocked** by
rewriting `/api/jobs` in the browser, and their file names say so. The other
states are real saved runs in the QA workspace.

Results:
- 0 px horizontal overflow and no console errors in all captures.
- At 390 px the relationship table fits its container: 330 of 330 px.

Defects seen in the screenshots and fixed:
- the stage state text inherited the table's line marker;
- the log download link was unstyled;
- the narrow table clipped its score column;
- the disabled primary button had low contrast (1.97:1, now 7.3:1);
- the recovery text said "on the left", which is wrong on narrow screens.

**Contrast.** All text/background colour pairs were computed against WCAG AA
(≥ 4.5:1). The lowest is 5.36:1, for muted text on the selected history row.
This was computed from the stylesheet tokens, not checked with an automated
page audit.

**End-to-end.** The unchanged `scripts/web/browser_qa.cjs` passed **30/30**
against the redesigned page. It ran one real sample run
(`b1cc6d9aecf94dc5a6686038f4394aa1`: 100 steps, 1,162 edge records, the same
as before; 121 s including browser overhead) and one cancelled run. It
covered:
- keyboard order and visible focus on 9 stops, including the new header status
  link;
- focus surviving a refresh;
- invalid, empty and corrupt inputs; one-job-at-a-time;
- table and frame matching `graphs.json` at all 100 steps;
- Play/Pause, arrow keys, End, slider and timeline seeking;
- the six exports and the allowlist;
- process cleanup after cancel;
- history reopening; 390 px layout; no console errors.

**Suite.** 403 passed, 4 skipped. `node --check` and `git diff --check` pass.

**Not checked:** screen readers, browsers other than Edge, and
Windows high-contrast mode.
