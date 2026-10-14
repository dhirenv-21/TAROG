# Local website validation — 2026-10-14

This is engineering validation, not a new scientific evaluation. No research
decisions, thresholds, models, or old artifacts were changed.

## Real pipeline check

A 12-frame MP4 was built from the first 12 already-local pipeline smoke frames,
at 10 fps. No dataset or model was downloaded. The website's HTTP upload endpoint
ran perception, EGTR, maintenance, and viewer generation successfully with
Fixed Buffer + frozen flag, cutoff `6.648e-7`, buffer 10. Output: four relationship
steps, 48 edge records. These are output counts, not accuracy measurements.

- Run: `D:/TAORG_runs/web/fb791c1412e24b3a869e7d076398ca62`
- Input SHA-256: `2530a18dcda98d9cad759d441ccff7c0a1719cf64d982164ea995dd0425b05e3`
- Config SHA-256: `c5716975de61af81b89cd26a654006649005edcf9b7e68ff0dff205e9217e882`
- Pipeline base: `4131448a7288a740754b9b2aac8ef5fc7938e7f3` with the uncommitted
  website implementation in this change; the metadata's Git SHA is the base,
  not a claim that the website was already committed.

## Browser check

`scripts/web/browser_smoke.cjs` ran in headless Microsoft Edge using the installed
Playwright runtime. It checked file selection, preview element, method/flag
compatibility, desktop and 390-pixel mobile layout, no horizontal page overflow,
the completed embedded viewer canvas, actual form submission of an invalid
video, failed-run messaging, logs, and recovery of the Start button. No page
JavaScript errors were reported. Screenshots were inspected locally at
`D:/TAORG_runs/web-qa`; they contain existing dataset imagery and are not committed.
The malformed-video run is intentionally retained in local history.

The video preview depends on browser codecs; this generated MPEG-4 clip can be
processed by OpenCV even when the browser cannot preview it. A preview error
does not prevent analysis. The graph viewer uses decoded JPEG frames.

## Automated tests

Final full main-environment suite: **395 passed, 4 skipped** (83.93 seconds).
The skips are the two BoxMOT-only appearance tests and two EGTR-environment tests.
Both perception environments were exercised by the real website smoke above;
their separate full test suites were not rerun for this wrapper change. Ruff
passed for the website Python and tests; Node syntax checks passed for both JS
files; `git diff --check` passed.

Nine backend tests cover settings validation, hashes and configuration snapshots,
single-worker enforcement, interrupted history, incomplete uploads, stage order
and interpreter selection, stage failure, host/origin/token checks, upload limits,
artifact allowlisting, metadata-write failure recovery, and cancellation of a real
child process. All nine passed.
The initial sandboxed attempt failed because it could not open localhost sockets
or replace temporary files; the unchanged tests passed with local execution access.

Run the optional browser check with a running server and a completed run named
`website-smoke.mp4` in its history:

```powershell
node scripts/web/browser_smoke.cjs D:/TAORG_runs/web-smoke.mp4 D:/TAORG_runs/web-qa
```

Playwright must be available to Node (or set `PLAYWRIGHT_MODULE` to its module
path), and Microsoft Edge must be installed. This test deliberately adds a failed
run to history; it is not part of the ordinary Python suite.
