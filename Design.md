# TAORG / V-HPED — Design Guide

## 0. Scope
TAORG is primarily a research/code project, not a frontend product. This file defines the visual language for **paper figures, plots, architecture diagrams, README visuals, and any optional minimal demo**. Do not build a dashboard unless explicitly requested after the research pipeline is working.

## 1. Visual direction
Style: clean academic / technical / conference-paper ready.

Principles:
- high information density without clutter
- readable when printed or viewed in a two-column paper
- color is supportive, never the only encoding
- no neon/cyberpunk aesthetic
- no excessive gradients, glassmorphism, or decorative animation
- figures should survive grayscale conversion where possible

## 2. Semantic visual language
Use consistent semantics across figures:
- observed/confirmed relationship: solid line / strong marker
- occluded or censored interval: dashed or hatched representation
- visible negative evidence / termination: distinct marker or line style
- predicted persistence: continuous temporal segment
- expiry/drop event: clear `x` or vertical marker

Do not rely solely on red/green distinctions; combine color with line style/shape for accessibility.

## 3. Suggested palette roles
Exact colors can be chosen at figure-generation time using an accessible palette, but roles must stay stable:
- primary method / V-HPED: one consistent accent
- simple baselines: neutral/secondary hues
- occlusion/censored state: muted cool hue or hatch
- termination/negative evidence: warm hue
- ground truth: dark neutral

For paper plots, use a colorblind-safe palette and distinct markers/dashes.

## 4. Typography
### Paper/figures
Prefer fonts that embed reliably and remain legible at small sizes. Match the eventual conference template where practical. For matplotlib, use a standard sans-serif unless the paper template requires otherwise.

### Optional web/demo
- UI: Inter or system sans-serif
- monospace/code: JetBrains Mono or a system monospace

Do not make font downloads a project dependency.

## 5. Plot standards
Every plot must have:
- descriptive axis labels with units
- legend labels using paper method names
- readable font size at final paper dimensions
- uncertainty shown when relevant
- fixed ordering of methods across plots
- export to vector format (PDF/SVG) when possible plus PNG preview

Avoid:
- 3D charts
- unnecessary smoothing
- truncated axes that visually exaggerate differences without clear justification
- plotting only the best seed

## 6. Key paper figure design
### Persistence vs termination trade-off
This is the central figure.
- x-axis: false persistence / termination cost (exact metric frozen before use)
- y-axis: Edge Persistence F1 or persistence recall/F1
- curves/points across thresholds
- separate panels should be avoided if one clear plot can show occlusion lengths; otherwise produce separate figures rather than unreadable overlays
- highlight V-HPED consistently but retain readable baselines

### Occlusion-length robustness
- x-axis: injected gap length in frames
- y-axis: persistence metric
- mean across fixed seeds; uncertainty bars/bands

### Efficiency
- x-axis: active edge count A or density
- y-axis: temporal update latency
- eager vs lazy/heap
- also report operation counters so latency is not the only evidence

## 7. Architecture diagram
Use a left-to-right or top-to-bottom flow:

`Video/Dataset -> Detector/GT -> Tracker -> Base VidSGG -> Canonical Edges + Visibility -> V-HPED -> Robust Graph -> Metrics`

Inside V-HPED show three conceptual blocks:
- visibility-aware dynamics
- cross-predicate interactions
- lazy expiry heap

Do not imply that V-HPED is itself the detector/tracker/base scene-graph model.

## 8. Qualitative sequence figure
For a selected clip, show a short row of frames or schematic states:
- before occlusion
- during occlusion
- after reappearance
- true termination comparison

Overlay only essential relation labels. The point is to make “occluded or ended?” visually obvious.

## 9. Optional minimal demo
Only if later requested:
- single video pane
- current detected/tracked objects
- raw base-model edges vs V-HPED output
- small temporal state/visibility indicator
- no authentication, social features, dashboard analytics, or unrelated UI

The demo is for qualitative inspection, not the primary scientific result.

## 10. Naming conventions in visuals
Use consistently:
- **TAORG** for the overall framework
- **V-HPED** for the proposed temporal edge model
- `Vis-Gated EMA` for visibility-gated EMA in compact legends
- `No Memory`, `Fixed Window`, `EMA`, `HMM` for baselines

Never rename methods between plots because it complicates paper review and result auditing.
