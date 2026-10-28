# IEEE conference paper — TAORG / V-HPED

**Title:** *Occluded or Ended? Visibility-Thinned Hawkes Filtering with Exact Lazy
Maintenance for Video Scene-Graph Relationships*

All quantitative results in this paper are **synthetic**. No real-video experiment has
been run. Author names, affiliations and the acknowledgment are red placeholders.

## Contents

| Path | What it is |
|---|---|
| `main.tex` | Full paper (`\documentclass[conference]{IEEEtran}`) |
| `references.bib` | Bibliography (Crossref/arXiv-checked; notes flag remaining checks) |
| `tables/*.tex` | Result tables, generated from saved runs |
| `figures/*.pdf` / `*.png` | Figures, generated from saved runs |
| `make_artifacts.py` | Regenerates `tables/`, `figures/` and `supplementary/provenance.json` from `../outputs/runs` (reads files only; runs no experiment) |
| `supplementary/provenance.json` | Run IDs behind every table and figure |
| `supplementary/experimental_checklist.md` | Experiments not yet done |
| `supplementary/reproducibility.md` | How to reproduce everything |
| `reviewer_assessment.md` | Critical, reviewer-style assessment (not part of the paper) |

## Compile

Needs a TeX distribution with `IEEEtran`, `algorithm`/`algpseudocode` (algorithmicx),
`tikz`, `booktabs` (TeX Live or MiKTeX; or upload the folder to Overleaf).

```bash
pdflatex main
```
```bash
bibtex main
```
```bash
pdflatex main
```
```bash
pdflatex main
```

**Compilation status:** not verified. No LaTeX compiler was available on the machine where
the paper was written. A static check passed: every `\cite` key is in `references.bib`
(and every entry is cited), every `\ref` has a `\label`, every included figure and table
file exists, and braces, math delimiters and environments balance. Compile once and fix
anything that remains before submission.

## Regenerate tables and figures

From this folder, with the project's Python environment:
```bash
python make_artifacts.py
```
It reads only the saved run directories listed in `supplementary/provenance.json`.

## Before submission

1. Replace the author/affiliation placeholders and the acknowledgment.
2. Re-check the bibliography notes, especially the TCDSG last-author spelling and the
   year of the Wang et al. entry (published online 2010, journal issue 2012).
3. Adapt to the target conference's page limit and template options.
4. Read `reviewer_assessment.md`. The paper's main weakness is the absence of any
   real-data result.
