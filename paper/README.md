# Paper: An Area-Preserving Hénon-Map Model for the Riemann Zeros

LaTeX source and built PDF of the manuscript (Frontiers in Physics, ID 1833442)
and the point-by-point response to reviewers.

## Files

| File | Description |
| :--- | :--- |
| `manuscript.tex` | Main manuscript (Frontiers Harvard class), revised version |
| `manuscript.pdf` | Compiled PDF (17 pages) |
| `references.bib` | Bibliography |
| `rebuttal.md` | Point-by-point response to Reviewers 1, 5, 6 + cover note |
| `figures/` | Figures 1–8 (`image2.png`–`image10.png`) |
| `FrontiersinHarvard.cls`, `Frontiers-Harvard.bst` | Frontiers template class & bib style |
| `logo*.{eps,pdf}`, `YM-logo.*` | Header logos required by the class |

The figures are produced by the analysis scripts in the repository root
(`1-…` through `9b-…`). Figures 8–9 (spectral diagnostics and robustness) come
from `8-spectral_diagnostics.py` and `9b-robustness_reoptimized.py`.

## Build

```bash
pdflatex manuscript
bibtex   manuscript
pdflatex manuscript
pdflatex manuscript
```

Uses `pdflatex`; the `*-eps-converted-to.pdf` logo files are included so no
`--shell-escape` / `epstopdf` step is needed.
