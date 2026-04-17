# Report Build Instructions

## Overleaf (recommended)

1. Create a new blank project on Overleaf.
2. Upload the entire `report/` folder.
3. Set the main document to `main.tex`.
4. Compile with pdfLaTeX (default).

## Local compilation

Requires `latexmk` and a TeX distribution (TeX Live, MiKTeX).

```bash
cd report
latexmk -pdf main.tex
```

The output is `main.pdf`.

## Figure generation

All figures are produced by `figures/generate_figures.py` from CSV logs in `../results/logs/`:

```bash
cd report/figures
python generate_figures.py
```

This writes:
- `parameter-efficiency.pdf`
- `fewshot-curves.pdf`
- `training-curves.pdf`
- `prompt-length.pdf`

The VPT-Deep confusion matrix (`vpt-deep-cm.pdf`) is copied from the evaluation notebook.

## NTU Logo

Drop an `ntu-logo.png` file into `figures/` for the title page.
