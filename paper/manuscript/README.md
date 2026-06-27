# legoESM — Nature paper bundle

Compile-ready LaTeX for the legoESM manuscript.

- `main.tex` — main article (title, authors, abstract, intro, results, Methods).
- `supplementary.tex` — Supplementary Information (full equations §1, then §2–§8).
- `references.bib` — bibliography (shared by both documents).
- `figures/` — all display items. Main: `f2_*`–`f6_*` (Fig. 1 is a TikZ schematic
  inside `main.tex`); Supplementary: `si_*`. `f6_scaling.pdf` is generated.

## Compile
```bash
latexmk -pdf main.tex
latexmk -pdf supplementary.tex
```
(or pdflatex → bibtex → pdflatex × 2). Builds cleanly on TeX Live with no extra packages.

## Push to the existing Overleaf project (recommended: clone, copy, push)
Get a Git token at Overleaf → Account Settings → Git, then:
```bash
# 1. clone the existing Overleaf project (uses the token as the password)
git clone https://git@git.overleaf.com/6a398dcf2cfe1ae37fe8b69f overleaf_legoesm

# 2. copy this bundle's files in
cp -r main.tex supplementary.tex references.bib figures overleaf_legoesm/

# 3. commit and push
cd overleaf_legoesm
git add main.tex supplementary.tex references.bib figures
git commit -m "Add legoESM manuscript, SI, references and figures"
git push
```
Then open the Overleaf project and set the main document to `main.tex`.

## Alternative: push this repo directly (only if the Overleaf project is empty)
```bash
git remote add overleaf https://git@git.overleaf.com/6a398dcf2cfe1ae37fe8b69f
git push -u overleaf master   # add --force to overwrite existing Overleaf content
```

## To verify before submission
- Author affiliations are best-effort inferences — please confirm each.
- Repository URL is set to https://github.com/gentine/legoESM (from CITATION.cff).
- Scaling numbers in Fig. 6 come from `docs/performance/scaling/`; re-confirm before submission.
