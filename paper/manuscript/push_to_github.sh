#!/usr/bin/env bash
# Push the merged manuscript to https://github.com/gentine/legoESM-paper
# Usage:  ./push_to_github.sh  [branch]     (default branch: main)
set -euo pipefail
BRANCH="${1:-main}"
SRC="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
git clone https://github.com/gentine/legoESM-paper.git "$TMP/repo"
cd "$TMP/repo"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
cp "$SRC/main.tex" "$SRC/supplementary.tex" "$SRC/references.bib" .
mkdir -p figures && cp -R "$SRC/figures/." figures/
git add -A main.tex supplementary.tex references.bib figures
git commit -m "Merge Pierre's abstract/intro/section rewrites onto latest manuscript

- adopt rewritten abstract + introduction (typos fixed, em dashes removed per style rule)
- bring in section edits: tensor-in/tendency-out, Adam/MUON, GM-Redi, MicroHH/SAM/ESMValTool,
  Derecho/Levante scaling runs, faithfulness oracle, SFNO module-swap ladder
- keep recent rounds: 13 authors, Extended Data Tables 1-10, JCM/differentiable-modelling
  paragraph, redesigned Fig 1, A100 scaling + land-bias figures, back-matter + summary caveats
- repo URL -> github.com/gentine/legoESM"
git push origin "$BRANCH"
echo "pushed to gentine/legoESM-paper ($BRANCH)"
