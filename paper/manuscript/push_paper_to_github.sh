#!/usr/bin/env bash
# Create a clean, paper-only GitHub repo from this manuscript bundle and push it,
# so Overleaf's "Import from GitHub" sees main.tex at the repo ROOT (default branch).
#
# Usage:
#   ./push_paper_to_github.sh [DEST_DIR] [GITHUB_REPO]
# Defaults:
#   DEST_DIR     = ~/Documents/legoESM-paper
#   GITHUB_REPO  = gentine/legoESM-paper   (change to your account/name)
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/Documents/legoESM-paper}"
REPO="${2:-gentine/legoESM-paper}"

echo ">> Building clean paper repo in: $DEST"
rm -rf "$DEST" && mkdir -p "$DEST"
cp -r "$SRC/main.tex" "$SRC/supplementary.tex" "$SRC/references.bib" \
      "$SRC/figures" "$SRC/README.md" "$DEST"/

cd "$DEST"
cat > .gitignore <<'EOF'
*.aux
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.out
*.toc
*.log
*.synctex.gz
EOF

git init -q
git add -A
git commit -q -m "legoESM Nature manuscript (main text, SI, figures, references)"
git branch -M main

if command -v gh >/dev/null 2>&1; then
  echo ">> Creating GitHub repo $REPO and pushing (via gh CLI)..."
  gh repo create "$REPO" --private --source=. --remote=origin --push
  echo ">> Done. Repo: https://github.com/$REPO  (branch: main)"
else
  echo ">> GitHub CLI (gh) not found. Create an empty repo named '$REPO' on github.com, then run:"
  echo "     cd \"$DEST\""
  echo "     git remote add origin https://github.com/$REPO.git"
  echo "     git push -u origin main"
fi

echo ""
echo ">> Finally, in Overleaf:  New Project -> Import from GitHub -> $REPO"
echo "   main.tex is at the repo root, so it will appear immediately."
