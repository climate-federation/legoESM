#!/bin/bash
# Create a PINNED run worktree the safe way, and refuse if it cannot be exact.
#
# Replaces the pattern that failed twice on 2026-08-14: create a worktree at a
# SHA, then hand-copy uncommitted files into it.  That is fragile (`cp` is
# aliased to prompt here, so a plain `cp` AND a `cp -f` both silently did
# nothing and a job ran against stale code), and it defeats the pin -- a
# worktree with copied-in files no longer matches the SHA it claims.
#
#   usage: pin_worktree.sh <name>      e.g. pin_worktree.sh run8
#
# Aborts if the source worktree is dirty, because that is the only way a
# hand-copy is ever tempting.  Commit first; git then places every file.
set -euo pipefail
SRC=/burg-archive/glab/users/pg2328/legoESM_fv3jax
NAME=${1:?usage: pin_worktree.sh <name>}
DST=/burg-archive/glab/users/pg2328/_fv3jax_$NAME

cd "$SRC"
DIRTY=$(git status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ]; then
  echo "REFUSING: $SRC is dirty ($DIRTY files). Commit first -- a pinned run"
  echo "must be reproducible from its SHA, and copying files in defeats that."
  git status --porcelain | head -20
  exit 2
fi
SHA=$(git rev-parse HEAD)

[ -e "$DST" ] && git worktree remove --force "$DST" 2>/dev/null || true
rm -rf "$DST"
git worktree add --detach "$DST" "$SHA" >/dev/null

GOT=$(git -C "$DST" rev-parse HEAD)
[ "$GOT" = "$SHA" ] || { echo "PIN MISMATCH: $GOT != $SHA"; exit 3; }
DDIRTY=$(git -C "$DST" status --porcelain | wc -l)
[ "$DDIRTY" -eq 0 ] || { echo "PINNED TREE IS DIRTY on creation"; exit 4; }

echo "$DST"
echo "pinned at $SHA (verified clean, no files copied)"
