#!/usr/bin/env python3
"""Re-anchor every `<basename>:N` span of the files this round edited.

Method (standing brief, CITATION RE-ANCHOR RULE): build the old->new line map
from the file's own diff with difflib.SequenceMatcher -- equal blocks map 1:1,
a changed line maps to the nearest equal line below plus its offset -- and
apply it ONCE to every span in the gate's CITATION_MAP and in every receipt.
"""
import difflib, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASE = next((a for a in sys.argv[1:] if not a.startswith("-")), "HEAD~1")
APPLY = "--apply" in sys.argv

FILES = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only",
                        BASE, "HEAD"], capture_output=True, text=True,
                       check=True).stdout.split()
FILES = [f for f in FILES if f.endswith(".py")
         and subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                             f"{BASE}:{f}"], capture_output=True).returncode == 0]

def line_map(path):
    old = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:{path}"],
                         capture_output=True, text=True, check=True
                         ).stdout.splitlines()
    new = (REPO / path).read_text().splitlines()
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    m = {}
    blocks = sm.get_matching_blocks()
    for i, j, n in blocks:
        for k in range(n):
            m[i + k + 1] = j + k + 1
    # a changed/deleted old line maps to the nearest surviving line BELOW it
    survivors = sorted(m)
    for ln in range(1, len(old) + 1):
        if ln in m:
            continue
        nxt = next((s for s in survivors if s > ln), None)
        m[ln] = (m[nxt] - (nxt - ln)) if nxt is not None else (
            len(new) - (len(old) - ln))
    return m

MAPS = {Path(f).name: line_map(f) for f in FILES}
if not MAPS:
    print(f"no .py file changed between {BASE} and HEAD; nothing to re-anchor")
    raise SystemExit(0)
for name, m in MAPS.items():
    moved = sum(1 for k, v in m.items() if k != v)
    print(f"{name}: {len(m)} lines, {moved} shifted")

TARGETS = [REPO / "scripts/validate/ocean_fidelity/testcases"
           / "nemo_testcase_receipt_citation_gate.py"]
# Scope: the gate's CITATION_MAP and the receipt the gate reads.  Older
# receipts keep the line numbers that were right when they were written; the
# gate does not read them (AUDITED_ROUNDS).
TARGETS += [REPO / "docs/ocean/fidelity/testcases"
            / "nemo_testcases_l2_gyre_phase3_round8_receipt.md"]

NAMES = "|".join(re.escape(n) for n in MAPS)
# `<basename>:N` or `<basename>:N-M` or `<basename>:a,b,c`
PAT = re.compile(rf"\b({NAMES}):((?:\d+)(?:[-,]\d+)*)")

def remap(match):
    name, span = match.group(1), match.group(2)
    m = MAPS[name]
    out = re.sub(r"\d+", lambda d: str(m.get(int(d.group()), int(d.group()))),
                 span)
    return f"{name}:{out}"

total = 0
for path in TARGETS:
    text = path.read_text()
    new = PAT.sub(remap, text)
    if new != text:
        n = sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b)
        total += n
        print(f"{'rewrote' if APPLY else 'would rewrite'} "
              f"{path.relative_to(REPO)} ({n} lines)")
        if APPLY:
            path.write_text(new)
print(f"total lines touched: {total}")
