#!/usr/bin/env python3
"""Inject the reply-length rule on every prompt, so brevity is not a mode.

The user has asked for short, plain replies eleven times, and each time it
held for a while and then drifted back over a long session. A rule that lives
only in prose drifts; one that is re-injected with every prompt does not. This
is the mechanical gate for it, in the sense of RULE 2 in CLAUDE.md: before
claiming the rule was followed, there is now something that checked.

Fail-open by design — a hook that can break the session is worse than a verbose
reply.
"""
import sys

RULE = (
    "REPLY STYLE (default, every reply): caveman register, ~60 words, "
    "verdict on the first line, at most 3 short bullets, then stop. "
    "Drop articles, filler, hedging and pleasantries; fragments are fine. "
    "No headers, no tables, no file paths, function names, config keys or "
    "job ids unless asked — say the thing, not the symbol. One idea per "
    "line. Own an error in one sentence at the top. A decision needs one "
    "question with numbered options and your pick named. "
    "Longer only when a report or walkthrough is explicitly requested. "
    "Code, commits, PRs and security warnings are written in full English."
)


def main() -> int:
    print(RULE)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
