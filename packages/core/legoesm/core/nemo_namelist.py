"""NEMO namelist reader: source files and NEMO's resolved dump share one grammar.

``parse_namelist`` maps ``(ns, group, key)`` to normalised value tokens;
``overlay`` merges parsed files in the order NEMO reads them (``namelist_ref``
then ``namelist_cfg``: a later file's key replaces the earlier one).
"""

from __future__ import annotations

import re
from pathlib import Path

_KEY = re.compile(r"(?:^|(?<=[\s,]))([A-Za-z][A-Za-z0-9_%]*(?:\(\s*\d+\s*\))?)\s*=")
_QUOTED = re.compile(r"'[^']*'?|\"[^\"]*\"?")


def strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
    return "".join(out)


def _split(text: str) -> list[str]:
    parts, cur, quote = [], [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == ",":
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def norm(token: str):
    t = token.strip()
    if t[:1] in ("'", '"'):
        return t[1:-1].rstrip() if len(t) > 1 and t[-1] == t[0] else t[1:].rstrip()
    low = t.lower()
    if low in (".true.", "t", ".t.", "true"):
        return True
    if low in (".false.", "f", ".f.", "false"):
        return False
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        return low


def values(text: str) -> list:
    out = []
    for part in _split(text):
        m = re.fullmatch(r"(\d+)\*(.*)", part)
        if m and not part.startswith(("'", '"')):
            out.extend([norm(m.group(2))] * int(m.group(1)))
        else:
            out.append(norm(part))
    return out


def parse_namelist(path: Path, ns: str) -> dict[tuple[str, str, str], list]:
    """``(ns, group, key)`` -> normalised value tokens, in file order."""

    found: dict[tuple[str, str, str], list] = {}
    group, key, buf = None, None, ""

    def flush():
        if group and key is not None:
            vals = values(buf)
            name = (ns, group, key)
            if name in found and found[name] != vals:
                found[name] = vals            # NEMO keeps the last read
            else:
                found[name] = vals

    def extend(text: str):
        nonlocal buf
        text = text.strip()
        if key is not None and text:
            buf = buf.rstrip() + ("" if not buf.strip() or buf.rstrip().endswith(",") else ",") + text

    for raw in Path(path).read_text(errors="replace").splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            flush()
            name = line[1:].split()[0]
            group, key, buf = name.lower(), None, ""
            line = line[1 + len(name):].strip()
        if group is None:
            continue
        # Quoted text is masked at equal length, so '=' or '/' inside a string stays a value.
        masked = _QUOTED.sub(lambda q: "x" * len(q.group()), line)
        slash = masked.find("/")
        body = line if slash < 0 else line[:slash]
        hits = list(_KEY.finditer(masked[: len(body)]))
        extend(body[: hits[0].start()] if hits else body)
        for i, m in enumerate(hits):
            flush()
            key = re.sub(r"\s+", "", m.group(1)).lower()
            buf = body[m.end(): hits[i + 1].start() if i + 1 < len(hits) else len(body)]
        if slash >= 0:
            flush()
            group, key, buf = None, None, ""
    flush()
    return found


def overlay(*parsed: dict) -> dict:
    out: dict = {}
    for p in parsed:
        out.update(p)
    return out


__all__ = ("norm", "overlay", "parse_namelist", "strip_comment")
