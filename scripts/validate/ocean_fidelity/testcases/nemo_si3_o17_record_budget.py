#!/usr/bin/env python3
"""Exact ORCA2 O17 stream-record layout, size, and unified acquisition budget.

Every recorded field's extent is derived from source, never typed: the O17
writer's ``put`` calls name either a NEMO module array (resolved through the
shared allocation catalog over the real ORCA2_SAS_ICE ppsrc) or a writer dummy,
which is followed to the observer patch / writer call that passes it and then
to that caller's NEMO declaration.  The deck extents come from the admitted
ocean.output (``Ni0glo=180``, ``Nj0glo=148``, ``nn_hls=2`` -> 184 x 152).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4.nemo_acquisition_extents import (
    DeckDimensions,
    ExtentError,
    _actual_shape,
    _dimension_size,
    _paren_body,
    _split_top_level,
    build_catalog,
    logical_statements,
)

HERE = Path(__file__).resolve().parent
ACQ = HERE / "nemo_si3_o8_o16_master_acquisition"
O17_WRITER = ACQ / "o17_master_writer.F90"
MASTER_WRITER = ACQ / "o8_o16_master_writer.F90"
DECK_LOG = HERE / "nemo_si3_o7_orca2_sas_acquisition/ocean.output.excerpt"
DEFAULT_PPSRC = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo"
)
DEFAULT_NEMO_SRC = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src")
STEPS = 100  # ORCA2_SAS_ICE namelist_cfg nn_itend (master run.sh render_deck)
RANKS = 1  # master manifest 'ranks': 1
STREAM_HEADER = 44  # o17_master_writer.F90:107-113 magic(8)+version(4)+meta(8*4)
FIELD_HEADER = 52  # o17_master_writer.F90:30-37 name(32)+rank(4)+dims(4*4)
STREAM_END = 52  # o17_master_writer.F90:81-84
MASTER_BYTES = 140_965_556_460  # nemo_si3_o8_o16_master_gate.size_report()
WALL_CLOCK_SECONDS = 21_600
# o17_state(kind) kinds (o17_master_writer.F90:122-131) and the other writers.
STATE_KINDS = (22, 23, 24, 25, 26, 27, 28, 31)
KIND_ROUTINE = {**{k: "o17_state" for k in STATE_KINDS}, 29: "o17_prather",
                30: "o17_prather", 32: "o17_raw_blk1", 33: "o17_raw_blk2",
                34: "o17_tau_private"}
_PUT = re.compile(r"call\s+put(?:_o17_f2)?\s*\(\s*u\s*,\s*'(\w+)'\s*,\s*(.+)\)\s*$", re.I)


def _routines(path: Path) -> dict[str, dict]:
    """Writer subroutines: dummies, ordered puts (conditional flag), calls, copies."""
    out: dict[str, dict] = {}
    current = None
    depth = 0
    for st in logical_statements(path.read_text()):
        head = re.match(r"subroutine\s+(\w+)\s*\((.*)\)\s*$", st, re.I)
        if head:
            current = out.setdefault(head.group(1).lower(), {
                "dummies": [a.strip().lower() for a in head.group(2).split(",")],
                "puts": [], "calls": [], "locals": set()})
            depth = 0
            continue
        if current is None:
            continue
        if re.match(r"end\s+subroutine", st, re.I):
            current = None
            continue
        low = st.lower()
        if re.match(r"if\s*\(.*\)\s*then$", low):
            depth += 1
            continue
        if re.match(r"end\s*if$", low):
            depth -= 1
            continue
        inline_if = re.match(r"if\s*\((.*?)\)\s*(call\s+.*)$", st, re.I)
        body = inline_if.group(2) if inline_if else st
        put = _PUT.match(body)
        if put:
            current["puts"].append((put.group(1), put.group(2).strip(), bool(inline_if) or depth > 0))
            continue
        call = re.match(r"call\s+(\w+)\s*\((.*)\)\s*$", body, re.I)
        if call:
            current["calls"].append((call.group(1).lower(), _split_top_level(call.group(2))))
        decl = re.match(r"real\s*\(\s*real64\s*\)\s*::\s*(.*)$", st, re.I)
        if decl:
            current["locals"].update(v.strip().lower() for v in decl.group(1).split(","))
    return out


def _copies(path: Path) -> dict[str, tuple[str, str]]:
    """Module copies such as ``cached_fast_t=fast_t`` -> (routine, source dummy)."""
    copies: dict[str, tuple[str, str]] = {}
    current = None
    for st in logical_statements(path.read_text()):
        head = re.match(r"subroutine\s+(\w+)", st, re.I)
        if head:
            current = head.group(1).lower()
            continue
        match = re.match(r"(cached_\w+)\s*=\s*(\w+)\s*$", st, re.I)
        if match and current:
            copies[match.group(1).lower()] = (current, match.group(2).lower())
    return copies


def _patch_calls(patch_dir: Path, nemo_src: Path) -> dict[str, list[tuple[str, str, list[str]]]]:
    """Observer CALLs in the patches with the pristine NEMO subroutine enclosing them."""
    found: dict[str, list[tuple[str, str, list[str]]]] = {}
    for patch in sorted(patch_dir.glob("*.patch")):
        unit = patch.name.split("_o")[0]
        sub = "OCE/SBC" if unit == "sbcblk" else "ICE"
        pristine = (nemo_src / sub / f"{unit}.F90").read_text().splitlines()
        anchor = 0
        pending = ""
        for raw in patch.read_text().splitlines():
            hunk = re.match(r"@@ -(\d+)", raw)
            if hunk:
                anchor = int(hunk.group(1))
                continue
            if not raw.startswith("+") or raw.startswith("+++"):
                continue
            line = raw[1:].split("!", 1)[0].strip()
            if line.startswith("&"):
                line = line[1:].strip()
            pending += " " + line.rstrip("&")
            if line.endswith("&"):
                continue
            call = re.match(r"\s*call\s+(\w+)\s*\((.*)\)\s*$", pending, re.I)
            pending = ""
            if not call:
                continue
            enclosing = None
            for text in reversed(pristine[:anchor]):
                head = re.match(r"\s*(?:end\s+)?subroutine\s+(\w+)", text, re.I)
                if head:
                    enclosing = None if text.strip().lower().startswith("end") else head.group(1).lower()
                    break
            if enclosing is None:
                raise ExtentError(f"observer call outside a subroutine: {patch.name}:{anchor}")
            found.setdefault(call.group(1).lower(), []).append(
                (unit, enclosing, _split_top_level(call.group(2)))
            )
    return found


@functools.lru_cache(maxsize=64)
def _declared_shapes(ppsrc: Path, unit: str, routine: str,
                     deck: DeckDimensions) -> dict[str, tuple[int, ...]]:
    """Explicit-shape declarations inside one NEMO routine (ppsrc macros expanded)."""
    shapes: dict[str, tuple[int, ...]] = {}
    inside = False
    for st in logical_statements((ppsrc / f"{unit}.f90").read_text()):
        if re.match(rf"subroutine\s+{routine}\b", st, re.I):
            inside = True
            continue
        if inside and re.match(rf"end\s+subroutine\s+{routine}\b", st, re.I):
            break
        if not inside or "::" not in st:
            continue
        head, tail = st.split("::", 1)
        attr = re.search(r"\bdimension\s*\(", head, re.I)
        if not attr:
            continue
        body, _ = _paren_body(head, head.find("(", attr.start()))
        dims = _split_top_level(body)
        if any(dim.strip() == ":" for dim in dims):
            continue  # assumed shape takes the actual's extent
        for item in _split_top_level(tail):
            name = re.match(r"\s*(\w+)", item).group(1).lower()
            shapes[name] = tuple(_dimension_size(dim, deck.values) for dim in dims)
    return shapes


@functools.lru_cache(maxsize=4)
def record_layout(
    ppsrc: Path = DEFAULT_PPSRC,
    deck: DeckDimensions | None = None,
    nemo_src: Path = DEFAULT_NEMO_SRC,
) -> dict[int, tuple[tuple[str, tuple[int, ...], bool], ...]]:
    """Per kind, the writer's ordered ``(field, shape, conditional)`` tuple."""
    deck = deck or DeckDimensions.from_ocean_output(DECK_LOG)
    catalog = build_catalog(sorted(ppsrc.glob("*.f90")))
    routines = _routines(O17_WRITER) | {
        k: v for k, v in _routines(MASTER_WRITER).items() if k in {"o17_tau_private", "o8o16_tau"}
    }
    copies = _copies(O17_WRITER)
    observers = _patch_calls(ACQ, nemo_src)

    def nemo_shape(expr: str, unit: str, routine: str) -> tuple[int, ...]:
        base = re.match(r"\s*(\w+)", expr).group(1).lower()
        declared = _declared_shapes(ppsrc, unit, routine, deck).get(base)
        if declared is not None and "(" not in expr:
            return declared  # local declaration shadows a module array
        return _actual_shape(expr.replace("current_kbb", "1"), catalog, deck)

    def writer_shape(expr: str, routine: str) -> tuple[int, ...]:
        base = re.match(r"\s*(\w+)", expr).group(1).lower()
        if base in copies:
            return writer_shape(copies[base][1], copies[base][0])
        info = routines[routine]
        if base in info["locals"]:
            return ()
        if base not in info["dummies"]:  # writer USE-associated NEMO module array
            return _actual_shape(expr.replace("current_kbb", "1"), catalog, deck)
        index = info["dummies"].index(base)
        shapes = set()
        for caller, caller_info in routines.items():  # writer-to-writer calls
            for callee, args in caller_info["calls"]:
                if callee == routine:
                    shapes.add(writer_shape(args[index], caller))
        for unit, enclosing, args in observers.get(routine, ()):
            shapes.add(nemo_shape(args[index], unit, enclosing))
        if len(shapes) != 1:
            raise ExtentError(f"{routine}:{base} resolves to {sorted(shapes)}")
        return shapes.pop()

    layout = {}
    for kind, routine in KIND_ROUTINE.items():
        layout[kind] = tuple(
            (name, writer_shape(expr, routine), conditional)
            for name, expr, conditional in routines[routine]["puts"]
        )
    return layout


def record_bytes(shapes) -> int:
    """Bytes written by one stream record, including every field header."""
    return STREAM_HEADER + sum(FIELD_HEADER + 8 * math.prod(s) for s in shapes) + STREAM_END


def size_report(master_bytes: int = MASTER_BYTES) -> dict[str, int | float]:
    """Exact schema maximum (every conditional field present) and one 4% disk budget."""
    layout = record_layout()
    per_kind = {kind: record_bytes(s for _, s, _ in fields) for kind, fields in layout.items()}
    per_step = sum(per_kind.values()) * RANKS
    exact_max = STEPS * per_step
    combined = master_bytes + exact_max
    deck = DeckDimensions.from_ocean_output(DECK_LOG)
    return {
        "jpi": deck.jpi,
        "jpj": deck.jpj,
        "owned": [deck.jpi - 2 * deck.nn_hls, deck.jpj - 2 * deck.nn_hls],
        "record_bytes_by_kind": {str(k): v for k, v in sorted(per_kind.items())},
        "o17_bytes_per_step_max": per_step,
        "o17_record_bytes_exact_max": exact_max,
        "o8_o16_master_bytes_worst_case": master_bytes,
        "combined_record_bytes_max": combined,
        "combined_disk_budget_kib_4pct": math.ceil(combined * 1.04 / 1024),
        "wall_clock_budget_seconds": WALL_CLOCK_SECONDS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-bytes", type=int, default=MASTER_BYTES)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--layout", action="store_true")
    args = parser.parse_args()
    if args.layout:
        print(json.dumps({str(k): [[n, list(s), c] for n, s, c in v]
                          for k, v in record_layout().items()}, indent=1))
        return 0
    report = size_report(args.master_bytes)
    if args.shell:
        print(report["o17_record_bytes_exact_max"], report["combined_disk_budget_kib_4pct"])
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
