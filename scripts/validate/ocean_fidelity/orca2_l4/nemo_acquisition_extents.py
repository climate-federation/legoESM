#!/usr/bin/env python3
"""Shared fail-closed NEMO allocation and acquisition-writer extent gate."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeckDimensions:
    jpi: int
    jpj: int
    jpk: int
    nn_hls: int
    jpl: int = 5
    nlay_i: int = 10
    nlay_s: int = 5
    jpt: int = 2

    @property
    def values(self) -> dict[str, int]:
        return {
            "jpi": self.jpi, "jpj": self.jpj, "jpk": self.jpk,
            "jpkm1": self.jpk - 1, "nn_hls": self.nn_hls,
            "nis0": 1 + self.nn_hls, "nie0": self.jpi - self.nn_hls,
            "njs0": 1 + self.nn_hls, "nje0": self.jpj - self.nn_hls,
            "ntsi": 1 + self.nn_hls, "ntei": self.jpi - self.nn_hls,
            "ntsj": 1 + self.nn_hls, "ntej": self.jpj - self.nn_hls,
            "jpij": (self.jpi - 2 * self.nn_hls) * (self.jpj - 2 * self.nn_hls),
            "jpl": self.jpl, "nlay_i": self.nlay_i, "nlay_s": self.nlay_s,
            "jpt": self.jpt,
        }

    @classmethod
    def from_ocean_output(cls, path: Path) -> DeckDimensions:
        text = path.read_text()
        def value(pattern: str) -> int:
            matches = re.findall(pattern, text, flags=re.I)
            if not matches:
                raise ExtentError(f"missing deck dimension {pattern} in {path}")
            return int(matches[-1])
        ni = value(r"\bNi0glo\s*=\s*(\d+)")
        nj = value(r"\bNj0glo\s*=\s*(\d+)")
        jpk = value(r"\bjpkglo\s*=\s*(\d+)")
        halo = value(r"\bnn_hls\s*=\s*(\d+)")
        return cls(ni + 2 * halo, nj + 2 * halo, jpk, halo)


@dataclass(frozen=True)
class ArrayExtent:
    name: str
    dimensions: tuple[str, ...]
    source: str


class ExtentError(ValueError):
    pass


def _without_comment(line: str) -> str:
    quote = None
    out: list[str] = []
    for char in line:
        if char in "'\"":
            quote = None if quote == char else (char if quote is None else quote)
        if char == "!" and quote is None:
            break
        out.append(char)
    return "".join(out)


def logical_statements(text: str) -> list[str]:
    statements: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = _without_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            line = line[1:].lstrip()
        continued = line.endswith("&")
        if continued:
            line = line[:-1]
        current += " " + line
        if not continued:
            statements.extend(part.strip() for part in current.split(";") if part.strip())
            current = ""
    if current:
        raise ExtentError("unterminated Fortran continuation")
    return statements


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    for pos, char in enumerate(text + ","):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:pos].strip())
            start = pos + 1
    if depth:
        raise ExtentError(f"unbalanced Fortran expression: {text}")
    return [part for part in parts if part]


def _paren_body(text: str, start: int) -> tuple[str, int]:
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "(":
            depth += 1
        elif text[pos] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:pos], pos
    raise ExtentError(f"unbalanced parentheses: {text}")


def _expand_macros(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        halo = match.group(2)
        if match.group(1).lower() == "a2d":
            return f"Nis0-({halo}):Nie0+({halo}),Njs0-({halo}):Nje0+({halo})"
        return f"ntsi-({halo}):ntei+({halo}),ntsj-({halo}):ntej+({halo})"
    return re.sub(r"\b(A2D|T2D)\s*\(\s*([^()]+)\s*\)", repl, text, flags=re.I)


def build_catalog(paths: Iterable[Path]) -> dict[str, tuple[ArrayExtent, ...]]:
    """Index ALLOCATE shapes and explicit declaration shapes in preprocessed NEMO."""
    found: dict[str, list[ArrayExtent]] = {}
    for path in paths:
        for statement in logical_statements(path.read_text()):
            low = statement.lower()
            allocate = re.match(r"\s*allocate\s*\(", low)
            if allocate:
                body, _ = _paren_body(statement, statement.find("(", allocate.start()))
                for item in _split_top_level(_expand_macros(body)):
                    match = re.match(r"\s*([a-z]\w*)\s*\((.*)\)\s*$", item, re.I)
                    if not match or match.group(1).lower() == "stat":
                        continue
                    extent = ArrayExtent(
                        match.group(1).lower(), tuple(_split_top_level(match.group(2))),
                        f"{path}:{statement[:80]}",
                    )
                    found.setdefault(extent.name, []).append(extent)
            if "::" not in statement or not re.match(
                r"\s*(?:integer|real|logical|complex|character|type\s*\()\b", statement, re.I
            ):
                continue
            head, tail = statement.split("::", 1)
            attr = re.search(r"\bdimension\s*\(", head, re.I)
            attr_dims: tuple[str, ...] | None = None
            if attr:
                body, _ = _paren_body(head, head.find("(", attr.start()))
                if ":" not in body.replace("0:", ""):
                    attr_dims = tuple(_split_top_level(_expand_macros(body)))
            for item in _split_top_level(tail):
                match = re.match(r"\s*([a-z]\w*)\s*(?:\((.*)\))?", item, re.I)
                if not match:
                    continue
                dims = (
                    tuple(_split_top_level(_expand_macros(match.group(2))))
                    if match.group(2) else attr_dims
                )
                if not dims or all(dim.strip() in {":", "0:"} for dim in dims):
                    continue
                extent = ArrayExtent(match.group(1).lower(), dims, f"{path}:{statement[:80]}")
                found.setdefault(extent.name, []).append(extent)
    return {name: tuple(values) for name, values in found.items()}


def _eval_int(expression: str, values: dict[str, int]) -> int:
    clean = re.sub(r"(?<=\d)_(?:wp|i\w*)\b", "", expression.lower()).strip()
    tree = ast.parse(clean, mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)
        ):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left // right
        raise ExtentError(f"unsupported extent expression: {expression}")

    return visit(tree)


def _dimension_size(expression: str, values: dict[str, int]) -> int:
    pieces = _split_top_level(expression.replace(":", ",")) if ":" in expression else []
    if not pieces:
        return _eval_int(expression, values)
    if len(pieces) == 2:
        return _eval_int(pieces[1], values) - _eval_int(pieces[0], values) + 1
    raise ExtentError(f"unsupported array bound: {expression}")


def resolved_shape(extent: ArrayExtent, deck: DeckDimensions) -> tuple[int, ...]:
    return tuple(_dimension_size(dim, deck.values) for dim in extent.dimensions)


def unique_extent(
    catalog: dict[str, tuple[ArrayExtent, ...]], name: str, deck: DeckDimensions
) -> ArrayExtent:
    candidates = catalog.get(name.lower(), ())
    if not candidates:
        raise ExtentError(f"missing allocation/declaration for recorded array {name}")
    by_shape: dict[tuple[int, ...], ArrayExtent] = {}
    for candidate in candidates:
        try:
            by_shape.setdefault(resolved_shape(candidate, deck), candidate)
        except (ExtentError, SyntaxError):
            continue
    if not by_shape:
        raise ExtentError(f"no resolvable allocation for recorded array {name}")
    if len(by_shape) != 1:
        raise ExtentError(f"ambiguous allocation shapes for {name}: {sorted(by_shape)}")
    return next(iter(by_shape.values()))


def _actual_shape(
    expression: str,
    catalog: dict[str, tuple[ArrayExtent, ...]],
    deck: DeckDimensions,
) -> tuple[int, ...]:
    match = re.match(r"\s*([a-z]\w*)\s*(?:\((.*)\))?\s*$", expression, re.I)
    if not match:
        raise ExtentError(f"unsupported writer actual argument: {expression}")
    extent = unique_extent(catalog, match.group(1), deck)
    base = resolved_shape(extent, deck)
    if match.group(2) is None:
        return base
    indices = _split_top_level(match.group(2))
    if len(indices) != len(base):
        raise ExtentError(f"rank mismatch in writer actual argument: {expression}")
    shape: list[int] = []
    for index, size in zip(indices, base, strict=True):
        index = index.strip()
        if index == ":":
            shape.append(size)
        elif ":" in index:
            shape.append(_dimension_size(index, deck.values))
        else:
            _eval_int(index.replace("kbb", "1"), deck.values)
    return tuple(shape)


def writer_put_calls(writer: Path) -> tuple[tuple[str, str], ...]:
    calls: list[tuple[str, str]] = []
    for statement in logical_statements(writer.read_text()):
        match = re.match(r"call\s+put_([a-z0-9]+)\s*\((.*)\)\s*$", statement, re.I)
        if match:
            modes = {
                "i2", "h1", "f2", "i3", "ii4", "is4", "g1", "gi",
                "gs", "gi0", "gs0", "c0", "c1",
            }
            if match.group(1).lower() not in modes:
                continue
            args = _split_top_level(match.group(2))
            if match.group(1).lower() == "idx" and len(args) == 1:
                continue
            if len(args) != 2:
                raise ExtentError(f"unexpected put call: {statement}")
            calls.append((match.group(1).lower(), args[1]))
    if not calls:
        raise ExtentError(f"no writer put calls found in {writer}")
    return tuple(calls)


def _mode_accepts(mode: str, shape: tuple[int, ...], deck: DeckDimensions) -> bool:
    v = deck.values
    interior = (v["ntei"] - v["ntsi"] + 1, v["ntej"] - v["ntsj"] + 1)
    full = (deck.jpi, deck.jpj)
    expected = {
        "h1": {(interior[0] + 2, interior[1] + 2)},
        "f2": {full}, "i2": {interior, full},
        "i3": {interior + (deck.jpl,), full + (deck.jpl,)},
        "ii4": {full + (deck.nlay_i, deck.jpl)},
        "is4": {full + (deck.nlay_s, deck.jpl)},
        "gi": {(v["jpij"], deck.nlay_i)}, "gs": {(v["jpij"], deck.nlay_s)},
        "gi0": {(v["jpij"], deck.nlay_i + 1)}, "gs0": {(v["jpij"], deck.nlay_s + 1)},
        "c0": {(deck.jpl + 1,)}, "c1": {(deck.jpl,)},
        "g1": {(v["jpij"],)},
    }
    return shape in expected.get(mode, set())


def validate_writer(
    ppsrc: Path, writer: Path, deck: DeckDimensions, plant: str | None = None,
) -> dict[str, object]:
    paths = tuple(ppsrc.glob("*.f90")) + (writer,)
    catalog = build_catalog(paths)
    aliases = {
        "pu_oce": "uu(:,:,1,1)", "pv_oce": "vv(:,:,1,1)",
        "ptmod_io": "tmod_io", "putau_oce": "utau_oce", "pvtau_oce": "vtau_oce",
    }
    if plant == "missing-allocation":
        catalog.pop("rcdu_ice", None)
    if plant == "dummy-actual":
        aliases["pu_oce"] = "qlead"
    checked = []
    for mode, actual in writer_put_calls(writer):
        expression = aliases.get(actual.lower(), actual)
        effective_mode = "f2" if plant == "writer-mode" and actual.lower() == "qlead" else mode
        shape = _actual_shape(expression, catalog, deck)
        if not _mode_accepts(effective_mode, shape, deck):
            raise ExtentError(
                f"STATUS PLANT-FIRED: writer {effective_mode} extent rejects {actual} {shape}"
            )
        checked.append({"mode": effective_mode, "actual": actual, "shape": list(shape)})
    return {
        "status": "PASS_SOURCE_DERIVED_WRITER_EXTENTS", "arrays_checked": len(checked),
        "deck": deck.values,
        "rCdU_ice_shape": list(
            resolved_shape(unique_extent(catalog, "rCdU_ice", deck), deck)
        ),
        "calls": checked,
    }


FIXTURE_ARRAYS = {
    "tmask": "tmask", "umask": "umask", "vmask": "vmask", "smask0": "smask0",
    "qsr": "qsr", "qns": "qns", "emp": "emp", "sfx": "sfx", "fr_i": "fr_i",
    "fwfice": "fwfice", "qns_tot": "qns_tot", "qsr_tot": "qsr_tot", "frq_m": "frq_m",
    "sss_m": "sss_m", "sst_m": "sst_m", "cloud_fra": "cloud_fra", "sprecip": "sprecip",
    "tprecip": "tprecip", "utau": "utau", "vtau": "vtau", "taum": "taum",
    "qsr_oce": "qsr_oce", "qns_oce": "qns_oce", "qemp_oce": "qemp_oce",
    "qemp_ice": "qemp_ice", "emp_oce": "emp_oce", "emp_ice": "emp_ice",
    "qsr_ice": "qsr_ice", "qns_ice": "qns_ice", "qevap_ice": "qevap_ice",
    "qcn_ice": "qcn_ice", "qml_ice": "qml_ice", "qtr_ice_top": "qtr_ice_top",
    "tn_ice": "tn_ice", "alb_ice": "alb_ice", "snwice_mass": "snwice_mass",
    "snwice_mass_b": "snwice_mass_b", "snwice_fmass": "snwice_fmass",
    "utau_ice": "utau_ice", "vtau_ice": "vtau_ice", "rCdU_ice": "rCdU_ice",
    "converged": "l_t_converged", "pu": "uu(:,:,1,1)", "pv": "vv(:,:,1,1)",
    "zkappa_i": "zkappa_i", "zradab_i": "zradab_i", "zradtr_i": "zradtr_i",
    "zkappa_s": "zkappa_s", "zradab_s": "zradab_s", "zradtr_s": "zradtr_s",
}


def fixture_include(
    ppsrc: Path, writer: Path, deck: DeckDimensions, plant: str | None = None
) -> str:
    catalog = build_catalog(tuple(ppsrc.glob("*.f90")) + (writer,))
    lines = [
        f"jpi={deck.jpi} ; jpj={deck.jpj} ; jpk={deck.jpk} ; "
        f"jpkm1={deck.jpk - 1} ; nn_hls={deck.nn_hls}",
        f"Nis0={1 + deck.nn_hls} ; Nie0={deck.jpi - deck.nn_hls} ; "
        f"Njs0={1 + deck.nn_hls} ; Nje0={deck.jpj - deck.nn_hls}",
        "ntsi=Nis0 ; ntei=Nie0 ; ntsj=Njs0 ; ntej=Nje0",
        f"jpij={(deck.jpi - 2 * deck.nn_hls) * (deck.jpj - 2 * deck.nn_hls)} ; "
        f"jpni=1 ; jpnj=1 ; jpl={deck.jpl} ; nlay_i={deck.nlay_i} ; "
        f"nlay_s={deck.nlay_s} ; npti=jpij",
    ]
    for target, source in FIXTURE_ARRAYS.items():
        match = re.match(r"([a-z]\w*)", source, re.I)
        assert match
        extent = unique_extent(catalog, match.group(1), deck)
        dims = extent.dimensions
        if "(" in source:
            shape = _actual_shape(source, catalog, deck)
            dims = tuple(str(value) for value in shape)
        if plant == "fixture-allocation" and target.lower() == "rcdu_ice":
            dims = ("jpi", "jpj")
        lines.append(f"ALLOCATE({target}({','.join(dims)}))")
    if plant == "fixture-allocation":
        planted = ArrayExtent("rCdU_ice", ("jpi", "jpj"), "fixture plant")
        source_shape = resolved_shape(unique_extent(catalog, "rCdU_ice", deck), deck)
        if resolved_shape(planted, deck) != source_shape:
            raise ExtentError(
                "STATUS PLANT-FIRED: generated fixture allocation differs from source rCdU_ice"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppsrc", type=Path, required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--fixture-include", type=Path)
    parser.add_argument(
        "--plant",
        choices=("missing-allocation", "writer-mode", "dummy-actual", "fixture-allocation"),
    )
    parser.add_argument("--jpi", type=int, default=184)
    parser.add_argument("--jpj", type=int, default=152)
    parser.add_argument("--jpk", type=int, default=31)
    parser.add_argument("--nn-hls", type=int, default=2)
    parser.add_argument("--deck-log", type=Path)
    args = parser.parse_args(argv)
    deck = (
        DeckDimensions.from_ocean_output(args.deck_log)
        if args.deck_log else DeckDimensions(args.jpi, args.jpj, args.jpk, args.nn_hls)
    )
    try:
        report = validate_writer(args.ppsrc, args.writer, deck, args.plant)
        if args.fixture_include:
            args.fixture_include.write_text(
                fixture_include(args.ppsrc, args.writer, deck, args.plant)
            )
            report["fixture_include"] = str(args.fixture_include)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ExtentError, OSError, SyntaxError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
