#!/usr/bin/env python3
"""Admit (or refuse) a VORTEX kt=1..10 step-record acquisition.

NOTE BD, binding: this checker NEVER predicts a record's size or its header
tuple.  It reads the record's own header, derives every dimension from it, and
verifies each payload length against the dimensions that record declares.  The
only hard-coded expectations are the magic strings and the list of record
families the round needs.

It also refuses unless the instrumented run's NEMO restart is byte-identical to
the un-instrumented reference run's restart at the same step: that, and not a
stream-to-stream comparison between two differently instrumented builds, is how
note AS says a writer's passivity is judged.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import struct
import sys
from pathlib import Path

# (magic, number of header integers after the magic, number of 3-D arrays,
#  number of extra 2-D arrays).  Everything else is read from the header.
_FAMILIES = {
    "oracle_step_entry_kt": ("NEMO_L1_ENTRY_1", 8, "ts+uuvv", 1),
    # stprk3.F90:349 writes SIX integers after the magic (version, step,
    # time level, jpi, jpj, bit width) -- one fewer than the other families,
    # because the barotropic record has no vertical dimension.
    "oracle_bt_frames_kt": ("NEMO_L1_BTFRM_1", 6, "b2d", 0),
    "oracle_stage_kt": ("NEMO_L1_STAGE_1", 9, "ts+uuvv", 1),
    # NOTE: the stage record's filename carries the stage after the step, so
    # its step digits are read from the "kt" split above, not from the tail.
    "oracle_rhs_kt": ("NEMO_L1_RHS___1", 7, "uuvv", 0),
    # Round 4's per-term momentum record.  Sixteen header integers and then
    # (name, rank, n1, n2, n3, payload) groups to end of file -- the
    # self-describing shape note BD makes binding, so nothing about its size
    # is written down here.
    "oracle_rhsterm_kt": ("NEMO_L1_RHSTRM1", 16, "groups", 0),
}
# The groups every per-term record must carry, by name.  This list, the magic
# and the format version are the ONLY hard-coded expectations.
_RHSTERM_GROUPS = ("uu_rhs", "vv_rhs", "ww", "r3t_Kaa")
# The boundaries stp_2D is instrumented at, in NEMO's own execution order;
# the per-term increments are the differences between consecutive records.
_RHSTERM_BOUNDARIES = ("hpg", "ldf", "vor", "wzv", "keg", "zad")


class Refusal(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise Refusal(message)


def parse_record(path: Path, corrupt_header: bool = False) -> dict:
    """Parse one record from its own header.  Raises Refusal on any mismatch."""
    family = next(
        (name for name in _FAMILIES if path.name.startswith(name)), None)
    _require(family is not None, f"{path.name}: unknown record family")
    magic_expected, n_header, layout, n_2d = _FAMILIES[family]
    raw = path.read_bytes()
    _require(len(raw) > 16 + 4 * n_header, f"{path.name}: truncated")
    magic = raw[:16].decode("ascii", "replace").rstrip()
    header = list(struct.unpack(f"={n_header}i", raw[16:16 + 4 * n_header]))
    if corrupt_header:
        header[-2] += 1
    _require(magic == magic_expected,
             f"{path.name}: magic {magic!r} is not {magic_expected!r}")
    _require(header[0] == 1,
             f"{path.name}: record format version {header[0]}, this checker "
             "reads version 1")
    bits = header[-1]
    _require(bits == 64, f"{path.name}: records must be 64-bit, header says {bits}")
    # The step is in the FILENAME and in the header; a writer that drifted
    # between them would silently mislabel the whole ladder.
    stamped = "".join(ch for ch in path.stem.split("kt")[-1][:8] if ch.isdigit())
    if stamped:
        _require(int(stamped) == header[1],
                 f"{path.name}: filename says step {int(stamped)}, its header "
                 f"says {header[1]}")
    if layout == "groups":
        return _parse_groups(path, raw, magic, header, n_header)
    payload = len(raw) - (16 + 4 * n_header)
    _require(payload % 8 == 0, f"{path.name}: payload is not a whole number of f64")
    values = payload // 8
    # The trailing header integers are the dimensions the WRITER declared.
    if layout == "ts+uuvv":          # ts(:,:,:,:), uu, vv, ssh
        nx, ny, nz, ntr = header[-5:-1]
        cell = nx * ny * nz
        expected = ntr * cell + 2 * cell + n_2d * nx * ny
    elif layout == "uuvv":           # uu, vv
        nx, ny, nz = header[-4:-1]
        cell = nx * ny * nz
        ntr = 0
        expected = 2 * cell
    else:                            # four 2-D fields
        nx, ny = header[-3:-1]
        nz = ntr = 0
        expected = 4 * nx * ny
    _require(
        values == expected,
        f"{path.name}: payload holds {values} doubles, but its OWN header "
        f"{(nx, ny, nz, ntr)} implies {expected}")
    return {"file": path.name, "magic": magic, "header": header,
            "nx": nx, "ny": ny, "nz": nz, "ntr": ntr, "doubles": values}


def _parse_groups(path: Path, raw: bytes, magic: str, header: list,
                  n_header: int) -> dict:
    """Walk (name, rank, n1, n2, n3, payload) groups to end of file.

    Nothing here predicts a size: each payload's length is checked against the
    rank and extents that group itself declares, and the file must end exactly
    on a group boundary.
    """
    offset = 16 + 4 * n_header
    groups = {}
    while offset < len(raw):
        _require(offset + 32 <= len(raw),
                 f"{path.name}: a group header is truncated at byte {offset}")
        name = raw[offset:offset + 16].decode("ascii", "replace").rstrip()
        rank, n1, n2, n3 = struct.unpack("=4i", raw[offset + 16:offset + 32])
        offset += 32
        _require(rank in (2, 3), f"{path.name}: group {name!r} has rank {rank}")
        _require(min(n1, n2, n3) > 0,
                 f"{path.name}: group {name!r} declares a nonpositive extent")
        count = n1 * n2 * (n3 if rank == 3 else 1)
        _require(offset + 8 * count <= len(raw),
                 f"{path.name}: group {name!r} declares {count} doubles, but "
                 f"only {(len(raw) - offset) // 8} remain in the file")
        groups[name] = {"rank": rank, "shape": [n1, n2, n3][:rank],
                        "doubles": count}
        offset += 8 * count
    _require(offset == len(raw),
             f"{path.name}: {len(raw) - offset} trailing bytes after the last "
             "group; the file does not end on a group boundary")
    missing = [name for name in _RHSTERM_GROUPS if name not in groups]
    _require(not missing, f"{path.name}: missing group(s) {missing}")
    declared = header[10]
    _require(declared == len(groups),
             f"{path.name}: the header declares {declared} groups and the "
             f"file carries {len(groups)}")
    return {"file": path.name, "magic": magic, "header": header,
            "nx": header[6], "ny": header[7], "nz": header[8], "ntr": 0,
            "groups": groups,
            "doubles": sum(g["doubles"] for g in groups.values())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="the INSTRUMENTED run directory")
    parser.add_argument("--reference-dir", required=True, type=Path,
                        help="the UN-instrumented reference run directory")
    parser.add_argument("--restart", required=True,
                        help="restart file name common to both runs")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--rhs-terms", action="store_true",
                        help="also require round 4's per-term momentum "
                             "records, one per instrumented boundary")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true",
                        help="corrupt one parsed header; MUST exit non-zero")
    args = parser.parse_args(argv)

    report = {"run_dir": str(args.run_dir), "records": [], "plant": args.plant}
    try:
        # 1. Passivity (note AS): the restart must be byte-identical.
        run_restart = args.run_dir / args.restart
        ref_restart = args.reference_dir / args.restart
        for path in (run_restart, ref_restart):
            _require(path.is_file(), f"missing restart {path}")
        identical = filecmp.cmp(run_restart, ref_restart, shallow=False)
        report["restart_byte_identical"] = identical
        _require(identical,
                 "the step-record writer PERTURBS NEMO: the instrumented and "
                 "reference restarts differ byte for byte")

        # 2. Every record the writer emits must exist and parse.  Naming only
        # the step entries would admit a build whose stage and barotropic
        # writers were silently dropped, which is exactly the "stale build"
        # failure the tanks' acquisitions guard against.
        wanted = [args.run_dir / f"oracle_step_entry_kt{kt:08d}.bin"
                  for kt in range(1, args.steps + 1)]
        wanted += [args.run_dir / f"oracle_bt_frames_kt{kt:08d}.bin"
                   for kt in range(1, args.steps + 1)]
        wanted += [args.run_dir / f"oracle_stage_kt00000001_s{s}.bin"
                   for s in (1, 2, 3)]
        wanted.append(args.run_dir / "oracle_rhs_kt00000001.bin")
        if args.rhs_terms:
            wanted += [args.run_dir / f"oracle_rhsterm_kt00000001_{term}.bin"
                       for term in _RHSTERM_BOUNDARIES]
        for path in wanted:
            _require(path.is_file(), f"the run did not write {path.name}")
        for path in sorted(args.run_dir.glob("oracle_*.bin")):
            report["records"].append(
                parse_record(path, corrupt_header=args.plant
                             and path == wanted[0]))
        _require(len(report["records"]) >= args.steps,
                 "fewer oracle records than the requested ladder")
        # 3. Every step-entry record must agree with the others on geometry.
        entries = [r for r in report["records"]
                   if r["magic"] == "NEMO_L1_ENTRY_1"]
        shapes = {(r["nx"], r["ny"], r["nz"], r["ntr"]) for r in entries}
        _require(len(shapes) == 1,
                 f"step-entry records disagree on geometry: {sorted(shapes)}")
        report["entry_geometry"] = sorted(shapes)[0]
        if args.rhs_terms:
            # Every boundary must be present and must agree with the others
            # on the shape of each group it carries; a writer that dropped
            # one boundary would leave a term silently unmeasured.
            terms = {r["file"]: r for r in report["records"]
                     if r["magic"] == "NEMO_L1_RHSTRM1"}
            _require(len(terms) == len(_RHSTERM_BOUNDARIES),
                     f"{len(terms)} per-term records, expected "
                     f"{len(_RHSTERM_BOUNDARIES)}")
            group_shapes = {
                name: {tuple(r["groups"][name]["shape"]) for r in
                       terms.values()}
                for name in _RHSTERM_GROUPS}
            disagree = {k: sorted(v) for k, v in group_shapes.items()
                        if len(v) != 1}
            _require(not disagree,
                     f"per-term records disagree on group shapes: {disagree}")
            report["rhsterm_group_shapes"] = {
                k: sorted(v)[0] for k, v in group_shapes.items()}
        report["status"] = "ADMITTED"
    except Refusal as error:
        report["status"] = "REFUSED"
        report["reason"] = str(error)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["status"] == "ADMITTED" else 1


if __name__ == "__main__":
    sys.exit(main())
