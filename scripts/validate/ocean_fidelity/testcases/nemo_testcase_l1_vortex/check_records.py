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
    # Round 4's per-term momentum record.  FIFTEEN header integers -- version,
    # step, the four time-level indices, the three extents, the group count,
    # four spare and the word size LAST -- and then (name, rank, n1, n2, n3,
    # payload) groups to end of file.  The count is the writer's, read off
    # the committed instrument rather than assumed: the first draft of this
    # line said sixteen because a comment in the writer said so, and the
    # acquisition's admission caught it by reading a group NAME where it
    # expected the word size.  Nothing about the record's SIZE is written
    # down here; every payload is checked against its own declared extents.
    "oracle_rhsterm_kt": ("NEMO_L1_RHSTRM1", 15, "groups", 0),
    # Round 8's stage-2/3 momentum-term record.  Fifteen header integers:
    # version, step, stage, four time-level indices, three extents, group
    # count, three spare integers and word size.  Payload sizes remain wholly
    # self-described by the groups that follow.
    "oracle_stage_terms_kt": ("NEMO_L1_STGTRM1", 15, "groups", 0),
    # Round 196's per-substep barotropic (dyn_spg_ts) record.  Fifteen header
    # integers: version, step, the four time-level indices, the three
    # extents, the substep count icycle, the four interior-domain bounds and
    # the word size LAST.  It carries NO group count: the number of groups is
    # a function of icycle, which the header already states, so writing it
    # twice would be a second thing to keep in step.  Every payload remains
    # self-described by its own (rank, n1, n2, n3).
    "oracle_spgts_kt": ("NEMO_L1_SPGTS1", 15, "groups", 0),
    # Round 200's FLUX-card stage record.  Same header shape as round 192's
    # (fifteen integers after the magic) and the same self-describing group
    # stream, but it opens at EVERY stage and carries the advective
    # transports the flux-form advection call consumes.
    "oracle_stage_flux_terms_kt": ("NEMO_L1_STGFLX1", 15, "groups", 0),
    # Round 218's TRACER-term record (VORTEX_SMT round 7).  Same header
    # shape and the same self-describing group stream; it opens at every
    # stage and carries the tracer path's operands and outputs.
    "oracle_tracer_terms_kt": ("NEMO_L1_TRATRM1", 15, "groups", 0),
}
# The groups every per-term record must carry, by name.  This list, the magic
# and the format version are the ONLY hard-coded expectations.
_RHSTERM_GROUPS = ("uu_rhs", "vv_rhs", "ww", "r3t_Kaa")
# The boundaries stp_2D is instrumented at, in NEMO's own execution order;
# the per-term increments are the differences between consecutive records.
_RHSTERM_BOUNDARIES = ("hpg", "ldf", "vor", "wzv", "keg", "zad")
_STAGE_TERM_COMMON = (
    "kmm_u", "kmm_v", "ssh_kmm", "ww",
    "base_u", "base_v", "hpg_u", "hpg_v", "vor_u", "vor_v",
    "keg_u", "keg_v", "zad_u", "zad_v", "out_u", "out_v",
)
_STAGE_TERM_BY_STAGE = {
    2: _STAGE_TERM_COMMON + ("update_u", "update_v"),
    3: _STAGE_TERM_COMMON + ("ldf_u", "ldf_v", "zdf_u", "zdf_v"),
}
# Round 200's FLUX-card stage record.  Stage 1 runs ONE momentum statement
# (the flux-form advection call); stages 2 and 3 add HPG and VOR before it,
# and stage 3 replaces the explicit update with LDF + the implicit ZDF
# integration.  The transports zfu/zfv/zfw are the advection call's operands.
_STAGE_FLUX_COMMON = (
    "kmm_u", "kmm_v", "ssh_kmm", "ww", "zfu", "zfv", "zfw",
    "base_u", "base_v", "adv_u", "adv_v", "out_u", "out_v",
)
_STAGE_FLUX_BY_STAGE = {
    1: _STAGE_FLUX_COMMON + ("update_u", "update_v"),
    2: _STAGE_FLUX_COMMON + ("hpg_u", "hpg_v", "vor_u", "vor_v",
                             "update_u", "update_v"),
    3: _STAGE_FLUX_COMMON + ("hpg_u", "hpg_v", "vor_u", "vor_v",
                             "ldf_u", "ldf_v", "zdf_u", "zdf_v"),
}

# Round 218's tracer-term record.  The same fifteen groups at every stage:
# the three advective transports as tra_adv receives them, the cross-level
# velocity, the before/now tracer fields, the three surface-ratio time
# levels, the tracer right-hand side after advection + the surface boundary
# condition, and the after-tracer at the end of the stage.
_TRACER_TERM_GROUPS = (
    "zfu", "zfv", "zfw", "ww",
    "tsb_t", "tsb_s", "tsm_t", "tsm_s",
    "r3t_kbb", "r3t_kmm", "r3t_kaa",
    "adv_t", "adv_s", "out_t", "out_s",
)
_SMT3_TRACER_TERM_BY_STAGE = {
    1: _TRACER_TERM_GROUPS,
    2: _TRACER_TERM_GROUPS,
    3: _TRACER_TERM_GROUPS + ("ldf_t", "ldf_s"),
}


# Round 196's per-substep barotropic record.  Three frame kinds: the
# loop-entry frame 'i000_', one frame 'jNNN_' per sub-time-step, and the
# loop-exit frame 'o000_'.  These names, the magic and the format version
# are the ONLY hard-coded expectations; sizes are never predicted.
_SPGTS_ENTRY = (
    "ssh_frc", "zu_frc", "zv_frc", "un_e", "vn_e", "ub_e", "vb_e",
    "ubb_e", "vbb_e", "sshn_e", "sshb_e", "sshbb_e",
    "hu_e", "hv_e", "hur_e", "hvr_e", "zCdU_u", "zCdU_v",
    "wgtbtp1", "wgtbtp2", "entry_sc",
)
_SPGTS_SUBSTEP = (
    "ua_ext", "va_ext", "sshp2_mid", "htp2_e", "hup2_e", "hvp2_e", "ext_coef",
    "zhU", "zhV", "ssha_e", "un_adv", "vn_adv", "sshu_a", "sshv_a",
    "sshp2_bck", "zu_spg", "zv_spg", "bck_coef", "cor_u", "cor_v",
    "trd_u", "trd_v", "ua_new", "va_new", "hu_e", "hv_e", "hur_e", "hvr_e",
    "uub_sum", "vvb_sum", "ssh_sum", "sum_coef",
)
_SPGTS_EXIT = ("un_adv", "vn_adv", "uu_b_aa", "vv_b_aa", "ssh_aa")
_SPGTS_PLANTS = ("header", "field-name", "truncated", "missing-frame")


class Refusal(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise Refusal(message)


def parse_record(path: Path, corrupt_header: bool = False,
                 plant: str | None = None) -> dict:
    """Parse one record from its own header.  Raises Refusal on any mismatch."""
    family = next(
        (name for name in _FAMILIES if path.name.startswith(name)), None)
    _require(family is not None, f"{path.name}: unknown record family")
    magic_expected, n_header, layout, n_2d = _FAMILIES[family]
    raw = path.read_bytes()
    if plant == "truncated":
        # Lose the last eight bytes: the file must then stop reading mid-group
        # and the group walk must say so rather than silently accepting it.
        raw = raw[:-8]
    if plant == "header":
        corrupt_header = True
    _require(len(raw) > 16 + 4 * n_header, f"{path.name}: truncated")
    magic = raw[:16].decode("ascii", "replace").rstrip()
    header = list(struct.unpack(f"={n_header}i", raw[16:16 + 4 * n_header]))
    if corrupt_header and layout != "groups":
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
        return _parse_groups(path, raw, magic, header, n_header, family,
                             corrupt_header, plant)
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
                  n_header: int, family: str, corrupt_extent: bool,
                  plant: str | None = None) -> dict:
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
        if corrupt_extent and not groups:
            n1 += 1
        offset += 32
        # Rank 1 exists only in the round-196 barotropic record, which
        # carries the two time-filter weight vectors and the per-substep
        # coefficients; the older families are 2-D and 3-D only, and
        # accepting a rank they never write would weaken their guard.
        allowed = (1, 2, 3) if family == "oracle_spgts_kt" else (2, 3)
        _require(rank in allowed,
                 f"{path.name}: group {name!r} has rank {rank}")
        _require(min(n1, n2, n3) > 0,
                 f"{path.name}: group {name!r} declares a nonpositive extent")
        _require(name not in groups,
                 f"{path.name}: duplicate group name {name!r}")
        count = n1 * (n2 if rank >= 2 else 1) * (n3 if rank == 3 else 1)
        _require(offset + 8 * count <= len(raw),
                 f"{path.name}: group {name!r} declares {count} doubles, but "
                 f"only {(len(raw) - offset) // 8} remain in the file")
        groups[name] = {"rank": rank, "shape": [n1, n2, n3][:rank],
                        "doubles": count}
        offset += 8 * count
    _require(offset == len(raw),
             f"{path.name}: {len(raw) - offset} trailing bytes after the last "
             "group; the file does not end on a group boundary")
    if family == "oracle_spgts_kt":
        return _finish_spgts(path, magic, header, groups, plant)
    if plant == "field-name":
        victim = "ldf_t" if "ldf_t" in groups else next(iter(groups))
        groups[f"{victim}_plant"] = groups.pop(victim)
    if family == "oracle_rhsterm_kt":
        required = _RHSTERM_GROUPS
        declared_index = 9
        stage = None
    elif family == "oracle_stage_flux_terms_kt":
        stage = header[2]
        _require(stage in _STAGE_FLUX_BY_STAGE,
                 f"{path.name}: unsupported stage {stage}")
        required = _STAGE_FLUX_BY_STAGE[stage]
        declared_index = 10
    elif family == "oracle_tracer_terms_kt":
        stage = header[2]
        _require(stage in (1, 2, 3),
                 f"{path.name}: unsupported stage {stage}")
        if stage == 3 and header[10] == len(_SMT3_TRACER_TERM_BY_STAGE[3]):
            required = _SMT3_TRACER_TERM_BY_STAGE[3]
        else:
            required = _TRACER_TERM_GROUPS
        declared_index = 10
    else:
        stage = header[2]
        _require(stage in _STAGE_TERM_BY_STAGE,
                 f"{path.name}: unsupported stage {stage}")
        required = _STAGE_TERM_BY_STAGE[stage]
        declared_index = 10
    missing = [name for name in required if name not in groups]
    _require(not missing, f"{path.name}: missing group(s) {missing}")
    unexpected = [name for name in groups if name not in required]
    _require(not unexpected, f"{path.name}: unexpected group(s) {unexpected}")
    declared = header[declared_index]
    _require(declared == len(groups),
             f"{path.name}: the header declares {declared} groups and the "
             f"file carries {len(groups)}")
    if family == "oracle_rhsterm_kt":
        nx, ny, nz = header[6:9]
    else:
        nx, ny, nz = header[7:10]
    return {"file": path.name, "magic": magic, "header": header,
            "nx": nx, "ny": ny, "nz": nz, "ntr": 0,
            "stage": stage,
            "groups": groups,
            "doubles": sum(g["doubles"] for g in groups.values())}


def _finish_spgts(path: Path, magic: str, header: list, groups: dict,
                  plant: str | None) -> dict:
    """Verify the frame structure the record's OWN header implies.

    The header states icycle; the file must therefore carry the loop-entry
    frame, exactly icycle substep frames and the loop-exit frame, each
    complete.  Nothing about a size is written down here.
    """
    icycle = header[9]
    _require(icycle > 0, f"{path.name}: header declares icycle {icycle}")
    if plant == "field-name":
        # Rename one required operand: a record that lost a field under a
        # typo must be refused, not quietly scored with the field missing.
        victim = f"j001_zhU"
        _require(victim in groups,
                 f"{path.name}: plant target {victim!r} is absent")
        groups[victim.replace("zhU", "zhX")] = groups.pop(victim)
    if plant == "missing-frame":
        for name in [n for n in groups if n.startswith(f"j{icycle:03d}_")]:
            groups.pop(name)
    frames = {"i000": _SPGTS_ENTRY, "o000": _SPGTS_EXIT}
    for jn in range(1, icycle + 1):
        frames[f"j{jn:03d}"] = _SPGTS_SUBSTEP
    missing, unexpected = [], []
    seen = set()
    for prefix, required in frames.items():
        for name in required:
            key = f"{prefix}_{name}"
            seen.add(key)
            if key not in groups:
                missing.append(key)
    unexpected = [name for name in groups if name not in seen]
    _require(not missing,
             f"{path.name}: missing group(s) {missing[:6]} "
             f"({len(missing)} in total)")
    _require(not unexpected,
             f"{path.name}: unexpected group(s) {unexpected[:6]} "
             f"({len(unexpected)} in total)")
    nx, ny, nz = header[6:9]
    return {"file": path.name, "magic": magic, "header": header,
            "nx": nx, "ny": ny, "nz": nz, "ntr": 0, "stage": None,
            "icycle": icycle, "frames": len(frames),
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
    parser.add_argument("--stage-terms", action="store_true",
                        help="also require round 8's stage-2 and stage-3 "
                             "momentum term records")
    parser.add_argument("--stage-flux-terms", action="store_true",
                        help="also require round 200's flux-card stage-1, "
                             "stage-2 and stage-3 momentum term records")
    parser.add_argument("--tracer-terms", action="store_true",
                        help="also admit round 218's per-stage tracer-term "
                             "record")
    parser.add_argument("--smt3-tracer-terms", action="store_true",
                        help="also admit round 224's per-stage tracer record; "
                             "stage 3 must include the post-LDF boundary")
    parser.add_argument("--spgts-terms", action="store_true",
                        help="also require round 196's per-substep "
                             "barotropic (dyn_spg_ts) records")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", nargs="?", const="header",
                        choices=_SPGTS_PLANTS,
                        help="corrupt one parsed record; MUST exit non-zero. "
                             "'header' bumps a declared extent, 'field-name' "
                             "renames a required operand, 'truncated' drops "
                             "the file's last eight bytes and 'missing-frame' "
                             "deletes the last substep frame")
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
        if args.stage_terms:
            wanted += [args.run_dir /
                       f"oracle_stage_terms_kt00000001_s{stage}.bin"
                       for stage in (2, 3)]
        if args.stage_flux_terms:
            wanted += [args.run_dir /
                       f"oracle_stage_flux_terms_kt00000001_s{stage}.bin"
                       for stage in (1, 2, 3)]
        if args.tracer_terms or args.smt3_tracer_terms:
            wanted += [args.run_dir /
                       f"oracle_tracer_terms_kt00000001_s{stage}.bin"
                       for stage in (1, 2, 3)]
        if args.spgts_terms:
            wanted += [args.run_dir / f"oracle_spgts_kt{kt:08d}.bin"
                       for kt in range(1, args.steps + 1)]
        for path in wanted:
            _require(path.is_file(), f"the run did not write {path.name}")
        if args.spgts_terms:
            corrupt_path = args.run_dir / "oracle_spgts_kt00000001.bin"
        elif args.tracer_terms or args.smt3_tracer_terms:
            corrupt_path = (args.run_dir /
                            "oracle_tracer_terms_kt00000001_s1.bin")
        elif args.stage_flux_terms:
            corrupt_path = (args.run_dir /
                            "oracle_stage_flux_terms_kt00000001_s1.bin")
        elif args.stage_terms:
            corrupt_path = wanted[-1]
        else:
            corrupt_path = wanted[0]
        for path in sorted(args.run_dir.glob("oracle_*.bin")):
            planted = args.plant if path == corrupt_path else None
            report["records"].append(
                parse_record(path, corrupt_header=bool(planted)
                             and planted == "header", plant=planted))
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
        if args.stage_terms:
            stage_records = [r for r in report["records"]
                             if r["magic"] == "NEMO_L1_STGTRM1"]
            stages = {r["stage"] for r in stage_records}
            _require(stages == {2, 3},
                     f"stage-term records cover {sorted(stages)}, expected [2, 3]")
            _require(len(stage_records) == 2,
                     f"{len(stage_records)} stage-term records, expected 2")
            for record in stage_records:
                stage = record["stage"]
                groups = record["groups"]
                for name, meta in groups.items():
                    if name == "ssh_kmm":
                        _require(meta["rank"] == 2,
                                 f"stage {stage} {name} is not rank 2")
                    else:
                        _require(meta["rank"] == 3,
                                 f"stage {stage} {name} is not rank 3")
            report["stage_term_groups"] = {
                str(r["stage"]): sorted(r["groups"])
                for r in stage_records}
        if args.stage_flux_terms:
            flux_records = [r for r in report["records"]
                            if r["magic"] == "NEMO_L1_STGFLX1"]
            stages = {r["stage"] for r in flux_records}
            _require(stages == {1, 2, 3},
                     f"flux stage-term records cover {sorted(stages)}, "
                     "expected [1, 2, 3]")
            _require(len(flux_records) == 3,
                     f"{len(flux_records)} flux stage-term records, expected 3")
            for record in flux_records:
                for name, meta in record["groups"].items():
                    want = 2 if name == "ssh_kmm" else 3
                    _require(meta["rank"] == want,
                             f"stage {record['stage']} {name} is not "
                             f"rank {want}")
            report["stage_flux_term_groups"] = {
                str(r["stage"]): sorted(r["groups"]) for r in flux_records}
        if args.tracer_terms or args.smt3_tracer_terms:
            tra_records = [r for r in report["records"]
                           if r["magic"] == "NEMO_L1_TRATRM1"]
            stages = {r["stage"] for r in tra_records}
            _require(stages == {1, 2, 3},
                     f"tracer-term records cover {sorted(stages)}, "
                     "expected [1, 2, 3]")
            _require(len(tra_records) == 3,
                     f"{len(tra_records)} tracer-term records, expected 3")
            if args.smt3_tracer_terms:
                for record in tra_records:
                    required = _SMT3_TRACER_TERM_BY_STAGE[record["stage"]]
                    _require(set(record["groups"]) == set(required),
                             f"stage {record['stage']} SMT-3 tracer groups "
                             f"differ: {sorted(record['groups'])}")
            for record in tra_records:
                for name, meta in record["groups"].items():
                    want = 2 if name.startswith("r3t_") else 3
                    _require(meta["rank"] == want,
                             f"stage {record['stage']} {name} is not "
                             f"rank {want}")
            report["tracer_term_groups"] = {
                str(r["stage"]): sorted(r["groups"]) for r in tra_records}
        if args.spgts_terms:
            spgts = [r for r in report["records"]
                     if r["magic"] == "NEMO_L1_SPGTS1"]
            _require(len(spgts) == args.steps,
                     f"{len(spgts)} barotropic substep records, expected "
                     f"{args.steps}")
            cycles = {r["icycle"] for r in spgts}
            _require(len(cycles) == 1,
                     f"the substep records disagree on icycle: {sorted(cycles)}")
            report["spgts_icycle"] = sorted(cycles)[0]
            report["spgts_frames_per_step"] = sorted(
                {r["frames"] for r in spgts})
            report["spgts_groups_per_step"] = sorted(
                {len(r["groups"]) for r in spgts})
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
