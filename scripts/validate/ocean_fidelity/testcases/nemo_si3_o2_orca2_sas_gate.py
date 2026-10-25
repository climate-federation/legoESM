#!/usr/bin/env python3
"""Committed validation, layout-fixture, log-check, and admission gate for SI3 Rung O2 (`si3_r7`).

Covers (`DECISION 153`, `DECISION 159`, `docs/ice/fidelity/si3_r7_o2_preregister.md`):
- Extends Rung O1's gate (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py`)
  by importing its binary snapshot readers/stitchers, wet-masking helpers, dry `-0.0` fixture,
  log verifier, and state comparators.
- Reads and stitches the O2 initial-state snapshot (`O2SI3I01`, `o2_si3_istate_rankXXXX.bin`)
  written by `o2_si3_state_writer.F90` right after NEMO builds the scalar initial state and
  levitating-ice initial SSH (`src/ICE/iceistate.F90:304`, `:427`).
- Computes the category fit through rung O3 (`si3_o3_istate_fill_fields`),
  then scores O2's post-fit sums, mass, and levitating-ice SSH statements
  under the glibc-`exp` certification arm (`transcendentals="libm"`).
- Verifies all negative-control plants P1-P7.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.config import build_si3_orca2_o2_config
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.si3_orca2_state import (
    SI3AllPratherMoments,
    SI3O2InitConfig,
    SI3O2ScalarInitialState,
    SI3ORCA2O2Forcing,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    read_si3_netcdf_restart,
    si3_dom_uniq_t_orca2,
    si3_eos_fzp,
    si3_glob_2dsum_ddpdd,
    si3_hfn_category_bounds,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_o2_scalar_initial_state,
    si3_o2_step,
    si3_rst_to_netcdf_dict,
    write_si3_netcdf_restart,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    BUILD_CONFIG_FILES,
    GateError as SameProgramError,
    same_program_observers,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import (
    _FIXTURE_FIELD_OFFSETS,
    _PROG_FIELDS,
    _compare_si3_states_wet_and_full,
    _expected_fixture_field,
    _verify_run_completion_and_logs,
    _verify_sha256_manifest,
    build_synthetic_orca2_o1_state,
    first_dry_ij,
    first_wet_ij,
    max_normalized_diff,
    max_ulp_diff,
    o1_snapshot_to_si3_state,
    read_o1_binary_snapshot,
    restart_payload_view,
    stitch_o1_binary_snapshots,
    verify_dry_signed_zero_fixture,
    verify_ocean_output_log,
    wet_mask_for_field,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

WRITER_UNIT = "o2_si3_state_writer"
OBSERVER_UNITS = ("icedyn_adv_pra", "iceistate", "icestp")

_O2_ISTATE_FIELD_SPECS: tuple[tuple[str, float, int, bool], ...] = (
    # Domain & surface inputs (13 x 2D)
    ("tmask", 1.1, 2, False),
    ("umask", 2.1, 2, False),
    ("vmask", 3.1, 2, False),
    ("ssmask", 6.0, 2, False),
    ("smask0", 6.25, 2, True),
    ("smask0_i", 6.5, 2, True),
    ("e1e2t", 7.0, 2, False),
    ("ff_t", 7.5, 2, False),
    ("gphit", 8.0, 2, False),
    ("sst_m", 4.0, 2, False),
    ("sss_m", 5.0, 2, False),
    ("ssh_before_kbb", 8.6, 2, False),
    ("ssh_before_kmm", 8.7, 2, False),
    # Pre-ice_var_itd O2 scalar initial state (14 x 2D)
    ("t_bo", 9.0, 2, True),
    ("ztfrz", 140.0, 2, False),
    ("zmsk", 141.0, 2, False),
    ("zht_i_ini", 142.0, 2, False),
    ("zht_s_ini", 143.0, 2, False),
    ("zat_i_ini", 144.0, 2, False),
    ("zsm_i_ini", 145.0, 2, False),
    ("ztm_i_ini", 146.0, 2, False),
    ("zt_su_ini", 147.0, 2, False),
    ("ztm_s_ini", 148.0, 2, False),
    ("zapnd_ini", 149.0, 2, False),
    ("zhpnd_ini", 150.0, 2, False),
    ("zhlid_ini", 151.0, 2, False),
    ("rn_amax_2d", 9.5, 2, False),
    # Post-ice_istate aggregated volumes, snow/ice mass, and levitating-ice SSH (10 x 2D)
    ("vt_i", 10.0, 3, False),
    ("vt_s", 11.0, 3, False),
    ("at_i", 112.0, 2, False),
    ("at_ip", 18.0, 3, False),
    ("vt_ip", 19.0, 3, False),
    ("vt_il", 20.0, 3, False),
    ("snwice_mass", 26.0, 2, False),
    ("snwice_mass_b", 27.0, 2, False),
    ("ssh_kbb", 160.1, 2, False),
    ("ssh_kmm", 160.2, 2, False),
)

_O2_SCALAR_STATE_FIELDS: tuple[str, ...] = (
    "t_bo",
    "ztfrz",
    "zmsk",
    "zht_i_ini",
    "zht_s_ini",
    "zat_i_ini",
    "zsm_i_ini",
    "ztm_i_ini",
    "zt_su_ini",
    "ztm_s_ini",
    "zapnd_ini",
    "zhpnd_ini",
    "zhlid_ini",
    "rn_amax_2d",
    "vt_i",
    "vt_s",
    "at_i",
    "at_ip",
    "vt_ip",
    "vt_il",
    "snwice_mass",
    "snwice_mass_b",
    "ssh_kbb",
    "ssh_kmm",
)


def read_o2_istate_snapshot(path: str | Path) -> dict[str, Any]:
    """Read one rank-local `o2_si3_istate_rankXXXX.bin` file written by `o2_si3_state_writer.F90`."""
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < 308:
        raise ValueError(f"Truncated O2 istate header in {p}: {len(raw)} bytes < 308")

    magic = raw[:8].decode("ascii", errors="replace")
    if magic != "O2SI3I01":
        raise ValueError(f"Invalid O2 istate magic {magic!r} in {p}, expected 'O2SI3I01'")
    version = int(np.frombuffer(raw, dtype="<i4", count=1, offset=8)[0])
    if version != 1:
        raise ValueError(f"Unsupported O2 istate version {version} in {p}")

    i_meta = np.frombuffer(raw, dtype="<i4", count=22, offset=12)
    r_meta = np.frombuffer(raw, dtype="<f8", count=26, offset=100)
    jpi = int(i_meta[3])
    jpj = int(i_meta[4])

    meta = {
        "kt": int(i_meta[0]),
        "mpprank": int(i_meta[1]),
        "narea": int(i_meta[2]),
        "jpi": jpi,
        "jpj": jpj,
        "jpl": int(i_meta[5]),
        "nlay_i": int(i_meta[6]),
        "nlay_s": int(i_meta[7]),
        "nn_hls": int(i_meta[8]),
        "Ni0glo": int(i_meta[9]),
        "Nj0glo": int(i_meta[10]),
        "nimpp": int(i_meta[11]),
        "njmpp": int(i_meta[12]),
        "Nis0": int(i_meta[13]),
        "Nie0": int(i_meta[14]),
        "Njs0": int(i_meta[15]),
        "Nje0": int(i_meta[16]),
        "Kbb": int(i_meta[17]),
        "Kmm": int(i_meta[18]),
        "nn_iceini_file": int(i_meta[19]),
        "ln_iceini": int(i_meta[20]),
        "ln_ice_embd": int(i_meta[21]),
        "rn_thres_sst": float(r_meta[0]),
        "rn_hti_ini_n": float(r_meta[1]),
        "rn_hti_ini_s": float(r_meta[2]),
        "rn_hts_ini_n": float(r_meta[3]),
        "rn_hts_ini_s": float(r_meta[4]),
        "rn_ati_ini_n": float(r_meta[5]),
        "rn_ati_ini_s": float(r_meta[6]),
        "rn_smi_ini_n": float(r_meta[7]),
        "rn_smi_ini_s": float(r_meta[8]),
        "rn_tmi_ini_n": float(r_meta[9]),
        "rn_tmi_ini_s": float(r_meta[10]),
        "rn_tsu_ini_n": float(r_meta[11]),
        "rn_tsu_ini_s": float(r_meta[12]),
        "rn_tms_ini_n": float(r_meta[13]),
        "rn_tms_ini_s": float(r_meta[14]),
        "rn_apd_ini_n": float(r_meta[15]),
        "rn_apd_ini_s": float(r_meta[16]),
        "rn_hpd_ini_n": float(r_meta[17]),
        "rn_hpd_ini_s": float(r_meta[18]),
        "rn_hld_ini_n": float(r_meta[19]),
        "rn_hld_ini_s": float(r_meta[20]),
        "rn_amax_n": float(r_meta[21]),
        "rn_amax_s": float(r_meta[22]),
        "rt0": float(r_meta[23]),
        "area": float(r_meta[24]),
        "zsshadj": float(r_meta[25]),
    }

    offset = 308
    shape = (jpi, jpj)
    count = jpi * jpj
    nbytes = count * 8
    fields: dict[str, np.ndarray] = {}
    for fname, *_ in _O2_ISTATE_FIELD_SPECS:
        if offset + nbytes > len(raw):
            raise ValueError(
                f"Truncated O2 istate payload at field {fname} in {p}: need {offset + nbytes}, got {len(raw)}"
            )
        arr = np.frombuffer(raw, dtype="<f8", count=count, offset=offset).reshape(
            shape, order="F"
        ).copy()
        fields[fname] = arr
        offset += nbytes

    if offset != len(raw):
        raise ValueError(
            f"Trailing bytes in {p}: consumed {offset}, file size {len(raw)}"
        )

    return {
        "magic": magic,
        "version": version,
        "meta": meta,
        "fields": fields,
    }


def stitch_o2_istate_snapshots(snaps: list[dict[str, Any]]) -> dict[str, Any]:
    """Stitch rank-local O2 `istate` binary snapshots onto the global `(Ni0glo + 2*nn_hls, Nj0glo + 2*nn_hls)` grid."""
    if not snaps:
        raise ValueError("Cannot stitch empty O2 istate snapshot list")
    if len(snaps) == 1:
        return copy.deepcopy(snaps[0])

    meta0 = copy.deepcopy(snaps[0]["meta"])
    nn_hls = int(meta0["nn_hls"])
    ni0glo = int(meta0["Ni0glo"])
    nj0glo = int(meta0["Nj0glo"])
    g_jpi = ni0glo + 2 * nn_hls
    g_jpj = nj0glo + 2 * nn_hls

    stitched_fields: dict[str, np.ndarray] = {
        fname: np.zeros((g_jpi, g_jpj), dtype=np.float64)
        for fname, *_ in _O2_ISTATE_FIELD_SPECS
    }

    for snap in snaps:
        m = snap["meta"]
        nis0, nie0 = int(m["Nis0"]), int(m["Nie0"])
        njs0, nje0 = int(m["Njs0"]), int(m["Nje0"])
        gi0 = int(m["nimpp"]) + nis0 - 2
        gi1 = int(m["nimpp"]) + nie0 - 1
        gj0 = int(m["njmpp"]) + njs0 - 2
        gj1 = int(m["njmpp"]) + nje0 - 1
        if gi0 < 0 or gj0 < 0 or gi1 > g_jpi or gj1 > g_jpj:
            raise ValueError(
                f"Subdomain slice [{gi0}:{gi1}, {gj0}:{gj1}] out of global bounds ({g_jpi}, {g_jpj})"
            )
        for fname in stitched_fields:
            stitched_fields[fname][gi0:gi1, gj0:gj1] = snap["fields"][fname][
                nis0 - 1 : nie0, njs0 - 1 : nje0
            ]

    meta0["jpi"] = g_jpi
    meta0["jpj"] = g_jpj
    meta0["nimpp"] = 1
    meta0["njmpp"] = 1
    meta0["Nis0"] = 1 + nn_hls
    meta0["Nie0"] = g_jpi - nn_hls
    meta0["Njs0"] = 1 + nn_hls
    meta0["Nje0"] = g_jpj - nn_hls
    return {
        "magic": snaps[0]["magic"],
        "version": snaps[0]["version"],
        "meta": meta0,
        "fields": stitched_fields,
    }


def o2_istate_snapshot_to_config(
    snap: dict[str, Any],
    *,
    ln_teos10: bool = True,
    ln_eos80: bool = False,
) -> SI3O2InitConfig:
    """Build `SI3O2InitConfig` from a stitched `O2SI3I01` snapshot's header metadata."""
    m = snap["meta"]
    return SI3O2InitConfig(
        ln_iceini=bool(m["ln_iceini"]),
        nn_iceini_file=int(m["nn_iceini_file"]),
        rn_thres_sst=float(m["rn_thres_sst"]),
        rn_hti_ini_n=float(m["rn_hti_ini_n"]),
        rn_hti_ini_s=float(m["rn_hti_ini_s"]),
        rn_hts_ini_n=float(m["rn_hts_ini_n"]),
        rn_hts_ini_s=float(m["rn_hts_ini_s"]),
        rn_ati_ini_n=float(m["rn_ati_ini_n"]),
        rn_ati_ini_s=float(m["rn_ati_ini_s"]),
        rn_smi_ini_n=float(m["rn_smi_ini_n"]),
        rn_smi_ini_s=float(m["rn_smi_ini_s"]),
        rn_tmi_ini_n=float(m["rn_tmi_ini_n"]),
        rn_tmi_ini_s=float(m["rn_tmi_ini_s"]),
        rn_tsu_ini_n=float(m["rn_tsu_ini_n"]),
        rn_tsu_ini_s=float(m["rn_tsu_ini_s"]),
        rn_tms_ini_n=float(m["rn_tms_ini_n"]),
        rn_tms_ini_s=float(m["rn_tms_ini_s"]),
        rn_apd_ini_n=float(m["rn_apd_ini_n"]),
        rn_apd_ini_s=float(m["rn_apd_ini_s"]),
        rn_hpd_ini_n=float(m["rn_hpd_ini_n"]),
        rn_hpd_ini_s=float(m["rn_hpd_ini_s"]),
        rn_hld_ini_n=float(m["rn_hld_ini_n"]),
        rn_hld_ini_s=float(m["rn_hld_ini_s"]),
        rn_amax_n=float(m["rn_amax_n"]),
        rn_amax_s=float(m["rn_amax_s"]),
        ln_ice_embd=bool(m["ln_ice_embd"]),
        ln_rstart=False,
        ln_teos10=bool(ln_teos10 and not ln_eos80),
    )


def same_program(
    new_bld: Path,
    ref_bld: Path,
    pra_patch: Path,
    ist_patch: Path,
    stp_patch: Path,
    *,
    new_cfg: str = "ORCA2_SAS_ICE_O2_REC",
    ref_cfg: str = "ORCA2_SAS_ICE_O2_REF",
) -> dict[str, Any]:
    """Adapt the landed shared ORCA2 same-program proof for O2's observers."""
    try:
        report = same_program_observers(
            new_bld,
            ref_bld,
            dict(
                zip(
                    OBSERVER_UNITS,
                    (pra_patch, ist_patch, stp_patch),
                    strict=True,
                )
            ),
            (WRITER_UNIT,),
            new_cfg=new_cfg,
            ref_cfg=ref_cfg,
        )
    except SameProgramError as exc:
        raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
    return {
        **report,
        "status": "PASS_SAME_PROGRAM",
        "reference_bld": str(ref_bld),
        "new_bld": str(new_bld),
    }


def _expected_o2_istate_fixture_field(
    jpi: int,
    jpj: int,
    nis0: int,
    nie0: int,
    njs0: int,
    nje0: int,
    offset: float,
    src_ndim: int,
    jpl: int = 5,
    is_a2d: bool = False,
) -> np.ndarray:
    """Reproduce the exact rank/field pattern written by `o2_si3_layout_fixture.F90` for `O2SI3I01`."""
    if src_ndim == 2:
        return _expected_fixture_field(
            jpi=jpi,
            jpj=jpj,
            nis0=nis0,
            nie0=nie0,
            njs0=njs0,
            nje0=nje0,
            offset=offset,
            ndim=2,
            jpl=jpl,
            is_a2d=is_a2d,
        )
    if src_ndim == 3:
        arr_3d = _expected_fixture_field(
            jpi=jpi,
            jpj=jpj,
            nis0=nis0,
            nie0=nie0,
            njs0=njs0,
            nje0=nje0,
            offset=offset,
            ndim=3,
            jpl=jpl,
            is_a2d=False,
        )
        acc = np.zeros((jpi, jpj), dtype=np.float64)
        for jl in range(jpl):
            acc = acc + arr_3d[:, :, jl]
        return acc
    raise ValueError(f"Unsupported src_ndim={src_ndim}")


def verify_layout_fixture(
    records_dir: str | Path,
    plant: str | None = None,
) -> dict[str, Any]:
    """Verify the 4-rank compiled `gfortran` layout fixture for both `O2SI3I01` and `O1SI3R01`."""
    rdir = Path(records_dir)
    expected_istate_bytes = 26948
    expected_state_bytes = 868548
    istate_paths = [rdir / f"o2_si3_istate_rank{r:04d}.bin" for r in range(4)]
    entry_paths = [rdir / f"o2_si3_state_entry_rank{r:04d}.bin" for r in range(4)]
    step1_paths = [rdir / f"o2_si3_state_step0001_rank{r:04d}.bin" for r in range(4)]

    if plant == "missing-frame":
        istate_paths = istate_paths[:-1]

    if len(istate_paths) != 4 or len(entry_paths) != 4 or len(step1_paths) != 4:
        raise ValueError("STATUS PLANT-FIRED: missing rank file in O2 layout fixture")

    for p in istate_paths:
        if not p.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing expected istate fixture file {p}")
        if p.stat().st_size != expected_istate_bytes:
            raise ValueError(
                f"STATUS PLANT-FIRED: istate file size {p.stat().st_size} != {expected_istate_bytes}"
            )
    for p in entry_paths + step1_paths:
        if not p.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing expected state fixture file {p}")
        if p.stat().st_size != expected_state_bytes:
            raise ValueError(
                f"STATUS PLANT-FIRED: state file size {p.stat().st_size} != {expected_state_bytes}"
            )

    if plant == "truncation":
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            bad_file = Path(tmp) / "truncated_istate.bin"
            bad_file.write_bytes(istate_paths[0].read_bytes()[:-16])
            try:
                read_o2_istate_snapshot(bad_file)
            except ValueError as exc:
                raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
            raise RuntimeError("Truncation plant did not fire")

    if plant == "bad-magic":
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            bad_file = Path(tmp) / "bad_magic_istate.bin"
            raw = bytearray(istate_paths[0].read_bytes())
            raw[:8] = b"BADMAGIC"
            bad_file.write_bytes(bytes(raw))
            try:
                read_o2_istate_snapshot(bad_file)
            except ValueError as exc:
                raise ValueError(f"STATUS PLANT-FIRED: {exc}") from exc
            raise RuntimeError("Bad-magic plant did not fire")

    snaps_istate = [read_o2_istate_snapshot(p) for p in istate_paths]
    snaps_entry = [read_o1_binary_snapshot(p) for p in entry_paths]
    snaps_step1 = [read_o1_binary_snapshot(p) for p in step1_paths]

    if plant == "rank-collapse":
        snaps_istate[1]["fields"]["zht_i_ini"] = snaps_istate[0]["fields"]["zht_i_ini"].copy()
    elif plant == "wrong-offset":
        snaps_istate[2]["meta"]["nimpp"] += 1
    elif plant == "field-swap":
        tmp_arr = snaps_istate[0]["fields"]["zht_i_ini"].copy()
        snaps_istate[0]["fields"]["zht_i_ini"] = snaps_istate[0]["fields"]["zht_s_ini"].copy()
        snaps_istate[0]["fields"]["zht_s_ini"] = tmp_arr

    # Verify rank uniqueness across all 4 ranks for both istate and state
    for r in range(4):
        for s in range(r + 1, 4):
            if np.allclose(
                snaps_istate[r]["fields"]["zht_i_ini"],
                snaps_istate[s]["fields"]["zht_i_ini"],
            ):
                raise ValueError(
                    f"STATUS PLANT-FIRED: ranks {r} and {s} have identical zht_i_ini data"
                )
            if np.allclose(
                snaps_entry[r]["fields"]["v_i"], snaps_entry[s]["fields"]["v_i"]
            ):
                raise ValueError(
                    f"STATUS PLANT-FIRED: ranks {r} and {s} have identical v_i data"
                )

    bounds = si3_hfn_category_bounds(jpl=5, rn_himean=2.0, opt_hips=0, rn_himax=99.0)
    for irank in range(4):
        ist_snap = snaps_istate[irank]
        ent_snap = snaps_entry[irank]
        meta_i = ist_snap["meta"]
        meta_e = ent_snap["meta"]
        rx = irank % 2
        ry = irank // 2
        if meta_i["mpprank"] != irank or meta_i["narea"] != irank + 1:
            raise ValueError("STATUS PLANT-FIRED: istate rank/narea metadata mismatch")
        if meta_i["nimpp"] != 1 + rx * 6 or meta_i["njmpp"] != 1 + ry * 5:
            raise ValueError("STATUS PLANT-FIRED: istate subdomain nimpp/njmpp offset mismatch")
        if max_ulp_diff(ent_snap["hi_max"], np.asarray(bounds.hi_max)) != 0:
            raise ValueError("STATUS PLANT-FIRED: hi_max bounds mismatch in fixture")
        if max_ulp_diff(ent_snap["hi_mean"], np.asarray(bounds.hi_mean)) != 0:
            raise ValueError("STATUS PLANT-FIRED: hi_mean bounds mismatch in fixture")

        rbase = 10000.0 * float(irank + 1)
        for fname, foff, src_ndim, is_a2d in _O2_ISTATE_FIELD_SPECS:
            expected_arr = _expected_o2_istate_fixture_field(
                jpi=meta_i["jpi"],
                jpj=meta_i["jpj"],
                nis0=meta_i["Nis0"],
                nie0=meta_i["Nie0"],
                njs0=meta_i["Njs0"],
                nje0=meta_i["Nje0"],
                offset=rbase + foff,
                src_ndim=src_ndim,
                jpl=meta_i["jpl"],
                is_a2d=is_a2d,
            )
            if fname == "tmask":
                expected_arr[meta_i["Nis0"] - 1, meta_i["Njs0"] - 1] = 0.0
            elif fname == "zht_i_ini":
                expected_arr[meta_i["Nis0"] - 1, meta_i["Njs0"] - 1] = -0.0
            ulp = max_ulp_diff(ist_snap["fields"][fname], expected_arr)
            if ulp != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: istate rank {irank} field {fname} ULP={ulp}"
                )
        dry_i = meta_i["Nis0"] - 1
        dry_j = meta_i["Njs0"] - 1
        dry_value = ist_snap["fields"]["zht_i_ini"][dry_i, dry_j]
        if ist_snap["fields"]["tmask"][dry_i, dry_j] != 0.0 or not np.signbit(dry_value):
            raise ValueError("STATUS PLANT-FIRED: real-writer dry -0.0 fixture was not preserved")

        for fname, foff, ndim, nlay, is_a2d in _FIXTURE_FIELD_OFFSETS:
            expected_arr = _expected_fixture_field(
                jpi=meta_e["jpi"],
                jpj=meta_e["jpj"],
                nis0=meta_e["Nis0"],
                nie0=meta_e["Nie0"],
                njs0=meta_e["Njs0"],
                nje0=meta_e["Nje0"],
                offset=rbase + foff,
                ndim=ndim,
                nlay=nlay,
                jpl=meta_e["jpl"],
                is_a2d=is_a2d,
            )
            if fname == "tmask":
                expected_arr[meta_e["Nis0"] - 1, meta_e["Njs0"] - 1] = 0.0
            ulp = max_ulp_diff(ent_snap["fields"][fname], expected_arr)
            if ulp != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: state rank {irank} field {fname} ULP={ulp}"
                )

    stitched_istate = stitch_o2_istate_snapshots(snaps_istate)
    stitched_entry = stitch_o1_binary_snapshots(snaps_entry)
    stitched_step1 = stitch_o1_binary_snapshots(snaps_step1)
    if (
        stitched_istate["meta"]["jpi"] != 16
        or stitched_istate["meta"]["jpj"] != 14
        or stitched_entry["meta"]["jpi"] != 16
        or stitched_entry["meta"]["jpj"] != 14
    ):
        raise ValueError("STATUS PLANT-FIRED: stitched global grid shape mismatch")
    if stitched_step1["meta"]["lrst_ice"] != 1 or stitched_entry["meta"]["lrst_ice"] != 0:
        raise ValueError("STATUS PLANT-FIRED: lrst_ice metadata mismatch")

    for irank, snap in enumerate(snaps_istate):
        meta = snap["meta"]
        nis0, nie0 = meta["Nis0"], meta["Nie0"]
        njs0, nje0 = meta["Njs0"], meta["Nje0"]
        gi0 = meta["nimpp"] + nis0 - 2
        gi1 = meta["nimpp"] + nie0 - 1
        gj0 = meta["njmpp"] + njs0 - 2
        gj1 = meta["njmpp"] + nje0 - 1
        for fname, *_ in _O2_ISTATE_FIELD_SPECS:
            sub_local = snap["fields"][fname][nis0 - 1 : nie0, njs0 - 1 : nje0]
            sub_global = stitched_istate["fields"][fname][gi0:gi1, gj0:gj1]
            if max_ulp_diff(sub_local, sub_global) != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: stitched istate interior mismatch on rank {irank} field {fname}"
                )

    return {
        "status": "PASS",
        "ranks_verified": 4,
        "istate_fields_per_rank": len(_O2_ISTATE_FIELD_SPECS),
        "istate_bytes_per_rank": expected_istate_bytes,
        "state_fields_per_rank": len(_FIXTURE_FIELD_OFFSETS),
        "state_bytes_per_rank": expected_state_bytes,
        "stitched_shape": [
            int(stitched_istate["meta"]["jpi"]),
            int(stitched_istate["meta"]["jpj"]),
        ],
        "max_ulp": 0,
    }


@functools.lru_cache(maxsize=1)
def _load_orca2_sas_ice_bundle_inputs() -> dict[str, np.ndarray] | None:
    """Load real ORCA2_SAS_ICE domain and SAS kt=1 SST/SSS from input tarballs in memory if present."""
    orca2_tar = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz")
    sas_tar = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz"
    )
    if not orca2_tar.is_file() or not sas_tar.is_file():
        return None

    with tarfile.open(orca2_tar, "r:gz") as tf:
        dom_member = next(m for m in tf.getmembers() if m.name.endswith("ORCA_R2_zps_domcfg.nc"))
        dom_bytes = tf.extractfile(dom_member).read()  # type: ignore[union-attr]
    with tarfile.open(sas_tar, "r:gz") as tf:
        sas_member = next(m for m in tf.getmembers() if m.name.endswith("sas_grid_T.nc"))
        sas_bytes = tf.extractfile(sas_member).read()  # type: ignore[union-attr]

    with netCDF4.Dataset("domcfg_mem.nc", memory=dom_bytes) as ds:
        e1t_int = np.squeeze(np.asarray(ds.variables["e1t"][...], dtype=np.float64)).T
        e2t_int = np.squeeze(np.asarray(ds.variables["e2t"][...], dtype=np.float64)).T
        ff_t_int = np.squeeze(np.asarray(ds.variables["ff_t"][...], dtype=np.float64)).T
        gphit_int = np.squeeze(np.asarray(ds.variables["gphit"][...], dtype=np.float64)).T
        bottom_level = np.squeeze(np.asarray(ds.variables["bottom_level"][...], dtype=np.int32)).T
        tmask_int = (bottom_level > 0).astype(np.float64)

    with netCDF4.Dataset("sas_mem.nc", memory=sas_bytes) as ds:
        sst_r0 = np.asarray(ds.variables["sosstsst"][0], dtype=np.float64).T
        sst_r1 = np.asarray(ds.variables["sosstsst"][1], dtype=np.float64).T
        sss_r0 = np.asarray(ds.variables["sosaline"][0], dtype=np.float64).T
        sss_r1 = np.asarray(ds.variables["sosaline"][1], dtype=np.float64).T

    # Time interpolation at kt = 1 for 120h (5-day) yearly climatology with rn_Dt = 10800s:
    # zt = (kt - 0.5) * rn_Dt / (120 * 3600) + 0.5 = 0.0125 + 0.5 = 123 / 240;
    # NEMO fld_interp (fldread.F90) weight between record 0 (t=-0.5) and record 1 (t=+0.5):
    w1 = (1.5 * 3600.0) / (120.0 * 3600.0) + 0.5  # 123 / 240
    w0 = 1.0 - w1
    sst_int = w0 * sst_r0 + w1 * sst_r1
    sss_int = w0 * sss_r0 + w1 * sss_r1

    def _pad_lbc(arr_180_148: np.ndarray, psgn: float = 1.0) -> np.ndarray:
        padded = np.zeros((184, 152), dtype=np.float64)
        padded[2:-2, 2:-2] = arr_180_148
        return np.asarray(
            si3_lbc_lnk_orca2(
                jnp.asarray(padded, dtype=jnp.float64),
                cd_nat="T",
                psgn=psgn,
                nn_hls=2,
                ldfull=True,
            ),
            dtype=np.float64,
        )

    tmask = _pad_lbc(tmask_int, 1.0)
    e1t = _pad_lbc(e1t_int, 1.0)
    e2t = _pad_lbc(e2t_int, 1.0)
    e1e2t = e1t * e2t
    ff_t = _pad_lbc(ff_t_int, 1.0)
    gphit = _pad_lbc(gphit_int, 1.0)
    sst_m = _pad_lbc(sst_int, 1.0)
    sss_m = _pad_lbc(sss_int, 1.0)
    return {
        "tmask": tmask,
        "ssmask": tmask.copy(),
        "e1e2t": e1e2t,
        "ff_t": ff_t,
        "gphit": gphit,
        "sst_m": sst_m,
        "sss_m": sss_m,
    }


def build_synthetic_orca2_o2_inputs(
    config: SI3ORCA2StateConfig | None = None,
    o2_config: SI3O2InitConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic full ORCA2 `(184, 152)` O2 inputs (preferring real bundle geometry/SAS when present)."""
    cfg = config or SI3ORCA2StateConfig.for_orca2_sas_ice()
    o2_cfg = o2_config or SI3O2InitConfig.for_orca2_sas_ice()
    state0, tmask_syn, umask_syn, vmask_syn = build_synthetic_orca2_o1_state(cfg)
    # O1's builder selects the native arm; the O3 fit's EXP needs glibc's.
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    bundle = _load_orca2_sas_ice_bundle_inputs()
    if bundle is not None:
        tmask = jnp.asarray(bundle["tmask"], dtype=jnp.float64)
        ssmask = jnp.asarray(bundle["ssmask"], dtype=jnp.float64)
        e1e2t = jnp.asarray(bundle["e1e2t"], dtype=jnp.float64)
        ff_t = jnp.asarray(bundle["ff_t"], dtype=jnp.float64)
        gphit = jnp.asarray(bundle["gphit"], dtype=jnp.float64)
        sst_m = jnp.asarray(bundle["sst_m"], dtype=jnp.float64)
        sss_m = jnp.asarray(bundle["sss_m"], dtype=jnp.float64)
        umask = tmask
        vmask = tmask
        state0 = state0._replace(
            snwice_mass=state0.snwice_mass * tmask,
            snwice_mass_b=state0.snwice_mass_b * tmask,
        )
    else:
        jpi, jpj = 184, 152
        tmask = tmask_syn
        umask = umask_syn
        vmask = vmask_syn
        ssmask = tmask_syn
        jj = np.linspace(-78.0, 88.0, jpj, dtype=np.float64)[None, :]
        gphit_np = np.broadcast_to(jj, (jpi, jpj)).copy()
        ff_t_np = 2.0 * 7.292115083046062e-5 * np.sin(np.deg2rad(gphit_np))
        e1e2t_np = np.full((jpi, jpj), 4.0e10, dtype=np.float64)
        sst_np = np.where(np.abs(gphit_np) > 60.0, -1.85, 15.0)
        sss_np = np.full((jpi, jpj), 34.7, dtype=np.float64)
        e1e2t = si3_lbc_lnk_orca2(jnp.asarray(e1e2t_np), cd_nat="T", psgn=1.0, nn_hls=2, ldfull=True)
        ff_t = si3_lbc_lnk_orca2(jnp.asarray(ff_t_np), cd_nat="T", psgn=1.0, nn_hls=2, ldfull=True)
        gphit = si3_lbc_lnk_orca2(jnp.asarray(gphit_np), cd_nat="T", psgn=1.0, nn_hls=2, ldfull=True)
        sst_m = si3_lbc_lnk_orca2(jnp.asarray(sst_np), cd_nat="T", psgn=1.0, nn_hls=2, ldfull=True)
        sss_m = si3_lbc_lnk_orca2(jnp.asarray(sss_np), cd_nat="T", psgn=1.0, nn_hls=2, ldfull=True)

    smask0_i = ssmask * si3_dom_uniq_t_orca2(
        int(tmask.shape[0]), int(tmask.shape[1]), nn_hls=cfg.nn_hls
    )
    ssh_before_kbb = jnp.where(ssmask != 0.0, 0.125, 0.0)
    ssh_before_kmm = jnp.where(ssmask != 0.0, -0.25, 0.0)
    return {
        "cfg": cfg,
        "o2_cfg": o2_cfg,
        "state0": state0,
        "tmask": tmask,
        "umask": umask,
        "vmask": vmask,
        "ssmask": ssmask,
        "smask0_i": smask0_i,
        "e1e2t": e1e2t,
        "ff_t": ff_t,
        "gphit": gphit,
        "sst_m": sst_m,
        "sss_m": sss_m,
        "ssh_before_kbb": ssh_before_kbb,
        "ssh_before_kmm": ssh_before_kmm,
    }


def _compare_o2_scalar_state_against_recipe_and_ref(
    ref_fields: dict[str, np.ndarray | jax.Array],
    ref_area: float,
    ref_zsshadj: float,
    actual: SI3O2ScalarInitialState,
    tmask: np.ndarray | jax.Array,
    *,
    ssh_before_kbb: np.ndarray | jax.Array | None = None,
    ssh_before_kmm: np.ndarray | jax.Array | None = None,
    allow_itd_5cat_1ulp: bool = False,
) -> dict[str, Any]:
    """Verify O2-owned fields against the NEMO record."""
    max_ulp_wet = 0
    max_ulp_full = 0
    max_norm_wet = 0.0
    itd_sum_fields = {"vt_i", "vt_s", "at_i", "at_ip", "vt_ip", "vt_il", "snwice_mass", "snwice_mass_b"}

    for fname in _O2_SCALAR_STATE_FIELDS:
        exp_arr = ref_fields[fname]
        act_arr = getattr(actual, fname)
        # NEMO defines t_bo only on the kbnd=0 interior; the other scalar
        # fields are defined through the wet seam/fold halos by kbnd=nn_hls.
        score_mask = np.asarray(tmask) != 0.0
        if fname == "t_bo":
            score_mask = score_mask.copy()
            score_mask[:2, :] = False
            score_mask[-2:, :] = False
            score_mask[:, :2] = False
            score_mask[:, -2:] = False
        ulp_w = max_ulp_diff(exp_arr, act_arr, mask=score_mask)
        ulp_f = max_ulp_diff(exp_arr, act_arr)
        norm_w = max_normalized_diff(exp_arr, act_arr, mask=score_mask)
        if allow_itd_5cat_1ulp and fname in itd_sum_fields:
            if ulp_w > 1 or norm_w > 1e-15:
                raise ValueError(
                    f"STATUS PLANT-FIRED: O2 5-category ITD sum field {fname} "
                    f"wet_ULP={ulp_w}, wet_norm={norm_w:.3e}"
                )
            continue
        max_ulp_wet = max(max_ulp_wet, ulp_w)
        max_ulp_full = max(max_ulp_full, ulp_f)
        max_norm_wet = max(max_norm_wet, norm_w)
        if ulp_w != 0 or ulp_f != 0 or norm_w > 1e-15:
            raise ValueError(
                f"STATUS PLANT-FIRED: O2 scalar initial state field {fname} "
                f"wet_ULP={ulp_w}, full_ULP={ulp_f}, wet_norm={norm_w:.3e}"
            )

    ulp_area = max_ulp_diff(np.array([ref_area]), np.asarray(actual.area).reshape(1))
    ulp_zsshadj = max_ulp_diff(
        np.array([ref_zsshadj]), np.asarray(actual.zsshadj).reshape(1)
    )
    if ulp_area != 0 or ulp_zsshadj != 0:
        raise ValueError(
            f"STATUS PLANT-FIRED: O2 scalar reduction mismatch "
            f"area_ULP={ulp_area}, zsshadj_ULP={ulp_zsshadj}"
        )

    return {
        "max_ulp_wet": max_ulp_wet,
        "max_ulp_full": max_ulp_full,
        "max_norm_wet": max_norm_wet,
        "area": float(np.asarray(actual.area)),
        "zsshadj": float(np.asarray(actual.zsshadj)),
    }


def run_self_test(plant: str | None = None) -> dict[str, Any]:
    """Run the full O2 gate self-test and negative-control plants P2-P6."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    inp = build_synthetic_orca2_o2_inputs()
    cfg: SI3ORCA2StateConfig = inp["cfg"]
    o2_cfg: SI3O2InitConfig = inp["o2_cfg"]
    state0: SI3State = inp["state0"]
    tmask = inp["tmask"]
    umask = inp["umask"]
    vmask = inp["vmask"]
    ssmask = inp["ssmask"]
    smask0_i = inp["smask0_i"]
    e1e2t = inp["e1e2t"]
    ff_t = inp["ff_t"]
    gphit = inp["gphit"]
    sst_m = inp["sst_m"]
    sss_m = inp["sss_m"]
    ssh_before_kbb = inp["ssh_before_kbb"]
    ssh_before_kmm = inp["ssh_before_kmm"]

    wi, wj = first_wet_ij(tmask, interior_only=True, nn_hls=cfg.nn_hls)
    wi_u, wj_u = first_wet_ij(umask, interior_only=True, nn_hls=cfg.nn_hls)
    wi_v, wj_v = first_wet_ij(vmask, interior_only=True, nn_hls=cfg.nn_hls)

    # Reference O2 scalar initial state (category fit computed by rung O3).
    ref_o2 = si3_o2_scalar_initial_state(
        sst_m=sst_m,
        sss_m=sss_m,
        tmask=tmask,
        ssmask=ssmask,
        smask0_i=smask0_i,
        e1e2t=e1e2t,
        ff_t=ff_t,
        gphit=gphit,
        ssh_before_kbb=ssh_before_kbb,
        ssh_before_kmm=ssh_before_kmm,
        config=o2_cfg,
        state_config=cfg,
    )

    cand_o2 = ref_o2
    if plant == "1ulp-istate":
        ht_np = np.asarray(cand_o2.zht_i_ini).copy()
        ht_np[wi, wj] = np.nextafter(ht_np[wi, wj], np.inf)
        cand_o2 = cand_o2._replace(zht_i_ini=jnp.asarray(ht_np))
    elif plant == "1ulp-ssh":
        ssh_np = np.asarray(cand_o2.ssh_kmm).copy()
        ssh_np[wi, wj] = np.nextafter(ssh_np[wi, wj], np.inf)
        cand_o2 = cand_o2._replace(ssh_kmm=jnp.asarray(ssh_np))
    elif plant == "eos-wrong-branch":
        cand_o2 = si3_o2_scalar_initial_state(
            sst_m=sst_m,
            sss_m=sss_m,
            tmask=tmask,
            ssmask=ssmask,
            smask0_i=smask0_i,
            e1e2t=e1e2t,
            ff_t=ff_t,
            gphit=gphit,
            ssh_before_kbb=ssh_before_kbb,
            ssh_before_kmm=ssh_before_kmm,
            config=o2_cfg,
            state_config=cfg,
            _plant_wrong_eos=True,
        )
    elif plant == "tbo-halo-leak":
        cand_o2 = si3_o2_scalar_initial_state(
            sst_m=sst_m, sss_m=sss_m, tmask=tmask, ssmask=ssmask,
            smask0_i=smask0_i, e1e2t=e1e2t, ff_t=ff_t, gphit=gphit,
            ssh_before_kbb=ssh_before_kbb, ssh_before_kmm=ssh_before_kmm,
            config=o2_cfg, state_config=cfg, _plant_tbo_halo_leak=True,
        )
    elif plant == "pond-unscaled":
        cand_o2 = si3_o2_scalar_initial_state(
            sst_m=sst_m, sss_m=sss_m, tmask=tmask, ssmask=ssmask,
            smask0_i=smask0_i, e1e2t=e1e2t, ff_t=ff_t, gphit=gphit,
            ssh_before_kbb=ssh_before_kbb, ssh_before_kmm=ssh_before_kmm,
            config=o2_cfg, state_config=cfg, _plant_pond_unscaled=True,
        )
    elif plant == "hemisphere-swap":
        cand_o2 = si3_o2_scalar_initial_state(
            sst_m=sst_m,
            sss_m=sss_m,
            tmask=tmask,
            ssmask=ssmask,
            smask0_i=smask0_i,
            e1e2t=e1e2t,
            ff_t=ff_t,
            gphit=gphit,
            ssh_before_kbb=ssh_before_kbb,
            ssh_before_kmm=ssh_before_kmm,
            config=o2_cfg,
            state_config=cfg,
            _plant_hemisphere_swap=True,
        )
    elif plant == "ssh-sign-or-div":
        cand_o2 = si3_o2_scalar_initial_state(
            sst_m=sst_m, sss_m=sss_m, tmask=tmask, ssmask=ssmask,
            smask0_i=smask0_i, e1e2t=e1e2t, ff_t=ff_t, gphit=gphit,
            ssh_before_kbb=ssh_before_kbb, ssh_before_kmm=ssh_before_kmm,
            config=o2_cfg, state_config=cfg, _plant_ssh_sign=True,
        )
    elif plant == "dom-uniq-fold":
        cand_o2 = si3_o2_scalar_initial_state(
            sst_m=sst_m, sss_m=sss_m, tmask=tmask, ssmask=ssmask,
            smask0_i=smask0_i, e1e2t=e1e2t, ff_t=ff_t, gphit=gphit,
            ssh_before_kbb=ssh_before_kbb, ssh_before_kmm=ssh_before_kmm,
            config=o2_cfg, state_config=cfg, _plant_omit_nfd_uniq=True,
        )

    ref_fields = {fname: getattr(ref_o2, fname) for fname in _O2_SCALAR_STATE_FIELDS}
    o2_cmp = _compare_o2_scalar_state_against_recipe_and_ref(
        ref_fields=ref_fields,
        ref_area=float(np.asarray(ref_o2.area)),
        ref_zsshadj=float(np.asarray(ref_o2.zsshadj)),
        actual=cand_o2,
        tmask=tmask,
        ssh_before_kbb=ssh_before_kbb,
        ssh_before_kmm=ssh_before_kmm,
    )

    # The dispatcher must select O2 and then refuse the still-unowned O16
    # heat/freshwater/salt response rather than synthesizing zeros.
    o2_forcing = SI3ORCA2O2Forcing(
        tmask=tmask,
        sst_m=sst_m,
        sss_m=sss_m,
        ff_t=ff_t,
        gphit=gphit,
        e1e2t=e1e2t,
        ssmask=ssmask,
        smask0_i=smask0_i,
        ssh_before_kbb=ssh_before_kbb,
        ssh_before_kmm=ssh_before_kmm,
        apply_halos=False,
    )
    state1, o2_init_step = si3_o2_step(
        state0,
        o2_forcing,
        config=cfg,
        o2_init_config=o2_cfg,
    )
    if o2_init_step is None:
        raise ValueError("STATUS PLANT-FIRED: si3_o2_step did not return SI3O2ScalarInitialState")
    sea_ice_cfg = build_si3_orca2_o2_config(cfg)
    z2 = jnp.zeros_like(tmask)
    try:
        step_sea_ice(
            state0,
            o2_forcing,
            z2,
            z2,
            z2,
            sea_ice_cfg,
            0.1,
            cfg.rDt_ice,
        )
    except NotImplementedError as exc:
        if "ice_update_flx" not in str(exc):
            raise
    else:
        raise ValueError("STATUS PLANT-FIRED: O2 dispatcher did not refuse unowned fluxes")

    dry_fixture = verify_dry_signed_zero_fixture(ref_o2.zht_i_ini, tmask, nn_hls=cfg.nn_hls)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        rst_o2_path = Path(tmp) / "ORCA2_SAS_00000001_restart_ice.nc"
        write_si3_netcdf_restart(rst_o2_path, state1, kt=1, config=cfg)
        state_o2_reloaded_init = read_si3_netcdf_restart(
            rst_o2_path, tmask=tmask, sss_m=sss_m, config=cfg, ln_rstart=False
        )
        state_o2_reloaded = si3_o1_step(
            state_o2_reloaded_init, tmask=tmask, sss_m=sss_m, config=cfg
        )
        _compare_si3_states_wet_and_full(
            state1, state_o2_reloaded, tmask, umask, vmask, check_prather=False
        )

        rst_path = Path(tmp) / "ORCA2_SAS_00000001_restart_ice_pra.nc"
        write_si3_netcdf_restart(
            rst_path, state1, kt=1, config=cfg, write_prather=True
        )
        state_reloaded_init = read_si3_netcdf_restart(
            rst_path, tmask=tmask, sss_m=sss_m, config=cfg, ln_rstart=True
        )
        state_reloaded = si3_o1_step(
            state_reloaded_init, tmask=tmask, sss_m=sss_m, config=cfg
        )

    if plant == "1ulp-state":
        v_i_np = np.asarray(state_reloaded.v_i).copy()
        v_i_np[wi, wj, 0] = np.nextafter(v_i_np[wi, wj, 0], np.inf)
        state_reloaded = state_reloaded._replace(v_i=jnp.asarray(v_i_np))
    elif plant == "1ulp-moment":
        sx_np = np.asarray(state_reloaded.prather.ice.sx).copy()
        sx_np[wi, wj, 0] = np.nextafter(sx_np[wi, wj, 0], np.inf)
        state_reloaded = state_reloaded._replace(
            prather=state_reloaded.prather._replace(
                ice=state_reloaded.prather.ice._replace(sx=jnp.asarray(sx_np))
            )
        )
    elif plant == "north-fold-sign":
        u_bad = si3_lbc_lnk_orca2(
            state1.u_ice, cd_nat="U", psgn=1.0, nn_hls=cfg.nn_hls, ldfull=True
        )
        state_reloaded = state_reloaded._replace(u_ice=u_bad)
    elif plant == "agg-formula":
        hm_bad = np.asarray(state_reloaded.aggregated.hm_i).copy()
        hm_bad[wi, wj] += 1e-6
        state_reloaded = state_reloaded._replace(
            aggregated=state_reloaded.aggregated._replace(hm_i=jnp.asarray(hm_bad))
        )

    cmp_metrics = _compare_si3_states_wet_and_full(
        state1, state_reloaded, tmask, umask, vmask, check_prather=True
    )

    # Differentiability gate (Plant P6: stop_gradient through O2 scalar initial state + SSH)
    def _o2_loss(sss_in: jax.Array, hti_n_in: jax.Array, v_i_in: jax.Array) -> jax.Array:
        o2_cfg_ad = o2_cfg._replace(rn_hti_ini_n=hti_n_in)
        s_in = state0._replace(v_i=v_i_in)
        s_out, o2_out = si3_o2_step(
            s_in,
            SI3ORCA2O2Forcing(
                tmask=tmask,
                sst_m=sst_m,
                sss_m=sss_in,
                ff_t=ff_t,
                gphit=gphit,
                e1e2t=e1e2t,
                ssmask=ssmask,
                smask0_i=smask0_i,
                ssh_before_kbb=ssh_before_kbb,
                ssh_before_kmm=ssh_before_kmm,
                apply_halos=False,
            ),
            config=cfg,
            o2_init_config=o2_cfg_ad,
            _plant_stop_gradient=plant == "stop-gradient",
        )
        assert o2_out is not None
        return (
            jnp.sum(
                o2_out.t_bo
                + o2_out.ztfrz
                + o2_out.zht_i_ini
                + o2_out.snwice_mass
                + o2_out.ssh_kmm
            )
            + o2_out.zsshadj
            + jnp.sum(s_out.aggregated.vt_i + s_out.snwice_mass)
        )

    loss_val, (grad_sss, grad_hti, grad_vi) = jax.jit(
        jax.value_and_grad(_o2_loss, argnums=(0, 1, 2))
    )(sss_m, jnp.asarray(o2_cfg.rn_hti_ini_n, dtype=jnp.float64), state0.v_i)
    grad_sss_norm = float(jnp.linalg.norm(grad_sss))
    grad_hti_val = float(jnp.abs(grad_hti))
    grad_vi_norm = float(jnp.linalg.norm(grad_vi))
    grad_norm = float(np.hypot(np.hypot(grad_sss_norm, grad_hti_val), grad_vi_norm))
    if (
        not np.isfinite(float(loss_val))
        or not np.isfinite(grad_norm)
        or grad_sss_norm == 0.0
        or grad_hti_val == 0.0
        or grad_vi_norm == 0.0
    ):
        raise ValueError(
            f"STATUS PLANT-FIRED: O2 AD gate failed (loss={loss_val}, "
            f"grad_sss={grad_sss_norm}, grad_hti={grad_hti_val}, grad_vi={grad_vi_norm})"
        )

    return {
        "status": "PASS",
        "grid_shape": [184, 152],
        "jpl": cfg.jpl,
        "nlay_i": cfg.nlay_i,
        "nlay_s": cfg.nlay_s,
        "first_wet_t_ij": [wi, wj],
        "first_wet_u_ij": [wi_u, wj_u],
        "first_wet_v_ij": [wi_v, wj_v],
        "dry_signed_zero_fixture": dry_fixture,
        "o2_scalar_fields_checked": len(_O2_SCALAR_STATE_FIELDS),
        "o2_area": o2_cmp["area"],
        "o2_zsshadj": o2_cmp["zsshadj"],
        "o2_max_ulp_wet": o2_cmp["max_ulp_wet"],
        "o2_max_ulp_full": o2_cmp["max_ulp_full"],
        "prognostic_fields_checked": len(_PROG_FIELDS),
        "prather_arrays_checked": 55,
        "equivalent_fields_checked": len(state1.equivalent._fields),
        "aggregated_fields_checked": len(state1.aggregated._fields),
        "max_ulp": max(o2_cmp["max_ulp_wet"], cmp_metrics["max_ulp_wet"]),
        "max_ulp_full": max(o2_cmp["max_ulp_full"], cmp_metrics["max_ulp_full"]),
        "max_norm_wet": max(o2_cmp["max_norm_wet"], cmp_metrics["max_norm_wet"]),
        "ad_grad_norm": grad_norm,
        "ad_grad_sss_norm": grad_sss_norm,
        "ad_grad_hti_abs": grad_hti_val,
    }


def admit_evidence_directory(
    evidence_dir: str | Path,
    plant: str | None = None,
    *,
    expect_commit: str | None = None,
) -> dict[str, Any]:
    """Admit a completed NEMO 5.0.2 `ORCA2_SAS_ICE` O2 acquisition directory."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    ev = Path(evidence_dir)
    manifest_dir = ev / "manifest"
    ref_dir = ev / "ref_2step"
    rec_dir = ev / "rec_2step"
    rst_dir = ev / "rst_1step"

    # 1. Producer commit and provenance manifests
    commit_file = manifest_dir / "producer_commit.txt"
    if not commit_file.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing {commit_file}")
    recorded_commit = commit_file.read_text(encoding="utf-8").strip()
    if plant == "stamp":
        recorded_commit = "0" * 40
    if not re.fullmatch(r"[0-9a-f]{40}", recorded_commit):
        raise ValueError(
            f"STATUS PLANT-FIRED: malformed producer commit {recorded_commit!r}"
        )
    commit_object = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{recorded_commit}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if commit_object.returncode != 0:
        raise ValueError(
            f"STATUS PLANT-FIRED: producer commit {recorded_commit} is not a commit object"
        )
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", recorded_commit, "HEAD"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise ValueError(
            f"STATUS PLANT-FIRED: producer commit {recorded_commit} is not an ancestor of HEAD"
        )
    if expect_commit is not None and recorded_commit != expect_commit.strip():
        raise ValueError(
            f"STATUS PLANT-FIRED: producer commit {recorded_commit} != expected {expect_commit.strip()}"
        )

    same_prog_path = manifest_dir / "same_program.json"
    if not same_prog_path.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing {same_prog_path}")
    same_prog_data = json.loads(same_prog_path.read_text(encoding="utf-8"))
    if plant == "same-program":
        same_prog_data["status"] = "FAIL"
    if same_prog_data.get("status") != "PASS_SAME_PROGRAM":
        raise ValueError(
            f"STATUS PLANT-FIRED: same_program.json status={same_prog_data.get('status')!r}"
        )

    binaries_checked = _verify_sha256_manifest(
        manifest_dir / "binaries.sha256",
        label="binaries.sha256",
        plant_mismatch=plant == "binary-sha",
    )
    binary_targets = {
        Path(line.strip().split(None, 1)[1]).resolve()
        for line in (manifest_dir / "binaries.sha256").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    try:
        expected_binary_targets = {
            (Path(same_prog_data["reference_bld"]) / "bin/nemo.exe").resolve(),
            (Path(same_prog_data["new_bld"]) / "bin/nemo.exe").resolve(),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "STATUS PLANT-FIRED: same_program.json lacks reference/new build paths"
        ) from exc
    if binary_targets != expected_binary_targets or binaries_checked != 2:
        raise ValueError(
            "STATUS PLANT-FIRED: binaries.sha256 is not the two same-program nemo.exe files"
        )
    inputs_checked = _verify_sha256_manifest(
        manifest_dir / "inputs.sha256",
        label="inputs.sha256",
        plant_mismatch=plant == "input-sha",
    )

    gate_manifest = manifest_dir / "gate.sha256"
    gate_changed_since_run = None
    if gate_manifest.is_file():
        gate_parts = gate_manifest.read_text(encoding="utf-8").strip().split(None, 1)
        if len(gate_parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", gate_parts[0]):
            raise ValueError("STATUS PLANT-FIRED: malformed gate.sha256")
        current_gate = Path(gate_parts[1])
        gate_changed_since_run = (
            not current_gate.is_file()
            or hashlib.sha256(current_gate.read_bytes()).hexdigest() != gate_parts[0]
        )

    # 2. Completion markers and log checks on all three runs
    for rdir, rlabel in (
        (ref_dir, "ref_2step"),
        (rec_dir, "rec_2step"),
        (rst_dir, "rst_1step"),
    ):
        run_stamp = rdir / "RUN_PRODUCER_COMMIT"
        if (
            not run_stamp.is_file()
            or run_stamp.read_text(encoding="utf-8").strip() != recorded_commit
        ):
            raise ValueError(f"STATUS PLANT-FIRED: producer stamp mismatch in {rlabel}")
        _verify_run_completion_and_logs(
            rdir,
            rlabel,
            plant_missing_complete=plant == "missing-complete" and rlabel == "rst_1step",
            plant_missing_stop0=plant == "missing-stop0" and rlabel == "rst_1step",
        )

    # 3. Reference vs instrumented restart byte-identity (steps 1 and 2)
    for step in (1, 2):
        rst_name = f"ORCA2_SAS_{step:08d}_restart_ice.nc"
        ref_rst = ref_dir / rst_name
        rec_rst = rec_dir / rst_name
        if not ref_rst.is_file() or not rec_rst.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing restart file {rst_name}")
        if plant == "ref-rec-diff" or ref_rst.read_bytes() != rec_rst.read_bytes():
            raise ValueError(
                f"STATUS PLANT-FIRED: instrumented restart {rst_name} differs from reference"
            )

    # 4. Score O2 scalar initial state & levitating-ice SSH (`o2_si3_istate_rank*.bin`)
    rec_records = rec_dir / "o2_si3_records"
    istate_files = sorted(rec_records.glob("o2_si3_istate_rank*.bin"))
    entry_files = sorted(rec_records.glob("o2_si3_state_entry_rank*.bin"))
    step1_files = sorted(rec_records.glob("o2_si3_state_step0001_rank*.bin"))
    step2_files = sorted(rec_records.glob("o2_si3_state_step0002_rank*.bin"))
    if not istate_files or not entry_files or not step1_files or not step2_files:
        raise ValueError("STATUS PLANT-FIRED: missing O2 binary records in rec_2step")

    snap_istate = stitch_o2_istate_snapshots([read_o2_istate_snapshot(p) for p in istate_files])
    snap_entry = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in entry_files])
    snap_step1 = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in step1_files])
    snap_step2 = stitch_o1_binary_snapshots([read_o1_binary_snapshot(p) for p in step2_files])

    state_entry, cfg = o1_snapshot_to_si3_state(snap_entry)
    o2_cfg = o2_istate_snapshot_to_config(
        snap_istate, ln_teos10=True, ln_eos80=False
    )._replace(ln_pnd=cfg.ln_pnd, ln_pnd_lids=cfg.ln_pnd_lids)

    f_ist = snap_istate["fields"]
    tmask = jnp.asarray(f_ist["tmask"], dtype=jnp.float64)
    umask = jnp.asarray(f_ist["umask"], dtype=jnp.float64)
    vmask = jnp.asarray(f_ist["vmask"], dtype=jnp.float64)
    ssmask = jnp.asarray(f_ist["ssmask"], dtype=jnp.float64)
    e1e2t = jnp.asarray(f_ist["e1e2t"], dtype=jnp.float64)
    ff_t = jnp.asarray(f_ist["ff_t"], dtype=jnp.float64)
    gphit = jnp.asarray(f_ist["gphit"], dtype=jnp.float64)
    sst_istate = jnp.asarray(f_ist["sst_m"], dtype=jnp.float64)
    sss_istate = jnp.asarray(f_ist["sss_m"], dtype=jnp.float64)
    ssh_before_kbb = jnp.asarray(f_ist["ssh_before_kbb"], dtype=jnp.float64)
    ssh_before_kmm = jnp.asarray(f_ist["ssh_before_kmm"], dtype=jnp.float64)

    wi, wj = first_wet_ij(tmask, interior_only=True, nn_hls=cfg.nn_hls)
    dry_fixture = verify_dry_signed_zero_fixture(f_ist["zht_i_ini"], tmask, nn_hls=cfg.nn_hls)

    o2_scalar_jit = jax.jit(
        lambda sst, sss, tm, ssm, sm0i, area_t, coriolis, lat, ssh_b, ssh_m: si3_o2_scalar_initial_state(
            sst_m=sst,
            sss_m=sss,
            tmask=tm,
            ssmask=ssm,
            smask0_i=sm0i,
            e1e2t=area_t,
            ff_t=coriolis,
            gphit=lat,
            ssh_before_kbb=ssh_b,
            ssh_before_kmm=ssh_m,
            config=o2_cfg,
            state_config=cfg,
        )
    )
    actual_o2 = o2_scalar_jit(
        sst_istate,
        sss_istate,
        tmask,
        ssmask,
        jnp.asarray(f_ist["smask0_i"], dtype=jnp.float64),
        e1e2t,
        ff_t,
        gphit,
        ssh_before_kbb,
        ssh_before_kmm,
    )

    if plant == "1ulp-istate":
        ht_np = np.asarray(actual_o2.zht_i_ini).copy()
        ht_np[wi, wj] = np.nextafter(ht_np[wi, wj], np.inf)
        actual_o2 = actual_o2._replace(zht_i_ini=jnp.asarray(ht_np))
    elif plant == "1ulp-ssh":
        ssh_np = np.asarray(actual_o2.ssh_kmm).copy()
        ssh_np[wi, wj] = np.nextafter(ssh_np[wi, wj], np.inf)
        actual_o2 = actual_o2._replace(ssh_kmm=jnp.asarray(ssh_np))

    o2_cmp = _compare_o2_scalar_state_against_recipe_and_ref(
        ref_fields=f_ist,
        ref_area=float(snap_istate["meta"]["area"]),
        ref_zsshadj=float(snap_istate["meta"]["zsshadj"]),
        actual=actual_o2,
        tmask=tmask,
        ssh_before_kbb=ssh_before_kbb,
        ssh_before_kmm=ssh_before_kmm,
        allow_itd_5cat_1ulp=False,
    )

    # 5. Score rec_2step (entry, step 1, step 2) over wet cells
    sss_entry = jnp.asarray(snap_entry["fields"]["sss_m"], dtype=jnp.float64)
    sss_step2_np = np.asarray(snap_step2["fields"]["sss_m"], dtype=np.float64)
    sss_step2 = jnp.asarray(sss_step2_np, dtype=jnp.float64)
    smask0_i = jnp.asarray(f_ist["smask0_i"], dtype=jnp.float64)
    o2_step_jit = jax.jit(
        lambda step_state, step_sss: si3_o2_step(
            step_state,
            SI3ORCA2O2Forcing(
                tmask=tmask,
                sst_m=sst_istate,
                sss_m=step_sss,
                ff_t=ff_t,
                gphit=gphit,
                e1e2t=e1e2t,
                ssmask=ssmask,
                smask0_i=smask0_i,
                ssh_before_kbb=ssh_before_kbb,
                ssh_before_kmm=ssh_before_kmm,
                apply_halos=False,
            ),
            config=cfg,
            o2_init_config=o2_cfg,
        )
    )
    o1_step_jit = jax.jit(
        lambda step_state, step_sss: si3_o1_step(
            step_state, tmask=tmask, sss_m=step_sss, config=cfg
        )
    )
    state_kt1, _ = o2_step_jit(state_entry, sss_entry)
    no_ice_wet = (np.asarray(tmask) != 0.0) & np.any(
        np.asarray(state_kt1.v_i) == 0.0, axis=2
    )
    sss_plant_i, sss_plant_j = first_wet_ij(
        no_ice_wet, interior_only=True, nn_hls=cfg.nn_hls
    )
    if plant == "kt2-sss-perturb":
        sss_step2_plant = sss_step2_np.copy()
        branch_point = cfg.rn_simin / cfg.rn_sinew
        sss_step2_plant[sss_plant_i, sss_plant_j] = np.nextafter(
            branch_point, -np.inf
        )
        sss_step2 = jnp.asarray(sss_step2_plant, dtype=jnp.float64)
    state_kt2 = o1_step_jit(state_kt1, sss_step2)

    if plant == "1ulp-state":
        v_np = np.asarray(state_kt1.v_i).copy()
        v_np[wi, wj, 0] = np.nextafter(v_np[wi, wj, 0], np.inf)
        state_kt1 = state_kt1._replace(v_i=jnp.asarray(v_np))

    nemo_kt1, _ = o1_snapshot_to_si3_state(snap_step1, config=cfg)
    nemo_kt2, _ = o1_snapshot_to_si3_state(snap_step2, config=cfg)

    cmp_kt1 = _compare_si3_states_wet_and_full(
        nemo_kt1, state_kt1, tmask, umask, vmask, check_fmass=True, check_prather=True
    )
    cmp_kt2 = _compare_si3_states_wet_and_full(
        nemo_kt2, state_kt2, tmask, umask, vmask, check_fmass=True, check_prather=True
    )

    # 6. Score rst_1step: NetCDF restart read + 1-step trajectory vs rec_2step step 2
    rst_records = rst_dir / "o2_si3_records"
    rst_entry_files = sorted(rst_records.glob("o2_si3_state_entry_rank*.bin"))
    rst_step1_pattern = (
        "o2_si3_state_step0001_rank*.bin"
        if plant == "rst-step-name"
        else "o2_si3_state_step0002_rank*.bin"
    )
    rst_step1_files = sorted(rst_records.glob(rst_step1_pattern))
    rst_nc = rst_dir / "output_restart/ORCA2_SAS_00000002_restart_ice_o2_output.nc"
    rst_ocean_nc = rst_dir / "output_restart/ORCA2_SAS_00000002_restart_o2_output.nc"
    rst_input = rst_dir / "input_restart/ORCA2_SAS_00000001_restart_ice.nc"
    rst_ocean_input = rst_dir / "input_restart/ORCA2_SAS_00000001_restart.nc"
    producer_restart = rec_dir / "ORCA2_SAS_00000001_restart_ice.nc"
    producer_ocean_restart = rec_dir / "ORCA2_SAS_00000001_restart.nc"
    producer_ocean_restart_checked = (
        rec_dir / "PLANTED_MISSING_SAS_OCEAN_RESTART.nc"
        if plant == "ocean-restart-missing"
        else producer_ocean_restart
    )
    rst_nc_checked = rst_nc.with_name("PLANTED_MISSING_restart_ice.nc") if plant == "rst-missing" else rst_nc
    if (
        not rst_entry_files
        or not rst_step1_files
        or not rst_nc_checked.is_file()
        or not rst_ocean_nc.is_file()
        or not rst_input.is_file()
        or not rst_ocean_input.is_file()
        or not producer_restart.is_file()
        or not producer_ocean_restart_checked.is_file()
    ):
        raise ValueError(
            "STATUS PLANT-FIRED: rst_1step requires entry/step0002 records, both copied inputs, and both step-2 outputs"
        )
    if rst_input.resolve() == rst_nc.resolve() or rst_input.read_bytes() != producer_restart.read_bytes():
        raise ValueError("STATUS PLANT-FIRED: ice restart input/output isolation failed")
    if rst_ocean_input.resolve() == rst_ocean_nc.resolve() or rst_ocean_input.read_bytes() != producer_ocean_restart_checked.read_bytes():
        raise ValueError("STATUS PLANT-FIRED: ocean restart input/output isolation failed")

    rst_snap_entry = stitch_o1_binary_snapshots(
        [read_o1_binary_snapshot(p) for p in rst_entry_files]
    )
    rst_snap_step1 = stitch_o1_binary_snapshots(
        [read_o1_binary_snapshot(p) for p in rst_step1_files]
    )
    if int(rst_snap_entry["meta"]["kt"]) != 0 or int(rst_snap_step1["meta"]["kt"]) != 2:
        raise ValueError("STATUS PLANT-FIRED: rst_1step metadata must be entry kt=0 and step kt=2")
    nemo_rst_entry, _ = o1_snapshot_to_si3_state(rst_snap_entry, config=cfg)
    nemo_rst_step1, _ = o1_snapshot_to_si3_state(rst_snap_step1, config=cfg)
    rst_sss_entry = jnp.asarray(rst_snap_entry["fields"]["sss_m"], dtype=jnp.float64)
    rst_sss_step2 = jnp.asarray(rst_snap_step1["fields"]["sss_m"], dtype=jnp.float64)

    lego_rst_entry = read_si3_netcdf_restart(
        producer_restart,
        tmask=tmask,
        sss_m=rst_sss_entry,
        config=cfg,
        ln_rstart=True,
    )
    lego_rst_step1 = o1_step_jit(lego_rst_entry, rst_sss_step2)

    if plant == "rst-1ulp":
        v_rst_np = np.asarray(lego_rst_step1.v_i).copy()
        v_rst_np[wi, wj, 0] = np.nextafter(v_rst_np[wi, wj, 0], np.inf)
        lego_rst_step1 = lego_rst_step1._replace(v_i=jnp.asarray(v_rst_np))

    cmp_rst_entry = _compare_si3_states_wet_and_full(
        nemo_rst_entry,
        lego_rst_entry,
        tmask,
        umask,
        vmask,
        check_fmass=True,
        check_prather=True,
    )
    cmp_rst_step1 = _compare_si3_states_wet_and_full(
        nemo_rst_step1,
        lego_rst_step1,
        tmask,
        umask,
        vmask,
        check_fmass=True,
        check_prather=True,
    )
    cmp_rst_vs_kt2 = _compare_si3_states_wet_and_full(
        nemo_kt2,
        lego_rst_step1,
        tmask,
        umask,
        vmask,
        check_fmass=True,
        check_prather=True,
    )

    # Verify serialized NetCDF restart variables match NEMO's ORCA2_SAS_00000001_restart_ice.nc at 0 ULP
    expected_rst_dict = si3_rst_to_netcdf_dict(state_kt1, kt=1, config=cfg)
    max_ulp_restart_variables = 0
    with netCDF4.Dataset(rec_dir / "ORCA2_SAS_00000001_restart_ice.nc", "r") as ds:
        for vname, exp_arr in expected_rst_dict.items():
            if vname not in ds.variables:
                raise ValueError(
                    f"STATUS PLANT-FIRED: missing variable {vname} in NEMO restart file"
                )
            raw = np.asarray(ds.variables[vname][...], dtype=np.float64)
            dims = tuple(ds.variables[vname].dimensions)
            if "time_counter" in dims and dims[0] == "time_counter":
                raw = raw[0]
                dims = dims[1:]
            if raw.ndim == 2 and dims == ("y", "x"):
                raw = raw.T
            elif raw.ndim == 3 and dims == ("numcat", "y", "x"):
                raw = np.transpose(raw, (2, 1, 0))
            m = (
                wet_mask_for_field(vname, tmask, umask, vmask)
                if raw.ndim >= 2
                else None
            )
            exp_scored, mask_scored = restart_payload_view(
                exp_arr, raw, m, nn_hls=cfg.nn_hls
            )
            ulp_nc = max_ulp_diff(exp_scored, raw, mask=mask_scored)
            max_ulp_restart_variables = max(max_ulp_restart_variables, ulp_nc)
            if ulp_nc != 0:
                raise ValueError(
                    f"STATUS PLANT-FIRED: NetCDF restart variable {vname} ULP={ulp_nc} != 0"
                )

    tmask_np = np.asarray(tmask) != 0.0
    h = cfg.nn_hls
    interior_t = np.zeros_like(tmask_np)
    interior_t[h:-h, h:-h] = True
    i_idx = np.arange(tmask_np.shape[0])[:, None]
    j_idx = np.arange(tmask_np.shape[1])[None, :]
    seam_t = ((i_idx < h) | (i_idx >= tmask_np.shape[0] - h)) & (
        (j_idx >= h) & (j_idx < tmask_np.shape[1] - h)
    )
    fold_t = np.broadcast_to(j_idx >= tmask_np.shape[1] - h, tmask_np.shape)
    state_arrays_checked = (
        len(_PROG_FIELDS)  # t1_ice excluded, snwice_fmass included: net zero
        + len(SI3AllPratherMoments._fields) * len(SI3PratherMoments._fields)
        + len(state_kt1.equivalent._fields)
        + len(state_kt1.aggregated._fields)
    )

    return {
        "status": "PASS",
        "evidence_dir": str(ev),
        "producer_commit": recorded_commit,
        "gate_changed_since_run": gate_changed_since_run,
        "binaries_checked": binaries_checked,
        "inputs_checked": inputs_checked,
        "execution_backend": jax.default_backend(),
        "execution_jit": True,
        "execution_x64": bool(jax.config.jax_enable_x64),
        "first_wet_t_ij": [wi, wj],
        "dry_signed_zero_fixture": dry_fixture,
        "o2_arrays_checked": len(_O2_SCALAR_STATE_FIELDS),
        "state_arrays_checked_per_comparison": state_arrays_checked,
        "restart_variables_checked": len(expected_rst_dict),
        "wet_t_cell_counts": {
            "interior": int(np.count_nonzero(tmask_np & interior_t)),
            "seam": int(np.count_nonzero(tmask_np & seam_t)),
            "fold": int(np.count_nonzero(tmask_np & fold_t)),
        },
        "o2_area": o2_cmp["area"],
        "o2_zsshadj": o2_cmp["zsshadj"],
        "max_ulp_o2_istate": o2_cmp["max_ulp_wet"],
        "max_ulp_o2_istate_full": o2_cmp["max_ulp_full"],
        "max_ulp_kt1": cmp_kt1["max_ulp_wet"],
        "max_ulp_kt1_full": cmp_kt1["max_ulp_full"],
        "max_ulp_kt2": cmp_kt2["max_ulp_wet"],
        "max_ulp_kt2_full": cmp_kt2["max_ulp_full"],
        "max_ulp_rst_entry": cmp_rst_entry["max_ulp_wet"],
        "max_ulp_rst_entry_full": cmp_rst_entry["max_ulp_full"],
        "max_ulp_rst_step1": cmp_rst_step1["max_ulp_wet"],
        "max_ulp_rst_step1_full": cmp_rst_step1["max_ulp_full"],
        "max_ulp_rst_vs_kt2": cmp_rst_vs_kt2["max_ulp_wet"],
        "max_ulp_rst_vs_kt2_full": cmp_rst_vs_kt2["max_ulp_full"],
        "max_ulp_restart_variables": max_ulp_restart_variables,
        "max_norm_o2_istate": o2_cmp["max_norm_wet"],
        "max_norm_kt1": cmp_kt1["max_norm_wet"],
        "max_norm_kt2": cmp_kt2["max_norm_wet"],
        "max_norm_rst_step1": cmp_rst_step1["max_norm_wet"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SI3 Rung O2 gate and verifier.")
    parser.add_argument("--verify-layout-fixture", type=Path, default=None)
    parser.add_argument("--verify-log", type=Path, default=None)
    parser.add_argument("--expect-icedyn", choices=("T", "F"), default=None)
    parser.add_argument("--expect-icethd", choices=("T", "F"), default=None)
    parser.add_argument("--expect-rstart", choices=("T", "F"), default=None)
    parser.add_argument("--same-program", type=Path, default=None)
    parser.add_argument("--reference-bld", type=Path, default=None)
    parser.add_argument("--pra-patch", type=Path, default=None)
    parser.add_argument("--ist-patch", type=Path, default=None)
    parser.add_argument("--stp-patch", type=Path, default=None)
    parser.add_argument("--new-cfg", type=str, default="ORCA2_SAS_ICE_O2_REC")
    parser.add_argument("--reference-cfg", type=str, default="ORCA2_SAS_ICE_O2_REF")
    parser.add_argument("--admit-evidence", type=Path, default=None)
    parser.add_argument("--expect-commit", type=str, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.verify_layout_fixture is not None:
            report = verify_layout_fixture(args.verify_layout_fixture, plant=args.plant)
        elif args.verify_log is not None:
            exp_dyn = None if args.expect_icedyn is None else (args.expect_icedyn == "T")
            exp_thd = None if args.expect_icethd is None else (args.expect_icethd == "T")
            exp_rst = None if args.expect_rstart is None else (args.expect_rstart == "T")
            report = verify_ocean_output_log(
                args.verify_log,
                expect_icedyn=exp_dyn,
                expect_icethd=exp_thd,
                expect_rstart=exp_rst,
                plant=args.plant,
            )
        elif args.same_program is not None:
            if (
                args.reference_bld is None
                or args.pra_patch is None
                or args.ist_patch is None
                or args.stp_patch is None
            ):
                parser.error(
                    "--same-program requires --reference-bld, --pra-patch, --ist-patch, and --stp-patch"
                )
            report = same_program(
                args.same_program,
                args.reference_bld,
                args.pra_patch,
                args.ist_patch,
                args.stp_patch,
                new_cfg=args.new_cfg,
                ref_cfg=args.reference_cfg,
            )
        elif args.admit_evidence is not None:
            report = admit_evidence_directory(
                args.admit_evidence,
                plant=args.plant,
                expect_commit=args.expect_commit,
            )
        elif args.self_test:
            report = run_self_test(plant=args.plant)
        else:
            parser.error(
                "Specify one of --verify-layout-fixture, --verify-log, --same-program, --admit-evidence, or --self-test"
            )
    except Exception as exc:
        print(f"REFUSE: GATE_FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
