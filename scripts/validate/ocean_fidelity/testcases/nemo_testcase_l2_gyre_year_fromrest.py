#!/usr/bin/env python3
"""The GYRE from-rest YEAR and its ensemble spread floor.

Preregistration: ``docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_fromrest.md``.
Read it first; every constant below is fixed there and none of them is a
judgement call made at scoring time.

This is the GYRE counterpart of the DINO from-rest protocol.  The question is
not "do the ten-step operators match" -- the kt=1..10 ladder answers that -- but
"after a YEAR from rest, is what separates legoESM from NEMO bigger than what
separates each model from ITSELF under an infinitesimal initial perturbation".

PHASES
  --member SEED     one legoESM member, 360 days, snapshots every 30 days
  --score-phase0    the legoESM-only floor; writes phase0_floor.json
  --score           the full verdict against the NEMO members
  --figures         surface and subsurface maps at day 30 and day 360
  --self-check      the arithmetic, with synthetic violations

PLANTS (each must exit non-zero, and each is exercised by the unit test)
  --plant perturbation-zero    seed 1 returns an all-zero perturbation
  --plant perturbation-relative the perturbation scales with T instead of being absolute
  --plant floor-inflate        the floor is multiplied by 1e6
  --plant gap-zero             the gap is forced to zero
  --plant operand-mismatch     the card's latitude is offset from NEMO's mesh
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- constants ---
CASE = "GYRE-zco"
DT_S = 14400.0                 # namelist_cfg &namdom rn_Dt
YEAR_DAYS = 360                # nn_leapy = 30 -> nyear_len(1) = 360
STEPS_PER_DAY = int(round(86400.0 / DT_S))          # 6
YEAR_STEPS = YEAR_DAYS * STEPS_PER_DAY              # 2160
SNAP_DAYS = 30                                      # nn_stock = 180 steps
SNAP_STEPS = SNAP_DAYS * STEPS_PER_DAY              # 180
SCORED_DAYS = tuple(range(SNAP_DAYS, YEAR_DAYS + 1, SNAP_DAYS))
SEEDS = (0, 1, 2, 3)           # 0 is the unperturbed control, by definition

# NEMO cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183, copied verbatim.
PERT_AMPLITUDE_K = 1.0e-10
PERT_DEPTH_MULT = 73
PERT_LAT_SCALE = 1000.0
PERT_LAT_MULT = 179
PERT_SEED_MULT = 997

# The verdict rule.  Preregistered; never relaxed, only measured against.
VERDICT_FACTOR = 2.0
# n=4 gives a 1/sqrt(2(n-1)) = 41% relative standard error on a spread
# estimate (dino_1226/floor90_ensemble.py:71), so a ratio inside this band is
# reported as MARGINAL rather than as a verdict.
MARGINAL_BAND = (0.5, 2.0)

# Preregistered expectations, section 8.  Constants, not opinions.
P1_FLOOR_360_MAX_K = 1.0e-3
P2_FLOOR_GROWTH_MAX = 1.0e3
P3_GAP_MIN_K = 2.8e-3          # the measured 40-hour gap
P3_GAP_MAX_K = 3.0
BOUNDED_OFFSET_RATIO = 10.0

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest")
DEFAULT_NEMO_MESH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/mesh_mask.nc")
DAILY_RESET_FAMILIES = (
    "control", "tracer", "temperature", "salinity", "vector", "ssh", "tke")
DAILY_RESET_VARIABLES = {
    "control": (),
    "tracer": ("tn", "sn"),
    "temperature": ("tn",),
    "salinity": ("sn",),
    "vector": ("un", "vn", "uu_n", "vv_n",
               "ub_e", "ubb_e", "vb_e", "vbb_e"),
    "ssh": ("sshn", "ssha", "sshb_e", "sshbb_e"),
    "tke": ("en", "avm_k", "avt_k", "dissl"),
}
DAILY_RESET_PLANTS = (
    "daily-source-ulp", "daily-family-registry", "daily-cadence-registry")
# Depth bands exist because the floor and the gap otherwise live in DIFFERENT
# VOLUMES: the 1e-10 K seed is uniform to the bottom, while a year-1 from-rest
# gap lives in the top few hundred metres.  A whole-column rms therefore
# dilutes the floor relative to the gap by roughly sqrt(volume ratio) and
# biases the verdict toward DISTINGUISHABLE.  Both independent reviews of the
# preregistration raised this; the bands are the fix, and they cost nothing
# because they come from the same saved states.
DEPTH_BANDS = (("0_100", 0.0, 100.0), ("100_1000", 100.0, 1000.0),
               ("1000P", 1000.0, 1.0e9))
FIELD_ROWS = (("T3D", "S3D", "SST", "SSH")
              + tuple(f"T3D_{name}" for name, _, _ in DEPTH_BANDS))
# PSI is reported SIGNED, as two rows.  A double gyre has a subtropical and a
# subpolar cell; max|PSI| reports one of them, is blind to the other, and is
# blind to a sign flip.
SCALAR_ROWS = ("PSI_MAX", "PSI_MIN", "QNET")
ROW_UNITS = {"T3D": "K", "S3D": "g/kg", "SST": "K", "SSH": "m",
             "PSI_MAX": "Sv", "PSI_MIN": "Sv", "QNET": "W"}
ROW_UNITS.update({f"T3D_{name}": "K" for name, _, _ in DEPTH_BANDS})
# A cell-count diagnostic, not a verdict row: how many wet cells hold a
# temperature difference above 1 mK.  A threshold process (NEMO's ln_zdfevd
# convective switch, rn_evd = 100 m2/s, a 1e7 jump on one branch) converts a
# rounding-level perturbation into a finite difference in ONE cell in ONE
# step, and that shows up here as a cell COUNT long before it shows up in an
# rms.
CELL_COUNT_THRESHOLD_K = 1.0e-3

# The ALIGNMENT GATE.  Every number in this file is an index-by-index
# difference between two models, so a frame error does not raise -- it returns
# a plausible number.  Three checks close the three readers, and each one is
# exercised by a plant that must turn it red:
#   A1  legoESM's initial state against NEMO's BEFORE level at the entry of
#       step 1, BIT-EXACT (0 cells unequal, not "inside a tolerance").  This
#       binds the BINARY record reader and the certified gate's own masks.
#   A2  the card's latitude and depth against NEMO's mesh_mask, to 0.0.  This
#       binds the MESH reader (reconcile_operands, already enforced).
#   A3  orientation discrimination on the netCDF RESTART reader, which neither
#       A1 nor A2 touches: the identity index mapping must beat every axis
#       reversal by a wide margin.  A TRANSPOSED mapping cannot reach this
#       check -- the array is 22x32 and a transpose raises on shape.
DEFAULT_ENTRY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/"
    "oracle_step_entry_kt00000001.bin")
# A3's thresholds are on a GEOMETRY IDENTITY, never on the models agreeing.
# The restart stores nav_lat/nav_lon in float32, so the identity mapping must
# reproduce the mesh's own gphit/glamt to f32 rounding (1.19e-7 x ~50 deg =
# 6e-6); a reversal misses by the domain's own extent, about 20 degrees.  The
# ratio is what is actually required, so the check cannot pass vacuously on a
# domain too small to discriminate.
FRAME_COORD_TOL_DEG = 1.0e-4
FRAME_COORD_RATIO = 1.0e4
# The figure depths are NOT a free choice: they are the preregistered depth
# bands' own boundaries (section 4), so the maps and the scored rows localise
# a difference at the same two depths.
FIGURE_DEPTHS_M = (100.0, 1000.0)


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate_module():
    """The certified GYRE phase-3 gate, loaded (never copied) for its stepping.

    Rule 10: the year is stepped by exactly the functions the kt=1..10 ladder
    steps this card with -- ``_surface_forcings``, ``expected_masks`` and
    ``lego_fields`` -- so a year cannot silently run a different program from
    the ladder that certified it.  The file's SHA-256 goes in every artifact.
    """
    path = Path(__file__).with_name("nemo_testcase_l2_gyre_phase3_gate.py")
    require(path.is_file(), f"missing {path}")
    spec = importlib.util.spec_from_file_location("gyre_phase3_gate", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha256(path)


# ------------------------------------------------------------ perturbation ---
def _nint(values):
    """Fortran ``NINT``: round half AWAY FROM ZERO.

    ``np.round`` is round-half-to-EVEN.  ``gphit*1000`` lands on exact halves
    on a regular grid, so the two disagree on real cells; this is not pedantry.
    """
    values = np.asarray(values, dtype=np.float64)
    return np.trunc(values + np.copysign(0.5, values))


def nemo_istate_perturbation(t_depth_m, lat_deg, tmask, seed: int, *,
                             plant: str | None = None):
    """NEMO's ``nn_pert_seed`` temperature perturbation, statement for statement.

    ``cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183``::

        IF( nn_pert_seed /= 0 ) THEN
           pts(:,:,jk,jp_tem) = pts(:,:,jk,jp_tem) + 1.e-10_wp
              * SIN( REAL( NINT(pdept(:,:,jk))*73
                         + NINT(gphit(:,:)*1000._wp)*179
                         + nn_pert_seed*997, wp ) ) * ptmask(:,:,jk)

    GYRE does not ship this block; ``nemo_testcase_l2_gyre_year_fromrest_members/``
    carries the additive MY_SRC patch that adds it, and the control member
    proves the seed-0 path is byte-identical to the unperturbed run.

    ``seed = 0`` returns exactly zero, as NEMO's ``IF`` guard does.
    """
    shape = np.broadcast(np.asarray(t_depth_m), np.asarray(lat_deg),
                         np.asarray(tmask)).shape
    if plant == "perturbation-zero":
        return np.zeros(shape)
    if seed == 0:
        return np.zeros(shape)
    argument = (_nint(t_depth_m) * PERT_DEPTH_MULT
                + _nint(np.asarray(lat_deg) * PERT_LAT_SCALE) * PERT_LAT_MULT
                + seed * PERT_SEED_MULT)
    amplitude = PERT_AMPLITUDE_K
    field = amplitude * np.sin(argument) * np.asarray(tmask)
    if plant == "perturbation-relative":
        field = field * 20.0          # a relative-looking perturbation
    return field


def assert_perturbation_properties(pert, lat_deg, tmask) -> dict:
    """The two properties that are easy to get wrong, asserted not assumed.

    1.  1e-10 K ABSOLUTE.  About 5e5 ulp on a ~20 K field: far above fp64
        rounding, physically meaningless, never written to a file.
    2.  NOT zonally uniform -- GYRE's grid is rotated 45 degrees
        (``usrdef_hgr.F90:125-126``) so ``gphit`` varies along a row by ~21
        degrees.  DINO's zonal-uniformity assertion is FALSE here and copying
        it would have measured the rotation, not the perturbation.  What the
        expression really implies is that at fixed level the field is constant
        within each ``NINT(gphit*1000)`` group.  That is what is checked, plus
        a non-degeneracy check so the assertion cannot pass vacuously.
    """
    pert = np.asarray(pert, dtype=np.float64)
    wet = np.asarray(tmask, dtype=bool)
    values = pert[wet]
    require(values.size > 0, "perturbation: empty wet mask")
    peak = float(np.max(np.abs(values)))
    require(peak <= PERT_AMPLITUDE_K * (1.0 + 1e-12),
            f"perturbation amplitude {peak:.6e} exceeds the absolute "
            f"{PERT_AMPLITUDE_K:.1e} K NEMO applies")
    require(peak > 0.1 * PERT_AMPLITUDE_K,
            f"perturbation peak {peak:.6e} is far below the {PERT_AMPLITUDE_K:.1e} K "
            "NEMO applies; the seed did not reach the field")
    groups = np.broadcast_to(
        _nint(np.asarray(lat_deg) * PERT_LAT_SCALE), pert.shape)
    worst, n_groups = 0.0, 0
    nlev = pert.shape[-1]
    for level in range(nlev):
        active = wet[..., level]
        if not active.any():
            continue
        keys = groups[..., level][active]
        vals = pert[..., level][active]
        for key in np.unique(keys):
            member = vals[keys == key]
            n_groups += 1
            worst = max(worst, float(np.max(member) - np.min(member)))
    require(worst == 0.0,
            f"perturbation is not constant within a NINT(gphit*1000) group "
            f"(worst spread {worst:.3e}); the transcription does not match "
            "the NEMO expression")
    distinct = int(np.unique(np.round(values, 20)).size)
    require(n_groups > 1 and distinct > 1,
            "the group-constancy assertion is vacuous: "
            f"{n_groups} groups, {distinct} distinct values")
    zonal = 0.0
    for level in range(nlev):
        active = wet[..., level]
        if not active.any():
            continue
        rows = pert[..., level]
        for j in range(rows.shape[0]):
            row = rows[j][active[j]]
            if row.size:
                zonal = max(zonal, float(np.max(row) - np.min(row)))
    return {
        "absolute_amplitude_K": PERT_AMPLITUDE_K,
        "measured_peak_K": peak,
        "n_latitude_groups": n_groups,
        "n_distinct_values": distinct,
        "max_within_group_spread": worst,
        "max_within_row_spread": zonal,
        "zonally_uniform": bool(zonal == 0.0),
        "expected_zonally_uniform": False,
        "source": "cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183",
    }


# --------------------------------------------------------------- operands ----
def nemo_operands(mesh_path: Path) -> dict:
    """NEMO's OWN gphit / gdept_0 / tmask, read from the mesh, not look-alikes."""
    import netCDF4

    require(Path(mesh_path).is_file(), f"missing NEMO mesh {mesh_path}")
    with netCDF4.Dataset(mesh_path) as mesh:
        gphit = np.asarray(mesh.variables["gphit"][0], dtype=np.float64)
        tmask = np.asarray(mesh.variables["tmask"][0], dtype=np.float64
                           ).transpose(1, 2, 0)
        name = "gdept_0" if "gdept_0" in mesh.variables else "gdept_1d"
        gdept = np.asarray(mesh.variables[name][:], dtype=np.float64)
        umask = np.asarray(mesh.variables["umask"][0], dtype=np.float64
                           ).transpose(1, 2, 0)
    if gdept.ndim == 4:
        gdept = gdept[0].transpose(1, 2, 0)
    elif gdept.ndim == 2:
        gdept = np.broadcast_to(gdept[0], gphit.shape + (gdept.shape[1],))
    return {"gphit": gphit, "gdept_0": gdept, "tmask": tmask, "umask": umask,
            "gdept_source": name, "mesh": str(mesh_path),
            "mesh_sha256": sha256(mesh_path)}


def reconcile_operands(card, mesh: dict, *, plant: str | None = None) -> dict:
    """Rule 1e: two readings of the same operand must CONVERGE before use.

    The card's latitude/depth and NEMO's mesh are independent readings of the
    same analytic definitions.  If they disagree, one of them has a bug and
    neither is recorded.
    """
    lat_card = np.asarray(card.recipe.grid.native_lat_T_deg, dtype=np.float64)
    if plant == "operand-mismatch":
        # One full unit of NINT(gphit*1000).  A smaller offset would leave the
        # ROUNDED operand unchanged and the plant would prove nothing -- which
        # is what the first version of it did.
        lat_card = lat_card + 1.0e-3
    dep_card = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    lat_mesh = mesh["gphit"]
    dep_mesh = mesh["gdept_0"][..., :dep_card.shape[-1]]
    require(lat_card.shape == lat_mesh.shape,
            f"latitude shape {lat_card.shape} vs mesh {lat_mesh.shape}")
    require(dep_card.shape == dep_mesh.shape,
            f"depth shape {dep_card.shape} vs mesh {dep_mesh.shape}")
    lat_err = float(np.max(np.abs(lat_card - lat_mesh)))
    dep_err = float(np.max(np.abs(dep_card - dep_mesh)))
    # The perturbation reads NINT(gphit*1000) and NINT(pdept), so what has to
    # agree is the ROUNDED operand, exactly.  A sub-rounding disagreement in
    # the raw value is recorded but cannot change the perturbation; a rounding
    # disagreement changes it completely.
    lat_round = int(np.count_nonzero(
        _nint(lat_card * PERT_LAT_SCALE) != _nint(lat_mesh * PERT_LAT_SCALE)))
    dep_round = int(np.count_nonzero(_nint(dep_card) != _nint(dep_mesh)))
    require(lat_round == 0 and dep_round == 0,
            f"the card and NEMO's mesh disagree on the ROUNDED perturbation "
            f"operands ({lat_round} latitudes, {dep_round} depths); the "
            "perturbation would not be NEMO's")
    return {"max_abs_latitude_deg": lat_err, "max_abs_depth_m": dep_err,
            "rounded_latitude_mismatches": lat_round,
            "rounded_depth_mismatches": dep_round,
            "gdept_source": mesh["gdept_source"],
            "mesh_sha256": mesh["mesh_sha256"]}


# ------------------------------------------------------------------ member ---
def _snapshot(state, gate) -> dict:
    fields = gate.lego_fields(state)
    return {key: np.asarray(value, dtype=np.float64)
            for key, value in fields.items()}


def _daily_restart_path(root: Path, step: int) -> Path:
    return Path(root) / f"GYRE_OMIP_L2_P3_{step:08d}_restart.nc"


def _load_daily_reset_payload(path: Path, nlev: int) -> dict:
    """Read one admitted daily restart with explicit axis contracts."""
    from netCDF4 import Dataset

    require(path.is_file(), f"missing admitted daily restart {path}")
    with Dataset(path, "r") as handle:
        require(int(np.asarray(handle.variables["kt"][...]))
                == int(path.name.split("_")[-2]),
                f"{path}: scalar kt disagrees with filename")

        def xyz(name: str) -> np.ndarray:
            variable = handle.variables[name]
            require(variable.dimensions == ("time_counter", "nav_lev", "y", "x"),
                    f"{path.name}/{name}: axes {variable.dimensions}")
            values = np.asarray(variable[0], dtype=np.float64).transpose(1, 2, 0)
            require(values.shape[-1] >= nlev,
                    f"{path.name}/{name}: {values.shape[-1]} levels < {nlev}")
            return values[..., :nlev]

        def xy(name: str) -> np.ndarray:
            variable = handle.variables[name]
            require(variable.dimensions == ("time_counter", "y", "x"),
                    f"{path.name}/{name}: axes {variable.dimensions}")
            return np.asarray(variable[0], dtype=np.float64)

        payload = {
            name: xyz(name) for name in
            ("tn", "sn", "un", "vn", "en", "avm_k", "avt_k", "dissl")
        }
        payload.update({
            name: xy(name) for name in
            ("uu_n", "vv_n", "sshn", "ssha", "ub_e", "ubb_e",
             "vb_e", "vbb_e", "sshb_e", "sshbb_e")
        })
    for name, values in payload.items():
        require(bool(np.all(np.isfinite(values))),
                f"{path.name}/{name}: non-finite daily reset operand")
    payload["path"] = str(path)
    payload["sha256"] = sha256(path)
    return payload


def _daily_record_contract(audit_path: Path, record_root: Path,
                           expect_commit: str) -> tuple[dict, dict[str, str]]:
    """Bind a reset run to an admitted record and its closed SHA manifest."""
    require(audit_path.is_file(), f"missing daily-record audit {audit_path}")
    audit = json.loads(audit_path.read_text())
    require(audit.get("status") == "ADMITTED" and audit.get("admitted") is True,
            f"{audit_path}: daily record is not ADMITTED")
    require(Path(audit["daily_root"]).resolve() == Path(record_root).resolve(),
            f"{audit_path}: admitted root {audit['daily_root']} != {record_root}")
    require(audit.get("tool", {}).get("commit") == expect_commit,
            f"{audit_path}: admission commit does not equal {expect_commit}")
    required = set(audit["requirements"]["required_variables"])
    expected = set().union(*map(set, DAILY_RESET_VARIABLES.values()))
    require(expected <= required,
            f"{audit_path}: admission omits reset variables {sorted(expected-required)}")
    stamp = audit.get("acquisition_stamp")
    require(stamp is not None and stamp.get("verified_file_count") == YEAR_DAYS,
            f"{audit_path}: acquisition SHA manifest was not verified")
    manifest_path = Path(stamp["manifest"])
    require(manifest_path.is_file(), f"missing admitted SHA manifest {manifest_path}")
    require(sha256(manifest_path) == stamp["manifest_sha256"],
            f"{manifest_path}: changed since record admission")
    entries = {}
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split()
        entries[name] = digest
    return audit, entries


def _require_payload_digest(payload: dict, expected_sha256: str, *,
                            plant: bool = False) -> None:
    observed = payload["sha256"]
    if plant:
        observed = ("0" if observed[0] != "0" else "1") + observed[1:]
    require(observed == expected_sha256,
            f"{Path(payload['path']).name}: payload SHA-256 {observed} != "
            f"admitted {expected_sha256}")


def validate_daily_reset_registry(registry=None) -> None:
    actual = DAILY_RESET_VARIABLES if registry is None else registry
    require(tuple(actual) == DAILY_RESET_FAMILIES,
            "daily reset family registry order/set changed")
    require(actual["control"] == (), "control arm must reset no variables")
    expected = {
        "tracer": ("tn", "sn"),
        "temperature": ("tn",),
        "salinity": ("sn",),
        "vector": ("un", "vn", "uu_n", "vv_n",
                   "ub_e", "ubb_e", "vb_e", "vbb_e"),
        "ssh": ("sshn", "ssha", "sshb_e", "sshbb_e"),
        "tke": ("en", "avm_k", "avt_k", "dissl"),
    }
    require(all(tuple(actual[name]) == values
                for name, values in expected.items()),
            "daily reset family registry omits or adds a source variable")


def validate_daily_reset_interval(interval_days: int) -> None:
    require(isinstance(interval_days, int) and not isinstance(interval_days, bool),
            "daily reset interval must be an integer number of days")
    require(1 <= interval_days < YEAR_DAYS,
            f"daily reset interval {interval_days} is outside [1, {YEAR_DAYS})")


def daily_reset_self_check(plant: str | None) -> int:
    """Failing controls for the source hash and exact family registry."""
    validate_daily_reset_registry()
    for interval_days in (1, 2, 4, 8):
        validate_daily_reset_interval(interval_days)
    if plant is None:
        _require_payload_digest(
            {"path": "synthetic.nc", "sha256": "a" * 64}, "a" * 64)
        print("SELF-CHECK OK: daily source digest, seven-family registry, "
              "and cadence registry")
        return 0
    try:
        if plant == "daily-source-ulp":
            _require_payload_digest(
                {"path": "synthetic.nc", "sha256": "a" * 64},
                "a" * 64, plant=True)
        elif plant == "daily-family-registry":
            broken = dict(DAILY_RESET_VARIABLES)
            broken["ssh"] = ("sshn", "sshb_e", "sshbb_e")
            validate_daily_reset_registry(broken)
        elif plant == "daily-cadence-registry":
            validate_daily_reset_interval(0)
        else:  # pragma: no cover - argparse constrains the value
            raise AssertionError(plant)
    except GateError as error:
        print(f"REFUSE Round-134 {plant}: {error}", file=sys.stderr)
        print(f"STATUS PLANT-FIRED: {plant}")
        return 1
    print(f"REFUSE Round-134 plant {plant} did not fire", file=sys.stderr)
    return 3


def _face_history(values: np.ndarray, face: str) -> np.ndarray:
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)

    values3 = np.asarray(values, dtype=np.float64)[..., None]
    if face == "u":
        return _u_east_to_face(values3)[..., 0]
    if face == "v":
        return _v_north_to_face(values3)[..., 0]
    raise ValueError(face)


def _load_kamm_twin_module():
    """Load the existing TKE restart bridge; never duplicate its mapping."""
    name = "_gyre_year_kamm_twin"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent.parent / "dino_1226" / "kamm_twin_90d.py"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load TKE restart bridge {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_daily_reset(state, payload: dict, family: str, card):
    """Replace exactly one registered family; return state and one-step ssha."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)

    require(family in DAILY_RESET_FAMILIES and family != "control",
            f"unknown reset family {family!r}")
    nlev = state.T.data.shape[-1]
    if family in ("tracer", "temperature", "salinity"):
        active = jnp.asarray(card.recipe.z_coord.is_active)
        replacements = {}
        if family in ("tracer", "temperature"):
            T = neumann_fill_cgrid(jnp.asarray(payload["tn"][..., :nlev]),
                                   active, card.recipe.grid)
            replacements["T"] = state.T.replace(data=T)
        if family in ("tracer", "salinity"):
            S = neumann_fill_cgrid(jnp.asarray(payload["sn"][..., :nlev]),
                                   active, card.recipe.grid)
            replacements["S"] = state.S.replace(data=S)
        return state._replace(**replacements), None
    if family == "vector":
        require(state.uu_b is not None and state.vv_b is not None,
                "vector reset requires live uu_b/vv_b state")
        require(state.bt_hist is not None and len(state.bt_hist) == 6,
                "vector reset requires six carried barotropic histories")
        hist = state.bt_hist
        new_hist = (
            jnp.asarray(_face_history(payload["ub_e"], "u")),
            jnp.asarray(_face_history(payload["ubb_e"], "u")),
            jnp.asarray(_face_history(payload["vb_e"], "v")),
            jnp.asarray(_face_history(payload["vbb_e"], "v")),
            hist[4], hist[5],
        )
        return state._replace(
            u=state.u.replace(data=jnp.asarray(
                _u_east_to_face(payload["un"][..., :nlev]))),
            v=state.v.replace(data=jnp.asarray(
                _v_north_to_face(payload["vn"][..., :nlev]))),
            uu_b=state.uu_b.replace(data=jnp.asarray(
                _face_history(payload["uu_n"], "u"))),
            vv_b=state.vv_b.replace(data=jnp.asarray(
                _face_history(payload["vv_n"], "v"))),
            bt_hist=new_hist,
        ), None
    if family == "ssh":
        require(state.bt_hist is not None and len(state.bt_hist) == 6,
                "ssh reset requires six carried barotropic histories")
        hist = state.bt_hist
        new_hist = (*hist[:4], jnp.asarray(payload["sshb_e"]),
                    jnp.asarray(payload["sshbb_e"]))
        return state._replace(
            eta=state.eta.replace(data=jnp.asarray(payload["sshn"])),
            bt_hist=new_hist), jnp.asarray(payload["ssha"])
    if family == "tke":
        require(all(getattr(state, name) is not None for name in (
            "tke", "tke_avm", "tke_avt", "tke_dissl", "tke_avm_surface")),
            "TKE reset requires live TKE state and coefficient memory")
        reset = _load_kamm_twin_module().bridge_tke_from_restart(
            state, payload["en"], np.asarray(state.land_mask.data),
            restart_avm=payload["avm_k"], restart_avt=payload["avt_k"],
            restart_dissl=payload["dissl"])
        return reset, None
    raise AssertionError(family)  # pragma: no cover


def run_member(seed: int, out_root: Path, *, days: int = YEAR_DAYS,
               mesh_path: Path = DEFAULT_NEMO_MESH, tag: str = "",
               plant: str | None = None, snap_steps: int = SNAP_STEPS,
               daily_reset_family: str | None = None,
               daily_reset_interval_days: int = 1,
               daily_record_root: Path | None = None,
               daily_record_audit: Path | None = None,
               expect_commit: str | None = None) -> int:
    """One legoESM member: from rest, ``days`` days, a snapshot every 30 days.

    ``snap_steps`` exists so a FINER record can be taken through the SAME
    stepping loop the scored members used, rather than by a second copy of it.
    Its default is the preregistered 180-step (30-day) cadence, so every
    scored member is byte-unchanged by this parameter; the day-by-day owner
    round passes 6 (one day) and writes into its own root.  The snapshot
    filename carries the DAY, so a cadence that is not a whole number of days
    is refused rather than silently rounding two steps onto one name."""
    require(snap_steps >= 1 and snap_steps % STEPS_PER_DAY == 0,
            f"snap_steps={snap_steps} is not a positive whole number of days "
            f"({STEPS_PER_DAY} steps); the snapshot name carries the day")
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    import jax.numpy as jnp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    producer_stamp = worktree_stamp()
    if expect_commit is not None:
        require(producer_stamp["commit"] == expect_commit,
                f"member commit {producer_stamp['commit']} != "
                f"--expect-commit {expect_commit}")
    if daily_reset_family is not None:
        validate_daily_reset_registry()
        validate_daily_reset_interval(daily_reset_interval_days)
        require(daily_reset_family in DAILY_RESET_FAMILIES,
                f"unknown daily reset family {daily_reset_family!r}")
        require(daily_reset_family != "control"
                or daily_reset_interval_days == 1,
                "control arm requires the one-day registered interval")
        require(seed == 0, "daily reset attribution is registered for member 0")
        require(days == YEAR_DAYS,
                "daily reset attribution requires the full 360-day member")
        require(snap_steps == STEPS_PER_DAY,
                "daily reset attribution requires six-step daily snapshots")
        require(expect_commit is not None,
                "daily reset attribution requires --expect-commit")
        require(daily_record_root is not None and daily_record_audit is not None,
                "daily reset attribution requires an admitted record and audit")
        record_audit, admitted_hashes = _daily_record_contract(
            daily_record_audit, daily_record_root, expect_commit)
    else:
        require(daily_reset_interval_days == 1,
                "--daily-reset-interval-days requires --daily-reset-family")
        record_audit, admitted_hashes = None, {}
    gate, gate_sha = _gate_module()
    card = build_nemo_testcase_card(CASE)
    require(card.dt_s == DT_S, f"card dt {card.dt_s} != {DT_S}")

    mesh = nemo_operands(mesh_path)
    operands = reconcile_operands(card, mesh, plant=plant)
    active = np.asarray(card.recipe.z_coord.is_active) & (
        np.asarray(card.recipe.land_mask) > 0.5)[..., None]
    depth3 = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    lat3 = np.broadcast_to(
        np.asarray(card.recipe.grid.native_lat_T_deg,
                   dtype=np.float64)[..., None], depth3.shape)
    pert = nemo_istate_perturbation(depth3, lat3, active.astype(np.float64),
                                    seed, plant=plant)
    properties = (assert_perturbation_properties(pert, lat3, active)
                  if seed != 0 else
                  {"absolute_amplitude_K": PERT_AMPLITUDE_K,
                   "measured_peak_K": 0.0,
                   "reason": "seed 0 is the unperturbed control, by definition"})
    require(seed != 0 or float(np.max(np.abs(pert))) == 0.0,
            "seed 0 produced a non-zero perturbation; the control is not the "
            "unperturbed path")

    state = card.recipe.initial_state
    state = state._replace(T=state.T.replace(
        data=jnp.asarray(np.asarray(state.T.data) + pert, dtype=jnp.float64)))
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    out = Path(out_root) / (f"lego_seed{seed}" + (f"_{tag}" if tag else ""))
    if daily_reset_family is not None:
        require(not out.exists() or not any(out.iterdir()),
                f"daily reset output {out} is not empty; refusing stale rows")
    out.mkdir(parents=True, exist_ok=True)
    n_steps = days * STEPS_PER_DAY
    started = time.time()
    reset_sources = []
    pending_ssha = None
    for completed in range(n_steps):
        kt = completed + 1
        freshwater, surface = gate._surface_forcings(card, state, kt)
        state = model.step(state, dt=card.dt_s,
                           freshwater=freshwater, surface_forcing=surface,
                           _nemo_stage1_zad_eta_after_override=pending_ssha)
        pending_ssha = None
        if kt % snap_steps == 0:
            arrays = _snapshot(state, gate)
            require(all(np.all(np.isfinite(value)) for value in arrays.values()),
                    f"seed {seed}: non-finite state at day {kt // STEPS_PER_DAY}")
            np.savez(out / f"day{kt // STEPS_PER_DAY:03d}.npz", **arrays)
            print(f"  seed {seed} day {kt // STEPS_PER_DAY:3d}  "
                  f"{time.time() - started:7.1f} s", flush=True)
        if (daily_reset_family not in (None, "control")
                and kt % (STEPS_PER_DAY * daily_reset_interval_days) == 0
                and kt < n_steps):
            path = _daily_restart_path(daily_record_root, kt)
            payload = _load_daily_reset_payload(path, state.T.data.shape[-1])
            expected_hash = admitted_hashes.get(path.name)
            require(expected_hash is not None,
                    f"{path.name}: absent from admitted acquisition manifest")
            _require_payload_digest(payload, expected_hash)
            state, pending_ssha = apply_daily_reset(
                state, payload, daily_reset_family, card)
            reset_sources.append({
                "step": kt, "day": kt // STEPS_PER_DAY,
                "path": str(path), "sha256": payload["sha256"],
            })
    manifest = {
        "format": "nemo-testcase-l2-gyre-year-fromrest-member-v1",
        "case": CASE, "seed": seed, "tag": tag, "days": days,
        "steps": n_steps,
        "dt_s": DT_S, "snapshot_step_interval": snap_steps,
        "snapshot_days": list(range(snap_steps // STEPS_PER_DAY, days + 1,
                                    snap_steps // STEPS_PER_DAY)),
        "phase3_gate_sha256": gate_sha,
        "perturbation": properties,
        "operands": operands,
        "wall_seconds": time.time() - started,
        "worktree": producer_stamp,
    }
    if daily_reset_family is not None:
        manifest["daily_reset"] = {
            "family": daily_reset_family,
            "registered_variables": list(
                DAILY_RESET_VARIABLES[daily_reset_family]),
            "interval_days": daily_reset_interval_days,
            "timing": "after pre-reset daily snapshot; consumed next step",
            "applied_boundary_count": len(reset_sources),
            "applied_boundaries": reset_sources,
            "record_root": str(daily_record_root),
            "record_audit": str(daily_record_audit),
            "record_audit_sha256": sha256(daily_record_audit),
            "record_producer_commit": (
                record_audit["acquisition_stamp"]["producer_commit"]),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"member": str(out), "wall_s": manifest["wall_seconds"]}))
    return 0


# ------------------------------------------------------------------- census --
# The fields watched by --census.  A THROWAWAY PROBE'S NUMBER IS UNMEASURED, so
# the probe that found the step-48 blocker lives here rather than in a heredoc:
# it is re-runnable against a changed model, and it FAILS LOUDLY instead of
# printing a table someone has to read.
CENSUS_FIELDS = ("eta", "u", "v", "w", "T", "S", "tke", "tke_avm", "tke_avt",
                 "tke_dissl", "tke_avm_surface", "uu_b", "vv_b")


def census(steps: int, *, every: int = 1, start: int = 1,
           corner: tuple[int, int, int] | None = None) -> int:
    """Step the card and report every prognostic field's range, per step.

    Exits NON-ZERO the moment any watched field stops being finite, and names
    the step and the field.  That is the whole point: the abort this was
    written for reports a vertical-geometry guard, which is several operators
    downstream of the field that actually diverged.
    """
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    gate, gate_sha = _gate_module()
    card = build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    print(f"# case={CASE} dt_s={card.dt_s} steps={steps} "
          f"phase3_gate_sha256={gate_sha}")
    state = card.recipe.initial_state
    _C_EPS = float(
        card.recipe.model_config.physics.vertical_mixing.tke.c_eps)
    corner_state = {"tke": None, "dissl": None}
    for kt in range(1, steps + 1):
        freshwater, surface = gate._surface_forcings(card, state, kt)
        try:
            state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                               surface_forcing=surface)
        except Exception as error:                       # noqa: BLE001
            print(f"STEP {kt} RAISED: {str(error).splitlines()[0][:200]}")
            print(f"STATUS ABORTED at step {kt} of {steps}")
            return 1
        cells, nonfinite = [], []
        # WHERE the peak TKE sits, not just how big it is.  An independent
        # review identified the runaway as the explicit half of the
        # dissipation split acting as a source, and predicted the peak would
        # sit on the DEEPEST active interface; that prediction is only
        # testable if the index is printed.  It also prints the implied
        # mixing length L = sqrt(en)/dissl, because NEMO's `dissl` is the RATE
        # sqrt(en)/L (zdftke.F90:717), not a length -- so L is already inside
        # any census that carries both fields.
        tke_field = getattr(state, "tke", None)
        if tke_field is not None:
            values = np.asarray(getattr(tke_field, "data", tke_field))
            flat = int(np.nanargmax(np.where(np.isfinite(values), values, -np.inf)))
            index = np.unravel_index(flat, values.shape)
            peak = float(values[index])
            dissl = getattr(state, "tke_dissl", None)
            length = float("nan")
            if dissl is not None:
                rate = float(np.asarray(getattr(dissl, "data", dissl))[index])
                if rate > 0.0 and peak > 0.0:
                    length = float(np.sqrt(peak) / rate)
            cells.append("argmax_tke(j=%d,i=%d,k=%d)=%.3e L=%.4f"
                         % (index[0], index[1], index[2], peak, length))
            if corner is not None:
                # P2's falsifier, measured on the CARRIED state only: the
                # deepest TKE row's step map.  NEMO's explicit half of the
                # dissipation split (zdftke.f90:422-425, zfact3=0.5*rn_ediss)
                # contributes dt*0.5*rn_ediss*dissl(n-1)*e(n-1); if that term
                # alone accounts for the increment, the row carries NO implicit
                # balance -- which is the claim.  ``dissl`` and ``tke`` are both
                # carried, so this needs no reach into the solver.
                jj, ii, kk = corner
                e_new = float(values[jj, ii, kk])
                d_old = corner_state["dissl"]
                e_old = corner_state["tke"]
                predicted = (float("nan") if d_old is None else
                             card.dt_s * 0.5 * _C_EPS * d_old * e_old)
                actual = (float("nan") if e_old is None else e_new - e_old)
                share = (float("nan") if not np.isfinite(actual) or actual == 0.0
                         else predicted / actual)
                ratio15 = (float("nan") if e_old is None or e_old <= 0.0
                           else (e_new - e_old) / e_old ** 1.5)
                print("CORNER kt=%d e_old=%.8e e_new=%.8e dissl_old=%s "
                      "d_predicted=%.8e d_actual=%.8e explicit_share=%.6f "
                      "de_over_e15=%.6f neighbour_k-1=%.8e"
                      % (kt, float("nan") if e_old is None else e_old, e_new,
                         "None" if d_old is None else "%.8e" % d_old,
                         predicted, actual, share, ratio15,
                         float(values[jj, ii, kk - 1])), flush=True)
                _d = getattr(state, "tke_dissl", None)
                corner_state["tke"] = e_new
                corner_state["dissl"] = (
                    None if _d is None
                    else float(np.asarray(getattr(_d, "data", _d))[jj, ii, kk]))
        for name in CENSUS_FIELDS:
            field = getattr(state, name, None)
            if field is None:
                continue
            values = np.asarray(getattr(field, "data", field))
            if values.dtype.kind != "f":
                continue
            bad = int(np.count_nonzero(~np.isfinite(values)))
            if bad:
                nonfinite.append((name, bad))
            # RAW min/max, so an `inf` PRINTS as inf.  The first version of
            # this line reduced over the finite subset only and so printed a
            # healthy-looking range for a half-infinite field -- the repo's
            # own "nanmax hides failures" rule, in its subtlest form, inside
            # the probe written to catch exactly this.  A test pins the fix by
            # source, so the discarded spelling must not appear anywhere in
            # this file, comments included.
            lo, hi = float(values.min()), float(values.max())
            cells.append(f"{name}[{lo:.3e},{hi:.3e}]"
                         + (f"!NONFINITE{bad}" if bad else ""))
        if kt >= start and (kt % every == 0 or nonfinite):
            print(f"kt={kt:4d} " + " ".join(cells), flush=True)
        if nonfinite:
            names = ", ".join(f"{name} ({bad} cells)" for name, bad in nonfinite)
            print(f"STATUS NONFINITE at step {kt}: {names}")
            return 1
    print(f"STATUS FINITE through step {steps}")
    return 0



def ladder_dump(steps: int, out: Path) -> int:
    """Write every prognostic field at kt=1..steps, for a BIT comparison.

    Rule 12 asks whether a change leaves the certified short ladder
    bit-identical.  Printed ranges cannot answer that -- three decimal digits
    hide everything below 1e-3 -- so this writes the arrays and the comparison
    is ``np.array_equal`` on the two files.  Same card, same forcing, same
    fp64/libm policy as :func:`census` and :func:`run_member`.
    """
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    gate, gate_sha = _gate_module()
    card = build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    arrays = {}
    for kt in range(1, steps + 1):
        freshwater, surface = gate._surface_forcings(card, state, kt)
        state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
        for name in CENSUS_FIELDS:
            field = getattr(state, name, None)
            if field is None:
                continue
            values = np.asarray(getattr(field, "data", field))
            if values.dtype.kind != "f":
                continue
            arrays[f"kt{kt:03d}_{name}"] = values
    require(bool(arrays), "ladder dump captured no field")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"# ladder dump {out} fields={len(arrays)} steps={steps} "
          f"phase3_gate_sha256={gate_sha}")
    return 0


# ------------------------------------------------------------------ scoring --
def _rms(values, wet) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values[wet] ** 2)))


def _psi_field_sv(u3d, dz, dy):
    """Barotropic streamfunction [Sv], identical formula on both models.

    Depth-integrated zonal transport, accumulated meridionally.  Reference
    thicknesses are used on BOTH sides, so the free-surface contribution --
    ~1e-4 of the column -- cancels out of the comparison rather than being
    modelled differently in each.
    """
    u3d = np.asarray(u3d, dtype=np.float64)
    transport = np.sum(u3d * dz[None, None, :], axis=2) * dy
    return np.cumsum(transport, axis=0) / 1.0e6


def _qnet_w(arrays, day, card, area, wet2) -> float:
    """Area-integrated net surface heat flux [W], from the saved state.

    ``qns`` is a 40 W/m2/K restoring to GYRE's analytic ``t_star``
    (``usrdef_sbc.F90:118-120``), so it is a function of the model's own SST.
    That makes this an INTEGRAL, budget-closing discriminator that no rms of
    a field provides, and it is the quantity a surface-flux or top-cell
    mismatch shows up in first.
    """
    from legoesm.ocean.eos import nemo_potential_temperature_from_conservative
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_qns
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        gyre_surface_boundary_condition)
    import numpy as _np

    sbc = gyre_surface_boundary_condition(card, float(day) * 86400.0)
    sst = _np.asarray(arrays["T"][..., 0], dtype=_np.float64)
    sss = _np.asarray(arrays["S"][..., 0], dtype=_np.float64)
    sst_m = _np.asarray(
        nemo_potential_temperature_from_conservative(sst, sss), dtype=_np.float64)
    qns = _np.asarray(nemo_gyre_qns(sst, sst_m, _np.asarray(sbc.t_star_c),
                                    _np.asarray(sbc.qsr_w_m2),
                                    _np.asarray(sbc.emp_kg_m2_s)),
                      dtype=_np.float64)
    total = qns + _np.asarray(sbc.qsr_w_m2, dtype=_np.float64)
    return float(_np.sum(total[wet2] * area[wet2]))


def _fields(arrays, wet3, wet2, dz, dy, band_masks=None, day=None,
            card=None, area=None) -> dict:
    psi = _psi_field_sv(arrays["u"], dz, dy)
    out = {
        "T3D": (arrays["T"], wet3),
        "S3D": (arrays["S"], wet3),
        "SST": (arrays["T"][..., 0], wet2),
        "SSH": (arrays["ssh"], wet2),
        "PSI_MAX": (float(np.max(psi)), None),
        "PSI_MIN": (float(np.min(psi)), None),
    }
    if band_masks is not None:
        for name, mask in band_masks.items():
            out[f"T3D_{name}"] = (arrays["T"], mask)
    if card is not None:
        out["QNET"] = (_qnet_w(arrays, day, card, area, wet2), None)
    return out


def _distance(row: str, left, right, mask) -> float:
    if row in SCALAR_ROWS:
        return float(abs(left - right))
    return _rms(np.asarray(left) - np.asarray(right), mask)


def _load_lego(root: Path, seed: int, day: int) -> dict:
    path = Path(root) / f"lego_seed{seed}" / f"day{day:03d}.npz"
    require(path.is_file(), f"missing legoESM snapshot {path}")
    with np.load(path) as handle:
        return {key: np.asarray(handle[key], dtype=np.float64)
                for key in handle.files}


def _load_nemo(root: Path, seed: int, day: int, nlev: int, *,
               directory: Path | None = None) -> dict:
    import netCDF4

    step = day * STEPS_PER_DAY
    directory = (Path(root) / f"nemo_seed{seed}"
                 if directory is None else Path(directory))
    matches = sorted(directory.glob(f"*_{step:08d}_restart.nc"))
    require(len(matches) == 1,
            f"expected exactly one NEMO restart for seed {seed} step {step} "
            f"in {directory}, found {len(matches)}")
    with netCDF4.Dataset(matches[0]) as handle:
        recorded = int(np.asarray(handle.variables["kt"][...]))
        require(recorded == step,
                f"{matches[0]}: kt={recorded}, expected {step}")

        def xyz(name):
            # The transpose is an API, so it is READ, not assumed: a restart
            # written with a different axis order would otherwise be silently
            # reindexed.  Shape alone would catch a scramble here (22, 32 and
            # 31 are all distinct) but not on a square domain.
            require(handle.variables[name].dimensions
                    == ("time_counter", "nav_lev", "y", "x"),
                    f"{name}: axes {handle.variables[name].dimensions}, "
                    "expected (time_counter, nav_lev, y, x)")
            return np.asarray(handle.variables[name][0],
                              dtype=np.float64).transpose(1, 2, 0)[..., :nlev]

        # restart.F90:176-182 writes 'tn'/'sn'/'un'/'vn'/'sshn' from the Kbb
        # slot in the RK3 branch, and the swap Nbb <- Naa happens before
        # rst_write in the step routine THIS CARD COMPILES --
        # cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/MY_SRC/stprk3.F90:237 and :280.
        # (The shipped src/OCE/stprk3.F90 is the same code at :213 and :256;
        # a review read the shipped file and flagged the citation as off by
        # 43 lines, which is exactly the offset between the two.  The card
        # runs its MY_SRC copy, so that is the one cited.)
        # So the file stamped kt=n is the state AFTER n completed steps, which
        # is legoESM after n model.step calls.  Measured, not assumed: see the
        # preregistration's section 0.
        fields = {"T": xyz("tn"), "S": xyz("sn"), "u": xyz("un"),
                  "v": xyz("vn"),
                  "ssh": np.asarray(handle.variables["sshn"][0],
                                    dtype=np.float64)}
        require(handle.variables["sshn"].dimensions
                == ("time_counter", "y", "x"),
                f"sshn: axes {handle.variables['sshn'].dimensions}")
        # legoESM's own snapshots are checked for finiteness as they are
        # written; NEMO's were not checked at all, so a blown-up oracle member
        # would have entered the floor as a plausible number.
        for name, values in fields.items():
            require(bool(np.all(np.isfinite(values))),
                    f"{matches[0]}: NEMO field {name} is not finite "
                    f"({int(np.count_nonzero(~np.isfinite(values)))} cells)")
        fields.update({"path": str(matches[0]), "sha256": sha256(matches[0])})
        return fields


def ensemble_provenance(root: Path) -> dict:
    """ONE MODEL VERSION PER ENSEMBLE, read from the members' own manifests.

    The report's own stamp records the SCORING tree, not the tree each member
    ran in, so an ensemble split across two commits would produce a floor that
    is partly a CODE DIFFERENCE and there would be no red anywhere.  Raised by
    an independent review of this round -- in a round whose entire provenance
    problem was exactly that.
    """
    members = {}
    for seed in SEEDS:
        manifest = Path(root) / f"lego_seed{seed}" / "manifest.json"
        require(manifest.is_file(), f"missing member manifest {manifest}")
        record = json.loads(manifest.read_text())
        members[str(seed)] = {
            "commit": record["worktree"]["commit"],
            "clean": bool(record["worktree"]["clean"]),
            "phase3_gate_sha256": record["phase3_gate_sha256"],
            "steps": int(record["steps"]), "dt_s": float(record["dt_s"])}
    commits = sorted({row["commit"] for row in members.values()})
    require(len(commits) == 1,
            f"the four legoESM members ran at {len(commits)} different "
            f"commits {commits}; their spread would be partly a code "
            "difference and the floor would not be a floor")
    dirty = [seed for seed, row in members.items() if not row["clean"]]
    require(not dirty,
            f"legoESM members {dirty} ran on a DIRTY worktree; the commit does "
            "not identify the code that produced them")
    gates = sorted({row["phase3_gate_sha256"] for row in members.values()})
    require(len(gates) == 1,
            f"the members were stepped by {len(gates)} different versions of "
            "the certified gate")
    steps = sorted({row["steps"] for row in members.values()})
    require(steps == [YEAR_STEPS],
            f"the members carry {steps} steps, not [{YEAR_STEPS}]")
    return members


def _ensemble_spread(prepared: list[dict], row: str) -> float:
    """MAX over the within-ensemble pairs -- the sample RANGE, at n=4 about 2x
    a standard deviation BY CONSTRUCTION.  Recorded so the floor is never
    quoted as if it were a std."""
    worst = 0.0
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            worst = max(worst, _distance(row, prepared[i][row][0],
                                         prepared[j][row][0],
                                         prepared[i][row][1]))
    return worst


def _cells_over(left, right, mask) -> int:
    """How many wet cells hold a temperature difference above 1 mK."""
    difference = np.abs(np.asarray(left["T"]) - np.asarray(right["T"]))
    return int(np.count_nonzero(difference[mask] > CELL_COUNT_THRESHOLD_K))


def _geometry(card, mesh_path: Path):
    mesh = nemo_operands(mesh_path)
    nlev = card.recipe.z_coord.n_levels
    wet3 = np.asarray(mesh["tmask"][..., :nlev]) > 0.5
    wet2 = wet3[..., 0]
    dz = np.asarray(card.recipe.z_coord.dz_ref, dtype=np.float64)
    dy = np.asarray(card.recipe.grid.dy_u, dtype=np.float64)[:, 1:]
    area = (np.asarray(card.recipe.grid.dx_T, dtype=np.float64)
            * np.asarray(card.recipe.grid.dy_T, dtype=np.float64))
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0,
                       dtype=np.float64)[..., :nlev]
    bands = {name: wet3 & (depth >= lo) & (depth < hi)
             for name, lo, hi in DEPTH_BANDS}
    for name, mask in bands.items():
        require(bool(mask.any()), f"depth band {name} is empty")
    return mesh, wet3, wet2, dz, dy, area, bands


def score(root: Path, *, phase0_only: bool, mesh_path: Path = DEFAULT_NEMO_MESH,
          plant: str | None = None) -> dict:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, dz, dy, area, bands = _geometry(card, mesh_path)
    nlev = card.recipe.z_coord.n_levels
    _, gate_sha = _gate_module()

    def prepare(arrays, day):
        return _fields(arrays, wet3, wet2, dz, dy, band_masks=bands, day=day,
                       card=card, area=area)

    members = ensemble_provenance(root)
    commits = {row["commit"] for row in members.values()}

    rows, days, diagnostics = {}, [], {}
    dry_faces = np.asarray(mesh["umask"][..., :nlev]) > 0.5
    # ...and one NEMO binary per ensemble, by the same argument.
    nemo_members = {}
    if not phase0_only:
        for seed in SEEDS:
            stamp = Path(root) / f"nemo_seed{seed}" / "binary.sha256"
            require(stamp.is_file(), f"missing NEMO binary stamp {stamp}")
            nemo_members[str(seed)] = stamp.read_text().split()[0]
        require(len(set(nemo_members.values())) == 1,
                f"the four NEMO members ran different binaries "
                f"{sorted(set(nemo_members.values()))}")

    repro = _repro_floor(root, prepare, wet3)
    for day in SCORED_DAYS:
        lego = [_load_lego(root, seed, day) for seed in SEEDS]
        lego_p = [prepare(state, day) for state in lego]
        if not phase0_only:
            nemo = [_load_nemo(root, seed, day, nlev) for seed in SEEDS]
            nemo_p = [prepare(state, day) for state in nemo]
        days.append(day)
        # _psi_field_sv sums u over the whole column with no mask and cumsums
        # from the boundary row, so a non-zero velocity on a DRY face would
        # enter the streamfunction as transport.  Measured, not assumed, on
        # both models at every scored day (raised by review).
        for label, states in (("lego", lego),
                              *(() if phase0_only else (("nemo", nemo),))):
            for index, state in enumerate(states):
                leak = float(np.max(np.abs(
                    np.asarray(state["u"])[~dry_faces])))
                require(leak == 0.0,
                        f"day {day}: {label} member {index} carries "
                        f"{leak:.3e} m/s on a DRY u-face; the barotropic "
                        "streamfunction would integrate it as transport")
        diagnostics[str(day)] = {
            "dry_u_face_velocity_both_models": 0.0,
            "ensemble_cells_over_1mK_lego":
                max(_cells_over(lego[i], lego[j], wet3)
                    for i in range(len(lego)) for j in range(i + 1, len(lego))),
        }
        if not phase0_only:
            diagnostics[str(day)]["gap_cells_over_1mK"] = _cells_over(
                lego[0], nemo[0], wet3)
            diagnostics[str(day)]["ensemble_cells_over_1mK_nemo"] = max(
                _cells_over(nemo[i], nemo[j], wet3)
                for i in range(len(nemo)) for j in range(i + 1, len(nemo)))
        for row in FIELD_ROWS + SCALAR_ROWS:
            entry = rows.setdefault(row, {"unit": ROW_UNITS[row], "days": {}})
            spread_lego = _ensemble_spread(lego_p, row)
            record = {"spread_lego": spread_lego}
            if phase0_only:
                floor = spread_lego
                record["floor_basis"] = "legoESM only (PHASE 0)"
            else:
                spread_nemo = _ensemble_spread(nemo_p, row)
                floor = float(np.hypot(spread_lego, spread_nemo))
                record["spread_nemo"] = spread_nemo
                record["floor_basis"] = "sqrt(spread_lego^2 + spread_nemo^2)"
            if plant == "floor-inflate":
                floor = floor * 1.0e6
            record["floor"] = floor
            if not phase0_only:
                gap = _distance(row, lego_p[0][row][0], nemo_p[0][row][0],
                                lego_p[0][row][1])
                if plant == "gap-zero":
                    gap = 0.0
                across = [_distance(row, left[row][0], right[row][0],
                                    left[row][1])
                          for left in lego_p for right in nemo_p]
                # LIKE FOR LIKE.  The floor is the MAX over six within-model
                # pairs; scoring a SINGLE control-vs-control pair against it
                # compares an extreme of six with one draw and inflates the
                # floor by roughly 2x, biasing every row toward
                # INDISTINGUISHABLE.  An independent review of this diff
                # caught it.  The headline ratio therefore uses the max over
                # the sixteen cross-model pairs, the same order statistic as
                # the floor; the control-vs-control ratio is reported beside
                # it and never silently replaces it.
                gap_max = float(np.max(across))

                def classify(numerator):
                    if floor <= 0.0 or not np.isfinite(floor):
                        return float("inf"), "UNMEASURED_ZERO_FLOOR"
                    value = numerator / (VERDICT_FACTOR * floor)
                    if not np.isfinite(value):
                        return value, "UNMEASURED_NONFINITE"
                    if MARGINAL_BAND[0] <= value <= MARGINAL_BAND[1]:
                        return value, "MARGINAL"
                    return value, ("INDISTINGUISHABLE" if value <= 1.0
                                   else "DISTINGUISHABLE")

                ratio, verdict = classify(gap_max)
                ratio_control, verdict_control = classify(gap)
                record.update({
                    # PREREG section 6 defines gap_d as CONTROL vs CONTROL and
                    # relegates the pooled cross-ensemble distance to "a
                    # supporting column".  The preregistration is frozen, so
                    # the control pair is the verdict of record; the
                    # like-for-like max is reported beside it and both are
                    # printed.  A review of this scoring round caught the
                    # harness quoting the max as the headline.
                    "verdict_preregistered": verdict_control,
                    "ratio_preregistered": float(ratio_control),
                    "gap_control_pair": gap,
                    "gap_max_across_pairs": gap_max,
                    "pooled_across_min": float(np.min(across)),
                    "pooled_across_max": gap_max,
                    "ratio_gap_over_2floor": float(ratio),
                    "statistic": ("max over 16 cross-model pairs vs max over "
                                  "6 within-model pairs (like for like)"),
                    "ratio_control_pair_over_2floor": float(ratio_control),
                    "verdict_control_pair": verdict_control,
                    "verdict": verdict,
                })
            entry["days"][str(day)] = record

    # QNET IS NOT THE INDEPENDENT BUDGET-CLOSING ROW THE PREREGISTRATION
    # HOPED FOR, and this measures how far it falls short rather than leaving
    # the claim standing.  `_qnet_w` evaluates legoESM's OWN transcription of
    # NEMO's restoring on each model's saved SST; NEMO's own `qns` is not in
    # the members' output.  With a 40 W/m2/K restoring the row is therefore an
    # area-weighted rescaling of the SST difference, blind to a
    # forcing-transcription error.  Raised by an independent review; the
    # preregistration is frozen, so the row stays scored and is RELABELLED
    # here by its own measured correlation.
    qnet_note = {"status": "UNMEASURED", "reason": "PHASE 0 has no gap"}
    if not phase0_only:
        ordered = [str(day) for day in days]
        q = np.array([rows["QNET"]["days"][day]["gap_control_pair"]
                      for day in ordered])
        s = np.array([rows["SST"]["days"][day]["gap_control_pair"]
                      for day in ordered])
        qnet_note = {
            "status": "MEASURED",
            "pearson_r_against_SST_gap": float(np.corrcoef(q, s)[0, 1]),
            "consequence": ("QNET is a rescaled SST row on this card, not an "
                            "independent budget-closing discriminator; it is "
                            "reported as such"),
            "why": ("qns is 40 W/m2/K restoring to the analytic t_star "
                    "(usrdef_sbc.F90:118-120) and is recomputed from each "
                    "model's own SST by legoESM's transcription"),
        }

    report = {
        "format": ("nemo-testcase-l2-gyre-year-fromrest-phase0-v1"
                   if phase0_only else
                   "nemo-testcase-l2-gyre-year-fromrest-verdict-v1"),
        "case": CASE, "seeds": list(SEEDS), "scored_days": days,
        "dt_s": DT_S, "year_steps": YEAR_STEPS,
        "verdict_factor": VERDICT_FACTOR, "marginal_band": list(MARGINAL_BAND),
        "phase3_gate_sha256": gate_sha,
        "mesh_sha256": mesh["mesh_sha256"],
        "lego_members": members,
        "lego_member_commit": sorted(commits)[0],
        "nemo_members": nemo_members,
        "floor_kind": (
            "INITIAL-CONDITION-PERTURBATION spread, not run-to-run "
            "irreproducibility.  On a flow that damps an IC perturbation the "
            "two are NOT the same thing and this one can collapse toward "
            "zero; see repro_floor for the same-binary measurement and the "
            "FLOOR_COLLAPSE classification"),
        "repro_floor": repro,
        "spread_statistic": ("max over the 6 within-ensemble pairwise "
                             "distances = the sample RANGE; at n=4 about 2x a "
                             "std by construction, so the 2*floor bar is "
                             "roughly 3.4 sigma once the two models' ranges "
                             "are combined in quadrature"),
        "relative_standard_error_of_spread": float(1.0 / np.sqrt(2 * (len(SEEDS) - 1))),
        "cell_count_threshold_K": CELL_COUNT_THRESHOLD_K,
        "diagnostics": diagnostics,
        "QNET_row_is_a_rescaled_SST_row": qnet_note,
        "rows": rows,
        "worktree": worktree_stamp(),
    }
    report.update(_expectations(rows, phase0_only=phase0_only, repro=repro))
    report["vacuity_gate"] = _vacuity(rows, phase0_only=phase0_only)
    return report


def _repro_floor(root: Path, prepare, wet3) -> dict:
    """The SAME-BINARY floor: seed 0 run twice under a different thread count.

    The IC-perturbation ensemble answers "how far apart does an infinitesimal
    initial difference put this model".  The owner's bar is "each model's own
    run-to-run spread", which on a NON-chaotic flow is a DIFFERENT and usually
    much smaller number.  Measuring both keeps the verdict honest: a zero
    reproducibility floor is itself the finding that the IC floor is the only
    floor available.
    """
    directory = Path(root) / "lego_seed0_repro"
    if not directory.is_dir():
        return {"status": "UNMEASURED",
                "reason": f"{directory} does not exist; run --member 0 --tag repro"}
    out = {"status": "MEASURED", "path": str(directory), "days": {}}
    for day in SCORED_DAYS:
        candidate = directory / f"day{day:03d}.npz"
        if not candidate.is_file():
            continue
        with np.load(candidate) as handle:
            other = {key: np.asarray(handle[key], dtype=np.float64)
                     for key in handle.files}
        base = _load_lego(root, 0, day)
        out["days"][str(day)] = {
            "T3D": _rms(base["T"] - other["T"], wet3),
            "bit_identical": bool(all(
                np.array_equal(base[key], other[key]) for key in base)),
        }
    return out


def _expectations(rows: dict, *, phase0_only: bool, repro: dict) -> dict:
    """Preregistered P1..P5, scored HELD or REFUTED.  Constants from section 8."""
    temperature = rows["T3D"]["days"]
    last, first = str(YEAR_DAYS), str(SNAP_DAYS)
    mid = str(YEAR_DAYS // 4)
    floor_360 = temperature[last]["floor"]
    out = {"P1_floor_360_below_1e-3_K": {
        "measured": floor_360, "bound": P1_FLOOR_360_MAX_K,
        "status": "HELD" if floor_360 < P1_FLOOR_360_MAX_K else "REFUTED"}}
    growth_floor = (floor_360 / temperature[first]["floor"]
                    if temperature[first]["floor"] > 0 else np.inf)
    out["P2_floor_growth_below_1e3"] = {
        "measured": float(growth_floor), "bound": P2_FLOOR_GROWTH_MAX,
        "status": "HELD" if growth_floor < P2_FLOOR_GROWTH_MAX else "REFUTED"}
    # A floor that DECAYS is the outcome that makes the whole verdict
    # uninformative: an initial-condition perturbation on a damped flow shrinks,
    # the floor collapses toward zero, and then ANY operator difference reads
    # DISTINGUISHABLE for a reason that has nothing to do with fidelity.  It is
    # classified here rather than discovered later.
    floor_30 = temperature[first]["floor"]
    out["floor_shape"] = {
        "floor_30": floor_30, "floor_360": floor_360,
        "classification": ("FLOOR_COLLAPSE" if floor_360 < floor_30 else
                           "FLOOR_GROWING"),
        "consequence": ("a collapsing floor makes gap/(2*floor) a statement "
                        "about the perturbation decaying, not about fidelity; "
                        "the repro_floor row and the maps carry the round in "
                        "that case"),
        "repro_floor_status": repro.get("status"),
    }
    if phase0_only:
        return out
    # PREREG section 6: the gap is control vs control.  An earlier version of
    # this function scored P3, P4 and the section-1 test on the max over the
    # 16 cross-model pairs; that is the supporting column, not the gap.
    gaps = {day: temperature[day]["gap_control_pair"] for day in temperature}
    gaps_max = {day: temperature[day]["gap_max_across_pairs"]
                for day in temperature}
    lo, hi = min(gaps.values()), max(gaps.values())
    out["P3_gap_band"] = {
        "statistic": "control vs control (PREREG section 6)",
        "min_over_days": lo, "max_over_days": hi,
        "min_over_days_max_pair": min(gaps_max.values()),
        "max_over_days_max_pair": max(gaps_max.values()),
        "band": [P3_GAP_MIN_K, P3_GAP_MAX_K],
        "status": ("HELD" if lo >= P3_GAP_MIN_K and hi <= P3_GAP_MAX_K
                   else "REFUTED")}
    verdicts = {day: temperature[day]["verdict_preregistered"]
                for day in temperature}
    # PREREG P4 is REFUTED "if any scored day gives gap_d <= 2*floor_d".  It
    # is NOT refuted by the MARGINAL label, which is this harness's own
    # sampling-error caveat and not part of the preregistered rule; scoring it
    # as a refutation would have made a ratio of 1.5 refute P4 in the code
    # while holding it in the document.  Caught by review.
    crossing = next((day for day in sorted(temperature, key=int)
                     if temperature[day]["ratio_preregistered"] <= 1.0), None)
    out["P4_distinguishable_every_day"] = {
        "verdicts": verdicts, "crossing_day": crossing,
        "rule": "REFUTED iff any scored day has gap <= 2*floor",
        "marginal_days": [day for day in temperature
                          if verdicts[day] == "MARGINAL"],
        "status": ("HELD" if all(
            temperature[day]["ratio_preregistered"] > 1.0
            for day in temperature) else "REFUTED")}
    g_gap = gaps[last] / gaps[mid] if gaps[mid] > 0 else np.inf
    g_floor = (temperature[last]["floor"] / temperature[mid]["floor"]
               if temperature[mid]["floor"] > 0 else np.inf)
    shape_ratio = g_floor / g_gap if g_gap > 0 else np.inf
    if shape_ratio >= BOUNDED_OFFSET_RATIO:
        shape = "BOUNDED_DETERMINISTIC_OFFSET"
    elif shape_ratio <= 1.0 / BOUNDED_OFFSET_RATIO:
        shape = "GAP_AMPLIFYING"
    else:
        shape = "CO_GROWING"
    out["section1_bounded_offset_test"] = {
        "G_gap_360_over_90": float(g_gap), "G_floor_360_over_90": float(g_floor),
        "ratio": float(shape_ratio), "classification": shape,
        "status": "HELD" if shape != "BOUNDED_DETERMINISTIC_OFFSET" else "REFUTED"}
    return out


def _vacuity(rows: dict, *, phase0_only: bool = True) -> dict:
    """The floor must be able to see anything at all -- on BOTH models.

    An earlier version ran only on PHASE 0, so NEMO's own spread entered the
    verdict unchecked: four bit-identical NEMO members would have given a
    floor made entirely of legoESM's spread and every row would have read
    DISTINGUISHABLE for a reason that has nothing to do with fidelity.
    Caught by an independent review of this scoring round.
    """
    days = rows["T3D"]["days"]
    zero_days = [day for day, record in days.items() if record["floor"] <= 0.0]
    out = {
        "floor_positive_every_day": not zero_days,
        "zero_floor_days": zero_days,
        "phase1_may_run": not zero_days,
        "reason": ("a zero floor means the seed never reached the state, and "
                   "would be read as DISTINGUISHABLE -- the exact false "
                   "positive this harness exists to avoid"),
    }
    if not phase0_only:
        for side in ("spread_lego", "spread_nemo"):
            dead = [day for day, record in days.items()
                    if record.get(side, 0.0) <= 0.0]
            out[f"{side}_positive_every_day"] = not dead
            out[f"{side}_zero_days"] = dead
        out["both_ensembles_live"] = (out["spread_lego_positive_every_day"]
                                      and out["spread_nemo_positive_every_day"])
    return out


# ----------------------------------------------------------- alignment gate --
def _frame_mappings():
    """The index mappings a silent frame error could be hiding in.

    Only reversals: a transpose is excluded by SHAPE (22 != 32), so it cannot
    silently pass and does not need a row here.  Recorded rather than argued,
    because "a transpose would raise" is a claim about the array, and the
    array's shape is asserted below.
    """
    return {
        "identity": lambda a: a,
        "reverse_j": lambda a: a[::-1],
        "reverse_i": lambda a: a[:, ::-1],
        "reverse_both": lambda a: a[::-1, ::-1],
    }


def alignment_gate(root: Path, *, mesh_path: Path = DEFAULT_NEMO_MESH,
                   entry_path: Path = DEFAULT_ENTRY,
                   plant: str | None = None) -> dict:
    """Prove the two models are compared cell for cell.  Never argue it."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    gate, gate_sha = _gate_module()
    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _, _, _, _ = _geometry(card, mesh_path)
    nlev = card.recipe.z_coord.n_levels

    # A2 -- the mesh reader.  Raises on any rounded-operand disagreement.
    operands = reconcile_operands(card, mesh, plant=plant)

    # A2b -- THE WET MASK, because latitude alone cannot see a diagonal shift.
    # An independent review of this round measured that GYRE's 45-degree
    # rotation makes the latitude EXACTLY invariant along the anti-diagonal:
    # in the interior overlap, |gphit[j+1,i-1] - gphit[j,i]| is 7.105e-15 deg.
    # So a (j+1,i-1) misread would pass A2, and A1 is horizontally blind, and
    # A3a compares two NEMO-side files that would shift together.  The MASK is
    # the operand that does see it -- GYRE's wet rectangle is 30x20 inside a
    # 32x22 array, so shifting it moves the closed-boundary ring -- and the
    # non-vacuity of that statement is MEASURED here, not asserted.
    nemo_wet = np.asarray(mesh["tmask"][..., :nlev]) > 0.5
    card_wet = np.asarray(gate.expected_masks(card)["T"], dtype=bool)
    if plant == "mask-shift":
        card_wet = np.roll(np.roll(card_wet, 1, axis=0), -1, axis=1)
    require(card_wet.shape == nemo_wet.shape,
            f"A2b: card mask {card_wet.shape} vs NEMO tmask {nemo_wet.shape}")
    differing = int(np.count_nonzero(card_wet != nemo_wet))
    require(differing == 0,
            f"A2b: the card's wet mask and NEMO's tmask disagree on "
            f"{differing} cells; the two models are not on the same domain")
    surface = nemo_wet[..., 0]
    shift_power = {}
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj == 0 and di == 0:
                continue
            rows, cols = surface.shape
            j0, j1 = max(0, dj), min(rows, rows + dj)
            i0, i1 = max(0, di), min(cols, cols + di)
            left = surface[j0:j1, i0:i1]
            right = surface[j0 - dj:j1 - dj, i0 - di:i1 - di]
            shift_power[f"j{dj:+d}_i{di:+d}"] = {
                "mask_cells_differing": int(np.count_nonzero(left != right)),
                "overlap_cells": int(left.size)}
    blind_shifts = [name for name, row in shift_power.items()
                    if row["mask_cells_differing"] == 0]
    require(not blind_shifts,
            f"A2b would be VACUOUS: the mask is invariant under {blind_shifts}, "
            "so it cannot discriminate that shift and the frame is not closed")

    # A1 -- the binary reader, against the state legoESM starts from.
    require(Path(entry_path).is_file(), f"missing NEMO entry record {entry_path}")
    oracle = gate.read_entry(Path(entry_path))
    require((oracle["kt"], oracle["Nbb"]) == (1, 1),
            f"{entry_path}: kt={oracle['kt']} Nbb={oracle['Nbb']}, "
            "expected the BEFORE level at the entry of step 1")
    initial = card.recipe.initial_state
    if plant == "alignment-initial":
        import jax.numpy as jnp
        # One cell, one ulp-scale nudge.  A1 is a BIT-EXACT check, so this is
        # the smallest possible synthetic violation and it must still be red.
        bumped = np.asarray(initial.T.data).copy()
        bumped[5, 5, 0] = np.nextafter(bumped[5, 5, 0], np.inf)
        initial = initial._replace(
            T=initial.T.replace(data=jnp.asarray(bumped)))
    candidate = gate.lego_fields(initial)
    masks = gate.expected_masks(card)
    identity = {}
    for field in ("T", "S", "u", "v", "ssh"):
        reference = (oracle[field] if field == "ssh"
                     else oracle[field][..., :nlev])
        mask = np.asarray(masks[field], dtype=bool)
        left = np.asarray(reference, dtype=np.float64)
        right = np.asarray(candidate[field], dtype=np.float64)
        require(left.shape == right.shape == mask.shape,
                f"A1 {field}: shapes {left.shape} / {right.shape} / {mask.shape}")
        require(bool(mask.any()), f"A1 {field}: empty mask")
        unequal = int(np.count_nonzero(left[mask] != right[mask]))
        identity[field] = {
            "cells": int(np.count_nonzero(mask)),
            "cells_unequal": unequal,
            "max_abs_difference": float(np.max(np.abs(left[mask] - right[mask])))
            if mask.any() else 0.0,
            "exact": unequal == 0,
        }
    unexact = [name for name, row in identity.items() if not row["exact"]]
    require(not unexact,
            "A1 FAILED: legoESM's initial state is not BIT-EXACT against "
            f"NEMO's before level at the entry of step 1 on {unexact}; the "
            "two models do not start from the same state")

    # WHAT A1 CANNOT SEE (Rule 2), MEASURED rather than reasoned about.  An
    # independent review of this gate showed that GYRE's day-0 state is
    # horizontally UNIFORM -- T and S are functions of depth alone and u, v,
    # ssh are identically zero -- and that NEMO's tmask is symmetric under
    # every axis reversal.  A1 therefore passes IDENTICALLY under all four
    # horizontal mappings and carries NO horizontal frame evidence.  An
    # earlier version of this file said A1 "binds the frame"; that is
    # RETRACTED.  A1 binds the VALUES and, because the day-0 profile is
    # monotonic in depth, the VERTICAL axis.  The horizontal frame is carried
    # entirely by A2 and A3a.
    mask_t = np.asarray(masks["T"], dtype=bool)
    reference_t = np.asarray(oracle["T"][..., :nlev], dtype=np.float64)
    horizontal = max(
        float(reference_t[..., k][mask_t[..., k]].max()
              - reference_t[..., k][mask_t[..., k]].min())
        for k in range(nlev) if mask_t[..., k].any())
    both = mask_t[..., 1:] & mask_t[..., :-1]
    vertical = float(np.max(np.abs(np.diff(reference_t, axis=-1)[both])))
    blindness = {
        "max_horizontal_spread_of_day0_T_K": horizontal,
        "day0_field_is_horizontally_uniform": horizontal == 0.0,
        "mask_symmetric_under": [name for name, fn in _frame_mappings().items()
                                 if np.array_equal(fn(mask_t), mask_t)],
        "max_level_to_level_step_of_day0_T_K": vertical,
        "sees": ("the VALUES both models start from, and -- because the day-0 "
                 "profile varies from level to level -- the VERTICAL axis"),
        "blind_to": ("the HORIZONTAL frame: a reversed j or i passes A1 "
                     "identically on this card.  A2 and A3a carry that."),
    }

    # A3 -- the netCDF restart reader.  A1 and A2 never touch it.
    #
    # A3a is the one that GATES, and it never looks at legoESM: every restart
    # carries its own nav_lat/nav_lon, so the reader's frame is tied to the
    # mesh's gphit/glamt by an IDENTITY, and A2 already ties that mesh to the
    # card.  A1 + A2 + A3a is then a chain of three exact measurements and the
    # conclusion does not depend on the two models agreeing about anything.
    #
    # An earlier version of A3 gated on "the identity index mapping minimises
    # the MODEL-MODEL difference by 10x".  That was a defect in the
    # instrument, not a check: at day 360 the real gap is 0.41 K against a
    # 1.5 K reversal signal, so a correct frame failed a 10x margin.  The
    # minimisation survives below as a DIAGNOSTIC, gating nothing.
    import netCDF4

    mappings = _frame_mappings()
    with netCDF4.Dataset(mesh_path) as handle:
        mesh_lat = np.asarray(handle.variables["gphit"][0], dtype=np.float64)
        mesh_lon = np.asarray(handle.variables["glamt"][0], dtype=np.float64)
    chain = {}
    for seed in SEEDS:
        for day in (SNAP_DAYS, YEAR_DAYS):
            step = day * STEPS_PER_DAY
            matches = sorted((Path(root) / f"nemo_seed{seed}").glob(
                f"*_{step:08d}_restart.nc"))
            require(len(matches) == 1,
                    f"A3a: seed {seed} step {step}: {len(matches)} restarts")
            with netCDF4.Dataset(matches[0]) as handle:
                lat = np.asarray(handle.variables["nav_lat"][:], dtype=np.float64)
                lon = np.asarray(handle.variables["nav_lon"][:], dtype=np.float64)
            if plant == "frame-flip":
                lat, lon = lat[::-1], lon[::-1]
            require(lat.shape == mesh_lat.shape,
                    f"A3a: restart nav_lat {lat.shape} vs mesh {mesh_lat.shape}")
            errors = {name: max(float(np.max(np.abs(fn(lat) - mesh_lat))),
                                float(np.max(np.abs(fn(lon) - mesh_lon))))
                      for name, fn in mappings.items()}
            worst_reversal = min(value for name, value in errors.items()
                                 if name != "identity")
            row = {"max_abs_coord_error_deg": errors,
                   "identity": errors["identity"],
                   "nearest_reversal": worst_reversal,
                   "ratio": (worst_reversal / errors["identity"]
                             if errors["identity"] > 0 else float("inf"))}
            chain[f"seed{seed}_day{day}"] = row
            require(errors["identity"] <= FRAME_COORD_TOL_DEG,
                    f"A3a seed {seed} day {day}: the restart's own nav_lat/"
                    f"nav_lon miss NEMO's mesh by {errors['identity']:.3e} deg "
                    f"under the identity mapping, over the "
                    f"{FRAME_COORD_TOL_DEG:.1e} f32-rounding bound; the "
                    "restart is being read in the wrong frame")
            require(row["ratio"] >= FRAME_COORD_RATIO,
                    f"A3a seed {seed} day {day}: the identity mapping beats "
                    f"the nearest reversal by only {row['ratio']:.3g}x, under "
                    f"{FRAME_COORD_RATIO:.0e}; this domain cannot discriminate "
                    "a reversal and the check would be vacuous")

    # A3b -- DIAGNOSTIC ONLY.  Reported because it is the thing a reader
    # expects to see, and because its MARGIN shrinking with the day is itself
    # a statement about how large the gap has become.
    orientation = {}
    for day in SCORED_DAYS:
        lego = _load_lego(root, 0, day)["T"]
        nemo = _load_nemo(root, 0, day, nlev)["T"]
        require(lego.shape == nemo.shape == wet3.shape,
                f"A3b day {day}: shapes {lego.shape} / {nemo.shape}")
        scores = {name: _rms(lego - fn(nemo), wet3)
                  for name, fn in mappings.items()}
        best = min(scores, key=scores.get)
        others = min(value for name, value in scores.items()
                     if name != "identity")
        orientation[str(day)] = {
            "rms_by_mapping": scores, "best": best,
            "margin_over_next": float(others / scores["identity"])
            if scores["identity"] > 0 else float("inf")}

    return {
        "format": "nemo-testcase-l2-gyre-year-fromrest-alignment-v1",
        "case": CASE,
        "status": "ALIGNED",
        "A1_day0_identity": {
            "entry_record": str(entry_path),
            "entry_sha256": sha256(Path(entry_path)),
            "time_level": oracle["registry_level"],
            "kt": int(oracle["kt"]), "Nbb": int(oracle["Nbb"]),
            "fields": identity,
            "binds": ("the binary record reader's VALUES and vertical axis "
                      "(36x26x31 Fortran order, cropped [2:-2,2:-2], "
                      "transposed to (j,i,k)) and the certified gate's masks"),
            "blind_spot": blindness,
        },
        "A2_operand_reconciliation": operands,
        "A2b_wet_mask_identity": {
            "cells_differing": differing,
            "wet_cells": int(np.count_nonzero(nemo_wet)),
            "total_cells": int(nemo_wet.size),
            "mask_discriminates_every_pm1_shift": shift_power,
            "binds": ("the HORIZONTAL SHIFT that latitude cannot see: GYRE's "
                      "rotation makes gphit invariant along the anti-diagonal "
                      "(7.105e-15 deg in the interior overlap), and the mask "
                      "is not"),
        },
        "A3a_restart_coordinate_chain": {
            "mappings": sorted(mappings),
            "identity_tolerance_deg": FRAME_COORD_TOL_DEG,
            "ratio_required": FRAME_COORD_RATIO,
            "restarts": chain,
            "binds": ("the netCDF restart reader ((t,z,j,i) -> (j,i,k)), by "
                      "the restart's OWN nav_lat/nav_lon against the mesh's "
                      "gphit/glamt.  legoESM does not enter this check.  A "
                      "TRANSPOSE cannot reach it either -- the array is "
                      f"{wet3.shape[0]}x{wet3.shape[1]} and would raise"),
        },
        "A3b_orientation_diagnostic": {
            "gates": False,
            "days": orientation,
            "note": ("the identity mapping minimising the MODEL-MODEL "
                     "difference is evidence about the frame only while the "
                     "gap is small; by day 360 it is not, which is why this "
                     "row gates nothing"),
        },
        "frames": {
            "nemo_restart_netcdf": "x=32, y=22, nav_lev=31 (no halo written)",
            "nemo_binary_record": "36x26x31 = the nn_hls=2 halo allocation",
            "compared_array": list(wet3.shape),
            "wet_cells_3d": int(np.count_nonzero(wet3)),
            "wet_cells_2d": int(np.count_nonzero(wet2)),
            "note": ("the closed-boundary ring is masked out of every scored "
                     "number by NEMO's OWN tmask; 30x20 of the 32x22 columns "
                     "are wet"),
        },
        "phase3_gate_sha256": gate_sha,
        "mesh_sha256": mesh["mesh_sha256"],
        "worktree": worktree_stamp(),
    }


# ------------------------------------------------------------------ figures --
def _level_for_depth(depth_1d, target_m: float) -> int:
    """The level whose centre is nearest ``target_m``, from NEMO's own depths."""
    return int(np.argmin(np.abs(np.asarray(depth_1d, dtype=np.float64) - target_m)))


def figures(root: Path, out_dir: Path,
            mesh_path: Path = DEFAULT_NEMO_MESH) -> dict:
    """The maps.  A DELIVERABLE of this round whatever the verdict says.

    Seven quantities x three columns (legoESM, NEMO, the difference) at day 30
    and at day 360: the three surface fields and, at each of the two
    preregistered depth-band boundaries, temperature and salinity.  The
    figure carries its own provenance -- both models' commits, the NEMO
    restart's SHA-256 and the mesh's -- because a map with no provenance is a
    picture, not a measurement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _, _, _, _ = _geometry(card, mesh_path)
    nlev = card.recipe.z_coord.n_levels
    depth_1d = np.asarray(card.recipe.z_coord.t_depth_ref,
                          dtype=np.float64)[:nlev]
    levels = [( _level_for_depth(depth_1d, target), target)
              for target in FIGURE_DEPTHS_M]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = worktree_stamp()
    written, provenance = {}, {}
    for day in (SNAP_DAYS, YEAR_DAYS):
        lego = _load_lego(root, 0, day)
        nemo = _load_nemo(root, 0, day, nlev)
        panels = [("SST", "degC", lego["T"][..., 0], nemo["T"][..., 0], wet2),
                  ("SSS", "g/kg", lego["S"][..., 0], nemo["S"][..., 0], wet2),
                  ("SSH", "m", lego["ssh"], nemo["ssh"], wet2)]
        for index, target in levels:
            actual = depth_1d[index]
            panels.append((f"T at {actual:.0f} m (level {index + 1}, "
                           f"nearest {target:.0f} m)", "degC",
                           lego["T"][..., index], nemo["T"][..., index],
                           wet3[..., index]))
            panels.append((f"S at {actual:.0f} m (level {index + 1}, "
                           f"nearest {target:.0f} m)", "g/kg",
                           lego["S"][..., index], nemo["S"][..., index],
                           wet3[..., index]))
        figure, axes = plt.subplots(len(panels), 3,
                                    figsize=(13.5, 2.5 * len(panels)))
        for row, (name, unit, left, right, mask) in enumerate(panels):
            masked_left = np.where(mask, left, np.nan)
            masked_right = np.where(mask, right, np.nan)
            difference = np.where(mask, left - right, np.nan)
            lo = float(np.nanmin([np.nanmin(masked_left), np.nanmin(masked_right)]))
            hi = float(np.nanmax([np.nanmax(masked_left), np.nanmax(masked_right)]))
            span = float(np.nanmax(np.abs(difference)))
            span = span if span > 0 else 1.0
            for column, (field, title, kwargs) in enumerate((
                    (masked_left, f"legoESM {name} [{unit}]",
                     {"cmap": "viridis", "vmin": lo, "vmax": hi}),
                    (masked_right, f"NEMO {name} [{unit}]",
                     {"cmap": "viridis", "vmin": lo, "vmax": hi}),
                    (difference, f"legoESM - NEMO [{unit}]  max|d|={span:.3e}",
                     {"cmap": "RdBu_r", "vmin": -span, "vmax": span}))):
                axis = axes[row][column]
                image = axis.pcolormesh(field.T, shading="auto", **kwargs)
                axis.set_title(title, fontsize=8)
                axis.set_xlabel("j (NEMO y)", fontsize=7)
                axis.set_ylabel("i (NEMO x)", fontsize=7)
                axis.tick_params(labelsize=6)
                figure.colorbar(image, ax=axis)
        source = json.loads(
            (Path(root) / "lego_seed0" / "manifest.json").read_text())
        footer = (
            f"GYRE from-rest year, day {day} of 360 "
            f"({day * STEPS_PER_DAY} steps at dt={DT_S:.0f} s), control "
            f"members (seed 0, unperturbed) | "
            f"legoESM commit {source['worktree']['commit'][:12]} | "
            f"scoring commit {stamp['commit'][:12]} "
            f"(clean={stamp['clean']}) | "
            f"NEMO {Path(nemo['path']).name} sha256 {nemo['sha256'][:16]} | "
            f"mesh sha256 {mesh['mesh_sha256'][:16]} | fp64, libm, CPU | "
            f"masked by NEMO's own tmask")
        figure.suptitle(footer, fontsize=7, y=0.999)
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        path = out_dir / f"gyre_year_fromrest_maps_day{day:03d}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written[str(path)] = sha256(path)
        provenance[str(day)] = {
            "figure": str(path), "sha256": written[str(path)],
            "lego_member": str(Path(root) / "lego_seed0"),
            "lego_commit": source["worktree"]["commit"],
            "lego_clean": source["worktree"]["clean"],
            "nemo_restart": nemo["path"], "nemo_sha256": nemo["sha256"],
            "mesh_sha256": mesh["mesh_sha256"],
            "levels": [{"index": int(index), "depth_m": float(depth_1d[index]),
                        "requested_m": float(target)} for index, target in levels],
            "scoring_commit": stamp["commit"],
        }
    (out_dir / "figure_provenance.json").write_text(
        json.dumps(provenance, indent=2))
    written[str(out_dir / "figure_provenance.json")] = sha256(
        out_dir / "figure_provenance.json")
    return written


# --------------------------------------------------------------- self-check --
def self_check() -> int:
    """Runnable checks of every piece of arithmetic this harness does not import."""
    assert _nint(0.5) == 1.0 and _nint(-0.5) == -1.0, "NINT is not half-away"
    assert _nint(1.5) == 2.0 and _nint(2.5) == 3.0, "NINT fell back to half-even"
    assert np.round(2.5) == 2.0, "numpy changed: half-even assumption is stale"

    # Two latitude groups of TWO cells each, per level: a one-cell group has
    # zero spread by construction and the constancy assertion could never fail
    # on it.  (That is exactly how the first draft of this self-check passed
    # while proving nothing.)
    depth = np.broadcast_to(np.array([10.0, 20.0]), (2, 2, 2)).copy()
    lat = np.broadcast_to(np.array([[30.0], [31.0]])[..., None],
                          (2, 2, 2)).copy()
    mask = np.ones_like(depth)
    zero = nemo_istate_perturbation(depth, lat, mask, 0)
    assert np.all(zero == 0.0), "seed 0 is not the unperturbed path"
    one = nemo_istate_perturbation(depth, lat, mask, 1)
    two = nemo_istate_perturbation(depth, lat, mask, 2)
    assert np.max(np.abs(one)) <= PERT_AMPLITUDE_K, "amplitude exceeded"
    assert not np.array_equal(one, two), "two seeds gave the same perturbation"
    assert np.max(np.abs(one - two)) > 0.1 * PERT_AMPLITUDE_K, "seeds barely differ"

    # the group-constancy assertion CAN fail -- a synthetic violation
    wet = mask.astype(bool)
    assert_perturbation_properties(one, lat, wet)
    broken = one.copy()
    broken[0, 0, 0] += 1e-12
    try:
        assert_perturbation_properties(broken, lat, wet)
    except GateError:
        pass
    else:                                   # pragma: no cover
        raise AssertionError("group-constancy assertion cannot fail")

    # the amplitude assertion CAN fail
    try:
        assert_perturbation_properties(one * 20.0, lat, wet)
    except GateError:
        pass
    else:                                   # pragma: no cover
        raise AssertionError("amplitude assertion cannot fail")

    # the non-degeneracy check CAN fail: a constant field passes constancy
    try:
        assert_perturbation_properties(
            np.full_like(one, PERT_AMPLITUDE_K),
            np.zeros_like(lat), wet)
    except GateError:
        pass
    else:                                   # pragma: no cover
        raise AssertionError("non-degeneracy check cannot fail")

    # the year arithmetic
    assert STEPS_PER_DAY == 6 and YEAR_STEPS == 2160 and SNAP_STEPS == 180
    assert SCORED_DAYS[0] == 30 and SCORED_DAYS[-1] == 360
    assert len(SCORED_DAYS) == 12

    # the rms distance is a distance, and the difference of two rms values is not
    a = np.array([[[1.0, 2.0]]])
    b = np.array([[[1.0, 3.0]]])
    wet1 = np.ones_like(a, dtype=bool)
    assert abs(_rms(a - b, wet1) - np.sqrt(0.5)) < 1e-12
    assert _rms(a - a, wet1) == 0.0
    assert abs(_rms(a, wet1) - _rms(b, wet1)) != _rms(a - b, wet1)

    # PSI is in Sv, is linear in u, and is SIGNED (a double gyre has two cells;
    # a max|PSI| row reports one of them and is blind to a sign flip).
    dz = np.full(2, 500.0)
    dy = np.full((1, 1), 1.0e5)
    u = np.ones((1, 1, 2)) * 0.01
    psi = _psi_field_sv(u, dz, dy)
    assert abs(float(psi.max()) - 1.0) < 1e-12, psi
    assert abs(float(_psi_field_sv(2 * u, dz, dy).max()) - 2.0) < 1e-12
    two_cells = _psi_field_sv(
        np.array([[[0.01, 0.01]], [[-0.02, -0.02]]]), dz, dy)
    assert float(two_cells.max()) > 0 > float(two_cells.min()), two_cells
    assert abs(float(two_cells.min()) + 1.0) < 1e-12, two_cells

    # the depth bands partition the column exactly once
    edges = [(lo, hi) for _, lo, hi in DEPTH_BANDS]
    assert edges[0][0] == 0.0 and edges[-1][1] > 5000.0
    for (_, hi), (lo, _) in zip(edges[:-1], edges[1:]):
        assert hi == lo, (hi, lo)

    # the cell-count diagnostic can see a single switched cell
    left = {"T": np.zeros((2, 2, 2))}
    right = {"T": np.zeros((2, 2, 2))}
    right["T"][0, 0, 0] = 1.0e-2
    wet_all = np.ones((2, 2, 2), dtype=bool)
    assert _cells_over(left, right, wet_all) == 1
    right["T"][0, 0, 0] = 1.0e-6
    assert _cells_over(left, right, wet_all) == 0

    # the verdict rule and its marginal band
    assert MARGINAL_BAND[0] < 1.0 < MARGINAL_BAND[1]
    # No square-root factor is applied to the floor anywhere: the
    # within-ensemble statistic is already a pairwise DIFFERENCE, so the
    # single-run-std conversion factor does not apply.  An earlier version of
    # this check grepped for one exact spelling and would have walked past the
    # others -- and, twice, matched the very line that named it.  Two
    # non-self-referential properties instead: the module-level constant is
    # GONE, and no executable line multiplies a floor by a root.
    import sys as _sys
    source = Path(__file__).read_text()
    assert not hasattr(_sys.modules[__name__], "SQRT" + "2"), (
        "the square-root constant is back; the pairwise floor does not take one")
    import re as _re
    multiply = _re.compile(
        r"(floor\s*=[^=].*sqrt)|(sqrt[^)]*\)\s*\*\s*floor)|(floor\s*\*[^=]*sqrt)",
        _re.I)
    offenders = [line for line in source.splitlines()
                 if multiply.search(line) and "hypot" not in line
                 and not line.lstrip().startswith("#")
                 and "selfcheck-probe" not in line]
    assert not offenders, f"a root factor is applied to the floor: {offenders}"
    # The check must be able to FAIL -- a grep that matches nothing by
    # construction is the defect it exists to prevent.
    assert multiply.search("floor = floor * np.sqrt(2.0)")  # selfcheck-probe
    assert not multiply.search("floor = np.hypot(a, b)")     # selfcheck-probe
    print("SELF-CHECK OK: Fortran NINT, seed-0 no-op, three assertions each "
          "shown to fail on a synthetic violation, 2160-step year, rms is a "
          "distance, signed PSI in Sv, depth bands partition the column, "
          "cell-count sees one switched cell, no sqrt(2) double-count")
    return 0


# ------------------------------------------------------------------- driver --
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=int, default=None,
                        help="run one legoESM member with this seed")
    parser.add_argument("--daily-reset-family", choices=DAILY_RESET_FAMILIES,
                        help="Round-134/135 oracle-reset arm")
    parser.add_argument("--daily-reset-interval-days", type=int, default=1,
                        help="whole-day interval between admitted resets")
    parser.add_argument("--daily-record-root", type=Path)
    parser.add_argument("--daily-record-audit", type=Path)
    parser.add_argument("--expect-commit",
                        help="full clean producer commit required by reset arms")
    parser.add_argument("--daily-reset-self-check", action="store_true")
    parser.add_argument("--days", type=int, default=YEAR_DAYS)
    parser.add_argument("--snap-steps", type=int, default=SNAP_STEPS,
                        help="snapshot cadence in STEPS; the preregistered "
                             "members use 180 (30 days) and are unchanged by "
                             "this flag.  Must be a whole number of days.")
    parser.add_argument("--tag", default="",
                        help="suffix for the member directory; \"repro\" is the same-binary reproducibility re-run of seed 0")
    parser.add_argument("--census", type=int, default=None,
                        metavar="STEPS",
                        help="step the card and report every field per step; exits non-zero on the first non-finite value")
    parser.add_argument("--census-every", type=int, default=1)
    parser.add_argument("--census-start", type=int, default=1)
    parser.add_argument(
        "--ladder-dump", type=Path, default=None,
        help=("write the kt=1..N states to an .npz -- the Rule-12 "
              "before/after bit-comparison of the certified short ladder"))
    parser.add_argument(
        "--census-corner", default=None,
        help=("J,I,K -- print the per-step TKE map at one cell during "
              "--census (the deepest-row falsifier)"))
    parser.add_argument("--score-phase0", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--alignment-gate", action="store_true",
                        help=("prove the two models are compared cell for "
                              "cell before any gap is quoted"))
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY,
                        help="NEMO's kt=1 BEFORE-level record, for A1")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_NEMO_MESH)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plant", default=None,
                        choices=["perturbation-zero", "perturbation-relative",
                                 "floor-inflate", "gap-zero",
                                 "operand-mismatch", "frame-flip",
                                 "alignment-initial", "mask-shift",
                                 *DAILY_RESET_PLANTS])
    args = parser.parse_args(argv)

    if args.daily_reset_self_check:
        if args.plant is not None and args.plant not in DAILY_RESET_PLANTS:
            parser.error("--daily-reset-self-check accepts only daily plants")
        return daily_reset_self_check(args.plant)
    if args.self_check:
        return self_check()
    if args.census is not None:
        if args.ladder_dump is not None:
            return ladder_dump(args.census, args.ladder_dump)
        _corner = None
        if args.census_corner:
            _corner = tuple(int(v) for v in args.census_corner.split(","))
            if len(_corner) != 3:
                raise SystemExit("--census-corner needs exactly J,I,K")
        return census(args.census, every=args.census_every,
                      start=args.census_start, corner=_corner)
    if args.member is not None:
        return run_member(args.member, args.root, days=args.days,
                          mesh_path=args.mesh, tag=args.tag,
                          plant=args.plant, snap_steps=args.snap_steps,
                          daily_reset_family=args.daily_reset_family,
                          daily_reset_interval_days=(
                              args.daily_reset_interval_days),
                          daily_record_root=args.daily_record_root,
                          daily_record_audit=args.daily_record_audit,
                          expect_commit=args.expect_commit)
    if args.alignment_gate:
        report = alignment_gate(args.root, mesh_path=args.mesh,
                                entry_path=args.entry, plant=args.plant)
        target = args.json or (args.root / "alignment_gate.json")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("STATUS ALIGNED")
        return 0
    if args.figures:
        written = figures(args.root, args.root / "maps", mesh_path=args.mesh)
        print(json.dumps(written, indent=2))
        return 0
    if not (args.score_phase0 or args.score):
        parser.error("one of --member/--score-phase0/--score/"
                     "--alignment-gate/--figures/--self-check is required")

    # The gap is never quoted before the frame is proved.  A1/A2/A3 RAISE
    # rather than return, so reaching the next line IS the alignment result.
    alignment = None
    if args.score:
        alignment = alignment_gate(args.root, mesh_path=args.mesh,
                                   entry_path=args.entry, plant=args.plant)
        (args.root / "alignment_gate.json").write_text(
            json.dumps(alignment, indent=2))

    report = score(args.root, phase0_only=args.score_phase0,
                   mesh_path=args.mesh, plant=args.plant)
    if alignment is not None:
        report["alignment_gate"] = alignment
    target = args.json or (args.root / ("phase0_floor.json"
                                        if args.score_phase0
                                        else "verdict_year.json"))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key != "rows"}, indent=2))
    temperature = report["rows"]["T3D"]["days"]
    header = ("day   spread_lego   spread_nemo         floor     gap_ctrl"
              "      gap_max  gap/(2*floor)  verdict"
              if args.score else "day    floor[K]")
    print(header)
    for day in sorted(temperature, key=int):
        record = temperature[day]
        if args.score:
            print(f"{int(day):3d}  {record['spread_lego']:.6e}  "
                  f"{record['spread_nemo']:.6e}  {record['floor']:.6e}  "
                  f"{record['gap_control_pair']:.6e}  "
                  f"{record['gap_max_across_pairs']:.6e}  "
                  f"{record['ratio_gap_over_2floor']:13.4f}  "
                  f"{record['verdict']}")
        else:
            print(f"{int(day):3d}  {record['floor']:.6e}")
    failures = [name for name, value in report.items()
                if isinstance(value, dict) and value.get("status") == "REFUTED"]
    if args.score_phase0 and not report["vacuity_gate"]["phase1_may_run"]:
        print("STATUS VACUOUS (the floor is zero somewhere; PHASE 1 refused)")
        return 1
    if args.score:
        print("\nday-360 verdict, EVERY scored row (only T3D gates P1-P5):")
        for name in sorted(report["rows"]):
            last = report["rows"][name]["days"][str(YEAR_DAYS)]
            print(f"  {name:<12s} {report['rows'][name]['unit']:<6s} "
                  f"gap={last['gap_max_across_pairs']:.6e} "
                  f"floor={last['floor']:.6e} "
                  f"ratio={last['ratio_gap_over_2floor']:.4f} "
                  f"{last['verdict']}")
    if failures:
        print("STATUS REFUTED " + " ".join(failures))
        return 1
    print("STATUS HELD")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        sys.exit(2)
