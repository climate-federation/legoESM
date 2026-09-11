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
    if gdept.ndim == 4:
        gdept = gdept[0].transpose(1, 2, 0)
    elif gdept.ndim == 2:
        gdept = np.broadcast_to(gdept[0], gphit.shape + (gdept.shape[1],))
    return {"gphit": gphit, "gdept_0": gdept, "tmask": tmask,
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


def run_member(seed: int, out_root: Path, *, days: int = YEAR_DAYS,
               mesh_path: Path = DEFAULT_NEMO_MESH, tag: str = "",
               plant: str | None = None) -> int:
    """One legoESM member: from rest, ``days`` days, a snapshot every 30 days."""
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
    out.mkdir(parents=True, exist_ok=True)
    n_steps = days * STEPS_PER_DAY
    started = time.time()
    for completed in range(n_steps):
        kt = completed + 1
        freshwater, surface = gate._surface_forcings(card, state, kt)
        state = model.step(state, dt=card.dt_s,
                           freshwater=freshwater, surface_forcing=surface)
        if kt % SNAP_STEPS == 0:
            arrays = _snapshot(state, gate)
            require(all(np.all(np.isfinite(value)) for value in arrays.values()),
                    f"seed {seed}: non-finite state at day {kt // STEPS_PER_DAY}")
            np.savez(out / f"day{kt // STEPS_PER_DAY:03d}.npz", **arrays)
            print(f"  seed {seed} day {kt // STEPS_PER_DAY:3d}  "
                  f"{time.time() - started:7.1f} s", flush=True)
    manifest = {
        "format": "nemo-testcase-l2-gyre-year-fromrest-member-v1",
        "case": CASE, "seed": seed, "tag": tag, "days": days,
        "steps": n_steps,
        "dt_s": DT_S, "snapshot_days": list(range(SNAP_DAYS, days + 1, SNAP_DAYS)),
        "phase3_gate_sha256": gate_sha,
        "perturbation": properties,
        "operands": operands,
        "wall_seconds": time.time() - started,
        "worktree": worktree_stamp(),
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


def census(steps: int, *, every: int = 1, start: int = 1) -> int:
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


def _load_nemo(root: Path, seed: int, day: int, nlev: int) -> dict:
    import netCDF4

    step = day * STEPS_PER_DAY
    directory = Path(root) / f"nemo_seed{seed}"
    matches = sorted(directory.glob(f"*_{step:08d}_restart.nc"))
    require(len(matches) == 1,
            f"expected exactly one NEMO restart for seed {seed} step {step} "
            f"in {directory}, found {len(matches)}")
    with netCDF4.Dataset(matches[0]) as handle:
        recorded = int(np.asarray(handle.variables["kt"][...]))
        require(recorded == step,
                f"{matches[0]}: kt={recorded}, expected {step}")

        def xyz(name):
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
        # legoESM's own snapshots are checked for finiteness as they are
        # written; NEMO's were not checked at all, so a blown-up oracle member
        # would have entered the floor as a plausible number.
        for name, values in fields.items():
            require(bool(np.all(np.isfinite(values))),
                    f"{matches[0]}: NEMO field {name} is not finite "
                    f"({int(np.count_nonzero(~np.isfinite(values)))} cells)")
        fields.update({"path": str(matches[0]), "sha256": sha256(matches[0])})
        return fields


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

    rows, days, diagnostics = {}, [], {}
    repro = _repro_floor(root, prepare, wet3)
    for day in SCORED_DAYS:
        lego = [_load_lego(root, seed, day) for seed in SEEDS]
        lego_p = [prepare(state, day) for state in lego]
        if not phase0_only:
            nemo = [_load_nemo(root, seed, day, nlev) for seed in SEEDS]
            nemo_p = [prepare(state, day) for state in nemo]
        days.append(day)
        diagnostics[str(day)] = {
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

    report = {
        "format": ("nemo-testcase-l2-gyre-year-fromrest-phase0-v1"
                   if phase0_only else
                   "nemo-testcase-l2-gyre-year-fromrest-verdict-v1"),
        "case": CASE, "seeds": list(SEEDS), "scored_days": days,
        "dt_s": DT_S, "year_steps": YEAR_STEPS,
        "verdict_factor": VERDICT_FACTOR, "marginal_band": list(MARGINAL_BAND),
        "phase3_gate_sha256": gate_sha,
        "mesh_sha256": mesh["mesh_sha256"],
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
        "rows": rows,
        "worktree": worktree_stamp(),
    }
    report.update(_expectations(rows, phase0_only=phase0_only, repro=repro))
    if phase0_only:
        report["vacuity_gate"] = _vacuity(rows)
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
    gaps = {day: temperature[day]["gap_max_across_pairs"]
            for day in temperature}
    lo, hi = min(gaps.values()), max(gaps.values())
    out["P3_gap_band"] = {
        "min_over_days": lo, "max_over_days": hi,
        "band": [P3_GAP_MIN_K, P3_GAP_MAX_K],
        "status": ("HELD" if lo >= P3_GAP_MIN_K and hi <= P3_GAP_MAX_K
                   else "REFUTED")}
    verdicts = {day: temperature[day]["verdict"] for day in temperature}
    crossing = next((day for day in sorted(temperature, key=int)
                     if temperature[day]["ratio_gap_over_2floor"] <= 1.0), None)
    out["P4_distinguishable_every_day"] = {
        "verdicts": verdicts, "crossing_day": crossing,
        "status": ("HELD" if all(value == "DISTINGUISHABLE"
                                 for value in verdicts.values())
                   else "REFUTED")}
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


def _vacuity(rows: dict) -> dict:
    """PHASE 0's only gate: the floor must be able to see anything at all."""
    zero_days = [day for day, record in rows["T3D"]["days"].items()
                 if record["floor"] <= 0.0]
    return {
        "floor_positive_every_day": not zero_days,
        "zero_floor_days": zero_days,
        "phase1_may_run": not zero_days,
        "reason": ("a zero floor means the seed never reached the state, and "
                   "would be read as DISTINGUISHABLE -- the exact false "
                   "positive this harness exists to avoid"),
    }


# ------------------------------------------------------------------ figures --
def figures(root: Path, out_dir: Path,
            mesh_path: Path = DEFAULT_NEMO_MESH) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    _, wet3, wet2, _, _, _, _ = _geometry(card, mesh_path)
    nlev = card.recipe.z_coord.n_levels
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for day in (SNAP_DAYS, YEAR_DAYS):
        lego = _load_lego(root, 0, day)
        nemo = _load_nemo(root, 0, day, nlev)
        difference = lego["T"] - nemo["T"]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        surface = np.where(wet2, difference[..., 0], np.nan)
        deep = np.where(wet3[..., 9], difference[..., 9], np.nan)
        for axis, field, title in (
                (axes[0][0], np.where(wet2, lego["T"][..., 0], np.nan),
                 f"legoESM SST, day {day} [C]"),
                (axes[0][1], np.where(wet2, nemo["T"][..., 0], np.nan),
                 f"NEMO SST, day {day} [C]"),
                (axes[1][0], surface, f"SST difference, day {day} [K]"),
                (axes[1][1], deep, f"level-10 T difference, day {day} [K]")):
            image = axis.pcolormesh(field.T, shading="auto",
                                    cmap="RdBu_r" if "difference" in title
                                    else "viridis")
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("j")
            axis.set_ylabel("i")
            figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = out_dir / f"gyre_year_fromrest_day{day:03d}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        written[str(path)] = sha256(path)
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
    parser.add_argument("--days", type=int, default=YEAR_DAYS)
    parser.add_argument("--tag", default="",
                        help="suffix for the member directory; \"repro\" is the same-binary reproducibility re-run of seed 0")
    parser.add_argument("--census", type=int, default=None,
                        metavar="STEPS",
                        help="step the card and report every field per step; exits non-zero on the first non-finite value")
    parser.add_argument("--census-every", type=int, default=1)
    parser.add_argument("--census-start", type=int, default=1)
    parser.add_argument("--score-phase0", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_NEMO_MESH)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plant", default=None,
                        choices=["perturbation-zero", "perturbation-relative",
                                 "floor-inflate", "gap-zero",
                                 "operand-mismatch"])
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check()
    if args.census is not None:
        return census(args.census, every=args.census_every,
                      start=args.census_start)
    if args.member is not None:
        return run_member(args.member, args.root, days=args.days,
                          mesh_path=args.mesh, tag=args.tag,
                          plant=args.plant)
    if args.figures:
        written = figures(args.root, args.root / "figures", mesh_path=args.mesh)
        print(json.dumps(written, indent=2))
        return 0
    if not (args.score_phase0 or args.score):
        parser.error("one of --member/--score-phase0/--score/--figures/"
                     "--self-check is required")

    report = score(args.root, phase0_only=args.score_phase0,
                   mesh_path=args.mesh, plant=args.plant)
    target = args.json or (args.root / ("phase0_floor.json"
                                        if args.score_phase0
                                        else "verdict_year.json"))
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key != "rows"}, indent=2))
    temperature = report["rows"]["T3D"]["days"]
    header = ("day    floor[K]      gap[K]   gap/(2*floor)  verdict"
              if args.score else "day    floor[K]")
    print(header)
    for day in sorted(temperature, key=int):
        record = temperature[day]
        if args.score:
            print(f"{int(day):3d}  {record['floor']:.6e}  {record['gap']:.6e}  "
                  f"{record['ratio_gap_over_2floor']:13.4f}  {record['verdict']}")
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
