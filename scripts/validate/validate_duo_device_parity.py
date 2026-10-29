"""Gate: an FV3-duo AMIP run on 1 device and on 2 devices agree after the
first physics step, per field group, relative to each field's peak.

``dump`` wraps ``run_amip`` (production deck, same command line in both
arms; only the launcher and ``--distributed --distributed-mode spmd``
differ).  On the first model step that runs physics it records the step's
input bundle, the post-dynamics bundle, the post-step column state and the
physics in/out (gathered to whole arrays with the model's own ``_to_host``);
at the start of the next step it writes the driver's own checkpoint (land
included) and exits every process::

    python scripts/validate/validate_duo_device_parity.py dump --out-dir D1 \\
        -- <run_amip args>
    mpiexec -n 2 ... python scripts/validate/validate_duo_device_parity.py \\
        dump --out-dir D2 -- <run_amip args> --distributed --distributed-mode spmd
    python scripts/validate/validate_duo_device_parity.py compare D1 D2

``compare`` exits 1 when any field is outside its group's bound, a key or
shape differs, or the dry-soil KNOWN DEFECT below grows.

KNOWN DIFFERENCE (C24 deck, 1 vs 2 devices, one step, x64; user-approved
bounds 2026-10-29):
* dynamics: not bitwise -- summation order across devices; worst field
  1e-13 of peak (pkz), temperature 2.9e-11 K.
* radiation runs in fp32 on this lane (the driver's compute_fp32): its
  heating 4.8e-5, total heating rate 3.0e-6, longwave fluxes 1.3-1.8e-6,
  shortwave 1.1e-7, temperature after the step 4.4e-8, sea-ice skin
  temperature 2.0e-10.  With radiation forced to fp64 in both arms all of
  these fall to <= 1.1e-12 (sea ice bitwise).
* cloud-water subgrid variance (relvar): 6.7e-10, unchanged with fp64
  radiation -- dynamics round-off amplified by cloud physics.
* land, every column but the two below: <= 1.3e-7.
* The 1-day difference (O(1) in cloud fields) is this gap grown by the
  weather -- PLAUSIBLE, not measured.

KNOWN DEFECT (not a tolerance): two very dry-topsoil columns, 470 and 495
(face 0), jump in the fp32-radiation run -- soil moisture 7.3e-3 of peak
(column 470, layer 4: 3.2e-3 m3/m3), soil suction 1.1e-3, albedo 1.9e-4,
soil temperature 8.6e-6.  Soil-water totals agree (column 470: 922.132139
kg/m2 in both arms, difference 2.6e-12; column 495: 969.918787, -1.9e-12):
water moves between layers, none is created.  The jump vanishes with fp64
radiation, so it is the dry-soil land solve's sensitivity to a ~1e-6
forcing change, not a device-split error.  Those columns are reported by
name on every run, never hidden; a new column, a new field, or a size
above the measured one fails the gate.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from scripts.validate.validate_duo_land_device_split import leaf_rel  # noqa: E402

N_COLUMNS = 6 * 24 * 24

TOL_ATMOSPHERE = 1e-10
TOL_RADIATION = 1e-4
TOL_CLOUD_VARIANCE = 6.7e-9
TOL_LAND = 1e-6

_RAD_SLOTS = (0, 1, 3, 8, 9, 11)  # sw/lw net sfc, lw up toa, sw/lw down sfc, lw up toa clr
RADIATION_KEYS = {
    "model_step.npz": (
        {"s0_physout_rad_heating", "s0_physout_held_physics_dT_dt_flat_index_0",
         "s0_new_T_flat_index_0", "s0_new_native_state_pt"}
        | {f"s0_physout_held_physics_{f}_flat_index_0" for f in (
            "sw_net_sfc", "lw_net_sfc", "lw_up_toa", "sw_down_sfc",
            "lw_down_sfc", "lw_up_toa_clr")}
        | {f"s0_sfc_{i}_flat_index_0" for i in _RAD_SLOTS}),
    "driver_checkpoint.npz": (
        {"physstate_rad_heating", "T", "state_pt", "ice_T_skin"}
        | {f"cmor_fluxsum_{i}" for i in _RAD_SLOTS}),
}
CLOUD_VARIANCE_KEYS = {"model_step.npz": {"s0_physout_relvar"},
                       "driver_checkpoint.npz": {"physstate_relvar"}}

# measured max|d|/peak per (land field, column), fp32-radiation C24 pair
DRY_SOIL_DEFECT = {
    ("land_ml_theta_soil", 470): 0.007317916876823379,
    ("land_ml_theta_soil", 495): 0.0008840520035645059,
    ("land_ml_psi_soil", 470): 0.0004043122562504943,
    ("land_ml_psi_soil", 495): 0.001125322371841656,
    ("land_held_albedo", 470): 2.917594547470961e-05,
    ("land_held_albedo", 495): 0.00018766495014062518,
    ("land_ml_T_soil", 470): 2.8197581978050917e-06,
    ("land_ml_T_soil", 495): 8.631109943871692e-06,
}
DRY_SOIL_COLUMNS = (470, 495)


def group_of(fname: str, key: str):
    """``(group, bound)`` of one array; anything not listed is held to the
    atmosphere bound."""
    if key in RADIATION_KEYS[fname]:
        return "radiation", TOL_RADIATION
    if key in CLOUD_VARIANCE_KEYS[fname]:
        return "cloud_variance", TOL_CLOUD_VARIANCE
    if key.startswith("land_"):
        return "land", TOL_LAND
    return "atmosphere", TOL_ATMOSPHERE


def _rel(a, b) -> float:
    return leaf_rel({"x": a}, {"x": b})["['x']"]


def _column_rel(a, b, col) -> float:
    """max|d|/peak over column ``col`` only (peak of the whole field)."""
    only = a.copy()
    only[col] = b[col]
    return _rel(a, only)


def soil_water_kg_m2(ckpt, col) -> float:
    from legoesm import constants
    return float(np.sum(ckpt["land_ml_theta_soil"][col] * ckpt["land_soil_dz"])
                 * constants.rho_water)


def compare_files(fname: str, ref, got) -> list[str]:
    """Failures for one file pair; prints every group's worst field and the
    dry-soil columns."""
    fails = []
    if fname == "driver_checkpoint.npz" and (
            np.shape(ref.get("land_ml_theta_soil"))[:1] != (N_COLUMNS,)):
        return [f"{fname}: land_ml_theta_soil is not (6*24*24, ...); the recorded "
                "gaps are for the C24 deck"]
    for k in sorted(set(ref) ^ set(got)):
        fails.append(f"{fname}: {k} only in {'ref' if k in ref else 'cmp'}")
    for k in sorted(RADIATION_KEYS[fname] | CLOUD_VARIANCE_KEYS[fname]):
        if k not in ref:
            fails.append(f"{fname}: grouped key {k} absent")
    worst = {}
    for k in sorted(set(ref) & set(got)):
        a, b = np.asarray(ref[k]), np.asarray(got[k])
        try:
            r = _rel(a, b)
        except ValueError as exc:
            fails.append(f"{fname}: {exc}")
            continue
        group, tol = group_of(fname, k)
        if group == "land" and a.dtype.kind == "f" and a.ndim and a.shape[0] == N_COLUMNS:
            # the whole-field bound covers every other column; each dry-soil
            # column is checked and printed on its own below
            rest = b.copy()
            for c in DRY_SOIL_COLUMNS:
                rest[c] = a[c]
                rc = _column_rel(a, b, c)
                known = DRY_SOIL_DEFECT.get((k, c))
                if known is not None:
                    print(f"  KNOWN DEFECT {k} column {c}: {rc:.3e} (measured {known:.3e})")
                    if rc > known:
                        fails.append(f"{fname}: {k} column {c} {rc:.3e} > measured {known:.3e}")
                elif rc > tol:
                    fails.append(f"{fname}: {k} column {c} {rc:.3e} > {tol:g} "
                                 "(a new field in the dry-soil defect)")
            r = _rel(a, rest)
            if r > tol:
                d = np.abs(np.nan_to_num(rest - a)).reshape(N_COLUMNS, -1).max(axis=1)
                k = f"{k} (worst column {int(np.argmax(d))})"
        if r > worst.get(group, (-1.0, ""))[0]:
            worst[group] = (r, k)
        if r > tol:
            fails.append(f"{fname}: [{group}] {k} {r:.3e} > {tol:g}")
    for group, (r, k) in sorted(worst.items()):
        print(f"  {fname} {group:15s} worst {r:.3e} ({k})")
    if fname == "driver_checkpoint.npz" and "land_ml_theta_soil" in ref:
        for c in DRY_SOIL_COLUMNS:
            w1, w2 = soil_water_kg_m2(ref, c), soil_water_kg_m2(got, c)
            print(f"  KNOWN DEFECT column {c} soil water: ref {w1:.6f} cmp "
                  f"{w2:.6f} kg/m2, cmp-ref {w2 - w1:.3e}")
    return fails


def compare(ref_dir: Path, cmp_dir: Path) -> int:
    fails = []
    for fname in ("model_step.npz", "driver_checkpoint.npz"):
        with np.load(ref_dir / fname) as ref, np.load(cmp_dir / fname) as got:
            fails += compare_files(fname, dict(ref), dict(got))
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAILED ({len(fails)})")
    return 1 if fails else 0


def _flatten(prefix, tree, rec):
    import jax
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnModel
    if tree is None:
        return
    for path, x in jax.tree_util.tree_flatten_with_path(
            FV3DuoColumnModel._to_host(tree))[0]:
        if x is None:
            continue
        key = re.sub(r"[^A-Za-z0-9_]+", "_",
                     prefix + jax.tree_util.keystr(path)).strip("_")
        a = np.asarray(x)
        if a.dtype.kind not in "fiub":
            raise TypeError(f"{key}: dtype {a.dtype} cannot be compared")
        rec[key] = a


def _install_dump(out_dir: Path, radiation_fp64: bool):
    import jax
    from legoesm.atmosphere.dynamics.gcm import fv3_duo_column as fdc

    if radiation_fp64:
        # the driver builds RRTMGPConfig(compute_fp32=True) for rrtmgp
        from legoesm.atmosphere.physics.radiation import config as rc
        orig_cfg = rc.RRTMGPConfig

        def fp64_cfg(*a, **k):
            k["compute_fp32"] = False
            return orig_cfg(*a, **k)

        rc.RRTMGPConfig = fp64_cfg

    model_cls = fdc.FV3DuoColumnModel
    orig_step = model_cls.step
    rec = {}

    def driver():
        # looked up at step time: importing the driver module up front
        # would initialise the backend before the multi-process launcher
        import gc
        found = [o for o in gc.get_objects() if type(o).__name__ == "ModelDriver"]
        if len(found) != 1:
            raise RuntimeError(f"expected one ModelDriver, found {len(found)}")
        return found[0]

    def step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        n = getattr(self, "_parity_n", 0)
        self._parity_n = n + 1
        if getattr(self, "_parity_done", False):
            drv = driver()
            day = float(drv.config.start_day)
            drv.save_checkpoint(n, day)
            if jax.process_index() == 0:
                np.savez(out_dir / "model_step.npz", **rec)
                shutil.copy2(drv._output_dir / f"checkpoint_day_{int(round(day)):04d}.npz",
                             out_dir / "driver_checkpoint.npz")
                print(f"PARITY dump: {len(rec)} model arrays (steps 0..{n - 1}) "
                      f"+ driver checkpoint at the start of step {n} -> {out_dir}",
                      flush=True)
            sys.stdout.flush()
            os._exit(0)
        dyn_cls = type(self.dyn)
        orig_dyn = dyn_cls.step
        cap = {}

        def dstep(dself, *a, **k):
            out = orig_dyn(dself, *a, **k)
            cap["in"], cap["out"] = a[0], out
            return out

        dyn_cls.step = dstep
        try:
            out = orig_step(self, state, dt, physics_fn=physics_fn,
                            forcing=forcing, phys_state=phys_state)
        finally:
            dyn_cls.step = orig_dyn
        tag = f"s{n}"
        _flatten(f"{tag}_pre", cap["in"], rec)
        _flatten(f"{tag}_dyn", cap["out"], rec)
        _flatten(f"{tag}_new", out, rec)
        if physics_fn is not None:
            _flatten(f"{tag}_physin", phys_state, rec)
            _flatten(f"{tag}_forcing", forcing, rec)
            _flatten(f"{tag}_physout", self._phys_state, rec)
            _flatten(f"{tag}_sfc", self._sfc_diag, rec)
            self._parity_done = True
        return out

    model_cls.step = step


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, run_args = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--radiation-fp64", action="store_true",
                   help="force radiation to fp64 (the lane runs it in fp32)")
    c = sub.add_parser("compare")
    c.add_argument("ref_dir", type=Path)
    c.add_argument("cmp_dir", type=Path)
    opts = p.parse_args(argv)
    if opts.cmd == "compare":
        return compare(opts.ref_dir, opts.cmp_dir)
    if not run_args:
        p.error("dump needs run_amip arguments after --")
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    _install_dump(opts.out_dir, opts.radiation_fp64)
    import runpy
    sys.argv = ["run_amip.py"] + run_args
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "run" / "run_amip.py"),
                   run_name="__main__")
    raise SystemExit("run_amip finished without reaching a physics step")


if __name__ == "__main__":
    sys.exit(main())
