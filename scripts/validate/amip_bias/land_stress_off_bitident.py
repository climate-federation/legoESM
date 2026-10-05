"""Off-path bit-identity probe for the land-stress PR (user Q13).

Runs short MPAS driver integrations from the CURRENT checkout (cwd = repo
root on PYTHONPATH) and saves every atmosphere and land state array, so the
PR branch with the land stress OFF can be compared bit-for-bit with the
pre-PR commit.  Uses the repo's own driver test harness
(tests/unit/test_multilayer_land_driver.py: _small_cfg + synthetic land
loaders), present on both sides.

  run OUT.npz     cases: simple_seb+louis and two_leaf+clubb, land flux
                  handoff on, 4 steps; on a checkout that has the switch it
                  is set explicitly False ("off") and also left unset on an
                  ineligible config ("auto_inel": beta_soil off)
  compare A.npz B.npz   exact array equality on the common keys; prints the
                  max abs difference per differing key and a verdict line
"""
import importlib.util
import subprocess
import sys
import tempfile

import numpy as np


def _harness():
    spec = importlib.util.spec_from_file_location(
        "_h", "tests/unit/test_multilayer_land_driver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cases(base, has_switch):
    from legoesm.driver.config import DycoreConfig, GridConfig
    mesh = dict(grid=GridConfig(grid_type="mpas", resolution=2, nlev=8),
                dycore=DycoreConfig(dt=600.0, discretization="mpas"),
                days=2401.0 / 86400.0, mpas_land_params_refresh=False)
    cases = {
        "seb_louis": base._replace(**mesh, mpas_land_beta_soil=True,
                                   turbulence="louis",
                                   land_surface_scheme="simple_seb"),
        "twoleaf_clubb": base._replace(**mesh, mpas_land_beta_soil=True,
                                       turbulence="clubb",
                                       land_surface_scheme="two_leaf",
                                       snow_albedo_feedback=True),
        "seb_louis_nobeta": base._replace(**mesh, mpas_land_beta_soil=False,
                                          turbulence="louis",
                                          land_surface_scheme="simple_seb"),
    }
    if has_switch:
        for k in ("seb_louis", "twoleaf_clubb"):
            cases[k] = cases[k]._replace(mpas_land_stress_from_land=False)
    return cases


def run(out):
    import pytest
    from legoesm.driver.model_driver import ModelDriver
    h = _harness()
    base = h._small_cfg()
    has_switch = "mpas_land_stress_from_land" in base._fields
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    arrays = {"_meta_sha": np.array(sha), "_meta_switch": np.array(has_switch)}
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as td:
        h._patch_land_loaders(mp)
        for name, cfg in _cases(base, has_switch).items():
            cfg.validate_strict()
            d = ModelDriver(cfg, output_dir=f"{td}/{name}")
            d.setup()
            status = d.run()
            if status != "COMPLETED":
                raise SystemExit(f"{name}: run status {status}")
            st = d.state
            fields = {f: getattr(st, f, None) for f in ("u", "T", "p_s", "w")}
            fields.update({f"tracer_{k}": v
                           for k, v in (st.tracers or {}).items()})
            fields.update({f"land_{f}": getattr(d._land_ml_state, f)
                           for f in d._land_ml_state._fields})
            for k, v in fields.items():
                while hasattr(v, "data"):
                    v = v.data
                if v is None or isinstance(v, tuple):
                    continue
                a = np.asarray(v)
                if a.dtype != object:
                    arrays[f"{name}/{k}"] = a
    np.savez(out, **arrays)
    print(f"saved {len(arrays) - 2} arrays from {sha} (switch={has_switch}) "
          f"to {out}")


def compare(a, b):
    A, B = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    print("A", A["_meta_sha"], "switch", A["_meta_switch"])
    print("B", B["_meta_sha"], "switch", B["_meta_switch"])
    keys = sorted(k for k in set(A.files) & set(B.files)
                  if not k.startswith("_meta"))
    only = sorted((set(A.files) ^ set(B.files)) - {"_meta_sha", "_meta_switch"})
    bad = []
    for k in keys:
        x, y = A[k], B[k]
        if x.dtype == object or y.dtype == object:
            print(f"SKIP {k}: object array (not a numeric state field)")
            continue
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
            d = (np.nanmax(np.abs(x - y)) if x.shape == y.shape else "shape")
            bad.append(k)
            print(f"DIFF {k}: {d}")
    print(f"{len(keys)} common arrays, {len(bad)} differ, "
          f"{len(only)} present on one side only {only[:5]}")
    print("VERDICT: BIT-IDENTICAL" if not bad and keys else "VERDICT: DIFFERENT")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
