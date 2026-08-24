"""#1492: perturb NEMO's day-180 restart for the 90-day NEMO-side noise-floor
ensemble (RUN_FLOOR90_M{1,2,3}), matching the legoESM-side 90-day ensemble's
own T-perturbation convention exactly so the two floors are comparable.

WHY THIS DIFFERS FROM ``scripts/tmp/_perturb_restart_ensemble.py``.  That tool
perturbs ``tn`` AND ``tb`` (same random draw, both time levels) across 16
per-rank restart tiles, for the DIFFERENT 10-year NEMO-perturbing-NEMO floor
ensemble.  The legoESM-side 90-day ensemble
(``kamm_twin_90d.py --perturb-seed``, read at
``scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py:653-662``)
perturbs ONLY the now-level T.  With ``--bridge-before`` -- which is what
``floor90_ensemble.py`` actually runs -- the legoESM twin state DOES carry a
before-level T (bridged from NEMO's ``tb``), but the perturbation is applied
to ``state.T`` (now) only; ``state.T_before`` keeps the original,
UNPERTURBED bridged value. To measure a comparable floor this tool matches
that asymmetry rather than "fixing" it: ``tn`` is perturbed, ``tb`` is left
BIT-IDENTICAL to the source. This is inherited from the twin harness, not a
choice made here, and it means a NEMO run started this way carries a real
tn/tb mismatch (a leapfrog computational-mode kick) that the tn+tb same-draw
convention was built to avoid -- flagged, not fixed (out of scope for this
tool).

Also unlike the 16-tile RUN_20Y source, ``RUN_90D_TWIN``'s own input restart
(``DINO_00005760_restart.nc``) is a SINGLE GLOBAL file (x=52, y=199 -- the
full domain, not a per-rank tile) even though the run itself is 16-rank:
NEMO scatters the global restart at read time. So there is exactly one file
to perturb per member, not 16.

Usage
-----
  perturb_nemo_tn_90d.py <seed> <dest_dir> [--src PATH]     # perturb + verify
  perturb_nemo_tn_90d.py                                    # self-check only
"""
import shutil
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

SRC = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
           "RUN_90D_TWIN/DINO_00005760_restart.nc")
EPS = 1e-14


def _attr_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def perturb(seed: int, dest_dir: Path, src: Path = SRC) -> dict:
    """Copy ``src`` (a NEMO restart, single global file) into ``dest_dir``,
    multiply ``tn`` by ``(1 + EPS * N(0,1))`` per grid point in place
    (``rng = numpy.random.default_rng(seed)``), leave every other variable
    and every attribute (global + per-variable) bit-identical. Returns a
    verification report; raises if anything besides ``tn`` moved."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src.name
    shutil.copy2(src, dst)  # dereferences the RUN_90D_TWIN symlink -> real file

    rng = np.random.default_rng(seed)
    d = nc.Dataset(dst, "a")
    tn = np.asarray(d["tn"][:], dtype=np.float64)
    d["tn"][:] = tn * (1.0 + EPS * rng.standard_normal(tn.shape))
    d.close()

    report = _verify(src, dst)
    if not (report["other_vars_bit_identical"] and report["attrs_bit_identical"]):
        raise SystemExit(
            f"perturbation of {dst} touched something besides tn: "
            f"changed_vars={report['changed_vars_besides_tn']} "
            f"attrs_bit_identical={report['attrs_bit_identical']}")
    return report


def _verify(src: Path, dst: Path) -> dict:
    s, r = nc.Dataset(src), nc.Dataset(dst)
    s_tn = np.asarray(s["tn"][:], dtype=np.float64)
    r_tn = np.asarray(r["tn"][:], dtype=np.float64)
    diff = r_tn - s_tn
    nz = s_tn != 0
    rel = np.zeros_like(diff)
    rel[nz] = np.abs(diff[nz] / s_tn[nz])

    other_ok, max_other_diff, changed_vars = True, 0.0, []
    for vname in s.variables:
        a, b = np.asarray(s[vname][:]), np.asarray(r[vname][:])
        if a.shape != b.shape:
            other_ok, changed_vars = False, changed_vars + [vname]
            continue
        d_ = (float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
              if a.dtype.kind in "fc" and a.size else
              (0.0 if np.array_equal(a, b) else 1.0))
        if vname != "tn":
            # max_other_diff excludes tn itself -- it is the max diff among
            # the vars that are SUPPOSED to be untouched, not tn's own
            # (expected) perturbation size (review finding, was folding tn's
            # diff into "max_other_var_diff" before this exclusion existed).
            max_other_diff = max(max_other_diff, d_)
            if d_ != 0.0:
                other_ok, changed_vars = False, changed_vars + [vname]

    attrs_ok = s.ncattrs() == r.ncattrs() and all(
        _attr_equal(getattr(s, a), getattr(r, a)) for a in s.ncattrs())
    for vname in s.variables:
        if s[vname].ncattrs() != r[vname].ncattrs():
            attrs_ok = False
        else:
            attrs_ok = attrs_ok and all(
                _attr_equal(getattr(s[vname], a), getattr(r[vname], a))
                for a in s[vname].ncattrs())
    s.close()
    r.close()

    return {
        "max_abs_tn_diff": float(np.max(np.abs(diff))),
        "max_rel_tn_diff": float(np.max(rel)) if nz.any() else 0.0,
        "n_tn_changed": int(np.sum(diff != 0)),
        "n_tn_total": int(diff.size),
        "other_vars_bit_identical": other_ok,
        "max_other_var_diff": max_other_diff,
        "changed_vars_besides_tn": changed_vars,
        "attrs_bit_identical": attrs_ok,
    }


def _demo_self_check():
    """Runnable check, no cluster restart needed beyond the committed source:
    only tn changes, tb + every other var/attr stays bit-identical, and the
    relative perturbation lands in (0, 1e-12) -- the same bounds the sibling
    ``_perturb_restart_ensemble.py`` self-check asserts."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # perturb() itself already raises SystemExit if other_vars_bit_identical
        # or attrs_bit_identical comes back False (review: asserting those two
        # flags again here was dead code -- unreachable given that guard).
        rep = perturb(seed=999, dest_dir=Path(td))
        assert 0 < rep["max_rel_tn_diff"] < 1e-12, rep["max_rel_tn_diff"]
        # land cells carry tn==0 exactly, and factor*0==0, so those stay
        # unchanged by construction -- only wet cells are expected to move.
        assert 0 < rep["n_tn_changed"] < rep["n_tn_total"], rep["n_tn_changed"]
    print("self-check OK: only tn perturbed (tb + every other var/attr "
          "bit-identical), relative scale in (0, 1e-12)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _demo_self_check()
        sys.exit(0)
    seed_ = int(sys.argv[1])
    dest_ = Path(sys.argv[2])
    rep_ = perturb(seed_, dest_)
    print(f"seed={seed_} dest={dest_}")
    print(f"  max|dT|={rep_['max_abs_tn_diff']:.3e}  "
          f"max_rel|dT/T|={rep_['max_rel_tn_diff']:.3e}  "
          f"n_changed(tn)={rep_['n_tn_changed']}/{rep_['n_tn_total']}  "
          f"other_vars_identical={rep_['other_vars_bit_identical']}  "
          f"attrs_identical={rep_['attrs_bit_identical']}")
