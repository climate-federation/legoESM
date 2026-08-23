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
convention was built to avoid.

``both_levels=True`` (CLI ``--both-levels``) is the OTHER convention, added
for the #1455 kick-asymmetry discriminator
(``PREREG_kick_asymmetry.md``): the SAME per-grid-point relative draw
multiplies ``tn`` AND ``tb``, so the perturbed state carries no tn-tb
mismatch beyond the one the source restart already had. Two runs that differ
only in this flag differ only in how much leapfrog COMPUTATIONAL mode the
kick projects onto. The legoESM twin's matching flag is
``kamm_twin_90d.py --perturb-both-levels``.

Which fields move is stated by the caller and verified afterwards: the
verification is over EVERY variable and EVERY attribute in the file, and any
variable outside the requested target set that moved raises.

Also unlike the 16-tile RUN_20Y source, ``RUN_90D_TWIN``'s own input restart
(``DINO_00005760_restart.nc``) is a SINGLE GLOBAL file (x=52, y=199 -- the
full domain, not a per-rank tile) even though the run itself is 16-rank:
NEMO scatters the global restart at read time. So there is exactly one file
to perturb per member, not 16.

Usage
-----
  perturb_nemo_tn_90d.py <seed> <dest_dir> [--both-levels]  # perturb + verify
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


def perturb(seed: int, dest_dir: Path, src: Path = SRC,
            both_levels: bool = False) -> dict:
    """Copy ``src`` (a NEMO restart, single global file) into ``dest_dir``,
    multiply the target temperature field(s) by ``(1 + EPS * N(0,1))`` per
    grid point in place (``rng = numpy.random.default_rng(seed)``), leave
    every other variable and every attribute (global + per-variable)
    bit-identical. Returns a verification report; raises if anything outside
    the target set moved.

    ``both_levels=False`` (default, unchanged): targets ``tn`` only -- the
    one-time-level kick the recorded 90-day and 360-day ensembles used.
    ``both_levels=True``: targets ``tn`` AND ``tb`` with the SAME draw, i.e.
    the same RELATIVE perturbation at both leapfrog time levels. ONE draw is
    taken and reused; drawing twice would give the two levels INDEPENDENT
    perturbations, which is a bigger tn-tb mismatch than the one-level kick,
    not a smaller one -- the exact opposite of what this option is for.
    """
    targets = ("tn", "tb") if both_levels else ("tn",)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src.name
    shutil.copy2(src, dst)  # dereferences the RUN_90D_TWIN symlink -> real file

    rng = np.random.default_rng(seed)
    d = nc.Dataset(dst, "a")
    shape = np.asarray(d["tn"][:]).shape
    factor = 1.0 + EPS * rng.standard_normal(shape)   # ONE draw, reused
    for name in targets:
        v = np.asarray(d[name][:], dtype=np.float64)
        if v.shape != shape:
            d.close()
            raise SystemExit(f"{name} has shape {v.shape}, tn has {shape} -- "
                             f"the same draw cannot be applied to both")
        d[name][:] = v * factor
    d.close()

    report = _verify(src, dst, targets)
    if not (report["other_vars_bit_identical"] and report["attrs_bit_identical"]):
        raise SystemExit(
            f"perturbation of {dst} touched something besides {targets}: "
            f"changed_vars={report['changed_vars_besides_targets']} "
            f"attrs_bit_identical={report['attrs_bit_identical']}")
    return report


def _verify(src: Path, dst: Path, targets: tuple = ("tn",)) -> dict:
    """Compare EVERY variable and EVERY attribute of ``dst`` against ``src``.

    ``targets`` is the set of variables the caller INTENDED to change. Each
    one gets its own magnitude report; every other variable must be
    bit-identical, and so must every global and per-variable attribute. The
    per-target magnitudes are excluded from ``max_other_var_diff`` -- that
    column is the max diff among the variables that were SUPPOSED to be
    untouched, not the (expected) size of the perturbation itself.
    """
    s_, r = nc.Dataset(src), nc.Dataset(dst)

    per_target = {}
    for name in targets:
        a = np.asarray(s_[name][:], dtype=np.float64)
        b = np.asarray(r[name][:], dtype=np.float64)
        diff = b - a
        nz = a != 0
        rel = np.zeros_like(diff)
        rel[nz] = np.abs(diff[nz] / a[nz])
        per_target[name] = {
            "max_abs_diff": float(np.max(np.abs(diff))),
            "max_rel_diff": float(np.max(rel)) if nz.any() else 0.0,
            "n_changed": int(np.sum(diff != 0)),
            "n_total": int(diff.size),
        }

    other_ok, max_other_diff, changed_vars = True, 0.0, []
    for vname in s_.variables:
        a, b = np.asarray(s_[vname][:]), np.asarray(r[vname][:])
        if a.shape != b.shape:
            other_ok, changed_vars = False, changed_vars + [vname]
            continue
        d_ = (float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
              if a.dtype.kind in "fc" and a.size else
              (0.0 if np.array_equal(a, b) else 1.0))
        if vname not in targets:
            max_other_diff = max(max_other_diff, d_)
            if d_ != 0.0:
                other_ok, changed_vars = False, changed_vars + [vname]

    attrs_ok = s_.ncattrs() == r.ncattrs() and all(
        _attr_equal(getattr(s_, a), getattr(r, a)) for a in s_.ncattrs())
    for vname in s_.variables:
        if s_[vname].ncattrs() != r[vname].ncattrs():
            attrs_ok = False
        else:
            attrs_ok = attrs_ok and all(
                _attr_equal(getattr(s_[vname], a), getattr(r[vname], a))
                for a in s_[vname].ncattrs())
    s_.close()
    r.close()

    tn = per_target["tn"]
    return {
        "targets": tuple(targets),
        "per_target": per_target,
        # tn-named keys kept because verdict360.setup_nemo prints them.
        "max_abs_tn_diff": tn["max_abs_diff"],
        "max_rel_tn_diff": tn["max_rel_diff"],
        "n_tn_changed": tn["n_changed"],
        "n_tn_total": tn["n_total"],
        "other_vars_bit_identical": other_ok,
        "max_other_var_diff": max_other_diff,
        "changed_vars_besides_targets": changed_vars,
        "attrs_bit_identical": attrs_ok,
    }


def _demo_self_check():
    """Runnable check of BOTH conventions on the committed source restart.

    one-level : tn moves, tb stays bit-identical (the recorded convention).
    two-level : tn AND tb move, with the SAME RELATIVE draw -- asserted cell
                by cell, which is the property the whole discriminator rests
                on. Drawing twice instead of reusing one draw would still
                move both fields and still pass a naive "both changed" test;
                this check is written so it FAILS in that case.
    Both also assert the relative scale lands in (0, 1e-12) and that every
    other variable and attribute is untouched.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # perturb() itself already raises SystemExit if other_vars_bit_identical
        # or attrs_bit_identical comes back False, so re-asserting them here
        # would be dead code.
        one = Path(td) / "one"
        rep = perturb(seed=999, dest_dir=one)
        assert rep["targets"] == ("tn",), rep["targets"]
        assert 0 < rep["max_rel_tn_diff"] < 1e-12, rep["max_rel_tn_diff"]
        # land cells carry tn==0 exactly, and factor*0==0, so those stay
        # unchanged by construction -- only wet cells are expected to move.
        assert 0 < rep["n_tn_changed"] < rep["n_tn_total"], rep["n_tn_changed"]
        src_tb = np.asarray(nc.Dataset(SRC)["tb"][:], dtype=np.float64)
        one_tb = np.asarray(nc.Dataset(one / SRC.name)["tb"][:], dtype=np.float64)
        assert np.array_equal(src_tb, one_tb), "one-level kick moved tb"

        two = Path(td) / "two"
        rep2 = perturb(seed=999, dest_dir=two, both_levels=True)
        assert rep2["targets"] == ("tn", "tb"), rep2["targets"]
        for name in ("tn", "tb"):
            pt = rep2["per_target"][name]
            assert 0 < pt["max_rel_diff"] < 1e-12, (name, pt)
            assert 0 < pt["n_changed"] < pt["n_total"], (name, pt)
        # THE discriminating assertion: one draw, reused. The per-cell
        # relative change must be IDENTICAL on tn and tb wherever both are
        # nonzero. Two independent draws fail this by ~1e-14 relative.
        d = nc.Dataset(two / SRC.name)
        s_ = nc.Dataset(SRC)
        rel = {}
        for name in ("tn", "tb"):
            a = np.asarray(s_[name][:], dtype=np.float64)
            b = np.asarray(d[name][:], dtype=np.float64)
            rel[name] = np.where(a != 0, (b - a) / np.where(a != 0, a, 1.0), np.nan)
        both = np.isfinite(rel["tn"]) & np.isfinite(rel["tb"])
        assert both.sum() > 0, "no cell where both tn and tb are nonzero"
        dev = np.max(np.abs(rel["tn"][both] - rel["tb"][both]))
        scale = float(np.nanmax(np.abs(rel["tn"])))
        # The reconstructed relative change carries an fp64 round-off floor of
        # ~1 ulp = 2.2e-16 ABSOLUTE (b = fl(a*(1+f)) loses ulp(a), and dividing
        # by a turns that into ~eps_machine on f). Against a draw of size
        # ~4e-14 that is a ~0.5% deviation. INDEPENDENT draws would put dev at
        # the same order as the draw itself, i.e. ratio ~1 -- so the bar sits
        # two decades above the measured round-off and two decades below the
        # failure mode it exists to catch. MEASURED here: dev/scale ~ 5e-3.
        assert dev / scale < 0.05, (
            f"tn and tb did not get the SAME draw: max per-cell deviation "
            f"{dev:.3e} is {dev / scale:.3f} of the draw size {scale:.3e} "
            f"(round-off alone gives ~5e-3; independent draws give ~1)")
        # and the one-level tn kick must be the SAME draw as the two-level one
        # (same seed) -- otherwise the two arms differ in more than the levels.
        one_tn = np.asarray(nc.Dataset(one / SRC.name)["tn"][:], dtype=np.float64)
        two_tn = np.asarray(d["tn"][:], dtype=np.float64)
        assert np.array_equal(one_tn, two_tn), \
            "same seed gave different tn between the one- and two-level arms"
        d.close()
        s_.close()
    print("self-check OK: one-level moves tn only; two-level moves tn AND tb "
          "with the SAME per-cell relative draw (deviation <5% of the draw, "
          "round-off only); same seed "
          "gives the same tn in both arms; every other var/attr bit-identical; "
          "relative scale in (0, 1e-12)")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--both-levels"]
    both = "--both-levels" in sys.argv[1:]
    if not argv:
        if both:
            raise SystemExit("--both-levels takes a seed and a dest_dir")
        _demo_self_check()
        sys.exit(0)
    if len(argv) != 2:
        raise SystemExit(__doc__.split("Usage")[1])
    seed_ = int(argv[0])
    dest_ = Path(argv[1])
    rep_ = perturb(seed_, dest_, both_levels=both)
    print(f"seed={seed_} dest={dest_} targets={rep_['targets']}")
    for name_, pt_ in rep_["per_target"].items():
        print(f"  {name_}: max|d|={pt_['max_abs_diff']:.3e}  "
              f"max_rel|d/x|={pt_['max_rel_diff']:.3e}  "
              f"n_changed={pt_['n_changed']}/{pt_['n_total']}")
    print(f"  other_vars_identical={rep_['other_vars_bit_identical']}  "
          f"max_other_var_diff={rep_['max_other_var_diff']:.3e}  "
          f"attrs_identical={rep_['attrs_bit_identical']}")
