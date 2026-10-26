#!/usr/bin/env python
"""#1455 -- the NONLINEAR momentum terms at the wall, partitioned three ways.

Pre-registration: ``PREREG_wall_nonlinear_partition.md``, written before any
number here existed. Every linear candidate is closed (drag refuted
over-determined, friction refuted on magnitude under both weightings, the EEN
vorticity flux's promotion void), the forcing matches to 2.2e-19 Pa, and the
deficit is a pure recirculation. What is left is the nonlinear terms.

DINO runs ``ln_dynadv_vec=.true.`` with ``nn_dynkeg=1``, so the horizontal
momentum advection is the KINETIC-ENERGY GRADIENT (Hollingsworth branch) plus
VERTICAL ADVECTION, with the rotational part in the already-decomposed EEN
flux.

RETRACTED PREMISE (review finding 2). This probe was motivated by the idea that
NEMO's Hollingsworth meridional stencil reads the land row while legoESM fills
it by edge replication, so that the two agree "only because the stored velocity
there is exactly zero". THAT IS FALSE. legoESM's fill
(``ocean_pe_latlon_cgrid.py:1531``) returns the TRUE row j-1 value for every
scored row; it differs from a raw index only at row 0 and the last row, both of
which are entirely dry. The two stencils therefore agree for ANY row-0 value.
The motivating mechanism did not exist; the measurement below stands on its
own. NEMO dumps all three of keg / hpg / zad in isolation, and legoESM exposes
the KE and pressure gradients separately from the same helper that builds them,
so the "advection + KE/pressure union" the residual list assumed is actually a
THREE-WAY partition and is scored as one.

THE THREE LEGS, each on the statistic its own bar was calibrated for -- this is
the rescore's lesson applied in advance, since scoring the enrichment leg with
the magnitude statistic is what promoted a candidate that never qualified:
  MAGNITUDE   signed, MASS-weighted wall-row mean; >=5.9e-11 clears, <5.9e-12
              refutes. The level mean is printed but NOT scored.
  ENRICHMENT  row_report's row-mean ABSOLUTE ratio, on BOTH interior sets;
              straddling the 3x bar means undecided, not re-cut.
  SIGN        NEMO - legoESM must be POSITIVE (eastward) on >=3 of 4 wall rows,
              since legoESM's recirculating lobe is too strong westward.

Nothing is re-derived: the bridge, the reductions, the dump reader and the
face-convention mapping all come from the siblings that established them.

This probe prints numbers; every verdict string is computed from them.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.wall_nonlinear_partition
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import wall_term_discriminators as W  # noqa: E402
from acc_momentum_budget import _load_full_3d, _u_to_nemo  # noqa: E402


def _v_to_nemo(a):
    """legoESM v-face array (n_lat+1, n_lon, ...) -> NEMO v-point layout.

    legoESM v-face j is the SOUTH face of T-row j; NEMO's V-point j is the
    NORTH face of T-row j == legoESM face j+1.
    """
    return np.asarray(a)[1:]
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

RUN, JPI, JPJ, JPKM1, HLS = W.RUN, W.JPI, W.JPJ, W.JPKM1, W.HLS
WALL_ROWS, FAR = W.WALL_ROWS, W.INTERIOR_ROWS_FAR
BAR_MAG, BAR_REFUTE, BAR_ENRICH = W.BAR_MAG, W.BAR_MAG_REFUTE, W.BAR_ENRICH
BAR_CLOSURE = 1e-15          # control 1: NEMO's own partition, relative to RMS
_OUT = REPO_ROOT / "results" / "dino_1455_nonlinear"


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  run={RUN}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
    print(f"PROVENANCE  bars: magnitude>={BAR_MAG:.2e} (MASS-weighted), "
          f"refute<{BAR_REFUTE:.2e}, enrichment>={BAR_ENRICH} (row_report "
          f"mean|.|), sign>=3/4 positive")
    nml = (RUN / "namelist_cfg").read_text().splitlines()
    for key in ("ln_dynadv_vec", "nn_dynkeg", "ln_dynzad", "ln_hpg_sco"):
        for ln in nml:
            sl = ln.split("!")[0]
            if key in sl and "=" in sl:
                print(f"BUILD  namelist_cfg: {sl.strip()}")
                break


def _finite(name, a):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise SystemExit(f"{name} carries non-finite values -- NaN is fatal")
    return a


def dump(name):
    return _finite(name, _load_full_3d(str(RUN / name), JPI, JPJ, JPKM1, HLS))


def lego_terms(g, br, cfg, mc):
    """legoESM's KE gradient, pressure gradient and vertical advection.

    Recomputed from the SAME helpers the tendency function calls, then checked
    against the model's own published diagnostics before anything is scored.
    """
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_geometry_and_density,
        _bc_ke_and_pressure_gradients,
        _bc_vertical_and_depthmean_velocity,
        _bc_vertical_momentum_advection,
    )
    from legoesm.ocean.vertical import OceanPartialCellCoordinate

    st, z, grid = br.state, br.z_coord, br.geometry
    u, v, mask = st.u.data, st.v.data, st.land_mask.data
    if isinstance(z, OceanPartialCellCoordinate):
        um3, vm3 = compute_face_masks_3d(z.is_active, grid)
    else:
        um3, vm3 = st.u_mask.data[..., None], st.v_mask.data[..., None]
    eta_safe = jnp.maximum(
        st.eta.data, mc.min_water_column_m - st.H_bathy.data) * mask
    J, h_k, rho_p, p_p = _bc_geometry_and_density(
        eta_safe, st.H_bathy.data, z, mc, st.T.data, st.S.data, mask, grid,
        mc.rho_0, mc.g)
    h_u, h_v, _fd, w, u_p, v_p = _bc_vertical_and_depthmean_velocity(
        h_k, u, v, um3, vm3, grid, z, st.u_mask.data, st.v_mask.data)
    dKE_dx, dp_dx, _a, _b = _bc_ke_and_pressure_gradients(
        u, v, p_p, rho_p, grid, mc, z, eta_safe, st.H_bathy.data, mc.g, mask)
    zu = jnp.zeros_like(dKE_dx)
    zv = jnp.zeros_like(_a)
    _du, _dv, vadv_u, _vadv_v = _bc_vertical_momentum_advection(
        zu, zv, u_p, v_p, w, h_u, h_v, um3, vm3, grid,
        mc.momentum_advection, None, mc, True, u_full=u, v_full=v)
    def _n(x):
        return _u_to_nemo(np.asarray(x, dtype=np.float64))[..., :JPKM1]

    def _nv(x):
        return _v_to_nemo(np.asarray(x, dtype=np.float64))[..., :JPKM1]
    return dict(keg=_n(-dKE_dx * um3), hpg=_n(-dp_dx / mc.rho_0 * um3),
                zad=_n(vadv_u * um3),
                ke_pgf=_n((-dKE_dx - dp_dx / mc.rho_0) * um3),
                # REVIEW FINDING 5: the v dumps exist and were never read, so
                # "the nonlinear terms are refuted" was a ZONAL-momentum
                # statement. The meridional component costs one line each.
                keg_v=_nv(-_a * vm3), hpg_v=_nv(-_b / mc.rho_0 * vm3),
                zad_v=_nv(_vadv_v * vm3))


def control_reassembly(g, br, cfg, mc, L):
    """legoESM's recomputed pieces must match its own published diagnostics."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    print("\n=== CONTROL 3 -- reassembly vs the model's own diagnostics ===")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    _t, d = model.tendencies_with_diagnostics(
        br.state, surface_forcing=None, dt=float(cfg.dt))
    ok = True
    for key, field in (("ke_pgf", d.KE_PGF_u), ("zad", d.vertadv_u)):
        ref = _u_to_nemo(np.asarray(field.data, dtype=np.float64))[..., :JPKM1]
        dd = float(np.abs(L[key] - _finite(key, ref)).max())
        print(f"  {key:<8s} max|recomputed - diagnostic| = {dd:.3e} m/s2  "
              f"{'PASS' if dd == 0.0 else 'FAIL'}")
        ok &= dd == 0.0
    if not ok:
        raise SystemExit(
            "the recomputed legoESM terms differ from the model's own "
            "diagnostics -- the partition being scored is not the one the "
            "model integrates")


def control_closure(N):
    """A SHAPE-AND-SELF-CONSISTENCY check. NOT a validation of the reader.

    REVIEW FINDING 4, acted on. The pre-registration called this "a known-answer
    check on the dump reader". It is not: ``dynadv.F90:92`` dumps ``puu(Krhs)``
    after ``dyn_keg`` and ``:101`` forms the vertical-advection dump as
    ``puu(Krhs) - zkeg``, so ``keg + zad == adv`` is an ALGEBRAIC IDENTITY that
    holds for any reader, any layout, any halo strip and regardless of whether
    the vector-form assumption is right. It prints 0.000e+00 by construction.

    What it DOES check is that the three files have the same shape and are
    mutually self-consistent, which is worth one line. The reader is really
    validated by the shift/zero controls in ``--self-test`` and by the fact
    that a misaligned comparison scores 10+ orders louder (control 4).
    """
    print("\n=== CONTROL 1 -- the three dumps are self-consistent "
          "(an IDENTITY, not a reader check) ===")
    lhs = N["adv"]
    rhs = N["keg"] + N["zad"]
    rms = float(np.sqrt(np.mean(lhs ** 2)))
    rel = float(np.abs(lhs - rhs).max()) / rms
    print(f"  max|dyn_adv - (keg + zad)| / RMS = {rel:.3e}  vs bar "
          f"{BAR_CLOSURE:.0e}  {'PASS' if rel < BAR_CLOSURE else 'FAIL'}")
    if rel >= BAR_CLOSURE:
        raise SystemExit(
            f"NEMO's vector-form partition does not close ({rel:.3e}) -- the "
            "dump reader or the vector-form assumption is wrong, and nothing "
            "attributed to keg or zad below would mean anything")


def score(name, diff, wet3, h, term=None):
    """The three legs, each on its registered statistic."""
    d = _finite(name, diff)
    sg_l, _ = W.coherent_rows(d, wet3, None)
    sg_m, _ = W.coherent_rows(d, wet3, h)
    mag = float(np.mean([abs(sg_m[j]) for j in WALL_ROWS]))
    mag_l = float(np.mean([abs(sg_l[j]) for j in WALL_ROWS]))
    far_m = float(np.mean([abs(sg_m[j]) for j in FAR]))
    # ENRICHMENT on the REGISTERED statistic (row_report's mean|.|), both
    # interior sets, via the sibling that owns it.
    # ``term=`` supplies the whole-term magnitude, without which the table
    # cannot distinguish "the two models agree" from "this term is too small
    # to matter at the wall whatever either model computes" (review finding 1).
    W.row_report(name, d, wet3, term=term)
    r = W.LAST_ROW[name]
    pos = sum(int(sg_m[j] > 0) for j in WALL_ROWS)
    # THE NON-CANCELLING UPPER BOUND. Every reduction above can cancel, and
    # this campaign has already been burned once by crediting a term with a
    # difference that vanished under projection. The MAXIMUM |difference| over
    # every wet cell in the wall rows cannot cancel by construction: if even
    # that is below the carry bar, the term is refuted over-determined and no
    # choice of reduction can rescue it.
    wall_cells = np.zeros(d.shape[0], bool)
    wall_cells[WALL_ROWS] = True
    m = wet3 & wall_cells[:, None, None]
    pw = float(np.abs(d[m]).max()) if m.any() else float("nan")
    legs = dict(
        magnitude=(mag >= BAR_MAG),
        refuted=(mag < BAR_REFUTE),
        enrichment=(r["enrich"] >= BAR_ENRICH),
        enrichment_far=(r["enrich_far"] >= BAR_ENRICH),
        straddles=r["straddles"],
        sign=(pos >= 3),
    )
    head = float("nan")
    if term is not None:
        # THE HEADROOM. Score the term as if legoESM computed ZERO for it: that
        # is the largest error this term could possibly contribute at the wall.
        # If the headroom itself is below the carry bar, the term is refuted by
        # its own SIZE and the measured agreement is not doing the work.
        hs, _ = W.coherent_rows(_finite(name + " term", term), wet3, h)
        head = float(np.mean([abs(hs[j]) for j in WALL_ROWS]))
    return dict(name=name, mag=mag, mag_level=mag_l, far=far_m, pw=pw,
                headroom=head,
                enrich=r["enrich"], enrich_far=r["enrich_far"], pos=pos,
                rows_mass=[float(sg_m[j]) for j in WALL_ROWS], legs=legs)


def verdict(s):
    L = s["legs"]
    if L["refuted"]:
        # The magnitude leg is a DOUBLY-cancelling statistic (signed in the
        # vertical and signed in the zonal). Review finding: the non-cancelling
        # maximum was printed but never gated, so it was decorative. It is a
        # gate now -- a term may only be called refuted if its LARGEST single
        # cell in the wall rows is also under the carry bar.
        if s["pw"] >= BAR_MAG:
            return (f"NOT refuted: mean {s['mag']:.2e} is small but the "
                    f"largest single cell is {s['pw']:.2e} >= {BAR_MAG:.1e} "
                    "-- the mean is cancelling")
        return (f"REFUTED (mean {s['mag']:.2e} < {BAR_REFUTE:.1e} AND max "
                f"cell {s['pw']:.2e} < {BAR_MAG:.1e})")
    if L["magnitude"] and L["enrichment"] and L["sign"]:
        return "CLEARS ALL THREE LEGS -- candidate"
    fails = []
    if not L["magnitude"]:
        fails.append(f"magnitude {s['mag']:.2e} < {BAR_MAG:.1e}")
    if not L["enrichment"]:
        fails.append(f"enrichment {s['enrich']:.2f}x < {BAR_ENRICH}"
                     + (" (INTERIOR SETS STRADDLE -> undecided)"
                        if L["straddles"] else ""))
    if not L["sign"]:
        fails.append(f"sign {s['pos']}/4 positive < 3")
    return "REFUTED as owner: " + "; ".join(fails)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()

    g, br, cfg, mc = W.build_lego()
    W._gate_dry_faces(br)                      # control 2, on the scoring path
    for k in ("ke_gradient_scheme", "momentum_advection",
              "vertical_momentum_scheme", "adaptive_implicit_vertadv",
              "vorticity_scheme", "een_metric_weighting"):
        print(f"  resolved: {k} = {getattr(mc, k, '<absent>')!r}")

    N = dict(keg=dump("keg_dump_du.bin"), zad=dump("zad_dump_du.bin"),
             hpg=dump("hpg_dump_du.bin"),
             adv=dump("stp_dump_03_dynadv_du.bin"))
    control_closure(N)
    L = lego_terms(g, br, cfg, mc)
    control_reassembly(g, br, cfg, mc, L)

    umask3 = np.asarray(g.umask)[..., :JPKM1] > 0.5
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    from legoesm.ocean.vertical import compute_layer_thickness
    st = br.state
    h_u = _u_to_nemo(np.asarray(min_cell_to_uface(jnp.asarray(np.asarray(
        compute_layer_thickness(
            np.maximum(np.asarray(st.eta.data, dtype=np.float64),
                       mc.min_water_column_m
                       - np.asarray(st.H_bathy.data, dtype=np.float64))
            * np.asarray(st.land_mask.data, dtype=np.float64),
            st.H_bathy.data, br.z_coord,
            min_water_column_m=mc.min_water_column_m),
        dtype=np.float64))), dtype=np.float64))[..., :JPKM1]

    print("\n=== THE THREE-WAY PARTITION AT THE WALL (u-momentum) ===")
    out = []
    for key, label in (("keg", "KE gradient (Hollingsworth)"),
                       ("hpg", "pressure gradient"),
                       ("zad", "vertical advection"),
                       ("ke_pgf", "KE+pressure UNION (cross-check)")):
        nemo = N["keg"] + N["hpg"] if key == "ke_pgf" else N[key]
        out.append(score(label, nemo - L[key], umask3, h_u, term=nemo))

    print(f"\n  {'term':<34s}{'MAG(mass)':>12s}{'HEADROOM':>12s}"
          f"{'head/bar':>10s}{'max|.| wall':>13s}{'enr':>7s}{'sign':>6s}"
          f"  verdict")
    for s in out:
        print(f"  {s['name']:<34s}{s['mag']:>12.4e}{s['headroom']:>12.4e}"
              f"{s['headroom'] / BAR_MAG:>10.2f}{s['pw']:>13.4e}"
              f"{s['enrich']:>7.2f}{s['pos']:>4d}/4  {verdict(s)}")
    print()
    print("  HEADROOM is the score legoESM would earn by computing ZERO for "
          "the term -- i.e. the")
    print("  largest error it could contribute at the wall at 100% error. "
          "Where head/bar < 1 the")
    print("  term is refuted BY ITS OWN SIZE and the measured agreement is "
          "not doing the work;")
    print("  only a term with real headroom tests the two models against each "
          "other. (Review")
    print("  finding: the table could not previously tell those two "
          "situations apart.)")
    print()
    for s in out:
        od = s["pw"] / BAR_MAG
        print(f"  {s['name']:<34s} even its LARGEST single-cell difference in "
              f"the wall rows is {od:.3e}x the carry bar"
              + ("  -> refuted over-determined, no reduction can rescue it"
                 if od < 1.0 else "  -> NOT bounded below the bar"))
    print(f"\n  bars: magnitude >= {BAR_MAG:.2e} (< {BAR_REFUTE:.2e} refutes), "
          f"enrichment >= {BAR_ENRICH} on the REGISTERED mean|.| statistic, "
          f"sign >= 3/4 positive")
    for s in out:
        print(f"  {s['name']:<34s} wall rows (mass) "
              + " ".join(f"{v:+.3e}" for v in s["rows_mass"]))

    # ---- CONTROL 4: THE INSTRUMENT MUST BE ABLE TO SEE A DIFFERENCE -------
    # Three refutations in a row is exactly when to ask whether the comparison
    # can fail at all. Deliberately MISPAIR the terms (NEMO's KE gradient
    # against legoESM's vertical advection, and vice versa): if the machinery
    # is sound those must score ENORMOUSLY, and the matched pairs' tiny numbers
    # are genuine agreement rather than a self-comparison, a zero array, or a
    # mask that empties the scored set.
    print("\n=== CONTROL 4 -- positive control: mispaired terms must SCREAM ===")
    worst_matched = max(s["pw"] for s in out)
    mis = []
    for a, b in (("keg", "zad"), ("zad", "keg"), ("hpg", "zad")):
        d = N[a] - L[b]
        m = np.zeros(d.shape[0], bool)
        m[WALL_ROWS] = True
        pw = float(np.abs(d[umask3 & m[:, None, None]]).max())
        mis.append(pw)
        print(f"  NEMO {a:<4s} vs legoESM {b:<4s}: max|diff| in wall rows "
              f"{pw:.4e} m/s2  ({pw / max(worst_matched, 1e-300):.3e}x the "
              f"worst MATCHED pair)")
    if min(mis) <= 100.0 * worst_matched:
        raise SystemExit(
            f"a MISPAIRED comparison scores {min(mis):.3e}, within 100x of the "
            f"worst matched pair {worst_matched:.3e} -- this probe cannot "
            "distinguish the right terms from the wrong ones, so its "
            "refutations mean nothing")
    print("  matched pairs are >=100x quieter than every mispairing, so the "
          "comparison is discriminating, not vacuous")

    # ---- THE MERIDIONAL COMPONENT (review finding 5) ---------------------
    print("\n=== THE SAME PARTITION FOR v-MOMENTUM ===")
    vmask3 = np.asarray(g.vmask)[..., :JPKM1] > 0.5
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_vface
    h_v = _v_to_nemo(np.asarray(min_cell_to_vface(jnp.asarray(np.asarray(
        compute_layer_thickness(st.eta.data, st.H_bathy.data, br.z_coord,
                                min_water_column_m=mc.min_water_column_m),
        dtype=np.float64)), br.geometry), dtype=np.float64))[..., :JPKM1]
    print(f"  {'term':<30s}{'MAG(mass)':>12s}{'HEADROOM':>12s}"
          f"{'head/bar':>10s}{'max|.| wall':>13s}")
    for key, label in (("keg_v", "KE gradient (v)"),
                       ("hpg_v", "pressure gradient (v)"),
                       ("zad_v", "vertical advection (v)")):
        nv = dump(f"{key.split('_')[0]}_dump_dv.bin")
        sv = score(label, nv - L[key], vmask3, h_v, term=nv)
        print(f"  {label:<30s}{sv['mag']:>12.4e}{sv['headroom']:>12.4e}"
              f"{sv['headroom'] / BAR_MAG:>10.2f}{sv['pw']:>13.4e}")

    # ---- THE DOMAIN-WIDE BOUND (review finding 6) ------------------------
    # "Candidates exhausted" is a domain-wide claim; every bound above is
    # wall-only. The largest single-cell difference ANYWHERE, times 90 days of
    # perfectly coherent accumulation, caps the velocity error each term could
    # ever build -- against the 4.6e-4 m/s the deficit represents.
    print("\n=== THE DOMAIN-WIDE BOUND ===")
    tau = 90.0 * 86400.0
    target = 4.6e-4
    for key, label in (("keg", "KE gradient"), ("hpg", "pressure gradient"),
                       ("zad", "vertical advection")):
        gmax = float(np.abs((N[key] - L[key])[umask3]).max())
        print(f"  {label:<24s} max|diff| over EVERY wet cell {gmax:.3e} m/s2 "
              f"-> at most {gmax * tau:.2e} m/s in 90 d, "
              f"{gmax * tau / target:.2e}x the {target:.1e} m/s target")

    if args.self_test:
        return self_test(g, br, cfg, mc, N, L, umask3, h_u, out)
    return 0


def self_test(g, br, cfg, mc, N, L, umask3, h_u, out) -> int:
    """Non-vacuity, and the dry-row trap on the term that reads land by design."""
    print("\n=== SELF-TEST ===")
    base = score("st base", N["keg"] - L["keg"], umask3, h_u)["mag"]
    # (i) a WET wall-row plant must move the score.
    st = br.state
    uu = np.asarray(st.u.data, dtype=np.float64).copy()
    uu[WALL_ROWS[0], :, :] += 1e-2
    br2 = br._replace(state=st._replace(u=st.u.replace(data=uu)))
    L2 = lego_terms(g, br2, cfg, mc)
    m2 = score("st wet plant", N["keg"] - L2["keg"], umask3, h_u)["mag"]
    assert m2 != base, ("planting a wet wall row moved nothing -- the KE leg "
                        "cannot fail and proves nothing")
    print(f"(i) wet wall-row plant moved the KE score {base:.4e} -> {m2:.4e}")
    # (ii) NEMO-SIDE CONTROLS. REVIEW FINDING 3, acted on: the previous check
    #      planted on the DRY row and moved only legoESM, which duplicated (i)
    #      and proved nothing about the reference. Nothing in this probe could
    #      detect a misaligned, mispaired or zeroed NEMO field, because that
    #      side is a frozen file. So corrupt it deliberately: every corruption
    #      must land at or above the carry bar, while the true pairing sits
    #      orders below.
    ref = N["keg"]
    corrupt = {
        "shifted 1 row": np.roll(ref, 1, axis=0),
        "shifted 1 column": np.roll(ref, 1, axis=1),
        "shifted 1 level": np.roll(ref, 1, axis=2),
        "replaced by zeros": np.zeros_like(ref),
        "mispaired with zad": N["zad"],
    }
    print(f"(ii) NEMO-side controls on the KE gradient (true pairing scores "
          f"{base:.3e}):")
    for lbl, bad in corrupt.items():
        sc = score(f"st {lbl}", bad - L["keg"], umask3, h_u)["mag"]
        print(f"       {lbl:<20s} {sc:.4e}  ({sc / BAR_MAG:.2f}x the carry bar)")
        assert sc > 100.0 * base, (
            f"corrupting the NEMO reference ({lbl}) scored {sc:.3e}, within "
            f"100x of the TRUE pairing {base:.3e} -- this probe cannot tell a "
            "correct reference from a broken one, so its refutations are void")
    print("     every corruption lands >=100x above the true pairing, so the "
          "agreement is not an")
    print("     alignment artifact, a self-comparison, or a comparison against "
          "zero.")

    # (iii) the verdict rule must reach its branches.
    seen = {verdict(dict(mag=1e-13, mag_level=0, far=0, enrich=9, enrich_far=9,
                         pos=4, rows_mass=[], name="", pw=1e-13,
                         legs=dict(refuted=True, magnitude=False,
                                   enrichment=True, enrichment_far=True,
                                   straddles=False, sign=True))),
            verdict(dict(mag=1e-9, mag_level=0, far=0, enrich=9, enrich_far=9,
                         pos=4, rows_mass=[], name="", pw=1e-9,
                         legs=dict(refuted=False, magnitude=True,
                                   enrichment=True, enrichment_far=True,
                                   straddles=False, sign=True))),
            verdict(dict(mag=1e-9, mag_level=0, far=0, enrich=1, enrich_far=1,
                         pos=0, rows_mass=[], name="", pw=1e-9,
                         legs=dict(refuted=False, magnitude=True,
                                   enrichment=False, enrichment_far=False,
                                   straddles=False, sign=False)))}
    assert len(seen) == 3, f"the verdict rule collapsed to {seen}"
    print("(iii) the verdict rule reaches refute / clear / partial-fail")
    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
