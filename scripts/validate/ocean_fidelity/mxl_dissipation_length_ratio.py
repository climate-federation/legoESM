"""How much more does NEMO nn_mxl=2 dissipate than nn_mxl=3, and where?

THE QUESTION.  ORCA1's ``namelist_cfg`` sets ``nn_mxl = 2``; every day-90
tripole arm has run our ``tke_mxl_choice=3`` (NEMO nn_mxl=3).  The two choices
share the eddy-coefficient length ``l_k = min(lup, ldn)`` and differ ONLY in
the dissipation length used by ``eps = c_eps * e^{3/2} / l_eps``:

    choice 3 (nn_mxl=3):  l_eps = sqrt(lup * ldn)
    choice 4 (nn_mxl=2):  l_eps = min(lup, ldn) = l_k

Since ``min(a,b) <= sqrt(a*b)`` for positive a, b, choice 4 always dissipates
at least as much.  This probe measures the ratio

    R = l_eps(choice 3) / l_eps(choice 4) = sqrt(lup*ldn) / min(lup, ldn) >= 1

on a SAVED model state, per latitude band and depth bin.  R is a pure geometric
property of the lup/ldn pair, so it answers the one question a 90-day run
cannot answer cheaply: **is the difference between the two NEMO settings
seasonally selective?**  In a stably stratified summer boundary layer lup ~= ldn
and R -> 1 (the two settings coincide, which the measured day-30 A/B confirms
at 0.9 m of mixed layer).  In autumn convection lup is bounded by the distance
to the surface while ldn runs to the pycnocline, so lup << ldn and R grows.

WHAT R IS NOT.  R is NOT a mixed-layer-depth prediction and must never be
quoted as one.  The dissipation length feeds back through the TKE solve over
90 days; only the integration gives the MLD response.  R bounds the MECHANISM:
R ~= 1 in the Southern Ocean at day 90 REFUTES nn_mxl as the autumn lever
outright, and no run is needed.

INSTRUMENT CONTROLS, all asserted at runtime (the run aborts if one fails):
  C1  ``l_k`` must be BIT-IDENTICAL between the two choices.  It is the same
      expression in both branches, so any difference is a probe bug.  This is
      a check with a known answer, run before any number is reported.
  C2  ``l_eps(3) >= l_eps(4)`` everywhere, from ``sqrt(a*b) >= min(a,b)``.
  C3  R == 1 exactly wherever ``lup == ldn`` (recovered as
      ``l_eps(4) == max(lup,ldn)``-free identity: where R != 1 the two sweeps
      must genuinely differ).  Reported as the fraction of wet interfaces at
      R < 1 + 1e-12, i.e. the fraction where the two settings are the same
      operator.
  C4  A synthetic uniform column with e3t constant and a single sharp
      pycnocline is run first, where the sign of the effect is known by hand.

THE N-SQUARED INSTRUMENT.  The mixing-length sweeps are seeded by the buoyancy
length sqrt(2e)/N.  This probe uses the model's own NEMO ``bn2``
(``eos.compute_buoyancy_frequency_nemo_bn2``), the same instrument
``arctic_n2_compare.py`` uses, NOT the ``n2_mode="insitu"`` form the production
card selects.  The two differ by the compressibility term (measured +4.3e-5
elsewhere in this campaign).  Both choices are fed the IDENTICAL N2, so R --
a ratio of two lengths built from the same seed -- is insensitive to that at
first order; the ABSOLUTE lengths are not, and are reported as context only.

THE WIND ANCHOR.  ``ln_mxl0`` sets the surface seed
``l_sfc = max(rn_mxl0, vkarmn*2e5/(rho0*g)*|tau|)``.  The snapshot carries no
surface stress, so l_sfc is swept over two values and BOTH are reported:
the windless floor ``rn_mxl0 = 0.04 m`` and 1.5 m, a Southern-Ocean storm-track
value for |tau| ~ 0.35 N/m^2.  A larger anchor raises lup near the surface and
therefore SHRINKS R, so the floor case is the optimistic end of the bracket --
if R is small there too, the mechanism is dead either way.

Usage (CPU, seconds; wrap in sbatch per the login-node policy):
    python scripts/validate/ocean_fidelity/mxl_dissipation_length_ratio.py \
        --snapshot results/omip_nemo/nemolev_trp_mlefix_d90/snapshot_day0090.npz \
        --mesh-mask data/grids/eORCA1.2_mesh_mask.nc --label d90
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_REPO / "packages" / _p))
sys.path.insert(0, str(_REPO))

from global_tracer_content import (  # noqa: E402
    _native, load_mesh_depth_1d, load_mesh_latitude, load_mesh_metrics,
)

# Bands, south-to-north, matching compare_omip_nemo.py so the numbers can sit
# next to the MLD biases without a relabelling step.
_BANDS = (
    ("antarctic_S_of_45S", -91.0, -45.0),
    ("SH_midlat_45S_23S", -45.0, -23.0),
    ("tropics_23S_23N", -23.0, 23.0),
    ("NH_midlat_23N_45N", 23.0, 45.0),
    ("arctic_N_of_45N", 45.0, 91.0),
)
_DEPTH_BINS = (0.0, 20.0, 50.0, 100.0, 200.0, 500.0, 6000.0)


def _mesh_gdepw(mesh_mask_path):
    """Interface depth ladder gdepw_1d, same accessor as arctic_n2_compare."""
    import netCDF4 as nc
    ds = nc.Dataset(mesh_mask_path)
    try:
        return np.asarray(ds.variables["gdepw_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()


def _n2_bn2(T, S, gdept, gdepw_int):
    """Model's own NEMO bn2. T/S level-LAST, per the function's contract."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(gdept), jnp.asarray(gdepw_int)))


def _cards(iwm_enabled: bool):
    """The two ORCA1 zdftke cards, one variable apart (nn_mxl=3 vs nn_mxl=2).

    Built by the DRIVER's own factory so the probe cannot drift from the card
    the runs use; only ``mxl_choice`` differs between the two.
    """
    sys.path.insert(0, str(_REPO / "scripts" / "run"))
    from run_omip_core2 import orca1_zdftke_config
    common = dict(iwm_enabled=iwm_enabled, surface_bc="nemo_dirichlet",
                  prognostic=True)
    c3 = orca1_zdftke_config(mxl_choice=3, **common)
    c4 = orca1_zdftke_config(mxl_choice=4, **common)
    diff = [f for f in c3._fields if getattr(c3, f) != getattr(c4, f)]
    if diff != ["tke_mxl_choice"]:
        raise SystemExit(
            f"FATAL: the two cards differ in {diff}, expected only "
            "['tke_mxl_choice'] -- this is not a one-variable pair.")
    return c3, c4


def _lengths(e, N2, dz_half, dz_cell, cfg, l_sfc_value):
    """(l_k, l_eps) from the MODEL's compute_mixing_lengths -- not a re-derivation.

    ``signed_n2=False`` matches the production card (``n2_mode="insitu"``);
    ``boundary_cap`` is None because NEMO applies the direct distance-to-
    boundary bound only in nn_mxl=0, not in nn_mxl=2/3 (zdftke.F90:662,
    confirmed by codex 9393928).
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing.tke import compute_mixing_lengths
    anchor = jnp.full(e.shape[:-1], float(l_sfc_value))
    l_k, l_eps = compute_mixing_lengths(
        jnp.asarray(e), jnp.asarray(N2), jnp.asarray(dz_half), cfg,
        signed_n2=False, dz_cell=jnp.asarray(dz_cell), boundary_cap=None,
        l_surface_anchor=anchor)
    return np.asarray(l_k), np.asarray(l_eps)


def _synthetic_selftest(c3, c4):
    """C4: a hand-checkable column before any real number is reported.

    One column, uniform 10 m cells, TKE uniform.  Upper half is NEUTRAL
    (N2 = tiny) so the buoyancy length is huge and the sweeps are bounded only
    by the |dl/dz| <= e3t slope limit -- i.e. lup grows from the surface anchor
    while ldn grows from the bottom, so they differ strongly near the surface
    and R must exceed 1 there.  Lower half is STRONGLY stratified so the
    buoyancy length collapses below the cell size and BOTH sweeps are pinned to
    the same small value -- so R must be ~1 there.  A probe that reports R ~ 1
    in the neutral half, or R >> 1 in the stratified half, is broken.
    """
    nlev = 20
    dz_cell = np.full((1, nlev), 10.0)
    dz_half = np.full(nlev - 1, 10.0)
    e = np.full((1, nlev - 1), 1.0e-4)
    N2 = np.where(np.arange(nlev - 1) < (nlev - 1) // 2, 1.0e-12, 1.0e-2)[None, :]
    lk3, le3 = _lengths(e, N2, dz_half, dz_cell, c3, 0.04)
    lk4, le4 = _lengths(e, N2, dz_half, dz_cell, c4, 0.04)
    if not np.array_equal(lk3, lk4):
        raise SystemExit("FATAL C1(synthetic): l_k differs between the choices.")
    R = le3 / np.maximum(le4, 1e-30)
    top = float(np.max(R[0, : (nlev - 1) // 2]))
    bot = float(np.max(R[0, (nlev - 1) // 2:]))
    print(f"[selftest] neutral-half max R = {top:.4f} (must be > 1.05)")
    print(f"[selftest] stratified-half max R = {bot:.4f} (must be < 1.05)")
    if not (top > 1.05):
        raise SystemExit(
            "FATAL C4: the probe reports no dissipation-length separation in a "
            "neutral column, where the sweeps provably diverge. Probe is wrong.")
    if not (bot < 1.05):
        raise SystemExit(
            "FATAL C4: the probe reports a separation in a strongly stratified "
            "column, where both sweeps are pinned to the buoyancy length.")
    print("[selftest] PASS — the metric responds to the regime it names.\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--l-sfc", type=float, nargs="+", default=[0.04, 1.5],
                   help="ln_mxl0 surface anchors [m] to sweep (see module doc).")
    p.add_argument("--no-iwm", action="store_true",
                   help="build the card with iwm_enabled=False (the d90 arms "
                        "all pass --iwm, so the default True matches them).")
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    c3, c4 = _cards(iwm_enabled=not a.no_iwm)
    print(f"[card] nn_mxl=3 -> tke_mxl_choice={c3.tke_mxl_choice}, "
          f"nn_mxl=2 -> tke_mxl_choice={c4.tke_mxl_choice}, "
          f"one variable apart\n")
    _synthetic_selftest(c3, c4)

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)   # (nlev,nj,ni)
    lat = load_mesh_latitude(a.mesh_mask)                   # (nj,ni)
    gdept = load_mesh_depth_1d(a.mesh_mask)                 # (nlev,)
    gdepw = _mesh_gdepw(a.mesh_mask)
    gdepw_int = gdepw[1:]                                   # (nlev-1,)

    z = dict(np.load(a.snapshot))
    T = np.transpose(_native(z["T"]), (1, 2, 0))            # (nj,ni,nlev)
    S = np.transpose(_native(z["S"]), (1, 2, 0))
    tke = np.transpose(_native(z["tke"]), (1, 2, 0))        # (nj,ni,nlev-1)
    if T.shape[:2] != lat.shape:
        raise SystemExit(f"FATAL: snapshot {T.shape[:2]} vs mesh {lat.shape}")

    N2 = _n2_bn2(T, S, gdept, gdepw_int)                    # (nj,ni,nlev-1)
    dz_cell = np.transpose(e3t, (1, 2, 0))                  # (nj,ni,nlev) LOCAL
    dz_half = np.diff(gdept)                                # (nlev-1,)

    # Wet interior interfaces: both bracketing cells wet.
    wet_c = np.transpose(tmask, (1, 2, 0)) > 0.5            # (nj,ni,nlev)
    wet_i = wet_c[..., :-1] & wet_c[..., 1:]
    # Interface volume weights, same construction as arctic_n2_compare.
    dV = np.transpose(e1t[None] * e2t[None] * e3t * (tmask > 0.5), (1, 2, 0))
    w_i = 0.5 * (dV[..., :-1] + dV[..., 1:]) * wet_i
    depth_i = 0.5 * (gdept[:-1] + gdept[1:])                # (nlev-1,)

    out = {"snapshot": a.snapshot, "mesh_mask": a.mesh_mask, "label": a.label,
           "n2_instrument": "compute_buoyancy_frequency_nemo_bn2",
           "iwm_enabled": (not a.no_iwm), "sweeps": {}}
    try:
        out["git_sha"] = subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        out["git_sha"] = "unknown"

    for l_sfc in a.l_sfc:
        lk3, le3 = _lengths(tke, N2, dz_half, dz_cell, c3, l_sfc)
        lk4, le4 = _lengths(tke, N2, dz_half, dz_cell, c4, l_sfc)
        # ---- C1: the eddy-coefficient length is the SAME expression -------
        if not np.allclose(lk3[wet_i], lk4[wet_i], rtol=0, atol=0):
            bad = float(np.max(np.abs(lk3[wet_i] - lk4[wet_i])))
            raise SystemExit(
                f"FATAL C1: l_k differs between the two choices by {bad:.3e}. "
                "They are the same expression in tke.py; this is a probe bug, "
                "and every ratio below would be meaningless.")
        # ---- C2: sqrt(a*b) >= min(a,b) ------------------------------------
        viol = int(np.sum((le3[wet_i] < le4[wet_i] * (1 - 1e-12))))
        if viol:
            raise SystemExit(
                f"FATAL C2: l_eps(nn_mxl=3) < l_eps(nn_mxl=2) at {viol} wet "
                "interfaces, which sqrt(a*b) >= min(a,b) forbids.")
        R = np.where(wet_i, le3 / np.maximum(le4, 1e-30), np.nan)
        if not np.isfinite(R[wet_i]).all():
            raise SystemExit("FATAL: non-finite R at a wet interface — "
                             "nanmean would have hidden this.")
        # ---- C3: where the two settings are literally the same operator ---
        same = float(np.sum(w_i[wet_i] * (R[wet_i] < 1 + 1e-12))
                     / np.sum(w_i[wet_i]))
        print(f"########## l_sfc = {l_sfc:g} m   "
              f"(volume fraction where the two settings coincide: {same:.3f})")
        print(f"{'band':22s} {'depth_m':>14s} {'R_volmean':>10s} "
              f"{'R_p90':>8s} {'R_max':>8s} {'l_eps3_m':>9s} {'l_eps2_m':>9s}")
        rows = {}
        for bname, lo, hi in _BANDS:
            inb = (lat >= lo) & (lat < hi)
            rows[bname] = {}
            for k0, k1 in zip(_DEPTH_BINS[:-1], _DEPTH_BINS[1:]):
                kk = (depth_i >= k0) & (depth_i < k1)
                m = wet_i & inb[..., None] & kk[None, None, :]
                if not m.any():
                    continue
                ww = w_i[m]
                den = ww.sum()
                if den <= 0:
                    continue
                rv = float((R[m] * ww).sum() / den)
                rows[bname][f"{k0:g}-{k1:g}"] = {
                    "R_volmean": rv,
                    "R_p90": float(np.percentile(R[m], 90)),
                    "R_max": float(R[m].max()),
                    "l_eps_nnmxl3_m": float((le3[m] * ww).sum() / den),
                    "l_eps_nnmxl2_m": float((le4[m] * ww).sum() / den),
                    "n_interfaces": int(m.sum()),
                }
                d = rows[bname][f"{k0:g}-{k1:g}"]
                print(f"{bname:22s} {k0:6.0f}-{k1:<7.0f} {rv:10.3f} "
                      f"{d['R_p90']:8.3f} {d['R_max']:8.3f} "
                      f"{d['l_eps_nnmxl3_m']:9.3f} {d['l_eps_nnmxl2_m']:9.3f}")
        out["sweeps"][f"l_sfc_{l_sfc:g}"] = {"coincide_volfrac": same,
                                             "bands": rows}
        print()

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[json] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
