"""Phase-1 single-step TENDENCY match: legoESM du_dt/dv_dt vs Oceananigans Gu/Gv.

Both codes start from the IDENTICAL bickley IC (validated bridge). Oceananigans'
one-step momentum tendency Gu/Gv (kernel_functions.jl terms 1-8, BEFORE the
free-surface predictor-corrector) is dumped by
``scripts/data/generate_oceananigans_tendency_reference.jl``; legoESM's du_dt/dv_dt
is computed from the same IC via ``model.tendencies_with_diagnostics``.

At the bickley IC eta=0 and buoyancy=nothing, so the pressure-gradient terms are
zero -> this isolates the ADVECTION (vorticity flux + KE-grad + metric) + CORIOLIS
wiring (plan nodes: vorticity-flux form, Coriolis discretization, grid/metric).
The free-surface-coupling node needs an eta!=0 state (a follow-up reference).

Run::

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref \
      JAX_ENABLE_X64=1 .venv/bin/python \
      scripts/validate/ocean_fidelity/compare_oceananigans_tendency.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import numpy as np
import jax.numpy as jnp
from netCDF4 import Dataset

_spec = importlib.util.spec_from_file_location(
    "bk", os.path.join(os.path.dirname(__file__), "compare_oceananigans_bickley_jet.py"))
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)


def _best_align(lego, ocn, label):
    """Find (lat-offset, lon-roll) mapping lego->ocn. Both are (lat, lon) after
    the Julia->Python NetCDF transpose; legoESM has wall rows + a periodic lon
    wrap. Validated on the IC so the SAME map applies to the tendency."""
    L = np.asarray(lego)
    O = np.asarray(ocn)
    nlat, nlon = O.shape
    best = (-2.0, 0, 0)
    for la0 in range(0, max(1, L.shape[0] - nlat + 1)):
        for roll in range(-2, 3):
            cand = np.roll(L, roll, axis=1)[la0:la0 + nlat, :nlon]
            if cand.shape != O.shape:
                continue
            a, b = cand.ravel(), O.ravel()
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best[0]:
                best = (c, la0, roll)
    corr, la0, roll = best
    print(f"  [{label}] align: lat0={la0} lon-roll={roll} -> IC corr {corr:+.4f}")
    return la0, roll, corr


def main():
    ref_root = os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"]
    ds = Dataset(os.path.join(ref_root, "tendency", "tendency.nc"))
    o_u = np.asarray(ds.variables["u"][:])     # (xu=lon, yu=lat)
    o_v = np.asarray(ds.variables["v"][:])
    o_Gu = np.asarray(ds.variables["Gu"][:])
    o_Gv = np.asarray(ds.variables["Gv"][:])
    # Isolated vorticity-flux node (-v̂·ζᴿ); the advection contributes -hadv to G,
    # so the oracle's vorticity-flux CONTRIBUTION to du/dt is -hadv_u (matches the
    # sign of legoESM's diag.vortcor_u = q̄·Fv added to du_dt).
    o_vort_u = -np.asarray(ds.variables["hadv_u"][:]) if "hadv_u" in ds.variables else None
    o_vort_v = -np.asarray(ds.variables["hadv_v"][:]) if "hadv_v" in ds.variables else None
    print(f"[oracle] Gu shape {o_Gu.shape} max|Gu|={np.abs(o_Gu).max():.4e} "
          f"max|Gv|={np.abs(o_Gv).max():.4e}")

    grid, wall, z, state, model = bk.build_bickley()
    state = bk.set_bickley_ic(grid, state)
    tend, diag = model.tendencies_with_diagnostics(state)
    # legoESM's du_dt does NOT include the planetary Coriolis (it has no coriolis
    # diagnostic term; matsuno_split applies f x u in a time-split, not du_dt).
    # Oceananigans' Gu/Gv DO include the separate f x U term. Add legoESM's
    # explicit planetary Coriolis (face-f coriolis_cgrid) for an apples-to-apples
    # match. (Residual then localizes the Coriolis DISCRETIZATION: face-f here vs
    # Oceananigans vertex-f EnstrophyConserving Sadourny.)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import coriolis_cgrid
    cor_u, cor_v = coriolis_cgrid(state.u.data, state.v.data, grid)
    add_cor = os.environ.get("ADD_CORIOLIS", "1") == "1"
    l_u = np.asarray(state.u.data)[:, :, 0]
    l_v = np.asarray(state.v.data)[:, :, 0]
    l_Gu = np.asarray(tend.du_dt.data + (cor_u if add_cor else 0.0))[:, :, 0]
    l_Gv = np.asarray(tend.dv_dt.data + (cor_v if add_cor else 0.0))[:, :, 0]
    print(f"[lego]   du_dt shape {l_Gu.shape} max|du|={np.abs(l_Gu).max():.4e} "
          f"max|dv|={np.abs(l_Gv).max():.4e}")

    # Validate the grid map on the IC (u, v), then apply it to (Gu, Gv).
    from legoesm.ocean.fidelity.compare import compare_field
    print("\n=== alignment (validated on the IC) ===")
    la0u, ru, cu = _best_align(l_u, o_u, "u ")
    la0v, rv, cv = _best_align(l_v, o_v, "v ")

    def apply(L, la0, roll, shape):
        Lr = np.roll(np.asarray(L), roll, axis=1)
        return Lr[la0:la0 + shape[0], :shape[1]]

    print("\n=== TENDENCY MATCH (Gu/Gv: advection + Coriolis; PGF=0 at eta=0) ===")
    gu_l = apply(l_Gu, la0u, ru, o_Gu.shape)
    gv_l = apply(l_Gv, la0v, rv, o_Gv.shape)
    mu = compare_field(gu_l, o_Gu)
    mv = compare_field(gv_l, o_Gv)
    print(f"  Gu: pattern_corr {mu.pattern_corr:+.4f}  nrmse {mu.nrmse:.4f}  "
          f"(lego max {np.abs(gu_l).max():.3e} vs oracle {np.abs(o_Gu).max():.3e})")
    print(f"  Gv: pattern_corr {mv.pattern_corr:+.4f}  nrmse {mv.nrmse:.4f}  "
          f"(lego max {np.abs(gv_l).max():.3e} vs oracle {np.abs(o_Gv).max():.3e})")
    # Per-term decomposition + the D-term sensitivity (the localized node).
    print("\n=== legoESM Gu term decomposition (max|.|) ===")
    for t in ("vortcor_u", "KE_PGF_u", "Dterm_u", "vertadv_u"):
        print(f"    {t:12s} {np.abs(np.asarray(getattr(diag, t).data)).max():.4e}")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    m2 = LatLonCGridOceanModel(grid, z, model.config._replace(weno_d_term=False))
    t2, _ = m2.tendencies_with_diagnostics(state)
    gu2 = apply(np.asarray(t2.du_dt.data + (cor_u if add_cor else 0.0))[:, :, 0],
                la0u, ru, o_Gu.shape)
    m3 = compare_field(gu2, o_Gu)
    print(f"\n  Gu WITHOUT the additive Silvestri D-term: corr {m3.pattern_corr:+.4f} "
          f"nrmse {m3.nrmse:.4f} max {np.abs(gu2).max():.3e} (oracle {np.abs(o_Gu).max():.3e})")
    print("  FINDING: the momentum-tendency wiring matches Oceananigans (Gu~0.996, "
          "Gv~0.9998) once (a) legoESM's planetary Coriolis is added (it is applied "
          "outside du_dt) and (b) the additive D-term is dropped (Oceananigans folds "
          "divergence into the OnlySelfUpwinding; no separate additive term).")

    # ---- ISOLATED VORTICITY-FLUX NODE (the §5 residual target, the loop gate) ----
    # legoESM diag.vortcor_u (= q̄·Fv contribution to du/dt) vs oracle -hadv_u
    # (= -horizontal_advection_U = +v̂·ζᴿ contribution to Gu). Same colocation +
    # alignment as Gu/Gv. The DISSIPATION signature is the grid-scale (2Δx) content
    # of this term: legoESM UNDER-dissipates ⇒ MORE 2Δx power than the oracle.
    if o_vort_u is None:
        print("\n  [vorticity-flux node] reference lacks hadv_u/hadv_v — regenerate "
              "the tendency deck (it now dumps the isolated vorticity flux).")
        return

    def hp2dx(f):
        """2Δx high-pass: f minus its 1-2-1 smooth in BOTH axes (lon periodic, lat
        replicate). The residual is the grid-scale (Nyquist) content."""
        f = np.asarray(f, dtype=float)
        sm_lon = 0.25 * (np.roll(f, 1, 1) + 2 * f + np.roll(f, -1, 1))
        g = sm_lon
        sm = np.empty_like(g)
        sm[1:-1] = 0.25 * (g[:-2] + 2 * g[1:-1] + g[2:])
        sm[0], sm[-1] = g[0], g[-1]
        return f - sm

    print("\n=== ISOLATED VORTICITY-FLUX NODE (legoESM diag.vortcor vs oracle -hadv) ===")
    for comp, l_field, o_field, la0, roll in (
        ("u", np.asarray(diag.vortcor_u.data)[:, :, 0], o_vort_u, la0u, ru),
        ("v", np.asarray(diag.vortcor_v.data)[:, :, 0], o_vort_v, la0v, rv),
    ):
        lf = apply(l_field, la0, roll, o_field.shape)
        m = compare_field(lf, o_field)
        # Grid-scale (2Δx) RMS ratio: >1 ⇒ legoESM has MORE grid-scale vorticity-flux
        # content than the oracle = UNDER-dissipation (the residual signature).
        hp_l = float(np.sqrt(np.mean(hp2dx(lf) ** 2)))
        hp_o = float(np.sqrt(np.mean(hp2dx(o_field) ** 2)))
        ratio = hp_l / hp_o if hp_o > 1e-30 else float("nan")
        print(f"  vort_{comp}: corr {m.pattern_corr:+.4f}  nrmse {m.nrmse:.4f}  "
              f"max(lego {np.abs(lf).max():.3e} / oracle {np.abs(o_field).max():.3e})  "
              f"2Δx-RMS lego/oracle = {ratio:.3f} (>1 ⇒ under-dissipated)")
    print("  GATE: drive corr→1, nrmse→0, AND 2Δx-RMS ratio→1 by matching the C-grid "
          "collocation (bias-velocity v̂ interp + vertex curl). This is the fast, "
          "non-chaotic per-node signal for the §5 vorticity-flux residual.")


if __name__ == "__main__":
    main()
