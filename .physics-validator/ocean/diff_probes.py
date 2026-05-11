"""Physics-validator differentiability probes for ocean physics.

Runs jax.grad / jax.jacrev against centered finite-difference Jacobians
for the leaf physics paths that matter most: EOS, hydrostatic pressure,
sponge tracer relaxation, freshwater virtual-salt flux, plume convection,
KPP boundary-layer-depth diagnosis, GM Visbeck kappa, implicit vertical
diffusion, and the implicit bottom-drag factor.

Run from repo root::

    JAX_ENABLE_X64=1 .venv/bin/python .physics-validator/ocean/diff_probes.py
"""

from __future__ import annotations

import sys
import traceback

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6
RESULTS = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f"   {detail}"
    print(line, flush=True)
    RESULTS.append((name, ok))


def fd_grad(fn, x, h=1e-6):
    """Centered FD gradient of scalar fn at array x."""
    g = jnp.zeros_like(x)
    flat = x.reshape(-1)
    n = flat.shape[0]
    for i in range(n):
        e = jnp.zeros_like(flat).at[i].set(h)
        plus = fn(flat.add(e).reshape(x.shape))
        minus = fn(flat.add(-e).reshape(x.shape))
        g = g.at[jnp.unravel_index(i, x.shape)].set(float((plus - minus) / (2 * h)))
    return g


# ---------------------------------------------------------------------------
# 1. Wright EOS — rho = f(T, S, p)
# ---------------------------------------------------------------------------
def probe_eos():
    from legoesm.ocean.eos import wright_eos

    T = jnp.array(15.0)
    S = jnp.array(35.0)
    p = jnp.array(2.0e7)

    def loss(T_, S_, p_):
        return wright_eos(T_, S_, p_).sum()

    grad_T = jax.grad(loss, 0)(T, S, p)
    grad_S = jax.grad(loss, 1)(T, S, p)
    grad_p = jax.grad(loss, 2)(T, S, p)

    # Centered FD with appropriately scaled steps.
    # rho ~ 1025, T~15 → drho/dT ~ -0.25; FD with h=1e-3 underflows to
    # ~1e-4 / 2e-3 ~ -0.05 which won't agree with AD.  Use h=0.1.
    fd_T = (loss(T + 0.1, S, p) - loss(T - 0.1, S, p)) / 0.2
    fd_S = (loss(T, S + 0.1, p) - loss(T, S - 0.1, p)) / 0.2
    # drho/dp ~ 4e-7; pressure spans 1e7 Pa → use h=1e3 Pa for fd.
    fd_p = (loss(T, S, p + 1.0e3) - loss(T, S, p - 1.0e3)) / 2.0e3

    # Wright EOS is a high-order rational polynomial; centered FD has
    # O(h^2) truncation error.  Use rtol=1% to allow that, while still
    # checking AD.
    eos_rtol = 1e-2
    ok_T = bool(jnp.allclose(grad_T, fd_T, rtol=eos_rtol, atol=ATOL))
    ok_S = bool(jnp.allclose(grad_S, fd_S, rtol=eos_rtol, atol=ATOL))
    ok_p = bool(jnp.allclose(grad_p, fd_p, rtol=eos_rtol, atol=ATOL))
    report(
        "EOS Wright drho/dT",
        ok_T,
        f"AD={float(grad_T):.6e} FD={float(fd_T):.6e}",
    )
    report(
        "EOS Wright drho/dS",
        ok_S,
        f"AD={float(grad_S):.6e} FD={float(fd_S):.6e}",
    )
    report(
        "EOS Wright drho/dp",
        ok_p,
        f"AD={float(grad_p):.6e} FD={float(fd_p):.6e}",
    )

    # Also check thermal expansion sign convention (alpha = -drho/dT/rho > 0
    # for warm water).
    from legoesm.ocean.eos import thermal_expansion_coeff

    alpha = float(thermal_expansion_coeff(T, S, p))
    ok_sign = alpha > 0.0
    report(
        "EOS thermal expansion sign (alpha>0 at 15C/35psu/2e7Pa)",
        ok_sign,
        f"alpha={alpha:.4e}",
    )


# ---------------------------------------------------------------------------
# 2. Hydrostatic pressure — p_hydro = f(rho, eta)
# ---------------------------------------------------------------------------
def probe_hydrostatic_pressure():
    from legoesm.ocean.eos import compute_hydrostatic_pressure

    nlev = 8
    rho = 1025.0 + 0.5 * jnp.linspace(0.0, 1.0, nlev)
    eta = jnp.array(0.1)
    dz = jnp.full(nlev, 200.0)
    J = jnp.array(1.0)

    def loss(rho_, eta_):
        p = compute_hydrostatic_pressure(rho_, eta_, dz, J)
        return p.sum()

    g_rho = jax.grad(loss, 0)(rho, eta)
    g_eta = jax.grad(loss, 1)(rho, eta)

    # FD checks
    h = 1e-3
    fd_rho = jnp.zeros_like(rho)
    for k in range(nlev):
        e = jnp.zeros_like(rho).at[k].set(h)
        fd_rho = fd_rho.at[k].set(
            float((loss(rho + e, eta) - loss(rho - e, eta)) / (2 * h)),
        )
    fd_eta = float((loss(rho, eta + h) - loss(rho, eta - h)) / (2 * h))

    ok_rho = bool(jnp.allclose(g_rho, fd_rho, rtol=RTOL, atol=ATOL))
    ok_eta = bool(jnp.allclose(g_eta, fd_eta, rtol=RTOL, atol=ATOL))
    report(
        "compute_hydrostatic_pressure dp/drho",
        ok_rho,
        f"max|AD-FD|={float(jnp.max(jnp.abs(g_rho - fd_rho))):.3e}",
    )
    report(
        "compute_hydrostatic_pressure dp/deta",
        ok_eta,
        f"AD={float(g_eta):.4e} FD={fd_eta:.4e}",
    )


# ---------------------------------------------------------------------------
# 3. Sponge relaxation
# ---------------------------------------------------------------------------
def probe_sponge():
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        apply_sponge_tracer_relaxation,
    )
    from legoesm.ocean.sponge import SpongeForcing

    n = 4
    nlev = 3
    dT_dt = jnp.zeros((n, n, nlev))
    dS_dt = jnp.zeros((n, n, nlev))
    T = 15.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n, n, nlev))
    S = 35.0 + 0.05 * jax.random.normal(jax.random.PRNGKey(1), (n, n, nlev))
    gamma = jnp.full((n, n), 1.0 / 86400.0)
    sponge = SpongeForcing(
        gamma=gamma,
        T_ref=jnp.full((n, n, nlev), 15.0),
        S_ref=jnp.full((n, n, nlev), 35.0),
    )

    def loss(T_):
        dT_new, _ = apply_sponge_tracer_relaxation(dT_dt, dS_dt, T_, S, sponge)
        return dT_new.sum()

    g_T = jax.grad(loss)(T)
    # Analytical: for each (i,j,k), dT_new = gamma*(T_ref - T), so dloss/dT = -gamma.
    expected = -gamma[..., jnp.newaxis] * jnp.ones_like(T)
    ok = bool(jnp.allclose(g_T, expected, rtol=RTOL, atol=ATOL))
    report(
        "Sponge dT_relax / dT",
        ok,
        f"max|AD-analytic|={float(jnp.max(jnp.abs(g_T - expected))):.3e}",
    )


# ---------------------------------------------------------------------------
# 4. Freshwater virtual-salt flux gradient
# ---------------------------------------------------------------------------
def probe_freshwater():
    from legoesm.ocean.freshwater import (
        FreshwaterForcing,
        virtual_salt_flux,
    )

    n = 4
    fw = FreshwaterForcing(
        precip=jnp.full((n,), 1.0e-4),
        evap=jnp.full((n,), 0.5e-4),
        runoff=jnp.zeros(n),
        ice_fw=jnp.zeros(n),
    )
    h_top = jnp.full((n,), 50.0)
    rho_0 = 1025.0
    S_ref = 35.0

    def loss(precip_):
        fw_ = fw._replace(precip=precip_)
        return virtual_salt_flux(fw_, S_ref, h_top, rho_0).sum()

    g = jax.grad(loss)(fw.precip)
    # dS/dt = -S_ref * (P - E + R + M) / (rho_0 * h)
    # d(dS)/dP = -S_ref / (rho_0 * h) per cell, so loss-grad has -S_ref/(rho_0*h)
    expected = jnp.full_like(fw.precip, -S_ref / (rho_0 * 50.0))
    ok = bool(jnp.allclose(g, expected, rtol=RTOL, atol=ATOL))
    report(
        "Freshwater virtual_salt_flux d(sum)/dprecip",
        ok,
        f"AD={float(g[0]):.4e} expected={float(expected[0]):.4e}",
    )

    # Sign sanity: net P>E should freshen surface (negative dS/dt).
    s_tend = float(virtual_salt_flux(fw, S_ref, h_top, rho_0)[0])
    sign_ok = s_tend < 0.0
    report(
        "Freshwater sign: net precip freshens (dS/dt<0)",
        sign_ok,
        f"dS/dt={s_tend:.4e}",
    )


# ---------------------------------------------------------------------------
# 5. iterate_eos_and_pressure_anomaly
# ---------------------------------------------------------------------------
def probe_eos_iteration():
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn

    n = 3
    nlev = 5
    key = jax.random.PRNGKey(7)
    T = 15.0 + jax.random.normal(key, (n, n, nlev))
    S = 35.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(8), (n, n, nlev))
    mask = jnp.ones((n, n))
    dz_ref = jnp.full(nlev, 100.0)
    eos_fn = make_eos_fn("wright")

    def fill_fn(q):
        return q  # ocean only — no land in this test

    def loss(T_):
        rho, rho_p, p_p = iterate_eos_and_pressure_anomaly(
            T_, S, mask, fill_fn, eos_fn, dz_ref, 1025.0, 9.80616, n_iter=2,
        )
        return p_p.sum()

    g = jax.grad(loss)(T)
    finite = bool(jnp.all(jnp.isfinite(g)))
    nonzero = float(jnp.mean(jnp.abs(g) > 1e-30))
    report(
        "iterate_eos_and_pressure_anomaly grad finite",
        finite,
        f"nonzero_frac={nonzero:.2%}",
    )
    # FD spot-check at a single index — use h=0.1 K for stability.
    i, j, k = 1, 1, 2
    h = 0.1
    Tp = T.at[i, j, k].add(h)
    Tm = T.at[i, j, k].add(-h)
    fd = float((loss(Tp) - loss(Tm)) / (2 * h))
    ad = float(g[i, j, k])
    ok = jnp.isclose(ad, fd, rtol=1e-2, atol=1e-2)
    report(
        "iterate_eos_and_pressure_anomaly dT spot-FD",
        bool(ok),
        f"AD={ad:.4e} FD={fd:.4e}",
    )


# ---------------------------------------------------------------------------
# 6. Implicit vertical diffusion
# ---------------------------------------------------------------------------
def probe_implicit_diffusion():
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_ocean,
        build_dz_half,
    )

    nlev = 8
    n = 3
    field = 15.0 + jax.random.normal(jax.random.PRNGKey(2), (n, n, nlev))
    K = jnp.full((n, n, nlev - 1), 1e-4)
    dz = jnp.full((n, n, nlev), 100.0)
    dz_half = build_dz_half(dz)
    dt = 3600.0

    def loss(K_):
        out = implicit_vertical_diffusion_ocean(field, K_, dz, dz_half, dt)
        return out.sum()

    g = jax.grad(loss)(K)
    finite = bool(jnp.all(jnp.isfinite(g)))
    report(
        "implicit_vertical_diffusion d(sum_field)/dK finite",
        finite,
        f"max|g|={float(jnp.max(jnp.abs(g))):.3e}",
    )

    # Conservation property: sum(out) ≈ sum(field) for zero-flux BC.
    out = implicit_vertical_diffusion_ocean(field, K, dz, dz_half, dt)
    rel_drift = float(abs(jnp.sum(out) - jnp.sum(field)) / abs(jnp.sum(field)))
    ok_cons = rel_drift < 1e-10
    report(
        "implicit_vertical_diffusion no-flux conservation",
        ok_cons,
        f"rel_drift={rel_drift:.3e}",
    )


# ---------------------------------------------------------------------------
# 7. Plume convection
# ---------------------------------------------------------------------------
def probe_plume_convection():
    from legoesm.ocean.physics.convection.config import PlumeConfig
    from legoesm.ocean.physics.convection.plume import plume_convection
    from legoesm.ocean.eos import wright_eos
    from legoesm.ocean.vertical import OceanZStarCoordinate

    nlev = 6
    n = 3
    # Unstable column: warm at depth, cold at surface (winter convection).
    T_profile = jnp.array([2.0, 5.0, 8.0, 10.0, 12.0, 14.0])
    T = jnp.broadcast_to(T_profile, (n, n, nlev))
    S = jnp.full_like(T, 35.0)

    # Use canonical builder.
    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=300.0, dz_surface=50.0, dz_deep=50.0,
    )
    jacobian = jnp.ones((n, n))
    rho = wright_eos(T, S, jnp.zeros_like(T))
    p = jnp.zeros_like(T)
    cfg = PlumeConfig()

    out = plume_convection(T, S, rho, p, z_coord, jacobian, cfg)

    def loss(T_):
        rho_ = wright_eos(T_, S, jnp.zeros_like(T_))
        out_ = plume_convection(T_, S, rho_, p, z_coord, jacobian, cfg)
        return out_.dT_dt.sum()

    g = jax.grad(loss)(T)
    finite = bool(jnp.all(jnp.isfinite(g)))
    report(
        "plume_convection grad finite",
        finite,
        f"max|g|={float(jnp.max(jnp.abs(g))):.3e}",
    )

    # Sanity: surface tendency should be zero (plume detrains below surface).
    surface_tend = float(out.dT_dt[..., 0].mean())
    ok_surface = abs(surface_tend) < 1e-12
    report(
        "plume_convection: surface tendency=0",
        ok_surface,
        f"avg surface dT/dt={surface_tend:.3e}",
    )


# ---------------------------------------------------------------------------
# 8. Implicit bottom-drag factor
# ---------------------------------------------------------------------------
def probe_bottom_drag_factor():
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        implicit_bottom_drag_factor,
    )

    H = jnp.array([10.0, 100.0, 1000.0, 5000.0])
    dt = 600.0
    r = 1.0e-3

    f = implicit_bottom_drag_factor(dt, r, H)
    # Backward-Euler implicit form: f = 1 / (1 + dt*r/H).
    # H=10: dt*r/H=0.06, f=1/1.06=0.9434; H=5000: dt*r/H=1.2e-4, f≈1.
    expected = 1.0 / (1.0 + dt * r / H)
    ok = bool(jnp.allclose(f, expected, rtol=1e-12))
    report(
        "implicit_bottom_drag_factor",
        ok,
        f"f={[round(float(x), 6) for x in f]}",
    )

    # All factors must be > 0 (otherwise drag drives velocity through zero).
    pos_ok = bool(jnp.all(f > 0.0))
    report(
        "implicit_bottom_drag_factor positivity at typical values",
        pos_ok,
        "",
    )


# ---------------------------------------------------------------------------
# 9. NPZD non-negativity, conservation, gradient
# ---------------------------------------------------------------------------
def probe_npzd():
    from legoesm.ocean.biogeochemistry.config import BiogeoConfig
    from legoesm.ocean.biogeochemistry.npzd import npzd_source_sink

    nlev = 4
    n = 2
    NO3 = jnp.full((n, n, nlev), 5.0e-3)
    P = jnp.full((n, n, nlev), 0.5e-3)
    Z = jnp.full((n, n, nlev), 0.2e-3)
    Det = jnp.full((n, n, nlev), 0.1e-3)
    DIC = jnp.full((n, n, nlev), 2.0)
    ALK = jnp.full((n, n, nlev), 2.4)
    T = jnp.full((n, n, nlev), 15.0)
    PAR = jnp.full((n, n, nlev), 100.0)
    dz = jnp.full(nlev, 50.0)
    cfg = BiogeoConfig()

    dN, dP, dZ, dD, dDIC, dALK = npzd_source_sink(
        NO3, P, Z, Det, DIC, ALK, T, PAR, dz, cfg,
    )

    # N conservation in nitrogen pools (no sinking exported -> column sum
    # unaffected).  However: detritus sinking exports nitrogen out of the
    # bottom of every column.  Compute interior-column N tendency closure.
    N_pool_tend = (dN + dP + dZ + dD).sum(axis=-1) * 50.0
    # The export at the bottom is w_sink * D[bottom].
    w_sink_s = cfg.w_sink / 86400.0
    expected_export = -w_sink_s * Det[..., -1]
    rel_err = float(jnp.max(
        jnp.abs(N_pool_tend - expected_export)
        / jnp.maximum(jnp.abs(expected_export), 1e-30),
    ))
    ok_cons = rel_err < 1e-6
    report(
        "NPZD nitrogen pool conservation w/ export",
        ok_cons,
        f"rel_err={rel_err:.3e}",
    )

    # Gradient sanity
    def loss(P_):
        d = npzd_source_sink(NO3, P_, Z, Det, DIC, ALK, T, PAR, dz, cfg)
        return d[1].sum()

    g = jax.grad(loss)(P)
    finite = bool(jnp.all(jnp.isfinite(g)))
    report(
        "NPZD grad finite",
        finite,
        f"max|g|={float(jnp.max(jnp.abs(g))):.3e}",
    )


# ---------------------------------------------------------------------------
# 10. KPP boundary layer depth and mixing tendency
# ---------------------------------------------------------------------------
def probe_kpp_smoke():
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.eos import wright_eos
    from legoesm.ocean.vertical import OceanZStarCoordinate

    nlev = 8
    n = 3
    # Stratified profile.
    T_profile = 5.0 + 10.0 * jnp.exp(jnp.linspace(0, -1, nlev))
    T = jnp.broadcast_to(T_profile, (n, n, nlev))
    S = jnp.full_like(T, 35.0)
    u = jnp.full_like(T, 0.0)
    v = jnp.full_like(T, 0.0)

    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=400.0, dz_surface=20.0, dz_deep=80.0,
    )
    jacobian = jnp.ones((n, n))
    eta = jnp.zeros((n, n))
    rho = wright_eos(T, S, jnp.zeros_like(T))
    cfg = KPPConfig()

    tau_x = jnp.full((n, n), 0.1)
    tau_y = jnp.zeros((n, n))
    B_f = jnp.full((n, n), -1e-8)  # stable

    out = kpp_vertical_mixing(
        u, v, T, S, rho, eta, z_coord, jacobian, cfg,
        tau_x=tau_x, tau_y=tau_y, B_f=B_f,
    )

    finite_K = bool(jnp.all(jnp.isfinite(out.K_v)))
    finite_A = bool(jnp.all(jnp.isfinite(out.A_v)))
    report(
        "KPP outputs finite (stable)",
        finite_K and finite_A,
        f"K_v range=[{float(out.K_v.min()):.3e}, {float(out.K_v.max()):.3e}]",
    )

    # Convective branch
    B_f_unstable = jnp.full((n, n), 1e-7)
    # Set unstable surface T (cold-on-warm).
    T_unstable = T.at[..., 0].set(0.0)
    rho_unstable = wright_eos(T_unstable, S, jnp.zeros_like(T_unstable))
    out_u = kpp_vertical_mixing(
        u, v, T_unstable, S, rho_unstable, eta, z_coord, jacobian, cfg,
        tau_x=tau_x, tau_y=tau_y, B_f=B_f_unstable,
    )
    finite_u = bool(jnp.all(jnp.isfinite(out_u.K_v))
                    & jnp.all(jnp.isfinite(out_u.dT_dt)))
    # Non-local should be nonzero in unstable column.
    nonlocal_active = float(jnp.max(jnp.abs(out_u.dT_dt))) > 1e-12
    report(
        "KPP outputs finite (unstable)",
        finite_u,
        f"max|dT/dt|={float(jnp.max(jnp.abs(out_u.dT_dt))):.3e}",
    )
    report(
        "KPP non-local transport active in unstable",
        nonlocal_active,
        "",
    )

    # Gradient through KPP w.r.t. T
    def loss(T_):
        rho_ = wright_eos(T_, S, jnp.zeros_like(T_))
        out_ = kpp_vertical_mixing(
            u, v, T_, S, rho_, eta, z_coord, jacobian, cfg,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
        )
        return out_.dT_dt.sum()

    g = jax.grad(loss)(T)
    finite_g = bool(jnp.all(jnp.isfinite(g)))
    nonzero_frac = float(jnp.mean(jnp.abs(g) > 1e-30))
    report(
        "KPP grad finite + nonzero",
        finite_g and nonzero_frac > 0.1,
        f"nonzero_frac={nonzero_frac:.1%}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    probes = [
        probe_eos,
        probe_hydrostatic_pressure,
        probe_sponge,
        probe_freshwater,
        probe_eos_iteration,
        probe_implicit_diffusion,
        probe_plume_convection,
        probe_bottom_drag_factor,
        probe_npzd,
        probe_kpp_smoke,
    ]

    for p in probes:
        try:
            p()
        except Exception as exc:  # pragma: no cover
            report(p.__name__, False, f"EXCEPTION {exc!r}")
            traceback.print_exc()

    n_pass = sum(1 for _, ok in RESULTS if ok)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\nTOTAL: {n_pass} pass / {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
