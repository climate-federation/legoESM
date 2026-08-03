"""Differentiability sweep for the atmosphere turbulence / gravity-wave-drag /
radiation schemes NOT covered by ``tests/unit/test_diff_atmosphere_physics.py``.

That file's category 2d only parametrizes turbulence over a handful of schemes
w.r.t. ``T``, and its category 2b covers ``gray`` radiation.  ``gravity_wave_drag``
had **ZERO** differentiability coverage in the diff sweep — this file closes that
gap first, then adds the uncovered turbulence schemes / leaf helpers and the
uncovered radiation modules (solar geometry, ozone, RRTMGP).

Coverage added here
-------------------
gravity_wave_drag  (NEW — no prior diff-sweep coverage of ANY kind)
  rayleigh, lindzen, mcfarlane, hines, prognostic_spectral, e3sm_cam
  (orographic / frontal / convective sources), ml_emulator, ``+``-composites,
  oro_source.depth_averaged_oro_source, frontogenesis.compute_frontogenesis,
  and the ``make_gwd_physics`` hydrostatic bridge.
turbulence
  tke, mynn25, clubb_lite, clubb, edmf, ysu, holtslag_boville, plus the leaf
  helpers vertical_diffusion, surface_layer, pbl_height, amd, vreman, lasd_core.
radiation
  solar (zenith / insolation / declination / orbit), ozone_mls, ozone_ml,
  rrtmgp_radiation.

Assertion standard
------------------
Every test asserts **finite AND non-zero AND structured** — "is finite" alone is
not acceptable (this sweep found pre-existing tests that passed while measuring
an identically-zero quantity).  Where a gradient is *genuinely* zero because the
input only enters a step function or a lookup index, the zero is asserted
EXACTLY and documented with its measured number — a result, not a hidden failure.

Vertical index convention (load-bearing)
----------------------------------------
Every column leaf in this package reads the SURFACE at index ``-1``
(``u_sfc = u[:, -1]``, ``p_s = p_half[:, -1]``), so **pressure must INCREASE with
index** and height must DECREASE with index.  A previous gray-radiation fixture
inverted this and silently clamped the whole column optical depth to zero,
making every gradient exactly 0.0 while the test still "passed".  All fixtures
below build the column from the model's own ``create_sigma_coordinate`` +
``compute_heights_from_sigma`` + ``compute_rho`` helpers so the convention (and
the hydrostatic consistency) cannot drift.

Run with ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402


# ===========================================================================
# Shared assertion helpers
# ===========================================================================

def nonzero_frac(grad) -> float:
    """Fraction of gradient entries that are exactly non-zero."""
    return float(jnp.mean(jnp.abs(jnp.asarray(grad)) > 0.0))


def assert_finite(grad, name=""):
    g = jnp.asarray(grad)
    assert jnp.all(jnp.isfinite(g)), (
        f"{name}: gradient has NaN/Inf "
        f"({int(jnp.sum(~jnp.isfinite(g)))}/{g.size} entries)"
    )


def assert_gradient_ok(grad, name="", min_nonzero_frac=0.1,
                       require_structure=True):
    """Finite + non-zero + structured.

    ``require_structure`` guards the failure mode where a scheme collapses to a
    single global scalar path: the gradient is then non-zero everywhere but
    IDENTICAL everywhere, which a plain non-zero check cannot distinguish from a
    healthy spatially-varying sensitivity.
    """
    g = jnp.asarray(grad)
    assert_finite(g, name)
    frac = nonzero_frac(g)
    assert frac >= min_nonzero_frac, (
        f"{name}: only {frac * 100:.1f}% of gradient entries are non-zero "
        f"(need >= {min_nonzero_frac * 100:.0f}%); max|g| = "
        f"{float(jnp.max(jnp.abs(g))):.3e}"
    )
    if require_structure:
        gmax = float(jnp.max(jnp.abs(g)))
        spread = float(jnp.std(g))
        assert spread > 1.0e-12 * max(gmax, 1.0e-300), (
            f"{name}: gradient is spatially UNIFORM (std={spread:.3e}, "
            f"max|g|={gmax:.3e}) — a constant gradient means the scheme "
            "collapsed to one global scalar path."
        )


def assert_exactly_zero(grad, name="", reason=""):
    """Assert a MEASURED identically-zero gradient (a documented result)."""
    g = jnp.asarray(grad)
    assert_finite(g, name)
    gmax = float(jnp.max(jnp.abs(g)))
    assert gmax == 0.0, (
        f"{name}: expected an identically-zero gradient ({reason}) but measured "
        f"max|g| = {gmax:.3e}. The scheme gained a dependence on this input — "
        "update the documented expectation."
    )


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    """Drop compiled executables after every test.

    JAX aborts (rc=134) after roughly three dozen accumulated model-graph
    compilations in one process — not OOM.  Each scheme here traces a fresh
    graph, so clear between cases.
    """
    yield
    jax.clear_caches()


# ===========================================================================
# Shared column fixtures
# ===========================================================================

def _gwd_column(ncol=8, nlev=20, seed=0):
    """Stably stratified column in which an orographic wave actually BREAKS.

    Index 0 = model top, index -1 = surface (pressure increases with index).
    Heights and density come from the model's own ``_shared`` helpers so the
    hydrostatic column the GWD backends see is the one the production bridge
    builds.

    Profile choice is load-bearing, not cosmetic
    --------------------------------------------
    The orographic closures cap the propagating stress at
    ``tau_sat = eff * fcrit2 * rho * |U_proj|^3 * k / N`` and only deposit
    momentum where the launch stress ``tau_0`` OVERTAKES ``tau_sat``.  A first
    version of this fixture used a strong upward-increasing jet (5 -> 40 m/s);
    there ``rho * U^3`` still grows with height, ``tau_sat`` never drops below
    ``tau_0``, and BOTH orographic schemes returned ``du_dt`` identically zero
    (measured: McFarlane max|d(loss)/du| = 0.0, Lindzen 1.3% non-zero).  That
    is CORRECT physics — a wave that never saturates deposits nothing — but it
    makes the test vacuous.

    A weakly sheared profile (10 -> 14 m/s) lets the ~100x density drop
    dominate, so ``tau_sat`` falls below ``tau_0`` in the upper troposphere and
    the wave breaks over the top of the column, which is the branch under test.
    Temperature is 230 K (top) -> 290 K (surface): a ~6.5 K/km tropospheric
    lapse rate, comfortably stable (N ~ 0.0105 s^-1), rather than the ~9.7 K/km
    near-adiabatic profile the first version produced (N^2 ~ 0, which makes
    every N-dependent term numerically degenerate).
    """
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.physics._shared import (
        compute_heights_from_sigma,
        compute_rho,
    )

    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
    sigma_full = jnp.asarray(sigma.sigma_full, dtype=jnp.float64)     # (nlev,)
    sigma_half = jnp.asarray(sigma.sigma_half, dtype=jnp.float64)     # (nlev+1,)
    assert float(sigma_half[0]) < float(sigma_half[-1]), (
        "fixture convention broken: sigma must increase with index (TOA first)"
    )

    # Smooth per-column multipliers give the gradient spatial structure without
    # per-level noise (which at 0.4 km layer spacing can locally flip the sign
    # of N^2 and silently switch a scheme into its stable-cutoff branch).
    span = jnp.linspace(-1.0, 1.0, ncol)
    p_s = 1.0e5 * (1.0 + 0.01 * span)                                  # (ncol,)
    p_half = p_s[:, None] * sigma_half[None, :]                        # (ncol,nlev+1)
    p_full = p_s[:, None] * sigma_full[None, :]                        # (ncol,nlev)

    T = ((230.0 + 60.0 * sigma_full)[None, :]
         * (1.0 + 0.02 * span)[:, None])
    u = ((10.0 + 4.0 * (1.0 - sigma_full))[None, :]
         * (1.0 + 0.08 * span)[:, None])
    v = 3.0 * (1.0 - 0.05 * span)[:, None] * jnp.ones((1, nlev))
    q_v = (1.2e-2 * sigma_full[None, :] ** 3) * jnp.ones((ncol, 1))

    z_full, z_half = compute_heights_from_sigma(T, p_half, q_v=q_v)
    rho = compute_rho(T, p_full, q_v=q_v)
    lat = jnp.linspace(-jnp.pi / 3.0, jnp.pi / 3.0, ncol)

    return dict(u=u, v=v, T=T, q_v=q_v, p_full=p_full, p_half=p_half,
                z_full=z_full, z_half=z_half, rho=rho, lat=lat,
                sigma_full=sigma_full, p_s=p_s)


def _gwd_args(col, **override):
    """Positional GWD backend signature: (u,v,T,p_full,p_half,z_full,z_half,rho,lat,dt)."""
    c = dict(col)
    c.update(override)
    return (c["u"], c["v"], c["T"], c["p_full"], c["p_half"],
            c["z_full"], c["z_half"], c["rho"], c["lat"])


# ===========================================================================
# A.  Gravity-wave drag — column leaves  (NEW COVERAGE)
# ===========================================================================

# Backends sharing the plain (u,v,T,...,config) signature.
GWD_PLAIN = ["rayleigh", "hines"]
GWD_OROGRAPHIC = ["lindzen", "mcfarlane"]
GWD_DIAGNOSTIC = GWD_PLAIN + GWD_OROGRAPHIC


def _gwd_leaf(scheme):
    """Return (fn, config) for a single-scheme GWD backend."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import get_gwd_fn

    _name, fn, cfg = get_gwd_fn(GravityWaveDragConfig(scheme=scheme))
    return fn, cfg


class TestGWDLeafGrad:
    """Gradients through each GWD column backend.

    Before this file the GWD package had no entry in the differentiability
    sweep at all: a dead ``du_dt`` path (a saturation gate that latched off, a
    critical-level filter that detached) would have been invisible to it.
    """

    @pytest.mark.parametrize("scheme", GWD_DIAGNOSTIC)
    def test_grad_du_dt_wrt_u(self, scheme):
        """d(sum du_dt^2)/du — the primary momentum sensitivity."""
        fn, cfg = _gwd_leaf(scheme)
        col = _gwd_column()

        def loss(u):
            out = fn(*_gwd_args(col, u=u), 300.0, cfg)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        # rayleigh: drag is active only in the sigma>sigma_b boundary layer and
        # the sponge, so a minority of levels carry sensitivity by design.
        min_frac = 0.05 if scheme == "rayleigh" else 0.10
        assert_gradient_ok(grad, f"GWD({scheme}) du_dt w.r.t. u",
                           min_nonzero_frac=min_frac)

    @pytest.mark.parametrize("scheme", GWD_DIAGNOSTIC)
    def test_grad_dv_dt_wrt_v(self, scheme):
        fn, cfg = _gwd_leaf(scheme)
        col = _gwd_column()

        def loss(v):
            out = fn(*_gwd_args(col, v=v), 300.0, cfg)
            return jnp.sum(out.dv_dt ** 2)

        grad = jax.grad(loss)(col["v"])
        min_frac = 0.05 if scheme == "rayleigh" else 0.10
        assert_gradient_ok(grad, f"GWD({scheme}) dv_dt w.r.t. v",
                           min_nonzero_frac=min_frac)

    @pytest.mark.parametrize("scheme", ["lindzen", "mcfarlane", "hines"])
    def test_grad_du_dt_wrt_T(self, scheme):
        """d(sum du_dt^2)/dT — the buoyancy (Brunt-Vaisala) path.

        Every saturation closure here scales the breaking stress with
        ``N = sqrt(g/theta dtheta/dz)``, so a dead dT path would mean the
        scheme's wave breaking no longer responds to stratification at all
        (i.e. the launch/saturation profile is frozen).
        """
        fn, cfg = _gwd_leaf(scheme)
        col = _gwd_column()

        def loss(T):
            out = fn(*_gwd_args(col, T=T), 300.0, cfg)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad, f"GWD({scheme}) du_dt w.r.t. T",
                           min_nonzero_frac=0.05)

    def test_rayleigh_grad_wrt_T_is_exactly_zero_by_construction(self):
        """MEASURED ZERO (not a bug).

        ``rayleigh_gwd`` builds its drag coefficient purely from
        ``sigma = p_full / p_half[:, -1]`` and the config rates — temperature
        never enters.  Rayleigh friction is buoyancy-independent by definition,
        so d(du_dt)/dT is identically 0.  Recorded here so the zero is a
        documented, asserted property rather than a silent hole: if the scheme
        ever gains an N-dependence this test goes red and the expectation is
        revisited deliberately.
        """
        fn, cfg = _gwd_leaf("rayleigh")
        col = _gwd_column()

        def loss(T):
            out = fn(*_gwd_args(col, T=T), 300.0, cfg)
            return jnp.sum(out.du_dt ** 2) + jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_exactly_zero(
            grad, "GWD(rayleigh) w.r.t. T",
            reason="rayleigh_gwd reads only sigma = p_full/p_s, never T",
        )

    @pytest.mark.parametrize("scheme", GWD_DIAGNOSTIC)
    def test_grad_dT_dt_wrt_u(self, scheme):
        """Wave-breaking heating must respond to the wind that drives it.

        ``dT_dt = -(u du_dt + v dv_dt)/c_pd`` is the KE->heat closure; a dead
        gradient here means the thermal side of the drag is detached from the
        momentum side and column energy would not close under perturbation.
        """
        fn, cfg = _gwd_leaf(scheme)
        col = _gwd_column()

        def loss(u):
            out = fn(*_gwd_args(col, u=u), 300.0, cfg)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        min_frac = 0.05 if scheme == "rayleigh" else 0.10
        assert_gradient_ok(grad, f"GWD({scheme}) dT_dt w.r.t. u",
                           min_nonzero_frac=min_frac)

    @pytest.mark.parametrize("scheme", GWD_DIAGNOSTIC)
    def test_grad_eps_gwd_wrt_u(self, scheme):
        """Column KE dissipation diagnostic is differentiable (energy-budget
        losses in parameter estimation reduce to exactly this quantity)."""
        fn, cfg = _gwd_leaf(scheme)
        col = _gwd_column()

        def loss(u):
            out = fn(*_gwd_args(col, u=u), 300.0, cfg)
            return jnp.sum(out.eps_gwd ** 2)

        grad = jax.grad(loss)(col["u"])
        min_frac = 0.05 if scheme == "rayleigh" else 0.10
        assert_gradient_ok(grad, f"GWD({scheme}) eps_gwd w.r.t. u",
                           min_nonzero_frac=min_frac)

    @pytest.mark.parametrize("scheme", GWD_OROGRAPHIC)
    def test_grad_wrt_h_topo_col(self, scheme):
        """Per-column subgrid orography drives the launch stress tau_0 ~ h^2.

        This is the single most important *boundary* input for orographic GWD
        (and the natural target for calibrating drag against a reanalysis), so
        a detached h_topo would make the scheme untrainable in the one knob
        that varies spatially.
        """
        fn, cfg = _gwd_leaf(scheme)
        col = _gwd_column()
        ncol = col["u"].shape[0]
        # Two constraints on this range, both measured:
        #  * ABOVE ~0.2 Pa-equivalent launch stress, else the column-top
        #    saturation stress is never overtaken and the column is silently
        #    non-breaking (a legitimately-zero gradient entry).
        #  * BELOW the McFarlane Froude cap.  ``_mcfarlane_launch_stress`` uses
        #    ``h_eff^2 = min(h_disp^2, fcrit2*(U/N)^2)``; with U ~ 10.4 m/s and
        #    N ~ 0.0105 s^-1 the cap sits at ``h_disp^2 = 9.9e5`` m^2, i.e.
        #    ``h ~ 497 m`` once the E3SM ``h_disp = 2*sgh`` doubling activates
        #    (which it does as soon as a per-column ``h_topo_col`` is wired).
        #    Above the cap ``tau_0`` is h-INDEPENDENT — see
        #    ``test_mcfarlane_froude_capped_h_topo_grad_is_exactly_zero``.
        h_topo = jnp.linspace(250.0, 450.0, ncol)

        def loss(h):
            out = fn(*_gwd_args(col), 300.0, cfg, h_topo_col=h)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(h_topo)
        assert_gradient_ok(grad, f"GWD({scheme}) du_dt w.r.t. h_topo_col",
                           min_nonzero_frac=0.5)

    def test_mcfarlane_froude_capped_h_topo_grad_is_exactly_zero(self):
        """MEASURED ZERO — the Froude cap, working as specified.

        ``tau_0 = G_0 rho N k min(h_disp^2, fcrit2 (U/N)^2) U``.  Once the
        streamline displacement exceeds the marginal-instability limit the
        ``min`` selects the Froude branch, ``tau_0`` no longer contains ``h``
        at all, and ``d(tau_0)/dh`` is identically 0: a taller mountain
        launches NO extra stress.

        This is physically correct (McFarlane 1987 / E3SM gw_oro.F90:166) but
        it has a concrete consequence for calibration that is worth having
        pinned: over genuinely mountainous columns the orographic amplitude
        parameter is UNIDENTIFIABLE from the drag — the gradient a tuner would
        follow is exactly zero, not merely small.  An earlier version of the
        test above spanned 400-900 m and measured 25% non-zero for exactly this
        reason (the 6 columns above 497 m were all Froude-limited).
        """
        fn, cfg = _gwd_leaf("mcfarlane")
        col = _gwd_column()
        ncol = col["u"].shape[0]
        # Well above the ~497 m cap for this column's U/N.
        h_topo = jnp.linspace(1500.0, 2500.0, ncol)

        def loss(h):
            out = fn(*_gwd_args(col), 300.0, cfg, h_topo_col=h)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(h_topo)
        assert_exactly_zero(
            grad, "GWD(mcfarlane) du_dt w.r.t. h_topo_col (Froude-capped)",
            reason="h_eff^2 = min(h_disp^2, fcrit2*(U/N)^2) selects the "
                   "Froude branch, which contains no h",
        )


class TestGWDPrognosticSpectralGrad:
    """``prognostic_spectral`` carries a wave-action spectrum through a
    ``lax.scan`` over levels — the AD hotspot the other GWD schemes lack."""

    def _run(self, col, cfg, spectrum_in, u=None, T=None):
        from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
            prognostic_spectral_gwd,
        )
        over = {}
        if u is not None:
            over["u"] = u
        if T is not None:
            over["T"] = T
        return prognostic_spectral_gwd(
            *_gwd_args(col, **over), 300.0, cfg, spectrum_in,
        )

    def _setup(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            PrognosticSpectralConfig,
        )
        col = _gwd_column()
        # 6 wavenumbers instead of the 20 default keeps the scan graph small.
        cfg = PrognosticSpectralConfig(n_azimuths=4, n_wavenumbers=6)
        ncol = col["u"].shape[0]
        spec = jnp.full((ncol, cfg.n_azimuths, cfg.n_wavenumbers),
                        cfg.launch_flux)
        return col, cfg, spec

    def test_grad_du_dt_wrt_u(self):
        col, cfg, spec = self._setup()

        def loss(u):
            out, _spec_new = self._run(col, cfg, spec, u=u)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "GWD(prognostic_spectral) du_dt w.r.t. u",
                           min_nonzero_frac=0.10)

    def test_grad_du_dt_wrt_T(self):
        col, cfg, spec = self._setup()

        def loss(T):
            out, _spec_new = self._run(col, cfg, spec, T=T)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad, "GWD(prognostic_spectral) du_dt w.r.t. T",
                           min_nonzero_frac=0.05)

    def test_grad_wrt_spectrum_carry(self):
        """The prognostic carry itself must be differentiable end to end.

        The spectrum is the state a multi-step (BPTT / 4D-Var) window would
        thread; if d(spectrum_out)/d(spectrum_in) were dead the carry would be
        a constant and the whole prognostic branch would degrade to a
        diagnostic one under optimisation without any error.
        """
        col, cfg, spec = self._setup()

        def loss(spectrum_in):
            out, spec_new = self._run(col, cfg, spectrum_in)
            return jnp.sum(spec_new ** 2) + jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(spec)
        assert_gradient_ok(grad, "GWD(prognostic_spectral) w.r.t. spectrum_in",
                           min_nonzero_frac=0.10)

    def test_grad_through_two_chained_steps(self):
        """Two sequential spectrum updates: gradient must accumulate, not
        vanish or explode across the carried state."""
        col, cfg, spec = self._setup()

        def loss(u):
            out1, spec1 = self._run(col, cfg, spec, u=u)
            out2, spec2 = self._run(col, cfg, spec1, u=u)
            return jnp.sum(out2.du_dt ** 2) + jnp.sum(spec2 ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "GWD(prognostic_spectral) 2-step w.r.t. u",
                           min_nonzero_frac=0.10)


class TestGWDE3SMCAMGrad:
    """The faithful E3SM/CAM ``gw_drag_prof`` solver — all three wave sources.

    This is the most intricate GWD path in the package (per-phase-speed stress
    scan, critical-level filtering, tendency limiters, energy-conservation
    fix-up).  A ``lax.scan`` over the phase-speed spectrum plus integer-valued
    source-level selection makes it the most plausible place for a detached
    gradient, and it had no differentiability coverage in the diff sweep.
    """

    def _col(self, ncol=4, nlev=24):
        return _gwd_column(ncol=ncol, nlev=nlev, seed=3)

    def test_orographic_grad_wrt_u(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            E3SMCAMConfig,
            E3SMOrographicConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

        col = self._col()
        cfg = E3SMCAMConfig(
            source="orographic",
            orographic=E3SMOrographicConfig(sgh_default=200.0),
        )

        def loss(u):
            out = e3sm_cam_gwd(*_gwd_args(col, u=u), 1800.0, cfg)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "GWD(e3sm_cam/orographic) du_dt w.r.t. u",
                           min_nonzero_frac=0.05)

    def test_orographic_grad_wrt_sgh(self):
        """Subgrid-orography amplitude (``sgh``) is the calibration knob."""
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            E3SMCAMConfig,
            E3SMOrographicConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

        col = self._col()
        ncol = col["u"].shape[0]
        cfg = E3SMCAMConfig(
            source="orographic",
            orographic=E3SMOrographicConfig(sgh_default=200.0),
        )
        sgh = jnp.linspace(150.0, 600.0, ncol)

        def loss(h):
            out = e3sm_cam_gwd(*_gwd_args(col), 1800.0, cfg, h_topo_col=h)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(sgh)
        assert_gradient_ok(grad, "GWD(e3sm_cam/orographic) w.r.t. sgh",
                           min_nonzero_frac=0.5)

    def test_orographic_grad_wrt_land_frac(self):
        """``utgw *= landfrac`` (gw_drag.F90:904-906) — the ocean/land mask that
        zeroes orographic drag over water.  Differentiable so a coupled run can
        propagate sensitivity through the surface-type blend."""
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            E3SMCAMConfig,
            E3SMOrographicConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

        col = self._col()
        ncol = col["u"].shape[0]
        cfg = E3SMCAMConfig(
            source="orographic",
            orographic=E3SMOrographicConfig(sgh_default=300.0),
        )
        land_frac = jnp.linspace(0.2, 1.0, ncol)

        def loss(lf):
            out = e3sm_cam_gwd(*_gwd_args(col), 1800.0, cfg,
                               land_frac_col=lf)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(land_frac)
        assert_gradient_ok(grad, "GWD(e3sm_cam/orographic) w.r.t. land_frac",
                           min_nonzero_frac=0.5)

    def test_frontal_grad_wrt_frontgf_is_exactly_zero_step_trigger(self):
        """MEASURED ZERO — the frontal launch is a pure ON/OFF step.

        ``gw_cm_src`` uses the frontogenesis function exactly once::

            launch     = frontgf[:, kfront] > frontgfc          # boolean
            tau_launch = where(launch, fav, 0.0)                # fav has NO frontgf

        so the launched stress is the FIXED Gaussian spectrum ``fav`` gated by
        a boolean — the frontogenesis magnitude sets *whether* waves launch,
        never *how strongly*.  ``d(du_dt)/d(frontgf)`` is therefore identically
        0 (measured max|g| = 0.0, at every one of the 3x24 entries).

        This is FAITHFUL to the oracle (E3SM gw_front.F90:169 does exactly the
        same comparison), so it is a documented property, not a defect.  The
        consequence for AD users is real though: a frontal-GWD experiment
        cannot be differentiated w.r.t. the frontal activity itself — only
        w.r.t. the wind/temperature that set the spectrum and the propagation
        (covered by ``test_frontal_grad_wrt_u``).  ``frontgfc``, the threshold,
        is likewise unreachable.
        """
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            E3SMCAMConfig,
            E3SMFrontalConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

        col = self._col(ncol=3, nlev=24)
        ncol, nlev = col["u"].shape
        cfg = E3SMCAMConfig(
            source="frontal", pgwv=4, dc=5.0,
            frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
        )
        frontgf = jnp.full((ncol, nlev), 1e-9) * (
            1.0 + 0.3 * jnp.linspace(-1.0, 1.0, ncol)[:, None]
        )

        def loss(f):
            out = e3sm_cam_gwd(*_gwd_args(col), 1800.0, cfg, frontgf_col=f)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(frontgf)
        assert_exactly_zero(
            grad, "GWD(e3sm_cam/frontal) du_dt w.r.t. frontgf",
            reason="frontgf enters only the boolean launch test "
                   "frontgf[:,kfront] > frontgfc (E3SM gw_front.F90:169)",
        )
        # Guard against a vacuous zero: the forward drag must be non-trivial,
        # otherwise "gradient is zero" would just mean "nothing happened".
        out = e3sm_cam_gwd(*_gwd_args(col), 1800.0, cfg, frontgf_col=frontgf)
        assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0, (
            "frontal source produced NO drag at all — the exactly-zero "
            "gradient above is vacuous, not a step-function finding."
        )

    def test_frontal_grad_wrt_u(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            E3SMCAMConfig,
            E3SMFrontalConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

        col = self._col(ncol=3, nlev=24)
        ncol, nlev = col["u"].shape
        cfg = E3SMCAMConfig(
            source="frontal", pgwv=4, dc=5.0,
            frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
        )
        frontgf = jnp.full((ncol, nlev), 1e-9)

        def loss(u):
            out = e3sm_cam_gwd(*_gwd_args(col, u=u), 1800.0, cfg,
                               frontgf_col=frontgf)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "GWD(e3sm_cam/frontal) du_dt w.r.t. u",
                           min_nonzero_frac=0.05)

    def test_convective_grad_wrt_netdt(self):
        """Beres convective source: gradient w.r.t. the convective heating
        profile is the coupling that makes convection->GWD trainable.

        NOTE the Beres source ALSO indexes an offline ``mfcc`` lookup with an
        integer NINT of the heating depth; the analytic stand-in table is used
        here (``use_stand_in_table``), so what is measured is the amplitude
        path (``q0^2/AL``), not the table index — see the companion
        exactly-zero documentation in the class docstring of the table test.
        """
        from legoesm.atmosphere.physics.gravity_wave_drag.config import E3SMCAMConfig
        from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd

        col = self._col(ncol=3, nlev=24)
        ncol, nlev = col["u"].shape
        cfg = E3SMCAMConfig(source="convective", pgwv=4, dc=5.0, effgw=0.4)
        # Deep heating (~ surface to mid-troposphere) so the depth test fires.
        sig = col["sigma_full"]
        netdt = 8.0e-4 * jnp.exp(-((sig - 0.6) / 0.25) ** 2)[None, :] * (
            1.0 + 0.2 * jnp.linspace(-1.0, 1.0, ncol)[:, None]
        )

        def loss(q):
            out = e3sm_cam_gwd(*_gwd_args(col), 1800.0, cfg, netdt_col=q)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(netdt)
        assert_gradient_ok(grad, "GWD(e3sm_cam/convective) du_dt w.r.t. netdt",
                           min_nonzero_frac=0.02)


class TestGWDMLEmulatorGrad:
    """Neural GWD emulator — gradient must reach BOTH the column inputs and the
    network weights (the latter is the whole point of an emulator)."""

    def _setup(self):
        import jax.random as jr
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GWDMLEmulatorConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
            GWDEmulator,
        )

        cfg = GWDMLEmulatorConfig(n_hidden=16, n_layers=2)
        model = GWDEmulator(cfg.n_input, cfg.n_hidden, cfg.n_layers,
                            cfg.n_output, key=jr.PRNGKey(cfg.seed))
        return _gwd_column(ncol=4, nlev=12, seed=5), cfg, model

    def test_grad_wrt_u(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import ml_gwd

        col, cfg, model = self._setup()

        def loss(u):
            out = ml_gwd(*_gwd_args(col, u=u), 300.0, cfg, model)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "GWD(ml_emulator) du_dt w.r.t. u",
                           min_nonzero_frac=0.5)

    def test_grad_wrt_network_weights(self):
        """d(loss)/d(MLP weights) — an emulator whose weights are unreachable
        by AD cannot be trained online at all."""
        import equinox as eqx
        from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import ml_gwd

        col, cfg, model = self._setup()

        def loss(m):
            out = ml_gwd(*_gwd_args(col), 300.0, cfg, m)
            return jnp.sum(out.du_dt ** 2) + jnp.sum(out.dT_dt ** 2)

        grads = eqx.filter_grad(loss)(model)
        leaves = [g for g in jax.tree_util.tree_leaves(grads)
                  if isinstance(g, jnp.ndarray) and g.size > 0]
        assert leaves, "ml_emulator: no differentiable weight leaves found"
        flat = jnp.concatenate([g.ravel() for g in leaves])
        assert_gradient_ok(flat, "GWD(ml_emulator) w.r.t. MLP weights",
                           min_nonzero_frac=0.3)


class TestGWDCompositeGrad:
    """``+``-composite (issue #834): orographic + non-orographic summed.

    The composite fans out to several backends and threads one shared
    wave-action spectrum; if the spectrum carry were re-seeded (rather than
    threaded) inside the fan-out, the gradient w.r.t. it would be exactly zero
    while every forward test still passed.
    """

    def test_stateless_composite_grad_wrt_u(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            get_gwd_fn,
        )

        col = _gwd_column()
        ncol = col["u"].shape[0]
        _name, fn, cfg = get_gwd_fn(
            GravityWaveDragConfig(scheme="mcfarlane+hines"))

        def loss(u):
            out, _spec = fn(*_gwd_args(col, u=u), 300.0, cfg, None,
                            h_topo_col=jnp.full((ncol,), 400.0))
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "GWD(mcfarlane+hines) du_dt w.r.t. u",
                           min_nonzero_frac=0.10)

    def test_spectrum_composite_grad_wrt_spectrum(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
            PrognosticSpectralConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            get_gwd_fn,
        )

        col = _gwd_column()
        ncol = col["u"].shape[0]
        ps = PrognosticSpectralConfig(n_azimuths=4, n_wavenumbers=6)
        _name, fn, cfg = get_gwd_fn(GravityWaveDragConfig(
            scheme="mcfarlane+prognostic_spectral", prognostic_spectral=ps))
        spec = jnp.full((ncol, ps.n_azimuths, ps.n_wavenumbers), ps.launch_flux)

        def loss(spectrum_in):
            out, spec_new = fn(*_gwd_args(col), 300.0, cfg, spectrum_in,
                               h_topo_col=jnp.full((ncol,), 400.0))
            return jnp.sum(spec_new ** 2) + jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(spec)
        assert_gradient_ok(
            grad, "GWD(mcfarlane+prognostic_spectral) w.r.t. spectrum_in",
            min_nonzero_frac=0.10)


class TestGWDOroSourceGrad:
    """``depth_averaged_oro_source`` — E3SM's dp-weighted source-region average
    (gw_oro.F90:119-145).  It selects the source interface with an integer
    index, so the interesting question is whether the *averaged* quantities
    stay differentiable through that selection."""

    def _inputs(self, ncol=4, nlev=20):
        col = _gwd_column(ncol=ncol, nlev=nlev, seed=7)
        dpm = jnp.abs(col["p_half"][:, 1:] - col["p_half"][:, :-1])
        # Brunt-Vaisala from the model helper (no re-derivation).
        from legoesm.atmosphere.physics._shared import brunt_vaisala_n_full
        nm = brunt_vaisala_n_full(col["T"], col["p_full"], col["z_full"])
        hdsp = jnp.linspace(300.0, 1200.0, ncol)
        return col, dpm, nm, hdsp

    def test_grad_usrc_wrt_u(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.oro_source import (
            depth_averaged_oro_source,
        )

        col, dpm, nm, hdsp = self._inputs()

        def loss(u):
            rsrc, usrc, vsrc, nsrc, _lvl = depth_averaged_oro_source(
                u, col["v"], col["rho"], hdsp, col["p_half"], dpm,
                col["z_full"], nm,
            )
            return jnp.sum(usrc ** 2) + jnp.sum(rsrc ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "oro_source usrc w.r.t. u",
                           min_nonzero_frac=0.05)

    def test_grad_wrt_hdsp_is_exactly_zero_step_selection(self):
        """MEASURED ZERO — a genuine step function, documented not hidden.

        ``hdsp`` enters ONLY through the boolean penetration test
        ``hdsp > sqrt(zm[i] zm[i+1])``, which picks an integer source level.
        A boolean comparison has zero derivative almost everywhere, so
        d(usrc)/d(hdsp) is identically 0: the source-region DEPTH is not a
        differentiable control here (its amplitude effect enters later, through
        ``tau_0 ~ hdsp^2`` in the caller, which IS differentiable — see
        ``TestGWDE3SMCAMGrad.test_orographic_grad_wrt_sgh``).  Recording the
        exact zero keeps the boundary between the two paths explicit.
        """
        from legoesm.atmosphere.physics.gravity_wave_drag.oro_source import (
            depth_averaged_oro_source,
        )

        col, dpm, nm, hdsp = self._inputs()

        def loss(h):
            rsrc, usrc, vsrc, nsrc, _lvl = depth_averaged_oro_source(
                col["u"], col["v"], col["rho"], h, col["p_half"], dpm,
                col["z_full"], nm,
            )
            return (jnp.sum(usrc ** 2) + jnp.sum(vsrc ** 2)
                    + jnp.sum(rsrc ** 2) + jnp.sum(nsrc ** 2))

        grad = jax.grad(loss)(hdsp)
        assert_exactly_zero(
            grad, "oro_source w.r.t. hdsp",
            reason="hdsp enters only the boolean penetration test "
                   "hdsp > sqrt(zm[i]*zm[i+1]) -> integer level selection",
        )


class TestFrontogenesisGrad:
    """``compute_frontogenesis`` (E3SM FRONTGF) feeds the frontal GWD source.

    The existing unit test differentiates w.r.t. a single SCALAR amplitude,
    which cannot detect a structurally dead cell; here the gradient is taken
    w.r.t. the FULL temperature and wind fields.
    """

    def _grid_fields(self, n_lat=8):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.physics.gravity_wave_drag.frontogenesis import (
            frontogenesis_supported,
        )

        grid = create_latlon_grid(n_lat, dtype=jnp.float64)
        if not frontogenesis_supported(grid):
            pytest.skip("uniform-global lat-lon predicate rejected the fixture "
                        "grid; frontogenesis has no operator for it")
        n_lon = grid.lon.shape[0]
        nlev = 4
        lat2d = jnp.asarray(grid.lat2d)[:, :, None]
        lon2d = jnp.asarray(grid.lon2d)[:, :, None]
        lev = jnp.arange(nlev, dtype=jnp.float64)[None, None, :]
        # Smooth global wave field: a real deformation pattern so FRONTGF is
        # genuinely non-zero (a solid-body field would give F == 0 by design).
        u = 20.0 * jnp.cos(lat2d) * jnp.cos(lon2d) + 0.0 * lev
        v = 8.0 * jnp.sin(2.0 * lon2d) * jnp.cos(lat2d) + 0.0 * lev
        T = 280.0 - 30.0 * jnp.sin(lat2d) ** 2 + 4.0 * jnp.cos(lon2d) + 0.0 * lev
        p_full = jnp.broadcast_to(
            jnp.linspace(3.0e4, 9.0e4, nlev)[None, None, :], T.shape)
        return grid, u, v, T, p_full

    def test_grad_frontgf_wrt_T(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.frontogenesis import (
            compute_frontogenesis,
        )

        grid, u, v, T, p_full = self._grid_fields()

        def loss(T_in):
            frontgf, _frontga = compute_frontogenesis(u, v, T_in, p_full, grid)
            return jnp.sum(frontgf ** 2)

        grad = jax.grad(loss)(T)
        assert_gradient_ok(grad, "frontogenesis FRONTGF w.r.t. T",
                           min_nonzero_frac=0.5)

    def test_grad_frontgf_wrt_u(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.frontogenesis import (
            compute_frontogenesis,
        )

        grid, u, v, T, p_full = self._grid_fields()

        def loss(u_in):
            frontgf, _frontga = compute_frontogenesis(u_in, v, T, p_full, grid)
            return jnp.sum(frontgf ** 2)

        grad = jax.grad(loss)(u)
        assert_gradient_ok(grad, "frontogenesis FRONTGF w.r.t. u",
                           min_nonzero_frac=0.5)


def _bridge_gwd_config(scheme):
    """``GravityWaveDragConfig`` for the C4 bridge tests.

    ``E3SMOrographicConfig.sgh_default`` is 0.0 in production ("no oro waves
    unless the dataset supplies sgh"), and the C4 test grid carries no
    ``subgrid_topo_stddev`` attribute, so a bare ``scheme="e3sm_cam"`` launches
    NOTHING and every gradient is legitimately zero (measured max|g| = 0.0).
    Give that one scheme a real mountain so the bridge test is not vacuous.
    """
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        E3SMCAMConfig,
        E3SMOrographicConfig,
        GravityWaveDragConfig,
    )

    if scheme == "e3sm_cam":
        return GravityWaveDragConfig(
            scheme=scheme,
            e3sm_cam=E3SMCAMConfig(
                source="orographic",
                orographic=E3SMOrographicConfig(sgh_default=250.0),
            ),
        )
    return GravityWaveDragConfig(scheme=scheme)


class TestGWDHydrostaticBridgeGrad:
    """``make_gwd_physics(model_type="hydrostatic")`` on a C4 cubed sphere.

    The bridge is where ``T`` also flows into ``z_full/z_half`` and ``rho``
    (via the hydrostatic helpers), so this covers the FULL temperature
    sensitivity that the leaf-level test (which holds z/rho fixed) cannot.
    """

    def _state(self, n=4, nlev=12):
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate

        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
        sigma_full = jnp.asarray(sigma.sigma_full, dtype=jnp.float64)
        shape_3d = (6, n, n, nlev)
        # Same profile rationale as ``_gwd_column``: weak shear + a stable
        # ~6.5 K/km lapse rate so the orographic wave saturates aloft.  The
        # per-face multiplier supplies horizontal structure.
        face = 1.0 + 0.03 * jnp.arange(6, dtype=jnp.float64)[:, None, None, None]
        ones = jnp.ones(shape_3d, dtype=jnp.float64)
        T = (230.0 + 60.0 * sigma_full) * face * ones
        u = (10.0 + 4.0 * (1.0 - sigma_full)) * face * ones
        v = 3.0 * ones
        state = HydrostaticState(
            u=Field(u, name="u"),
            v=Field(v, name="v"),
            T=Field(T, name="T"),
            p_s=Field(1.0e5 * jnp.ones((6, n, n)), name="p_s"),
            phis=Field(jnp.zeros((6, n, n)), name="phis"),
            tracers={"q_v": Field(1e-3 * jnp.ones(shape_3d), name="q_v")},
        )
        return state, grid, sigma

    @pytest.mark.parametrize(
        "scheme",
        ["rayleigh", "lindzen", "mcfarlane", "hines", "prognostic_spectral",
         "e3sm_cam", "ml_emulator", "mcfarlane+hines"],
    )
    def test_bridge_grad_wrt_u(self, scheme):
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            make_gwd_physics,
        )

        state, grid, sigma = self._state()
        cfg = _bridge_gwd_config(scheme)
        gwd_fn = make_gwd_physics(cfg, model_type="hydrostatic", dt=300.0)

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            tend, _prog = gwd_fn(s, grid, sigma)
            return jnp.sum(tend.du_dt.data ** 2)

        grad = jax.grad(loss)(state.u.data)
        min_frac = 0.05 if scheme == "rayleigh" else 0.10
        assert_gradient_ok(grad, f"make_gwd_physics({scheme}) du_dt w.r.t. u",
                           min_nonzero_frac=min_frac)

    @pytest.mark.parametrize(
        "scheme", ["lindzen", "mcfarlane", "hines", "prognostic_spectral"])
    def test_bridge_grad_wrt_T_full_path(self, scheme):
        """T flows into N^2 AND into the hydrostatic heights/density here."""
        from legoesm.atmosphere.physics.gravity_wave_drag.config import (
            GravityWaveDragConfig,
        )
        from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
            make_gwd_physics,
        )

        state, grid, sigma = self._state()
        cfg = _bridge_gwd_config(scheme)
        gwd_fn = make_gwd_physics(cfg, model_type="hydrostatic", dt=300.0)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, _prog = gwd_fn(s, grid, sigma)
            return jnp.sum(tend.du_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, f"make_gwd_physics({scheme}) du_dt w.r.t. T",
                           min_nonzero_frac=0.05)


# ===========================================================================
# B.  Turbulence — column leaves
# ===========================================================================
#
# The existing sweep (test_diff_atmosphere_physics.py::TestTurbulenceGrad)
# tests a subset of schemes at the FACTORY level w.r.t. T only, and does not
# touch mynn25 or full CLUBB at all.  Here every scheme is exercised at the
# LEAF level against the inputs the factory test never varies: the wind, the
# moisture, the prognostic TKE/qke carry, and the surface temperature.

def _turb_column(ncol=8, nlev=12, seed=1):
    """Sheared, convectively active column (index 0 = top, -1 = surface).

    Turbulence closures are gated on stability: an isothermal/stable column
    leaves the faithful stable-cutoff branches at K ~ 0, where a dead gradient
    is indistinguishable from correct behaviour.  This fixture is
    super-adiabatic near the surface with real shear so every scheme's ACTIVE
    branch is the one under test.

    (The PBL-height diagnostics need the OPPOSITE regime — a Richardson number
    sitting ON its critical value rather than saturated far from it — so they
    use the dedicated ``_pbl_column`` instead.)
    """
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.physics._shared import (
        compute_heights_from_sigma,
        compute_rho,
    )
    from legoesm.thermo import saturation_mixing_ratio

    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
    sigma_full = jnp.asarray(sigma.sigma_full, dtype=jnp.float64)
    sigma_half = jnp.asarray(sigma.sigma_half, dtype=jnp.float64)

    key = jax.random.PRNGKey(seed)
    kT, ku, kv, kq = jax.random.split(key, 4)
    p_s = 1.0e5 * (1.0 + 0.005 * jnp.linspace(-1.0, 1.0, ncol))
    p_half = p_s[:, None] * sigma_half[None, :]
    p_full = p_s[:, None] * sigma_full[None, :]

    # Super-adiabatic near the surface, capped aloft.
    T = (240.0 + 60.0 * sigma_full[None, :]
         + 1.0 * jax.random.normal(kT, (ncol, nlev)))
    u = 8.0 + 6.0 * (1.0 - sigma_full)[None, :] + jax.random.normal(ku, (ncol, nlev))
    v = 2.0 + 0.5 * jax.random.normal(kv, (ncol, nlev))
    q_v = (2.0e-3 + 6.0e-3 * sigma_full[None, :] ** 2
           + 2.0e-4 * jax.random.uniform(kq, (ncol, nlev)))

    z_full, z_half = compute_heights_from_sigma(T, p_half, q_v=q_v)
    rho = compute_rho(T, p_full, q_v=q_v)
    # Warm surface -> upward sensible heat flux -> active convective BL.
    T_sfc = T[:, -1] + 3.0
    q_sfc = saturation_mixing_ratio(T_sfc, p_full[:, -1])
    tke = jnp.full((ncol, nlev), 0.4)

    return dict(u=u, v=v, T=T, q_v=q_v, tke=tke, p_full=p_full, p_half=p_half,
                z_full=z_full, z_half=z_half, T_sfc=T_sfc, q_sfc=q_sfc,
                rho=rho, sigma_full=sigma_full)


# scheme -> (carries a prognostic TKE/qke carry?)
TURB_CARRY = ["tke", "mynn25", "clubb_lite", "edmf"]
TURB_DIAGNOSTIC = ["ysu", "holtslag_boville"]
TURB_ALL = TURB_CARRY + TURB_DIAGNOSTIC


def _turb_leaf(scheme):
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        get_turbulence_fn,
        turbulence_scheme_traits,
    )

    name, fn, cfg = get_turbulence_fn(TurbulenceConfig(scheme=scheme))
    return fn, cfg, turbulence_scheme_traits(name).carries_energy


def _run_turb(scheme, col, **override):
    """Call a turbulence backend with its scheme-appropriate signature."""
    fn, cfg, carries = _turb_leaf(scheme)
    c = dict(col)
    c.update(override)
    if carries:
        out, carry_new = fn(
            c["u"], c["v"], c["T"], c["q_v"], c["tke"],
            c["p_full"], c["p_half"], c["z_full"], c["z_half"],
            c["T_sfc"], c["q_sfc"], c["rho"], 300.0, cfg,
        )
        return out, carry_new
    out = fn(
        c["u"], c["v"], c["T"], c["q_v"],
        c["p_full"], c["p_half"], c["z_full"], c["z_half"],
        c["T_sfc"], c["q_sfc"], c["rho"], 300.0, cfg,
    )
    return out, None


class TestTurbulenceLeafGrad:

    @pytest.mark.parametrize("scheme", TURB_ALL)
    def test_grad_du_dt_wrt_u(self, scheme):
        """Momentum mixing must respond to the shear that produces it."""
        col = _turb_column()

        def loss(u):
            out, _c = _run_turb(scheme, col, u=u)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, f"turbulence({scheme}) du_dt w.r.t. u",
                           min_nonzero_frac=0.05)

    @pytest.mark.parametrize("scheme", TURB_ALL)
    def test_grad_dT_dt_wrt_T(self, scheme):
        col = _turb_column()

        def loss(T):
            out, _c = _run_turb(scheme, col, T=T)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad, f"turbulence({scheme}) dT_dt w.r.t. T",
                           min_nonzero_frac=0.05)

    @pytest.mark.parametrize("scheme", TURB_ALL)
    def test_grad_dq_v_dt_wrt_q_v(self, scheme):
        """Moisture mixing: a dead dq_v path means the scheme transports heat
        but not water, which silently breaks the column water budget's
        sensitivity (and any humidity-based training signal)."""
        col = _turb_column()

        def loss(q_v):
            out, _c = _run_turb(scheme, col, q_v=q_v)
            return jnp.sum(out.dq_v_dt ** 2)

        grad = jax.grad(loss)(col["q_v"])
        assert_gradient_ok(grad, f"turbulence({scheme}) dq_v_dt w.r.t. q_v",
                           min_nonzero_frac=0.05)

    @pytest.mark.parametrize("scheme", TURB_ALL)
    def test_grad_wrt_T_sfc(self, scheme):
        """Surface coupling: d(tendencies)/d(T_sfc).

        This is the atmosphere->surface adjoint link (SST / land-skin
        sensitivity in a coupled 4D-Var); it enters only through the bulk
        surface flux, so it is easy to detach without any forward symptom.
        """
        col = _turb_column()

        def loss(T_sfc):
            out, _c = _run_turb(scheme, col, T_sfc=T_sfc)
            return (jnp.sum(out.dT_dt ** 2) + jnp.sum(out.shflx ** 2)
                    + jnp.sum(out.lhflx ** 2))

        grad = jax.grad(loss)(col["T_sfc"])
        assert_gradient_ok(grad, f"turbulence({scheme}) w.r.t. T_sfc",
                           min_nonzero_frac=0.5)

    @pytest.mark.parametrize("scheme", TURB_CARRY)
    def test_grad_wrt_tke_carry(self, scheme):
        """The prognostic TKE/qke carry must be differentiable both ways.

        A carry with a dead d(carry_out)/d(carry_in) degrades a prognostic
        closure to a diagnostic one across a training window without changing
        any single-step forward result.
        """
        col = _turb_column()

        def loss(tke):
            out, carry_new = _run_turb(scheme, col, tke=tke)
            return jnp.sum(carry_new ** 2) + jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(col["tke"])
        assert_gradient_ok(grad, f"turbulence({scheme}) carry w.r.t. tke_in",
                           min_nonzero_frac=0.05)

    # Schemes whose diffusivity is a DIRECT function of the thermal state
    # (a Richardson number / buoyancy-flux stability function evaluated this
    # step) vs schemes that diagnose K purely from the carried TKE and a
    # geometric length scale.  Measured, not assumed — see the two tests below.
    TURB_K_STABILITY = ["mynn25", "ysu", "holtslag_boville"]
    TURB_K_FROM_CARRY = ["tke", "clubb_lite", "edmf"]

    @pytest.mark.parametrize("scheme", TURB_K_STABILITY)
    def test_grad_diffusivity_wrt_T(self, scheme):
        """d(K_h)/dT — the stability dependence of the eddy diffusivity itself.

        These schemes gate K on a Richardson number or a surface buoyancy flux
        evaluated from THIS step's temperature; a zero gradient would mean the
        stability function has latched (the failure mode a faithful Lilly /
        MOST cutoff can produce)."""
        col = _turb_column()

        def loss(T):
            out, _c = _run_turb(scheme, col, T=T)
            return jnp.sum(out.Kh ** 2) + jnp.sum(out.Km ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad, f"turbulence({scheme}) Kh/Km w.r.t. T",
                           min_nonzero_frac=0.05)

    @pytest.mark.parametrize("scheme", TURB_K_FROM_CARRY)
    def test_grad_diffusivity_wrt_T_is_exactly_zero_carry_diagnosed(self, scheme):
        """MEASURED ZERO — K is diagnosed from the CARRIED TKE, not from T.

        ``tke.py`` states the closure explicitly::

            Km = Ck * l * sqrt(max(TKE, tke_min))
            Kh = Km / Pr_t          # CONSTANT Pr_t = 0.33

        Both factors read the INPUT carry and a geometric mixing length; the
        temperature never appears, and the Prandtl number is a constant rather
        than a stability function.  So ``d(K)/dT`` is identically 0 (measured
        max|g| = 0.0 for all of tke / clubb_lite / edmf) — stratification
        reaches the diffusivity only INDIRECTLY, one step later, through the
        buoyancy production term that evolves the carry.

        That is a legitimate structural property of a prognostic k-l closure,
        not a bug.  It is worth pinning because it is easy to mistake for one:
        a single-step sensitivity study will see zero response of K to a
        temperature perturbation, and only a MULTI-step window recovers it.
        The tendency-level tests above (``test_grad_dT_dt_wrt_T``) and the
        carry test (``test_grad_wrt_tke_carry``) confirm the two halves of that
        path are individually alive.
        """
        col = _turb_column()

        def loss(T):
            out, _c = _run_turb(scheme, col, T=T)
            return jnp.sum(out.Kh ** 2) + jnp.sum(out.Km ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_exactly_zero(
            grad, f"turbulence({scheme}) Kh/Km w.r.t. T",
            reason="K = Ck*l*sqrt(TKE_in)/Pr_t reads the carry and a geometric "
                   "length scale only; Pr_t is constant",
        )
        # Non-vacuous: K itself must be non-trivial.
        out, _c = _run_turb(scheme, col)
        assert float(jnp.max(jnp.abs(out.Kh))) > 0.0, (
            f"turbulence({scheme}): Kh is identically zero, so the "
            "exactly-zero gradient above is vacuous."
        )


class TestCLUBBFullGrad:
    """Full (higher-order, ADG1-PDF) CLUBB.

    Not present in the diff sweep at all.  CLUBB is the deepest turbulence
    graph in the package: pentadiagonal solves, a fixed-trip
    ``_bounded_while`` mixing-length search, hole filling, and PDF closure.
    ``_bounded_while`` is explicitly a ``lax.scan`` replacement for
    ``lax.while_loop`` *because* the latter has no reverse-mode rule — so this
    is the test that keeps that property honest.
    """

    def _col(self):
        # CLUBB wants enough levels to resolve a mixing length.
        return _turb_column(ncol=3, nlev=16, seed=13)

    def _run(self, col, **override):
        from legoesm.atmosphere.physics.turbulence.clubb import (
            CLUBBConfig,
            clubb_turbulence,
        )
        c = dict(col)
        c.update(override)
        return clubb_turbulence(
            c["u"], c["v"], c["T"], c["q_v"], c["tke"],
            c["p_full"], c["p_half"], c["z_full"], c["z_half"],
            c["T_sfc"], c["q_sfc"], c["rho"], 300.0, CLUBBConfig(),
        )

    def test_grad_dT_dt_wrt_T(self):
        col = self._col()

        def loss(T):
            out, _wp2 = self._run(col, T=T)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad, "CLUBB dT_dt w.r.t. T", min_nonzero_frac=0.05)

    def test_grad_du_dt_wrt_u(self):
        col = self._col()

        def loss(u):
            out, _wp2 = self._run(col, u=u)
            return jnp.sum(out.du_dt ** 2)

        grad = jax.grad(loss)(col["u"])
        assert_gradient_ok(grad, "CLUBB du_dt w.r.t. u", min_nonzero_frac=0.05)

    def test_grad_wp2_carry_wrt_tke(self):
        """The returned ``wp2`` carry — the prognostic state CLUBB threads."""
        col = self._col()

        def loss(tke):
            out, wp2 = self._run(col, tke=tke)
            return jnp.sum(wp2 ** 2) + jnp.sum(out.Kh ** 2)

        grad = jax.grad(loss)(col["tke"])
        assert_gradient_ok(grad, "CLUBB wp2 w.r.t. tke_in",
                           min_nonzero_frac=0.05)

    def test_grad_cloud_fraction_wrt_q_v(self):
        """CLUBB's PDF cloud fraction is routed to radiation under
        ``cloud_scheme="clubb"``, so d(cf)/d(q_v) is the link that lets a
        radiative loss train the sub-grid moisture PDF."""
        col = self._col()

        def loss(q_v):
            out, _wp2 = self._run(col, q_v=q_v)
            if out.cloud_fraction is None:
                pytest.skip("this CLUBB configuration produced no PDF cloud "
                            "fraction")
            return jnp.sum(out.cloud_fraction ** 2)

        grad = jax.grad(loss)(col["q_v"])
        assert_gradient_ok(grad, "CLUBB cloud_fraction w.r.t. q_v",
                           min_nonzero_frac=0.05)


# ===========================================================================
# C.  Turbulence — leaf helper modules
# ===========================================================================

class TestVerticalDiffusionGrad:
    """``implicit_vertical_diffusion`` — the shared batched Thomas solve every
    BL scheme's tendency ultimately passes through."""

    def _inputs(self, ncol=6, nlev=10):
        col = _turb_column(ncol=ncol, nlev=nlev, seed=21)
        z_half = col["z_half"]
        dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
        z_full = col["z_full"]
        dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
        K_half = 5.0 + 20.0 * jnp.linspace(0.1, 1.0, nlev - 1)[None, :] \
            * jnp.ones((ncol, 1))
        sfc_flux = jnp.linspace(0.01, 0.05, ncol)
        return col, K_half, dz, dz_half, sfc_flux

    def test_grad_wrt_phi(self):
        from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
            implicit_vertical_diffusion,
        )

        col, K_half, dz, dz_half, sfc_flux = self._inputs()

        def loss(phi):
            out = implicit_vertical_diffusion(
                phi, K_half, col["rho"], dz, dz_half, 300.0, sfc_flux)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss)(col["q_v"])
        assert_gradient_ok(grad, "implicit_vertical_diffusion w.r.t. phi",
                           min_nonzero_frac=0.9)

    def test_grad_wrt_K_half(self):
        """d(phi_new)/dK — the diffusivity is the tunable the whole closure
        produces; a dead gradient makes every BL scheme's output uncalibratable
        through the solver."""
        from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
            implicit_vertical_diffusion,
        )

        col, K_half, dz, dz_half, sfc_flux = self._inputs()

        def loss(K):
            out = implicit_vertical_diffusion(
                col["q_v"], K, col["rho"], dz, dz_half, 300.0, sfc_flux)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss)(K_half)
        assert_gradient_ok(grad, "implicit_vertical_diffusion w.r.t. K_half",
                           min_nonzero_frac=0.9)

    def test_grad_wrt_surface_flux(self):
        """Bottom boundary condition — the surface->atmosphere adjoint link."""
        from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
            implicit_vertical_diffusion,
        )

        col, K_half, dz, dz_half, sfc_flux = self._inputs()

        def loss(f):
            out = implicit_vertical_diffusion(
                col["q_v"], K_half, col["rho"], dz, dz_half, 300.0, f)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss)(sfc_flux)
        assert_gradient_ok(grad,
                           "implicit_vertical_diffusion w.r.t. surface_flux",
                           min_nonzero_frac=0.9)

    def test_theta_variant_grad_wrt_T(self):
        """The potential-temperature variant carries an extra Exner factor;
        it is the one the heat tendency actually uses."""
        from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
            implicit_vertical_diffusion_theta,
        )

        col, K_half, dz, dz_half, sfc_flux = self._inputs()

        def loss(T):
            out = implicit_vertical_diffusion_theta(
                T, K_half, col["rho"], dz, dz_half, col["p_full"], 300.0,
                sfc_flux)
            return jnp.sum(out ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad,
                           "implicit_vertical_diffusion_theta w.r.t. T",
                           min_nonzero_frac=0.9)


class TestSurfaceLayerGrad:
    """``compute_surface_fluxes`` — the MOST / bulk surface layer.

    ``bulk_scheme="coare3"`` runs a ``lax.fori_loop`` MOST iteration; the
    gradient must survive the fixed-point loop, not just the constant-C_d path.
    """

    def _inputs(self, ncol=6, nlev=8):
        """``compute_surface_fluxes`` takes LOWEST-LEVEL (ncol,) arrays, not
        (ncol, nlev) columns — it is the bulk formula, evaluated once."""
        col = _turb_column(ncol=ncol, nlev=nlev, seed=23)
        return dict(
            u=col["u"][:, -1], v=col["v"][:, -1], T=col["T"][:, -1],
            q_v=col["q_v"][:, -1], T_sfc=col["T_sfc"], q_sfc=col["q_sfc"],
            rho=col["rho"][:, -1],
        )

    @pytest.mark.parametrize("bulk_scheme", ["constant", "coare3"])
    def test_grad_shflx_wrt_T_sfc(self, bulk_scheme):
        from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
        from legoesm.atmosphere.physics.turbulence.surface_layer import (
            compute_surface_fluxes,
        )

        c = self._inputs()
        cfg = SurfaceLayerConfig(bulk_scheme=bulk_scheme)

        def loss(T_sfc):
            taux, tauy, shflx, lhflx, ustar = compute_surface_fluxes(
                c["u"], c["v"], c["T"], c["q_v"], T_sfc,
                c["q_sfc"], c["rho"], cfg)
            return jnp.sum(shflx ** 2) + jnp.sum(lhflx ** 2)

        grad = jax.grad(loss)(c["T_sfc"])
        assert_gradient_ok(grad,
                           f"surface_layer({bulk_scheme}) fluxes w.r.t. T_sfc",
                           min_nonzero_frac=0.9)

    @pytest.mark.parametrize("bulk_scheme", ["constant", "coare3"])
    def test_grad_ustar_wrt_u(self, bulk_scheme):
        """``coare3`` runs a ``lax.fori_loop`` MOST fixed-point iteration; the
        gradient must survive it, not just the constant-C_d path."""
        from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
        from legoesm.atmosphere.physics.turbulence.surface_layer import (
            compute_surface_fluxes,
        )

        c = self._inputs()
        cfg = SurfaceLayerConfig(bulk_scheme=bulk_scheme)

        def loss(u):
            taux, tauy, shflx, lhflx, ustar = compute_surface_fluxes(
                u, c["v"], c["T"], c["q_v"], c["T_sfc"],
                c["q_sfc"], c["rho"], cfg)
            return jnp.sum(ustar ** 2) + jnp.sum(taux ** 2)

        grad = jax.grad(loss)(c["u"])
        assert_gradient_ok(grad,
                           f"surface_layer({bulk_scheme}) ustar w.r.t. u",
                           min_nonzero_frac=0.9)


def _pbl_column(ncol=6, nlev=20):
    """Column whose bulk Richardson number CROSSES Ri_crit = 0.25 smoothly.

    Why this needs its own fixture: ``diagnose_pbl_height`` weights levels by
    ``sigmoid(sharpness * (Ri - Ri_crit))`` with ``sharpness = 20``.  On the
    convective ``_turb_column`` (Ri strongly negative near the surface, strongly
    positive aloft) that sigmoid is SATURATED at every level and the gradient
    underflows: measured max|d(h_pbl)/dT| = 7.9e-86, i.e. numerically dead even
    though formally non-zero.  That is a property of the profile, not of the
    code — but it is a real trap for anyone training against a PBL-height
    target, so the fixture is built to sit ON the transition instead.

    Construction: weak stable stratification (theta = 288 K + 1.5 K/km * z) with
    a surface-concentrated shear that saturates aloft
    (``u = u0 + 18 (1 - exp(-z/600))``).  Because the shear saturates while the
    buoyancy keeps growing, Ri rises monotonically through 0.25 near z ~ 1.8 km
    with 2-3 levels inside the sigmoid's transition width.
    """
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.physics._shared import (
        compute_heights_from_sigma,
        compute_rho,
        exner_function,
    )

    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
    sigma_full = jnp.asarray(sigma.sigma_full, dtype=jnp.float64)
    sigma_half = jnp.asarray(sigma.sigma_half, dtype=jnp.float64)
    span = jnp.linspace(-1.0, 1.0, ncol)
    p_s = 1.0e5 * (1.0 + 0.005 * span)
    p_half = p_s[:, None] * sigma_half[None, :]
    p_full = p_s[:, None] * sigma_full[None, :]
    q_v = (4.0e-3 * sigma_full[None, :] ** 2) * jnp.ones((ncol, 1))

    # Pass 1: a first-guess T to get heights, then rebuild T from the target
    # theta(z) profile and recompute the heights consistently.
    T_guess = (230.0 + 60.0 * sigma_full)[None, :] * jnp.ones((ncol, 1))
    z_full, _z_half = compute_heights_from_sigma(T_guess, p_half, q_v=q_v)
    theta = 288.0 + 1.5e-3 * z_full
    T = theta * exner_function(p_full)
    z_full, z_half = compute_heights_from_sigma(T, p_half, q_v=q_v)

    theta = 288.0 + 1.5e-3 * z_full
    T = theta * exner_function(p_full)
    rho = compute_rho(T, p_full, q_v=q_v)
    # Surface-concentrated, height-saturating shear (per-column amplitude gives
    # the gradient spatial structure).
    u_amp = 18.0 * (1.0 + 0.05 * span)[:, None]
    u = 2.0 + u_amp * (1.0 - jnp.exp(-z_full / 600.0))
    v = 0.5 * jnp.ones_like(u)
    return dict(u=u, v=v, T=T, q_v=q_v, p_full=p_full, p_half=p_half,
                z_full=z_full, z_half=z_half, rho=rho)


class TestPBLHeightGrad:
    """``diagnose_pbl_height`` uses a SIGMOID-weighted first crossing of the
    critical bulk Richardson number precisely so the PBL top stays
    differentiable; that smoothing is what these tests protect.

    The gradient is INHERENTLY localized — only levels inside the sigmoid's
    transition width contribute — so the non-zero fraction is expected to be
    small.  What matters is that the surviving entries are numerically usable,
    which is why every assertion here also carries an explicit magnitude floor.
    """

    _MAG_FLOOR = 1.0e-20   # below this an optimizer sees nothing in float64

    def _assert_usable(self, grad, name, min_nonzero_frac):
        assert_gradient_ok(grad, name, min_nonzero_frac=min_nonzero_frac)
        gmax = float(jnp.max(jnp.abs(grad)))
        assert gmax > self._MAG_FLOOR, (
            f"{name}: gradient is formally non-zero but numerically DEAD "
            f"(max|g| = {gmax:.3e} < {self._MAG_FLOOR:.0e}). The Ri sigmoid has "
            "saturated at every level, so no optimizer can move the PBL height."
        )

    def test_grad_h_pbl_wrt_T(self):
        from legoesm.atmosphere.physics.turbulence.pbl_height import (
            diagnose_pbl_height,
        )

        col = _pbl_column()

        def loss(T):
            h = diagnose_pbl_height(T, col["q_v"], col["u"], col["v"],
                                    col["p_full"], col["z_full"])
            return jnp.sum(h ** 2)

        grad = jax.grad(loss)(col["T"])
        self._assert_usable(grad, "diagnose_pbl_height w.r.t. T", 0.05)

    def test_grad_h_pbl_wrt_u(self):
        from legoesm.atmosphere.physics.turbulence.pbl_height import (
            diagnose_pbl_height,
        )

        col = _pbl_column()

        def loss(u):
            h = diagnose_pbl_height(col["T"], col["q_v"], u, col["v"],
                                    col["p_full"], col["z_full"])
            return jnp.sum(h ** 2)

        grad = jax.grad(loss)(col["u"])
        self._assert_usable(grad, "diagnose_pbl_height w.r.t. u", 0.05)

    def test_grad_bulk_richardson_wrt_T(self):
        """``compute_bulk_richardson`` returns ``(Ri_bulk, theta_v)`` — both
        halves must be differentiable, so the loss uses both."""
        from legoesm.atmosphere.physics.turbulence.pbl_height import (
            compute_bulk_richardson,
        )

        col = _pbl_column()

        def loss(T):
            Ri, theta_v = compute_bulk_richardson(
                T, col["q_v"], col["u"], col["v"], col["p_full"],
                col["z_full"])
            return jnp.sum(Ri ** 2) + jnp.sum(theta_v ** 2)

        grad = jax.grad(loss)(col["T"])
        assert_gradient_ok(grad, "compute_bulk_richardson w.r.t. T",
                           min_nonzero_frac=0.5)

    def test_grad_first_crossing_height_wrt_values(self):
        """The smooth crossing operator itself, isolated from the Ri physics."""
        from legoesm.atmosphere.physics.turbulence.pbl_height import (
            first_crossing_height,
        )

        col = _pbl_column()
        # Monotone-ish profile crossing 0.25 somewhere in the interior.
        values = jnp.linspace(1.0, 0.0, col["T"].shape[1])[None, :] \
            * jnp.linspace(0.8, 1.2, col["T"].shape[0])[:, None]

        def loss(v):
            h = first_crossing_height(v, col["z_full"], 0.25, 20.0)
            return jnp.sum(h ** 2)

        grad = jax.grad(loss)(values)
        assert_gradient_ok(grad, "first_crossing_height w.r.t. values",
                           min_nonzero_frac=0.3)


class TestLESSubgridClosureGrad:
    """AMD / Vreman / LASD sub-grid eddy viscosities.

    The existing faithfulness tests differentiate LASD w.r.t. a single SCALAR
    multiplier, which cannot detect a structurally dead cell.  These take the
    gradient w.r.t. the FULL field so a per-cell mask failure is visible.
    """

    def _grads(self, shape, seed):
        rng = np.random.default_rng(seed)
        return [jnp.asarray(rng.standard_normal(shape) * 0.4) for _ in range(9)]

    @pytest.mark.parametrize("closure", ["amd", "vreman"])
    def test_grad_nu_t_wrt_velocity_gradient(self, closure):
        if closure == "amd":
            from legoesm.atmosphere.physics.turbulence.amd import amd_nu_t as fn
            coeff = 0.3
        else:
            from legoesm.atmosphere.physics.turbulence.vreman import (
                vreman_nu_t as fn,
            )
            coeff = 0.07
        shape = (4, 6, 5)
        a = self._grads(shape, seed=31)
        dx = dy = dz = 10.0

        def loss(a11):
            nu = fn(a11, a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                    dx, dy, dz, coeff)
            return jnp.sum(nu ** 2)

        grad = jax.grad(loss)(a[0])
        # Both closures mask nu_t to exactly zero where the resolved flow needs
        # no SGS dissipation (AMD's minimum-dissipation property), so a
        # substantial fraction of cells is legitimately dead.
        assert_gradient_ok(grad, f"{closure}_nu_t w.r.t. a11",
                           min_nonzero_frac=0.2)

    def test_lasd_cs2_grad_wrt_full_velocity_field(self):
        from legoesm.atmosphere.physics.turbulence.lasd_core import lasd_cs2

        ny, nx, nz = 8, 8, 4
        rng = np.random.default_rng(37)
        uc = jnp.asarray(rng.standard_normal((ny, nx, nz)))
        vc = jnp.asarray(rng.standard_normal((ny, nx, nz)))
        wc = jnp.asarray(rng.standard_normal((ny, nx, nz)))
        S = [jnp.asarray(rng.standard_normal((ny, nx, nz)) * 0.5)
             for _ in range(6)]
        Smag = jnp.sqrt(
            2.0 * (S[0] ** 2 + S[1] ** 2 + S[2] ** 2)
            + 4.0 * (S[3] ** 2 + S[4] ** 2 + S[5] ** 2)
        )
        # ``lasd_cs2`` reshapes ``delta`` to (1, 1, nz): it is a PER-LEVEL
        # filter width, not a scalar.
        delta = jnp.full((nz,), 10.0)

        def loss(u):
            cs2 = lasd_cs2(u, vc, wc, S[0], S[1], S[2], S[3], S[4], S[5],
                           Smag, delta)
            return jnp.sum(cs2 ** 2)

        grad = jax.grad(loss)(uc)
        assert_gradient_ok(grad, "lasd_cs2 w.r.t. uc (full field)",
                           min_nonzero_frac=0.3)


# ===========================================================================
# D.  Radiation — solar geometry
# ===========================================================================

class TestSolarGeometryGrad:
    """``radiation/solar.py`` — insolation, zenith angle, orbital declination.

    Solar geometry is the boundary condition every radiation scheme is driven
    by, so its gradients are what an insolation- or orbit-perturbation
    experiment (and any seasonal-cycle loss) differentiates through.  The
    module carries hand-written AD-safety machinery around the polar
    singularity (a where-before-divide plus a strictly-interior ``arccos``
    clip); these tests keep that machinery honest, including AT the poles and
    AT solstice, where the naive form produces NaN.
    """

    def test_grad_daily_mean_insolation_wrt_lat(self):
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation,
        )

        lat = jnp.linspace(-1.4, 1.4, 24)

        def loss(l):
            return jnp.sum(daily_mean_insolation(l, 172.0) ** 2)

        grad = jax.grad(loss)(lat)
        assert_gradient_ok(grad, "daily_mean_insolation w.r.t. lat",
                           min_nonzero_frac=0.8)

    def test_grad_daily_mean_insolation_wrt_day_of_year(self):
        """The seasonal cycle must be differentiable in time — this is the
        knob an orbital / paleo sensitivity study perturbs."""
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation,
        )

        lat = jnp.linspace(-1.4, 1.4, 24)

        def loss(day):
            return jnp.sum(daily_mean_insolation(lat, day) ** 2)

        grad = jax.grad(loss)(jnp.asarray(172.0))
        assert_finite(grad, "daily_mean_insolation w.r.t. day_of_year")
        assert float(jnp.abs(grad)) > 0.0, (
            "daily_mean_insolation: d/d(day_of_year) is exactly zero — the "
            "seasonal cycle is detached from the declination."
        )

    def test_grad_at_poles_and_solstice_is_finite_and_nonzero(self):
        """AD-safety regression at the polar-day/polar-night singularity.

        ``cos(h_s) = -tan(lat) tan(delta)`` blows up as ``cos(lat) -> 0``.  The
        module masks the pole branch out of the divide entirely; without that
        the divide's ``-num/denom^2`` cotangent reaches ~1e76 and chains into a
        NaN.  Solstice (day 172) is the worst case because ``sin(delta) != 0``
        there, so the numerator does not vanish with the denominator.
        """
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation,
            daylight_fraction,
        )

        # Exactly at both poles plus a ring immediately inside them.
        lat = jnp.array([-jnp.pi / 2, -jnp.pi / 2 + 1e-9, -1.5,
                         0.0, 1.5, jnp.pi / 2 - 1e-9, jnp.pi / 2])

        def loss_q(l):
            return jnp.sum(daily_mean_insolation(l, 172.0) ** 2)

        def loss_f(l):
            return jnp.sum(daylight_fraction(l, 172.0) ** 2)

        g_q = jax.grad(loss_q)(lat)
        g_f = jax.grad(loss_f)(lat)
        assert_finite(g_q, "daily_mean_insolation at poles/solstice w.r.t. lat")
        assert_finite(g_f, "daylight_fraction at poles/solstice w.r.t. lat")
        # The mid-latitude entries must still carry real sensitivity: an
        # all-zero gradient would mean the pole guard swallowed the whole
        # array (finite, but useless).
        assert float(jnp.max(jnp.abs(g_q))) > 0.0, (
            "daily_mean_insolation: gradient is identically zero across the "
            "polar test ring — the AD-safety guard has swallowed the physics."
        )

    def test_grad_cos_zenith_wrt_lat_lon_and_hour(self):
        from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle

        lat = jnp.linspace(-1.2, 1.2, 12)
        lon = jnp.linspace(0.0, 2.0 * jnp.pi, 12, endpoint=False)

        def loss_lat(l):
            return jnp.sum(cos_zenith_angle(l, lon, 172.0, 12.0) ** 2)

        def loss_lon(lo):
            return jnp.sum(cos_zenith_angle(lat, lo, 172.0, 12.0) ** 2)

        def loss_hour(h):
            return jnp.sum(cos_zenith_angle(lat, lon, 172.0, h) ** 2)

        assert_gradient_ok(jax.grad(loss_lat)(lat),
                           "cos_zenith_angle w.r.t. lat", min_nonzero_frac=0.5)
        assert_gradient_ok(jax.grad(loss_lon)(lon),
                           "cos_zenith_angle w.r.t. lon", min_nonzero_frac=0.5)
        g_h = jax.grad(loss_hour)(jnp.asarray(12.0))
        assert_finite(g_h, "cos_zenith_angle w.r.t. hour")
        assert float(jnp.abs(g_h)) > 0.0, (
            "cos_zenith_angle: d/d(hour) is exactly zero — the diurnal cycle "
            "is detached."
        )

    def test_grad_earth_sun_distance_and_declination_wrt_day(self):
        """Berger (1978) orbital declination + (a/r)^2 eccentricity factor."""
        from legoesm.atmosphere.physics.radiation.solar import (
            earth_orbit,
            earth_sun_distance_factor,
            solar_declination,
        )

        orbit = earth_orbit()

        g_dist = jax.grad(
            lambda d: jnp.sum(earth_sun_distance_factor(d, orbit) ** 2)
        )(jnp.asarray(100.0))
        assert_finite(g_dist, "earth_sun_distance_factor w.r.t. day")
        assert float(jnp.abs(g_dist)) > 0.0, (
            "earth_sun_distance_factor: eccentricity is detached from day"
        )

        g_decl = jax.grad(
            lambda d: solar_declination(d, orbit=orbit) ** 2
        )(jnp.asarray(100.0))
        assert_finite(g_decl, "solar_declination w.r.t. day")
        assert float(jnp.abs(g_decl)) > 0.0, (
            "solar_declination: declination is detached from day"
        )

    def test_grad_daily_mean_insolation_wrt_S_0(self):
        """The total-solar-irradiance scaling — the natural forcing knob."""
        from legoesm.atmosphere.physics.radiation.solar import (
            daily_mean_insolation,
        )

        lat = jnp.linspace(-1.4, 1.4, 16)
        g = jax.grad(
            lambda s: jnp.sum(daily_mean_insolation(lat, 172.0, s) ** 2)
        )(jnp.asarray(constants.S_0))
        assert_finite(g, "daily_mean_insolation w.r.t. S_0")
        assert float(jnp.abs(g)) > 0.0, "S_0 unreachable by AD"


# ===========================================================================
# E.  Radiation — ozone
# ===========================================================================

class TestOzoneGrad:
    """MLS climatological ozone + the ML (ridge-regression) ozone predictor."""

    def test_mls_ozone_grad_wrt_p_full_in_table_range(self):
        """``mls_ozone_vmr`` is a log-log ``jnp.interp``: differentiable in p
        wherever p lies INSIDE the tabulated 0.0097-1054 hPa range."""
        from legoesm.atmosphere.physics.radiation.ozone_mls import mls_ozone_vmr

        # 50 hPa .. 900 hPa, comfortably inside the table.
        p_full = jnp.linspace(5.0e3, 9.0e4, 30)

        def loss(p):
            return jnp.sum(mls_ozone_vmr(p) ** 2)

        grad = jax.grad(loss)(p_full)
        assert_gradient_ok(grad, "mls_ozone_vmr w.r.t. p_full (in range)",
                           min_nonzero_frac=0.9)

    def test_mls_ozone_grad_is_exactly_zero_outside_table(self):
        """MEASURED ZERO — ``jnp.interp`` clamps to the table endpoints.

        Above 1054 hPa (and below 0.0097 hPa) the interpolant returns a
        CONSTANT endpoint value, so d(O3)/dp is identically 0 there.  This is
        the documented clamp behaviour, not a bug, but it means a column whose
        surface pressure exceeds the table top has no ozone sensitivity at its
        bottom level — worth having asserted rather than assumed.
        """
        from legoesm.atmosphere.physics.radiation.ozone_mls import mls_ozone_vmr

        p_out = jnp.linspace(1.2e5, 1.5e5, 8)   # 1200-1500 hPa, above the table

        def loss(p):
            return jnp.sum(mls_ozone_vmr(p) ** 2)

        grad = jax.grad(loss)(p_out)
        assert_exactly_zero(
            grad, "mls_ozone_vmr w.r.t. p_full (above table top)",
            reason="jnp.interp clamps to the 1053.63 hPa endpoint -> constant",
        )

    def _synthetic_ml_coefficients(self, n_lev=6, n_lat=4, n_lon=8, seed=0):
        """Build ``MLOzoneCoefficients`` directly (no NetCDF dependency)."""
        from legoesm.atmosphere.physics.radiation.ozone_ml import (
            MLOzoneCoefficients,
        )

        rng = np.random.default_rng(seed)
        coefs = np.zeros((n_lev, n_lev, n_lat, n_lon))
        for c in range(n_lev):
            coefs[c, c, :, :] = 1.0
        coefs += 1e-2 * rng.standard_normal(coefs.shape)
        z = np.arange(n_lev, dtype=np.float64)
        x_mean = np.broadcast_to((280.0 - 5.0 * z)[:, None, None],
                                 (n_lev, n_lat, n_lon)).copy()
        x_scale = np.full((n_lev, n_lat, n_lon), 5.0)
        y_mean = np.broadcast_to(
            (5e-6 * np.exp(-0.5 * ((z - n_lev // 2) / 1.5) ** 2))[:, None, None],
            (n_lev, n_lat, n_lon)).copy()
        y_scale = np.full((n_lev, n_lat, n_lon), 1e-6)
        return MLOzoneCoefficients(
            coefs=jnp.asarray(coefs),
            x_mean=jnp.asarray(x_mean),
            x_scale=jnp.asarray(x_scale),
            y_mean=jnp.asarray(y_mean),
            y_scale=jnp.asarray(y_scale),
            lat_uk=jnp.linspace(-80.0, 80.0, n_lat),
            lon_uk=jnp.linspace(0.0, 360.0, n_lon, endpoint=False),
            p_uk=jnp.linspace(100.0, 1.0e5, n_lev),
        )

    def _ml_column(self, ncol=5, nlev=8):
        T = jnp.linspace(220.0, 290.0, nlev)[None, :] * (
            1.0 + 0.02 * jnp.linspace(-1.0, 1.0, ncol)[:, None])
        lat = jnp.linspace(-1.0, 1.0, ncol)
        lon = jnp.linspace(0.3, 5.0, ncol)
        p_full = jnp.broadcast_to(
            jnp.linspace(2.0e3, 9.5e4, nlev)[None, :], (ncol, nlev))
        return T, lat, lon, p_full

    def test_predict_ozone_ml_grad_wrt_T(self):
        """The ridge predictor's documented differentiable input."""
        from legoesm.atmosphere.physics.radiation.ozone_ml import predict_ozone_ml

        coefs = self._synthetic_ml_coefficients()
        T, lat, lon, p_full = self._ml_column()

        def loss(T_in):
            return jnp.sum(predict_ozone_ml(T_in, lat, lon, p_full, coefs) ** 2)

        grad = jax.grad(loss)(T)
        assert_gradient_ok(grad, "predict_ozone_ml w.r.t. T",
                           min_nonzero_frac=0.5)

    def test_predict_ozone_ml_grad_wrt_lat_lon_is_nonzero(self):
        """MEASURED, and it CONTRADICTS the module docstring.

        ``predict_ozone_ml``'s docstring states "Lat/lon enter only as gather
        indices and are treated as static for AD purposes".  That is only half
        true: ``_bilinear_weights_1d`` returns a CONTINUOUS interpolation weight
        ``w_hi = (t - x0)/(x1 - x0)`` which depends smoothly on the target
        latitude/longitude, and that weight multiplies the gathered
        coefficients.  So the horizontal position IS reachable by AD (only the
        integer corner indices are not).

        This is not a bug — the code is MORE differentiable than advertised —
        but the docstring would mislead anyone deciding whether a
        column-position perturbation propagates.  Asserted here so the real
        behaviour is pinned; if the interpolation is ever replaced by a
        nearest-neighbour gather this goes red rather than silently changing
        the adjoint.
        """
        from legoesm.atmosphere.physics.radiation.ozone_ml import predict_ozone_ml

        coefs = self._synthetic_ml_coefficients()
        T, lat, lon, p_full = self._ml_column()

        def loss_lat(l):
            return jnp.sum(predict_ozone_ml(T, l, lon, p_full, coefs) ** 2)

        def loss_lon(lo):
            return jnp.sum(predict_ozone_ml(T, lat, lo, p_full, coefs) ** 2)

        g_lat = jax.grad(loss_lat)(lat)
        g_lon = jax.grad(loss_lon)(lon)
        assert_finite(g_lat, "predict_ozone_ml w.r.t. lat")
        assert_finite(g_lon, "predict_ozone_ml w.r.t. lon")
        assert float(jnp.max(jnp.abs(g_lat))) > 0.0, (
            "predict_ozone_ml: d/d(lat) measured EXACTLY zero — the bilinear "
            "weight has become a nearest-neighbour gather; the horizontal "
            "position is now genuinely non-differentiable."
        )
        assert float(jnp.max(jnp.abs(g_lon))) > 0.0, (
            "predict_ozone_ml: d/d(lon) measured EXACTLY zero (see above)."
        )

    def test_predict_ozone_ml_grad_wrt_p_full(self):
        """Pressure drives the two log-p vertical interpolations (model->UKESM
        levels and back), so the model's vertical grid is differentiable."""
        from legoesm.atmosphere.physics.radiation.ozone_ml import predict_ozone_ml

        coefs = self._synthetic_ml_coefficients()
        T, lat, lon, p_full = self._ml_column()

        def loss(p):
            return jnp.sum(predict_ozone_ml(T, lat, lon, p, coefs) ** 2)

        grad = jax.grad(loss)(p_full)
        assert_gradient_ok(grad, "predict_ozone_ml w.r.t. p_full",
                           min_nonzero_frac=0.3)


# ===========================================================================
# F.  Radiation — RRTMGP correlated-k
# ===========================================================================

def _rrtmgp_column(ncol=3, nlev=24):
    """Column inside the RRTMGP table's valid pressure/temperature range.

    Pressure INCREASES with index (100 Pa top -> 1e5 Pa surface) — the same
    convention the rest of this file uses, and the one whose inversion made a
    prior gray-radiation test vacuous.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = (jnp.linspace(200.0, 295.0, nlev)[None, :]
         * (1.0 + 0.01 * jnp.linspace(-1.0, 1.0, ncol)[:, None]))
    T_sfc = 295.0 + 2.0 * jnp.linspace(-1.0, 1.0, ncol)
    p_norm = p_full / p_full[:, -1:]
    q_v = 0.015 * p_norm ** 4
    cos_zen = jnp.linspace(0.3, 0.9, ncol)
    return T, p_full, p_half, T_sfc, q_v, cos_zen


class TestRRTMGPGrad:
    """RRTMGP correlated-k radiation.

    The existing sweep documents the absorption-table lookups as
    non-differentiable by design; that is NOT contested here.  What is measured
    is precisely WHICH inputs still carry gradient through the interpolation
    that sits on top of those tables.  The k-distribution is interpolated
    (bi-/tri-linearly) in (p, T, mixing ratio), so the continuous interpolation
    weights are differentiable even though the integer table CORNERS are not —
    the same structural distinction as the ozone-ML bilinear weights above.
    Every claim below is a measured number, not an assumption.
    """

    def test_grad_heating_rate_wrt_T(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()

        def loss(T_in):
            out = rrtmgp_radiation(T_in, p_full, p_half, T_sfc, q_v, cos_zen,
                                   cfg)
            return jnp.sum(out.heating_rate ** 2)

        grad = jax.grad(loss)(T)
        assert_gradient_ok(grad, "RRTMGP heating_rate w.r.t. T",
                           min_nonzero_frac=0.8)

    def test_grad_lw_flux_wrt_q_v(self):
        """Water vapour is the dominant LW absorber; its VMR is a continuous
        interpolation coordinate in the gas-optics table."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()

        def loss(q):
            out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q, cos_zen, cfg)
            return jnp.sum(out.lw_flux_up ** 2)

        grad = jax.grad(loss)(q_v)
        assert_gradient_ok(grad, "RRTMGP lw_flux_up w.r.t. q_v",
                           min_nonzero_frac=0.5)

    def test_grad_wrt_sfc_temperature(self):
        """Surface emission — the coupled surface->TOA adjoint link."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()

        def loss(ts):
            out = rrtmgp_radiation(T, p_full, p_half, ts, q_v, cos_zen, cfg)
            return jnp.sum(out.lw_flux_up ** 2)

        grad = jax.grad(loss)(T_sfc)
        assert_gradient_ok(grad, "RRTMGP lw_flux_up w.r.t. sfc_temperature",
                           min_nonzero_frac=0.9)

    def test_grad_sw_wrt_cos_zenith(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()

        def loss(cz):
            out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cz, cfg)
            return jnp.sum(out.sw_flux_down ** 2)

        grad = jax.grad(loss)(cos_zen)
        assert_gradient_ok(grad, "RRTMGP sw_flux_down w.r.t. cos_zenith",
                           min_nonzero_frac=0.9)

    def test_grad_wrt_surface_albedo_and_emissivity(self):
        """Surface optical properties — tier-1 tunables in the param spec."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()
        ncol = T.shape[0]
        albedo = jnp.full((ncol,), 0.15)
        emiss = jnp.full((ncol,), 0.97)

        g_alb = jax.grad(lambda a: jnp.sum(rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg,
            sfc_albedo_override=a).sw_flux_up ** 2))(albedo)
        assert_gradient_ok(g_alb, "RRTMGP sw_flux_up w.r.t. sfc_albedo",
                           min_nonzero_frac=0.9)

        g_eps = jax.grad(lambda e: jnp.sum(rrtmgp_radiation(
            T, p_full, p_half, T_sfc, q_v, cos_zen, cfg,
            sfc_emissivity_override=e).lw_flux_up ** 2))(emiss)
        assert_gradient_ok(g_eps, "RRTMGP lw_flux_up w.r.t. sfc_emissivity",
                           min_nonzero_frac=0.9)

    def test_grad_wrt_o3_vmr(self):
        """Prescribed ozone drives the SW stratospheric heating; this is the
        link that makes the ozone modules above matter to the radiation."""
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()
        o3 = jnp.full(p_full.shape, 5.0e-7)

        def loss(o):
            out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cos_zen,
                                   cfg, o3_vmr=o)
            return jnp.sum(out.sw_heating_rate ** 2)

        grad = jax.grad(loss)(o3)
        assert_gradient_ok(grad, "RRTMGP sw_heating_rate w.r.t. o3_vmr",
                           min_nonzero_frac=0.5)

    def test_grad_wrt_cloud_properties(self):
        """Cloud water paths / effective radii / fraction.

        These enter the CLOUD optics tables (a separate lookup from the gas
        k-distribution).  Measuring them separately answers the question the
        task poses directly: cloud properties are interpolated, not indexed, so
        they remain differentiable.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        # Smaller column than the clear-sky tests: the cloudy RRTMGP path is
        # the one ``cloud_optics.py`` warns "tips the executable over the
        # XLA-CPU LLVM-JIT code-region limit", and it dominates this file's
        # runtime.  All four cloud inputs are differentiated in ONE backward
        # pass (argnums tuple) rather than four separate ``jax.grad`` closures,
        # which would each trace and compile the cloudy solver again.
        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column(ncol=2, nlev=16)
        cfg = RRTMGPConfig(include_clouds=True)
        shape = p_full.shape
        nlev = shape[1]
        # A mid-level liquid deck and a high ice deck.
        band = lambda lo, hi, val: jnp.where(  # noqa: E731
            (jnp.arange(nlev) >= lo) & (jnp.arange(nlev) < hi), val, 0.0
        )[None, :] * jnp.ones((shape[0], 1))
        cwp_liq = band(nlev // 2, nlev // 2 + 4, 0.05)      # kg/m^2
        cwp_ice = band(nlev // 5, nlev // 5 + 3, 0.02)      # kg/m^2
        # UNITS ARE LOAD-BEARING: ``rrtmgp.py`` declares cloud_r_eff_* in
        # METERS and ``cloud_optics.compute_optical_properties`` multiplies by
        # 1e6 before indexing the table, then CLIPS to
        # [radius_liq_lower, radius_liq_upper] (~2.5-21 um).  Passing 10.0
        # (i.e. 10 metres) lands 1e6 beyond the upper bound, the clip
        # saturates, and d(flux)/d(r_eff) is EXACTLY 0 — which is how the first
        # version of this test measured max|g| = 0.0 and nearly got written up
        # as "effective radius is non-differentiable by table lookup".  It is
        # not: with physical values the interpolant is perfectly differentiable.
        r_liq = jnp.full(shape, 1.0e-5)                     # 10 um
        r_ice = jnp.full(shape, 2.0e-5)                     # 20 um radius
        cf = band(nlev // 5, nlev // 2 + 4, 0.8)

        def loss(lwp, iwp, reff_l, cloud_frac):
            out = rrtmgp_radiation(
                T, p_full, p_half, T_sfc, q_v, cos_zen, cfg,
                cloud_path_liq=lwp, cloud_path_ice=iwp,
                cloud_r_eff_liq=reff_l, cloud_r_eff_ice=r_ice,
                cloud_fraction=cloud_frac,
            )
            # Both SW and LW so an ice-only or liquid-only path cannot hide.
            return jnp.sum(out.sw_flux_up ** 2) + jnp.sum(out.lw_flux_up ** 2)

        g_lwp, g_iwp, g_reff, g_cf = jax.grad(loss, argnums=(0, 1, 2, 3))(
            cwp_liq, cwp_ice, r_liq, cf)

        assert_gradient_ok(g_lwp, "RRTMGP flux w.r.t. cloud_path_liq",
                           min_nonzero_frac=0.05)
        assert_gradient_ok(g_iwp, "RRTMGP flux w.r.t. cloud_path_ice",
                           min_nonzero_frac=0.02)
        assert_gradient_ok(g_reff, "RRTMGP flux w.r.t. cloud_r_eff_liq",
                           min_nonzero_frac=0.02)
        assert_gradient_ok(g_cf, "RRTMGP flux w.r.t. cloud_fraction",
                           min_nonzero_frac=0.05)

        # Companion MEASURED ZERO, reusing the SAME compiled backward pass
        # (identical shapes/dtypes -> JAX cache hit): outside the tabulated
        # particle-size range the clip saturates and r_eff sensitivity vanishes
        # exactly.  Worth pinning because the failure is silent — a column
        # supplied with r_eff in MICRONS (a natural mistake; the public
        # argument is in METRES, rrtmgp.py:50) sits 1e6 past the upper bound
        # and contributes no gradient at all while still returning perfectly
        # plausible fluxes.
        r_liq_out_of_range = jnp.full(shape, 10.0)          # 10 m, not 10 um
        _l, _i, g_clip, _c = jax.grad(loss, argnums=(0, 1, 2, 3))(
            cwp_liq, cwp_ice, r_liq_out_of_range, cf)
        assert_exactly_zero(
            g_clip, "RRTMGP flux w.r.t. cloud_r_eff_liq (out of range)",
            reason="clip to [radius_liq_lower, radius_liq_upper] saturates",
        )

    def test_grad_wrt_aerosol_optical_depth(self):
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()
        aod = jnp.full(p_full.shape, 0.02)

        def loss(a):
            out = rrtmgp_radiation(T, p_full, p_half, T_sfc, q_v, cos_zen,
                                   cfg, aerosol_optical_depth=a)
            return jnp.sum(out.sw_flux_up ** 2)

        grad = jax.grad(loss)(aod)
        assert_gradient_ok(grad, "RRTMGP sw_flux_up w.r.t. aerosol_optical_depth",
                           min_nonzero_frac=0.5)

    def test_grad_wrt_p_full_is_measured(self):
        """Pressure is BOTH a table interpolation coordinate AND the layer-mass
        weight in the heating-rate conversion.

        This is the input most likely to be genuinely index-only, so it is the
        sharpest test of the "which inputs survive the table lookup" question.
        The result is asserted as a measured non-zero rather than assumed; a
        zero here would be the documented "p only indexes the table" outcome
        and would need this expectation flipped deliberately.
        """
        from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        T, p_full, p_half, T_sfc, q_v, cos_zen = _rrtmgp_column()
        cfg = RRTMGPConfig()

        def loss(p):
            out = rrtmgp_radiation(T, p, p_half, T_sfc, q_v, cos_zen, cfg)
            return jnp.sum(out.heating_rate ** 2)

        grad = jax.grad(loss)(p_full)
        assert_finite(grad, "RRTMGP heating_rate w.r.t. p_full")
        frac = nonzero_frac(grad)
        assert frac > 0.0, (
            "RRTMGP: d(heating_rate)/d(p_full) measured EXACTLY zero across "
            "all levels — pressure reaches the solver only as a table index, "
            "so no pressure-perturbation experiment can be differentiated. "
            "Documented finding: flip this expectation to assert_exactly_zero."
        )
        assert frac >= 0.5, (
            f"RRTMGP: only {frac * 100:.1f}% of d(heating_rate)/d(p_full) "
            "entries are non-zero"
        )
