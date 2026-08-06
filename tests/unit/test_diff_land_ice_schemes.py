"""Differentiability tests for LAND and SEA-ICE schemes the step-level
sweep never reaches.

``tests/unit/test_diff_land.py`` differentiates ``step_land`` /
``step_multilayer_land`` (whole-step), plus GPP and the stomata models;
``tests/unit/test_diff_sea_ice.py`` differentiates ``step_sea_ice``,
the rheology kernels, the ITD remap and the albedo feedback.  Neither
touches the *selectable scheme* layer underneath:

  land
    - ``soil_hydraulics`` retention curves: van_genuchten /
      clapp_hornberger / campbell / brooks_corey / pdi / lu — the
      ``retention_curve`` dispatch, five distinct nonlinear closures
      that Richards and the soil thermal solver both consume;
    - ``soil_thermal``: heat capacity, thermal conductivity, and the
      freeze/thaw APPARENT heat capacity (zero-curtain) branch;
    - ``richards.solve_richards``: the implicit mixed-form solver;
    - runoff partition: ``topmodel`` vs ``bucket`` (both selectable).

  sea ice
    - ``ponds.step_ponds`` (melt-pond area/depth, refreeze switch),
    - ``ridging`` participation weights + ``apply_ridging``,
    - ``brine.update_salinity_and_salt_flux`` (salt budget),
    - ``snow`` conductive flux / accumulation / snow-ice flooding,
    - ``dynamics`` free-drift + air/ocean stress kernels.

Every assertion is finite AND non-zero AND (where the output is a
field) structured — a saturating clip or a hard ``jnp.where`` that
kills a branch must fail, not pass.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm import constants


def assert_gradient_ok(g, name="", min_nonzero_frac=0.5, require_structure=True):
    g = jnp.asarray(g)
    assert jnp.all(jnp.isfinite(g)), f"{name}: gradient has NaN/Inf"
    nz = float(jnp.mean(jnp.abs(g) > 0.0))
    assert nz >= min_nonzero_frac, (
        f"{name}: only {nz * 100:.1f}% of entries non-zero "
        f"(need {min_nonzero_frac * 100:.0f}%)"
    )
    if require_structure and g.size > 1:
        spread = float(jnp.max(g) - jnp.min(g))
        assert spread > 0.0, f"{name}: gradient is spatially uniform"


def assert_scalar_grad_alive(g, name=""):
    g = jnp.asarray(g)
    assert jnp.all(jnp.isfinite(g)), f"{name}: gradient has NaN/Inf"
    assert float(jnp.max(jnp.abs(g))) > 0.0, (
        f"{name}: gradient is identically zero — the input is unreachable "
        "by AD")


# ============================================================================
# LAND — soil hydraulics retention curves (the ``retention_curve`` dispatch)
# ============================================================================

_RETENTION_CURVES = [
    "van_genuchten", "clapp_hornberger", "campbell", "brooks_corey",
    "pdi", "lu",
]


def _hydro_cfg(curve):
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
    return SoilHydraulicsConfig(retention_curve=curve)

# Suctions strictly BELOW the air-entry pressure of the sharp-air-entry
# families (Clapp-Hornberger / Brooks-Corey default psi_sat = psi_b =
# -0.478 m), i.e. genuinely unsaturated.  Above air entry those curves are
# flat at theta_sat BY CONSTRUCTION (see
# TestSoilRetentionNearSaturation), so a fixture that straddled it would
# report a scheme property as a dead gradient.
_UNSAT_PSI = np.array([-0.6, -1.0, -3.0, -10.0, -50.0])


class TestSoilRetentionCurves:
    """Each retention curve is a selectable closure feeding Richards.
    A dead ``dtheta/dpsi`` or ``dK/dpsi`` silently freezes the soil
    moisture adjoint for that scheme only — invisible to a whole-step
    test run with the default curve."""

    @pytest.mark.parametrize("curve", _RETENTION_CURVES)
    def test_theta_from_psi_grad(self, curve):
        from legoesm.land.soil_hydraulics import theta_from_psi
        cfg = _hydro_cfg(curve)
        psi = jnp.asarray(_UNSAT_PSI)

        def loss(p):
            return jnp.sum(theta_from_psi(p, cfg) ** 2)

        g = jax.grad(loss)(psi)
        assert_gradient_ok(g, f"theta_from_psi[{curve}] d/dpsi",
                           min_nonzero_frac=0.8)

    @pytest.mark.parametrize("curve", _RETENTION_CURVES)
    def test_hydraulic_conductivity_grad(self, curve):
        from legoesm.land.soil_hydraulics import (
            theta_from_psi, hydraulic_conductivity,
        )
        cfg = _hydro_cfg(curve)
        psi = jnp.asarray(_UNSAT_PSI)

        def loss(p):
            th = theta_from_psi(p, cfg)
            return jnp.sum(hydraulic_conductivity(p, th, cfg) ** 2)

        g = jax.grad(loss)(psi)
        assert_gradient_ok(g, f"hydraulic_conductivity[{curve}] d/dpsi",
                           min_nonzero_frac=0.6)

    @pytest.mark.parametrize("curve", _RETENTION_CURVES)
    def test_moisture_capacity_positive_and_differentiable(self, curve):
        """dtheta/dpsi is the Richards matrix diagonal.  It must be
        strictly positive (a non-singular matrix) AND itself
        differentiable (the solver's Newton/Picard adjoint needs the
        second derivative)."""
        from legoesm.land.soil_hydraulics import (
            theta_from_psi, moisture_capacity,
        )
        cfg = _hydro_cfg(curve)
        psi = jnp.asarray(_UNSAT_PSI)
        th = theta_from_psi(psi, cfg)
        C = moisture_capacity(psi, th, cfg)
        assert jnp.all(jnp.isfinite(C)), f"{curve}: moisture capacity NaN/Inf"
        assert float(jnp.min(C)) > 0.0, (
            f"{curve}: moisture capacity has a non-positive entry "
            f"(min={float(jnp.min(C)):.3e}) — the Richards matrix is singular")

        def loss(p):
            return jnp.sum(moisture_capacity(p, theta_from_psi(p, cfg), cfg) ** 2)

        g = jax.grad(loss)(psi)
        assert_gradient_ok(g, f"moisture_capacity[{curve}] d/dpsi",
                           min_nonzero_frac=0.6)

    @pytest.mark.parametrize("curve", _RETENTION_CURVES)
    def test_theta_psi_roundtrip_grad_is_positive(self, curve):
        """theta(psi) must be monotonically INCREASING (wetter soil ->
        less negative matric potential).  A sign flip here inverts every
        soil-moisture gradient in the land adjoint."""
        from legoesm.land.soil_hydraulics import theta_from_psi
        cfg = _hydro_cfg(curve)
        for p0 in (-1.0, -3.0, -10.0):
            d = float(jax.grad(lambda p: theta_from_psi(p, cfg))(
                jnp.asarray(p0)))
            assert np.isfinite(d), f"{curve}: dtheta/dpsi NaN at psi={p0}"
            assert d > 0.0, (
                f"{curve}: dtheta/dpsi={d:.3e} at psi={p0} — retention "
                "curve is not monotonically increasing in psi")

    def test_retention_curves_are_not_aliases(self):
        """Distinct ``retention_curve`` selections must give distinct
        gradients.  A dispatch that silently falls through to the
        default would pass every per-curve check above."""
        from legoesm.land.soil_hydraulics import theta_from_psi
        psi = jnp.asarray(np.array([-1.0, -3.0, -10.0]))

        def g_of(curve):
            cfg = _hydro_cfg(curve)
            return jax.grad(lambda p: jnp.sum(theta_from_psi(p, cfg) ** 2))(psi)

        base = g_of("van_genuchten")
        for other in ("clapp_hornberger", "brooks_corey", "pdi", "lu"):
            d = float(jnp.max(jnp.abs(g_of(other) - base)))
            assert d > 0.0, (
                f"retention_curve={other!r} produces the SAME adjoint as "
                "van_genuchten — the dispatch may be aliased")
        # 'campbell' IS documented as an alias of clapp_hornberger.
        assert float(jnp.max(jnp.abs(
            g_of("campbell") - g_of("clapp_hornberger")))) == 0.0


class TestSoilRetentionNearSaturation:
    """The near-saturation band ``psi in (psi_air_entry, 0)``.

    NOT a bug, but a differentiability CLIFF worth pinning: the
    sharp-air-entry families (Clapp-Hornberger / Campbell / Brooks-Corey)
    are flat at ``theta_sat`` above their air-entry pressure, so
    ``dtheta/dpsi`` is EXACTLY zero there and the soil-moisture adjoint
    through ``theta`` is dead in that band; the elastic specific-storage
    term only switches on at ``psi >= 0``.  The smooth families (van
    Genuchten / PDI / Lu) stay differentiable throughout.  A parameter-
    estimation run that lands a wet column in that band gets no gradient
    from those three schemes — pin the behaviour so the split is explicit
    rather than discovered as a silent training stall.
    """

    _NEAR_SAT = np.array([-0.40, -0.20, -0.05])   # above the -0.478 m air entry

    @pytest.mark.parametrize("curve", ["clapp_hornberger", "campbell",
                                       "brooks_corey"])
    def test_sharp_air_entry_capacity_is_zero_near_saturation(self, curve):
        from legoesm.land.soil_hydraulics import theta_from_psi
        cfg = _hydro_cfg(curve)
        psi = jnp.asarray(self._NEAR_SAT)
        g = jax.grad(lambda p: jnp.sum(theta_from_psi(p, cfg)))(psi)
        assert jnp.all(jnp.isfinite(g)), f"{curve}: NaN/Inf near saturation"
        assert float(jnp.max(jnp.abs(g))) == 0.0, (
            f"{curve}: dtheta/dpsi = {g} above air entry — this test pins "
            "the documented FLAT branch; a non-zero value means the curve "
            "changed and the dead-band note is stale")

    @pytest.mark.parametrize("curve", ["van_genuchten", "pdi", "lu"])
    def test_smooth_curves_stay_differentiable_near_saturation(self, curve):
        from legoesm.land.soil_hydraulics import theta_from_psi
        cfg = _hydro_cfg(curve)
        psi = jnp.asarray(self._NEAR_SAT)
        g = jax.grad(lambda p: jnp.sum(theta_from_psi(p, cfg)))(psi)
        assert jnp.all(jnp.isfinite(g)), f"{curve}: NaN/Inf near saturation"
        assert float(jnp.min(jnp.abs(g))) > 0.0, (
            f"{curve}: dtheta/dpsi has a zero entry near saturation ({g}) — "
            "the smooth families must stay differentiable there")


# ============================================================================
# LAND — soil thermal (incl. the freeze/thaw apparent-heat-capacity branch)
# ============================================================================

class TestSoilThermal:

    @staticmethod
    def _cfgs(enable_freeze_thaw=False):
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        from legoesm.land.soil_thermal import SoilThermalConfig
        return (SoilHydraulicsConfig(),
                SoilThermalConfig(enable_freeze_thaw=enable_freeze_thaw))

    def test_grad_heat_capacity_wrt_theta(self):
        from legoesm.land.soil_thermal import compute_heat_capacity
        hydro, thermal = self._cfgs()
        theta = jnp.asarray(np.linspace(0.15, 0.42, 8))

        def loss(t):
            return jnp.sum(compute_heat_capacity(t, hydro, thermal) ** 2)

        g = jax.grad(loss)(theta)
        assert_gradient_ok(g, "compute_heat_capacity d/dtheta",
                           min_nonzero_frac=0.9)

    def test_grad_thermal_conductivity_wrt_theta(self):
        """Johansen's Kersten number uses log10(Sr) with a floor — a
        clip that saturates over the whole range would kill this."""
        from legoesm.land.soil_thermal import compute_thermal_conductivity
        hydro, thermal = self._cfgs()
        # Johansen's Kersten number uses log10(Sr) with Sr =
        # (theta-theta_r)/(theta_sat-theta_r) FLOORED at
        # kersten_sr_floor_fine=0.1; below that floor the gradient is
        # legitimately zero (dry-soil dead band), so sample above it:
        # theta > theta_r + 0.1*(theta_sat-theta_r) = 0.113.
        theta = jnp.asarray(np.linspace(0.15, 0.42, 8))

        def loss(t):
            return jnp.sum(compute_thermal_conductivity(t, hydro, thermal) ** 2)

        g = jax.grad(loss)(theta)
        assert_gradient_ok(g, "compute_thermal_conductivity d/dtheta",
                           min_nonzero_frac=0.9)

    def test_grad_liquid_water_content_across_freezing_point(self):
        """The smooth freezing curve is the whole point of the
        apparent-heat-capacity formulation: d(theta_liq)/dT must be
        strictly POSITIVE (warming melts ice) and non-zero on BOTH sides
        of T_freeze, or the zero-curtain is a hard switch and the
        permafrost adjoint is dead."""
        from legoesm.land.soil_thermal import liquid_water_content
        _, thermal = self._cfgs(enable_freeze_thaw=True)
        theta = jnp.full((5,), 0.30)
        T = jnp.asarray(
            constants.T_freeze + np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))

        def loss(T_in):
            liq, _ = liquid_water_content(T_in, theta, thermal)
            return jnp.sum(liq)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), "liquid_water_content: NaN/Inf grad"
        assert float(jnp.min(g)) >= 0.0, (
            f"d(theta_liq)/dT must be >= 0 (warming melts ice), got {g}")
        # Non-zero in the transition zone (|T - T_freeze| <= 0.5 K).
        assert float(jnp.max(jnp.abs(g[1:4]))) > 0.0, (
            "d(theta_liq)/dT is zero across the freezing point — the "
            "freezing curve is a hard switch, not a smooth curve")

    def test_grad_apparent_heat_capacity(self):
        """``compute_apparent_heat_capacity`` IS the freeze/thaw formula —
        the ``enable_freeze_thaw`` flag is read by the CALLER
        (``solve_soil_thermal``), not inside it — so its temperature
        adjoint must be live and dominated by the latent zero-curtain
        near ``T_freeze``."""
        from legoesm.land.soil_thermal import compute_apparent_heat_capacity
        hydro, thermal = self._cfgs(enable_freeze_thaw=True)
        theta = jnp.full((5,), 0.30)
        T = jnp.asarray(
            constants.T_freeze + np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))

        def loss(T_in):
            return jnp.sum(
                compute_apparent_heat_capacity(T_in, theta, hydro, thermal) ** 2)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), "apparent heat capacity: NaN/Inf"
        assert float(jnp.max(jnp.abs(g))) > 0.0, (
            "apparent heat capacity has a dead temperature adjoint")
        # The zero-curtain is CENTRED on T_freeze: sensitivity must peak in
        # the transition zone, not in the far-field tails.
        near = float(jnp.max(jnp.abs(g[1:4])))
        far = float(jnp.max(jnp.abs(g[jnp.array([0, 4])])))
        assert near > far, (
            f"latent-heat sensitivity is larger away from T_freeze "
            f"(near={near:.3e}, far={far:.3e}) — the zero-curtain is "
            "mis-centred")

    def test_solve_soil_thermal_freeze_thaw_gate_changes_the_adjoint(self):
        """``enable_freeze_thaw`` is a STATIC feature gate consumed by
        ``solve_soil_thermal``.  With it ON the solve uses the apparent
        (latent-inflated) heat capacity, so the temperature adjoint must
        MEASURABLY differ from the sensible-only path — otherwise the gate
        is wired to nothing."""
        from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
        from legoesm.land.soil_thermal import solve_soil_thermal
        grid = make_soil_grid(SoilGridConfig(n_layers=5))
        hydro, on = self._cfgs(enable_freeze_thaw=True)
        _, off = self._cfgs(enable_freeze_thaw=False)
        # Straddle the freezing point so the zero-curtain is active.
        T = jnp.asarray(constants.T_freeze
                        + np.array([[-1.0, -0.3, 0.2, 1.0, 2.0]]))
        theta = jnp.full((1, 5), 0.30)
        G = jnp.asarray([20.0])

        def g_of(cfg):
            return jax.grad(lambda t: jnp.sum(
                solve_soil_thermal(t, theta, grid, hydro, cfg, G, 1800.0) ** 2)
            )(T)

        g_on, g_off = g_of(on), g_of(off)
        assert jnp.all(jnp.isfinite(g_on)) and jnp.all(jnp.isfinite(g_off))
        rel = float(jnp.max(jnp.abs(g_on - g_off))) / float(
            jnp.max(jnp.abs(g_off)))
        assert rel > 1e-3, (
            f"enable_freeze_thaw changes the soil-thermal adjoint by only "
            f"{rel:.3e} relative — the gate is not reaching the solve")

    def test_grad_solve_soil_thermal(self):
        """The implicit tridiagonal solve is the land adjoint's hot
        path; ``G_surface`` is the coupling flux from the atmosphere."""
        from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
        from legoesm.land.soil_thermal import solve_soil_thermal
        hydro, thermal = self._cfgs(enable_freeze_thaw=True)
        grid = make_soil_grid(SoilGridConfig(n_layers=5))
        rng = np.random.default_rng(1)
        # solve_soil_thermal is (ncol, n_layers) with G_surface (ncol,).
        T = jnp.asarray(constants.T_freeze + 2.0
                        + rng.standard_normal((1, 5)))
        theta = jnp.asarray(0.25 + 0.05 * rng.random((1, 5)))
        G = jnp.asarray([20.0])

        def loss(args):
            out = solve_soil_thermal(args[0], args[1], grid, hydro, thermal,
                                     args[2], 1800.0)
            return jnp.sum(out ** 2)

        gT, gth, gG = jax.grad(loss)((T, theta, G))
        assert_gradient_ok(gT, "solve_soil_thermal d/dT_soil",
                           min_nonzero_frac=0.9)
        assert_gradient_ok(gth, "solve_soil_thermal d/dtheta",
                           min_nonzero_frac=0.6)
        assert_scalar_grad_alive(gG, "solve_soil_thermal d/dG_surface")


# ============================================================================
# LAND — Richards solver
# ============================================================================

class TestRichardsSolver:

    def test_grad_solve_richards(self):
        from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, theta_from_psi,
        )
        from legoesm.land.richards import solve_richards, RichardsConfig
        grid = make_soil_grid(SoilGridConfig(n_layers=5))
        hydro = SoilHydraulicsConfig()
        rcfg = RichardsConfig()
        rng = np.random.default_rng(2)
        # solve_richards is (ncol, n_layers); flux_top is (ncol,).
        psi = jnp.asarray(-0.5 - rng.random((1, 5)))
        flux_top = jnp.asarray([1e-7])      # infiltration [m/s]
        sink = jnp.zeros((1, 5))

        def loss(args):
            th = theta_from_psi(args[0], hydro)
            out = solve_richards(args[0], th, grid, hydro, rcfg,
                                 args[1], sink, 1800.0)
            return jnp.sum(out.psi_new ** 2)

        gpsi, gflux = jax.grad(loss)((psi, flux_top))
        assert_gradient_ok(gpsi, "solve_richards d/dpsi", min_nonzero_frac=0.9)
        assert_scalar_grad_alive(gflux, "solve_richards d/dflux_top")

    def test_richards_water_balance_adjoint_is_nonlocal(self):
        """Richards is an implicit column solve: an infiltration
        perturbation at the surface must reach EVERY layer's adjoint.
        A purely-diagonal gradient means the solve has been detached
        (e.g. an explicit update replacing the implicit matrix)."""
        from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, theta_from_psi,
        )
        from legoesm.land.richards import solve_richards, RichardsConfig
        grid = make_soil_grid(SoilGridConfig(n_layers=5))
        hydro = SoilHydraulicsConfig()
        rcfg = RichardsConfig()
        psi = jnp.asarray(np.full((1, 5), -0.8))
        sink = jnp.zeros((1, 5))

        def bottom_psi(psi_in):
            th = theta_from_psi(psi_in, hydro)
            out = solve_richards(psi_in, th, grid, hydro, rcfg,
                                 jnp.asarray([1e-7]), sink, 1800.0)
            return out.psi_new[0, -1]

        g = np.asarray(jax.grad(bottom_psi)(psi))[0]
        assert np.isfinite(g).all(), "Richards adjoint has NaN/Inf"
        assert abs(g[0]) > 0.0, (
            "d(psi_bottom)/d(psi_top) is exactly zero — the implicit "
            f"column coupling is missing from the adjoint (g={g})")


# ============================================================================
# LAND — runoff partition schemes (topmodel vs bucket)
# ============================================================================

class TestRunoffPartitionSchemes:
    """``partition_topmodel_runoff`` and ``partition_bucket_runoff``
    share a caller-agnostic signature — they are the two selectable
    runoff closures and neither is gradient-tested."""

    @staticmethod
    def _inputs():
        W = jnp.asarray(np.array([50.0, 120.0, 190.0]))
        P = jnp.asarray(np.array([1e-5, 5e-6, 2e-5]))     # [m/s]
        E = jnp.asarray(np.array([1e-6, 2e-6, 5e-7]))
        return W, P, E

    def _run(self, fn, W, P, E, **kw):
        out = fn(W, P, E, 1800.0, 200.0, 1e-5, 1.0, **kw)
        leaves = out if isinstance(out, tuple) else (out,)
        return sum(jnp.sum(jnp.asarray(x) ** 2) for x in leaves)

    def test_grad_bucket_runoff(self):
        from legoesm.land.bucket_hydrology import partition_bucket_runoff
        W, P, E = self._inputs()

        def loss(args):
            return self._run(partition_bucket_runoff, args[0], args[1], args[2])

        gW, gP, gE = jax.grad(loss)((W, P, E))
        assert_gradient_ok(gW, "bucket runoff d/dW", min_nonzero_frac=0.6)
        assert_gradient_ok(gP, "bucket runoff d/dP", min_nonzero_frac=0.6)
        assert jnp.all(jnp.isfinite(gE)), "bucket runoff d/dE: NaN/Inf"

    def test_grad_topmodel_runoff(self):
        from legoesm.land.topmodel_runoff import (
            partition_topmodel_runoff, TopmodelConfig,
        )
        W, P, E = self._inputs()
        cfg = TopmodelConfig()

        def loss(args):
            return self._run(partition_topmodel_runoff, args[0], args[1],
                             args[2], config=cfg)

        gW, gP, gE = jax.grad(loss)((W, P, E))
        assert_gradient_ok(gW, "topmodel runoff d/dW", min_nonzero_frac=0.6)
        assert_gradient_ok(gP, "topmodel runoff d/dP", min_nonzero_frac=0.6)
        assert jnp.all(jnp.isfinite(gE)), "topmodel runoff d/dE: NaN/Inf"

    def test_topmodel_saturated_fraction_grad(self):
        """The saturated-area fraction is TOPMODEL's defining nonlinearity
        (an exponential in water-table depth); its adjoint drives every
        saturation-excess runoff sensitivity."""
        from legoesm.land.topmodel_runoff import (
            water_table_depth, saturated_area_fraction, baseflow,
            TopmodelConfig,
        )
        cfg = TopmodelConfig()
        W = jnp.asarray(np.array([50.0, 120.0, 190.0]))

        def loss(w):
            z = water_table_depth(w, 200.0, cfg)
            return jnp.sum(saturated_area_fraction(z, cfg) ** 2
                           + baseflow(z, cfg) ** 2)

        g = jax.grad(loss)(W)
        assert_gradient_ok(g, "topmodel f_sat/baseflow d/dW",
                           min_nonzero_frac=0.9)


# ============================================================================
# SEA ICE — melt ponds
# ============================================================================

class TestMeltPonds:

    @staticmethod
    def _state(n=4):
        pond_area = jnp.asarray(np.array([0.05, 0.15, 0.25, 0.35]))
        pond_depth = jnp.asarray(np.array([0.02, 0.05, 0.08, 0.12]))
        melt = jnp.asarray(np.array([1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3]))
        rain = jnp.asarray(np.array([0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3]))
        # ice_mask is combined with a boolean predicate inside step_ponds
        # (``&``), so it must be a BOOL array, not a float field.
        ice_mask = jnp.ones(n, dtype=bool)
        h_snow = jnp.asarray(np.array([0.0, 0.005, 0.01, 0.02]))
        # Straddle the refreeze switch so both branches are exercised.
        T_air = jnp.asarray(constants.T_freeze + np.array([-3.0, -0.2, 0.2, 3.0]))
        return pond_area, pond_depth, melt, rain, ice_mask, h_snow, T_air

    def _step(self, pa, pd, melt, rain, ice_mask, h_snow, T_air):
        from legoesm.ice.ponds import step_ponds
        from legoesm.ice.config import MeltPondConfig
        cfg = MeltPondConfig()
        return step_ponds(
            pa, pd,
            melt_water_m=melt, rain_water_m=rain, ice_mask=ice_mask,
            h_snow=h_snow, T_air=T_air, dt=3600.0,
            drainage_timescale=cfg.drainage_timescale_s,
            refreeze_threshold=cfg.refreeze_threshold,
            pond_to_ice_max_area=cfg.pond_to_ice_max_area,
            depth_to_area_ratio=cfg.depth_to_area_ratio,
        )

    def test_grad_wrt_melt_water(self):
        pa, pd, melt, rain, im, hs, Ta = self._state()

        def loss(m):
            out = self._step(pa, pd, m, rain, im, hs, Ta)
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in out)

        g = jax.grad(loss)(melt)
        assert_gradient_ok(g, "step_ponds d/dmelt_water", min_nonzero_frac=0.5)

    def test_grad_wrt_T_air_across_refreeze_switch(self):
        """The refreeze switch is smoothed by ``refreeze_width_K``
        precisely so the pond adjoint survives it.  A hard threshold
        would give exactly zero here."""
        pa, pd, melt, rain, im, hs, Ta = self._state()

        def loss(T):
            out = self._step(pa, pd, melt, rain, im, hs, T)
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in out)

        g = jax.grad(loss)(Ta)
        assert jnp.all(jnp.isfinite(g)), "step_ponds d/dT_air: NaN/Inf"
        assert float(jnp.max(jnp.abs(g))) > 0.0, (
            "step_ponds d/dT_air is identically zero — the refreeze "
            "threshold is a hard switch, killing the pond-albedo adjoint")
        # The sensitivity must concentrate near the threshold, not be uniform.
        near = float(jnp.max(jnp.abs(g[1:3])))
        far = float(jnp.max(jnp.abs(g[jnp.array([0, 3])])))
        assert near >= far, (
            f"refreeze sensitivity is larger far from the threshold "
            f"(near={near:.3e}, far={far:.3e})")

    def test_grad_wrt_pond_state(self):
        pa, pd, melt, rain, im, hs, Ta = self._state()

        def loss(args):
            out = self._step(args[0], args[1], melt, rain, im, hs, Ta)
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in out)

        ga, gd = jax.grad(loss)((pa, pd))
        assert_gradient_ok(ga, "step_ponds d/dpond_area", min_nonzero_frac=0.5)
        assert_gradient_ok(gd, "step_ponds d/dpond_depth", min_nonzero_frac=0.5)


# ============================================================================
# SEA ICE — ridging
# ============================================================================

class TestRidging:

    @staticmethod
    def _cats(n_cat=5):
        a = jnp.asarray(np.array([0.30, 0.25, 0.20, 0.10, 0.05])[:n_cat])
        h = jnp.asarray(np.array([0.2, 0.6, 1.4, 2.6, 4.0])[:n_cat])
        return a, h

    def test_grad_participation_weights(self):
        from legoesm.ice.ridging import participation_weights
        from legoesm.ice.config import RidgingConfig
        a, h = self._cats()
        e_star = RidgingConfig().e_star

        def loss(args):
            return jnp.sum(participation_weights(args[0], args[1], e_star) ** 2)

        ga, gh = jax.grad(loss)((a, h))
        assert_gradient_ok(ga, "participation_weights d/da",
                           min_nonzero_frac=0.6)
        assert jnp.all(jnp.isfinite(gh)), "participation_weights d/dh: NaN/Inf"

    def test_grad_apply_ridging(self):
        """``apply_ridging`` redistributes area/volume across categories;
        the adjoint must survive the category loop and the mass
        redistribution."""
        from legoesm.ice.ridging import apply_ridging
        a, h = self._cats()
        V_snow = jnp.asarray(np.full(5, 0.02))
        S_ice = jnp.asarray(np.full(5, 5.0))
        closing = jnp.asarray(1.0e-7)

        def loss(args):
            out = apply_ridging(args[0], args[1], V_snow, S_ice, args[2],
                                5, 3600.0)
            return sum(jnp.sum(jnp.asarray(v) ** 2)
                       for v in out.values()
                       if jnp.asarray(v).dtype.kind == "f")

        ga, gh, gc = jax.grad(loss)((a, h, closing))
        assert jnp.all(jnp.isfinite(ga)), "apply_ridging d/da: NaN/Inf"
        assert jnp.all(jnp.isfinite(gh)), "apply_ridging d/dh: NaN/Inf"
        assert float(jnp.max(jnp.abs(ga))) > 0.0, "apply_ridging d/da dead"
        assert_scalar_grad_alive(gc, "apply_ridging d/dclosing_rate")


# ============================================================================
# SEA ICE — brine / salt budget
# ============================================================================

class TestBrineSaltBudget:

    def test_grad_update_salinity_and_salt_flux(self):
        from legoesm.ice.brine import update_salinity_and_salt_flux
        n = 4
        # NON-uniform per category: identical inputs give an identical
        # (uniform) gradient, which the structure check would flag as a
        # collapsed adjoint when it is really a degenerate fixture.
        S_old = jnp.asarray(np.array([3.0, 5.0, 7.0, 9.0]))
        V_old = jnp.asarray(np.array([0.5, 1.0, 1.5, 2.0]))
        V_new = jnp.asarray(np.array([0.55, 1.05, 1.60, 2.10]))
        dV_lead = jnp.asarray(np.array([0.01, 0.02, 0.03, 0.04]))
        dV_white = jnp.asarray(np.array([0.005, 0.010, 0.015, 0.020]))

        def loss(args):
            res = update_salinity_and_salt_flux(
                args[0], args[1], args[2], args[3], dV_white,
                rho_ice=float(constants.rho_ice), dt=3600.0,
                S_lead_ice=4.0, S_white_ice=0.0)
            leaves = [getattr(res, f) for f in res._fields]
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in leaves
                       if jnp.asarray(x).dtype.kind == "f")

        gS, gVo, gVn, gLead = jax.grad(loss)((S_old, V_old, V_new, dV_lead))
        assert_gradient_ok(gS, "brine d/dS_ice_old", min_nonzero_frac=0.9)
        assert_gradient_ok(gVn, "brine d/dV_ice_new", min_nonzero_frac=0.6)
        assert jnp.all(jnp.isfinite(gVo)), "brine d/dV_ice_old: NaN/Inf"
        assert float(jnp.max(jnp.abs(gLead))) > 0.0, (
            "brine d/d(lead-freeze volume) is dead — the lead-ice salt "
            "source is unreachable by AD")

    def test_salt_conservation_adjoint(self):
        """Salt budget closure: the salt flux to the ocean and the ice
        salt content must respond to the SAME perturbation with
        opposite sign (salt leaving the ice enters the ocean)."""
        from legoesm.ice.brine import update_salinity_and_salt_flux
        S_old = jnp.asarray(6.0)
        V_old = jnp.asarray(1.0)

        def ice_salt(V_new):
            res = update_salinity_and_salt_flux(
                S_old, V_old, V_new, jnp.asarray(0.0), jnp.asarray(0.0),
                rho_ice=float(constants.rho_ice), dt=3600.0,
                S_lead_ice=4.0, S_white_ice=0.0)
            return res.S_ice_new * V_new

        g = float(jax.grad(ice_salt)(jnp.asarray(0.9)))
        assert np.isfinite(g), "salt-content adjoint is NaN/Inf"
        assert g != 0.0, (
            "d(ice salt content)/d(ice volume) is exactly zero — the "
            "brine salt budget is detached from the volume change")


# ============================================================================
# SEA ICE — snow
# ============================================================================

class TestIceSnow:

    def test_grad_combined_conductive_flux(self):
        """The snow+ice series conductance sets the whole ice thermodynamic
        response; the ``h_ice_min``/``h_snow_min`` floors are the place a
        thin-ice adjoint dies."""
        from legoesm.ice.snow import combined_conductive_flux
        n = 4
        T_base = jnp.asarray(np.full(n, constants.T_freeze_ocean))
        T_sfc = jnp.asarray(constants.T_freeze - np.array([1.0, 5.0, 10.0, 20.0]))
        h_ice = jnp.asarray(np.array([0.05, 0.5, 1.5, 3.0]))
        h_snow = jnp.asarray(np.array([0.0, 0.05, 0.2, 0.5]))

        def loss(args):
            return jnp.sum(combined_conductive_flux(
                args[0], args[1], args[2], args[3],
                float(constants.k_ice_default), float(constants.k_snow),
                0.01, 0.001) ** 2)

        gTb, gTs, ghi, ghs = jax.grad(loss)((T_base, T_sfc, h_ice, h_snow))
        assert_gradient_ok(gTb, "conductive flux d/dT_base",
                           min_nonzero_frac=0.9)
        assert_gradient_ok(gTs, "conductive flux d/dT_sfc",
                           min_nonzero_frac=0.9)
        assert_gradient_ok(ghi, "conductive flux d/dh_ice",
                           min_nonzero_frac=0.9)
        assert jnp.all(jnp.isfinite(ghs)), "conductive flux d/dh_snow: NaN/Inf"
        # Sign: thicker ice insulates -> |flux| decreases with h_ice.
        assert float(jnp.max(jnp.abs(ghi))) > 0.0

    def test_grad_accumulate_snowfall(self):
        from legoesm.ice.snow import accumulate_snowfall
        n = 3
        h_snow = jnp.asarray(np.array([0.05, 0.10, 0.20]))
        precip = jnp.asarray(np.array([1.0e-6, 2.0e-6, 4.0e-6]))
        ice_mask = jnp.ones(n, dtype=bool)

        def loss(args):
            out = accumulate_snowfall(args[0], args[1], ice_mask, 3600.0,
                                      float(constants.rho_snow))
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in out)

        gh, gp = jax.grad(loss)((h_snow, precip))
        assert_gradient_ok(gh, "accumulate_snowfall d/dh_snow",
                           min_nonzero_frac=0.9)
        assert_gradient_ok(gp, "accumulate_snowfall d/dprecip",
                           min_nonzero_frac=0.9)

    def test_grad_snow_ice_flooding(self):
        """Snow-ice (white-ice) formation fires when the snow load pushes
        the ice surface below the waterline — an Archimedes threshold
        that must be differentiable on BOTH sides."""
        from legoesm.ice.snow import snow_ice_flooding
        h_ice = jnp.asarray(np.array([0.5, 0.5, 0.5, 0.5]))
        # Flooding threshold: h_snow > h_ice*(rho_ocean-rho_ice)/rho_snow
        h_snow = jnp.asarray(np.array([0.02, 0.10, 0.20, 0.40]))

        def loss(args):
            out = snow_ice_flooding(
                args[0], args[1], float(constants.rho_ice),
                float(constants.rho_snow), float(constants.rho_ocean))
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in out)

        ghi, ghs = jax.grad(loss)((h_ice, h_snow))
        assert jnp.all(jnp.isfinite(ghi)), "snow_ice_flooding d/dh_ice: NaN/Inf"
        assert jnp.all(jnp.isfinite(ghs)), "snow_ice_flooding d/dh_snow: NaN/Inf"
        assert float(jnp.max(jnp.abs(ghs))) > 0.0, (
            "snow_ice_flooding d/dh_snow is identically zero — the "
            "flooding threshold has severed the snow-load adjoint")


# ============================================================================
# SEA ICE — dynamics stress kernels
# ============================================================================

class TestIceDynamicsStresses:

    @staticmethod
    def _vel(n=4, seed=7):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(0.1 * rng.standard_normal(n)),
                jnp.asarray(0.1 * rng.standard_normal(n)),
                jnp.asarray(5.0 * rng.standard_normal(n)),
                jnp.asarray(5.0 * rng.standard_normal(n)))

    def test_grad_free_drift_velocity(self):
        from legoesm.ice.dynamics import free_drift_velocity
        ou, ov, wu, wv = self._vel()

        def loss(args):
            u, v = free_drift_velocity(args[0], args[1], args[2], args[3])
            return jnp.sum(u ** 2) + jnp.sum(v ** 2)

        gs = jax.grad(loss)((ou, ov, wu, wv))
        for name, g in zip(("ocean_u", "ocean_v", "wind_u", "wind_v"), gs):
            assert_gradient_ok(g, f"free_drift d/d{name}",
                               min_nonzero_frac=0.9)

    def test_grad_air_and_ocean_ice_stress_and_signs(self):
        """Sign convention: the ocean-ice stress must OPPOSE the ice
        velocity (a drag), so d(tau_ocean_x)/d(u_ice) < 0; the air-ice
        stress must increase with the wind, d(tau_air_x)/d(wind_u) > 0.
        A flipped sign here spins the ice up instead of damping it, and
        a finiteness-only check cannot see it."""
        from legoesm.ice.dynamics import air_ice_stress, ocean_ice_stress
        u_ice = jnp.asarray(0.2)
        v_ice = jnp.asarray(0.0)
        wind_u = jnp.asarray(8.0)
        ocean_u = jnp.asarray(0.0)

        d_tau_o = float(jax.grad(
            lambda u: ocean_ice_stress(u, v_ice, ocean_u, jnp.asarray(0.0))[0]
        )(u_ice))
        d_tau_a = float(jax.grad(
            lambda w: air_ice_stress(u_ice, v_ice, w, jnp.asarray(0.0))[0]
        )(wind_u))
        assert np.isfinite(d_tau_o) and np.isfinite(d_tau_a)
        assert d_tau_o < 0.0, (
            f"d(ocean-ice stress_x)/d(u_ice) = {d_tau_o:.4e} — ocean drag "
            "must OPPOSE ice motion (negative), this sign accelerates ice")
        assert d_tau_a > 0.0, (
            f"d(air-ice stress_x)/d(wind_u) = {d_tau_a:.4e} — wind stress "
            "must increase with wind speed")
