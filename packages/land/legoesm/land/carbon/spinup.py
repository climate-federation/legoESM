"""Semi-analytic soil-carbon spin-up for the DifferLand prognostic pools.

Standard land-carbon spin-up practice: a soil-carbon pool with centuries-to-
millennial turnover cannot be equilibrated by brute-force transient integration
in a feasible run (a ~270-yr pool needs ~1500-3000 model years, and cold-soil
pools far longer).  The community-standard remedy is a **semi-analytic
spin-up** (Xia et al. 2012, GMD 5, "A semi-analytical solution to accelerate
spin-up of a coupled carbon and nitrogen land model"; closely related to the
**accelerated-decomposition** methods of Thornton & Rosenbloom 2005 and Koven
et al. 2013): run a transient long enough to stationarise the fast pools and
the mean annual carbon fluxes, then solve the LINEAR slow-pool steady state
analytically and reset the pools to it, followed by a short verification
integration confirming the drift -> 0.

For a first-order (linear) pool ``dC/dt = I - k*C`` the turnover ``k`` is
independent of the pool size, so the equilibrium is simply ``C_eq = I / k``.
Given a spun-up pool ``C`` and its stationary annual input ``I_annual`` and loss
``loss_annual = k*C``, this is obtained without ever knowing ``k`` explicitly:

    C_eq = I_annual / k = C * (I_annual / loss_annual).

The DifferLand slow pools are wood (~decades) and SOM (~centuries); the fast
pools (labile/foliage/root/litter, sub-decadal) equilibrate within the
transient and are left untouched.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm.land.carbon.config import CarbonState

# --- time conversions (exact) ---
_SECS_PER_DAY = 86400.0
_SECS_PER_HOUR = 3600.0
_HOURS_PER_DAY = 24.0
_YEAR_DAYS = 365.0

# Prognostic carbon pools summed for the closed-column mass balance.
_POOL_FIELDS = ("C_lab", "C_fol", "C_root", "C_wood", "C_lit", "C_som")


class SlowPoolFluxes(NamedTuple):
    """Stationary mean-annual carbon fluxes for the slow-pool analytic solve.

    All fields are per-column arrays (or scalars) in [gC/m2/yr], taken as the
    mean over the final stationary year(s) of the transient spin-up.
    """
    a_wood: jnp.ndarray       # NPP allocated to wood  (wood input)
    wood_litter: jnp.ndarray  # wood turnover          (wood loss)
    lit_to_som: jnp.ndarray   # litter -> SOM          (SOM input from litter)
    r_het_som: jnp.ndarray    # SOM heterotrophic resp (SOM loss)


def analytic_slow_pool_equilibrium(
    carbon_state: CarbonState,
    fluxes: SlowPoolFluxes,
    cwd_humification_eff: float,
    eps: float = 1e-9,
) -> CarbonState:
    """Reset the slow pools (wood, SOM) of *carbon_state* to their analytic
    linear-pool steady state given stationary mean-annual *fluxes*.

    Wood:  loss = wood_litter = k_wood * C_wood, input = a_wood, so
           ``C_wood_eq = C_wood * a_wood / wood_litter``.
    SOM :  at wood equilibrium the wood -> SOM humified input is
           ``cwd_humification_eff * a_wood`` (wood in == out), so the total SOM
           input is ``lit_to_som + cwd_humification_eff*a_wood``; with
           loss = r_het_som = k_som * C_som,
           ``C_som_eq = C_som * som_input_eq / r_het_som``.

    A pool whose LOSS flux is ~0 (a dead/collapsed or masked column) has no
    inferable turnover ``k``, hence no finite analytic equilibrium, so it is
    LEFT UNCHANGED at its spun-up value rather than reset to an artefact
    (0 or ``C*I/eps``).  The fast pools are always returned unchanged.

    Parameters
    ----------
    carbon_state : CarbonState
        The spun-up state (fast pools + wood already ~stationary).
    fluxes : SlowPoolFluxes
        Stationary mean-annual slow-pool fluxes [gC/m2/yr].
    cwd_humification_eff : float
        Fraction of wood turnover that humifies to SOM (``CarbonConfig``).
    eps : float
        Threshold below which a loss flux is treated as ~0 (pool left as is).

    Returns
    -------
    CarbonState
        A copy of *carbon_state* with ``C_wood`` and ``C_som`` set to their
        analytic equilibrium (where the loss flux is > ``eps``); all other
        pools unchanged.
    """
    C_wood_eq = jnp.where(
        fluxes.wood_litter > eps,
        carbon_state.C_wood * fluxes.a_wood / jnp.maximum(fluxes.wood_litter, eps),
        carbon_state.C_wood,
    )
    wood_to_som_eq = cwd_humification_eff * fluxes.a_wood
    som_in_eq = fluxes.lit_to_som + wood_to_som_eq
    C_som_eq = jnp.where(
        fluxes.r_het_som > eps,
        carbon_state.C_som * som_in_eq / jnp.maximum(fluxes.r_het_som, eps),
        carbon_state.C_som,
    )
    return carbon_state._replace(C_wood=C_wood_eq, C_som=C_som_eq)


def _total_carbon(carbon_state: CarbonState) -> jnp.ndarray:
    """Column total of the six prognostic pools [gC/m2].

    The DifferLand column is closed (the only exchange with the atmosphere is
    NEE), so ``sum(dC_pools) == -NEE_day*dt`` exactly (see
    ``carbon_cycle.step_carbon_differland``; verified by
    ``test_carbon_*_conservation``).  The verified spin-up therefore recovers
    the model's authoritative annual NEE from the pool change, without
    threading the coupled ``TileResponse.co2_flux`` through ``step_fn``.
    """
    return (
        carbon_state.C_lab + carbon_state.C_fol + carbon_state.C_root
        + carbon_state.C_wood + carbon_state.C_lit + carbon_state.C_som
    )


def run_semi_analytic_spinup(
    step_fn: Callable,
    state0,
    carbon0: CarbonState,
    forcing_fn: Callable,
    *,
    n_spinup: int,
    n_verify: int,
    steps_per_year: int,
    dt: float,
    cwd_humification_eff: float,
):
    """Run a verified semi-analytic soil-carbon spin-up (Xia et al. 2012, GMD).

    Shared 3-phase driver behind BOTH the per-pixel ``land_carbon_equilibrium``
    validator and the batched ``global_init.equilibrate_archetypes`` archetype
    map — neither re-derives the phase logic:

    1. **Transient** (``n_spinup`` years): a nested ``lax.scan`` (outer = years,
       inner = sub-daily steps) advances ``step_fn`` under the repeating annual
       ``forcing_fn`` climate so the fast pools + wood + the mean-annual carbon
       fluxes stationarise.
    2. **Analytic slow-pool reset**: the LINEAR wood/SOM steady state is solved
       from the LAST spin-up year's mean-annual slow-pool fluxes
       (:func:`analytic_slow_pool_equilibrium`) and the pools reset to it.
    3. **Verification** (``n_verify`` years): re-integrate from the analytic
       equilibrium; the returned per-year diagnostics are this verified segment
       (its drift -> 0 confirms the equilibrium held).

    Parameters
    ----------
    step_fn : callable
        ``step_fn(state, carbon, forcing, doy) -> (new_state, new_carbon,
        diag)`` where ``diag`` is a :class:`~legoesm.land.carbon.config.
        CarbonDiagnostics` (or any NamedTuple carrying at least its fields).
        Each ``diag`` field is a per-day rate [gC/m2/day] (``lai`` is
        [m2/m2]); the slow-pool solve reads ``a_wood``/``wood_litter``/
        ``lit_to_som``/``r_het_som`` by name.
    state0, carbon0 :
        Initial land state (opaque to this driver) and :class:`CarbonState`
        (both batched over ``ncol`` columns).
    forcing_fn : callable
        ``forcing_fn(doy, hour) -> AtmToSurface`` — the repeating annual
        climate for every column (may be traced ``doy``/``hour`` scalars).
    n_spinup, n_verify : int
        Transient and verification year counts.
    steps_per_year : int
        Sub-daily steps per model year (``round(seconds_per_year / dt)``).
    dt : float
        Sub-daily timestep [s].
    cwd_humification_eff : float
        Fraction of wood turnover humified to SOM (``CarbonConfig``), for the
        analytic SOM-input balance.

    Returns
    -------
    (final_state, final_carbon, annual)
        ``final_carbon`` is the verified-equilibrium :class:`CarbonState`
        ``(ncol,)``.  ``annual`` is a dict of per-verify-year ``(n_verify,
        ncol)`` arrays: every ``diag`` flux field as an annual total
        [gC/m2/yr] (``sum(rate*dt_days)``), ``lai_sum``/``lai_max``, the
        allocation residual ``alloc_resid``, ``nsteps``, the model's
        mass-balance annual ``nee_model``, and the end-of-year pools
        ``C_lab``..``C_som``.
    """
    dt_days = dt / _SECS_PER_DAY

    def _inner_step(carry, step_idx):
        state, carbon = carry
        t_day = step_idx * dt_days
        doy = jnp.mod(t_day, _YEAR_DAYS)
        hour = jnp.mod(step_idx * dt / _SECS_PER_HOUR, _HOURS_PER_DAY)
        forcing = forcing_fn(doy, hour)
        new_state, new_carbon, diag = step_fn(state, carbon, forcing, doy)
        return (new_state, new_carbon), diag

    def _year_step(carry, _year_idx):
        _, carbon_start = carry
        total_start = _total_carbon(carbon_start)
        (state_end, carbon_end), diags = jax.lax.scan(
            _inner_step, carry, jnp.arange(steps_per_year))
        # Annual totals: every per-day flux integrated as sum(rate*dt_days)
        # -> gC/m2/yr; LAI (a state, not a rate) reduced to season sum + max.
        annual: dict = {}
        for field in diags._fields:
            if field == "lai":
                annual["lai_sum"] = jnp.sum(diags.lai, axis=0)
                annual["lai_max"] = jnp.max(diags.lai, axis=0)
            else:
                annual[field] = jnp.sum(getattr(diags, field) * dt_days, axis=0)
        # Allocation-closure residual: |sum(A) - max(NPP,0)| accumulated per
        # step (sum-of-abs, not |sum|, so winter NPP<0 cannot cancel it).
        resid = jnp.abs(
            diags.a_fol + diags.a_lab + diags.a_root + diags.a_wood
            - jnp.maximum(diags.npp, 0.0))
        annual["alloc_resid"] = jnp.sum(resid * dt_days, axis=0)
        # Model's authoritative annual NEE from the closed-column mass balance
        # (dC_total = -NEE); identical to integrating TileResponse.co2_flux.
        total_end = _total_carbon(carbon_end)
        annual["nee_model"] = -(total_end - total_start)
        annual["nsteps"] = jnp.full_like(total_end, float(steps_per_year))
        for pool in _POOL_FIELDS:
            annual[pool] = getattr(carbon_end, pool)
        return (state_end, carbon_end), annual

    # --- Phase 1: transient spin-up (fast pools + wood + stationary fluxes) ---
    (state_spun, carbon_spun), annual_spin = jax.lax.scan(
        _year_step, (state0, carbon0), jnp.arange(n_spinup))

    # --- Phase 2: analytic linear-pool equilibrium for the slow pools ---
    # Stationary mean-annual slow-pool fluxes from the final spin-up year
    # (already per-column, gC/m2/yr) fed to the shared analytic solver.
    fluxes = SlowPoolFluxes(
        a_wood=annual_spin["a_wood"][-1],
        wood_litter=annual_spin["wood_litter"][-1],
        lit_to_som=annual_spin["lit_to_som"][-1],
        r_het_som=annual_spin["r_het_som"][-1],
    )
    carbon_eq = analytic_slow_pool_equilibrium(
        carbon_spun, fluxes, cwd_humification_eff)

    # --- Phase 3: verification segment from the analytic equilibrium ---
    (final_state, final_carbon), annual_verify = jax.lax.scan(
        _year_step, (state_spun, carbon_eq), jnp.arange(n_verify))
    return final_state, final_carbon, annual_verify
