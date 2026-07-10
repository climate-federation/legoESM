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

The DifferLand slow pools are wood (~decades) and the 3-pool SOM cascade
active/slow/passive (~yrs / ~decades / ~centuries).  Because the SOM transfer
matrix is lower-triangular (forward cascade active->slow->passive, no
back-transfer), the coupled steady state is exact FORWARD SUBSTITUTION: solve
the active pool, feed its equilibrium transfer into the slow pool, then the
slow transfer into the passive pool (:func:`analytic_slow_pool_equilibrium`).
The fast pools (labile/foliage/root/litter, sub-decadal) equilibrate within the
transient and are left untouched.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm.land.carbon.config import (
    CarbonState,
    som_total,
    validate_som_transfer_fractions,
)

# --- time conversions (exact) ---
_SECS_PER_DAY = 86400.0
_SECS_PER_HOUR = 3600.0
_HOURS_PER_DAY = 24.0
_YEAR_DAYS = 365.0

# Prognostic carbon pools summed for the closed-column mass balance.  MUST equal
# ``CarbonState._fields`` (guarded by
# ``test_carbon_spinup.test_pool_fields_matches_carbon_state``); the SOM pool is
# resolved into active/slow/passive since phase A1.
_POOL_FIELDS = (
    "C_lab", "C_fol", "C_root", "C_wood", "C_lit",
    "C_som_active", "C_som_slow", "C_som_passive",
)


class SlowPoolFluxes(NamedTuple):
    """Stationary mean-annual carbon fluxes for the slow-pool analytic solve.

    All fields are per-column arrays (or scalars) in [gC/m2/yr], taken as the
    total over the final stationary year of the transient spin-up.  The three
    ``som_*_loss`` fields are each SOM pool's TOTAL decomposition ``D_X``
    (transfer to the next pool + respiration), used to infer that pool's
    turnover for the forward-substitution cascade equilibrium.
    """
    a_wood: jnp.ndarray            # NPP allocated to wood   (wood input)
    wood_litter: jnp.ndarray       # wood turnover           (wood loss)
    lit_to_som: jnp.ndarray        # litter -> active SOM    (active input from litter)
    som_active_loss: jnp.ndarray   # active SOM decomposition D_active (active loss)
    som_slow_loss: jnp.ndarray     # slow SOM decomposition   D_slow   (slow loss)
    som_passive_loss: jnp.ndarray  # passive SOM decomposition D_passive (passive loss)


def analytic_slow_pool_equilibrium(
    carbon_state: CarbonState,
    fluxes: SlowPoolFluxes,
    cwd_humification_eff: float,
    f_active_to_slow: float,
    f_slow_to_passive: float,
    eps: float = 1e-9,
) -> CarbonState:
    """Reset the slow pools (wood + the 3-pool SOM cascade) of *carbon_state* to
    their analytic linear steady state given stationary mean-annual *fluxes*.

    The SOM transfer matrix is LOWER-TRIANGULAR (forward cascade, no
    back-transfer), so the coupled steady state is exact FORWARD SUBSTITUTION —
    no singular 3x3 inverse.  Each pool's turnover ``k_X`` is inferred from its
    spun-up loss/stock (``k_X = D_X / C_X``), so the modifier (temperature *
    moisture * freeze) baked into the spun-up loss is carried through exactly:

    Wood:    ``C_wood_eq = C_wood * a_wood / wood_litter``.
    Active:  at wood equilibrium the humified wood input is
             ``cwd_humification_eff * a_wood`` (wood in == out), so
             ``I_active = lit_to_som + cwd_humification_eff*a_wood`` and
             ``C_active_eq = C_active * I_active / som_active_loss``.
             (at equilibrium the active LOSS == ``I_active``.)
    Slow:    input = ``f_active_to_slow * I_active_eq``, where ``I_active_eq``
             is the active pool's REALISED equilibrium loss -- ``I_active``
             when active is alive (``som_active_loss > eps``), exactly 0 when
             active is degenerate (left unchanged above).  A dead/collapsed
             active pool therefore hands the slow pool NOTHING, so
             ``C_slow_eq = C_slow * (f_active_to_slow*I_active_eq) / som_slow_loss``.
    Passive: input = ``f_slow_to_passive * I_slow_eq``, where ``I_slow_eq`` is
             the slow pool's OWN realised equilibrium loss (0 if slow itself
             received no realisable transfer), so
             ``C_passive_eq = C_passive * I_passive / som_passive_loss``.

    Each ``C_X_eq`` is a TRUE fixed point of the cascade: by construction the new
    loss ``k_X * C_X_eq`` equals the new input ``I_X`` per pool (verified by the
    drift-> 0 verification segment and the fixed-point unit test).

    A pool whose LOSS flux is ~0 (a dead/collapsed or masked column) has no
    inferable turnover ``k``, hence no finite analytic equilibrium, so it is
    LEFT UNCHANGED at its spun-up value rather than reset to an artefact
    (0 or ``C*I/eps``).  Critically, that degeneracy also propagates FORWARD
    through the cascade: the transfer a degenerate pool hands to the pool
    below it is gated to 0 (the pool's REALISED loss), never the raw,
    ungated forcing input.  Without this gate a dead/collapsed upstream pool
    (left unchanged, correctly) would still hand its would-be litter/CWD or
    humification input downstream unconditionally, manufacturing a spurious
    nonzero equilibrium for slow/passive from a transfer the (dead) upstream
    pool cannot actually sustain -- e.g. a boreal column with a frozen,
    non-decomposing active pool must NOT spuriously equilibrate its slow/
    passive pools from the active pool's raw litter input (codex finding #4).
    The fast pools are always returned unchanged.

    Parameters
    ----------
    carbon_state : CarbonState
        The spun-up state (fast pools + wood + SOM pools already ~stationary in
        turnover, i.e. each ``som_*_loss`` reflects the true turnover).
    fluxes : SlowPoolFluxes
        Stationary mean-annual slow-pool fluxes [gC/m2/yr].
    cwd_humification_eff : float
        Fraction of wood turnover that humifies to SOM (``CarbonConfig``).
    f_active_to_slow, f_slow_to_passive : float
        Inter-pool humification fractions (``CarbonConfig``) that set the
        cascade transfers active->slow->passive.
    eps : float
        Threshold below which a loss flux is treated as ~0 (pool left as is).

    Returns
    -------
    CarbonState
        A copy of *carbon_state* with ``C_wood``, ``C_som_active``,
        ``C_som_slow`` and ``C_som_passive`` set to their analytic equilibrium
        (per-pool, only where that pool's loss flux is > ``eps``); the fast
        pools unchanged.
    """
    # Fail-early on the STATIC fraction values (dispatch-hardening
    # discipline): this function is a public entry point called DIRECTLY by
    # ``run_lmip.py`` (not only via run_semi_analytic_spinup), so it cannot
    # rely on the step-level guard in carbon_cycle.step_carbon_differland.
    validate_som_transfer_fractions(
        f_active_to_slow, f_slow_to_passive, cwd_humification_eff)
    C_wood_eq = jnp.where(
        fluxes.wood_litter > eps,
        carbon_state.C_wood * fluxes.a_wood / jnp.maximum(fluxes.wood_litter, eps),
        carbon_state.C_wood,
    )
    # Active pool: input = litter->SOM + humified CWD (at wood equilibrium).
    wood_to_som_eq = cwd_humification_eff * fluxes.a_wood
    i_active = fluxes.lit_to_som + wood_to_som_eq
    C_som_active_eq = jnp.where(
        fluxes.som_active_loss > eps,
        carbon_state.C_som_active * i_active
        / jnp.maximum(fluxes.som_active_loss, eps),
        carbon_state.C_som_active,
    )
    # Slow pool: input = the humified transfer out of the equilibrated active
    # pool.  Gate on active's REALISED loss (0 if active is degenerate, i.e.
    # left unchanged above) rather than the raw ``i_active`` unconditionally --
    # a dead/collapsed active pool cannot actually sustain a steady transfer
    # to slow, so feeding it the raw litter/CWD input would manufacture a
    # spurious slow-pool equilibrium the (dead) active pool never produces
    # (codex finding #4).  A no-op when active is alive: at active equilibrium
    # its total loss == i_active, so i_active_eq == i_active exactly.
    i_active_eq = jnp.where(fluxes.som_active_loss > eps, i_active, 0.0)
    i_slow = f_active_to_slow * i_active_eq
    C_som_slow_eq = jnp.where(
        fluxes.som_slow_loss > eps,
        carbon_state.C_som_slow * i_slow
        / jnp.maximum(fluxes.som_slow_loss, eps),
        carbon_state.C_som_slow,
    )
    # Passive pool: same gating one level down -- input = the humified
    # transfer out of the equilibrated slow pool, using slow's OWN REALISED
    # loss (0 if slow is itself degenerate) rather than the raw ``i_slow``.
    i_slow_eq = jnp.where(fluxes.som_slow_loss > eps, i_slow, 0.0)
    i_passive = f_slow_to_passive * i_slow_eq
    C_som_passive_eq = jnp.where(
        fluxes.som_passive_loss > eps,
        carbon_state.C_som_passive * i_passive
        / jnp.maximum(fluxes.som_passive_loss, eps),
        carbon_state.C_som_passive,
    )
    return carbon_state._replace(
        C_wood=C_wood_eq,
        C_som_active=C_som_active_eq,
        C_som_slow=C_som_slow_eq,
        C_som_passive=C_som_passive_eq,
    )


def _total_carbon(carbon_state: CarbonState) -> jnp.ndarray:
    """Column total of every prognostic pool [gC/m2] (biomass + litter + all
    three SOM sub-pools via :func:`som_total`).

    The DifferLand column is closed (the only exchange with the atmosphere is
    NEE), so ``sum(dC_pools) == -NEE_day*dt`` exactly (see
    ``carbon_cycle.step_carbon_differland``; verified by
    ``test_carbon_*_conservation``).  The verified spin-up therefore recovers
    the model's authoritative annual NEE from the pool change, without
    threading the coupled ``TileResponse.co2_flux`` through ``step_fn``.
    """
    return (
        carbon_state.C_lab + carbon_state.C_fol + carbon_state.C_root
        + carbon_state.C_wood + carbon_state.C_lit + som_total(carbon_state)
    )


def step_doy_hour(step_idx, dt):
    """Day-of-year and hour-of-day for sub-daily ``step_idx`` at timestep ``dt``.

    The repeating-annual-climate time convention shared by every forward
    integrator in this module (the transient/verify scans in
    :func:`run_semi_analytic_spinup` and the raw transient in
    :func:`integrate_annual_pools`) AND the offline fast-analytic precompute's
    stationary-year recording scan
    (:func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`), so
    the time-of-year mapping is defined ONCE rather than re-derived per caller.
    """
    t_day = step_idx * (dt / _SECS_PER_DAY)
    doy = jnp.mod(t_day, _YEAR_DAYS)
    hour = jnp.mod(step_idx * dt / _SECS_PER_HOUR, _HOURS_PER_DAY)
    return doy, hour


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
    f_active_to_slow: float,
    f_slow_to_passive: float,
    remat: bool = False,
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
    f_active_to_slow, f_slow_to_passive : float
        Inter-pool humification fractions (``CarbonConfig``) that drive the
        forward-substitution active->slow->passive cascade equilibrium.
    remat : bool
        When True, wrap the per-year spin-up body in ``jax.checkpoint``
        (``jax.remat``) so reverse-mode AD recomputes each year's sub-daily inner
        scan during the backward pass instead of storing its full tape.  This
        bounds peak reverse-mode memory to a SINGLE year (needed when
        differentiating the equilibrium SOC w.r.t. the SOM parameters through a
        multi-year spin-up -- see
        ``global_init.equilibrate_archetypes_traced``).  Numerically identical to
        ``remat=False`` (checkpointing changes only the store-vs-recompute
        schedule, never the values); default False keeps the forward-only /
        static callers (``equilibrate_archetypes``, the drift validator)
        byte-for-byte unchanged.

    Returns
    -------
    (final_state, final_carbon, annual, reset_fluxes)
        ``final_carbon`` is the verified-equilibrium :class:`CarbonState`
        ``(ncol,)``.  ``annual`` is a dict of per-verify-year ``(n_verify,
        ncol)`` arrays: every ``diag`` flux field as an annual total
        [gC/m2/yr] (``sum(rate*dt_days)``), ``lai_sum``/``lai_max``, the
        allocation residual ``alloc_resid``, ``nsteps``, the model's
        mass-balance annual ``nee_model``, and the end-of-year pools
        ``C_lab``..``C_som_passive`` (the eight ``_POOL_FIELDS``).
        ``reset_fluxes`` is the :class:`SlowPoolFluxes` (per-column ``(ncol,)``)
        the analytic reset consumed -- the LAST-TRANSIENT-year slow-pool fluxes --
        so a caller can reproduce that exact reset (used by the fast-analytic SOC
        precompute; see the return statement's comment).
    """
    # Fail early on degenerate run controls (dispatch-hardening discipline): the
    # analytic reset reads the LAST spin-up year's fluxes (needs n_spinup >= 1),
    # the drift diagnostic needs >= 2 verification years, and each year is a
    # sub-daily scan of steps_per_year >= 1 steps (F7).  The transfer-fraction
    # check also lives here (not just inside analytic_slow_pool_equilibrium)
    # so an invalid config fails BEFORE the expensive Phase-1 transient scan,
    # not after it.
    if n_spinup < 1:
        raise ValueError(
            f"run_semi_analytic_spinup: n_spinup must be >= 1 (the analytic "
            f"slow-pool reset reads the last transient year's fluxes), got "
            f"{n_spinup}.")
    if n_verify < 2:
        raise ValueError(
            f"run_semi_analytic_spinup: n_verify must be >= 2 (drift needs at "
            f"least two verification years), got {n_verify}.")
    if steps_per_year < 1:
        raise ValueError(
            f"run_semi_analytic_spinup: steps_per_year must be >= 1, got "
            f"{steps_per_year}.")
    validate_som_transfer_fractions(
        f_active_to_slow, f_slow_to_passive, cwd_humification_eff)
    dt_days = dt / _SECS_PER_DAY

    def _inner_step(carry, step_idx):
        state, carbon = carry
        doy, hour = step_doy_hour(step_idx, dt)
        forcing = forcing_fn(doy, hour)
        new_state, new_carbon, diag = step_fn(state, carbon, forcing, doy)
        return (new_state, new_carbon), diag

    def _year_step(carry, _year_idx):
        _, carbon_start = carry
        total_start = _total_carbon(carbon_start)
        # TODO(perf/AD): accumulate annual sums/max inside _inner_step and
        # return yearly reductions, instead of materializing the full per-step
        # `diags` for every year before reducing (codex F6).  At hourly dt this
        # holds steps_per_year x n_diag_fields per year -> memory pressure and
        # poor reverse-mode AD ergonomics.  Deferred here on purpose: this is the
        # SHARED driver (equilibrate + validator + land_carbon_equilibrium) and
        # the annual-dict contract must stay byte-for-byte behaviour-preserving,
        # so the in-scan reduction (which reorders the float summation) is a
        # separately-validated follow-up.
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

    # Optionally checkpoint the per-year body so reverse-mode AD recomputes each
    # year's inner sub-daily scan rather than taping it (bounds peak memory to a
    # single year); numerically identical to the un-rematted body.
    year_step = jax.checkpoint(_year_step) if remat else _year_step

    # --- Phase 1: transient spin-up (fast pools + wood + stationary fluxes) ---
    (state_spun, carbon_spun), annual_spin = jax.lax.scan(
        year_step, (state0, carbon0), jnp.arange(n_spinup))

    # --- Phase 2: analytic linear-pool equilibrium for the slow pools ---
    # Stationary mean-annual slow-pool fluxes from the final spin-up year
    # (already per-column, gC/m2/yr) fed to the shared analytic solver.
    fluxes = SlowPoolFluxes(
        a_wood=annual_spin["a_wood"][-1],
        wood_litter=annual_spin["wood_litter"][-1],
        lit_to_som=annual_spin["lit_to_som"][-1],
        som_active_loss=annual_spin["som_active_loss"][-1],
        som_slow_loss=annual_spin["som_slow_loss"][-1],
        som_passive_loss=annual_spin["som_passive_loss"][-1],
    )
    carbon_eq = analytic_slow_pool_equilibrium(
        carbon_spun, fluxes, cwd_humification_eff,
        f_active_to_slow, f_slow_to_passive)

    # --- Phase 3: verification segment from the analytic equilibrium ---
    (final_state, final_carbon), annual_verify = jax.lax.scan(
        year_step, (state_spun, carbon_eq), jnp.arange(n_verify))
    # Return the LAST-TRANSIENT-year slow-pool ``fluxes`` the analytic reset
    # consumed (a :class:`SlowPoolFluxes` of per-column ``(ncol,)`` gC/m2/yr) as an
    # explicit 4th element, so a caller can reproduce that reset's INPUT chain.  The
    # fast-analytic SOC precompute
    # (:func:`legoesm.land.carbon.global_init.precompute_fast_analytic_inputs`)
    # records the SURROGATE's ``lit_to_som``/``a_wood`` inputs from THESE reset
    # fluxes (not from a separate post-verify stationary year), so the closed-form
    # ``analytic_som_soc`` reproduces ``analytic_slow_pool_equilibrium`` rather than
    # sampling a DIFFERENT phase of the still-equilibrating decadal wood pool: the
    # ~27-yr wood pool's NPP-to-wood allocation ``a_wood`` shifts between the
    # transient (pre-reset) and post-verify states (the reset boosts ``C_wood`` ->
    # raises maintenance respiration -> depresses NPP/allocation), so the two phases
    # differ by ~8-11% for woody columns and drive the whole active->slow->passive
    # CWD-humification cascade off (the systematic passive under-prediction).  Kept
    # OUT of the per-verify-year ``annual`` dict on purpose (it is a single reset-time
    # ``(ncol,)`` flux, not an ``(n_verify, ncol)`` series -- callers that reshape the
    # whole dict per verify year must not see it).
    return final_state, final_carbon, annual_verify, fluxes


def integrate_annual_pools(
    step_fn: Callable,
    state0,
    carbon0: CarbonState,
    forcing_fn: Callable,
    *,
    n_years: int,
    steps_per_year: int,
    dt: float,
):
    """Forward-integrate the coupled step ``n_years`` under the repeating annual
    ``forcing_fn``, recording the eight carbon pools at the END of each year.

    Unlike :func:`run_semi_analytic_spinup` this performs NO analytic slow-pool
    reset -- it is the RAW transient used to MEASURE how far a given IC drifts
    from equilibrium (the ``global_carbon_ic_map`` validator runs it from BOTH
    the mapped archetype-equilibrium IC and a cold IC, then compares drifts).
    The initial condition is returned as row 0 so the full trajectory
    (IC -> year ``n_years``) is available to a drift metric.

    Parameters
    ----------
    step_fn : callable
        ``step_fn(state, carbon, forcing, doy) -> (new_state, new_carbon,
        diag)`` -- the SAME signature :func:`run_semi_analytic_spinup` consumes;
        ``diag`` is ignored here (only the state/carbon trajectory is kept, and
        the pools evolve purely through ``new_carbon``).
    state0, carbon0 :
        Initial land state (opaque to this driver) and :class:`CarbonState`
        (both batched over ``ncol`` columns).
    forcing_fn : callable
        ``forcing_fn(doy, hour) -> AtmToSurface`` repeating annual climate.
    n_years : int
        Number of years to integrate forward.
    steps_per_year : int
        Sub-daily steps per model year (``round(seconds_per_year / dt)``).
    dt : float
        Sub-daily timestep [s].

    Returns
    -------
    dict[str, jnp.ndarray]
        One ``(n_years + 1, ncol)`` array per pool in ``_POOL_FIELDS``: row 0 is
        the initial condition, rows ``1..n_years`` the end-of-year pools.
    """
    def _inner_step(carry, step_idx):
        state, carbon = carry
        doy, hour = step_doy_hour(step_idx, dt)
        forcing = forcing_fn(doy, hour)
        new_state, new_carbon, _diag = step_fn(state, carbon, forcing, doy)
        return (new_state, new_carbon), None

    def _year_step(carry, _year_idx):
        (state_end, carbon_end), _ = jax.lax.scan(
            _inner_step, carry, jnp.arange(steps_per_year))
        pools = jnp.stack(
            [getattr(carbon_end, p) for p in _POOL_FIELDS], axis=0)  # (8, ncol)
        return (state_end, carbon_end), pools

    (_final_state, _final_carbon), pools_seq = jax.lax.scan(
        _year_step, (state0, carbon0), jnp.arange(n_years))
    # pools_seq: (n_years, 6, ncol).  Prepend the IC as year 0 so the trajectory
    # spans the whole run (drift is measured over IC -> final).
    ic = jnp.stack([getattr(carbon0, p) for p in _POOL_FIELDS], axis=0)
    allp = jnp.concatenate([ic[None], pools_seq], axis=0)  # (n_years+1, 8, ncol)
    return {p: allp[:, i, :] for i, p in enumerate(_POOL_FIELDS)}
