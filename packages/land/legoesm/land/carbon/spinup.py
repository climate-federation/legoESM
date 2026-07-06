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

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.land.carbon.config import CarbonState


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
