"""Shared Veros-faithful time-stepping composition (issue #433).

Every Veros fidelity recipe (``build_acc_recipe``, ``build_acc_basic_recipe``,
``build_global_{4deg,1deg,flexible}_recipe``) selects the SAME bundle of
stepping options to mimic Veros's outer integration:

    outer_integrator="ab2", ab2_scope="advective", barotropic_solver="rigid_lid",
    coriolis_scheme="explicit_ab2", momentum_friction_additive=True,
    implicit_vmix_dzw_slot=True   (+ a per-recipe ``dt_mom_ratio``)

This bundle was previously copy-pasted into each builder. That is exactly the
bug-magnet documented in ``veros_acc_basic_recipe.py``: acc_basic predated the
acc change, kept its OWN copy, and so silently ran ``matsuno_split`` / ``"total"``
— an energy audit showed ~+1.8 GW of spurious KE on the acc twin before the copy
was corrected. One shared source of truth removes that whole failure mode.

The ``with_surface_forcing=False`` path (the frozen-state tendency probe) keeps
the legoESM defaults so the probe stays bit-identical — this helper reproduces
that gating exactly.
"""

from __future__ import annotations


def veros_faithful_stepping(*, with_surface_forcing: bool,
                            dt_mom_ratio: float) -> dict:
    """Return the Veros-faithful stepping kwargs for a ``*OceanConfig``.

    Splat into the config constructor, e.g.::

        LatLonCGridOceanConfig(
            ...,
            **veros_faithful_stepping(with_surface_forcing=wsf, dt_mom_ratio=9.0),
            ...,
        )

    Parameters
    ----------
    with_surface_forcing : bool
        ``True`` for the free-run path → the faithful AB2 / rigid-lid /
        explicit-AB2-Coriolis bundle. ``False`` for the frozen-state
        tendency-probe path → legoESM defaults (forward-Euler, matsuno-split,
        ``ab2_scope="total"``), keeping the probe bit-identical.
    dt_mom_ratio : float
        ``dt_tracer / dt_mom`` for the asynchronous outer step (9.0 for ACC,
        48 for global-4deg, 1 for global-1deg, 8 for global-flexible). Ignored
        (forced to 1.0) when ``with_surface_forcing=False``.

    Returns
    -------
    dict
        The seven stepping fields, ready to splat into the config constructor.
    """
    if not with_surface_forcing:
        return dict(
            outer_integrator="forward_euler",
            dt_mom_ratio=1.0,
            barotropic_solver="explicit_substep",
            coriolis_scheme="matsuno_split",
            ab2_scope="total",
            momentum_friction_additive=False,
            implicit_vmix_dzw_slot=False,
        )
    return dict(
        outer_integrator="ab2",
        dt_mom_ratio=dt_mom_ratio,
        barotropic_solver="rigid_lid",
        coriolis_scheme="explicit_ab2",
        ab2_scope="advective",
        momentum_friction_additive=True,
        implicit_vmix_dzw_slot=True,
    )
