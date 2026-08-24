"""The NEMO transport metric weighting on the AL81/EEN vorticity flux.

NEMO's ``vor_een`` does not use the bare mass fluxes. It weights the meridional
transport by ``e1v`` -- the V-face zonal width (``dynvor.F90:791-792``,
``zwy = e1v*e3v*pv``) -- and divides the assembled u-tendency by ``e1u``
(``dynvor.F90:804``, ``r1_e1u``), and symmetrically ``e2u``/``r1_e2v`` for the
v-tendency. legoESM's ``pv_flux_al81_partial_cell`` computed the per-unit-width
form with neither, which is identical only where the horizontal metrics are
uniform. ``metric_widths`` selects NEMO's form; ``None`` (the default) keeps
the historical one bit-identical.

THE NON-VACUOUS CHECK is ``test_physical_energy_norm_is_conserved_only_with_``
``metric``: on a closed domain in the inviscid limit the vorticity flux must do
no net work. Which quadratic norm it conserves depends on the weighting --

  * WITH the metric, the domain-summed work in the PHYSICAL kinetic-energy norm
    ``sum e1*e2*h*u^2`` telescopes to zero, because the ``1/e1u`` cancels
    against the ``e1u`` in the u-cell volume;
  * WITHOUT it, the same sum is NOT zero, and what telescopes instead is the
    per-area norm ``sum h*u^2``, which is not a physical invariant on a
    stretched grid.

So the test fails in BOTH directions: delete the weighting and the physical
norm stops conserving; apply it unconditionally and the per-area norm result
changes. It cannot pass vacuously.

Measured consequence on the DINO oracle (#1455): supplying the weighting closes
96-98% of the wall-row disagreement against NEMO's own dumped vorticity
tendency, under both the level-mean and the mass-weighted reduction.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    curl_vertex_cgrid,
    pv_flux_al81_partial_cell,
)


def _state(n_lat=16, n_lon=24, seed=7):
    """Closed domain, flat bottom, all wet -- AL81's classical regime."""
    grid = create_latlon_grid(n_lat, n_lon)
    geom = ensure_geometry(grid)
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((n_lat, n_lon + 1, 1)) * 0.1)
    v = np.asarray(rng.standard_normal((n_lat + 1, n_lon, 1)) * 0.1)
    v[0] = 0.0                      # closed south wall
    v[-1] = 0.0                     # closed north wall
    v = jnp.asarray(v)
    u = u.at[:, -1, :].set(u[:, 0, :])          # periodic in longitude
    h = jnp.full((n_lat, n_lon, 1), 4000.0)
    h_u = jnp.full((n_lat, n_lon + 1, 1), 4000.0)
    h_v = jnp.concatenate(
        [jnp.zeros((1, n_lon, 1)), jnp.full((n_lat - 1, n_lon, 1), 4000.0),
         jnp.zeros((1, n_lon, 1))], axis=0)
    h_vtx = jnp.full((n_lat + 1, n_lon + 1, 1), 4000.0)
    u_mask = jnp.ones_like(h_u)
    v_mask = jnp.concatenate(
        [jnp.zeros((1, n_lon, 1)), jnp.ones((n_lat - 1, n_lon, 1)),
         jnp.zeros((1, n_lon, 1))], axis=0)
    vtx_mask = jnp.ones((n_lat + 1, n_lon + 1))
    zeta = curl_vertex_cgrid(u, v, geom)
    mw = (geom.dx_u[..., None][..., 0], geom.dx_v[..., None][..., 0],
          geom.dy_u[..., None][..., 0], geom.dy_v[..., None][..., 0])
    return dict(grid=geom, zeta=zeta, h_vtx=h_vtx, h_v=h_v, v=v, h_u=h_u, u=u,
                u_mask=u_mask, v_mask=v_mask, vtx_mask=vtx_mask, h=h,
                metric_widths=mw)


def _flux(st, metric_widths):
    return pv_flux_al81_partial_cell(
        st["zeta"], st["h_vtx"], st["h_v"], st["v"], st["h_u"], st["u"],
        st["u_mask"], st["v_mask"], st["vtx_mask"],
        metric_widths=metric_widths)


def _work(st, du, dv, physical):
    """Domain-summed work, in the physical or the per-area kinetic-energy norm.

    Physical: the cell VOLUME e1*e2*h. Per-area: the thickness h alone.
    Interior u-faces only (the periodic wrap column is a duplicate of column 0
    and would be double-counted).
    """
    g = st["grid"]
    wu = (g.dx_u * g.dy_u)[..., None] if physical else 1.0
    wv = (g.dx_v * g.dy_v)[..., None] if physical else 1.0
    a = float(jnp.sum((wu * st["h_u"] * st["u"] * du * st["u_mask"])[:, :-1]))
    b = float(jnp.sum(wv * st["h_v"] * st["v"] * dv * st["v_mask"]))
    return a + b


def _ke(st, physical):
    g = st["grid"]
    wu = (g.dx_u * g.dy_u)[..., None] if physical else 1.0
    wv = (g.dx_v * g.dy_v)[..., None] if physical else 1.0
    return float(
        jnp.sum((wu * st["h_u"] * st["u"] ** 2 * st["u_mask"])[:, :-1])
        + jnp.sum(wv * st["h_v"] * st["v"] ** 2 * st["v_mask"]))


def test_default_is_bit_identical():
    """``metric_widths=None`` must not change a single bit of the legacy form."""
    st = _state()
    a = _flux(st, None)
    b = pv_flux_al81_partial_cell(
        st["zeta"], st["h_vtx"], st["h_v"], st["v"], st["h_u"], st["u"],
        st["u_mask"], st["v_mask"], st["vtx_mask"])
    assert float(jnp.max(jnp.abs(a[0] - b[0]))) == 0.0
    assert float(jnp.max(jnp.abs(a[1] - b[1]))) == 0.0


def test_physical_energy_norm_is_conserved_only_with_metric():
    """The property the weighting exists for, asserted in both directions."""
    st = _state()
    du_off, dv_off = _flux(st, None)
    du_on, dv_on = _flux(st, st["metric_widths"])

    phys_off = abs(_work(st, du_off, dv_off, True)) / _ke(st, True)
    phys_on = abs(_work(st, du_on, dv_on, True)) / _ke(st, True)
    area_off = abs(_work(st, du_off, dv_off, False)) / _ke(st, False)

    # WITH the metric: the physical norm conserves to round-off.
    assert phys_on < 1e-15, (
        f"the metric-weighted flux does work {phys_on:.3e} per unit physical "
        "kinetic energy; NEMO's form must telescope to zero")
    # WITHOUT it: the same norm does NOT conserve, by many orders. This is the
    # half that fails if someone applies the weighting unconditionally, and the
    # half that fails if the operator is silently "fixed" elsewhere.
    assert phys_off > 1e4 * phys_on, (
        f"the unweighted flux conserves the physical norm to {phys_off:.3e}, "
        f"barely different from the weighted {phys_on:.3e} -- either the test "
        "grid has uniform metrics (it must not) or the weighting is being "
        "applied when it was not asked for")
    # And the unweighted form conserves the PER-AREA norm instead, which is the
    # positive statement about what it has always been doing.
    assert area_off < 1e-15, (
        f"the unweighted flux does not conserve the per-area norm either "
        f"({area_off:.3e}) -- the operator is broken independently of this "
        "option")


def test_metric_widths_is_equivalent_to_an_external_fold():
    """Folding the widths in externally must give the same answer.

    This is how the barotropic path has always applied the same weighting
    (``barotropic_latlon_cgrid.een_barotropic_coriolis``): scale the transport
    velocity by the neighbour width, divide the assembled tendency by the local
    one. If the in-operator version disagrees, one of the two is wrong.
    """
    st = _state()
    e1u, e1v, e2u, e2v = st["metric_widths"]
    du_on, dv_on = _flux(st, st["metric_widths"])
    st2 = dict(st)
    st2["h_v"] = st["h_v"] * e1v[..., None]
    st2["h_u"] = st["h_u"] * e2u[..., None]
    du_ext, dv_ext = _flux(st2, None)
    du_ext = du_ext / e1u[..., None]
    dv_ext = dv_ext / e2v[..., None]
    # The two folds differ only in the ORDER of one multiplication
    # (``h*width`` then ``*u``, versus ``h*u`` then ``*width``), so they agree
    # to floating-point reordering and no further. The u-tendency -- the one
    # the DINO wall investigation scores -- comes out at 1e-16 relative. The
    # v-tendency's four triad terms cancel by about eight orders on this state,
    # which amplifies that 1-ulp reordering to ~1e-8 relative; the ABSOLUTE
    # difference is still 4.5e-16, so both are asserted, absolute and relative.
    for nm, got, ref, rtol in (("du", du_on, du_ext, 1e-14),
                               ("dv", dv_on, dv_ext, 1e-6)):
        d = float(jnp.max(jnp.abs(got - ref)))
        scale = float(jnp.max(jnp.abs(ref)))
        assert d <= 1e-14, f"{nm}: absolute difference {d:.3e} is not roundoff"
        assert d <= rtol * scale, (
            f"{nm}: the in-operator weighting and the external fold disagree "
            f"by {d / scale:.3e} relative -- they are the same algebra, so one "
            "of them is wrong")


def test_malformed_metric_widths_raises():
    st = _state()
    with pytest.raises(ValueError, match="metric_widths must be the 4-tuple"):
        _flux(st, st["metric_widths"][:3])


def test_unknown_een_metric_weighting_raises():
    """Dispatch hardening: an unknown selector may never fall through."""
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux
    st = _state()
    z = jnp.zeros_like(st["u"])
    with pytest.raises(ValueError, match="unknown een_metric_weighting"):
        _bc_pv_flux(
            z, jnp.zeros_like(st["v"]), st["u"], st["v"], st["h_u"], st["h_v"],
            st["h"], st["u_mask"], st["v_mask"],
            jnp.ones((st["h"].shape[0], st["h"].shape[1])), st["grid"],
            "vector_invariant", "split",
            een_metric_weighting="nemoo")


# ----------------------------------------------------------------------
# The card selection, locked.
# ----------------------------------------------------------------------

def test_dino_mlf_card_selects_the_nemo_weighting():
    """The oracle card must ship what NEMO actually computes.

    Goes RED if the card is ever silently reverted to the per-unit-width form.
    That matters more than usual here because the revert is INVISIBLE in every
    90-day acceptance metric -- the A/B measured every gated number moving
    inside its noise floor -- so nothing else in the suite would notice.
    """
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    assert dino_config_for_recipe(
        "nemo_dino_kamm_mlf").een_metric_weighting == "nemo", (
        "the NEMO MLF oracle card no longer selects NEMO's own transport "
        "metric weighting (dynvor.F90:791-792, :804). It ships what NEMO "
        "computes; reverting it is a fidelity regression that no acceptance "
        "gate can see.")


def test_card_selection_reaches_the_model_config():
    """A card value nobody reads is not a selection.

    The card field and the model-config field are separate objects; this pins
    the pass-through, which is the half a card-only assertion cannot cover.
    """
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_model_config, dino_lat_lon_grid)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    mc, _ = dino_lat_lon_model_config(dino_lat_lon_grid(cfg), cfg)
    assert mc.een_metric_weighting == "nemo"


def test_other_cards_are_unchanged():
    """The flip is one card, not a global default change."""
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    for recipe in ("legoesm_default", "nemo_dino_kamm", "veros", "mitgcm"):
        assert dino_config_for_recipe(recipe).een_metric_weighting == "off", (
            f"{recipe} picked up the NEMO transport metric weighting; the "
            "#1455 flip was scoped to the NEMO MLF oracle card alone")
