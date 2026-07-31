"""Differentiability tests for the ocean OPERATOR and SCHEME layer.

The existing ``tests/unit/test_diff_ocean.py`` sweep covers the ocean at
MODEL level (``ocean_model``, ``ocean_model_latlon_cgrid``,
``ocean_model_mpas``), the EOS family, and one vertical-mixing scheme
(``richardson`` / ``kpp``).  This file is the complement: it puts
``jax.grad`` directly on the lat-lon C-grid *operators* and on the
selectable *schemes* that the model-level sweep never isolates.

Priority order (highest value first):

1. ``ocean/dynamics/latlon_cgrid_operators.py`` — ``pv_flux_ene`` and
   ``pv_flux_al81_partial_cell``.  Both were changed by the merged
   periodic-seam index fix (#1226 / PR #1382), which explicitly noted
   *"The 3-D baroclinic impact is not yet measured."*  They feed BOTH
   the barotropic Coriolis (``een_barotropic_coriolis``) and the 3-D
   baroclinic momentum path (``ocean_pe_latlon_cgrid._bc_vortcor``).
   The tests below assert the gradient is finite, non-zero and
   structured, and — the point of the exercise — that the ADJOINT is
   seam-consistent: the seam-mismatch functional
   ``Σ (diag_u[:, n_lon] - diag_u[:, 0])`` is structurally zero, so its
   gradient must vanish EXACTLY while a same-units seam functional
   ``Σ (diag_u[:, n_lon] + diag_u[:, 0])`` retains full sensitivity.
   A zonal-roll equivariance check on the gradient is the second,
   independent detector for the same defect class.

2. Ocean vertical-mixing / convection schemes beyond ``richardson``.

3. Ocean dynamics operators (``barotropic_common``,
   ``ocean_tendency_common``, ``eta_floor``, PGF helpers).

All tests run under ``JAX_ENABLE_X64=1``.  Grids are 8x16 / 6x12 with
3-4 levels so the whole file runs in seconds.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm import constants


# ============================================================================
# Shared helpers
# ============================================================================

def assert_gradient_ok(g, name="", min_nonzero_frac=0.1, require_structure=True):
    """Finite + genuinely non-zero + spatially structured.

    ``require_structure`` rejects a constant (all-identical) gradient,
    which is the signature of a collapsed / broadcast-only adjoint that
    a bare ``isfinite`` check would happily wave through.
    """
    g = jnp.asarray(g)
    assert jnp.all(jnp.isfinite(g)), f"{name}: gradient has NaN/Inf"
    nz = float(jnp.mean(jnp.abs(g) > 0.0))
    assert nz >= min_nonzero_frac, (
        f"{name}: only {nz * 100:.1f}% of entries non-zero "
        f"(need {min_nonzero_frac * 100:.0f}%)"
    )
    if require_structure and g.size > 1:
        # RELATIVE to the gradient's own scale: an operator whose output is
        # O(1e-8) (relative-vorticity-only PV flux) still has structure, and
        # an absolute floor would reject it for being small rather than flat.
        spread = float(jnp.max(g) - jnp.min(g))
        scale = float(jnp.max(jnp.abs(g)))
        assert spread > 1e-9 * scale, (
            f"{name}: gradient is spatially uniform (max-min={spread:.3e}, "
            f"scale={scale:.3e}) — the adjoint has collapsed to a broadcast"
        )


@pytest.fixture
def clear_jax_caches():
    """Drop XLA executables after a full-model gradient test.

    Accumulated model-graph compilations (not memory) are what aborts a
    long ocean pytest process — ``tests/ocean/unit/test_advection_grad_
    underflow.py`` SIGABRTs after ~36 of them at both 48 GB and 96 GB.
    Model-level classes below take this fixture; the pure-operator tests
    are cheap enough not to need it.
    """
    yield
    jax.clear_caches()


def taylor_remainders(loss_fn, x0, hs=(1e-2, 1e-3, 1e-4, 1e-5)):
    """|J(x+h·dx) - J(x) - h·<g,dx>| / h² for a steepest-descent dx.

    A correct first-order adjoint gives a BOUNDED, roughly constant
    sequence; a wrong gradient makes it blow up as h -> 0.
    """
    J0 = float(loss_fn(x0))
    g = jax.grad(loss_fn)(x0)
    gnorm = float(jnp.sqrt(jnp.sum(g ** 2)))
    assert gnorm > 0.0, "Taylor test: gradient is identically zero"
    dx = g / gnorm
    gdx = float(jnp.sum(g * dx))
    out = []
    for h in hs:
        Jh = float(loss_fn(x0 + h * dx))
        out.append(abs(Jh - J0 - h * gdx) / (h * h))
    return out


# ============================================================================
# PRIORITY 1 — pv_flux_ene / pv_flux_al81_partial_cell
# ============================================================================

_BIG_H = 1.0e30


def _wrap_lon(core):
    """(.., n_lon, ..) -> (.., n_lon+1, ..) with the periodic wrap column.

    Building every longitudinally-wrapped operator input this way ties
    column 0 and column ``n_lon`` to the SAME graph node, which is the
    convention the operators document and the precondition under which
    the seam-mismatch functional below is structurally zero.
    """
    return jnp.concatenate([core, core[:, 0:1]], axis=1)


def _pv_geometry(n_lat=8, n_lon=16, nlev=4, land=False, roll=0):
    """Static geometry for the PV-flux operators (not differentiated).

    ``roll`` rotates the land mask zonally — used by the roll-equivariance
    test, which rolls the geometry and the state together.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)

    ocean = np.ones((n_lat, n_lon))
    if land:
        # A coastline that STRADDLES the periodic seam (columns n_lon-1, 0, 1)
        # plus an isolated interior island — the seam is exactly where the
        # #1226 index bug lived, so the land case must probe it.
        j0 = n_lat // 2
        ocean[j0:j0 + 2, [n_lon - 1, 0, 1]] = 0.0
        ocean[2, 5:7] = 0.0
    ocean = np.roll(ocean, roll, axis=1)
    ocean_j = jnp.asarray(ocean, dtype=jnp.float64)

    is_active = jnp.broadcast_to(ocean_j[:, :, None], (n_lat, n_lon, nlev))
    if land:
        # Step bathymetry: the deepest level is dry on the southern half
        # (the partial-cell topology AL81 is supposed to handle).
        deep = jnp.asarray(
            (np.arange(n_lat)[:, None, None] >= n_lat // 2).astype(np.float64))
        bottom = jnp.asarray(
            (np.arange(nlev)[None, None, :] == nlev - 1).astype(np.float64))
        is_active = is_active * (1.0 - bottom * (1.0 - deep))

    u_mask_3d, v_mask_3d = compute_face_masks_3d(is_active)
    vtx_mask = compute_vertex_mask(ocean_j)
    return dict(
        grid=grid,
        n_lat=n_lat, n_lon=n_lon, nlev=nlev,
        is_active=is_active,
        u_mask_3d=jnp.asarray(u_mask_3d, dtype=jnp.float64),
        v_mask_3d=jnp.asarray(v_mask_3d, dtype=jnp.float64),
        vtx_mask=jnp.asarray(vtx_mask, dtype=jnp.float64),
    )


def _pv_cores(n_lat=8, n_lon=16, nlev=4, seed=0, roll=0):
    """Differentiated inputs, stored WITHOUT the redundant wrap column.

    ``zeta`` is an additive vertex-vorticity perturbation on top of the
    ``curl_vertex_cgrid(u, v)`` that the production caller computes, so
    the same fixture exercises both the composed (u,v -> curl -> q) path
    and the operator's own ``∂/∂ζ``.
    """
    rng = np.random.default_rng(seed)
    u = 0.1 * rng.standard_normal((n_lat, n_lon, nlev))
    v = 0.1 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    v[0] = 0.0            # pole walls
    v[-1] = 0.0
    zeta = 1.0e-6 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    h_cell = 50.0 + 10.0 * rng.random((n_lat, n_lon, nlev))
    if roll:
        u = np.roll(u, roll, axis=1)
        v = np.roll(v, roll, axis=1)
        zeta = np.roll(zeta, roll, axis=1)
        h_cell = np.roll(h_cell, roll, axis=1)
    return dict(
        u=jnp.asarray(u, dtype=jnp.float64),
        v=jnp.asarray(v, dtype=jnp.float64),
        zeta=jnp.asarray(zeta, dtype=jnp.float64),
        h_cell=jnp.asarray(h_cell, dtype=jnp.float64),
    )


def _pv_apply(cores, geom, scheme, use_f=False, q_boundary="neumann_fill"):
    """Pack cores into operator arguments and evaluate the PV flux."""
    from legoesm.grids.operators_latlon_cgrid import curl_vertex_cgrid
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        pv_flux_ene,
        pv_flux_al81_partial_cell,
        vertex_coriolis,
    )

    grid = geom["grid"]
    n_lon = geom["n_lon"]
    is_active = geom["is_active"]

    u = _wrap_lon(cores["u"])                     # (n_lat, n_lon+1, nlev)
    v = cores["v"]                                # (n_lat+1, n_lon, nlev)

    # MITgcm hFac convention: dry cells carry a BIG_H sentinel so the
    # min-rule returns the wet neighbour's thickness and a fully-dry
    # vertex gets q ~ 0.
    h_act = jnp.where(is_active > 0.5, cores["h_cell"], _BIG_H)
    h_west = jnp.roll(h_act, 1, axis=1)
    h_u = _wrap_lon(jnp.minimum(h_act, h_west))   # (n_lat, n_lon+1, nlev)
    zero_row = jnp.zeros_like(h_act[:1])
    h_v = jnp.concatenate(
        [zero_row, jnp.minimum(h_act[:-1], h_act[1:]), zero_row], axis=0)
    h_vtx_int = jnp.minimum(
        jnp.minimum(h_act[:-1], h_act[1:]),
        jnp.minimum(h_west[:-1], h_west[1:]),
    )
    h_vtx = jnp.concatenate([
        jnp.minimum(h_act[:1], h_west[:1]),
        h_vtx_int,
        jnp.minimum(h_act[-1:], h_west[-1:]),
    ], axis=0)
    h_vtx = _wrap_lon(h_vtx)                      # (n_lat+1, n_lon+1, nlev)

    zeta = curl_vertex_cgrid(u, v, grid) + _wrap_lon(cores["zeta"])
    f_vtx = vertex_coriolis(grid) if use_f else None

    args = (zeta, h_vtx, h_v, v, h_u, u,
            geom["u_mask_3d"], geom["v_mask_3d"], geom["vtx_mask"])
    if scheme == "ene":
        return pv_flux_ene(*args, f_vtx=f_vtx)
    if scheme == "al81":
        return pv_flux_al81_partial_cell(
            *args, f_vtx=f_vtx, q_boundary=q_boundary)
    raise ValueError(f"unknown pv scheme {scheme!r}")


def _pv_sq_loss(du, dv, n_lon):
    """Sum of squares over the PHYSICAL faces only.

    ``diag_u`` stores the periodic-seam face TWICE (column 0 and column
    ``n_lon``).  Summing all ``n_lon+1`` columns double-counts that one
    face, which makes the scalar depend on WHERE the array origin sits
    and silently breaks the zonal-roll invariance the equivariance test
    relies on.  Drop the redundant wrap column.
    """
    return jnp.sum(du[:, :n_lon, :] ** 2) + jnp.sum(dv ** 2)


# Full selectable surface: ENE (NEMO vor_ene) and AL81/EEN (NEMO vor_een),
# each in relative-only (f_vtx=None) and combined-q (f_vtx=f) form, plus
# the AL81 'nemo_live' q-boundary variant.
_PV_CASES = [
    ("ene", False, "neumann_fill"),
    ("ene", True, "neumann_fill"),
    ("al81", False, "neumann_fill"),
    ("al81", True, "neumann_fill"),
    ("al81", True, "nemo_live"),
]
_PV_IDS = [f"{s}-{'f' if f else 'rel'}-{q}" for s, f, q in _PV_CASES]


class TestPVFluxGradients:
    """``jax.grad`` of a scalar reduction of the PV flux w.r.t. the
    velocity / vorticity / thickness inputs."""

    @pytest.mark.parametrize("scheme,use_f,q_bnd", _PV_CASES, ids=_PV_IDS)
    @pytest.mark.parametrize("land", [False, True], ids=["allwet", "land"])
    def test_grad_finite_nonzero_structured(self, scheme, use_f, q_bnd, land):
        geom = _pv_geometry(land=land)
        cores = _pv_cores()

        def loss(c):
            du, dv = _pv_apply(c, geom, scheme, use_f, q_bnd)
            return _pv_sq_loss(du, dv, geom["n_lon"])

        g = jax.grad(loss)(cores)
        tag = f"pv[{scheme},{'f' if use_f else 'rel'},{q_bnd},land={land}]"
        # u/v/zeta all reach the output through distinct branches
        # (mass flux, mass flux, q) — each needs its own probe.
        assert_gradient_ok(g["u"], f"{tag} d/du", min_nonzero_frac=0.5)
        assert_gradient_ok(g["v"], f"{tag} d/dv", min_nonzero_frac=0.4)
        assert_gradient_ok(g["zeta"], f"{tag} d/dzeta", min_nonzero_frac=0.4)
        assert_gradient_ok(g["h_cell"], f"{tag} d/dh", min_nonzero_frac=0.4)

    @pytest.mark.parametrize("scheme,use_f,q_bnd", _PV_CASES, ids=_PV_IDS)
    def test_seam_mismatch_adjoint_vanishes(self, scheme, use_f, q_bnd):
        """#1226 / PR #1382 regression lock, stated in the ADJOINT.

        u-face column 0 and column ``n_lon`` are the SAME physical face.
        A seam-consistent operator therefore satisfies
        ``diag_u[:, n_lon] == diag_u[:, 0]`` IDENTICALLY (as a function
        of the inputs, not merely numerically at one state), so the
        seam-MISMATCH functional has an exactly-zero gradient.

        The pre-#1382 code rolled the ALREADY-WRAPPED (n_lon+1) array,
        which gave column 0 its own value as its west neighbour while
        column n_lon got the correct one — the two disagreed and the
        seam opened (21% relative seam in diag_u from seam-exact inputs;
        1.86e10 m^3 of fabricated volume per 68-substep window).  That
        defect shows up here as a NON-zero mismatch adjoint.

        Non-vacuity guard: the same-units seam SUM functional must keep
        full sensitivity, so a trivially-dead operator cannot pass.
        """
        geom = _pv_geometry(land=True)
        cores = _pv_cores()

        def seam_diff(c):
            du, _ = _pv_apply(c, geom, scheme, use_f, q_bnd)
            return jnp.sum(du[:, -1, :] - du[:, 0, :])

        def seam_sum(c):
            du, _ = _pv_apply(c, geom, scheme, use_f, q_bnd)
            return jnp.sum(du[:, -1, :] + du[:, 0, :])

        # Premise: the forward seam identity itself.
        du, _ = _pv_apply(cores, geom, scheme, use_f, q_bnd)
        seam_gap = float(jnp.max(jnp.abs(du[:, -1, :] - du[:, 0, :])))
        du_scale = float(jnp.max(jnp.abs(du)))
        assert du_scale > 0.0, "PV flux is identically zero — fixture is dead"
        assert seam_gap <= 1e-13 * du_scale, (
            f"{scheme}/{q_bnd}: FORWARD periodic seam is open — "
            f"|diag_u[n_lon]-diag_u[0]|max={seam_gap:.3e} vs "
            f"|diag_u|max={du_scale:.3e}"
        )

        g_diff = jax.grad(seam_diff)(cores)
        g_sum = jax.grad(seam_sum)(cores)
        ref = max(float(jnp.max(jnp.abs(g_sum[k]))) for k in g_sum)
        assert ref > 0.0, "seam-sum adjoint is dead — test would be vacuous"
        for k in ("u", "v", "zeta", "h_cell"):
            m = float(jnp.max(jnp.abs(g_diff[k])))
            assert m <= 1e-10 * ref, (
                f"{scheme}/{q_bnd}: seam-MISMATCH adjoint w.r.t. {k} is "
                f"non-zero ({m:.3e}, seam-sum reference {ref:.3e}) — the "
                "periodic seam is open in the adjoint (#1226 class)"
            )

    @pytest.mark.parametrize("scheme,use_f,q_bnd", _PV_CASES, ids=_PV_IDS)
    def test_grad_seam_columns_are_alive(self, scheme, use_f, q_bnd):
        """The seam-adjacent inputs must carry O(interior) sensitivity.

        A seam handled by zero-padding instead of a true periodic wrap
        would leave the gradient at the wrap-neighbouring columns
        (i = 0 and i = n_lon-1) an order of magnitude below the interior
        — silently degrading every adjoint that crosses the dateline.
        """
        geom = _pv_geometry(land=False)
        cores = _pv_cores()
        n_lon = geom["n_lon"]

        def loss(c):
            du, dv = _pv_apply(c, geom, scheme, use_f, q_bnd)
            return _pv_sq_loss(du, dv, geom["n_lon"])

        g = jax.grad(loss)(cores)["u"]
        col_norm = jnp.sqrt(jnp.sum(g ** 2, axis=(0, 2)))   # (n_lon,)
        interior = float(jnp.median(col_norm[2:n_lon - 2]))
        assert interior > 0.0
        for i in (0, 1, n_lon - 1):
            val = float(col_norm[i])
            assert val > 0.25 * interior, (
                f"{scheme}/{q_bnd}: d(loss)/du at seam column {i} is "
                f"{val:.3e} vs interior median {interior:.3e} — the "
                "periodic wrap is starving the seam adjoint"
            )

    @pytest.mark.parametrize("scheme,use_f,q_bnd", _PV_CASES, ids=_PV_IDS)
    @pytest.mark.parametrize("land", [False, True], ids=["allwet", "land"])
    def test_grad_zonal_roll_equivariance(self, scheme, use_f, q_bnd, land):
        """Independent detector for the same defect class as #1226.

        The PV-flux operators are pure array functions, periodic in the
        longitude axis; the lat-lon metric and the planetary ``f`` depend
        on latitude only.  Rotating the ENTIRE configuration (state,
        thickness, land mask) zonally by k must therefore rotate the
        output by k — and hence the gradient of a roll-invariant loss by
        k.  A seam index that is right on one column and wrong on
        another breaks this exactly at the wrap.
        """
        k = 3
        loss_kw = (scheme, use_f, q_bnd)

        def grad_at(roll):
            geom = _pv_geometry(land=land, roll=roll)
            cores = _pv_cores(roll=roll)

            def loss(c):
                du, dv = _pv_apply(c, geom, *loss_kw)
                return _pv_sq_loss(du, dv, geom["n_lon"])

            return jax.grad(loss)(cores)

        g0 = grad_at(0)
        gk = grad_at(k)
        for key in ("u", "v", "zeta", "h_cell"):
            expect = jnp.roll(g0[key], k, axis=1)
            got = gk[key]
            scale = float(jnp.max(jnp.abs(expect)))
            assert scale > 0.0, f"{key}: reference gradient is dead"
            err = float(jnp.max(jnp.abs(got - expect))) / scale
            assert err < 1e-10, (
                f"{scheme}/{q_bnd}/land={land}: gradient w.r.t. {key} is NOT "
                f"zonally roll-equivariant (rel err {err:.3e}) — a "
                "longitude-index asymmetry survives in the adjoint"
            )

    @pytest.mark.parametrize("scheme,use_f,q_bnd", _PV_CASES, ids=_PV_IDS)
    def test_taylor_second_order(self, scheme, use_f, q_bnd):
        """Gold-standard gradient-correctness check on the u branch."""
        geom = _pv_geometry(land=True)
        cores = _pv_cores()

        def loss(u_core):
            c = dict(cores, u=u_core)
            du, dv = _pv_apply(c, geom, scheme, use_f, q_bnd)
            return _pv_sq_loss(du, dv, geom["n_lon"])

        r = taylor_remainders(loss, cores["u"])
        assert all(np.isfinite(r)), f"{scheme}: Taylor remainders not finite: {r}"
        # 2nd-order: the remainder / h^2 ratio stays bounded as h shrinks.
        assert max(r) <= 20.0 * max(min(r), 1e-30), (
            f"{scheme}/{q_bnd}: Taylor remainder/h^2 is not bounded "
            f"({r}) — the adjoint disagrees with the forward operator"
        )

    def test_ene_and_al81_gradients_differ(self):
        """ENE and AL81 are different operators (2-point Sadourny vs the
        12-point triad), so their adjoints must differ too.  Guards
        against a dispatch regression that silently aliases one to the
        other — which no per-scheme finiteness check would catch."""
        geom = _pv_geometry(land=True)
        cores = _pv_cores()

        def g_of(scheme):
            def loss(c):
                du, dv = _pv_apply(c, geom, scheme, True, "neumann_fill")
                return _pv_sq_loss(du, dv, geom["n_lon"])
            return jax.grad(loss)(cores)["u"]

        g_ene, g_al = g_of("ene"), g_of("al81")
        rel = float(jnp.max(jnp.abs(g_ene - g_al))) / float(
            jnp.max(jnp.abs(g_ene)))
        assert rel > 1e-3, (
            f"ENE and AL81 adjoints are indistinguishable (rel diff "
            f"{rel:.3e}) — the vorticity_scheme dispatch may be aliased"
        )

    @pytest.mark.parametrize("scheme,use_f,q_bnd", _PV_CASES, ids=_PV_IDS)
    def test_jit_grad_matches_eager(self, scheme, use_f, q_bnd):
        geom = _pv_geometry(land=True)
        cores = _pv_cores()

        def loss(c):
            du, dv = _pv_apply(c, geom, scheme, use_f, q_bnd)
            return _pv_sq_loss(du, dv, geom["n_lon"])

        g_eager = jax.grad(loss)(cores)
        g_jit = jax.jit(jax.grad(loss))(cores)
        for key in g_eager:
            assert jnp.allclose(g_eager[key], g_jit[key], rtol=1e-10,
                                atol=1e-14), f"{scheme}: jit/eager grad differ ({key})"


class TestNeumannFillVertexAdjoint:
    """``neumann_fill_vertex`` is the q-boundary treatment inside both
    PV-flux operators and the WENO branch — an operator in its own right
    whose adjoint has never been probed."""

    @staticmethod
    def _fill(q_core, mask_core):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            neumann_fill_vertex,
        )
        return neumann_fill_vertex(_wrap_lon(q_core), _wrap_lon(mask_core))

    @pytest.mark.parametrize(
        "land_col",
        [
            7,
            pytest.param(0, marks=pytest.mark.xfail(
                strict=True,
                reason="BUG #1418: neumann_fill_vertex rolls the ALREADY-WRAPPED "
                       "(n_lat+1, n_lon+1) array, so the seam column takes ITSELF "
                       "as its west neighbour. Measured adjoint 0.000/1.000 at the "
                       "seam vs 0.500/0.500 interior. Same index-ordering family as "
                       "#1382/#1226, in a function #1382 did not touch. Strict-xfail "
                       "so it flips to a failure the moment the roll ordering is "
                       "fixed.")),
        ],
        ids=["interior-column", "seam-column"],
    )
    def test_fill_adjoint_reaches_both_zonal_neighbours(self, land_col):
        """A land vertex column flanked by two wet columns is filled with
        the AVERAGE of its west and east neighbours, so the adjoint must
        reach BOTH with equal weight.

        At the seam column this is exactly the ``roll`` ordering question
        that #1226 was about: ``jnp.roll(A, 1, axis=1)`` applied to the
        ALREADY-WRAPPED (n_lon+1) array returns ``A[:, n_lon] == A[:, 0]``
        at column 0, i.e. the vertex's OWN value instead of its true west
        neighbour ``A[:, n_lon-1]``.

        # BUG: MEASURED 2026-07-31 on branch test/diff-oli, n_lat=8,
        # n_lon=16, one all-land vertex column, interior row j=4:
        #   land_col = 7 (interior): dfill/dq_W = 0.500, dfill/dq_E = 0.500
        #   land_col = 0 (SEAM):     dfill/dq_W = 0.000, dfill/dq_E = 1.000
        # ``neumann_fill_vertex`` builds its zonal neighbours with
        # ``jnp.roll(filled, +/-1, axis=1)`` on the ALREADY-WRAPPED
        # ``(n_lat+1, n_lon+1)`` array, where column ``n_lon`` duplicates
        # column 0.  For ``A = [a_0 ... a_{n-1}, a_0]``,
        # ``roll(A, 1)[:, 0] = A[:, n_lon] = a_0`` — the seam vertex gets
        # ITSELF as its west neighbour instead of ``a_{n-1}``; the paired
        # mask roll makes ``m_w[:, 0] = m[:, 0] = 0`` at a land vertex, so
        # the true west neighbour is not even counted.  (Symmetrically
        # ``roll(A, -1)[:, n_lon] = a_0`` instead of ``a_1``.)  The final
        # ``filled.at[:, -1].set(filled[:, 0])`` re-sync then propagates
        # the one-sided column-0 value onto column ``n_lon``, OVERWRITING
        # the value column ``n_lon`` computed with the correct west
        # neighbour.
        # Consequence: PV ``q`` at a coastline touching the model's
        # longitude origin is a ONE-SIDED extrapolation, and the PV flux's
        # sensitivity to the true west-neighbour vorticity is DEAD.  The
        # seam stays CLOSED (col 0 == col n_lon), which is why the #1382
        # seam-mismatch adjoint test above passes and does not see this.
        # Same index-ordering family as #1226/PR #1382 ("roll the (n_lon)
        # array first, THEN wrap"), in a function that fix did not touch.
        """
        n_lat, n_lon = 8, 16
        rng = np.random.default_rng(3)
        q_core = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon)))
        mask = np.ones((n_lat + 1, n_lon))
        mask[:, land_col] = 0.0        # one fully-land vertex column
        mask_core = jnp.asarray(mask)

        j = n_lat // 2                 # interior row (no pole clamp)

        def loss(qc):
            return self._fill(qc, mask_core)[j, land_col]

        g = jax.grad(loss)(q_core)
        w = (land_col - 1) % n_lon
        e = (land_col + 1) % n_lon
        gw, ge = float(g[j, w]), float(g[j, e])
        assert abs(gw) > 0.0, (
            f"land column {land_col}: adjoint w.r.t. the WEST neighbour "
            f"(column {w}) is exactly zero (east neighbour weight {ge:.3f}) "
            "— the Neumann fill ignores it"
        )
        assert abs(gw - ge) < 1e-12, (
            f"land column {land_col}: W/E fill weights are asymmetric "
            f"(W={gw:.6f}, E={ge:.6f}) — the two zonal neighbours should "
            "contribute equally"
        )

    def test_fill_adjoint_finite_and_structured(self):
        """Generic health check of the fill adjoint on a realistic mask."""
        n_lat, n_lon = 8, 16
        rng = np.random.default_rng(4)
        q_core = jnp.asarray(rng.standard_normal((n_lat + 1, n_lon, 3)))
        mask = np.ones((n_lat + 1, n_lon))
        mask[0] = 0.0
        mask[-1] = 0.0
        mask[3:5, 2:5] = 0.0
        mask_core = jnp.asarray(mask)

        def loss(qc):
            return jnp.sum(self._fill(qc, mask_core) ** 2)

        g = jax.grad(loss)(q_core)
        assert_gradient_ok(g, "neumann_fill_vertex", min_nonzero_frac=0.5)


class TestBarotropicEENCoriolis:
    """``een_barotropic_coriolis`` — the barotropic consumer of
    ``pv_flux_al81_partial_cell`` (NEMO ``dyn_spg_ts::dyn_cor_2D``)."""

    @staticmethod
    def _setup(n_lat=8, n_lon=16, nlev=3, metric_complete=False):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            _build_een_barotropic_inputs,
        )
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            compute_face_masks,
        )

        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
        u_mask, v_mask = compute_face_masks(mask, grid=grid)
        h_k = jnp.full((n_lat, n_lon, nlev), 100.0, dtype=jnp.float64)
        pre = _build_een_barotropic_inputs(
            h_k, grid, mask, u_mask, v_mask, jnp.float64,
            metric_complete=metric_complete)
        return grid, pre

    @pytest.mark.parametrize("metric_complete", [False, True],
                             ids=["een", "een_metric"])
    def test_grad_wrt_barotropic_velocity(self, metric_complete):
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            een_barotropic_coriolis,
        )
        n_lat, n_lon = 8, 16
        _, pre = self._setup(n_lat, n_lon, metric_complete=metric_complete)
        rng = np.random.default_rng(11)
        U = jnp.asarray(0.05 * rng.standard_normal((n_lat, n_lon + 1)))
        U = _wrap_lon(U[:, :n_lon])          # seam-exact
        V = jnp.asarray(0.05 * rng.standard_normal((n_lat + 1, n_lon)))
        V = V.at[0].set(0.0).at[-1].set(0.0)

        def loss(args):
            cor_u, cor_v = een_barotropic_coriolis(args[0], args[1], pre)
            return jnp.sum(cor_u ** 2) + jnp.sum(cor_v ** 2)

        gU, gV = jax.grad(loss)((U, V))
        tag = f"een_barotropic_coriolis[{metric_complete}]"
        assert_gradient_ok(gU, f"{tag} d/dU", min_nonzero_frac=0.5)
        assert_gradient_ok(gV, f"{tag} d/dV", min_nonzero_frac=0.4)

    @pytest.mark.parametrize("metric_complete", [False, True],
                             ids=["een", "een_metric"])
    def test_seam_mismatch_adjoint_vanishes(self, metric_complete):
        """Same #1226 invariant one level up: the depth-integrated
        barotropic ``cor_u`` inherits the PV-flux seam identity."""
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            een_barotropic_coriolis,
        )
        n_lat, n_lon = 8, 16
        _, pre = self._setup(n_lat, n_lon, metric_complete=metric_complete)
        rng = np.random.default_rng(12)
        U_core = jnp.asarray(0.05 * rng.standard_normal((n_lat, n_lon)))
        V = jnp.asarray(0.05 * rng.standard_normal((n_lat + 1, n_lon)))
        V = V.at[0].set(0.0).at[-1].set(0.0)

        def seam(sign):
            def f(args):
                cor_u, _ = een_barotropic_coriolis(
                    _wrap_lon(args[0]), args[1], pre)
                return jnp.sum(cor_u[:, -1] + sign * cor_u[:, 0])
            return f

        g_diff = jax.grad(seam(-1.0))((U_core, V))
        g_sum = jax.grad(seam(+1.0))((U_core, V))
        ref = max(float(jnp.max(jnp.abs(g))) for g in g_sum)
        assert ref > 0.0, "barotropic seam-sum adjoint is dead"
        for name, g in zip(("U", "V"), g_diff):
            m = float(jnp.max(jnp.abs(g)))
            assert m <= 1e-10 * ref, (
                f"een_barotropic_coriolis[{metric_complete}]: seam-mismatch "
                f"adjoint w.r.t. {name} is {m:.3e} (ref {ref:.3e})"
            )


class TestVorticitySchemeModelLevel:
    """3-D BAROCLINIC path — the impact PR #1382 explicitly left unmeasured.

    ``vorticity_scheme`` selects which PV-flux operator the baroclinic
    momentum RHS uses (``ocean_pe_latlon_cgrid._bc_vortcor``).  Each
    selectable option must keep the momentum AND tracer adjoints alive
    through a full model step.
    """

    @staticmethod
    def _model(vorticity_scheme, een_q_boundary="neumann_fill"):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )

        nlev = 3
        grid = create_latlon_grid(n_lat=8, n_lon=16)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        # ``{ene,een}_total`` carry the planetary Coriolis INSIDE the
        # vorticity flux, and the model REJECTS them under the default
        # ``coriolis_scheme="matsuno_split"`` (the Matsuno rotation would
        # double-apply f).  ``explicit_ab2`` in turn requires an AB2/leapfrog
        # outer integrator (forward-Euler rotation is unconditionally
        # unstable).  So the ``_total`` cases run with that mandated pairing;
        # the relative-only ``al81``/``ene`` cases keep the default.  Each
        # case is an independent gradient-HEALTH probe, not a skill
        # comparison, so the differing integrator is not a confound here.
        extra = {}
        if vorticity_scheme in ("ene_total", "een_total"):
            extra = dict(coriolis_scheme="explicit_ab2",
                         outer_integrator="ab2")
        config = LatLonCGridOceanConfig.from_flat(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,
            vorticity_scheme=vorticity_scheme,
            een_q_boundary=een_q_boundary,
            **extra,
        )
        model = LatLonCGridOceanModel(grid, z_coord, config)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, land_lat_threshold=90.0)
        return model, state

    @pytest.mark.parametrize(
        "vorticity_scheme,q_bnd",
        [("al81", "neumann_fill"), ("al81", "nemo_live"),
         ("een_total", "neumann_fill"), ("een_total", "nemo_live"),
         ("ene", "neumann_fill"), ("ene_total", "neumann_fill")],
    )
    def test_grad_wrt_u_through_full_step(self, vorticity_scheme, q_bnd,
                                         clear_jax_caches):
        model, state = self._model(vorticity_scheme, q_bnd)
        key = jax.random.PRNGKey(7)
        u0 = 0.05 * jax.random.normal(key, state.u.data.shape)

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.u.data ** 2) + jnp.sum(out.v.data ** 2)

        g = jax.grad(loss)(u0)
        tag = f"3-D baroclinic vorticity_scheme={vorticity_scheme}/{q_bnd}"
        assert jnp.all(jnp.isfinite(g)), f"{tag}: gradient has NaN/Inf"
        assert float(jnp.mean(jnp.abs(g) > 0.0)) > 0.3, (
            f"{tag}: momentum adjoint is mostly dead")
        assert float(jnp.max(g) - jnp.min(g)) > 0.0, (
            f"{tag}: momentum adjoint is spatially uniform")

    @pytest.mark.parametrize("vorticity_scheme",
                             ["al81", "een_total", "ene", "ene_total"])
    def test_grad_wrt_T_through_full_step(self, vorticity_scheme,
                                         clear_jax_caches):
        """Cross-branch: the PV flux is a momentum term, but momentum
        feeds tracer advection, so a broken vorticity branch can sever
        the tracer adjoint too."""
        model, state = self._model(vorticity_scheme)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.T.data ** 2)

        g = jax.grad(loss)(state.T.data)
        assert_gradient_ok(
            g, f"3-D vorticity_scheme={vorticity_scheme} d/dT",
            min_nonzero_frac=0.3)


# ============================================================================
# PRIORITY 2 — ocean vertical mixing / convection / EOS schemes
# ============================================================================

class TestVerticalMixingSchemes:
    """Selectable ``VerticalMixingConfig.scheme`` values beyond the
    ``richardson``/``kpp`` pair the existing sweep covers."""

    @staticmethod
    def _model(scheme, nlev=4):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
        from legoesm.ocean.state import LatLonCGridOceanConfig
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.physics.combined import OceanPhysicsConfig
        from legoesm.ocean.physics.lateral_mixing.config import (
            LateralMixingConfig,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )

        grid = create_latlon_grid(n_lat=8, n_lon=16)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme=scheme),
            lateral_mixing=LateralMixingConfig(scheme="none"),
        )
        config = LatLonCGridOceanConfig.from_flat(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,
            implicit_vertical_mixing=True,
            physics=physics,
        )
        model = LatLonCGridOceanModel(grid, z_coord, config)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, land_lat_threshold=90.0)
        return model, state

    @pytest.mark.parametrize("scheme", ["constant", "tke", "catke"])
    def test_grad_wrt_T(self, scheme, clear_jax_caches):
        model, state = self._model(scheme)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.T.data ** 2)

        g = jax.grad(loss)(state.T.data)
        assert_gradient_ok(g, f"vmix={scheme} d/dT", min_nonzero_frac=0.3)

    @pytest.mark.parametrize("scheme", ["constant", "tke", "catke"])
    def test_grad_wrt_S(self, scheme, clear_jax_caches):
        """Salinity drives the same K profiles through the EOS/N² branch;
        a T-only probe can pass while the S branch is dead."""
        model, state = self._model(scheme)

        def loss(S_data):
            s = state._replace(S=state.S.replace(data=S_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.S.data ** 2)

        g = jax.grad(loss)(state.S.data)
        assert_gradient_ok(g, f"vmix={scheme} d/dS", min_nonzero_frac=0.3)


class TestCATKEClosureLeaf:
    """CATKE's diffusivity / dissipation kernels, exercised directly.

    The model-level probe above cannot separate a dead stability
    function from a dead mixing length; these do.
    """

    @staticmethod
    def _inputs(n_iface=6, seed=5):
        """Inputs spanning the CATKE stability function's TRANSITION band.

        The stability function is a smoothed step in ``Ri``; outside the
        band it saturates and ``dK/dS2`` is legitimately zero.  A fixture
        that sits entirely in the saturated tail would report a "dead"
        gradient that is really a scheme property, so ``N2``/``S2`` here
        are chosen to give ``Ri`` values straddling ``c_ri_lower``.
        """
        rng = np.random.default_rng(seed)
        e = jnp.asarray(1e-4 + 1e-3 * rng.random((n_iface,)))
        S2 = jnp.asarray(np.full((n_iface,), 1.0e-5))
        # Ri = N2/S2 in [-0.5, 1.5] across the interfaces.
        Ri = jnp.asarray(np.linspace(-0.5, 1.5, n_iface))
        N2 = Ri * S2
        N2_above = jnp.concatenate([N2[:1], N2[:-1]])
        depth = jnp.asarray(np.linspace(0.0, 400.0, n_iface))
        hab = jnp.asarray(np.linspace(400.0, 0.0, n_iface))
        H = jnp.asarray(400.0)
        Jb = jnp.asarray(1e-7)
        return e, N2, N2_above, S2, depth, hab, H, Jb

    def test_grad_diffusivities_wrt_tke(self):
        from legoesm.ocean.physics.vertical_mixing.catke import (
            catke_diffusivities, CATKEConfig,
        )
        e, N2, N2_above, S2, depth, hab, H, Jb = self._inputs()
        cfg = CATKEConfig()

        def loss(e_in):
            K_u, K_c, K_e = catke_diffusivities(
                e_in, N2, N2_above, S2, depth, hab, H, Jb, cfg)
            return jnp.sum(K_u ** 2 + K_c ** 2 + K_e ** 2)

        g = jax.grad(loss)(e)
        assert_gradient_ok(g, "catke_diffusivities d/de", min_nonzero_frac=0.5)

    def test_grad_diffusivities_wrt_N2_and_S2(self):
        """N² and S² enter through the Richardson stability function —
        a hard ``jnp.where`` on Ri would kill one of these."""
        from legoesm.ocean.physics.vertical_mixing.catke import (
            catke_diffusivities, CATKEConfig,
        )
        e, N2, N2_above, S2, depth, hab, H, Jb = self._inputs()
        cfg = CATKEConfig()

        def loss(args):
            K_u, K_c, K_e = catke_diffusivities(
                e, args[0], N2_above, args[1], depth, hab, H, Jb, cfg)
            return jnp.sum(K_u ** 2 + K_c ** 2 + K_e ** 2)

        gN2, gS2 = jax.grad(loss)((N2, S2))
        assert_gradient_ok(gN2, "catke d/dN2", min_nonzero_frac=0.4)
        assert_gradient_ok(gS2, "catke d/dS2", min_nonzero_frac=0.4)

    def test_grad_dissipation_rate(self):
        from legoesm.ocean.physics.vertical_mixing.catke import (
            catke_dissipation_rate, CATKEConfig,
        )
        e, N2, N2_above, S2, depth, hab, H, Jb = self._inputs()
        cfg = CATKEConfig()

        def loss(e_in):
            return jnp.sum(catke_dissipation_rate(
                e_in, N2, N2_above, S2, depth, hab, H, Jb, cfg) ** 2)

        g = jax.grad(loss)(e)
        assert_gradient_ok(g, "catke_dissipation_rate d/de",
                           min_nonzero_frac=0.5)

    def test_grad_surface_tke_flux(self):
        from legoesm.ocean.physics.vertical_mixing.catke import (
            catke_surface_tke_flux, CATKEConfig,
        )
        cfg = CATKEConfig()
        u_star = jnp.asarray(0.01)
        w3 = jnp.asarray(1e-6)

        def loss(args):
            return catke_surface_tke_flux(args[0], args[1], cfg)

        g_us, g_w3 = jax.grad(loss)((u_star, w3))
        assert jnp.isfinite(g_us) and jnp.isfinite(g_w3)
        assert abs(float(g_us)) > 0.0, "catke surface flux: d/du_star is zero"
        assert abs(float(g_w3)) > 0.0, "catke surface flux: d/dw_conv^3 is zero"


class TestAuxiliaryMixingSchemes:
    """Double diffusion, internal-wave mixing and tidal mixing are
    additive K contributions that never appear in the model-level
    T-gradient probe (they are gated behind ``implicit_vertical_mixing``
    and separate apply steps)."""

    def test_grad_double_diffusion(self):
        from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
            compute_ddm_diffusivity, DoubleDiffusionConfig,
        )
        rng = np.random.default_rng(6)
        n = 8
        N2 = jnp.asarray(1e-5 * rng.standard_normal((n,)))
        a_dTdz = jnp.asarray(1e-4 * rng.standard_normal((n,)))
        b_dSdz = jnp.asarray(1e-4 * rng.standard_normal((n,)))
        cfg = DoubleDiffusionConfig(enabled=True)

        def loss(args):
            out = compute_ddm_diffusivity(args[0], args[1], args[2], cfg)
            leaves = out if isinstance(out, tuple) else (out,)
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in leaves)

        gs = jax.grad(loss)((N2, a_dTdz, b_dSdz))
        for name, g in zip(("N2", "alpha_dTdz", "beta_dSdz"), gs):
            assert jnp.all(jnp.isfinite(g)), f"ddm d/d{name}: NaN/Inf"
        # The salt-finger / diffusive-convection branches are selected on
        # the SIGN of the two stratification terms, so at least the T and
        # S gradient branches must be live.
        assert float(jnp.max(jnp.abs(gs[1]))) > 0.0, "ddm d/dalpha_dTdz dead"
        assert float(jnp.max(jnp.abs(gs[2]))) > 0.0, "ddm d/dbeta_dSdz dead"

    def test_grad_internal_wave_mixing(self):
        from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
            compute_iwm_diffusivity, uniform_iwm_forcing, IWMConfig,
        )
        n_lat, n_lon, nlev = 4, 6, 5
        cfg = IWMConfig(enabled=True)
        forcing = uniform_iwm_forcing(cfg, (n_lat, n_lon), dtype=jnp.float64)
        rng = np.random.default_rng(8)
        # depth_cell is (..., nlev) at cell centres; dz_w and N2 live at the
        # (nlev-1) INTERIOR interfaces (NEMO zdfiwm e3w / rn2).
        depth_cell = jnp.asarray(
            np.broadcast_to(np.linspace(10.0, 400.0, nlev),
                            (n_lat, n_lon, nlev)).copy())
        dz_w = jnp.full((n_lat, n_lon, nlev - 1), 80.0, dtype=jnp.float64)
        H = jnp.full((n_lat, n_lon), 400.0, dtype=jnp.float64)
        N2 = jnp.asarray(1e-5 + 1e-6 * rng.random((n_lat, n_lon, nlev - 1)))

        def loss(N2_in):
            out = compute_iwm_diffusivity(
                forcing, depth_cell, dz_w, H, N2_in,
                cfg=cfg, rho_0=float(constants.rho_ocean))
            leaves = out if isinstance(out, tuple) else (out,)
            return sum(jnp.sum(jnp.asarray(x) ** 2) for x in leaves)

        g = jax.grad(loss)(N2)
        assert_gradient_ok(g, "iwm d/dN2", min_nonzero_frac=0.5)

    def test_grad_tidal_mixing(self):
        """Jayne & St-Laurent (2001) ``K = Gamma*q*E_BT*F(z)/(rho_0*N2)``.

        Two distinct adjoint branches: the tide-energy map ``E_BT``
        (linear) and the stratification ``N2`` (through a floor + a cap,
        the two places the gradient can die).
        """
        from legoesm.ocean.physics.vertical_mixing.tidal import (
            compute_tidal_diffusivity, TidalMixingConfig,
        )
        n_lat, n_lon, nlev = 4, 6, 5
        cfg = TidalMixingConfig(enabled=True)
        rng = np.random.default_rng(9)
        # N2 well ABOVE config.N_squared_min so the floor is not active
        # (below it the gradient is legitimately zero by construction).
        N2 = jnp.asarray(1.0e-5 + 1.0e-6 * rng.random((n_lat, n_lon, nlev)))
        E_BT = jnp.asarray(1.0e-3 * (1.0 + rng.random((n_lat, n_lon))))
        h_partial = jnp.full((n_lat, n_lon, nlev), 80.0, dtype=jnp.float64)
        H_bathy = jnp.full((n_lat, n_lon), 400.0, dtype=jnp.float64)
        layer_depths = jnp.asarray(
            np.broadcast_to(np.linspace(40.0, 360.0, nlev),
                            (n_lat, n_lon, nlev)).copy())

        def loss(args):
            K = compute_tidal_diffusivity(
                args[0], layer_depths, h_partial, H_bathy, args[1],
                config=cfg)
            return jnp.sum(K ** 2)

        gE, gN2 = jax.grad(loss)((E_BT, N2))
        assert_gradient_ok(gE, "tidal K d/dE_BT", min_nonzero_frac=0.9)
        assert_gradient_ok(gN2, "tidal K d/dN2", min_nonzero_frac=0.9)
        # SIGN: K ~ E_BT/N2, so K grows with the tide energy and shrinks
        # with stratification.  A same-sign pair means the 1/N2 has been
        # inverted and the scheme mixes MORE in a strongly stratified
        # column.
        def K_sum(args):
            return jnp.sum(compute_tidal_diffusivity(
                args[0], layer_depths, h_partial, H_bathy, args[1],
                config=cfg))

        dE, dN2 = jax.grad(K_sum)((E_BT, N2))
        assert float(jnp.min(dE)) > 0.0, (
            f"dK/dE_BT must be positive, got min {float(jnp.min(dE)):.3e}")
        assert float(jnp.max(dN2)) < 0.0, (
            f"dK/dN2 must be negative (stratification suppresses mixing), "
            f"got max {float(jnp.max(dN2)):.3e}")


class TestOceanConvectionSchemes:
    """``ocean/physics/convection`` — the enhanced-diffusion and plume
    schemes are dispatch-selectable and untested for AD."""

    @staticmethod
    def _column(n_lat=4, n_lon=6, nlev=5, seed=10):
        from legoesm.ocean.vertical import create_ocean_z_star
        rng = np.random.default_rng(seed)
        z_coord = create_ocean_z_star(nlev, H_max=400.0)
        # Unstable column at a few points so the convective branch fires.
        T = jnp.asarray(2.0 + 8.0 * rng.random((n_lat, n_lon, nlev)))
        S = jnp.asarray(34.0 + 1.0 * rng.random((n_lat, n_lon, nlev)))
        # The z* Jacobian is a per-COLUMN (horizontal) array, NOT per-level:
        # ``convective_K_A_flag`` derives ``dry_col = jacobian <= 0`` and then
        # expands it over the vertical.
        J = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
        return z_coord, T, S, J

    def test_grad_enhanced_diffusion(self):
        from legoesm.ocean.physics.convection.enhanced_diffusion import (
            enhanced_diffusion_convection,
        )
        from legoesm.ocean.physics.convection.config import (
            EnhancedDiffusionConfig,
        )
        from legoesm.ocean.eos import wright_eos
        z_coord, T, S, J = self._column()
        cfg = EnhancedDiffusionConfig()

        def loss(T_in):
            rho = wright_eos(T_in, S, jnp.zeros_like(T_in))
            out = enhanced_diffusion_convection(
                T_in, S, rho, z_coord, J, cfg, apply_diffusion=True, dt=300.0)
            return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dS_dt ** 2)

        g = jax.grad(loss)(T)
        assert_gradient_ok(g, "enhanced_diffusion d/dT", min_nonzero_frac=0.3)

    def test_grad_plume_convection(self):
        from legoesm.ocean.physics.convection.plume import plume_convection
        from legoesm.ocean.physics.convection.config import PlumeConfig
        from legoesm.ocean.eos import wright_eos
        z_coord, T, S, J = self._column()
        cfg = PlumeConfig()
        p_hydro = jnp.asarray(
            np.broadcast_to(
                np.linspace(0.0, 4.0e6, T.shape[-1]), T.shape).copy())

        def loss(T_in):
            rho = wright_eos(T_in, S, p_hydro)
            out = plume_convection(T_in, S, rho, p_hydro, z_coord, J, cfg)
            return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dS_dt ** 2)

        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g)), "plume_convection d/dT: NaN/Inf"
        assert float(jnp.max(jnp.abs(g))) > 0.0, (
            "plume_convection d/dT is identically zero — the plume "
            "trigger is not differentiable")


class TestEOSGsw:
    """``veros_gsw`` is the one member of ``VALID_EOS_SCHEMES`` the
    existing EOS gradient sweep omits (48-term TEOS-10 polynomial with a
    ``sqrt(S)`` branch — the most likely EOS to produce a NaN adjoint)."""

    def test_gsw_grad_signs(self):
        from legoesm.ocean.eos import veros_gsw_eos
        T = jnp.asarray(10.0)
        S = jnp.asarray(35.0)
        p = jnp.asarray(1.0e6)
        drho_dT = jax.grad(lambda t: veros_gsw_eos(t, S, p))(T)
        drho_dS = jax.grad(lambda s: veros_gsw_eos(T, s, p))(S)
        drho_dp = jax.grad(lambda q: veros_gsw_eos(T, S, q))(p)
        assert jnp.isfinite(drho_dT) and jnp.isfinite(drho_dS)
        assert jnp.isfinite(drho_dp)
        assert float(drho_dT) < 0.0, (
            f"veros_gsw: drho/dT must be negative (thermal expansion), "
            f"got {float(drho_dT):.4e}")
        assert float(drho_dS) > 0.0, (
            f"veros_gsw: drho/dS must be positive (haline contraction), "
            f"got {float(drho_dS):.4e}")
        assert float(drho_dp) > 0.0, (
            f"veros_gsw: drho/dp must be positive (compressibility), "
            f"got {float(drho_dp):.4e}")

    def test_gsw_analytic_derivative_matches_ad(self):
        """``veros_gsw_drhodT``/``drhodS`` are hand-differentiated
        kernels used by the buoyancy-frequency path.  They must agree
        with AD through ``veros_gsw_eos`` — a mismatch means the model's
        N² and its adjoint see different densities."""
        from legoesm.ocean.eos import (
            veros_gsw_eos, veros_gsw_drhodT, veros_gsw_drhodS,
        )
        T = jnp.asarray(8.0)
        S = jnp.asarray(34.5)
        p = jnp.asarray(2.0e6)
        ad_T = float(jax.grad(lambda t: veros_gsw_eos(t, S, p))(T))
        ad_S = float(jax.grad(lambda s: veros_gsw_eos(T, s, p))(S))
        an_T = float(veros_gsw_drhodT(T, S, p))
        an_S = float(veros_gsw_drhodS(T, S, p))
        assert abs(ad_T - an_T) <= 1e-8 * max(abs(ad_T), 1.0), (
            f"veros_gsw drho/dT: AD={ad_T:.8e} analytic={an_T:.8e}")
        assert abs(ad_S - an_S) <= 1e-8 * max(abs(ad_S), 1.0), (
            f"veros_gsw drho/dS: AD={ad_S:.8e} analytic={an_S:.8e}")

    def test_gsw_second_derivative_finite(self):
        """Cabbeling: d²rho/dT² must survive the sqrt(S) branch."""
        from legoesm.ocean.eos import veros_gsw_eos
        T, S, p = jnp.asarray(10.0), jnp.asarray(35.0), jnp.asarray(1.0e6)
        d2 = jax.grad(jax.grad(lambda t: veros_gsw_eos(t, S, p)))(T)
        assert jnp.isfinite(d2), "veros_gsw: d2rho/dT2 not finite"
        assert float(d2) != 0.0, "veros_gsw: d2rho/dT2 is exactly zero"


# ============================================================================
# PRIORITY 3 — ocean dynamics operators
# ============================================================================

class TestOceanTendencyCommon:
    """Shared baroclinic helpers (``ocean_tendency_common``) — every new
    ``ocean_pe_*`` scheme routes through these, so a dead adjoint here is
    a dead adjoint everywhere."""

    @pytest.mark.parametrize("quadrature",
                             ["cell_integral", "nemo_trapezoid"])
    def test_grad_iterate_eos_and_pressure_anomaly(self, quadrature):
        """The shared 2-pass EOS iteration + baroclinic pressure anomaly
        that every ``ocean_pe_*`` scheme routes through.

        The iteration is ``rho <- EOS(T, S, p_hydro(rho))`` unrolled
        ``n_iter`` times, so its adjoint is a short fixed-point chain: a
        dead T or S branch here kills the baroclinic pressure-gradient
        gradient in EVERY ocean dycore at once.  Both vertical
        quadratures are selectable and must be AD-connected.
        """
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            iterate_eos_and_pressure_anomaly,
        )
        from legoesm.ocean.eos import wright_eos
        n_lat, n_lon, nlev = 4, 6, 4
        rng = np.random.default_rng(19)
        T = jnp.asarray(4.0 + 6.0 * rng.random((n_lat, n_lon, nlev)))
        S = jnp.asarray(34.0 + rng.random((n_lat, n_lon, nlev)))
        mask = jnp.ones((n_lat, n_lon))
        dz_ref = jnp.asarray(np.full((nlev,), 100.0))

        def loss(args):
            rho, rho_p, p_prime = iterate_eos_and_pressure_anomaly(
                args[0], args[1], mask,
                lambda f: f,                 # no-op land fill (all-wet)
                wright_eos, dz_ref,
                float(constants.rho_ocean), float(constants.g),
                quadrature=quadrature,
            )
            return jnp.sum(p_prime ** 2) + jnp.sum(rho_p ** 2)

        gT, gS = jax.grad(loss)((T, S))
        assert_gradient_ok(gT, f"iterate_eos[{quadrature}] d/dT",
                           min_nonzero_frac=0.9)
        assert_gradient_ok(gS, f"iterate_eos[{quadrature}] d/dS",
                           min_nonzero_frac=0.9)
        # p' is a DOWNWARD cumulative integral of g·rho', so a
        # temperature perturbation at level k can only influence levels
        # >= k: the adjoint of the DEEPEST p' must reach every level,
        # while the adjoint of the SHALLOWEST must not reach below it.
        def p_at(k):
            def f(T_in):
                _, _, p_prime = iterate_eos_and_pressure_anomaly(
                    T_in, S, mask, lambda x: x, wright_eos, dz_ref,
                    float(constants.rho_ocean), float(constants.g),
                    quadrature=quadrature,
                )
                return p_prime[2, 3, k]
            return f

        g_top = np.asarray(jax.grad(p_at(0))(T))[2, 3]
        g_bot = np.asarray(jax.grad(p_at(nlev - 1))(T))[2, 3]
        assert np.all(np.abs(g_top[1:]) == 0.0), (
            f"{quadrature}: the SURFACE p' depends on deeper levels "
            f"(d p'[0]/dT = {g_top}) — the hydrostatic integral is "
            "not causal in depth")
        assert np.all(np.abs(g_bot) > 0.0), (
            f"{quadrature}: the DEEPEST p' is insensitive to some level "
            f"above it (d p'[-1]/dT = {g_bot}) — part of the column is "
            "missing from the hydrostatic adjoint")

    def test_grad_implicit_bottom_drag_factor(self):
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            implicit_bottom_drag_factor,
        )
        rng = np.random.default_rng(20)
        H = jnp.asarray(100.0 + 300.0 * rng.random((4, 6)))

        def loss(args):
            f = implicit_bottom_drag_factor(300.0, args[0], args[1])
            return jnp.sum(f ** 2)

        g_r, g_H = jax.grad(loss)((jnp.asarray(2.5e-3), H))
        assert jnp.isfinite(g_r) and float(jnp.abs(g_r)) > 0.0, (
            "implicit_bottom_drag_factor: d/d(drag_r) is dead — the drag "
            "coefficient is unreachable by AD")
        assert_gradient_ok(g_H, "implicit_bottom_drag_factor d/dH",
                           min_nonzero_frac=0.9)
        # Sign: deeper column -> weaker implicit drag -> factor closer to 1.
        assert float(g_r) != 0.0

    def test_grad_apply_sponge_tracer_relaxation(self):
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            apply_sponge_tracer_relaxation,
        )
        from legoesm.ocean.sponge import SpongeForcing
        rng = np.random.default_rng(21)
        shape = (4, 6, 3)
        T = jnp.asarray(5.0 + rng.random(shape))
        S = jnp.asarray(34.0 + rng.random(shape))
        dT = jnp.zeros(shape)
        dS = jnp.zeros(shape)
        gamma = jnp.asarray(1.0e-6 * np.ones((4, 6)))
        T_ref = jnp.asarray(4.0 * np.ones(shape))
        S_ref = jnp.asarray(35.0 * np.ones(shape))
        sponge = SpongeForcing(gamma=gamma, T_ref=T_ref, S_ref=S_ref)

        def loss(args):
            dTn, dSn = apply_sponge_tracer_relaxation(
                dT, dS, args[0], args[1], sponge)
            return jnp.sum(dTn ** 2) + jnp.sum(dSn ** 2)

        gT, gS = jax.grad(loss)((T, S))
        assert_gradient_ok(gT, "sponge d/dT", min_nonzero_frac=0.9)
        assert_gradient_ok(gS, "sponge d/dS", min_nonzero_frac=0.9)

        # SIGN: relaxation is a DAMPING, dT/dt = +gamma*(T_ref - T), so
        # d(dT/dt)/dT = -gamma < 0.  A positive value here is a runaway
        # source, and a finiteness check cannot see the flip.
        def tend_sum(T_in):
            dTn, _ = apply_sponge_tracer_relaxation(dT, dS, T_in, S, sponge)
            return jnp.sum(dTn)

        d = jax.grad(tend_sum)(T)
        assert float(jnp.max(d)) < 0.0, (
            f"sponge d(dT/dt)/dT = {float(jnp.max(d)):.3e} — restoring must "
            "be a damping (negative), this sign is a runaway source")

    def test_grad_apply_freshwater_virtual_salt_top(self):
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            apply_freshwater_virtual_salt_top,
        )
        from legoesm.ocean.freshwater import FreshwaterForcing
        rng = np.random.default_rng(22)
        shape = (4, 6, 3)
        dS = jnp.zeros(shape)
        precip = jnp.asarray(2.0e-5 * (1.0 + rng.random((4, 6))))
        evap = jnp.asarray(1.0e-5 * (1.0 + rng.random((4, 6))))
        runoff = jnp.asarray(5.0e-6 * rng.random((4, 6)))
        zeros = jnp.zeros((4, 6))
        h_top = jnp.asarray(10.0 + rng.random((4, 6)))
        mask = jnp.ones((4, 6))

        def loss(args):
            fw = FreshwaterForcing(
                precip=args[0], evap=args[1], runoff=args[2],
                ice_fw=zeros, restoring=zeros)
            out = apply_freshwater_virtual_salt_top(
                dS, fw, 35.0, args[3], float(constants.rho_ocean), mask)
            return jnp.sum(out ** 2)

        g_p, g_e, g_r, g_h = jax.grad(loss)((precip, evap, runoff, h_top))
        assert_gradient_ok(g_p, "virtual salt d/dprecip", min_nonzero_frac=0.9)
        assert_gradient_ok(g_r, "virtual salt d/drunoff", min_nonzero_frac=0.9)
        assert jnp.all(jnp.isfinite(g_h)), "virtual salt d/dh_top: NaN/Inf"

        # SIGN CONVENTION (evap is positive UPWARD = freshwater LEAVING the
        # ocean, precip/runoff positive INTO it): precip and evap must enter
        # the virtual-salt tendency with OPPOSITE sign.  A same-sign pair is
        # the classic evaporation-sign flip and it double-salinifies.
        def top_tend(args):
            fw = FreshwaterForcing(
                precip=args[0], evap=args[1], runoff=runoff,
                ice_fw=zeros, restoring=zeros)
            return jnp.sum(apply_freshwater_virtual_salt_top(
                dS, fw, 35.0, h_top, float(constants.rho_ocean), mask)[..., 0])

        dp, de = jax.grad(top_tend)((precip, evap))
        assert float(jnp.max(dp)) * float(jnp.max(de)) < 0.0, (
            f"d(dS/dt)/d(precip)={float(jnp.max(dp)):.3e} and "
            f"d(dS/dt)/d(evap)={float(jnp.max(de)):.3e} have the SAME sign — "
            "precip (into ocean) and evap (out of ocean) must dilute/"
            "concentrate salinity oppositely")

    def test_grad_ab2_blend_and_depth_mean(self):
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            ab2_blend, depth_mean,
        )
        rng = np.random.default_rng(23)
        f_new = jnp.asarray(rng.standard_normal((4, 6, 3)))
        f_old = jnp.asarray(rng.standard_normal((4, 6, 3)))

        def loss_ab2(args):
            return jnp.sum(ab2_blend(args[0], args[1], 0.1) ** 2)

        gn, go = jax.grad(loss_ab2)((f_new, f_old))
        assert_gradient_ok(gn, "ab2_blend d/df_new", min_nonzero_frac=0.9)
        assert_gradient_ok(go, "ab2_blend d/df_old", min_nonzero_frac=0.9)

        field = jnp.asarray(rng.standard_normal((4, 7, 3)))
        h_face = jnp.asarray(10.0 + rng.random((4, 7, 3)))

        def loss_dm(args):
            return jnp.sum(depth_mean(args[0], args[1], 1.0) ** 2)

        gf, gh = jax.grad(loss_dm)((field, h_face))
        assert_gradient_ok(gf, "depth_mean d/dfield", min_nonzero_frac=0.9)
        assert_gradient_ok(gh, "depth_mean d/dh_face", min_nonzero_frac=0.9)


class TestBarotropicCommon:
    """``barotropic_common`` blend / clip helpers sit inside the
    substep loop, so a broken adjoint there kills every barotropic
    gradient."""

    def test_grad_bebt_blend(self):
        from legoesm.ocean.dynamics.barotropic_common import bebt_blend
        rng = np.random.default_rng(30)
        eta_new = jnp.asarray(rng.standard_normal((4, 6)))
        eta_old = jnp.asarray(rng.standard_normal((4, 6)))

        def loss(args):
            return jnp.sum(bebt_blend(args[0], args[1], 0.281) ** 2)

        gn, go = jax.grad(loss)((eta_new, eta_old))
        assert_gradient_ok(gn, "bebt_blend d/deta_new", min_nonzero_frac=0.9)
        assert_gradient_ok(go, "bebt_blend d/deta_old", min_nonzero_frac=0.9)

    def test_grad_maxvel_clip_inside_and_outside_band(self):
        """The clip is a saturating nonlinearity: gradient must be 1
        inside the band and 0 outside.  A gradient of 1 everywhere means
        the clip is not applied; 0 everywhere means the barotropic
        adjoint is severed whenever the limiter is active."""
        from legoesm.ocean.dynamics.barotropic_common import maxvel_clip
        field = jnp.asarray([-30.0, -1.0, 0.0, 1.0, 30.0])
        g = jax.grad(lambda f: jnp.sum(maxvel_clip(f, 20.0)))(field)
        assert jnp.all(jnp.isfinite(g)), "maxvel_clip: NaN/Inf gradient"
        inside = np.asarray(g)[1:4]
        outside = np.asarray(g)[[0, 4]]
        assert np.allclose(inside, 1.0), (
            f"maxvel_clip: in-band gradient should be 1, got {inside}")
        assert np.allclose(outside, 0.0), (
            f"maxvel_clip: saturated gradient should be 0, got {outside}")

    def test_filter_weights_are_static_not_traced(self):
        """``compute_filter_weights`` takes a Python int substep count —
        an iteration COUNT, never a trainable.  Pin that it returns
        plain arrays that participate in a downstream gradient."""
        from legoesm.ocean.dynamics.barotropic_common import (
            compute_filter_weights,
        )
        w = compute_filter_weights(8, jnp.float64, use_cosine=False)
        arrs = [jnp.asarray(x) for x in w]
        assert all(jnp.all(jnp.isfinite(a)) for a in arrs)
        x = jnp.asarray(np.arange(arrs[0].shape[0], dtype=np.float64))

        def loss(xi):
            return jnp.sum(arrs[0] * xi)

        g = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.sum(jnp.abs(g))) > 0.0, (
            "barotropic filter weights are identically zero")


class TestEtaFloor:
    """``eta_floor.clamp_and_redistribute`` is a conservation fixer with
    a hard clamp + a global redistribution — the classic place for an
    adjoint to die (clamp) or to become non-local (global sum)."""

    def test_grad_clamp_and_redistribute(self):
        from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute
        rng = np.random.default_rng(40)
        n_lat, n_lon = 4, 6
        eta = jnp.asarray(0.1 * rng.standard_normal((n_lat, n_lon)))
        floor = jnp.full((n_lat, n_lon), -0.05, dtype=jnp.float64)
        mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
        area = jnp.asarray(1.0e10 * (1.0 + rng.random((n_lat, n_lon))))

        def loss(e):
            return jnp.sum(clamp_and_redistribute(e, floor, mask, area) ** 2)

        g = jax.grad(loss)(eta)
        assert_gradient_ok(g, "clamp_and_redistribute d/deta",
                           min_nonzero_frac=0.5)

    def test_redistribution_makes_adjoint_nonlocal(self):
        """The redistribution is a GLOBAL operation: perturbing one cell
        changes every other cell's floored eta.  The adjoint of a
        single-cell functional must therefore be non-zero away from that
        cell — if it is not, the redistribution has been silently
        detached from the graph (e.g. via ``lax.stop_gradient``) and
        volume is being created without the adjoint knowing."""
        from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute
        n_lat, n_lon = 4, 6
        # Deterministic: every cell starts WELL ABOVE the floor (so it has
        # headroom and receives the redistribution) except one cell driven
        # far below it, which is the mass source.  A randomly-clamped probe
        # cell would sit AT the floor, where the output is a constant and
        # every gradient is legitimately zero — a vacuous pass/fail.
        eta = jnp.full((n_lat, n_lon), 0.10, dtype=jnp.float64)
        eta = eta.at[1, 2].set(-0.5)
        floor = jnp.full((n_lat, n_lon), -0.05, dtype=jnp.float64)
        mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)
        area = jnp.full((n_lat, n_lon), 1.0e10, dtype=jnp.float64)

        def loss(e):
            return clamp_and_redistribute(e, floor, mask, area)[3, 5]

        g = np.asarray(jax.grad(loss)(eta))
        off = np.abs(g).sum() - abs(g[3, 5])
        assert np.isfinite(g).all(), "eta-floor adjoint has NaN/Inf"
        assert off > 0.0, (
            "eta-floor redistribution adjoint is purely local — the "
            "global mass redistribution is detached from the graph")


class TestPGFOperators:
    """Pressure-gradient-force helpers used by the z*/partial-cell
    baroclinic path."""

    def test_grad_partial_cell_pgf_correction(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            partial_cell_pgf_correction_x, partial_cell_pgf_correction_y,
        )
        n_lat, n_lon, nlev = 6, 12, 3
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        rng = np.random.default_rng(50)
        centroid = jnp.asarray(
            np.broadcast_to(np.linspace(5.0, 300.0, nlev),
                            (n_lat, n_lon, nlev)).copy()
            + 0.1 * rng.standard_normal((n_lat, n_lon, nlev)))
        rho_p = jnp.asarray(0.5 * rng.standard_normal((n_lat, n_lon, nlev)))

        def loss_x(r):
            return jnp.sum(partial_cell_pgf_correction_x(
                centroid, r, grid, float(constants.g)) ** 2)

        def loss_y(r):
            return jnp.sum(partial_cell_pgf_correction_y(
                centroid, r, grid, float(constants.g)) ** 2)

        gx = jax.grad(loss_x)(rho_p)
        gy = jax.grad(loss_y)(rho_p)
        assert_gradient_ok(gx, "partial_cell_pgf_x d/drho", min_nonzero_frac=0.4)
        assert_gradient_ok(gy, "partial_cell_pgf_y d/drho", min_nonzero_frac=0.4)

    def test_grad_density_jacobian_pgf_smc03(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            density_jacobian_pgf_smc03_x, density_jacobian_pgf_smc03_y,
        )
        n_lat, n_lon, nlev = 6, 12, 4
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        rng = np.random.default_rng(51)
        rho = jnp.asarray(float(constants.rho_ocean)
                          + rng.standard_normal((n_lat, n_lon, nlev)))
        # Partial cells: the deepest layer is thinner over a step, and
        # inactive on the southern half — the regime SMC03 exists for.
        h_partial = jnp.asarray(
            np.broadcast_to(np.array([50.0, 100.0, 150.0, 80.0])[:nlev],
                            (n_lat, n_lon, nlev)).copy())
        active = np.ones((n_lat, n_lon, nlev))
        active[: n_lat // 2, :, -1] = 0.0
        is_active = jnp.asarray(active)
        h_partial = h_partial * is_active

        def loss(r):
            fx = density_jacobian_pgf_smc03_x(
                r, h_partial, is_active, grid, float(constants.g))
            fy = density_jacobian_pgf_smc03_y(
                r, h_partial, is_active, grid, float(constants.g))
            return jnp.sum(fx ** 2) + jnp.sum(fy ** 2)

        g = jax.grad(loss)(rho)
        assert_gradient_ok(g, "pgf_smc03 d/drho", min_nonzero_frac=0.4)

        # The partial-cell thickness is the other differentiable input:
        # a dead d/dh means the PGF cannot feel bathymetry changes.
        def loss_h(h):
            fx = density_jacobian_pgf_smc03_x(
                rho, h, is_active, grid, float(constants.g))
            return jnp.sum(fx ** 2)

        gh = jax.grad(loss_h)(h_partial)
        assert jnp.all(jnp.isfinite(gh)), "pgf_smc03 d/dh_partial: NaN/Inf"
        assert float(jnp.max(jnp.abs(gh))) > 0.0, (
            "pgf_smc03 d/dh_partial is identically zero — the partial-cell "
            "thickness is unreachable by AD")


class TestViscosityOperators:
    """Lateral-viscosity operators on the lat-lon C-grid: each is a
    selectable ``lateral_viscosity`` option and none is gradient-tested."""

    @staticmethod
    def _state(n_lat=6, n_lon=12, nlev=3, seed=60):
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
        rng = np.random.default_rng(seed)
        u_core = jnp.asarray(0.1 * rng.standard_normal((n_lat, n_lon, nlev)))
        u = _wrap_lon(u_core)
        v = jnp.asarray(0.1 * rng.standard_normal((n_lat + 1, n_lon, nlev)))
        v = v.at[0].set(0.0).at[-1].set(0.0)
        return grid, u_core, v

    def test_grad_strain_rate(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            strain_rate_cgrid,
        )
        grid, u_core, v = self._state()

        def loss(args):
            D_T, D_S = strain_rate_cgrid(_wrap_lon(args[0]), args[1], grid)
            return jnp.sum(D_T ** 2) + jnp.sum(D_S ** 2)

        gu, gv = jax.grad(loss)((u_core, v))
        assert_gradient_ok(gu, "strain_rate d/du", min_nonzero_frac=0.5)
        assert_gradient_ok(gv, "strain_rate d/dv", min_nonzero_frac=0.4)

    def test_grad_smagorinsky_viscosity(self):
        """``A_smag = (C_smag*Delta)^2 * sqrt(D_T^2 + D_S^2 + eps)``.

        The ``sqrt`` is the classic NaN-gradient site at zero deformation
        (a rest state), which is exactly what the ``1e-30`` floor exists
        for — so BOTH a sheared state and a rest state are probed, and
        ``C_smag`` itself must be reachable by AD (it is a tunable).
        """
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            smagorinsky_viscosity_cgrid,
        )
        grid, u_core, v = self._state()

        def loss(args):
            A = smagorinsky_viscosity_cgrid(
                _wrap_lon(args[0]), args[1], grid, args[2])
            return jnp.sum(A ** 2)

        C_smag = jnp.asarray(0.2)
        gu, gv, gC = jax.grad(loss)((u_core, v, C_smag))
        assert_gradient_ok(gu, "smagorinsky d/du", min_nonzero_frac=0.5)
        assert_gradient_ok(gv, "smagorinsky d/dv", min_nonzero_frac=0.4)
        assert jnp.isfinite(gC) and float(jnp.abs(gC)) > 0.0, (
            "smagorinsky d/dC_smag is dead — the coefficient is "
            "unreachable by AD")

        # REST state: |D| = 0, so the sqrt sits exactly on its branch
        # point.  Without the epsilon floor this returns NaN and every
        # cold-start ocean adjoint is poisoned at step 0.
        zero_u = jnp.zeros_like(u_core)
        zero_v = jnp.zeros_like(v)
        g_rest = jax.grad(loss)((zero_u, zero_v, C_smag))
        for name, g in zip(("u", "v", "C_smag"), g_rest):
            assert jnp.all(jnp.isfinite(g)), (
                f"smagorinsky d/d{name} is NaN/Inf at a REST state — the "
                "sqrt branch-point guard has regressed")

    def test_grad_bilaplacian_cgrid(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            bilaplacian_cgrid,
        )
        grid, u_core, _ = self._state()
        field = jnp.asarray(u_core[:, :, 0])

        def loss(f):
            return jnp.sum(bilaplacian_cgrid(f, grid) ** 2)

        g = jax.grad(loss)(field)
        assert_gradient_ok(g, "bilaplacian_cgrid", min_nonzero_frac=0.5)
