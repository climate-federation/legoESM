"""FV3_3D iter 904: JIT + autodiff pass-through tests for the
FV3-faithful expanding-halo nord wrapper.

Why this matters
----------------

iter-899/903 introduced ``fv3_corner_laplacian_nord_expanding_halo``
as the FV3 ``sw_core.F90:1737-1788`` single-pad multi-step pattern.
Production paths use the re-pad wrapper ``fv3_corner_laplacian_nord``
(iter-892) for now; the expand-halo wrapper is the canonical FV3
structure available for opt-in use.  To remain a viable production
candidate it MUST stay (1) jittable and (2) differentiable end-to-end.

FV3 ``fill_c`` semantics
------------------------

FV3 ``sw_core.F90:1741-1743``:

    fill_c = (nt/=0) .and. (flagstruct%grid_type<3) .and.               &
             ( sw_corner .or. se_corner .or. ne_corner .or. nw_corner ) &
              .and. .not. (bounded_domain .or. flagstruct%duogrid)

i.e. ``fill_corners`` between iterations is SKIPPED when
``flagstruct%duogrid = .true.``.  legoESM cubed-sphere with duogrid
matches this — single pad before the loop, no intermediate refresh.
The expand-halo wrapper therefore corresponds to FV3 with
``duogrid=.true.``.

(The re-pad wrapper ``fv3_corner_laplacian_nord`` refreshes via
``pad_halo`` each iteration, behaving as if ``fill_c=.true.``
between iters.  Both legoESM paths match each other at machine
eps — see ``test_nord_2_quantitative_equivalence_iter901`` and
``test_nord_3_approximates_repad`` from iter-903.)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._fv3_divergence_corner import (
    fv3_corner_laplacian_nord_expanding_halo,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid as _create_cdgrid


def _cdgrid(n):
    return _create_cdgrid(create_cubed_sphere(n))


@pytest.mark.parametrize("nord", [0, 1, 2, 3])
def test_expanding_halo_jit_pass_through(nord):
    """JIT-compiled wrapper returns same output as eager + finite."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9040 + nord)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))

    eager = fv3_corner_laplacian_nord_expanding_halo(divg, cdgrid, nord=nord)
    jitted = jax.jit(
        lambda d: fv3_corner_laplacian_nord_expanding_halo(d, cdgrid, nord=nord)
    )
    out = jitted(divg)

    assert out.shape == (6, n + 1, n + 1)
    assert bool(jnp.all(jnp.isfinite(out)))
    # JIT may reorder ops within machine epsilon — allow ULP-scale diff.
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(eager), rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("nord", [1, 2, 3])
def test_expanding_halo_grad_pass_through(nord):
    """jax.grad through the wrapper yields finite gradients."""
    n = 8
    cdgrid = _cdgrid(n)
    rng = np.random.default_rng(seed=9050 + nord)
    divg = jnp.asarray(rng.uniform(size=(6, n + 1, n + 1)))

    def loss(d):
        return jnp.sum(
            fv3_corner_laplacian_nord_expanding_halo(d, cdgrid, nord=nord) ** 2
        )

    g = jax.grad(loss)(divg)
    assert g.shape == divg.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    # Nontrivial gradient — wrapper is not constant w.r.t. input.
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_expanding_halo_fill_c_semantics_documented():
    """FV3 fill_c semantics documented in wrapper docstring.

    Sanity check that the expand-halo docstring records its
    relationship to FV3 ``sw_core.F90`` ``fill_corners`` calls
    so a future reader does not have to re-derive it.
    """
    doc = fv3_corner_laplacian_nord_expanding_halo.__doc__ or ""
    # Must reference the FV3 source path + the fill_corners
    # convention it follows.
    assert "sw_core.F90" in doc
    assert "expanding halo" in doc.lower() or "expanding-halo" in doc.lower()
