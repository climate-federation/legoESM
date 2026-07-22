"""jax-mps negative-stride-reverse mis-fusion: probe, workaround, and the
downstream invariant it protects (2026-07-21 AMIP/CMIP audit).

On jax-mps <= 0.10.7 a 1-D ``lax.rev`` fused into an elementwise or scatter
consumer is silently dropped (``u[::-1] * 2`` returns ``u * 2``).  Cube-sphere
grid construction syncs corner lon/lat across faces via
``arr.at[face, ...].set(strip[::-1])``, so the bug corrupted the corner
metric matrices (negative Arakawa-Lamb determinants) and every cubed-sphere
run on mps blew up with 1e14 wind tendencies.  ``configure_backend`` now
re-routes ``rev_p`` through gather on the affected platform.

The determinant invariant is asserted backend-agnostically so this file has
teeth on CPU CI too: a healthy A-L gradient matrix has det(grad) > 0 at every
corner (the 2x2 system row vectors are east-ish/north-ish displacements).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp


def _has_mps() -> bool:
    try:
        return any(d.platform == "mps" for d in jax.devices())
    except Exception:
        return False


def test_reverse_fusion_correct_on_default_backend():
    """u[::-1] must actually reverse — fused with mul and scatter consumers.

    Runs on whatever backend the suite uses; on mps this exercises the
    configure_backend workaround, on CPU/GPU it is a trivial pass that
    pins the semantics the workaround must reproduce.
    """
    from legoesm.runtime.backend import configure_backend

    configure_backend()
    u = jnp.arange(13.0, dtype=jnp.float32)
    want = np.arange(13.0, dtype=np.float32)[::-1]

    got_mul = np.asarray(jax.jit(lambda t: t[::-1] * 2.0)(u))
    np.testing.assert_array_equal(got_mul, want * 2.0)

    x = jnp.zeros((6, 13, 13), jnp.float32)
    got_scatter = np.asarray(x.at[0, 0, :].set(u[::-1]))[0, 0]
    np.testing.assert_array_equal(got_scatter, want)

    got_interp = np.asarray(jax.jit(lambda t: 0.5 * (t[::-1][:-1] + t[::-1][1:]))(u))
    np.testing.assert_allclose(got_interp, 0.5 * (want[:-1] + want[1:]))


def test_cdgrid_al_gradient_determinant_positive():
    """Corner A-L gradient matrix must be well-conditioned on every corner.

    det(grad) = 1/(R^2 det(A)); the build clamps det(A) at 1e-20, so a
    corrupted corner shows up as |grad| ~ 1e12 (vs ~1e-6 healthy).  Guards
    the exact failure mode the mps rev bug produced, on ANY backend.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    base = create_cubed_sphere(12)
    cd = create_cubed_sphere_cdgrid(base)
    g00 = np.asarray(cd.grad_c00, dtype=np.float64)
    g01 = np.asarray(cd.grad_c01, dtype=np.float64)
    g10 = np.asarray(cd.grad_c10, dtype=np.float64)
    g11 = np.asarray(cd.grad_c11, dtype=np.float64)
    for name, g in (("g00", g00), ("g01", g01), ("g10", g10), ("g11", g11)):
        assert np.isfinite(g).all(), f"{name} has non-finite entries"
        assert np.abs(g).max() < 1e-3, (
            f"{name} magnitude {np.abs(g).max():.3e} — corner metric "
            "det collapsed to the 1e-20 clamp (halo/reverse corruption?)"
        )
    det_grad = g00 * g11 - g01 * g10
    assert (det_grad > 0).all(), (
        f"{int((det_grad <= 0).sum())} corners with det <= 0 — "
        "corner lon/lat sync produced a degenerate/flipped tangent basis"
    )


@pytest.mark.skipif(not _has_mps(), reason="mps backend not available")
def test_mps_probe_and_workaround_idempotent():
    """The probe detects the platform truthfully and re-applying is safe."""
    from legoesm.runtime.backend import (
        _apply_mps_rev_workaround, configure_backend,
    )

    configure_backend()
    _apply_mps_rev_workaround()  # second call must not raise or corrupt
    u = jnp.arange(7.0, dtype=jnp.float32)
    got = np.asarray(jax.jit(lambda t: t[::-1] * 3.0)(u))
    np.testing.assert_array_equal(got, np.arange(7.0, dtype=np.float32)[::-1] * 3.0)
