"""TurbulenceOutput.cloud_fraction — the PDF cloud-fraction exposure hook.

A moist higher-order closure (CLUBB) diagnoses a sub-grid cloud fraction from
its assumed PDF; this new optional field lets it flow to radiation
(cloud_scheme="clubb") instead of the RH-diagnosed grid-scale one.  These pin
the FIELD contract: default None (byte-identical for schemes without a PDF
cloud closure) + settable (the CLUBB path).  The physical cloud-fraction
diagnosis itself is covered by tests/unit/test_clubb_diagnostic.py.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput


def _dummy(ncol=2, nlev=4):
    z = jnp.zeros((ncol, nlev))
    s = jnp.zeros((ncol,))
    return TurbulenceOutput(z, z, z, z, z, z, s, s, s, s)


def test_cloud_fraction_defaults_none_byte_identical():
    """Schemes that don't set it (louis, tke, ...) get None => radiation keeps
    the grid-scale cloud scheme; the 10-positional-arg construction is unchanged."""
    out = _dummy()
    assert out.cloud_fraction is None
    # the pre-existing 10 fields are untouched / positionally stable
    assert out.du_dt.shape == (2, 4) and out.h_pbl.shape == (2,)


def test_cloud_fraction_settable_for_pdf_scheme():
    """CLUBB sets the ADG1-PDF liquid cloud fraction on the field."""
    cf = jnp.full((2, 4), 0.3)
    out = _dummy()._replace(cloud_fraction=cf)
    assert out.cloud_fraction is not None
    assert out.cloud_fraction.shape == (2, 4)
    assert bool(jnp.all((out.cloud_fraction >= 0.0) & (out.cloud_fraction <= 1.0)))
    # keyword construction with the field also works
    z = jnp.zeros((2, 4)); s = jnp.zeros((2,))
    out2 = TurbulenceOutput(
        du_dt=z, dv_dt=z, dT_dt=z, dq_v_dt=z, Km=z, Kh=z,
        shflx=s, lhflx=s, ustar=s, h_pbl=s, cloud_fraction=cf)
    assert bool(jnp.array_equal(out2.cloud_fraction, cf))
