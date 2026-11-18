"""CATKE closure KERNEL vs the ORIGINAL reference code, Oceananigans v0.116.1 CATKEVerticalDiffusivity.

Scope: kernel parity with oracle-fed state only.  It does NOT certify column parity: the port carries
e on interfaces and Oceananigans at centres; a one-step column comparison (report-only, in
oracle_audit/catke_carbonate_oracle) differs by a median 0.3-0.8 % in κc on the 128 m fixtures and
37 % on the 24 m column.

``oracle_fixtures/catke_oceananigans_v0_116_1.npz`` holds Oceananigans' own face/centre inputs and
outputs on seven single-column fixtures (free convection, wind, stable heating, entrainment, quiescent,
24 m column, and one where the entrainment length binds the tracer diffusivity), evaluated with the oracle's κuᶜᶜᶠ/κcᶜᶜᶠ/κeᶜᶜᶠ, dissipation_rate and top TKE flux.
The port is fed the oracle's own face N², N²_above, S², depth, height above bottom, H, Jb and
e_face := (oracle face w★)² (harness glue: Oceananigans carries e at centres and averages w★ to faces).

Gated at rel 1e-12: κu, κc, κe on non-convecting faces; the surface TKE flux.
KNOWN DIFFERENCES, pinned per case (rel 1e-6, abs 1e-12), from the different staggering:
- convecting faces: the convective length uses ℑz(w★³) in Oceananigans, (ℑz w★)³ here (max 0.74 %);
- centre dissipation rate: Oceananigans rebuilds centre Ri as the mean of face Ri, here N²/S² of
  centre-averaged N² and centre shear (max 9.7 % on F1-F6; swapping in the oracle Ri closes it to 9e-16).
  F7 adds one degenerate centre: zero shear at the mixed-layer base, where the face Ri jumps from -inf
  to +inf; the oracle's dissipation length there equals the column depth (gap 116x, pinned; the
  other 31 centres match to 7e-16).
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing import catke as ck
from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig

jax.config.update("jax_enable_x64", True)

_FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "catke_oceananigans_v0_116_1.npz")
_NAMES = [str(n) for n in _FIX["fixtures"]]
_CFG = CATKEConfig()


def _g(n, k):
    return _FIX[f"{n}__{k}"]


def _faces(n):
    args = [jnp.asarray(_g(n, "f_wstar")) ** 2] + [jnp.asarray(_g(n, f"f_{k}"))
                                                  for k in ("N2", "N2_above", "S2", "depth", "hab")]
    return ck.catke_diffusivities(*args, float(_g(n, "H")), float(_g(n, "Jb")), _CFG)


def test_default_bottom_coefficient_is_reference():
    # Tripwire on the default value only; the oracle gates below carry the behaviour.
    assert _CFG.c_bottom_shear == 0.967


@pytest.mark.parametrize("n", _NAMES)
def test_diffusivities_match_oceananigans(n):
    conv = (_g(n, "Jb") > _CFG.minimum_convective_buoyancy_flux) & (_g(n, "f_N2") < 0)
    assert (~conv).any()
    for k, ours in zip(("ku", "kc", "ke"), _faces(n)):
        ref = _g(n, f"f_{k}")
        ours = np.asarray(ours)
        np.testing.assert_allclose(ours[~conv], ref[~conv], rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(ours / ref - 1.0, _g(n, f"pin_{k}_rel"), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("n", _NAMES)
def test_dissipation_rate_gap_is_pinned(n):
    om = ck.catke_dissipation_rate(*(jnp.asarray(_g(n, f"c_{k}")) for k in ("e", "N2", "N2_above", "S2", "depth", "hab")),
                                   float(_g(n, "H")), float(_g(n, "Jb")), _CFG)
    np.testing.assert_allclose(np.asarray(om) / _g(n, "c_omega") - 1.0, _g(n, "pin_omega_rel"), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("n", _NAMES)
def test_surface_tke_flux_matches_oceananigans(n):
    q = ck.catke_surface_tke_flux(float(_g(n, "ustar")), max(float(_g(n, "Jb")), 0.0) * float(_g(n, "dz_top")), _CFG)
    np.testing.assert_allclose(float(q), float(_g(n, "Qe")), rtol=1e-12, atol=1e-20)
