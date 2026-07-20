"""CLM-ML turbulence-scheme selection against the REAL clm-ml-jax (importorskip).

The physics claim under test is an identity, not an approximation.  CLM-ML's
Harman & Finnigan roughness-sublayer ψ is Monin-Obukhov similarity plus a
roughness-sublayer correction ψ̂ (Bonan et al. 2018, eqs. A16/A19):

    psim = -psim1 + psim2 + c1*psihat_m(za) - c1*psihat_m(hc) + vkc/beta
    psic = -psic1 + psic2 + c1*psihat_h(za) - c1*psihat_h(hc)

so ``turbulence_scheme="most"`` must reduce these EXACTLY to the MOST terms.
That is a statement about the ψ FUNCTIONS only: β, the displacement height and
the ``vkc/beta`` canopy-top anchor still come from Harman & Finnigan canopy-drag
theory, so "most" is CLM-ML without its RSL correction, NOT the two-leaf
big-leaf surface layer.
legoESM implements that by pointing CLM-ML's ψ̂ lookup tables at zeros, which
only works while every ψ̂ consumer reads those tables — the gates below go red
if a clm-ml-jax refactor breaks that assumption, because a silent failure here
means "most" would quietly run RSL physics.

Sign/coordinate convention in scope: ψ are the integrated stability corrections
for the profile between canopy top ``hc`` and reference height ``za`` (both
heights measured UP from the ground), entering as ``ustar = uref*vkc/(zlog +
psim)``.  Removing ψ̂ must leave every remaining term, and their signs,
untouched — asserted term by term rather than by a norm.
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

turb = pytest.importorskip("multilayer_canopy.MLCanopyTurbulenceMod")

from legoesm.land.canopy.clm_ml_interface import (  # noqa: E402
    _PSIHAT_TABLE_ATTRS,
    _apply_turbulence_scheme,
    _ensure_clm_initialized,
)

# US-MMS-like tall deciduous canopy: 27 m canopy under a 46 m tower.
ZA, HC = 46.0, 27.0
DISP = 0.67 * HC
BETA, PRSC = 0.35, 0.5
# Both stability regimes — ψ̂ is tabulated separately for each sign of dt/L.
OBU_UNSTABLE, OBU_STABLE = -50.0, 200.0


@pytest.fixture(scope="module", autouse=True)
def _clm_ready():
    """Populate the ψ̂ lookup tables, and leave the process on the default."""
    _ensure_clm_initialized()
    yield
    _apply_turbulence_scheme("rsl_bonan")


def _psihat_lookups(zdt, dtL):
    """Every ψ̂ entry point: JAX (``_GetPsiRSL``) and scalar (root solvers)."""
    return {
        "M_jax": float(turb._LookupPsihatM(zdt, dtL)),
        "H_jax": float(turb._LookupPsihatH(zdt, dtL)),
        "M_scalar": turb._LookupPsihatM_scalar(zdt, dtL),
        "H_scalar": turb._LookupPsihatH_scalar(zdt, dtL),
    }


def test_rsl_default_has_a_nonzero_psihat_correction():
    """Non-vacuity gate: the term the 'most' option removes is really there.

    Without this, the zero-assertions below would pass against a ψ̂ that was
    already zero for unrelated reasons.
    """
    _apply_turbulence_scheme("rsl_bonan")
    values = _psihat_lookups(0.5, -0.2)
    assert all(abs(v) > 1.0e-6 for v in values.values()), values


@pytest.mark.parametrize("dtL", [-0.2, 0.2])
def test_most_zeroes_every_psihat_entry_point(dtL):
    """All four consumers — JAX and scalar, momentum and heat — must be 0."""
    _apply_turbulence_scheme("most")
    values = _psihat_lookups(0.5, dtL)
    assert all(v == 0.0 for v in values.values()), values


@pytest.mark.parametrize("obu", [OBU_UNSTABLE, OBU_STABLE])
def test_most_reduces_getpsirsl_to_pure_monin_obukhov(obu):
    """``psim``/``psic`` must equal the MOST terms alone, term for term."""
    _apply_turbulence_scheme("most")
    psim, psic, psim2, psim_hat2 = turb._GetPsiRSL(ZA, HC, DISP, obu, BETA, PRSC)

    psim1_mo = float(turb._psim_monin_obukhov((ZA - DISP) / obu))
    psim2_mo = float(turb._psim_monin_obukhov((HC - DISP) / obu))
    psic1_mo = float(turb._psic_monin_obukhov((ZA - DISP) / obu))
    psic2_mo = float(turb._psic_monin_obukhov((HC - DISP) / obu))

    # Momentum keeps the vkc/beta canopy-top anchor (u(hc) = ustar/beta);
    # only the two c1*psihat terms drop out.
    assert float(psim) == pytest.approx(
        -psim1_mo + psim2_mo + turb.vkc / BETA, rel=1e-12, abs=1e-12)
    assert float(psic) == pytest.approx(
        -psic1_mo + psic2_mo, rel=1e-12, abs=1e-12)
    # psim2 is the MOST ψ at hc and must be unaffected by the scheme;
    # psim_hat2 feeds the WITHIN-canopy wind profile and must vanish too, or
    # the in-canopy and above-canopy profiles would use different theories.
    assert float(psim2) == pytest.approx(psim2_mo, rel=1e-12, abs=1e-12)
    assert float(psim_hat2) == 0.0


@pytest.mark.parametrize("obu", [OBU_UNSTABLE, OBU_STABLE])
def test_rsl_and_most_actually_differ(obu):
    """The two schemes must produce different exchange, in both regimes."""
    _apply_turbulence_scheme("rsl_bonan")
    psim_rsl, psic_rsl, _, hat_rsl = turb._GetPsiRSL(ZA, HC, DISP, obu, BETA, PRSC)
    _apply_turbulence_scheme("most")
    psim_most, psic_most, _, _ = turb._GetPsiRSL(ZA, HC, DISP, obu, BETA, PRSC)

    assert float(hat_rsl) != 0.0
    assert abs(float(psim_rsl) - float(psim_most)) > 1.0e-6
    assert abs(float(psic_rsl) - float(psic_most)) > 1.0e-6


def test_switching_back_restores_the_rsl_tables_bit_for_bit():
    """No cross-run contamination: the tables are process-global CLM state.

    A one-shot destructive zeroing would leave every later ``rsl_bonan`` run in
    the same process silently running MOST.
    """
    _apply_turbulence_scheme("rsl_bonan")
    before = {a: getattr(turb, a).copy() for a in _PSIHAT_TABLE_ATTRS}

    _apply_turbulence_scheme("most")
    _apply_turbulence_scheme("rsl_bonan")

    for attr, ref in before.items():
        restored = getattr(turb, attr)
        assert restored.shape == ref.shape
        assert bool((restored == ref).all()), attr
    # And the restored tables are the RSL ones, not zeros.
    assert any(abs(v) > 1.0e-6 for v in _psihat_lookups(0.5, -0.2).values())
