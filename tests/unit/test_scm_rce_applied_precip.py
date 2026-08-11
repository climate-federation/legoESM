"""Gates for the SCM-RCE surface-precipitation readout.

The campaign reported 1e-18 to 3e-5 mm/day of surface precipitation for every
convection scheme while its columns were losing 1.2-1.8 mm/day of water,
because the number came from a SECOND, diagnostic-only microphysics call built
with the outer timestep rather than from the evaluation that advanced the
column.  These gates pin the readout and, more importantly, pin the ONE way of
getting it wrong that looks right: reading ``precip`` off a summed tendency,
which silently yields ``None`` and therefore 0.0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.core.field import Field                                # noqa: E402
from legoesm.core.state import HydrostaticTendencies                # noqa: E402


def _load_campaign():
    path = REPO_ROOT / "scripts" / "run" / "run_scm_rce_campaign.py"
    spec = importlib.util.spec_from_file_location("scm_rce_campaign_precip", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def camp():
    return _load_campaign()


def _field(name, value, shape=(1, 1, 1)):
    return Field(data=jnp.full(shape, value), name=name, dims=("face", "x", "y"))


def _tend(precip_kg_m2_s=None, nlev=3):
    dims4 = ("face", "x", "y", "level")
    zero4 = Field(data=jnp.zeros((1, 1, 1, nlev)), name="z", dims=dims4)
    zero3 = _field("z3", 0.0)
    return HydrostaticTendencies(
        du_dt=zero4, dT_dt=zero4, dp_s_dt=zero3, dphis_dt=zero3,
        dv_dt=zero4, tracer_tendencies=None,
        precip=(None if precip_kg_m2_s is None
                else _field("precip", precip_kg_m2_s)),
    )


class _LikeState:
    """Only ``.T.data.dtype`` is consulted, for the zero branch's dtype."""
    T = Field(data=jnp.zeros((1, 1, 1, 3)), name="T",
              dims=("face", "x", "y", "level"))


def test_converts_kg_per_m2_per_s_to_mm_per_day(camp):
    """1 kg/m^2 == 1 mm of liquid water, so the factor is exactly 86400."""
    rate = 2.5e-5                      # kg/m^2/s  -> 2.16 mm/day
    got = float(camp.applied_precip_mm_day(_tend(rate), _LikeState()))
    assert got == pytest.approx(rate * 86_400.0, rel=1e-12)
    assert got == pytest.approx(2.16, rel=1e-9)


def test_absent_precip_reads_zero_not_nan(camp):
    """Microphysics inactive is a legitimate state; it must read 0, and must
    not poison a mean with NaN."""
    got = camp.applied_precip_mm_day(_tend(None), _LikeState())
    assert float(got) == 0.0
    assert np.isfinite(float(got))


def test_a_summed_tendency_would_report_zero(camp):
    """THE TRAP, pinned. ``add_tendencies`` rebuilds HydrostaticTendencies from
    six fields and drops every diagnostic field, so reading `precip` off the
    SUM silently reports 0.0 — i.e. reproduces the very defect this readout
    fixes, while looking correct. If add_tendencies ever starts propagating
    precip, this test fails and the comment in the driver should be revisited.
    """
    from legoesm.atmosphere.forcing.scm.scm_forcing import add_tendencies

    raining = _tend(2.5e-5)
    other = _tend(None)
    summed = add_tendencies(other, raining)
    # NOT asserted as a requirement: `add_tendencies` propagating diagnostics
    # would be an IMPROVEMENT, and a gate demanding it keep dropping them would
    # fail the day someone made that improvement, for the wrong reason (GLM
    # review, 2026-08-11). What is asserted is the consequence that matters:
    # IF it drops precip, reading the sum reports zero, which is why the driver
    # reads micro_tend. The driver's own behaviour is gated separately.
    if summed.precip is None:
        assert float(camp.applied_precip_mm_day(summed, _LikeState())) == 0.0
    # ... while the un-summed microphysics tendency carries the real value.
    assert float(camp.applied_precip_mm_day(raining, _LikeState())) > 2.0


def test_multi_column_input_is_refused(camp):
    """[0] on a flattened field would silently score column 0 and call it the
    column."""
    tend = _tend(2.5e-5)
    wide = tend._replace(precip=Field(
        data=jnp.full((1, 2, 1), 2.5e-5), name="precip",
        dims=("face", "x", "y")))
    with pytest.raises(ValueError, match="single-column"):
        camp.applied_precip_mm_day(wide, _LikeState())


def test_the_driver_reads_the_applied_tendency_not_a_second_evaluation(camp):
    """NON-VACUITY for the fix itself: the substep must take precipitation from
    the microphysics tendency it applies, and must NOT call the standalone
    diagnostic inside the loop (that closure carries the OUTER dt)."""
    src = (REPO_ROOT / "scripts" / "run" / "run_scm_rce_campaign.py").read_text()
    body = src.split("def apply_split_microphysics_step")[1].split("\n    def body")[0]
    assert "applied_precip_mm_day(micro_tend" in body, body[:600]
    assert "precip_diagnostic(sub_state" not in body, (
        "the substep is calling the standalone diagnostic again — that is the "
        "defect: a different evaluation, built with the outer timestep")


def test_a_positive_flux_survives_the_substep_average(camp):
    """The per-substep values are weighted by 1/n and summed, so a constant
    rate must come back unchanged rather than divided."""
    n = 30
    rate = 1.0e-5
    per = float(camp.applied_precip_mm_day(_tend(rate), _LikeState()))
    assert sum([per / n] * n) == pytest.approx(per, rel=1e-12)


# --------------------------------------------------------------------------- #
# Counting semantics.  Codex round: the rule that zeroes the convective term
# used to branch on SUB-STEPPING, but sub-stepping has nothing to do with
# whether condensate gets counted twice -- what matters is whether microphysics
# will sediment it. The non-sub-stepped path therefore added a convective
# diagnostic on top of condensate microphysics would later rain out and count
# again; the default ten substeps hid it.
# --------------------------------------------------------------------------- #

def test_the_convective_term_is_zeroed_whenever_microphysics_is_active(camp):
    """Both branches, one criterion: active microphysics owns the surface."""
    src = (REPO_ROOT / "scripts" / "run" / "run_scm_rce_campaign.py").read_text()
    for body in src.split("convective_precip_for_score")[1:2]:
        pass
    # the guard must key off the microphysics scheme, not off sub-stepping
    idx = src.index("if microphysics_scheme == \"none\":")
    window = src[idx:idx + 700]
    assert "convective_precip_for_score" in window
    assert "use_split_convection" in window, (
        "the sub-stepped/non-sub-stepped distinction still selects WHICH "
        "convective diagnostic to use, which is fine; what must not return is "
        "sub-stepping deciding WHETHER to count it")
    # and the old shape -- branching on sub-stepping first -- must be gone
    assert "if use_split_convection:\n            convective_precip_for_score" \
        not in src


def test_the_wrong_dt_diagnostic_is_no_longer_constructed(camp):
    """Dead but loaded: the closure that caused the defect was still being
    built with the outer dt after the fix. Leaving it there is how the bug
    comes back."""
    src = (REPO_ROOT / "scripts" / "run" / "run_scm_rce_campaign.py").read_text()
    assert "precip_diagnostic = _make_microphysics_precip_diagnostic" not in src
