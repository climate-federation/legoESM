"""Every WB campaign deck's spectral timestep must satisfy the advective CFL.

The semi-implicit step treats ONLY the linear gravity-wave subsystem
implicitly; the RK3 advection stays explicit with a spectral stability bound

    u_ref * n_max * dt_sub / R_earth  <=  sqrt(3),   dt_sub = dt / si_substeps

(imaginary-axis limit of SSP-RK3 on the fastest resolved advective mode).
Measured 2026-08-23: the T63 campaign at dt=1800 with si_substeps=1
(CFL 2.67 at u_ref) blew up exponentially within 12 steps on every level
count, and every WB training run — classical and learned arms — optimized
against those mid-blowup 6-hour states (dycore-only p_s error 9.5 kPa vs a
0.26 kPa real 6-hour change).  dt<=900, or dt=1800 with si_substeps>=3, is
stable (~0.9 kPa).  The decks used to carry the comment "semi-implicit
stable large dt", which this test exists to keep dead.

u_ref is a gate parameter, not physics: 150 m/s bounds the strongest
resolved stratospheric jets with margin (the measured T63 blowup threshold
is ~97 m/s at dt_sub=1800).
"""
from pathlib import Path

import pytest
import yaml

# One bound, one home: the runtime guard (check_advective_cfl in
# spectral_pe.py, called from the training integrator factory) and this deck
# gate share the same constants and formula.
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    ADVECTIVE_CFL_U_REF as _U_REF,
)
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    RK3_ADVECTIVE_BOUND as _RK3_BOUND,
)
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    advective_cfl_courant,
)

_CAMPAIGN_DIR = Path(__file__).resolve().parents[2] / "config" / "wb" / "campaign"

_DECKS = sorted(_CAMPAIGN_DIR.glob("*.yaml"))


def _advective_cfl(spec: dict) -> float:
    # si_substeps only subdivides the step on the SEMI-IMPLICIT path
    # (neural_gcm_spectral._make_spectral_integrator); an explicit deck
    # advances at the full dt whatever the key says (codex diff review).
    n_sub = int(spec.get("si_substeps", 1))
    assert n_sub >= 1, f"si_substeps must be >= 1, got {n_sub}"
    dt_sub = float(spec["dt"]) / (n_sub if spec.get("semi_implicit", True) else 1)
    return advective_cfl_courant(int(spec["n_max"]), dt_sub)


@pytest.mark.parametrize("deck", _DECKS, ids=lambda p: p.stem)
def test_campaign_deck_within_advective_cfl(deck):
    spec = yaml.safe_load(deck.read_text()).get("spectral")
    if spec is None or "dt" not in spec or "n_max" not in spec:
        pytest.skip(f"{deck.name}: no spectral dt/n_max block")
    cfl = _advective_cfl(spec)
    assert cfl <= _RK3_BOUND, (
        f"{deck.name}: advective CFL {cfl:.2f} > {_RK3_BOUND:.2f} at "
        f"u_ref={_U_REF} m/s (n_max={spec['n_max']}, dt={spec['dt']}, "
        f"si_substeps={spec.get('si_substeps', 1)}). This is the exact "
        "configuration class that blew up the 2026-08 WB campaign; raise "
        "si_substeps or lower dt.")


def test_gate_is_not_vacuous():
    # The pre-fix T63 campaign (dt 1800, si_substeps 1) must FAIL the bound,
    # and the shipped fix (si_substeps 3) must pass it.
    assert _advective_cfl({"n_max": 63, "dt": 1800.0}) > _RK3_BOUND
    assert _advective_cfl({"n_max": 63, "dt": 1800.0, "si_substeps": 3}) \
        <= _RK3_BOUND
    # Substeps must NOT be credited on the explicit path (integrator ignores
    # them there).
    assert _advective_cfl({"n_max": 63, "dt": 1800.0, "si_substeps": 3,
                           "semi_implicit": False}) > _RK3_BOUND


def test_decks_found():
    assert len(_DECKS) >= 3, f"campaign dir moved? {_CAMPAIGN_DIR}"


def test_runtime_guard_check_advective_cfl():
    """The runtime companion of this gate (enforced when the training
    integrator is built) must refuse the measured blowup config and accept
    the shipped fix — including NOT crediting substeps on the explicit path."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig,
        check_advective_cfl,
    )

    bad = SpectralPEConfig(semi_implicit=True, si_substeps=1)
    with pytest.raises(ValueError, match="advective Courant"):
        check_advective_cfl(63, 1800.0, bad)
    check_advective_cfl(63, 1800.0, bad._replace(si_substeps=3))
    with pytest.raises(ValueError, match="advective Courant"):
        check_advective_cfl(
            63, 1800.0, bad._replace(si_substeps=3, semi_implicit=False))
