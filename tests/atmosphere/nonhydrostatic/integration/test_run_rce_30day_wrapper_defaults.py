"""Regression test for the plane CRM 30-day wrapper script defaults.

Locks the iter-14 production-measured config as a contract on
``scripts/run_rce_30day.sh`` env-var defaults. iter-58 found this
wrapper still had ``DT=1.0`` + ``N_ACOUSTIC=24`` in its
documentation (citing the iter-1 F1 ladder pre-F10 fix), while the
iter-14 production measurement + iter-38 structural regression
both use ``DT=5.0`` + ``N_ACOUSTIC=12``.

This test parses the bash-script env-var defaults via a simple
regex and asserts they match the iter-14 / iter-38 production
contract. A future edit that silently reverts the defaults trips
this in < 1 s before any of the slow nightly tests would catch
the drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
WRAPPER = REPO_ROOT / "scripts" / "run_rce_30day.sh"


def _parse_env_default(text: str, name: str) -> str | None:
    """Extract the ``${X:-VAL}`` default for env-var ``X`` from a
    bash assignment like ``X="${X:-VAL}"``. Returns the unquoted
    default string or ``None`` if no assignment found."""
    m = re.search(
        rf'^\s*{re.escape(name)}\s*=\s*"\$\{{{re.escape(name)}:-([^}}]+)\}}"\s*$',
        text, re.MULTILINE,
    )
    if m is None:
        return None
    return m.group(1).strip()


@pytest.fixture(scope="module")
def wrapper_text() -> str:
    assert WRAPPER.exists(), f"wrapper script missing: {WRAPPER}"
    return WRAPPER.read_text()


def test_dt_default_matches_iter14_production(wrapper_text):
    """iter-14 measured the 132x132 plane CRM 1-sim-hour PASS at
    dt=5.0 s (725 steps, max|w|=6.1e-3 m/s). The wrapper must
    default to that value; iter-58 found it stale at 1.0 s."""
    dt = _parse_env_default(wrapper_text, "DT")
    assert dt == "5.0", (
        f"run_rce_30day.sh DT default = {dt!r}, expected '5.0' "
        f"(iter-14 + iter-38 production contract)."
    )


def test_n_acoustic_default_matches_iter14_production(wrapper_text):
    """iter-14 + iter-38 measure N_ACOUSTIC=12 for dt=5.0 s.
    The wrapper must default to that value; iter-58 found it
    stale at 24 (the iter-1 dt=1.0 s value)."""
    n_ac = _parse_env_default(wrapper_text, "N_ACOUSTIC")
    assert n_ac == "12", (
        f"run_rce_30day.sh N_ACOUSTIC default = {n_ac!r}, "
        f"expected '12' (iter-14 + iter-38 production contract). "
        f"With dt=5.0 + N_ACOUSTIC=24 the acoustic CFL ratio "
        f"would halve, producing a slower / over-stable substep."
    )


def test_other_production_defaults(wrapper_text):
    """Lock the remaining production defaults so a partial revert
    surfaces the same way."""
    defaults = {
        "NX": "132",
        "NY": "132",
        "ADVECTION": "upwind1",
        "HYPERDIFF": "5.0e6",
        "BUBBLE_K": "0.0",   # F7 / F10 contract: clean Wing IC
        "QV_NOISE": "0.0",   # F7 / F10 contract: no qv noise
        # iter-61 Codex MEDIUM coverage gap fix: env defaults that
        # iter-58 missed.
        "DAYS": "30",
        "RANKS": "12",
        "USE_DD": "0",   # legacy rank-0-broadcast is the F8-stable default
    }
    for name, expected in defaults.items():
        actual = _parse_env_default(wrapper_text, name)
        assert actual == expected, (
            f"run_rce_30day.sh {name} default = {actual!r}, "
            f"expected {expected!r} (iter-12/14/38 production contract)."
        )


# iter-61 Codex MEDIUM silent-pass fix: the wrapper passes several
# production-relevant flags HARDCODED in the mpirun argv (not via
# env vars). A partial revert of these would slip past both this
# test + the driver-defaults test. Lock them here.


_HARDCODED_PASSTHROUGH_FLAGS = {
    "--acoustic-off-centering": "0.1",  # iter-14/38 production beta
    "--snapshot-hours": "24.0",          # daily snapshots
    "--snapshot-3d-hours": "1.0",        # hourly 3D snapshots
    "--profile-days": "5.0",
    "--log-every-steps": "100",
    # --semi-implicit-acoustic is a store_true; verified separately.
}


@pytest.mark.parametrize(
    "flag,expected_value",
    sorted(_HARDCODED_PASSTHROUGH_FLAGS.items()),
)
def test_wrapper_hardcoded_driver_flag(wrapper_text, flag, expected_value):
    """Each ``flag VALUE`` pair appears verbatim in the wrapper's
    mpirun invocation line (no env-var indirection). Catches a
    silent revert of any production tunable that the iter-58
    env-only coverage missed.
    """
    # The wrapper uses bash backslash-continuation; flags appear on
    # separate continuation lines like
    #   --acoustic-off-centering 0.1 \
    # Match flag then whitespace then the literal value.
    pattern = re.escape(flag) + r"\s+" + re.escape(expected_value) + r"\b"
    assert re.search(pattern, wrapper_text), (
        f"run_rce_30day.sh missing hardcoded {flag} {expected_value} "
        f"in the mpirun invocation. A partial revert of this "
        f"production tunable would slip past both the env-var "
        f"defaults test (iter-58) and the driver argparse test "
        f"(iter-59)."
    )


def test_wrapper_semi_implicit_acoustic_present(wrapper_text):
    """--semi-implicit-acoustic is a bare flag (no value); verify it
    appears in the mpirun invocation. Production contract since
    iter-1 + verified by iter-14/38."""
    assert re.search(r"\\\n\s*--semi-implicit-acoustic\b", wrapper_text), (
        "run_rce_30day.sh missing --semi-implicit-acoustic flag in "
        "mpirun invocation. The SI substep is the iter-14/38 "
        "production contract; removing it falls back to explicit "
        "forward-Euler which is dt-stability-bounded at the iter-1 "
        "ladder (dt <= 1.0 s)."
    )
