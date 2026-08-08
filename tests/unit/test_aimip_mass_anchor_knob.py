"""``fix_mass`` / ``anchor_mass_to_initial`` must reach the dycore config.

The spectral primitive-equation dycore has had a dry-mass anchor since iter-3,
but ``run_aimip`` built ``SpectralPEConfig`` without ever passing it, so the
anchor was False on every AIMIP arm regardless of what a suite asked for. The
two AIMIP arms that USE that dycore (classical, column_nn) drift by ~-16 hPa of
area-weighted mean sea-level pressure over a 10-day forecast, while the arm
that does not use it (sfno_full, whose SFNO physics projects the global mean
out of dlnps/dt) drifts by -0.8 hPa; 65% of classical's day-10 z500 MSE is that
bias term. A silently-dropped knob here reads exactly like "the anchor does not
help" — the failure mode this file exists to prevent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_aimip():
    """Import scripts/run/run_aimip.py by path (not an importable module)."""
    spec = importlib.util.spec_from_file_location(
        "run_aimip_mass_probe", _ROOT / "scripts" / "run" / "run_aimip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_aimip_mass_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _pe_config(mod, overrides, variant="classical"):
    base = mod._load_yaml(_ROOT / "config" / "aimip" / "aimip_era5.yaml")
    merged = mod._merge(base, overrides)
    merged["aimip_variant"] = variant
    return mod._build_spectral_config(merged).pe_config


def test_mass_anchor_defaults_off(run_aimip):
    """Every pre-existing suite omits both keys and must stay byte-identical.

    Asserted on the REAL builder rather than SpectralPEConfig's own defaults:
    the dataclass default was already False, and the bug was that run_aimip
    never forwarded the key at all.
    """
    pe = _pe_config(run_aimip, {})
    assert pe.fix_mass is False
    assert pe.anchor_mass_to_initial is False


def test_mass_anchor_is_reachable_from_a_suite(run_aimip):
    pe = _pe_config(run_aimip, {"fix_mass": True,
                                "anchor_mass_to_initial": True})
    assert pe.fix_mass is True
    assert pe.anchor_mass_to_initial is True


def test_anchor_requires_fix_mass_to_do_anything(run_aimip):
    """Both flags are independent knobs, and the dycore ANDs them.

    ``anchor_mass_to_initial`` alone must still forward as True — the guard
    that makes it inert lives in ``spectral_pe.step``, not in the builder, and
    a builder that silently corrected the combination would hide a misconfigured
    suite instead of letting the dycore's own condition speak.
    """
    pe = _pe_config(run_aimip, {"anchor_mass_to_initial": True})
    assert pe.fix_mass is False
    assert pe.anchor_mass_to_initial is True

    import inspect

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel,
    )

    # The function that RUNS, not a delegating wrapper: the anchor is applied
    # in ``_do_step`` (the compiled step body), gated on both flags together.
    src = inspect.getsource(SpectralPrimitiveEquationModel._do_step)
    assert "self.config.fix_mass" in src
    assert "self.config.anchor_mass_to_initial" in src
    assert "_apply_mass_fixer" in src


def test_the_ab_suite_pair_differs_only_in_the_anchor(run_aimip):
    """suite_curriculum_v2_massfix is the A/B partner of suite_curriculum_v2.

    An A/B whose two configs differ in a second field is a confound, and the
    two suites are separate files that can drift apart independently.
    """
    ace2 = _ROOT / "config" / "aimip" / "ace2"
    base_suite = run_aimip._load_yaml(ace2 / "suite_curriculum_v2.yaml")
    ab_suite = run_aimip._load_yaml(ace2 / "suite_curriculum_v2_massfix.yaml")

    assert base_suite["base"] == ab_suite["base"]
    assert base_suite["variants"] == ab_suite["variants"]

    base_ov = dict(base_suite["cfg_overrides"])
    ab_ov = dict(ab_suite["cfg_overrides"])
    changed = {k for k in set(base_ov) | set(ab_ov)
               if base_ov.get(k) != ab_ov.get(k)}
    assert changed == {"fix_mass", "anchor_mass_to_initial"}, changed

    pe_off = _pe_config(run_aimip, base_ov, variant="column_nn")
    pe_on = _pe_config(run_aimip, ab_ov, variant="column_nn")
    assert (pe_off.fix_mass, pe_off.anchor_mass_to_initial) == (False, False)
    assert (pe_on.fix_mass, pe_on.anchor_mass_to_initial) == (True, True)


def test_aimip_grid_refuses_a_grid_run_aimip_cannot_build(run_aimip):
    """``aimip_grid`` was declared by every config and read by nothing.

    The grid is hardcoded ``create_gaussian_grid`` at two sites, so a suite
    asking for "mpas" or "latlon" silently got a Gaussian spectral grid — the
    same class of trap as a config flag that reaches a code path the run never
    executes. It now refuses rather than advertising a choice that does not
    exist.
    """
    import pytest

    base = run_aimip._load_yaml(_ROOT / "config" / "aimip" / "aimip_era5.yaml")

    # The declared default still works.
    ok = run_aimip._merge(base, {"aimip_grid": "gaussian"})
    ok["aimip_variant"] = "column_nn"
    assert run_aimip._build_spectral_config(ok) is not None

    for bad in ("mpas", "latlon", "cubed_sphere"):
        cfg = run_aimip._merge(base, {"aimip_grid": bad})
        cfg["aimip_variant"] = "column_nn"
        with pytest.raises(SystemExit, match="aimip_grid"):
            run_aimip._build_spectral_config(cfg)

    # Absent key keeps the historical behaviour (gaussian), so no existing
    # suite changes.
    cfg = dict(base)
    cfg.pop("aimip_grid", None)
    cfg["aimip_variant"] = "column_nn"
    assert run_aimip._build_spectral_config(cfg) is not None
