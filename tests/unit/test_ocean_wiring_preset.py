"""Tests for the ``ocean.wiring`` named stepping-composition selector.

Issue #388 Ask#3: ``ocean: {wiring: veros_ab2}`` in an ocean template expands to
the documented Veros-faithful stepping composition (operator order / integrator
/ forcing placement) and raises on a preset/explicit-flag conflict — promoting
the composition from a loose set of flags to a named, tested code path a config
selects. The expansion reuses the canonical
``veros_stepping_common.VEROS_FAITHFUL_STEPPING_SIGNATURE`` (single source of
truth, ratchet-tested by the Veros free-run recipes), so it can never drift.
"""
from __future__ import annotations

import pytest

from legoesm.ocean.config import OceanExperimentConfig
from legoesm.ocean.fidelity.veros_stepping_common import (
    VEROS_FAITHFUL_STEPPING_SIGNATURE,
)


def _cfg(ocean: dict, grid_type: str = "latlon_cgrid") -> OceanExperimentConfig:
    return OceanExperimentConfig.from_dict(
        {"model": {"type": "ocean_only"},
         "grid": {"type": grid_type},
         "ocean": ocean})


class TestOceanWiringPreset:
    def test_veros_ab2_expands_to_canonical_signature(self):
        """Every field of the preset equals the canonical fragment — the
        expansion is LOCKED to the single source of truth (drift => red)."""
        cfg = _cfg({"wiring": "veros_ab2"}).to_ocean_config()
        for field, value in VEROS_FAITHFUL_STEPPING_SIGNATURE.items():
            assert getattr(cfg, field) == value, field

    def test_veros_ab2_equivalent_to_explicit_flags(self):
        """A wiring preset is exactly equivalent to setting the same flags by
        hand — the preset adds no hidden behaviour."""
        via_preset = _cfg({"wiring": "veros_ab2"}).to_ocean_config()
        via_flags = _cfg(dict(VEROS_FAITHFUL_STEPPING_SIGNATURE)).to_ocean_config()
        assert via_preset == via_flags

    def test_wiring_composes_with_other_dials(self):
        """A preset sets only its composition fields; unrelated scalar dials in
        the same ``ocean:`` block still apply."""
        cfg = _cfg({"wiring": "veros_ab2", "A_h": 12345.0}).to_ocean_config()
        assert cfg.A_h == 12345.0
        assert cfg.outer_integrator == "ab2"

    def test_conflict_with_explicit_flag_raises(self):
        with pytest.raises(ValueError, match=r"sets \['outer_integrator'\]"):
            _cfg({"wiring": "veros_ab2",
                  "outer_integrator": "forward_euler"}).to_ocean_config()

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown ocean.wiring preset"):
            _cfg({"wiring": "veros_ab9000"}).to_ocean_config()

    @pytest.mark.parametrize("bad", [["veros_ab2"], {"name": "veros_ab2"}, 7])
    def test_non_string_wiring_raises_clean_valueerror(self, bad):
        # YAML can produce a list/map; must be a clean ValueError, not a raw
        # TypeError from dict membership.
        with pytest.raises(ValueError, match="ocean.wiring must be a string"):
            _cfg({"wiring": bad}).to_ocean_config()

    def test_default_and_absent_are_no_op(self):
        bare = _cfg({}).to_ocean_config()
        for sentinel in ("default", ""):
            assert _cfg({"wiring": sentinel}).to_ocean_config() == bare
        # absent wiring is identical to a bare config
        assert _cfg({}).to_ocean_config() == bare

    def test_non_latlon_grid_raises(self):
        with pytest.raises(ValueError, match="does not apply to grid.type"):
            _cfg({"wiring": "veros_ab2"}, grid_type="cubed_sphere").to_ocean_config()

    def test_validate_strict_passes(self):
        _cfg({"wiring": "veros_ab2"}).validate_strict()

    def test_signature_tracks_wiring(self):
        a = _cfg({}).signature()
        b = _cfg({"wiring": "veros_ab2"}).signature()
        assert a != b, "ocean.wiring must change the resolved-config signature"

    def test_wiring_not_a_runtime_field(self):
        """`wiring` is a selector, not a LatLonCGridOceanConfig field — it must
        be consumed (popped), never leak as an 'unknown field' error or onto
        the runtime config."""
        cfg = _cfg({"wiring": "veros_ab2"}).to_ocean_config()
        assert not hasattr(cfg, "wiring")
