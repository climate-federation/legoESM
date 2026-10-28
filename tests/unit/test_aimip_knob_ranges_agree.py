"""AIMIP knob ranges agree with the scheme spec and the driver (one parameter,
one range)."""
from __future__ import annotations

from pathlib import Path

from legoesm.training.aimip_params import AIMIP_CLASSICAL_CONSTRAINTS

_C = {c.name: c for c in AIMIP_CLASSICAL_CONSTRAINTS}
_ROOT = Path(__file__).resolve().parents[2]


def test_aimip_cloud_q_c_range_is_the_spec_and_driver_range():
    """One parameter, one range (user 2026-10-28, Q35): the AIMIP knob's bounds
    equal the cloud scheme's __param_spec__ bounds and the driver's check."""
    import re
    c = _C["cloud_q_c_diagnostic"]
    src = (_ROOT / "packages/atmosphere/legoesm/atmosphere/physics/clouds/config.py").read_text()
    lo, hi = map(float, re.search(
        r'"q_c_diagnostic": \{"units": "kg/kg", "bounds": \(([^,]+), ([^)]+)\)', src).groups())
    drv = (_ROOT / "packages/coupler/legoesm/driver/config.py").read_text()
    dlo, dhi = map(float, re.search(
        r'\("cloud_q_c_diagnostic", ([0-9eE.+-]+), ([0-9eE.+-]+)\)', drv).groups())
    assert (c.min_val, c.max_val) == (lo, hi) == (dlo, dhi)
