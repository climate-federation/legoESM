"""Tests for #518 item 11: unified EPS_DIV + named WindStressConvention.

* All four GM/Redi + MLE variants now read the SAME ``_gm_redi_common.EPS_DIV``
  (1e-10).  gm_redi_mpas previously used 1e-30 — the one intentional value
  change (a bugfix; the others are unchanged).
* ``WindStressConvention`` / ``wind_stress_sign`` name the prescribed(+tau) vs
  external(-tau) momentum convention WITHOUT changing the signs.
"""

from __future__ import annotations

import pytest

from legoesm.ocean.physics.lateral_mixing._gm_redi_common import EPS_DIV
from legoesm.ocean.physics.lateral_mixing import (
    gm_redi_latlon_cgrid,
    gm_redi_mpas,
    mle_latlon_cgrid,
    mle_mpas,
)
from legoesm.ocean.physics.surface_forcing._shared import (
    WindStressConvention,
    wind_stress_sign,
)


def test_eps_div_unified_to_1e_10():
    assert EPS_DIV == 1e-10


def test_all_four_variants_share_eps_div():
    # Each module aliases the shared constant as its local _EPS_DIV.
    assert gm_redi_latlon_cgrid._EPS_DIV is EPS_DIV
    assert gm_redi_mpas._EPS_DIV is EPS_DIV
    assert mle_latlon_cgrid._EPS_DIV is EPS_DIV
    assert mle_mpas._EPS_DIV is EPS_DIV


def test_gm_redi_mpas_no_longer_1e_30():
    """The one intentional change: gm_redi_mpas was 1e-30, now 1e-10."""
    assert gm_redi_mpas._EPS_DIV == 1e-10
    assert gm_redi_mpas._EPS_DIV != 1e-30


def test_wind_stress_sign_values():
    assert wind_stress_sign(WindStressConvention.OCEAN_DIRECT) == 1.0
    assert wind_stress_sign(WindStressConvention.ATMOSPHERE_REACTION) == -1.0


def test_wind_stress_sign_opposite():
    s_p = wind_stress_sign(WindStressConvention.OCEAN_DIRECT)
    s_e = wind_stress_sign(WindStressConvention.ATMOSPHERE_REACTION)
    assert s_p == -s_e


def test_wind_stress_sign_rejects_unknown():
    with pytest.raises(ValueError):
        wind_stress_sign("ocean_direct")  # not the enum member -> hard error
