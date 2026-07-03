"""Per-PFT Vcmax25 lookup table + no-PFT defaults (DifferBESS sync).

The PFT_VCMAX25 tables previously held divergent values (e.g. GRA-cold 142,
DBF-cold 96, DNF 57) and were never consulted — the only fallback was a flat
60 / 40.  This checks the corrected DifferBESS CLM4.5/Jiang-Ryu values, the
lookup helper, and the new no-PFT defaults (DBF-temperate for C3, mean C4
grass/crop for C4) now used by the two-leaf canopy fallback.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.land.canopy.config import (
    PFT_VCMAX25_C3,
    PFT_VCMAX25_C4,
    VCMAX25_C3_DEFAULT,
    VCMAX25_C4_DEFAULT,
    lookup_vcmax25,
)


def test_corrected_c3_values_match_differbess():
    # The old transcription errors are gone.
    assert PFT_VCMAX25_C3["DBF"] == [41.0, 57.7, 57.7]   # was [66, 62, 96]
    assert PFT_VCMAX25_C3["GRA"] == [78.2, 78.2, 78.2]   # was [78, 78, 142]
    assert PFT_VCMAX25_C3["DNF"] == [39.1, 39.1, 39.1]   # was [57, 57, 57]
    assert PFT_VCMAX25_C3["EBF"][0] == 55.0              # tropical, was 41


def test_corrected_c4_values():
    assert PFT_VCMAX25_C4["GRA"] == [51.6, 51.6, 51.6]   # CLM4.5 C4 grass
    assert PFT_VCMAX25_C4["CRO"] == [37.0, 37.0, 37.0]   # JR C4 crop


def test_no_pft_defaults():
    # C3 default == DBF-temperate.
    assert VCMAX25_C3_DEFAULT == 57.7
    # C4 default == mean(C4 grass, C4 crop) temperate.
    assert VCMAX25_C4_DEFAULT == pytest.approx(0.5 * (51.6 + 37.0))


def test_lookup_helper():
    assert lookup_vcmax25("DBF", "temperate") == 57.7
    assert lookup_vcmax25("GRA", "boreal") == 78.2
    assert lookup_vcmax25("GRA", "temperate", c4=True) == 51.6
    # Unknown PFT -> no-PFT default.
    assert lookup_vcmax25("UNKNOWN", "temperate") == VCMAX25_C3_DEFAULT
    assert lookup_vcmax25("UNKNOWN", "temperate", c4=True) == VCMAX25_C4_DEFAULT
    # Unknown climate -> raises (no silent default).
    with pytest.raises(ValueError):
        lookup_vcmax25("DBF", "subtropical")


def test_two_leaf_fallback_uses_defaults_not_flat_60_40():
    """When canopy_params lacks Vcmax25, the canopy uses the DBF/mean defaults."""
    from legoesm.land.surface_scheme.two_leaf_canopy import _get

    ncol = 3
    Vc3 = _get(None, "Vcmax25_C3_leaf", jnp.full(ncol, VCMAX25_C3_DEFAULT))
    Vc4 = _get(None, "Vcmax25_C4_leaf", jnp.full(ncol, VCMAX25_C4_DEFAULT))
    assert jnp.allclose(Vc3, 57.7)
    assert jnp.allclose(Vc4, 44.3)
    # The old flat fallback values are no longer used.
    assert not jnp.allclose(Vc3, 60.0)
    assert not jnp.allclose(Vc4, 40.0)
