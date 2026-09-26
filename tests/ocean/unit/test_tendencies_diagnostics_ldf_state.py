"""``tendencies_with_diagnostics`` must forward ``ldf_state``.

NEMO evaluates lateral friction at the BEFORE level
(``CALL dyn_ldf( kstp, Nbb, Nnn, uu, vv, Nrhs )``, ``stpmlf.F90:319``) because
a leap-frog-centred diffusion is unconditionally unstable. legoESM's
production leap-frog path matches that by handing the tendency pass an
``ldf_state`` of before-level fields.

``LatLonCGridOceanModel.tendencies`` has always accepted ``ldf_state``; the
``tendencies_with_diagnostics`` wrapper silently dropped it, so every oracle
budget built on that wrapper evaluated lateral friction at the NOW level while
comparing against NEMO's BEFORE-level trend — measuring
``A_h * lap(u_now - u_before)``, a quantity the instrument manufactured. Found
2026-08-25 by adversarial review of the DINO zonal-wall momentum budget;
fixing it dropped that term's measured difference by 262x.
"""
from __future__ import annotations

import inspect

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)


def test_signature_accepts_ldf_state():
    sig = inspect.signature(LatLonCGridOceanModel.tendencies_with_diagnostics)
    assert "ldf_state" in sig.parameters, (
        "the diagnostics wrapper must expose ldf_state, or every oracle "
        "budget built on it compares lateral friction across time levels")
    assert sig.parameters["ldf_state"].default is None


def test_it_is_forwarded_not_merely_accepted():
    """A parameter that is accepted and dropped is worse than none at all.

    Source-level check, naming the symbol that RUNS: the wrapper's own body
    must pass ldf_state on to the tendency function.
    """
    src = inspect.getsource(
        LatLonCGridOceanModel.tendencies_with_diagnostics)
    assert "ldf_state=ldf_state" in src, src


def test_the_sibling_still_accepts_it_so_the_forward_lands():
    sig = inspect.signature(LatLonCGridOceanModel.tendencies)
    assert "ldf_state" in sig.parameters
