"""The FESOM arm runs on legoESM's constants. ALL of them, pi included.

USER DIRECTIVE 2026-08-11. A cross-dycore comparison is only controlled if
the arms share their physical constants, so the FESOM arm takes legoESM's
``g``, ``R_earth``, ``Omega`` and reference density.

``PI`` was initially left at FESOM2's truncated 3.14159265358979 -- its own
source says not to replace it -- and flagged. USER DIRECTIVE 2026-08-11
overrode that: pi now comes from ``legoesm.constants.PI`` like everything
else, so there is a single definition. The concern it was excepted for is
recorded rather than dropped: pi is baked into mesh-derived quantities at
LOAD time, so overriding it afterwards does not retroactively change a mesh
(see :func:`test_pi_override_does_not_silently_rescale_a_loaded_mesh`). The
numerical difference is 1e-15 relative.

THE COST, STATED: this breaks bit-fidelity with FESOM2. Gaps closed
(relative to legoESM): rho_0 4.9e-3, Omega 2.7e-3, R_earth 5.9e-4,
g 3.9e-4. ``FesomOceanConfig(constants="fesom")`` keeps FESOM2's values for
oracle work.

Omega deserves its own note: fesom_jax builds it from the SOLAR day
(``2*PI/86400``) where Earth's rotation is sidereal — a 0.27% error in the
Coriolis parameter, i.e. a real upstream defect, not a convention. It also
never appears outside ``fesom_jax.config``: it reaches the solution only
through the mesh's precomputed ``coriolis``/``coriolis_node`` arrays, so
the adapter rescales those too (:func:`rescale_mesh_coriolis`).
"""
from __future__ import annotations

import ast
import math
import pathlib

import pytest

from legoesm import constants as C

_ADAPTER = (pathlib.Path(__file__).resolve().parents[3] / "packages" /
            "ocean" / "legoesm" / "ocean" / "dynamics" / "ocean_model_fesom.py")


def _fesom_config():
    return pytest.importorskip(
        "fesom_jax.config",
        reason="fesom_jax not installed; the FESOM arm is optional")


def _apply():
    from legoesm.ocean.dynamics.ocean_model_fesom import use_legoesm_constants
    return use_legoesm_constants()


#: fesom_jax name -> the legoESM value it must take.
_SHARED = {
    "G": C.g,
    "R_EARTH": C.R_earth,
    "OMEGA": C.Omega,
    "DENSITY_0": C.rho_ocean,
    "PI": C.PI,
}


@pytest.mark.parametrize("name", sorted(_SHARED))
def test_override_makes_the_constant_match_legoesm(name):
    cfg = _fesom_config()
    _apply()
    assert float(getattr(cfg, name)) == pytest.approx(float(_SHARED[name]),
                                                      rel=1e-15)


def test_override_reaches_already_imported_consumers():
    """The part that is easy to get wrong.

    fesom_jax consumers do ``from .config import G``, which COPIES the
    binding at import time. Setting ``config.G`` alone reaches nothing that
    was already imported, so the override rebinds each consumer module too.
    """
    _fesom_config()
    pgf = pytest.importorskip("fesom_jax.pgf")
    eos = pytest.importorskip("fesom_jax.eos")
    _apply()
    assert float(pgf.G) == pytest.approx(float(C.g), rel=1e-15)
    assert float(eos.G) == pytest.approx(float(C.g), rel=1e-15)
    assert float(eos.DENSITY_0) == pytest.approx(float(C.rho_ocean), rel=1e-15)


def test_pi_comes_from_legoesm_core():
    """One definition of pi, per the 2026-08-11 directive."""
    cfg = _fesom_config()
    _apply()
    assert float(cfg.PI) == float(C.PI) == pytest.approx(math.pi, abs=0.0)
    # Derived from pi, so it must follow rather than keep the old value.
    assert float(cfg.CYCLIC_LENGTH_RAD) == pytest.approx(2.0 * float(C.PI),
                                                         abs=0.0)


def test_pi_override_does_not_silently_rescale_a_loaded_mesh():
    """The ordering trap, pinned.

    The mesh's Coriolis arrays are 2*PI_fesom*sin(lat), built at LOAD time.
    If the rescale read ``config.PI`` after the override it would divide by
    the NEW pi and get the wrong ratio -- a small, plausible, silently
    wrong number. It must use FESOM's original pi.
    """
    cfg = _fesom_config()
    mesh_mod = pytest.importorskip("fesom_jax.mesh")
    import numpy as np
    from legoesm.ocean.dynamics.ocean_model_fesom import rescale_mesh_coriolis
    _apply()                       # pi is now legoESM's
    mesh = mesh_mod.load_mesh(mesh_dir=mesh_mod.DEFAULT_PI_MESH_DIR)
    before = np.asarray(mesh.coriolis).copy()
    after = np.asarray(rescale_mesh_coriolis(mesh).coriolis)
    live = np.abs(before) > 1e-12
    ratio = float(np.median(after[live] / before[live]))
    native_pi = 3.14159265358979          # const-ok: FESOM's own truncated pi
    expected = float(C.Omega) / (2.0 * native_pi / 86400.0)
    assert ratio == pytest.approx(expected, rel=1e-12), (
        "the rescale used the overridden pi instead of the one the mesh was "
        "built with")
    # HONEST LIMIT of this test: for pi specifically the two candidate
    # denominators differ by 3e-15 relative, so the right and wrong answers
    # here agree to 8.9e-16 -- below double precision. MEASURED, not
    # assumed. So this pins the DISCIPLINE (use the value the mesh was
    # built with) rather than a detectable numerical difference; the
    # detectable case is Omega, at 2.7e-3, covered by
    # test_omega_is_rescaled_in_the_mesh_not_just_the_module.
    wrong = float(C.Omega) / (2.0 * float(cfg.PI) / 86400.0)
    assert abs(ratio - wrong) < 1e-14, (
        "if these ever diverge measurably, this test must be strengthened "
        "into a real discriminator instead of a discipline check")


def test_the_upstream_defaults_really_did_differ():
    """Non-vacuity, read from SOURCE so an earlier test's override cannot
    make this pass trivially. If upstream ever adopts legoESM's values the
    override becomes a no-op and this test says so."""
    src = pathlib.Path(_fesom_config().__file__).read_text()
    for token in ("G = 9.81", "R_EARTH = 6367500.0", "DENSITY_0 = 1030.0",
                  "OMEGA = 2.0 * PI / 86400.0", "PI = 3.14159265358979"):
        assert token in src, (
            f"fesom_jax's default {token!r} is gone -- re-derive the gaps "
            f"quoted in this module's docstring and in "
            f"FesomOceanConfig.constants")


def test_omega_is_rescaled_in_the_mesh_not_just_the_module():
    """Omega has NO consumer outside config: it is baked into the mesh's
    coriolis arrays at load. Rebinding the name alone would change nothing,
    so this checks the array actually moved, by the exact ratio."""
    cfg = _fesom_config()
    mesh_mod = pytest.importorskip("fesom_jax.mesh")
    import numpy as np
    from legoesm.ocean.dynamics.ocean_model_fesom import rescale_mesh_coriolis
    mesh = mesh_mod.load_mesh(mesh_dir=mesh_mod.DEFAULT_PI_MESH_DIR)
    before = np.asarray(mesh.coriolis).copy()
    after = np.asarray(rescale_mesh_coriolis(mesh).coriolis)
    live = before[np.abs(before) > 1e-12]
    assert live.size, "mesh has no non-zero Coriolis to test"
    ratio = np.median(after[np.abs(before) > 1e-12] / live)
    expected = float(C.Omega) / (2.0 * float(cfg.PI) / 86400.0)
    assert ratio == pytest.approx(expected, rel=1e-12)
    assert abs(ratio - 1.0) > 1e-4, (
        "control: the rescale must actually change f, or it proves nothing")


def test_unknown_constants_mode_raises():
    """Dispatch hardening: a typo must not silently pick one set."""
    pytest.importorskip("fesom_jax")
    from legoesm.ocean.dynamics.ocean_model_fesom import (
        FesomOceanConfig, FesomOceanModel, build_flat_bottom_mesh)
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from fesom_jax.mesh import load_mesh, DEFAULT_PI_MESH_DIR
    lc = LockExchangeConfig()
    mesh = build_flat_bottom_mesh(
        load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR), H_max=lc.H_max,
        nlev=lc.nlev, land_lat_threshold=lc.land_lat_threshold)
    z = create_ocean_z_star(n_levels=lc.nlev, H_max=lc.H_max)
    with pytest.raises(ValueError, match="constants="):
        FesomOceanModel(mesh, z, FesomOceanConfig(dt=300.0,
                                                  constants="legoESM"))


def test_the_arm_says_out_loud_that_it_is_no_longer_fesom2():
    """Silently changing another model's constants is the failure mode this
    whole file exists to prevent."""
    pytest.importorskip("fesom_jax")
    import warnings
    from legoesm.ocean.dynamics.ocean_model_fesom import (
        FesomOceanConfig, FesomOceanModel, build_flat_bottom_mesh)
    from legoesm.ocean.experiments.lock_exchange import LockExchangeConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from fesom_jax.mesh import load_mesh, DEFAULT_PI_MESH_DIR
    lc = LockExchangeConfig()
    mesh = build_flat_bottom_mesh(
        load_mesh(mesh_dir=DEFAULT_PI_MESH_DIR), H_max=lc.H_max,
        nlev=lc.nlev, land_lat_threshold=lc.land_lat_threshold)
    z = create_ocean_z_star(n_levels=lc.nlev, H_max=lc.H_max)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FesomOceanModel(mesh, z, FesomOceanConfig(dt=300.0))
    msgs = " ".join(str(c.message) for c in caught)
    assert "NOT FESOM2" in msgs and "pi" in msgs, (
        f"the arm must announce the constant swap and the pi exception; "
        f"got: {msgs!r}")


def test_legoesm_adapter_hardcodes_no_physical_constants():
    """The ADAPTER is ours, so it has no excuse for a second set."""
    cfg = _fesom_config()
    banned = {float(C.g), float(C.R_earth), float(C.Omega),
              float(C.rho_ocean), float(C.T_freeze),
              float(cfg.G), float(cfg.R_EARTH), float(cfg.OMEGA),
              float(cfg.DENSITY_0), float(cfg.VCPW)}
    tree = ast.parse(_ADAPTER.read_text())
    hits = [f"line {n.lineno}: {n.value}" for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, float) and n.value in banned]
    assert not hits, (
        "the FESOM adapter hardcodes a physical constant; import it from "
        "legoesm.constants instead:\n  " + "\n  ".join(hits))
