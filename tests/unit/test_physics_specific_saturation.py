"""Part B of the water-vapour convention (user decision 2026-09-29, option 1):
the schemes go SPECIFIC internally.  The state's water tracers are specific
humidity on total mass; every scheme now measures them against
``saturation_specific_humidity`` (and its ice / dT / blended forms), the
layer mass stays the TOTAL dp, and there is no basis conversion anywhere
(both reviewers rejected a tendency-level (1-Q) wrapper: flux-form
processes and the precipitation diagnostic do not survive it).

Gates:
  * the two new thermo helpers are the exact q(r) images of the mixing-
    ratio helpers (ice), and the analytic slope matches a central finite
    difference of ``saturation_specific_humidity`` (dT)
  * representation: a column at EXACTLY q = q_sat(T, p) is read as RH = 1
    by the cloud-cover dispatch, the warm-rain mixed-phase curve, the
    Sundqvist process rates (condensation gate fully open, zero
    supersaturation) and the Sundqvist cloud fraction (exactly 1); the OLD
    reference would read 1/(1 + r_sat) (0.977 at 300 K), so each gate
    fails with one site reverted
  * ratchet: no ``saturation_mixing_ratio*`` reference survives under
    atmosphere/physics outside the allowlisted files (thermo's own
    re-exports, the IFS-faithful path, CLUBB's native-r bridge, the
    r_s moist-lapse formula) except as the argument of the one exact map.
    CLUBB (turbulence/clubb.py) builds its own saturation from the vapour
    pressure (native rt/thl mixing ratios): not caught by this name-based
    ratchet; a separate bridge decision, registered.
"""

from __future__ import annotations

import ast
import pathlib
import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.thermo import (  # noqa: E402
    saturation_mixing_ratio,
    saturation_mixing_ratio_ice,
    saturation_specific_humidity,
    saturation_specific_humidity_dT,
    saturation_specific_humidity_ice,
)

T_GRID = jnp.asarray([230.0, 260.0, 280.0, 300.0, 310.0])
P_GRID = jnp.asarray([3.0e4, 6.0e4, 8.0e4, 1.0e5, 1.02e5])


def test_ice_helper_is_the_exact_image_of_the_mixing_ratio_helper():
    r_i = saturation_mixing_ratio_ice(T_GRID, P_GRID)
    np.testing.assert_allclose(np.asarray(saturation_specific_humidity_ice(T_GRID, P_GRID)),
                               np.asarray(r_i / (1.0 + r_i)), rtol=1e-14, atol=0)


def test_dT_helper_matches_a_central_finite_difference():
    h = 1e-3
    fd = (saturation_specific_humidity(T_GRID + h, P_GRID)
          - saturation_specific_humidity(T_GRID - h, P_GRID)) / (2 * h)
    an = saturation_specific_humidity_dT(T_GRID, P_GRID)
    np.testing.assert_allclose(np.asarray(an), np.asarray(fd), rtol=1e-6, atol=0)
    # the exact derivative of the VALUE function: it also holds where the
    # mixing-ratio slope helper's hard clip disagrees with the value's
    # softplus floor (e_sat -> p; codex/GLM 2026-09-29)
    T_x, p_x = jnp.asarray([300.0, 320.0]), jnp.asarray([100.0, 1000.0])
    fd_x = (saturation_specific_humidity(T_x + h, p_x)
            - saturation_specific_humidity(T_x - h, p_x)) / (2 * h)
    np.testing.assert_allclose(np.asarray(saturation_specific_humidity_dT(T_x, p_x)),
                               np.asarray(fd_x), rtol=1e-5, atol=0)
    # and the chain factor against the mixing-ratio slope in the ordinary regime
    r = saturation_mixing_ratio(T_GRID, P_GRID)
    from legoesm.thermo import saturation_mixing_ratio_dT
    np.testing.assert_allclose(np.asarray(an * (1.0 + r) ** 2),
                               np.asarray(saturation_mixing_ratio_dT(T_GRID, P_GRID)), rtol=1e-6)
    # differentiable: a second derivative exists and is finite
    g = jax.grad(lambda t: jnp.sum(saturation_specific_humidity_dT(t, P_GRID)))(T_GRID)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).min()) > 0.0


def _saturated_column(T=300.0, p=1.0e5, ncol=2, nlev=3):
    T = jnp.full((ncol, nlev), T)
    p = jnp.full((ncol, nlev), p)
    return T, p, saturation_specific_humidity(T, p)


def test_cloud_cover_dispatch_reads_a_saturated_column_as_rh_one():
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        CloudConfig, cover_saturation_specific_humidity, sundqvist_cloud_fraction)
    T, p, q = _saturated_column()
    for scheme in ("liquid", "mixed_phase"):
        cfg = CloudConfig(scheme="sundqvist", saturation_scheme=scheme)
        rh = q / cover_saturation_specific_humidity(T, p, cfg)
        np.testing.assert_allclose(np.asarray(rh), 1.0, rtol=1e-14, atol=0)
        assert float(sundqvist_cloud_fraction(rh, cfg).min()) == 1.0
    # the old reference is measurably NOT 1 on this column (the tripwire)
    assert float((q / saturation_mixing_ratio(T, p)).max()) < 0.98


def test_warm_rain_mixed_phase_curve_and_sundqvist_rates_are_specific():
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        mixed_phase_saturation_specific_humidity)
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from legoesm.atmosphere.physics.microphysics.sundqvist import (
        SundqvistConfig, diagnose_sundqvist_process_rates)
    from legoesm import constants
    T, p, q = _saturated_column()
    np.testing.assert_allclose(np.asarray(mixed_phase_saturation_specific_humidity(T, p)),
                               np.asarray(q), rtol=1e-14, atol=0)
    # interior of the mixed-phase ramp (codex: endpoints alone pass a
    # map-before-blend implementation): the blend is the exact image of
    # the blended MIXING RATIO, and lies strictly between the two curves
    T_c = jnp.full_like(T, 250.0)
    from legoesm.atmosphere.physics.microphysics._warm_rain import mixed_phase_liquid_fraction
    w = mixed_phase_liquid_fraction(T_c)
    r_b = w * saturation_mixing_ratio(T_c, p) + (1.0 - w) * saturation_mixing_ratio_ice(T_c, p)
    q_b = mixed_phase_saturation_specific_humidity(T_c, p)
    np.testing.assert_allclose(np.asarray(q_b), np.asarray(r_b / (1.0 + r_b)), rtol=1e-14, atol=0)
    assert 0.0 < float(w.min()) and float(w.max()) < 1.0            # interior weight
    q_l, q_i = saturation_specific_humidity(T_c, p), saturation_specific_humidity_ice(T_c, p)
    assert float((q_b - q_i).min()) > 0.0 and float((q_l - q_b).min()) > 0.0
    # map-before-blend would give w q_l + (1-w) q_i: measurably different
    assert float(jnp.abs(q_b - (w * q_l + (1.0 - w) * q_i)).max()) > 1e-9 * float(q_b.max())
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        CloudConfig, cover_saturation_specific_humidity, _ice_fraction)
    cfg = CloudConfig(scheme="sundqvist", saturation_scheme="mixed_phase")
    f = _ice_fraction(T_c, cfg)
    assert 0.0 < float(f.min()) and float(f.max()) < 1.0
    r_cov = (1.0 - f) * saturation_mixing_ratio(T_c, p) + f * saturation_mixing_ratio_ice(T_c, p)
    np.testing.assert_allclose(np.asarray(cover_saturation_specific_humidity(T_c, p, cfg)),
                               np.asarray(r_cov / (1.0 + r_cov)), rtol=1e-14, atol=0)
    z = jnp.zeros_like(q)
    hyd = HydrometeorState(q_c=z, q_r=z, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)
    rho = p / (constants.R_d * T)
    rates = diagnose_sundqvist_process_rates(
        T=T, q_v=q, hydrometeors=hyd, p_full=p, p_half=None, rho=rho,
        dz=jnp.full_like(T, 500.0), dt=300.0, config=SundqvistConfig())
    # exactly saturated: no supersaturation to remove, gate fully open
    assert float(jnp.abs(rates.condensation).max()) == 0.0
    # the basis discriminator (GLM 2026-09-29: the zero above also holds
    # for the OLD reference): a column halfway between q_sat and r_sat is
    # supersaturated in q and sub-saturated in r, so it MUST condense
    q_mid = 0.5 * (q + saturation_mixing_ratio(T, p))
    rates_mid = diagnose_sundqvist_process_rates(
        T=T, q_v=q_mid, hydrometeors=hyd, p_full=p, p_half=None, rho=rho,
        dz=jnp.full_like(T, 500.0), dt=300.0, config=SundqvistConfig())
    assert float(rates_mid.condensation.min()) > 0.0
    # a 1 % supersaturated column condenses; the OLD reading of a
    # saturated column was RH 0.977, i.e. it saw q - r_sat < 0 as well,
    # so the discriminating check is the exact zero above plus this:
    rates_ss = diagnose_sundqvist_process_rates(
        T=T, q_v=1.01 * q, hydrometeors=hyd, p_full=p, p_half=None, rho=rho,
        dz=jnp.full_like(T, 500.0), dt=300.0, config=SundqvistConfig())
    cfg = SundqvistConfig()
    gate = jax.nn.sigmoid(cfg.sigmoid_sharpness * (1.01 - cfg.rh_crit))   # the scheme's own RH gate at RH 1.01
    np.testing.assert_allclose(np.asarray(rates_ss.condensation * 300.0),
                               np.asarray(gate * 0.01 * q), rtol=1e-9)


# --- ratchet -------------------------------------------------------------
PHYS = pathlib.Path(__file__).resolve().parents[2] / "packages/atmosphere/legoesm/atmosphere/physics"
# Per-file reference COUNTS, shrink-only (codex 2026-09-29: a whole-file
# exemption would let one reverted Bechtold site through).  A count above
# the pin is a new mixing-ratio reference; a count below it means the pin
# must be lowered in the same PR.
ALLOWED_MIXING_RATIO_SATURATION = {          # AST Name + import-alias NODES, measured 2026-09-29
    "thermodynamics.py": 3,              # 2 re-export aliases + the r_s moist-lapse formula
    "convection/bechtold.py": 6,         # 3 import aliases + 3 sites that map r -> q themselves (oracle ZCOR form)
    "clouds/cloud_fraction.py": 6,       # 3 import aliases + blend (2) + Goff (1), each mapped once
    "microphysics/_warm_rain.py": 4,     # 2 import aliases + blend (2), mapped once
}
_PAT = re.compile(r"\bsaturation_mixing_ratio\w*\b")


def _reference_lines(src: str) -> list[int]:
    """One entry per referencing NODE (a Name or an import alias), so two
    imports on one line count twice and the pins are line-layout-proof."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and _PAT.fullmatch(node.id):
            out.append(node.lineno)
        elif isinstance(node, ast.alias) and _PAT.fullmatch(node.name):
            out.append(getattr(node, "lineno", 0))
    return sorted(out)


@pytest.mark.parametrize("path", sorted(p for p in PHYS.rglob("*.py")), ids=lambda p: str(p.relative_to(PHYS)))
def test_no_mixing_ratio_saturation_reference_in_physics(path):
    rel = str(path.relative_to(PHYS))
    lines = _reference_lines(path.read_text())
    allowed = ALLOWED_MIXING_RATIO_SATURATION.get(rel, 0)
    assert len(lines) <= allowed, (
        f"{rel}: {len(lines)} saturation_mixing_ratio* references at lines {lines}, pin is {allowed}; "
        "the tracer is SPECIFIC humidity (use saturation_specific_humidity*)")
    assert len(lines) == allowed, f"{rel}: {len(lines)} references, pin {allowed} is stale: lower it"


def test_allowlist_has_no_stale_entries():
    for rel in ALLOWED_MIXING_RATIO_SATURATION:
        assert (PHYS / rel).exists(), rel


def test_ratchet_detector_sees_a_reverted_site():
    assert _reference_lines("from legoesm.thermo import saturation_mixing_ratio\nq = saturation_mixing_ratio(T, p)\n") == [1, 2]
    assert _reference_lines("q = saturation_specific_humidity(T, p)\n") == []
