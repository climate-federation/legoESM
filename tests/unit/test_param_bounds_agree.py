"""One parameter, one range. Two disagreeing ranges mean neither is the range.

A tunable declares bounds twice: in its scheme's ``__param_spec__``, which the
registry-qualified ``--params`` path range-checks, and in the driver's
``validate_strict``, which the YAML and CLI path range-checks. When they
disagree, a value can be legal through one door and illegal through the other —
and the door a campaign actually uses is whichever the launcher happens to
take.

FOUND BY: a proposal to run ``cloud_q_c_diagnostic`` at 5e-6, described as
"an order of magnitude below its own declared lower bound". It is below the
SPEC's lower bound of 5e-5 and comfortably inside the DRIVER's, which starts at
1e-6. So the bound quoted in the discussion was not the bound that would have
guarded the run.

This is the hidden-choice class in CLAUDE.md: a scientific limit that exists in
two places and is enforced in one.

SHRINK-ONLY. The list below is debt, not permission. Adding an entry means
shipping a parameter whose legal range depends on which door you came through.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _ROOT / "packages" / "coupler" / "legoesm" / "driver" / "config.py"
_PACKAGES = _ROOT / "packages"

# name -> (driver range, spec range) as they stand today. Each needs the two
# declarations reconciled; until then the value that is legal depends on the
# entry point.  SHRINK-ONLY.
_DISAGREE: dict[str, str] = {
    "bechtold_delta_deep":
        "driver (2.475e-05, 2.25e-04) vs spec (3.3e-05, 3.0e-04)",
    "bechtold_downdraft_alpha":
        "driver (0, 0.9) vs spec (0.00495, 0.045) — the driver admits twenty "
        "times the spec's upper bound",
    "bechtold_epsilon_deep":
        "driver (5.775e-04, 3.5e-03) vs spec (7.0e-04, 4.2e-03)",
}

_TOL = 1e-12


def _driver_bounds() -> dict[str, tuple[float, float]]:
    """``("name", lo, hi)`` triples from the driver's range checks."""
    out = {}
    for m in re.finditer(
            r'\(\s*"([a-z_0-9]+)"\s*,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*\)',
            _DRIVER.read_text()):
        try:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        except ValueError:                       # not a numeric triple
            continue
    return out


def _spec_bounds() -> dict[str, list[tuple[tuple[float, float], str]]]:
    """``name -> [((lo, hi), file), ...]`` from EVERY ``__param_spec__``, in
    sorted file order.  All declarations are kept: a name two specs bound
    differently is a disagreement whichever file is read last."""
    out: dict[str, list] = {}
    for path in sorted(_PACKAGES.rglob("*.py")):
        text = path.read_text(errors="replace")
        if "__param_spec__" not in text:
            continue
        for m in re.finditer(
                r'"([a-z_0-9]+)"\s*:\s*\{[^}]*"bounds"\s*:\s*'
                r'\(\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*\)', text):
            try:
                b = (float(m.group(2)), float(m.group(3)))
            except ValueError:
                continue
            out.setdefault(m.group(1), []).append(
                (b, str(path.relative_to(_ROOT))))
    return out


def _disagreements(driver=None, spec=None) -> dict[str, str]:
    driver = _driver_bounds() if driver is None else driver
    spec = _spec_bounds() if spec is None else spec
    bad = {}
    for name, (lo, hi) in driver.items():
        for sname, decls in spec.items():
            # the driver prefixes the scheme onto the spec's field name
            if not (name == sname or name.endswith("_" + sname)
                    or name == "cloud_" + sname):
                continue
            for (slo, shi), where in decls:
                if (abs(lo - slo) > _TOL * max(1.0, abs(slo))
                        or abs(hi - shi) > _TOL * max(1.0, abs(shi))):
                    bad[name] = (f"driver ({lo:g}, {hi:g}) vs spec "
                                 f"({slo:g}, {shi:g}) in {where}")
    return bad


def test_a_second_spec_declaring_another_range_is_caught():
    """Non-vacuity: one agreeing and one diverging declaration of the same
    name is a disagreement, in either order."""
    driver = {"bechtold_dnoprc": (7.5e-5, 1.2e-3)}
    good, bad = ((7.5e-5, 1.2e-3), "a.py"), ((1.0e-4, 1.0e-3), "b.py")
    for decls in ([good, bad], [bad, good]):
        assert "bechtold_dnoprc" in _disagreements(driver, {"dnoprc": decls})
    assert not _disagreements(driver, {"dnoprc": [good, good]})


def test_the_comparison_finds_both_declarations():
    """Non-vacuity: if either side parses to nothing, everything 'agrees'."""
    driver, spec = _driver_bounds(), _spec_bounds()
    assert len(driver) >= 20, f"only {len(driver)} driver ranges parsed"
    assert len(spec) >= 100, f"only {len(spec)} spec ranges parsed"


def test_no_new_parameter_declares_two_different_ranges():
    new = sorted(set(_disagreements()) - set(_DISAGREE))
    assert not new, (
        f"{new} declare different legal ranges in the driver and in their "
        f"scheme's parameter spec, so whether a value is accepted depends on "
        f"which entry point set it. Reconcile the two, or add an entry to "
        f"_DISAGREE naming the reconciliation owed.")


def test_the_disagreement_list_only_shrinks():
    fixed = sorted(set(_DISAGREE) - set(_disagreements()))
    assert not fixed, (
        f"{fixed} now agree — delete their _DISAGREE entries. A stale entry "
        f"is a waiver covering nothing.")


@pytest.mark.parametrize("name", sorted(_DISAGREE))
def test_each_known_disagreement_is_still_real(name):
    """An entry that stops describing the code is a lie in a list."""
    assert name in _disagreements(), (
        f"{name} is listed as disagreeing but does not; remove it")


def test_ifs_ascent_port_shares_the_bechtold_ranges():
    """IFSAscentConfig ports the same rprcon / dnoprc as BechtoldConfig: one
    parameter, one range."""
    from legoesm.atmosphere.physics.convection import _ifs_ascent, config
    ifs = _ifs_ascent.__param_spec__["IFSAscentConfig"]["params"]
    bec = config.__param_spec__["BechtoldConfig"]["params"]
    for name in ("rprcon", "dnoprc"):
        assert ifs[name]["bounds"] == bec[name]["bounds"], name


# --- AIMIP trainable knobs: a third declaration of the same ranges -------------
# (user 2026-10-28, ASK N): every AIMIP knob's range is compared with its
# scheme's __param_spec__ and, where one exists, the driver's check.  Values
# are NOT reconciled here; today's disagreements are listed, SHRINK-ONLY.
_AIMIP_DISAGREE: dict[str, str] = {
    "tiedtke_tau_M_u_relax":
        "aimip (600.0, 14400.0) vs spec (600.0, 5400.0) vs driver None",
    "tiedtke_tau_MC_proxy":
        "aimip (1800.0, 28800.0) vs spec (1188.0, 10800.0) vs driver None",
    "tiedtke_downdraft_alpha":
        "aimip (0.05, 0.6) vs spec (0.0, 0.9) vs driver None",
    "tiedtke_delta_deep":
        "aimip (2e-05, 0.0005) vs spec (3.3e-05, 0.0003) vs driver None",
    "tiedtke_delta_shallow":
        "aimip (0.0001, 0.001) vs spec (9.9e-05, 0.0009) vs driver None",
    "tiedtke_delta_midlevel":
        "aimip (2e-05, 0.0005) vs spec (6.6e-05, 0.0006) vs driver None",
    "tiedtke_cmt_c_u":
        "aimip (0.3, 1.0) vs spec (0.0, 2.0) vs driver None",
    "tiedtke_cmt_c_d":
        "aimip (0.3, 1.0) vs spec (0.0, 2.0) vs driver None",
    "tiedtke_parcel_dq":
        "aimip (0.0001, 0.005) vs spec (0.0, 0.003) vs driver None",
    "tiedtke_cloud_depth_shallow_max":
        "aimip (500.0, 3000.0) vs spec (800.0, 2500.0) vs driver None",
    "tiedtke_mc_proxy_RH_crit":
        "aimip (0.4, 0.9) vs spec (0.3, 1.0) vs driver None",
    "louis_l_mix_max":
        "aimip (20.0, 400.0) vs spec (10.0, 300.0) vs driver None",
    "louis_Ri_crit":
        "aimip (0.1, 0.6) vs spec (0.1, 0.75) vs driver None",
    "louis_b_louis":
        "aimip (2.0, 10.0) vs spec (1.5, 15.0) vs driver None",
    "louis_c_louis":
        "aimip (5.0, 30.0) vs spec (5.0, 49.8) vs driver None",
    "louis_d_louis":
        "aimip (2.0, 15.0) vs spec (1.5, 15.0) vs driver None",
    "surface_Cd_neutral":
        "aimip (0.0005, 0.003) vs spec (0.0005, 0.005) vs driver None",
    "surface_Ch_neutral":
        "aimip (0.0005, 0.003) vs spec (0.0005, 0.005) vs driver None",
    "mcfarlane_h_topo":
        "aimip (100.0, 2000.0) vs spec (50.0, 2000.0) vs driver None",
    "mcfarlane_G_0":
        "aimip (0.1, 1.0) vs spec (0.1, 2.0) vs driver None",
    "mcfarlane_efficiency":
        "aimip (0.1, 1.0) vs spec (0.0, 1.0) vs driver None",
    "mcfarlane_min_wind":
        "aimip (0.5, 5.0) vs spec (0.5, 6.0) vs driver None",
    "mcfarlane_envelope_scale":
        "aimip (0.5, 2.0) vs spec (0.25, 4.0) vs driver None",
    "mcfarlane_directional_spread":
        "aimip (0.5, 2.0) vs spec (0.25, 4.0) vs driver None",
    "cloud_rh_crit":
        "aimip (0.5, 0.95) vs spec (0.5, 0.99) vs driver (0.5, 0.99)",
    "cloud_alpha_xr":
        "aimip (25.0, 400.0) vs spec (10.0, 1000.0) vs driver (10.0, 1000.0)",
    "cloud_p_xr":
        "aimip (0.1, 1.0) vs spec (0.05, 1.0) vs driver (0.05, 1.0)",
    "cloud_r_eff_liq":
        "aimip (5e-06, 3e-05) vs spec (4e-06, 3e-05) vs driver None",
    "cloud_r_eff_ice":
        "aimip (1e-05, 0.0001) vs spec (1e-05, 9e-05) vs driver None",
    "sundqvist_RH_crit":
        "aimip (0.5, 0.95) vs spec (0.5, 1.0) vs driver None",
    "sundqvist_evap_coeff":
        "aimip (1e-05, 0.005) vs spec (1e-05, 0.0015) vs driver None",
    "sbm_RH_ref":
        "aimip (0.5, 0.9) vs spec (0.4, 1.0) vs driver None",
}


def _aimip_knob_fields() -> dict[str, tuple[str, str]]:
    """knob -> (config class name, field), read from the code that maps them:
    ``aimip_params._canonical_scheme_defaults`` (``var = Class()`` then
    ``"knob": float(var.field)``)."""
    import inspect

    from legoesm.training import aimip_params
    tree = ast.parse(inspect.getsource(aimip_params._canonical_scheme_defaults))
    var_cls = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)):
            var_cls[node.targets[0].id] = node.value.func.id
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                attr = v.args[0]
                out[k.value] = (var_cls[attr.value.id], attr.attr)
    return out


def _aimip_rows():
    """``[(knob, aimip (lo, hi), spec (lo, hi) | None, driver (lo, hi) | None)]``."""
    from legoesm.training.aimip_params import AIMIP_CLASSICAL_CONSTRAINTS
    from legoesm.training.param_collector import build_registry
    spec = {}
    for m in build_registry():
        spec.setdefault((m.config_class, m.field), []).append(m)
    fields, driver = _aimip_knob_fields(), _driver_bounds()
    rows = []
    for c in AIMIP_CLASSICAL_CONSTRAINTS:
        metas = spec.get(fields[c.name], [])
        assert len(metas) <= 1, f"{c.name}: {len(metas)} spec entries"
        sb = tuple(map(float, metas[0].bounds)) if metas else None
        rows.append((c.name, (c.min_val, c.max_val), sb, driver.get(c.name)))
    return rows


def _close(a, b):
    return all(abs(x - y) <= _TOL * max(1.0, abs(y)) for x, y in zip(a, b))


def _aimip_disagreements(rows=None) -> dict[str, str]:
    bad = {}
    for name, a, sb, db in (_aimip_rows() if rows is None else rows):
        if any(o is not None and not _close(a, o) for o in (sb, db)):
            bad[name] = f"aimip {a} vs spec {sb} vs driver {db}"
    return bad


def test_aimip_disagreement_detector_catches_either_side():
    """Non-vacuity: a knob off from its spec OR its driver range is flagged;
    one with no other declaration is not."""
    rows = [("a", (0.0, 1.0), (0.0, 2.0), None),
            ("b", (0.0, 1.0), None, (0.5, 1.0)),
            ("c", (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ("d", (0.0, 1.0), None, None)]
    assert set(_aimip_disagreements(rows)) == {"a", "b"}


def test_every_aimip_knob_maps_to_a_config_field():
    """Non-vacuity: the knob -> field map covers every knob, and most knobs
    find a spec entry (if the lookup broke, everything would 'agree')."""
    from legoesm.training.aimip_params import AIMIP_CLASSICAL_CONSTRAINTS
    fields = _aimip_knob_fields()
    assert {c.name for c in AIMIP_CLASSICAL_CONSTRAINTS} <= set(fields)
    rows = _aimip_rows()
    assert sum(sb is not None for _, _, sb, _ in rows) >= len(rows) // 2


def test_no_new_aimip_knob_declares_another_range():
    new = sorted(set(_aimip_disagreements()) - set(_AIMIP_DISAGREE))
    assert not new, (
        f"{new}: the AIMIP knob range differs from its scheme spec or driver "
        f"range. Reconcile, or list it in _AIMIP_DISAGREE with what is owed.")


def test_the_aimip_disagreement_list_only_shrinks():
    fixed = sorted(set(_AIMIP_DISAGREE) - set(_aimip_disagreements()))
    assert not fixed, f"{fixed} now agree — delete their _AIMIP_DISAGREE entries."
