"""Unit tests for the CLUBB scheme configuration (``clubb_config.py``).

Guards the CAM-default flag/parameter values (the authoritative settings the
port targets) against accidental drift, and checks pytree/typing hygiene.

Part of the fuller CLUBB port — see ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics.turbulence.clubb_config import (
    CLUBBConfig,
    CLUBBFlags,
    CLUBBParams,
)
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig

# CAM namelist defaults: ../../CESM relative to the repo root (outside the
# repo). Present in the dev tree; absent in CI -> source-derived test skips.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NAMELIST = (
    _REPO_ROOT.parents[1]
    / "CESM/components/cam/bld/namelist_files/namelist_defaults_cam.xml"
)


def test_config_constructs_with_defaults():
    cfg = CLUBBConfig()
    assert isinstance(cfg.flags, CLUBBFlags)
    assert isinstance(cfg.params, CLUBBParams)
    assert isinstance(cfg.surface, SurfaceLayerConfig)
    assert cfg.clubb_dt == 300.0


def test_flags_are_static_no_leaves():
    """CLUBBFlags must contribute NO dynamic JAX leaves (static pytree node).

    This is the contract that lets kernels branch on flags with Python ``if``
    under jit. The flag values live in the treedef, not the leaves.
    """
    flag_leaves = jax.tree_util.tree_leaves(CLUBBFlags())
    assert flag_leaves == [], f"flags leaked dynamic leaves: {flag_leaves}"
    # Within a full config, the only leaves are params/surface/tolerances.
    cfg = CLUBBConfig()
    cfg_leaves = jax.tree_util.tree_leaves(cfg)
    param_leaves = jax.tree_util.tree_leaves(CLUBBParams())
    surface_leaves = jax.tree_util.tree_leaves(cfg.surface)
    # 5 tolerances + clubb_dt = 6 scalar fields on CLUBBConfig itself.
    assert len(cfg_leaves) == len(param_leaves) + len(surface_leaves) + 6


def test_flags_hashable_and_branchable_under_jit():
    """A jitted fn can branch on a static flag without tracer errors.

    Mirrors the real legoESM pattern: the scheme config is CLOSURE-captured by
    the physics factory (it also carries a ``str`` leaf, ``surface.bulk_scheme``,
    so it cannot be a traced arg), and only arrays cross the jit boundary. The
    static flag is then a concrete Python bool inside the trace.
    """
    assert hash(CLUBBFlags()) == hash(CLUBBFlags())

    def make_step(cfg):
        @jax.jit
        def step(x):
            if cfg.flags.l_use_cloud_cover:   # Python `if` on a static flag
                return x * 2.0
            return x * 3.0
        return step

    cfg = CLUBBConfig()
    assert float(make_step(cfg)(jnp.asarray(1.0))) == 2.0   # default True
    cfg_off = cfg._replace(flags=CLUBBFlags(l_use_cloud_cover=False))
    assert float(make_step(cfg_off)(jnp.asarray(1.0))) == 3.0


def test_is_pytree_round_trip():
    cfg = CLUBBConfig()
    leaves, treedef = jax.tree_util.tree_flatten(cfg)
    assert len(leaves) > 90  # params (~97 floats) + surface + tolerances
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == cfg


@pytest.mark.parametrize(
    "field, expected",
    [
        # CAM namelist OVERRIDES of the CLUBB library defaults (the key
        # correctness anchors — these must NOT silently revert to lib values).
        ("C2rt", 1.0),      # lib 2.0
        ("C2rtthl", 1.3),   # lib 2.0
        ("C4", 5.2),        # lib 2.0
        ("C6rt", 4.0),      # lib 2.0
        ("C8", 4.2),        # lib 0.5
        ("C8b", 0.0),       # lib 0.02
        ("C11", 0.7),       # lib 0.4
        ("C11b", 0.35),     # lib 0.4
        ("C14", 2.2),       # lib 1.0
        ("C_uu_shr", 0.3),  # lib 0.4
        ("C_wp3_pr_turb", 0.4),   # lib 0.0
        ("C_wp2_splat", 0.0),     # lib 2.0
        ("beta", 2.4),      # lib 1.0
        ("gamma_coef", 0.308),    # lib 0.25
        ("c_K1", 0.75),     # lib 0.2
        ("c_K8", 1.25),     # lib 5.0
        ("c_K10h", 0.3),    # lib 1.0
        ("nu2", 5.0),       # lib 1.0
        ("nu9", 20.0),      # lib 10.0
        ("mult_coef", 1.0), # lib 0.5
        ("lmin_coef", 0.1), # lib 0.5
        ("Skw_max_mag", 4.5),     # lib 10.0
        ("Skw_denom_coef", 0.0),  # lib 4.0
        ("up2_sfc_coef", 2.0),    # lib 4.0
        ("lambda0_stability_coef", 0.04),  # lib 0.03
        ("C_invrs_tau_N2", 0.1),  # lib 0.4
        ("C_invrs_tau_shear", 0.02),       # lib 0.15
        ("C_invrs_tau_N2_clear_wp3", 0.0), # lib 1.0
    ],
)
def test_cam_default_param_overrides(field, expected):
    assert getattr(CLUBBParams(), field) == expected


@pytest.mark.parametrize(
    "field, expected",
    [
        # CAM namelist flag values that DIFFER from the CLUBB library defaults.
        ("l_predict_upwp_vpwp", False),     # lib True
        ("l_use_cloud_cover", True),        # lib False
        ("l_use_C7_Richardson", False),     # lib True
        ("l_vert_avg_closure", True),       # lib False
        ("l_trapezoidal_rule_zt", True),    # lib False
        ("l_trapezoidal_rule_zm", True),    # lib False
        ("l_stability_correct_tau_zm", True),    # lib False
        ("l_rcm_supersat_adj", False),      # lib True
        ("l_use_tke_in_wp3_pr_turb_term", False),  # lib True
        ("l_call_pdf_closure_twice", True),  # lib False -> CAM True
        ("l_damp_wp2_using_em", False),      # lib True -> CAM False
        ("l_damp_wp3_Skw_squared", False),   # lib True -> CAM False
        ("l_diag_Lscale_from_tau", False),   # lib True -> CAM False
        ("penta_solve_method", 1),          # lib 2
        ("tridiag_solve_method", 1),        # lib 2
        ("grid_remap_method", 1),           # lib 2(ppm) -> CAM 1
        # Stable enum defaults (not in namelist -> library).
        ("iiPDF_type", 1),                  # ADG1
        ("saturation_formula", 3),          # flatau
        ("fill_holes_type", 2),
    ],
)
def test_cam_default_flag_values(field, expected):
    val = getattr(CLUBBFlags(), field)
    assert val == expected
    # bool/int distinction matters for downstream gating; verify exact type.
    assert type(val) is type(expected)


def _parse_namelist_clubb_base_defaults(text):
    """Extract base (unconditioned) clubb_l_*/enum defaults from the XML.

    A base entry has only whitespace between the tag name and ``>`` (no
    variant attributes like silhs="1"/phys="cam7"). Returns dict name->value
    (bool for .true./.false., int for integers).
    """
    out = {}
    # bool flags
    for m in re.finditer(
        r"<(clubb_l_[a-zA-Z0-9_]+)\s*>\s*\.(true|false)\.\s*</\1>", text
    ):
        out[m.group(1)] = (m.group(2) == "true")
    # integer enum flags
    for m in re.finditer(
        r"<(clubb_(?:penta_solve_method|tridiag_solve_method|grid_remap_method|"
        r"grid_adapt_in_time_method|fill_holes_type|ipdf_call_placement))\s*>\s*"
        r"(\d+)\s*</\1>",
        text,
    ):
        out[m.group(1)] = int(m.group(2))
    return out


@pytest.mark.skipif(not _NAMELIST.exists(), reason="CESM namelist not in tree")
def test_flags_match_cam_namelist_source():
    """Source-of-truth check: CLUBBFlags == namelist_defaults_cam.xml base.

    Non-vacuous: compares against the actual XML, not transcribed values. Only
    flags PRESENT in the namelist are checked (others fall back to library
    defaults, which this test does not police).
    """
    text = _NAMELIST.read_text()
    nl = _parse_namelist_clubb_base_defaults(text)
    assert nl, "parsed no clubb defaults — parser/namelist format drift"
    flags = CLUBBFlags()
    mismatches = []
    for nl_name, nl_val in nl.items():
        field = nl_name[len("clubb_"):]
        if not hasattr(flags, field):
            continue  # flag not modeled (e.g. host-side); skip
        got = getattr(flags, field)
        if bool(got) != bool(nl_val) if isinstance(nl_val, bool) else got != nl_val:
            mismatches.append((field, got, nl_val))
    assert not mismatches, f"CLUBBFlags disagree with CAM namelist: {mismatches}"


def test_all_params_are_float():
    p = CLUBBParams()
    for name in p._fields:
        v = getattr(p, name)
        assert isinstance(v, float), f"{name} should be float, got {type(v)}"


def test_overrides_compose():
    """Overriding nested config preserves the rest (NamedTuple._replace)."""
    cfg = CLUBBConfig()
    cfg2 = cfg._replace(params=cfg.params._replace(C8=9.9))
    assert cfg2.params.C8 == 9.9
    assert cfg2.params.C4 == 5.2          # untouched
    assert cfg2.flags is cfg.flags         # untouched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
