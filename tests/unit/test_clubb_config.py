"""Unit tests for the CLUBB scheme configuration (the config section of ``clubb.py``).

Guards the CAM-default parameter values and the CAM-default model-FLAG
reference table (the authoritative settings the port targets) against
accidental drift, and checks pytree/typing hygiene. The flags are no longer a
runtime config class — only the CAM-default tree is implemented — so the flag
tests parse the reference comment table at the end of ``clubb.py`` (keeping
the namelist source-of-truth tripwire alive on the comments themselves).

Part of the fuller CLUBB port — see ``docs/dev-notes/clubb.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import pytest
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod
from legoesm.atmosphere.physics.turbulence.clubb import (
    CLUBBConfig,
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

_CLUBB_PY = Path(clubb_mod.__file__)


def _parse_flag_reference_table():
    """Parse the CAM-default flag table from the comment block in clubb.py.

    The table rows are ``#   <name> = <value>`` lines after the
    "CAM-default CLUBB model-flag values (reference table)" header. Values are
    Python literals (True/False/int). Returns dict name -> value.
    """
    text = _CLUBB_PY.read_text()
    m = re.search(r"# CAM-default CLUBB model-flag values \(reference table\)\n(.*)\Z",
                  text, re.S)
    assert m, "flag reference table header missing from clubb.py"
    out = {}
    for row in re.finditer(r"^#   (\w+) = (True|False|\d+)", m.group(1), re.M):
        name, val = row.group(1), row.group(2)
        out[name] = {"True": True, "False": False}.get(val, None)
        if out[name] is None:
            out[name] = int(val)
    return out


def test_config_constructs_with_defaults():
    cfg = CLUBBConfig()
    assert isinstance(cfg.params, CLUBBParams)
    assert isinstance(cfg.surface, SurfaceLayerConfig)
    assert cfg.clubb_dt == 300.0
    # The flags field is GONE by design (only the CAM tree is implemented).
    assert "flags" not in cfg._fields


#: The scalar fields CLUBBConfig carries in its OWN right, beside ``params``
#: and ``surface``.  Pinned by NAME rather than by count: the old test asserted
#: "+ 10" with the names in a comment beside it, and a comment and a count
#: drift independently -- six legitimate float fields were added and the count
#: went red while the comment quietly stopped describing the class.  A name set
#: fails readably ("unexpected field: ..."), forces a one-line diff per
#: addition, and keeps the human review the count was there for.
_CLUBB_CONFIG_SCALAR_FIELDS = frozenset({
    "clubb_dt", "w_tol", "rt_tol", "thl_tol", "wp2_max", "tke_min", "T0",
    "prognostic", "cloud_buoyancy", "cloud_source",
    "trop_cloud_top_press", "trop_cloud_taper_lnp_width",
    "pdf_variance_scale", "q_flux_scale",
    "q_flux_scale_sigma_lo", "q_flux_scale_sigma_hi",
})


#: Flag-shaped leaves that ALREADY live in the pytree. Pre-existing debt, not
#: policy: ``prognostic`` and ``cloud_buoyancy`` are bools and ``cloud_source``
#: is a str selector, all three older than the rule they violate (they appear
#: in the original leaf-count comment). Shrink-only, and
#: ``test_the_bool_exemptions_have_not_outlived_their_reason`` deletes an entry
#: the moment it stops describing reality. Moving one out of the pytree changes
#: flatten structure and leaf order, so it belongs with the next
#: serialization-format change, not a standalone commit.
_BOOL_LEAF_EXEMPTIONS = frozenset({"prognostic", "cloud_buoyancy", "cloud_source"})


def test_config_has_no_flag_leaves():
    """The config's leaves are exactly params + surface + the scalar fields.

    The former ``CLUBBFlags`` static node was removed and nothing flag-like may
    leak back into the pytree leaves.  Note what that requires and what a
    weaker check would miss: a Python ``bool`` IS a scalar pytree leaf, so
    "every extra field contributes one leaf and is a scalar" would wave a new
    ``use_new_scheme: bool`` straight through -- exactly the class this test
    exists to catch.  The field NAMES are therefore pinned, and the count falls
    out of them.
    """
    cfg = CLUBBConfig()
    extra = set(cfg._fields) - {"params", "surface"}
    assert extra == set(_CLUBB_CONFIG_SCALAR_FIELDS), (
        f"CLUBBConfig's own scalar fields changed. Added: "
        f"{sorted(extra - _CLUBB_CONFIG_SCALAR_FIELDS)}; removed: "
        f"{sorted(set(_CLUBB_CONFIG_SCALAR_FIELDS) - extra)}. If the addition "
        f"is a genuine scalar tunable, add its name above; if it is a FLAG, "
        f"it does not belong in the pytree at all.")
    cfg_leaves = jax.tree_util.tree_leaves(cfg)
    param_leaves = jax.tree_util.tree_leaves(CLUBBParams())
    surface_leaves = jax.tree_util.tree_leaves(cfg.surface)
    assert len(cfg_leaves) == (len(param_leaves) + len(surface_leaves)
                               + len(_CLUBB_CONFIG_SCALAR_FIELDS)), (
        "a field on CLUBBConfig contributes more than one leaf -- a container "
        "leaked into the pytree")


def test_config_scalar_fields_are_not_flags():
    """None of the pinned scalar fields is a bool.

    The non-vacuity half of the test above: pinning names stops a NEW flag
    silently joining the set, and this stops an EXISTING name being changed
    into one.

    IT FOUND ONE ON ITS FIRST RUN, and the finding is recorded rather than
    fixed here: ``cloud_buoyancy`` is a bool on CLUBBConfig, i.e. a model flag
    sitting in the pytree leaves -- the thing the class docstring says may not
    happen. It is as old as ``prognostic`` (both are in the original count's
    own comment), so it is pre-existing and not this change's to move; moving
    a field out of the pytree is an API change with an owner decision behind
    it. The three grandfathered names are listed explicitly so the exemption
    is VISIBLE rather than implied, and so a fourth cannot join them quietly.
    ``cloud_source`` is a str selector (a scheme name, not a toggle).
    """
    cfg = CLUBBConfig()
    bad = [f for f in _CLUBB_CONFIG_SCALAR_FIELDS - _BOOL_LEAF_EXEMPTIONS
           if isinstance(getattr(cfg, f), bool)]
    assert not bad, (
        f"these CLUBBConfig fields are bools, i.e. model FLAGS living in the "
        f"pytree leaves, which is what CLUBBFlags was removed to prevent: "
        f"{bad}")


def test_the_bool_exemptions_have_not_outlived_their_reason():
    """An exemption that survives its justification is how debt becomes policy.

    Review finding on the first version of this: naming ``cloud_buoyancy`` as
    grandfathered makes it look deliberate forever unless something forces the
    entry to expire. So the exemption list must describe reality -- every name
    on it has to STILL be a flag-shaped leaf. The day someone gives
    ``cloud_buoyancy`` a proper static home, this test fails and the exemption
    is deleted in the same commit that earned the deletion.
    """
    cfg = CLUBBConfig()
    stale = [f for f in _BOOL_LEAF_EXEMPTIONS
             if not isinstance(getattr(cfg, f), (bool, str))]
    assert not stale, (
        f"these names are exempted from the no-flags rule but are no longer "
        f"flag-shaped: {stale}. The exemption has outlived its reason — "
        f"delete it from _BOOL_LEAF_EXEMPTIONS.")
    # A ratchet, not a policy: the list may SHRINK, never grow. A fourth
    # flag-shaped leaf has to be argued for, not appended.
    assert _BOOL_LEAF_EXEMPTIONS <= {"prognostic", "cloud_buoyancy", "cloud_source"}, (
        f"_BOOL_LEAF_EXEMPTIONS holds {sorted(_BOOL_LEAF_EXEMPTIONS)}. It is "
        f"pre-existing debt with a permitted direction: shrink only, and only "
        f"from the three grandfathered names.")
    # ...and the two pinned structures cannot drift apart.
    assert _BOOL_LEAF_EXEMPTIONS <= _CLUBB_CONFIG_SCALAR_FIELDS, (
        f"exempted names that are not CLUBBConfig scalar fields at all: "
        f"{sorted(_BOOL_LEAF_EXEMPTIONS - _CLUBB_CONFIG_SCALAR_FIELDS)}")


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
        # CAM namelist flag values that DIFFER from the CLUBB library defaults
        # (now reference-table rows, not config fields).
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
def test_cam_default_flag_table_values(field, expected):
    table = _parse_flag_reference_table()
    assert field in table, f"{field} missing from the clubb.py flag table"
    val = table[field]
    assert val == expected
    # bool/int distinction matters; the table must keep exact literal types.
    assert type(val) is type(expected)


def test_flag_table_complete():
    """The reference table keeps all 65 flags (8 enum + 57 logical)."""
    table = _parse_flag_reference_table()
    assert len(table) == 65, f"flag table has {len(table)} rows, expected 65"


def test_cam_fill_holes_type_constant_matches_table():
    """The hardcoded dispatch constant must agree with the reference table."""
    table = _parse_flag_reference_table()
    assert clubb_mod._CAM_FILL_HOLES_TYPE == table["fill_holes_type"]


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
def test_flag_table_matches_cam_namelist_source():
    """Source-of-truth check: the clubb.py flag table == namelist base values.

    Non-vacuous: compares against the actual XML, not transcribed values. Only
    flags PRESENT in the namelist are checked (others fall back to library
    defaults, which this test does not police).
    """
    text = _NAMELIST.read_text()
    nl = _parse_namelist_clubb_base_defaults(text)
    assert nl, "parsed no clubb defaults — parser/namelist format drift"
    table = _parse_flag_reference_table()
    mismatches = []
    for nl_name, nl_val in nl.items():
        field = nl_name[len("clubb_"):]
        if field not in table:
            continue  # flag not modeled (e.g. host-side); skip
        got = table[field]
        if bool(got) != bool(nl_val) if isinstance(nl_val, bool) else got != nl_val:
            mismatches.append((field, got, nl_val))
    assert not mismatches, f"clubb.py flag table disagrees with CAM namelist: {mismatches}"


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
    assert cfg2.surface is cfg.surface     # untouched


def test_toc_line_numbers_accurate():
    """Tripwire: the line-numbered TOC at the top of clubb.py must be exact.

    Each ``N.  [line  L] Title`` TOC row must point at the actual
    ``# N. Title`` section header line, and the flag-table pointer in the TOC
    header must point at the reference-table banner. Any edit that shifts the
    file must regenerate the TOC numbers (cheap: they are asserted here, so
    drift fails loudly instead of silently lying to readers).

    WHAT SHOULD TRIP THIS, and the one-command fix: any edit that moves a
    section header -- which is most edits to a 6700-line module, so treat a red
    here as bookkeeping, not as a defect in the code under test. The numbers
    are derived by ``scripts/validate/regen_clubb_toc.py``; run it to repin,
    or with ``--check`` to see the drift. On 2026-09-23 all 19 rows and the
    pointer had drifted at once, which is what a hand-maintained derived
    artifact does and why the generator exists.
    """
    lines = _CLUBB_PY.read_text().splitlines()
    toc_rows = []
    flag_ptr = None
    for ln in lines[:80]:
        m = re.match(r"  (\d+)\.\s+\[line\s+(\d+)\]", ln)
        if m:
            toc_rows.append((int(m.group(1)), int(m.group(2))))
        m2 = re.search(r"flag reference table at line (\d+)", ln)
        if m2:
            flag_ptr = int(m2.group(1))
    assert len(toc_rows) == 19, f"expected 19 TOC rows, parsed {len(toc_rows)}"
    for n, cited in toc_rows:
        actual = lines[cited - 1]
        assert re.match(rf"# {n}\. ", actual), (
            f"TOC row {n} cites line {cited}, but that line is: {actual!r}. "
            f"The numbers are DERIVED, not maintained: run "
            f"`python scripts/validate/regen_clubb_toc.py` to repin them "
            f"(`--check` reports the drift without writing).")
    assert flag_ptr is not None, "flag-table pointer missing from TOC header"
    assert "CAM-default CLUBB model-flag values" in lines[flag_ptr - 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
