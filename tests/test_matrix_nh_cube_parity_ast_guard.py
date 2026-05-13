"""new_test_dycores iter-9: AST guard for the iter-5/6/7 NH cube parity
fix in ``scripts/run_atmosphere_test_matrix.py``.

Iter-5/6/7 closed the cube NH parity gap for DCMIP TC1/TC2/TC3 by
enabling two FV3-faithful flags on each cube branch:

    use_fv3_vector_halo_uv      = True
    use_fv3_a2b_ord4_vector_uv  = True

(iter-697 vector halo + iter-698 a2b_ord4 4th-order corner cascade,
factored in ``make_fv3_faithful_nh_config``.)  Pre-iter-5 the matrix
runner had remained on the legacy scalar-halo path, which produced
cube |w|_max 22x worse than ico/spectral on TC1, 13x on TC2, and
2.2x on TC3.

This sentinel guards against an inadvertent removal during a future
refactor: it pattern-matches the runner source to confirm both flags
appear within EACH of the three NH cube ``CompressibleEulerConfig``
constructors (TC1, TC2, TC3 cube branches).  Catches a silent revert
that would otherwise only show up as a matrix-runner |w|_max
regression on the next full-matrix run.
"""
from __future__ import annotations

import re
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_atmosphere_test_matrix.py"
)


def _runner_source() -> str:
    return SCRIPT_PATH.read_text()


def _find_nh_config_blocks() -> dict[str, str]:
    """Extract the body of each ``CompressibleEulerConfig(...)`` call
    inside the three ``test_case ==`` branches of ``run_nonhydrostatic``.

    Returns a dict mapping ``{"tc1", "tc2a", "tc3"}`` to the substring
    spanning ``CompressibleEulerConfig(`` ... matching closing ``)``.
    """
    src = _runner_source()
    blocks: dict[str, str] = {}
    for tc in ("tc1", "tc2a", "tc3"):
        # Locate the test_case == "tcN" branch.
        anchor = re.search(rf'test_case == "{re.escape(tc)}"', src)
        assert anchor is not None, (
            f"matrix-runner branch for test_case={tc!r} not found"
        )
        # Find the first CompressibleEulerConfig( after the anchor.
        cfg_start = src.index("CompressibleEulerConfig(", anchor.end())
        # Walk parentheses to find matching close.
        depth = 0
        i = cfg_start
        while i < len(src):
            c = src[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    blocks[tc] = src[cfg_start : i + 1]
                    break
            i += 1
        else:
            raise AssertionError(
                f"unbalanced CompressibleEulerConfig(...) for {tc!r}"
            )
    return blocks


def test_tc1_cube_has_vector_halo_uv_flag():
    """iter-5 sentinel: TC1 cube enables ``use_fv3_vector_halo_uv=True``."""
    blocks = _find_nh_config_blocks()
    assert "use_fv3_vector_halo_uv=True" in blocks["tc1"], (
        "iter-5 regression: TC1 cube NH config lost "
        "``use_fv3_vector_halo_uv=True``.  This will re-open the cube "
        "NH TC1 parity gap (0.014 -> 0.327 m/s)."
    )


def test_tc1_cube_has_a2b_ord4_flag():
    """iter-5 sentinel: TC1 cube enables ``use_fv3_a2b_ord4_vector_uv=True``."""
    blocks = _find_nh_config_blocks()
    assert "use_fv3_a2b_ord4_vector_uv=True" in blocks["tc1"], (
        "iter-5 regression: TC1 cube NH config lost "
        "``use_fv3_a2b_ord4_vector_uv=True``."
    )


def test_tc2a_cube_has_vector_halo_uv_flag():
    """iter-6 sentinel: TC2 cube enables ``use_fv3_vector_halo_uv=True``."""
    blocks = _find_nh_config_blocks()
    assert "use_fv3_vector_halo_uv=True" in blocks["tc2a"], (
        "iter-6 regression: TC2 cube NH config lost "
        "``use_fv3_vector_halo_uv=True``.  This re-opens the TC2 cube "
        "NH parity gap (0.32 -> 4.65 m/s)."
    )


def test_tc2a_cube_has_a2b_ord4_flag():
    """iter-6 sentinel: TC2 cube enables ``use_fv3_a2b_ord4_vector_uv=True``."""
    blocks = _find_nh_config_blocks()
    assert "use_fv3_a2b_ord4_vector_uv=True" in blocks["tc2a"], (
        "iter-6 regression: TC2 cube NH config lost "
        "``use_fv3_a2b_ord4_vector_uv=True``."
    )


def test_tc3_cube_has_vector_halo_uv_flag():
    """iter-7 sentinel: TC3 cube enables ``use_fv3_vector_halo_uv=True``."""
    blocks = _find_nh_config_blocks()
    assert "use_fv3_vector_halo_uv=True" in blocks["tc3"], (
        "iter-7 regression: TC3 cube NH config lost "
        "``use_fv3_vector_halo_uv=True``.  This re-opens the TC3 cube "
        "NH parity gap (7.36 -> 23.07 m/s)."
    )


def test_tc3_cube_has_a2b_ord4_flag():
    """iter-7 sentinel: TC3 cube enables ``use_fv3_a2b_ord4_vector_uv=True``."""
    blocks = _find_nh_config_blocks()
    assert "use_fv3_a2b_ord4_vector_uv=True" in blocks["tc3"], (
        "iter-7 regression: TC3 cube NH config lost "
        "``use_fv3_a2b_ord4_vector_uv=True``."
    )


def test_all_nh_cube_branches_use_d_con_cv():
    """iter-12 sentinel: all 3 NH cube branches enable
    ``use_fv3_d_con_cv=True`` (c_v denominator for NH d_con heating;
    matches FV3-faithful factory default in
    ``make_fv3_faithful_nh_config``).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "use_fv3_d_con_cv=True" in blocks[tc], (
            f"iter-12 regression: {tc} cube NH config lost "
            "``use_fv3_d_con_cv=True``.  This re-introduces a ~40 % "
            "heating-magnitude error in the NH d_con term "
            "(c_pd vs c_v denominator)."
        )


def test_sw_cube_uses_iter1009_dual_target_config():
    """iter-1/8 sentinel: matrix runner SW W2/W5 cube branch routes
    through the canonical ``iter1009_dual_target_config(n)`` helper
    instead of an inline ``CDGridShallowWaterConfig(...)`` construction.

    The iter-1 calibration update (damp_v 0.06 -> 0.030) was factored
    via the iter-8 helper substitution.  A refactor that inlines a
    bare ``CDGridShallowWaterConfig(...)`` again would silently drop
    the iter-1030 W5-best stability margin if the inlined fields
    don't match the helper output bit-for-bit.
    """
    src = _runner_source()
    assert "iter1009_dual_target_config(n)" in src, (
        "iter-1/iter-8 regression: matrix runner SW cube branch no "
        "longer routes through the canonical "
        "``iter1009_dual_target_config(n)`` helper.  Re-route or pin "
        "the inlined config to ``damp_v=0.030`` / "
        "``div_damp=8*_div_damp_cube(n)``."
    )
