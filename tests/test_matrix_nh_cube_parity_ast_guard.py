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


def test_all_nh_cube_branches_use_dynamic_exner():
    """iter-13 sentinel: all 3 NH cube branches enable
    ``use_fv3_dynamic_exner=True``.  Matches FV3 live pkz at the
    slow-tendency d_con denominators (compose with iter-12 d_con_cv).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "use_fv3_dynamic_exner=True" in blocks[tc], (
            f"iter-13 regression: {tc} cube NH config lost "
            "``use_fv3_dynamic_exner=True``."
        )


def test_all_nh_cube_branches_use_metric_aware_d_con():
    """iter-13 sentinel: all 3 NH cube branches enable
    ``use_fv3_metric_aware_d_con=True`` (rsin2/cosa_s form at
    damp_v d_con site; PE 338 mirror).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "use_fv3_metric_aware_d_con=True" in blocks[tc], (
            f"iter-13 regression: {tc} cube NH config lost "
            "``use_fv3_metric_aware_d_con=True``."
        )


def test_all_nh_cube_branches_use_d_con_top_zero_levels():
    """iter-14 sentinel: all 3 NH cube branches set
    ``d_con_top_zero_levels=2`` (FV3 iter-431 sponge consistency;
    zero d_con heating in top 2 model levels per FV3
    ``dyn_core.F90:773-805``).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "d_con_top_zero_levels=2" in blocks[tc], (
            f"iter-14 regression: {tc} cube NH config lost "
            "``d_con_top_zero_levels=2``."
        )


def test_all_nh_cube_branches_use_heat_source_del2_iters():
    """iter-15 sentinel: all 3 NH cube branches set
    ``heat_source_del2_iters=2`` (FV3 iter-457; del-2 smoothing of
    ``_d_con_sum`` heat source, ``nf_ke=2`` at ``nord=1`` per FV3
    ``dyn_core.F90:1755-1756``).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "heat_source_del2_iters=2" in blocks[tc], (
            f"iter-15 regression: {tc} cube NH config lost "
            "``heat_source_del2_iters=2``."
        )


def test_all_nh_cube_branches_use_delt_max():
    """iter-16 sentinel: all 3 NH cube branches set ``delt_max=1.0``
    (factory default per-step heating cap; FV3 iter-218
    ``dyn_core.F90:1774``: |Δθ_p · Π| ≤ dt · delt_max).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "delt_max=1.0" in blocks[tc], (
            f"iter-16 regression: {tc} cube NH config lost "
            "``delt_max=1.0``."
        )


def test_all_nh_cube_branches_use_corner_div_damp_factory_pair():
    """iter-17 sentinel: all 3 NH cube branches set the FV3 factory
    ``corner_div_damp_nord=1`` + ``corner_div_damp_d4_bg=0.16``
    pair (del-4 corner-divergence damping; FV3 iter-168
    ``sw_core.F90:1641-1822 d_sw5``).
    """
    blocks = _find_nh_config_blocks()
    for tc in ("tc1", "tc2a", "tc3"):
        assert "corner_div_damp_nord=1" in blocks[tc], (
            f"iter-17 regression: {tc} cube NH config lost "
            "``corner_div_damp_nord=1``."
        )
        assert "corner_div_damp_d4_bg=0.16" in blocks[tc], (
            f"iter-17 regression: {tc} cube NH config lost "
            "``corner_div_damp_d4_bg=0.16``."
        )


def _find_pe_baroclinic_cube_config_block() -> str:
    """Locate the matrix-runner PE ``baroclinic`` cube
    ``PrimitiveEquationConfig(...)`` constructor (also used by
    ``rotated_baroclinic`` / ``rotated_steady`` / ``gravity_wave_3_1``).

    Anchor: the unique ``_resolve_dt_cube(n, label="baroclinic")``
    call, with whitespace tolerance.  Whitespace-flexible regex so
    a reformat (``label = "baroclinic"``) doesn't break the anchor.
    """
    src = _runner_source()
    anchor = re.search(
        r'_resolve_dt_cube\s*\(\s*n\s*,\s*label\s*=\s*"baroclinic"\s*\)',
        src,
    )
    assert anchor is not None, (
        "matrix-runner baroclinic cube ``_resolve_dt_cube(n, "
        "label=\"baroclinic\")`` anchor not found"
    )
    cfg_start = src.index(
        "PrimitiveEquationConfig(", anchor.end(),
    )
    # Sanity: anchor must be within ~2000 chars of the following PE
    # config opening so we don't accidentally grab a much later block.
    assert cfg_start - anchor.end() < 2000, (
        f"baroclinic anchor too far ({cfg_start - anchor.end()} "
        "chars) from next PrimitiveEquationConfig( — wrong block "
        "likely matched"
    )
    depth = 0
    i = cfg_start
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return src[cfg_start : i + 1]
        i += 1
    raise AssertionError("unbalanced PE baroclinic config block")


def test_pe_baroclinic_cube_uses_metric_aware_d_con():
    """iter-18 sentinel: PE baroclinic / rotated_steady /
    rotated_baroclinic / gravity_wave_3_1 cube branch enables
    ``use_fv3_metric_aware_d_con=True`` (PE iter-338 factory
    default).  Probed at C36 day-1 on gravity_wave_3_1 — cube max|v|
    27.5 -> 22.4 (-18.5%), now matches ico 20.0 / spec 20.3.
    """
    block = _find_pe_baroclinic_cube_config_block()
    assert "use_fv3_metric_aware_d_con=True" in block, (
        "iter-18 regression: PE baroclinic cube config lost "
        "``use_fv3_metric_aware_d_con=True``.  This re-opens the "
        "PE cube gravity_wave_3_1 max|v| gap (22.4 -> 27.5)."
    )


def test_pe_baroclinic_cube_uses_d_con_top_zero_levels():
    """iter-19 sentinel: PE baroclinic cube branch enables
    ``d_con_top_zero_levels=2`` (PE iter-433 factory default,
    analog of NH iter-14 enabled on the matrix in iter-14).
    """
    block = _find_pe_baroclinic_cube_config_block()
    assert "d_con_top_zero_levels=2" in block, (
        "iter-19 regression: PE baroclinic cube config lost "
        "``d_con_top_zero_levels=2``."
    )


def test_pe_baroclinic_cube_uses_delt_max():
    """iter-20 sentinel: PE baroclinic cube branch enables
    ``delt_max=1.0`` (PE iter-218 factory default per-step heating
    cap; analog of NH iter-16 enabled on the matrix).
    """
    block = _find_pe_baroclinic_cube_config_block()
    assert "delt_max=1.0" in block, (
        "iter-20 regression: PE baroclinic cube config lost "
        "``delt_max=1.0``."
    )


def test_pe_baroclinic_cube_uses_heat_source_del2_iters():
    """iter-21 sentinel: PE baroclinic cube branch enables
    ``heat_source_del2_iters=2`` (PE iter-458 factory default;
    del-2 smoothing of ``_d_con_sum`` heat source; analog of NH
    iter-15 enabled on the matrix in iter-15).
    """
    block = _find_pe_baroclinic_cube_config_block()
    assert "heat_source_del2_iters=2" in block, (
        "iter-21 regression: PE baroclinic cube config lost "
        "``heat_source_del2_iters=2``."
    )


def _find_pe_held_suarez_cube_config_block() -> str:
    """Locate the matrix-runner PE held_suarez cube branch
    ``PrimitiveEquationConfig(...)`` constructor (the one inside
    ``run_held_suarez``, distinguished from the baroclinic block by
    the unique ``smagorinsky_cs=`` field that only the held_suarez
    config sets).

    Anchor robust to whitespace variations around the assignment.
    """
    src = _runner_source()
    anchor = re.search(r"smagorinsky_cs\s*=\s*\S", src)
    assert anchor is not None, (
        "matrix-runner PE held_suarez ``smagorinsky_cs=`` anchor "
        "not found"
    )
    open_idx = src.rfind("PrimitiveEquationConfig(", 0, anchor.start())
    assert open_idx != -1, (
        "no PrimitiveEquationConfig( before held_suarez anchor"
    )
    depth = 0
    i = open_idx
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return src[open_idx : i + 1]
        i += 1
    raise AssertionError("unbalanced held_suarez config block")


def _find_pe_amip_cube_config_block() -> str:
    """Locate the matrix-runner PE AMIP cube branch
    ``PrimitiveEquationConfig(...)`` constructor (inside ``run_amip``,
    distinguished by the unique ``held_suarez_init(grid, sigma,
    T_init=…)`` call that follows it.  Anchor accepts any numeric
    value for ``T_init`` so a default-tweak doesn't silently break
    the sentinel).
    """
    src = _runner_source()
    anchor = re.search(
        r"held_suarez_init\(\s*grid\s*,\s*sigma\s*,\s*T_init\s*=\s*[\d.]+\s*\)",
        src,
    )
    assert anchor is not None, (
        "matrix-runner AMIP ``held_suarez_init(grid, sigma, T_init=...)``"
        " anchor not found"
    )
    open_idx = src.rfind("PrimitiveEquationConfig(", 0, anchor.start())
    assert open_idx != -1, (
        "no PrimitiveEquationConfig( before AMIP anchor"
    )
    # Sanity: the anchor must be within ~2000 chars of the preceding
    # ``PrimitiveEquationConfig(`` opening, otherwise an unrelated
    # earlier ``PrimitiveEquationConfig(`` (e.g., the held_suarez or
    # baroclinic block) was picked up and the walker would return
    # the wrong block.  Today's measured gap is 563 chars; the
    # ceiling here is generous to allow for typical comment growth.
    assert anchor.start() - open_idx < 2000, (
        f"AMIP anchor too far ({anchor.start() - open_idx} chars) "
        "from preceding PrimitiveEquationConfig( — wrong block "
        "likely matched"
    )
    depth = 0
    i = open_idx
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return src[open_idx : i + 1]
        i += 1
    raise AssertionError("unbalanced AMIP config block")


def test_sw_w6_wired_for_all_four_grids():
    """iter-24/25 sentinel: SW Williamson 6 (Rossby-Haurwitz wave-4)
    is wired for all 4 grid types in the matrix runner — the
    ``if g in (...):`` gate at the W6 ``matrix.append`` site must
    include ``cubed_sphere`` (iter-24) AND ``latlon`` (iter-25)
    alongside the pre-existing ``icosahedral`` and ``spectral``.
    """
    src = _runner_source()
    # Find the line that gates the W6 grid-type iteration.
    pat = re.search(
        r'if g in \([^)]*\):\s*\n\s*matrix\.append\(\s*TestCase\(\s*\n'
        r'\s*"shallow_water",\s*"williamson6"',
        src,
    )
    assert pat is not None, (
        "Could not locate W6 grid-type gate in matrix runner"
    )
    gate = pat.group(0)
    for grid in ("cubed_sphere", "latlon", "icosahedral", "spectral"):
        assert f'"{grid}"' in gate, (
            f"iter-24/25 regression: W6 grid-type gate no longer "
            f"includes ``{grid}``; that grid will be skipped on W6 "
            f"cross-grid runs."
        )


def test_w6_cube_init_uses_w6_winds_geo():
    """iter-24 sentinel: cube W6 init in the matrix runner SW
    branch uses ``_w6_winds_geo(lon_edge, lat_edge, R)`` for
    analytic edge-midpoint wind init.  A regression that strips
    the W6 branch + falls through to the W2/W5 ``u0*cos(lat)``
    init would silently use wrong winds.

    Whitespace-flexible regex (iter-30): tolerates a reflow /
    reindent of the call site.
    """
    src = _runner_source()
    pat = re.search(
        r"_w6_winds_geo\(\s*cdgrid\.lon_edge_x\s*,\s*cdgrid\.lat_edge_x",
        src,
    )
    assert pat is not None, (
        "iter-24 regression: cube W6 init no longer evaluates "
        "``_w6_winds_geo(cdgrid.lon_edge_x, cdgrid.lat_edge_x, ...)``."
    )


def test_w6_latlon_init_uses_w6_winds_geo():
    """iter-25 sentinel: latlon W6 init in the matrix runner SW
    branch uses ``_w6_winds_geo`` at u-face / v-face coords.

    Whitespace-flexible regex (iter-30): tolerates a reflow /
    reindent of the call site.
    """
    src = _runner_source()
    pat = re.search(
        r"_w6_winds_geo\(\s*_lon_f_full\s*\[\s*None\s*,\s*:\s*\]",
        src,
    )
    assert pat is not None, (
        "iter-25 regression: latlon W6 init no longer evaluates "
        "``_w6_winds_geo`` at u-face coords ``_lon_f_full[None, :]``."
    )


def test_sw_cube_propagating_tests_have_hyperdiff_override():
    """iter-31/33/42 sentinel: cube SW config overrides the iter1009
    baseline with ``hyperdiff_coeff=_hyperdiff_cube(n)`` for the
    propagating-wave test gate ``test_num in (2, 5, 6)``.

    Without the override:
      - cube W5 BLOWS UP at day 14.58 (15-day; iter-33).
      - cube W6 BLOWS UP at day  9.03 (14-day; iter-31).
      - cube W2 5-day v_ll_Linf = 3.65 m/s vs 0.82 with hyperdiff
        (iter-42 measurement; 4.5x parity gain).

    Latlon W2/W5/W6 at the same durations PASS without hyperdiff.
    This sentinel catches a regression that would re-introduce any
    of the three instabilities.
    """
    src = _runner_source()
    pat = re.search(
        r"if\s+test_num\s+in\s*\(\s*2\s*,\s*5\s*,\s*6\s*\)\s*:[^}]*?"
        r"hyperdiff_coeff\s*=\s*_hyperdiff_cube\(\s*n\s*\)",
        src,
        re.DOTALL,
    )
    assert pat is not None, (
        "iter-31/33/42 regression: cube SW propagating-test branch "
        "no longer overrides ``hyperdiff_coeff=_hyperdiff_cube(n)`` "
        "for (W2, W5, W6) — cube W5/W6 full-duration will re-BLOWUP "
        "and cube W2 5-day v_ll_Linf will regress from 0.82 to "
        "3.65 m/s (latlon stable; cube parity gap reopens)."
    )


def test_sw_cube_hyperdiff_gate_matches_propagating_tests():
    """iter-37/42 sentinel: the SW cube hyperdiff gate matches all
    propagating tests (W2, W5, W6) — exactly ``test_num in (2, 5, 6)``.

    iter-37 originally restricted the gate to ``{5, 6}`` under the
    assumption that hyperdiff would break the iter-1002 W2 1-day
    sentinel.  iter-42 measurement disproved this: matrix runner W2
    with hyperdiff actually IMPROVES cube W2 5-day v_ll_Linf from
    3.65 to 0.82 m/s (4.5x parity gain), and the iter-1002 sentinel
    is unaffected because it uses its own ``hyperdiff_coeff=0`` config
    (independent of matrix runner).

    This sentinel now pins the wider gate ``{2, 5, 6}``.  Regressions
    that shrink the gate back to ``{5, 6}`` or widen it to ``{1, 2,
    5, 6}`` (CB has test_num=1 if numbered, or no test_num) would
    trip this test.
    """
    src = _runner_source()
    gate_pat = re.search(
        r"if\s+test_num\s+in\s*\(\s*([\d,\s]+)\s*\)\s*:", src,
    )
    assert gate_pat is not None, (
        "iter-31/33/42 regression: ``if test_num in (...):`` gate "
        "not found in matrix runner SW cube branch."
    )
    gate_values = {
        int(v.strip()) for v in gate_pat.group(1).split(",") if v.strip()
    }
    assert gate_values == {2, 5, 6}, (
        f"iter-42 regression: SW cube hyperdiff gate now matches "
        f"test_num in {sorted(gate_values)} — must be exactly "
        "{{2, 5, 6}}.  Shrinking to {{5, 6}} would re-open the cube "
        "W2 5-day v_ll_Linf gap (0.82 -> 3.65 m/s)."
    )


def test_pe_factory_bundle_present_in_three_cube_branches():
    """iter-22 sentinel (tightened in iter-27 post-review): the PE
    iter-18..21 factory bundle (``use_fv3_metric_aware_d_con``,
    ``d_con_top_zero_levels=2``, ``delt_max=1.0``,
    ``heat_source_del2_iters=2``) is present in EACH of the 3 PE
    cube branches: baroclinic, held_suarez, amip.

    iter-27 fix (post-review): the iter-22 implementation counted
    GLOBAL occurrences across the file (NH + PE branches +
    comments), so a regression that stripped all 4 flags from a
    single PE branch (e.g., AMIP) would still pass because NH
    branches set the same flag names.  This version walks each PE
    branch independently and asserts presence of each flag in each
    branch's actual ``PrimitiveEquationConfig(...)`` constructor
    body — not in surrounding comments.
    """
    flags = (
        "use_fv3_metric_aware_d_con=True",
        "d_con_top_zero_levels=2",
        "delt_max=1.0",
        "heat_source_del2_iters=2",
    )
    block_finders = {
        "baroclinic":  _find_pe_baroclinic_cube_config_block,
        "held_suarez": _find_pe_held_suarez_cube_config_block,
        "amip":        _find_pe_amip_cube_config_block,
    }
    for branch_name, finder in block_finders.items():
        block = finder()
        for flag in flags:
            assert flag in block, (
                f"iter-22 regression: PE {branch_name} cube config "
                f"missing ``{flag}``."
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
