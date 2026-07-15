"""new_test_dycores iter-9: AST guard for the iter-5/6/7 NH cube parity
fix in ``scripts/matrix/run_atmosphere_test_matrix.py``.

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

import ast
import re
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "matrix"
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
    """iter-31/33/42/44 sentinel: cube SW config overrides the iter1009
    baseline with ``hyperdiff_coeff=2.0 * _hyperdiff_cube(n)`` for the
    propagating/strongly-nonlinear test gate ``test_num in (2, 5, 6, 8)``.

    Without the override:
      - cube W5 BLOWS UP at day 14.58 (15-day; iter-33).
      - cube W6 BLOWS UP at day  9.03 (14-day; iter-31).
      - cube W2 5-day v_ll_Linf = 3.65 m/s vs 0.51 with iter-44's
        2x hyperdiff (vs 0.82 at iter-42's 1x); h_err 72 -> 14 m.

    Latlon W2/W5/W6 at the same durations PASS without hyperdiff.
    iter-44 raised the coefficient from 1x to 2x after probing
    (further 38% W2 gain, W5/W6 stable, 4x cube has diminishing
    returns).

    #521/#529 (colliding modons, FV3 case 8, merged in commit
    750b157ac) joined the gate: the two-soliton collision is strongly
    nonlinear over a 100-day run and needs the SAME div-damp +
    biharmonic hyperdiff backstop as the long-duration propagating-wave
    cases.  test_num 8 is therefore a LEGITIMATE member of the override
    gate, not a regression — but the override must still cover W2
    (test_num 2) so the cube W2 5-day v_ll_Linf protection is intact.
    """
    src = _runner_source()
    # fv3_faithful iter-2: the literal ``2.0`` became an env-overridable
    # factor ``LEGOESM_SW_HYPERDIFF_FACTOR`` that DEFAULTS to 2.0, so the
    # override is preserved.  Accept either the original literal or the
    # factor form; if the factor form is used, the default must stay 2.0.
    # This test verifies the override EXPRESSION (``hyperdiff_coeff =
    # <factor> * _hyperdiff_cube(n)``) sits immediately under the gate.
    #
    # #521 (2026-07-03): colliding modons (test_num 8) LEFT the shared
    # gate for a DEDICATED ``if test_num == 8`` branch (duogrid grid +
    # per-case damping — the W2/W5/W6 calibration erodes the modon
    # cores below the cube seam-noise floor).  The shared-gate anchor
    # is therefore ``(2, 5, 6)`` now, and the modon branch's OWN
    # hyperdiff backstop (default 1.0x, env knob
    # LEGOESM_SW_MODON_HYPERDIFF_FACTOR) is pinned separately below —
    # zero hyperdiff blows the 100-day modon run up at ~day 40 even
    # with duogrid, so the backstop protection is unchanged, just
    # per-case.  The canonical SET-equality check on the shared gate
    # lives in ``test_sw_cube_hyperdiff_gate_matches_propagating_tests``.
    pat = re.search(
        r"if\s+test_num\s+in\s*\(\s*2\s*,\s*5\s*,\s*6\s*\)\s*:[^}]*?"
        r"hyperdiff_coeff\s*=\s*(?:2\.0|_sw_hd_fac)\s*\*\s*"
        r"_hyperdiff_cube\(\s*n\s*\)",
        src,
        re.DOTALL,
    )
    assert pat is not None, (
        "iter-31/33/42/44 regression: cube SW propagating-test "
        "branch no longer overrides ``hyperdiff_coeff=2.0 * "
        "_hyperdiff_cube(n)`` for (W2, W5, W6) — cube W5/W6 "
        "full-duration will re-BLOWUP and cube W2 5-day v_ll_Linf "
        "will regress (latlon stable; cube parity gap reopens)."
    )
    if "_sw_hd_fac" in pat.group(0):
        assert re.search(
            r'LEGOESM_SW_HYPERDIFF_FACTOR"\s*,\s*"2\.0"', src), (
            "LEGOESM_SW_HYPERDIFF_FACTOR default must remain \"2.0\" so "
            "the cube SW propagating-test hyperdiff override is preserved "
            "when the env knob is unset."
        )
    # Modon (#521) dedicated branch: its own hyperdiff backstop must
    # survive — zero hyperdiff blows the 100-day run up at ~day 40.  The
    # branch delegates to ``_modon_hyperdiff_coeff(n)`` (extracted so the
    # env-knob wiring is runtime-testable, #753); the backstop must both be
    # CALLED in the branch AND remain the ``_m_hd... * _hyperdiff_cube(n...)``
    # biharmonic inside that helper (``_hyperdiff_cube(n ...)`` allows the
    # optional #753 ``scaling_exponent=`` kwarg).
    pat8 = re.search(
        r"if\s+test_num\s*==\s*8\s*:[^}]*?"
        r"hyperdiff_coeff\s*=\s*_modon_hyperdiff_coeff\(\s*n\s*\)",
        src,
        re.DOTALL,
    )
    assert pat8 is not None, (
        "#521 regression: the dedicated colliding-modons branch "
        "(``test_num == 8``) no longer sets ``hyperdiff_coeff="
        "_modon_hyperdiff_coeff(n)`` — the 100-day cube run re-BLOWs-UP "
        "at ~day 40 without its biharmonic backstop."
    )
    pat8_helper = re.search(
        r"def\s+_modon_hyperdiff_coeff\(.*?"
        r"return\s+\w+\s*\*\s*_hyperdiff_cube\(\s*n\b[^)]*\)",
        src,
        re.DOTALL,
    )
    assert pat8_helper is not None, (
        "#521 regression: ``_modon_hyperdiff_coeff`` no longer returns "
        "``<factor> * _hyperdiff_cube(n ...)`` — the modon backstop was "
        "zeroed/removed inside the helper; the 100-day cube run BLOWs-UP."
    )
    # #800: the env-knob defaults come from the shared ``MODON_*`` constants
    # (single source of truth with the run_colliding_modons.py driver) — pin
    # that the matrix reads them via ``str(MODON_HYPERDIFF_{FACTOR,SCALING})``
    # AND that those canonical constants still hold their validated values.
    assert re.search(
        r'LEGOESM_SW_MODON_HYPERDIFF_FACTOR"\s*,\s*str\(MODON_HYPERDIFF_FACTOR\)',
        src), (
        "LEGOESM_SW_MODON_HYPERDIFF_FACTOR default must be str(MODON_HYPERDIFF_"
        "FACTOR) (the shared #800 source of truth), not a re-hardcoded literal."
    )
    assert re.search(
        r'LEGOESM_SW_MODON_HYPERDIFF_SCALING"\s*,\s*str\(MODON_HYPERDIFF_SCALING\)',
        src), (
        "LEGOESM_SW_MODON_HYPERDIFF_SCALING default must be str(MODON_HYPERDIFF_"
        "SCALING) (the shared #800 source of truth), not a re-hardcoded literal."
    )
    assert re.search(
        r'LEGOESM_SW_MODON_DIV_DAMP_FACTOR"\s*,\s*str\(MODON_DIV_DAMP_FACTOR\)',
        src), (
        "LEGOESM_SW_MODON_DIV_DAMP_FACTOR default must be str(MODON_DIV_DAMP_"
        "FACTOR) (the shared #800 source of truth), not a re-hardcoded literal."
    )
    assert re.search(
        r'LEGOESM_SW_MODON_DAMP_V"\s*,\s*str\(MODON_DAMP_V\)', src), (
        "LEGOESM_SW_MODON_DAMP_V default must be str(MODON_DAMP_V) (the shared "
        "#800 source of truth), not a re-hardcoded literal."
    )
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        MODON_DAMP_V,
        MODON_DIV_DAMP_FACTOR,
        MODON_HYPERDIFF_FACTOR,
        MODON_HYPERDIFF_SCALING,
    )
    # The canonical values: 1.0x biharmonic (the #521 sweep optimum: 0.5x erupts
    # at the collision transient, 0x blows up ~day 40 even with duogrid) + the
    # (ref/n)^2 law (scaling 2, the #753 item-1 default: C96 erupts under ^4 but
    # is stable under ^2, validated 100 days at C36/C48/C96; ^4 is now the opt-in
    # for pre-#753 byte-identical behaviour) + the tuned div_damp/damp_v.
    assert MODON_HYPERDIFF_FACTOR == 1.0
    assert MODON_HYPERDIFF_SCALING == 2
    assert MODON_DIV_DAMP_FACTOR == 8.0
    assert MODON_DAMP_V == 0.010, (
        "MODON_* modon config drifted — this is the #800 desync class; the "
        "matrix and run_colliding_modons.py both read these, keep them the "
        "2026-07-03 #521 sweep optimum."
    )


def test_sw_cube_hyperdiff_gate_matches_propagating_tests():
    """iter-37/42/45 sentinel: the SW cube hyperdiff gate matches all
    propagating / strongly-nonlinear cases (W2, W5, W6, colliding
    modons) — exactly ``test_num in (2, 5, 6, 8)``.

    iter-37 originally restricted the gate to ``{5, 6}`` under the
    assumption that hyperdiff would break the iter-1002 W2 1-day
    sentinel.  iter-42 measurement disproved this: matrix runner W2
    with hyperdiff actually IMPROVES cube W2 5-day v_ll_Linf from
    3.65 to 0.82 m/s (4.5x parity gain), and the W2 calibration
    sentinel is unaffected because it uses its own
    ``hyperdiff_coeff=0`` config (independent of matrix runner).

    iter-42 widened the gate to ``{2, 5, 6}``.  #521/#529 (colliding
    modons, FV3 case 8) then ADDED ``8`` in merged commit 750b157ac:
    the two-soliton collision is strongly nonlinear over a 100-day run
    and the matrix-runner branch documents that it needs the SAME
    div-damp + biharmonic hyperdiff backstop as the long-duration
    propagating-wave cases.  This sentinel therefore pins the current
    gate ``{2, 5, 6, 8}``.  The protection it guards is unchanged: the
    gate MUST still contain ``2`` (W2) and ``5``/``6`` (W5/W6) so the
    cube W2 5-day v_ll_Linf gap stays closed and W5/W6 don't re-blow-up.

    iter-45 review-driven fix: anchor the regex to the iter1009-helper
    call immediately following the gate so a future unrelated
    ``if test_num in (...):`` elsewhere in the file (e.g., line 2318
    spectral filter at ``(5, 6)``, line 4068 NH ``(11, 12)``) can't
    silently pin the wrong gate.
    """
    src = _runner_source()
    # Anchor to the iter1009_dual_target_config call immediately
    # following the gate (the gate is followed within ~300 chars by
    # ``config = iter1009_dual_target_config(...)`` with hyperdiff
    # kwarg).  This rules out the spectral filter conditional + the
    # NH test_num gate elsewhere in the file.
    gate_pat = re.search(
        r"if\s+test_num\s+in\s*\(\s*([\d,\s]+)\s*\)\s*:"
        r"[\s\S]{0,4000}?"
        r"config\s*=\s*iter1009_dual_target_config\s*\(\s*n\s*,",
        src,
    )
    assert gate_pat is not None, (
        "iter-31/33/42 regression: ``if test_num in (...):`` gate "
        "immediately followed by ``iter1009_dual_target_config(n, ...)``"
        " call not found in matrix runner SW cube branch."
    )
    gate_values = {
        int(v.strip()) for v in gate_pat.group(1).split(",") if v.strip()
    }
    assert gate_values == {2, 5, 6}, (
        f"iter-42 regression: SW cube hyperdiff gate now matches "
        f"test_num in {sorted(gate_values)} — must be exactly "
        "{{2, 5, 6}} (W2, W5, W6).  Dropping ``2`` re-opens the cube "
        "W2 5-day v_ll_Linf gap (0.51 -> 3.65 m/s); dropping "
        "``5``/``6`` re-blows-up cube W5/W6.  Colliding modons "
        "(test_num 8) moved to a DEDICATED ``if test_num == 8`` branch "
        "on 2026-07-03 (#521: duogrid grid + per-case damping) — its "
        "own hyperdiff backstop is pinned by the ``pat8`` check in "
        "``test_sw_cube_propagating_tests_have_hyperdiff_override``.  "
        "If a new SW cube case is added that legitimately needs the "
        "shared override, extend this set AND the override regex there."
    )
    # The modon branch must precede the shared gate (it is checked as
    # ``if test_num == 8: ... elif test_num in (2, 5, 6)``) — assert the
    # dedicated branch exists so the case can't silently fall back to
    # the plain iter1009 defaults with NO hyperdiff at all.
    assert re.search(
        r"if\s+test_num\s*==\s*8\s*:"
        r"[\s\S]{0,4000}?"
        r"config\s*=\s*iter1009_dual_target_config\s*\(\s*n\s*,",
        src,
    ), (
        "#521 regression: the dedicated colliding-modons config branch "
        "(``if test_num == 8`` -> ``iter1009_dual_target_config(n, ...)``)"
        " is gone — the case would fall through to the no-hyperdiff "
        "default and the 100-day cube run re-BLOWs-UP at ~day 40."
    )


def _parse_module(path: Path) -> ast.Module:
    """Parse a Python source file into an AST module (helper for the
    AST-based source tripwires below).  Keeps the guards robust against
    comment/docstring text that merely *mentions* the pattern being
    pinned (a pure ``str in src`` check is vacuous against that)."""
    return ast.parse(path.read_text(), filename=str(path))


def _find_function_def(
    module: ast.Module, name: str
) -> ast.FunctionDef | None:
    """Return the first top-level ``def <name>`` in ``module`` (or None).

    Recurses into class bodies too, so a method of the same name is also
    found; the iter1009 helper is module-level and the W2 fixture is
    module-level, so the top-level walk suffices, but recursion makes
    the helper reusable.
    """
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _arg_default(
    func: ast.FunctionDef, arg_name: str
) -> ast.expr | None:
    """Return the AST default expression for keyword/positional argument
    ``arg_name`` of ``func``, or None if the arg has no default."""
    args = func.args
    # Positional-or-keyword args: defaults align to the TAIL of the list.
    posargs = args.posonlyargs + args.args
    n_def = len(args.defaults)
    defaulted = posargs[len(posargs) - n_def:] if n_def else []
    for a, d in zip(defaulted, args.defaults):
        if a.arg == arg_name:
            return d
    # Keyword-only args have their own (possibly-None) defaults list.
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if a.arg == arg_name:
            return d
    return None


def _is_zero_constant(node: ast.expr | None) -> bool:
    """True iff ``node`` is a numeric literal equal to 0 — accepts the
    bare constants ``0`` / ``0.0`` / ``0e0`` AND the unary-signed forms
    ``-0.0`` / ``+0.0`` (which Python parses as ``UnaryOp(USub|UAdd,
    Constant(0.0))``, not a single ``Constant``).  Rejects booleans
    (``True``/``False`` are ``int`` subclasses == 1/0)."""
    # Unwrap a single unary +/- in front of a zero literal.
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub)
    ):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and float(node.value) == 0.0
    )


def _call_target_name(call: ast.Call) -> str | None:
    """Return the *called name* of an ``ast.Call`` for both bare-name
    (``f(...)`` -> ``"f"``) and attribute (``mod.f(...)`` /
    ``pkg.mod.f(...)`` -> ``"f"``) call forms.

    Matching on the trailing attribute (not the full dotted path) keeps
    these source tripwires robust to benign import-style refactors —
    ``from m import f; f()`` vs ``import m; m.f()`` vs
    ``import m as a; a.f()`` all resolve to ``"f"`` — which the
    round-2 review flagged as a false-positive risk when only
    ``ast.Name`` was matched.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_has_kwargs_unpack(call: ast.Call) -> bool:
    """True iff ``call`` uses ``**kwargs`` unpacking (a keyword whose
    ``arg`` is None).  Such a call can smuggle a nonzero
    ``hyperdiff_coeff`` past a per-keyword inspection
    (``iter1009_dual_target_config(**{"hyperdiff_coeff": 2.0})``), so
    the W2-test guard treats it as un-verifiable and fails loudly."""
    return any(kw.arg is None for kw in call.keywords)


def test_w2_calibration_baseline_independent_from_matrix_hyperdiff():
    """iter-45 review-driven sentinel (retargeted): pin the independence
    of the W2 calibration baseline from the matrix-runner SW cube
    hyperdiff override gate.

    iter-42 widened the matrix-runner hyperdiff gate (adding W2),
    relying on the invariant that the cube W2 calibration reference
    uses its OWN ``hyperdiff_coeff=0`` config — independent of the
    matrix-runner propagating-wave override (``2.0 * _hyperdiff_cube(n)``
    on ``test_num in (2, 5, 6, 8)``).  Without that independence, the
    iter-1030 dual-target calibration measured by codex
    iter-1009/1021/1030 (cube W2 1-day v_ll_Linf <= 0.119 m/s — the
    cube-imprint / W2 parity protection) would silently shift with the
    matrix-runner damping knobs.

    ORIGINAL form anchored ``tests/test_iter1002_w2_target_met.py`` and
    its private ``_make_iter1009_config(N)`` helper.  That file was
    curated out of the live suite by commit 096349afd ("curate 476
    ralph-loop iter tests -> 57 survivors") and archived (untracked)
    under ``scripts/tmp/dycore_iter_archive/``.  The invariant DID NOT
    go away — it moved to two LIVE, tracked locations, which this
    retargeted sentinel now pins instead of the deleted file.

    Codex-review hardening (round 1): the earlier retarget used
    ``str in src`` / signature-only regex, which can pass on stale
    docstring text or while the helper body ignores the zero default.
    This version parses both files with ``ast`` and asserts EXECUTABLE
    structure, so a docstring/comment that merely mentions the pattern
    cannot satisfy the guard:

      1. ``shallow_water_fv3_cdgrid.iter1009_dual_target_config`` — the
         single source of truth for the iter-1030 calibration — must
         (a) DEFAULT its ``hyperdiff_coeff`` parameter to a zero literal,
         AND (b) actually PASS that parameter (by name, not a recomputed
         nonzero literal) into the ``CDGridShallowWaterConfig(...)`` it
         returns.  (b) closes the "default stays 0.0 but the body hard-
         codes ``hyperdiff_coeff=_hyperdiff_cube(n)``" hole.

      2. The curation-survivor W2 regression test
         ``tests/atmosphere/dycore/regression/test_iter1032_dual_target_full_matrix.py``
         — successor to the deleted iter-1002 sentinel — must (a) call
         ``iter1009_dual_target_config`` with NO nonzero
         ``hyperdiff_coeff`` keyword (so it consumes the zero default; a
         nonzero override would decouple it from the measured
         calibration), AND (b) contain an executable
         ``assert <v_ll_Linf…> <= 0.119`` (the actual cube-imprint / W2
         parity threshold, anchored to an ``assert`` that references a
         ``v_ll``-named operand), not merely the string ``0.119`` in a
         docstring or a ``<= 0.119`` on some unrelated quantity.

    A static guard cannot prove the hyperdiff=0 config flows into that
    assertion (data-flow); that end-to-end guarantee is owned by the
    LIVE regression test, which runs the real dycore and FAILS if W2
    v_ll_Linf actually exceeds 0.119.  This sentinel pins the structural
    preconditions so the live test cannot be silently neutered.

    If a future refactor flips the helper default to nonzero, routes a
    nonzero hyperdiff into either consumer, or weakens the 0.119
    comparison, the cube W2 calibration is no longer measured at
    hyperdiff=0 and these structural assertions FIRE.
    """
    repo = Path(__file__).resolve().parents[1]

    # --- (1) Canonical preset: zero hyperdiff default AND pass-through ---
    pkg_path = (
        repo
        / "packages"
        / "atmosphere"
        / "legoesm"
        / "atmosphere"
        / "dynamics"
        / "gcm"
        / "shallow_water_fv3_cdgrid.py"
    )
    pkg_mod = _parse_module(pkg_path)
    helper = _find_function_def(pkg_mod, "iter1009_dual_target_config")
    assert helper is not None, (
        "iter-45 regression: ``iter1009_dual_target_config`` def not "
        "found in shallow_water_fv3_cdgrid.py — the canonical W2 "
        "calibration preset is the source of truth for the cube W2 "
        "hyperdiff=0 baseline; relocate this sentinel if the helper "
        "moved/renamed."
    )
    # (1a) default of hyperdiff_coeff is a zero literal.
    hd_default = _arg_default(helper, "hyperdiff_coeff")
    assert _is_zero_constant(hd_default), (
        "iter-45/#529 regression: ``iter1009_dual_target_config`` no "
        "longer DEFAULTS ``hyperdiff_coeff`` to a zero literal (parsed "
        f"default AST = {ast.dump(hd_default) if hd_default else None}). "
        "The cube W2 1-day calibration (v_ll_Linf <= 0.119 m/s) was "
        "measured at hyperdiff=0; a nonzero default would silently fold "
        "the matrix-runner propagating-wave override into the W2 baseline."
    )
    # (1b) the body must construct ``CDGridShallowWaterConfig`` passing
    # ``hyperdiff_coeff=hyperdiff_coeff`` (the PARAMETER, by name) — not
    # a recomputed nonzero expression that ignores the zero default.
    #
    # Round-2 review hardening: the existential check ("SOME call passes
    # through") was insufficient — a dead ``if False: CDGrid...(
    # hyperdiff_coeff=hyperdiff_coeff)`` branch alongside a live
    # ``return CDGrid...(hyperdiff_coeff=_hyperdiff_cube(n))`` would have
    # passed.  We now require BOTH:
    #   (i)  at least one ``CDGridShallowWaterConfig(...)`` call passes
    #        ``hyperdiff_coeff=hyperdiff_coeff`` through by name; AND
    #   (ii) NO ``CDGridShallowWaterConfig(...)`` call anywhere in the
    #        helper supplies a NONZERO ``hyperdiff_coeff`` (literal or
    #        otherwise non-pass-through) — so a hardcoded-nonzero return
    #        fails even if a dead pass-through branch also exists.
    # Attribute-form constructors (``mod.CDGridShallowWaterConfig(...)``)
    # are matched via ``_call_target_name`` so an import-style refactor
    # does not false-positive.
    cfg_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and _call_target_name(node) == "CDGridShallowWaterConfig"
    ]
    assert cfg_calls, (
        "iter-45 regression: ``iter1009_dual_target_config`` body no "
        "longer constructs ``CDGridShallowWaterConfig(...)`` — cannot "
        "verify the hyperdiff pass-through."
    )
    passes_param_through = False
    for call in cfg_calls:
        # A ``**kwargs`` unpack could smuggle a nonzero hyperdiff past
        # the per-keyword scan; treat it as un-verifiable -> fail.
        assert not _call_has_kwargs_unpack(call), (
            "iter-45 regression: ``iter1009_dual_target_config`` "
            "constructs ``CDGridShallowWaterConfig(**kwargs)`` with "
            "dictionary unpacking, which can hide a nonzero "
            "``hyperdiff_coeff``.  Pass ``hyperdiff_coeff=hyperdiff_coeff`` "
            "explicitly so this structural guard can verify the cube W2 "
            "hyperdiff=0 baseline."
        )
        for kw in call.keywords:
            if kw.arg != "hyperdiff_coeff":
                continue
            is_passthrough = (
                isinstance(kw.value, ast.Name)
                and kw.value.id == "hyperdiff_coeff"
            )
            if is_passthrough:
                passes_param_through = True
            else:
                # Any non-pass-through hyperdiff_coeff value (a literal,
                # a recomputed expression, etc.) in ANY constructor call
                # breaks the baseline unless it is an explicit zero.
                assert _is_zero_constant(kw.value), (
                    "iter-45 regression: ``iter1009_dual_target_config`` "
                    "passes a non-pass-through, nonzero ``hyperdiff_coeff`` "
                    f"into ``CDGridShallowWaterConfig`` (AST = "
                    f"{ast.dump(kw.value)}).  A hardcoded/recomputed "
                    "nonzero hyperdiff (e.g. ``_hyperdiff_cube(n)``) keeps "
                    "the 0.0 default yet breaks the cube W2 hyperdiff=0 "
                    "baseline — exactly the case this structural check "
                    "guards (even if a dead pass-through branch also "
                    "exists)."
                )
    assert passes_param_through, (
        "iter-45 regression: ``iter1009_dual_target_config`` no longer "
        "passes ``hyperdiff_coeff=hyperdiff_coeff`` (the parameter) into "
        "``CDGridShallowWaterConfig(...)``.  The returned config must "
        "thread the zero-default parameter through so the cube W2 "
        "hyperdiff=0 baseline holds."
    )

    # --- (2) Live W2 regression test: zero-default call + 0.119 target ---
    w2_test_path = (
        repo
        / "tests"
        / "atmosphere"
        / "dycore"
        / "regression"
        / "test_iter1032_dual_target_full_matrix.py"
    )
    assert w2_test_path.exists(), (
        "iter-45 regression: the live W2 calibration regression test "
        "``tests/atmosphere/dycore/regression/"
        "test_iter1032_dual_target_full_matrix.py`` (curation-survivor "
        "successor to the deleted iter-1002 sentinel) is missing.  It is "
        "the live owner of the cube W2 ``v_ll_Linf <= 0.119`` invariant; "
        "restore it or relocate this sentinel to its replacement."
    )
    w2_mod = _parse_module(w2_test_path)
    # (2a) at least one EXECUTABLE call to iter1009_dual_target_config,
    # and EVERY such call must omit a nonzero hyperdiff_coeff override
    # (it may omit the kwarg entirely -> zero default, or pass a zero
    # literal explicitly).  A nonzero override decouples the test from
    # the measured hyperdiff=0 calibration.  ``_call_target_name`` matches
    # both bare-name and ``module.helper`` attribute call forms so an
    # import-style refactor does not false-positive (round-2 review).
    helper_calls = [
        node
        for node in ast.walk(w2_mod)
        if isinstance(node, ast.Call)
        and _call_target_name(node) == "iter1009_dual_target_config"
    ]
    assert helper_calls, (
        "iter-45 regression: the live W2 regression test no longer "
        "CALLS ``iter1009_dual_target_config(...)`` (the canonical "
        "hyperdiff=0 preset) in executable code.  The cube W2 "
        "calibration is no longer measured at the iter-1030 baseline.  "
        "(A docstring mention does not count — this is an AST check.)"
    )
    for call in helper_calls:
        # ``**kwargs`` unpacking could smuggle a nonzero hyperdiff past
        # the per-keyword scan -> treat as un-verifiable and fail.
        assert not _call_has_kwargs_unpack(call), (
            "iter-45 regression: the live W2 regression test calls "
            "``iter1009_dual_target_config(**kwargs)`` with dictionary "
            "unpacking, which can hide a nonzero ``hyperdiff_coeff`` "
            "override.  Pass arguments explicitly so this guard can "
            "verify the cube W2 hyperdiff=0 baseline."
        )
        for kw in call.keywords:
            if kw.arg == "hyperdiff_coeff":
                assert _is_zero_constant(kw.value), (
                    "iter-45 regression: the live W2 regression test now "
                    "passes a nonzero ``hyperdiff_coeff`` override to "
                    "``iter1009_dual_target_config`` (AST = "
                    f"{ast.dump(kw.value)}).  The W2 1-day calibration "
                    "was measured at hyperdiff=0; any nonzero value "
                    "silently shifts the v_ll_Linf target away from the "
                    "cube W2 parity baseline."
                )
    # (2b) an EXECUTABLE ``assert <name…v_ll_Linf…> <= 0.119`` must exist
    # — the actual cube-imprint / W2 parity threshold.  Round-2 review
    # hardening: requiring the ``<= 0.119`` compare to live INSIDE an
    # ``assert`` AND reference a ``v_ll_Linf``-named operand rules out
    # (i) a stale docstring ``0.119``, (ii) a dead/unrelated ``<= 0.119``
    # compare that never gates the test, and (iii) a ``0.119`` threshold
    # applied to some unrelated quantity.  We scan only ``ast.Assert``
    # test expressions, and require a ``Name``/``Attribute`` whose
    # identifier contains ``v_ll`` on either side of the ``<=``.
    #
    # NOTE (inherent tripwire limit, surfaced in round-2 review): a
    # static guard cannot fully prove the config returned by the
    # hyperdiff=0 helper call is the SAME object fed into this assertion
    # (data-flow).  That end-to-end guarantee is owned by the LIVE
    # regression test itself — ``test_iter1032_dual_target_full_matrix``
    # runs the real dycore and FAILS if W2 v_ll_Linf actually exceeds
    # 0.119.  This sentinel pins the structural preconditions so the live
    # test cannot be silently neutered (helper deleted/renamed, default
    # flipped nonzero, override injected, threshold loosened).
    def _refs_v_ll(expr: ast.expr) -> bool:
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name) and "v_ll" in sub.id:
                return True
            if isinstance(sub, ast.Attribute) and "v_ll" in sub.attr:
                return True
        return False

    has_target_assert = False
    for node in ast.walk(w2_mod):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not any(isinstance(op, ast.LtE) for op in test.ops):
            continue
        # Operands of a Compare: left + comparators.
        operands = [test.left, *test.comparators]
        has_threshold = any(
            isinstance(o, ast.Constant)
            and isinstance(o.value, (int, float))
            and not isinstance(o.value, bool)
            and abs(float(o.value) - 0.119) < 1e-12
            for o in operands
        )
        has_v_ll = any(_refs_v_ll(o) for o in operands)
        if has_threshold and has_v_ll:
            has_target_assert = True
    assert has_target_assert, (
        "iter-45 regression: the live W2 regression test no longer "
        "contains an executable ``assert <v_ll_Linf…> <= 0.119`` "
        "comparison.  That threshold IS the cube-imprint / W2 parity "
        "protection (iter-1030 measured ~0.1138); do not weaken, raise, "
        "rename the operand away from ``v_ll``, or remove it.  (A "
        "docstring mention of 0.119, or a ``<= 0.119`` on some unrelated "
        "quantity, does not count — this is an AST check on the assert's "
        "comparison node.)"
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


def test_iter61_cb_latlon_has_anchored_mass_fixer():
    """new_test_dycores iter-61 sentinel: the latlon CB matrix branch
    must apply the cube-style anchored mass fixer (clip-negatives +
    rescale-positives to ``_mass_target_iter61``) at the tail of its
    ``step_fn``.

    Pre-iter-61 the latlon CB step was an intentional raw-FV benchmark
    with no mass correction; the iter-29 matrix-wide tolerance tighten
    (1e-2 -> 1e-4) silently turned it into a persistent FAIL because
    the raw 12-day drift sits at 5.35e-4.  iter-61 imported the cube
    CB fixer (transport_step's clip+rescale logic) and applied it to
    the latlon step_fn, bringing all 4 grids onto the same
    mass-conservation footing (drifts <1e-7) and apples-to-apples
    error norms.

    A refactor that drops the fixer or breaks the rescale call would
    re-introduce a latlon CB FAIL.  We anchor on the constant name
    ``_mass_target_iter61`` (specifically chosen with the iter-61
    tag so it isn't accidentally renamed) and the ``jnp.maximum`` /
    ``mass_target / max(mass_pos, 1.0)`` rescale pattern.
    Whitespace-tolerant.
    """
    src = _runner_source()
    assert "_mass_target_iter61" in src, (
        "iter-61 regression: latlon CB step_fn no longer references "
        "``_mass_target_iter61``.  Re-introduce the anchored mass "
        "fixer (clip negatives, rescale positives) on the latlon CB "
        "step_fn so the 12-day mass drift stays under the iter-29 "
        "1e-4 matrix tolerance."
    )
    assert re.search(
        r"_mass_target_iter61\s*/\s*jnp\.maximum\(\s*mass_pos\s*,\s*1\.0\s*\)",
        src,
    ), (
        "iter-61 regression: latlon CB rescale pattern "
        "``_mass_target_iter61 / jnp.maximum(mass_pos, 1.0)`` missing. "
        "The rescale is what brings 12-day drift from 5e-4 to 2e-8."
    )


def test_iter59_cb_cube_uses_n_sub_substepping():
    """iter-59 sentinel: matrix runner CB cube branch uses N=6 temporal
    substepping of ``transport_step`` inside the per-outer-step
    ``step_fn`` (dt_outer=1800s split into 6 sub-steps of dt_sub=300s).

    Probe ``_probe_iter59_substep.py`` measured cube CB 12-day L2 0.931
    (n_sub=1) -> 0.865 (n_sub=6), a 7.0 % improvement on top of iter-58.
    Root cause: cube CB previously did single-stage forward-Euler PPM
    transport (O(dt) phase error) while latlon CB uses SSP-RK3
    (O(dt^3)).  At outer dt=1800s temporal-truncation was contributing
    ~7 % to the 12-day error.

    A refactor that drops the substep scan (or reduces n_sub below 4)
    would silently lose the iter-59 improvement.  We anchor on the
    integer-literal assignment to the dispatch-local
    ``_CB_CUBE_N_SUB`` constant, and on the ``jax.lax.scan`` call that
    consumes it.  Whitespace-tolerant so trivial reformatting does
    not fire.
    """
    src = _runner_source()
    # Anchor 1: a numeric assignment of the constant within the
    # cube CB branch.  Allow integer >=4 to give a guardrail without
    # over-pinning (probe shows n_sub=4 still beats n_sub=1 by 5.9 %).
    m = re.search(r"_CB_CUBE_N_SUB\s*=\s*(\d+)", src)
    assert m is not None, (
        "iter-59 regression: ``_CB_CUBE_N_SUB`` constant no longer "
        "present in matrix runner.  Re-introduce N=6 temporal "
        "substepping in the CB cube step_fn."
    )
    n_sub = int(m.group(1))
    assert n_sub >= 4, (
        f"iter-59 regression: ``_CB_CUBE_N_SUB`` reduced to {n_sub}; "
        "probe shows n_sub<4 loses >2 % of the iter-59 gain.  Pin >=4 "
        "(matrix uses 6)."
    )
    # Anchor 2: the substep scan body must call transport_step.
    assert re.search(
        r"jax\.lax\.scan\(\s*_body\s*,\s*s\.h\s*,\s*None\s*,\s*"
        r"length\s*=\s*_CB_CUBE_N_SUB\s*\)",
        src,
    ), (
        "iter-59 regression: matrix CB cube step_fn no longer "
        "calls ``jax.lax.scan(_body, s.h, None, length=_CB_CUBE_N_SUB)``. "
        "The substep scan is what reduces the temporal-truncation "
        "error from O(dt) toward the spatial-limiter plateau."
    )


def test_iter66_cb_ico_uses_additive_correction():
    """iter-66 negative-result sentinel: matrix runner CB ico branch
    MUST keep its per-step additive uniform mass correction; it must
    NOT be switched to the multiplicative-rescale-to-initial-mass
    scheme used by cube (iter-58) and latlon (iter-61).

    iter-66 attempted that switch in pursuit of cross-grid fixer
    consistency.  Mass drift improved 580× (1.58e-6 → 2.71e-9) but
    Linf REGRESSED 5× (0.561 → 2.83) and L2 +24 % (0.620 → 0.772).
    Root cause: ico mesh is heterogeneous (12 pentagons alongside
    hexagons; cell-area ratio ~83 %); multiplicative rescale of
    clipped-positive cells concentrates mass into the smaller
    pentagons producing peak overshoot.  Reverted.

    This sentinel guards against future iter-66-like attempts.  We
    anchor on the additive-correction pattern:
    ``correction = (mass_old - mass_new) / total_area`` followed by
    ``h_new + correction`` in the ico CB step_fn.
    """
    src = _runner_source()
    # Anchor: the additive correction expression.  Whitespace-tolerant.
    assert re.search(
        r"correction\s*=\s*\(\s*mass_old\s*-\s*mass_new\s*\)\s*/\s*total_area",
        src,
    ), (
        "iter-66 regression: matrix CB ico branch no longer uses the "
        "per-step ADDITIVE correction ``(mass_old - mass_new) / "
        "total_area`` + ``h_new + correction``.  iter-66 NEGATIVE "
        "RESULT showed multiplicative-rescale on the heterogeneous "
        "(hex+pent) ico mesh regresses Linf 5x.  Keep the additive "
        "uniform correction — it is the natural choice on an "
        "unstructured mesh.  See matrix-runner ico CB branch comment "
        "for the full iter-66 lesson."
    )
    # Anchor 2: the ``h_new = h_new + correction`` line that applies
    # the additive correction.  Distinct from the multiplicative
    # ``h_pos * scale`` pattern cube/latlon use.
    assert re.search(
        r"h_new\s*=\s*h_new\s*\+\s*correction",
        src,
    ), (
        "iter-66 regression: matrix CB ico branch is missing the "
        "``h_new = h_new + correction`` additive fixer step.  See "
        "iter-66 lesson in matrix-runner comment."
    )
    # Anti-anchor: the iter-66-attempted multiplicative pattern must
    # NOT appear within the ico CB branch.  Locate the branch by
    # the unique ``thickness_flux`` import + scan only its window
    # (next ~3000 chars).  We look for the iter-66-named constant
    # ``_mass_target_iter66`` which is the smoking-gun of the
    # multiplicative attempt; if it appears in the ico CB branch
    # someone has resurrected the regressed scheme.
    ico_branch_start = src.find("from legoesm.core.operators_voronoi import")
    if ico_branch_start == -1:
        return  # No ico CB branch present (unlikely).
    ico_window = src[ico_branch_start:ico_branch_start + 5000]
    assert "_mass_target_iter66" not in ico_window, (
        "iter-66 regression: matrix CB ico branch contains "
        "``_mass_target_iter66`` — the iter-66-attempted "
        "multiplicative-rescale constant.  This was REVERTED "
        "because it regressed Linf 5x on the heterogeneous ico "
        "mesh.  See iter-66 lesson in matrix-runner comment."
    )


def test_iter102_tc2_cube_blowup_warning_present():
    """iter-105 sentinel: the iter-102 critical-finding warning comment
    must remain in the matrix runner TC2 cube branch.

    iter-102 discovered that NH TC2 cube blows up at day 0.13 (step
    8500) of the full 6-hour run, despite the iter-5/6/7/12..17 NH
    bundle which had measured |w|=0.32 m/s at quick-mode 5 minutes.
    iter-104 added an inline WARNING block citing the day-0.13 BLOWUP
    + 4 hypothesis-driven probe directions for iter-104+ investigation.

    This sentinel ensures the warning is preserved across future
    refactors so any maintainer touching the TC2 cube config sees the
    known full-mode issue and doesn't waste time re-discovering it.
    """
    src = _runner_source()
    # Anchor: a distinctive phrase from the iter-102 warning that is
    # unlikely to appear elsewhere in the matrix runner.
    assert "TC2 cube BLOWS UP at" in src, (
        "iter-105 regression: matrix CB cube branch no longer contains "
        "the iter-102 critical-finding WARNING comment "
        "(``TC2 cube BLOWS UP at day 0.13...``).  Re-add the warning "
        "block so future maintainers see the known full-mode blowup "
        "issue + the 4 hypothesis-driven probe directions documented "
        "in new_test_dycores.md iter-102/103."
    )
    # Anchor 2: the hypothesis list header.
    assert "Hypotheses (probe iter-104+" in src, (
        "iter-105 regression: matrix CB cube branch is missing the "
        "iter-104 hypothesis list.  The 4 probe directions "
        "(n_acoustic_substeps, hyperdiff, acoustic_off_centering, FV3 "
        "oracle) need to remain visible to future investigators."
    )


def test_iter118_timeseries_csv_excludes_private_keys():
    """iter-119 sentinel: `_save_timeseries_csv` + `_save_timeseries_plot`
    must exclude private-prefix (`_*`) keys from the csv/plot columns.

    iter-118 discovered that pre-fix the writer included `_blowup_info`
    (a dict value added on FAIL by `_run_timeloop`) in the column keys.
    Indexing `diag["_blowup_info"][i]` crashed silently after writing
    the header, leaving an empty csv that blocked post-mortem
    investigation of the iter-102 TC2 cube full-mode BLOWUP.

    The fix added `and not k.startswith("_")` to BOTH the csv writer
    and the plot writer's key-filter list comprehensions.  This
    sentinel ensures both filters remain in place.
    """
    src = _runner_source()
    # Anchor: distinctive iter-118 comment + 2x ``startswith("_")`` filter.
    assert "iter-118: exclude private-prefix keys" in src, (
        "iter-119 regression: `_save_timeseries_plot` no longer carries "
        "the iter-118 comment explaining the private-key filter.  Either "
        "the comment was deleted or the function moved — verify the "
        "private-key filter is still applied to BOTH "
        "`_save_timeseries_csv` and `_save_timeseries_plot`."
    )
    # Count the ``not k.startswith("_")`` filter occurrences — should be
    # at least 2 (csv + plot).
    n_filter = src.count('not k.startswith("_")')
    assert n_filter >= 2, (
        f"iter-119 regression: matrix runner has only {n_filter} "
        "occurrences of the private-key filter `not k.startswith(\"_\")`.  "
        "Expected at least 2 (`_save_timeseries_csv` + "
        "`_save_timeseries_plot`).  Either filter was removed or the "
        "writers were refactored without preserving the iter-118 fix.  "
        "Pre-iter-118 the missing filter caused FAILed runs to write an "
        "empty header-only csv, blocking blowup post-mortem."
    )


def test_iter108_tc3_cube_caution_present():
    """iter-112 sentinel: the iter-108 CAUTION comment must remain in
    the matrix runner TC3 cube branch.

    iter-108 added a pre-emptive caution noting that TC3 cube was
    cached only at quick-mode 4 min (per iter-89 audit), and that the
    same TC2-style full-mode BLOWUP at day 0.13 (iter-102 finding)
    could plausibly recur for TC3.  The caution lists the same 4
    probe directions (acoustic substeps, hyperdiff, off-centering,
    FV3 oracle).

    Once NH matrix refresh completes and TC3 cube full-mode result is
    known, this sentinel can be tightened (or removed if TC3 cube is
    confirmed stable at full duration).
    """
    src = _runner_source()
    # Anchor on the iter-108 CAUTION header.
    assert "iter-108 CAUTION" in src, (
        "iter-112 regression: matrix CB cube branch (TC3) no longer "
        "contains the iter-108 CAUTION comment.  Re-add the caution "
        "block so future maintainers see the pre-emptive note about "
        "TC3 cube full-mode behaviour being TBD."
    )
