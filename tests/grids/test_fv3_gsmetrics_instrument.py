"""Instrument math for the FV3 gridstruct-metric probes.

Codex retro-review (2026-08-11) found the per-cell parity fold
min(|o-p|, |o+p|) in ``compare_gs_metrics.py`` accepted ANY spatially
varying sign field (checkerboards, side-specific flips).  These tests
lock the replacement: a GLOBAL per-face sign that makes exactly those
classes visible, and the coherent boundary-metric perturbation whose
derived reciprocals must keep their builder invariants.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "validate" \
    / "fv3_native"
sys.path.insert(0, str(SCRIPTS))

from compare_gs_metrics import (  # noqa: E402
    resolve_sign,
    score,
    sentinel_mask,
    region_masks,
)


def _field(seed=0, m=12):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.9, size=(m, m))


def test_global_flip_resolves_to_minus_one_with_zero_residual():
    o = _field(1)
    s, e_c, e_o, det = resolve_sign(o, -o, np.zeros_like(o, dtype=bool))
    assert s == -1.0
    assert e_c == 0.0
    assert det


def test_checkerboard_flip_is_now_VISIBLE_not_absorbed():
    """The finding-1 class: the old per-cell fold scored this 0.0."""
    o = _field(2)
    ii, jj = np.indices(o.shape)
    checker = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    p = o * checker
    sent = np.zeros_like(o, dtype=bool)
    # the OLD fold: min(|o-p|, |o+p|) per cell == 0 everywhere -- the
    # exact blindness codex flagged.
    assert np.minimum(np.abs(o - p), np.abs(o + p)).max() == 0.0
    # the NEW global sign: no single s absorbs a checkerboard; the
    # residual is O(2*field) whichever sign wins.
    s, e_c, _e_o, _det = resolve_sign(o, p, sent)
    masks = {"all": np.ones_like(o, dtype=bool)}
    out, _ = score(o, p, masks, sent, sign=s)
    assert out["all"] > 0.4  # 2*min|o| over the 0.2-floor field


def test_side_specific_flip_is_visible():
    o = _field(3)
    p = o.copy()
    p[:, : o.shape[1] // 2] *= -1.0
    s, e_c, _e_o, _det = resolve_sign(o, p, np.zeros_like(o, dtype=bool))
    assert e_c > 0.4


def test_near_zero_field_reports_indeterminate_sign():
    o = np.full((8, 8), 1.0e-16)
    s, _e_c, _e_o, det = resolve_sign(o, o, np.zeros_like(o, dtype=bool))
    assert not det


def test_recip_sentinel_mask_catches_tiny_poison():
    # rarea at poisoned area cells is 1/(+-big_number) ~ 1e-30: tiny,
    # invisible to a magnitude ceiling, sentinel-class for _RECIP.
    o = np.array([[2.5e-11, 1.0e-30], [3.0e-8, 2.0e-9]])
    p = np.array([[2.5e-11, 3.0e-8], [3.0e-8, 2.0e-9]])
    sent = sentinel_mask(o, p, "rarea")
    assert sent[0, 1] and not sent[0, 0] and not sent[1, 0]


def test_region_masks_partition_the_plane():
    masks = region_masks(6, 3, 0, 1)
    tot = np.zeros_like(next(iter(masks.values())), dtype=int)
    for m in masks.values():
        tot += m.astype(int)
    assert (tot == 1).all()


@pytest.mark.slow
def test_coherent_perturbation_keeps_builder_invariants():
    """After perturb: area*rarea == 1 (the old probe made it fac^2),
    cos^2+sin^2 == 1 on rotated cells, ectx dx6 snapshots moved."""
    from full_step_oracle_parity import perturb_boundary_metrics_coherent
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    n, ng, eps = 12, 3, 1.0e-6
    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True)
    ref = [{k: np.array(v, copy=True)
            for k, v in gs.items() if isinstance(v, np.ndarray)}
           for gs in ctx["gs6"]]
    ref_dx6 = [np.array(a, copy=True) for a in ctx["ectx"]["dx6"]]
    ref_dy6 = [np.array(a, copy=True) for a in ctx["ectx"]["dy6"]]

    perturb_boundary_metrics_coherent(ctx, eps, n, ng)

    moved_any = False
    for t in range(6):
        gs = ctx["gs6"][t]
        # reciprocal invariants -- the class the OLD perturbation broke
        # (finding 3: area*rarea became fac^2), over EVERY recip pair
        for pk, rk in (("area", "rarea"), ("area_c", "rarea_c"),
                       ("dx", "rdx"), ("dy", "rdy"),
                       ("dxa", "rdxa"), ("dya", "rdya"),
                       ("dxc", "rdxc"), ("dyc", "rdyc")):
            p = np.asarray(gs[pk])
            r = np.asarray(gs[rk])
            live = (np.abs(p) < 1.0e29) & (np.abs(r) > 1.0e-20)
            assert np.abs(p[live] * r[live] - 1.0).max() < 1.0e-14, \
                (pk, rk)
        # angle coherence after rotation, ALL pairs (u, v, B-plane)
        for ck, sk in (("cosa_u", "sina_u"), ("cosa_v", "sina_v"),
                       ("cosa", "sina")):
            c, s = np.asarray(gs[ck]), np.asarray(gs[sk])
            lu = (np.abs(c) < 1.0e6) & (np.abs(s) < 1.0e6)
            assert np.abs(c[lu] ** 2 + s[lu] ** 2 - 1.0).max() < 1.0e-13
        # consumed metric summaries refreshed (codex instr r1 H2):
        # the stored scalars must equal the extremum of the PERTURBED
        # arrays over the builder's compute range
        sl_c = slice(ng, ng + n)
        assert float(gs["da_min"]) == float(
            np.asarray(gs["area"])[sl_c, sl_c].min())
        assert float(gs["da_min_c"]) == float(
            np.asarray(gs["area_c"])[sl_c, sl_c].min())
        # the field actually moved at boundary cells (non-vacuous
        # control -- a perturbation of zero cells is a no-op probe)
        if not np.array_equal(np.asarray(gs["dx"]), ref[t]["dx"]):
            moved_any = True
        # BOTH ectx snapshots refreshed (finding 2: c2l consumed stale
        # lengths)
        assert not np.array_equal(np.asarray(ctx["ectx"]["dx6"][t]),
                                  ref_dx6[t])
        assert not np.array_equal(np.asarray(ctx["ectx"]["dy6"][t]),
                                  ref_dy6[t])
        # strict interior BIT-EXACT, including the angle families the
        # arctan2 round-trip could silently re-round (codex instr r1
        # H1) and a derived family
        sl = slice(ng + 1, n + ng - 1)
        for k in ("dx", "cosa_u", "sina_u", "cosa", "sina",
                  "sin_sg", "cos_sg", "rsin_u", "divg_u", "rarea"):
            assert np.array_equal(np.asarray(gs[k])[sl, sl],
                                  ref[t][k][sl, sl]), k
    assert moved_any


def test_zero_cell_perturbation_is_refused():
    """A control that perturbs nothing must raise, not run."""
    from full_step_oracle_parity import perturb_boundary_metrics_coherent

    n, ng = 6, 3
    m_a, m_b = n + 2 * ng, n + 2 * ng + 1
    big = 1.0e30
    gs = {
        "dx": np.full((m_a, m_b), big), "dy": np.full((m_b, m_a), big),
        "dxa": np.full((m_a, m_a), big), "dya": np.full((m_a, m_a), big),
        "dxc": np.full((m_b, m_a), big), "dyc": np.full((m_a, m_b), big),
        "area": np.full((m_a, m_a), big),
        "area_c": np.full((m_b, m_b), big),
        "cosa_u": np.full((m_b, m_a), big),
        "sina_u": np.full((m_b, m_a), big),
        "cosa_v": np.full((m_a, m_b), big),
        "sina_v": np.full((m_a, m_b), big),
        "cosa": np.full((m_b, m_b), big),
        "sina": np.full((m_b, m_b), big),
        "cosa_s": np.full((m_a, m_a), big),
        "sin_sg": np.full((m_a, m_a, 9), big),
        "cos_sg": np.full((m_a, m_a, 9), big),
        "rdx": np.full((m_a, m_b), big), "rdy": np.full((m_b, m_a), big),
        "rdxa": np.full((m_a, m_a), big),
        "rdya": np.full((m_a, m_a), big),
        "rdxc": np.full((m_b, m_a), big),
        "rdyc": np.full((m_a, m_b), big),
        "rarea": np.full((m_a, m_a), big),
        "rarea_c": np.full((m_b, m_b), big),
        "rsina": np.full((m_b, m_b), big),
        "rsin_u": np.full((m_b, m_a), big),
        "rsin_v": np.full((m_a, m_b), big),
        "rsin2": np.full((m_a, m_a), big),
        "divg_u": np.full((m_a, m_b), big),
        "divg_v": np.full((m_b, m_a), big),
        "del6_u": np.full((m_a, m_b), big),
        "del6_v": np.full((m_b, m_a), big),
    }
    ctx = {"gs6": [gs] * 6, "ectx": None}
    with pytest.raises(SystemExit, match="PERTURBATION CONTROL FAILED"):
        perturb_boundary_metrics_coherent(ctx, 1.0e-10, n, ng)
