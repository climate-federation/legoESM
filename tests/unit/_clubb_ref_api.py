"""Bridge to the upstream CLUBB-JAX reference tree, for the parity tests.

The reference is an optional sibling checkout (``CLUBB-JAX`` beside the repo),
so every test that uses it skips when it is absent. Upstream's callable surface
moved when it became a full JAX port: its routines now take the level counts and
column count as leading STATIC arguments, and the grid is its own ``Grid``
NamedTuple rather than whatever object a caller happens to pass. Both of those
are encoded once here so the individual tests stay about physics.

Keeping this in one place also means the next upstream refresh has exactly one
adapter to update, instead of one hand-rolled grid stand-in per test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


def available() -> bool:
    """Is the optional reference checkout present?"""
    return (ROOT / "clubb_jax").exists()


def ensure_importable() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def ref_grid(gr):
    """Build the upstream ``Grid`` from one of our :class:`CLUBBGrid` objects.

    The heights are ours, so both sides discretise the same column; the derived
    spacings and interpolation weights are computed by UPSTREAM'S own helpers, so
    a parity failure cannot be blamed on a hand-copied weight formula.
    """
    ensure_importable()
    from clubb_jax.src.CLUBB_core import grid_class as ref_grid_class

    zm = np.asarray(gr.zm)
    zt = np.asarray(gr.zt)
    ngrdcol, nzm = zm.shape
    nzt = zt.shape[1]
    dzm, dzt, invrs_dzm, invrs_dzt = ref_grid_class._calc_grid_spacings(
        nzm, ngrdcol, zm, zt)
    return ref_grid_class.Grid(
        nzm=nzm, nzt=nzt, ngrdcol=ngrdcol,
        zm=gr.zm, zt=gr.zt,
        dzm=dzm, dzt=dzt, invrs_dzm=invrs_dzm, invrs_dzt=invrs_dzt,
        weights_zt2zm=ref_grid_class._calc_zt2zm_weights(nzm, nzt, ngrdcol, zm, zt),
        weights_zm2zt=ref_grid_class._calc_zm2zt_weights(nzt, ngrdcol, zm, zt, dzt),
        k_lb_zm=0, k_ub_zm=nzm - 1, k_lb_zt=0, k_ub_zt=nzt - 1,
        grid_dir_indx=1, grid_dir=1.0,
    )


def shape_of(gr):
    """``(ngrdcol, nzm, nzt)`` for the leading static arguments upstream wants."""
    ngrdcol, nzm = np.asarray(gr.zm).shape
    return ngrdcol, nzm, np.asarray(gr.zt).shape[1]


def ref_stats(ncol: int, max_nlev: int):
    """A disabled stats collector.

    Upstream threads a stats object through every advance so it can write budget
    terms. Parity tests care about the returned fields, not the diagnostics, so
    this is the collector with sampling switched off and no registered names.
    """
    ensure_importable()
    from clubb_jax.src.CLUBB_core.jax_stats import JaxStats

    return JaxStats.empty(l_sample=False, names=(), ncol=ncol, max_nlev=max_nlev)


def ref_flags(**overrides):
    """Upstream's default config flags, with any overrides applied.

    Our port implements only CAM's flag tree, so a parity test must hand the
    reference the CAM values wherever they differ from upstream's library
    defaults -- otherwise the two sides are running different physics and the
    comparison is a confound, not a result.
    """
    ensure_importable()
    from clubb_jax.src.CLUBB_core.model_flags import get_default_config_flags

    flags = get_default_config_flags()
    return flags._replace(**overrides) if overrides else flags


# CAM's overrides of the upstream library defaults, for the flags our port
# hard-wires. Mirrors the reference table at the foot of clubb.py; anything not
# listed already agrees between CAM and the library.
CAM_FLAGS = dict(
    l_predict_upwp_vpwp=False,
    l_diag_Lscale_from_tau=False,
    l_use_C7_Richardson=False,
    l_damp_wp2_using_em=False,
    l_vert_avg_closure=True,
    l_trapezoidal_rule_zt=True,
    l_trapezoidal_rule_zm=True,
    l_call_pdf_closure_twice=True,
    l_use_cloud_cover=True,
    l_stability_correct_tau_zm=True,
    l_rcm_supersat_adj=False,
    l_damp_wp3_Skw_squared=False,
    l_use_tke_in_wp3_pr_turb_term=False,
)


def assert_matches(mine, ref, name: str = "", rtol: float = 1e-13, atol: float = 0.0):
    """Compare our result against the reference on the levels we both define.

    Several of our term builders return only the interior band the solver
    consumes, where upstream returns the full column with its boundary levels
    zeroed. That is a storage convention, not a physics difference, so the
    reference is TRIMMED to our width rather than our result being padded --
    and the trimmed-off levels are asserted to be the zeros they claim to be,
    so a real value hiding in a boundary level cannot slip through.

    The default tolerance is a few tens of ulp rather than bit-exact. The
    reference's routines are jit-compiled, so XLA may contract a multiply-add
    that our eager form evaluates in two steps, and a multi-term expression may
    be summed in a different order. Both are last-bit effects. The tolerance is
    still some ten orders of magnitude tighter than any difference in the
    physics would be, so the gate fails on anything that matters.
    """
    mine = np.asarray(mine)
    ref = np.asarray(ref)
    if ref.shape[-1] == mine.shape[-1] + 2:
        edges = np.concatenate([ref[..., :1], ref[..., -1:]], axis=-1)
        np.testing.assert_array_equal(
            edges, np.zeros_like(edges),
            err_msg=f"{name}: reference boundary levels are not zero, so "
                    f"trimming them would hide a real difference")
        ref = ref[..., 1:-1]
    if rtol or atol:
        np.testing.assert_allclose(mine, ref, rtol=rtol, atol=atol, err_msg=name)
    else:
        np.testing.assert_array_equal(mine, ref, err_msg=name)


def ref_err_info(ngrdcol: int):
    """Upstream's per-column error record, initialised to "no error"."""
    ensure_importable()
    from clubb_jax.src.CLUBB_core.err_info import ErrInfo

    return ErrInfo.initialized(ngrdcol)


def ref_sclr_idx():
    """Passive-scalar index map with every scalar switched off.

    Our port carries no passive scalars at all, so every index is zero and the
    reference's scalar branches are unreachable -- which is what makes the two
    sides comparable in the first place.
    """
    ensure_importable()
    from clubb_jax.src.CLUBB_core.sclr_idx import SclrIdx

    return SclrIdx(0, 0, 0, 0, 0, 0)


def ref_nu(config, ngrdcol: int, nzm: int):
    """The grid-spacing-dependent diffusion coefficients, from OUR config.

    Upstream scales these by a factor derived from the average layer depth when
    that option is on; it is off in the CAM tree, where the factor is one, so
    the values pass through unchanged.
    """
    ensure_importable()
    from clubb_jax.src.CLUBB_core.nu_vert_res_dep import NuVertResDep

    p = config.params
    col = lambda v: np.full((ngrdcol,), float(v))  # noqa: E731
    return NuVertResDep(
        nzm=int(nzm), nu1=col(p.nu1), nu2=col(p.nu2), nu6=col(p.nu6),
        nu8=col(p.nu8), nu9=col(p.nu9), nu10=col(p.nu10), nu_hm=col(p.nu_hm))


def ref_pdf_coefs(nz: int, ngrdcol: int):
    """Zeroed implicit PDF coefficients (the ADG1 tree does not populate them)."""
    ensure_importable()
    from clubb_jax.src.CLUBB_core.pdf_params import init_pdf_implicit_coefs_terms_api

    return init_pdf_implicit_coefs_terms_api(nz, ngrdcol, 0)


def ref_params(config, ngrdcol: int):
    """Upstream's ``(ngrdcol, 102)`` parameter array, filled from OUR config.

    Both sides must run the same coefficients or a parity failure says nothing
    about the code under test. Our ``CLUBBParams`` carries the same 102 namelist
    names, so the mapping is by name; a name present upstream but missing from
    our config keeps upstream's default, and that list is returned so a caller
    can assert it is empty.
    """
    ensure_importable()
    from clubb_jax.src.CLUBB_core.parameters_tunable import _DEFAULTS, PARAM_NAMES

    ours = config.params._asdict()
    values = np.empty((ngrdcol, len(PARAM_NAMES)), dtype=np.float64)
    missing = []
    for i, name in enumerate(PARAM_NAMES):
        if name in ours:
            values[:, i] = float(ours[name])
        else:
            missing.append(name)
            values[:, i] = float(_DEFAULTS[name])
    return values, missing
