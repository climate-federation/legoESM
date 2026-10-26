"""Maximum-random-overlap subcolumns for cloud radiation (CAM McICA generator).

WHY THIS EXISTS.  The RRTMGP path carries no McICA, subcolumn or overlap
machinery: every layer's GRID-MEAN water path is handed to a single
homogeneous column, so a sky whose cloud is spread thinly over many partly
cloudy layers is solved as one deep uniform cloud.  Measured against a
Monte-Carlo independent-column reference on a real model state (prodA day 310,
25 columns x 24 subcolumns, generator validated against the model's own
:func:`maximum_random_overlap`), that costs

    cloud albedo  0.3995 -> 0.2786   (-0.1209 +- 0.0157, paired, 7.7 sigma)
    OLR           219.24 -> 237.78 W/m2   (reference rlut 239.0)

i.e. ~30% of the cloud albedo, and it moves the SW and LW errors toward
observations together.

ORACLE.  :func:`generate_subcolumns` is CAM's maximum-random generator,
``generate_stochastic_clouds_sw`` case(2) in
``rrtmg/aer_src/mcica_subcol_gen_sw.f90`` (cam_cesm2_1_rel_60, a03b84b7):
the recursion keeps the layer above's CDF value when that subcolumn was cloudy
(strict ``>``, line 365) and rescales the uniform into the clear part
otherwise, and a cell is cloudy when ``CDF >= 1 - cf`` (``>=``, line 398).
Given the same uniforms the masks are identical to the running CAM code
(``tests/atmosphere/test_max_random_subcolumns_oracle.py``).

UNIFORMS.  CAM draws fresh KISS uniforms per (subcolumn, column, layer),
seeded from the fractional parts of the bottom four layer pressures, so
columns with different pressure digits sample different subcolumns.  Here the uniforms are the
stratified set ``(i+0.5)/n_sub``, independently permuted per layer from a
table fixed at import, offset per column and layer by a shift seeded from the
fractional part of the bottom-layer pressure (:func:`cam_seed_shift`; the
same uniforms the ``mcica`` path uses; CAM seeds from four bottom pressures).  Over
columns whose shifts spread over [0, 1), a layer with ``cf < 1/n_sub`` is
cloudy in some columns and its cloud is kept on average to within the
shift's non-uniformity: layer ``k`` offsets by
``frac((1 + k*sqrt2)*shift + k*golden)``, which is not exactly uniform (for
an isolated cf = 0.01 in the second layer the mean cloudy fraction over
uniform shifts is 0.00984).  Columns with the same bottom
pressure (e.g. an idealized uniform state, or integer pressures, which
give a zero shift) share one set of subcolumns.  There is
no PRNG state in the hot loop, and the result is reproducible.  Without the
per-column shift (``shift=None``) every column would reuse ONE table and a
layer with ``0 < cf < 1/(2 n_sub)`` could never be cloudy unless cloud is
carried from the layer above (tier-2 oracle audit: -8.9 W/m2 TOA reflected
SW at ``n_sub = 8``).

WHY EIGHT (measured with the earlier, unshifted generator).  Subcolumn-count
sweep against the same reference:

    n_sub =  4   cloud albedo diff -0.0224   OLR diff +1.90 W/m2
    n_sub =  8   cloud albedo diff -0.0030   OLR diff +0.43 W/m2
    n_sub = 12   cloud albedo diff +0.0052   OLR diff -0.05 W/m2

Eight leaves 2.5% of a 0.121 signal for 8x the radiation cost; four leaves 18%.

REJECTED ALTERNATIVE, recorded so it is not retried: a closed-form column-level
inversion ``chi_col = gamma0 C / (gamma0 C + (1-C) tau_col)`` using the total
max-random cover ``C`` recovers only ~40% of the effect (cloud albedo 0.3509
against the 0.2786 reference).  It captures the clear/cloudy split but not the
VARIANCE of optical depth among cloudy subcolumns -- one subcolumn may cross
one cloud layer and another five -- and reflectance is concave in that spread.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

__physics_contract__ = {
    "summary": (
        "CAM McICA maximum-random-overlap subcolumn generator for cloud "
        "radiation: turns a layer cloud-fraction profile into n_sub binary "
        "cloud masks and the matching in-cloud water paths, so a partly "
        "cloudy sky is solved as independent columns instead of one "
        "homogeneous column carrying the grid-mean path."
    ),
    "inputs": {
        "cloud_fraction": "1 (0-1 area fraction, index 0 = model top)",
        "lwp": "kg/m^2 (GRID-MEAN liquid water path per layer)",
        "iwp": "kg/m^2 (GRID-MEAN ice water path per layer)",
        "n_sub": "1 (static subcolumn count)",
    },
    "outputs": {
        "mask": "bool (n_sub, ncol, nlev)",
        "subcolumn_paths": "kg/m^2 (IN-CLOUD paths, 0 in clear cells)",
    },
    "sign_convention": (
        "Diagnostic only (no state tendency): cloud_fraction clipped to "
        "[0,1]; water paths >= 0 and zero wherever the subcolumn is clear; "
        "cf=0 gives an all-clear mask and cf=1 an all-cloudy mask exactly."
    ),
    # The mean over subcolumns of the subcolumn water path returns the
    # grid-mean path: exact as n_sub -> inf, and to stratified-sampling
    # granularity at finite n_sub (column total within ~5% at n_sub=64).
    "conserves": ["moisture"],
    "differentiable": True,
    "reference": (
        "Raisanen, Barker, Khairoutdinov, Li & Randall (2004, QJRMS 130) "
        "maximum-random overlap subcolumn generator, as CAM implements it "
        "(cam_cesm2_1_rel_60 rrtmg/aer_src/mcica_subcol_gen_sw.f90 case(2)); "
        "stratified uniforms with a CAM-seeded per-column shift in place of "
        "the KISS draw"
    ),
    "idealized_test": (
        "tests/unit/test_cloud_subcolumns.py; cf=0 -> all clear, cf=1 -> all "
        "cloudy, per-layer P(cloudy)=cf, total cover matches "
        "maximum_random_overlap, adjacent layers overlap maximally, "
        "separated groups overlap randomly, grid-mean water conserved"
    ),
}

N_SUBCOLUMNS_DEFAULT = 8
_PERM_SEED = 20260731          # fixed at import: a table, never per-call noise
_CF_FLOOR = 1.0e-3             # matches clouds.cloud_fraction._INHOM_CF_FLOOR
# Per-layer McICA shift: irrational slope and step decorrelate layer offsets.
_SHIFT_LAYER_SLOPE = 2.0 ** 0.5
_SHIFT_LAYER_STEP = (5.0 ** 0.5 - 1.0) / 2.0


@functools.lru_cache(maxsize=32)
def _stratified_table_cached(n_sub: int, nlev: int) -> np.ndarray:
    """``(n_sub, nlev)`` stratified uniforms, independently permuted per layer.

    Cached on the STATIC ``(n_sub, nlev)`` pair, so it is built once per shape
    and enters the traced graph as a constant -- no PRNG state in the hot loop.
    """
    base = (np.arange(n_sub, dtype=np.float64) + 0.5) / n_sub
    rng = np.random.default_rng(_PERM_SEED)
    tab = np.stack([base[rng.permutation(n_sub)] for _ in range(nlev)], axis=1)
    # The cache hands the SAME object back on every hit, so make it read-only:
    # a caller mutating it would silently corrupt every later radiation call.
    tab.flags.writeable = False
    return tab


def _stratified_table(n_sub: int, nlev: int) -> np.ndarray:
    """Fresh copy of the cached table.

    The cache would otherwise hand the SAME object to every caller, and a
    read-only flag can be flipped back -- one mutation would then silently
    change every later radiation call. The table is n_sub x nlev floats, so
    copying costs nothing next to the solve it feeds.
    """
    return np.array(_stratified_table_cached(int(n_sub), int(nlev)), copy=True)


def generate_subcolumns(cloud_fraction, n_sub: int = N_SUBCOLUMNS_DEFAULT,
                        shift=None):
    """Boolean cloud mask ``(n_sub, ncol, nlev)`` under maximum-random overlap.

    Descending from the model top (index 0), a subcolumn already cloudy in the
    layer above keeps its CDF value, so adjacent cloudy layers overlap
    MAXIMALLY; otherwise the CDF is rescaled into the clear part, which makes
    the next cloud group overlap RANDOMLY.

    Parameters
    ----------
    cloud_fraction : jnp.ndarray
        ``(ncol, nlev)`` layer cloud fraction in [0, 1]; index 0 = model top.
    n_sub : int
        Number of subcolumns (STATIC -- it sets the traced batch size).
    shift : jnp.ndarray, optional
        ``(ncol,)`` per-column seed in [0, 1).  Layer ``k`` is offset by
        ``frac(shift*(1 + k*sqrt2) + k*golden)`` (added modulo 1 to the table),
        so each layer keeps its stratified marginal AND the offsets of different
        layers vary independently across columns.  A single offset shared by all
        layers kept every subcolumn's between-layer pairing fixed, so a given
        g-point saw the wrong overlap in every column (layers [0.5, 0, 0.5]:
        per-g-point cover 0.50-1.00 instead of 0.75; worst error 0.25 -> 0.03).
        Plain arithmetic, so CPU and GPU give the same masks.
    """
    cf = jnp.clip(jnp.asarray(cloud_fraction), 0.0, 1.0)
    ncol, nlev = cf.shape
    u = jnp.asarray(_stratified_table(int(n_sub), int(nlev)), dtype=cf.dtype)
    u = u[:, None, :]                                   # (n_sub, 1, nlev)
    if shift is not None:
        k = jnp.arange(nlev, dtype=cf.dtype)
        off = (jnp.asarray(shift, dtype=cf.dtype)[:, None] * (1.0 + k * _SHIFT_LAYER_SLOPE)
               + k * _SHIFT_LAYER_STEP)                  # (ncol, nlev)
        u = jnp.mod(u + off[None, :, :], 1.0)
    return max_random_mask(u, cf)


def max_random_mask(u, cloud_fraction):
    """CAM case(2) maximum-random recursion on given uniforms.

    ``u`` is ``(n_sub, ncol or 1, nlev)`` uniforms in [0, 1) (CAM's ``CDF``
    before the recursion), ``cloud_fraction`` ``(ncol, nlev)``, index 0 =
    model top.  Returns the boolean cloud mask ``(n_sub, ncol, nlev)``.
    """
    cf = jnp.clip(jnp.asarray(cloud_fraction), 0.0, 1.0)
    ncol, nlev = cf.shape
    u = jnp.asarray(u, dtype=cf.dtype)
    clear = 1.0 - cf                                    # (ncol, nlev)

    # nlev is static and small, so the recursion unrolls; a lax.scan would add
    # a carry for no benefit at this size.
    col = u[:, :, 0] * jnp.ones((1, ncol), dtype=cf.dtype)
    cdf = [col]
    for k in range(1, nlev):
        # CAM mcica_subcol_gen_sw.f90:365-369 (strict ``>``).
        above_cloudy = cdf[k - 1] > clear[None, :, k - 1]
        cdf.append(jnp.where(above_cloudy, cdf[k - 1],
                             u[:, :, k] * clear[None, :, k - 1]))
    # CAM mcica_subcol_gen_sw.f90:398: isCloudy = (CDF >= 1 - cldf).  ``>=``
    # (not ``>``) matters only at exact ties, which stratified dyadic
    # uniforms and quantised cloud fractions (e.g. cf = 0.0625) do produce.
    return jnp.stack(cdf, axis=-1) >= clear[None, :, :]


def cam_seed_shift(p_full):
    """Per-column subcolumn shift ``(ncol,)`` in [0, 1), state-seeded like CAM.

    CAM's McICA seeds KISS from the fractional parts of the bottom FOUR
    layer pressures (mcica_subcol_gen_sw.f90:293-296), so its draw is
    state-derived: independent of the domain decomposition, with no PRNG
    state to carry.  This shift (the one the ``mcica`` path has used) takes
    only the fractional part of the lowest-layer pressure [Pa] (index -1 =
    model bottom): columns with identical bottom pressure share a shift even
    where CAM's seeds would differ, and an fp32 pressure quantises it (to
    1/128 near 1e5 Pa).  Gradient-stopped: the mask is a sampling device,
    not a function of pressure.
    """
    return jax.lax.stop_gradient(jnp.mod(jnp.asarray(p_full)[:, -1], 1.0))


def expand_to_subcolumns(mask, *fields):
    """Replicate ``(ncol, ...)`` fields to ``(n_sub*ncol, ...)`` column-major.

    The subcolumn axis is FLATTENED INTO THE COLUMN AXIS so the whole set is
    one batched solver call.  RRTMGP is 1-D and columns are independent, so
    this is exact -- and it is the only shape that works: looping separate
    calls exhausts XLA compilation memory, and solving arms in separate calls
    of different batch size drifts by ~5e-4 on the float32 lookup tables.
    """
    n_sub = mask.shape[0]
    out = []
    for f in fields:
        f = jnp.asarray(f)
        out.append(jnp.reshape(jnp.broadcast_to(f[None, ...],
                                                (n_sub,) + f.shape),
                               (n_sub * f.shape[0],) + f.shape[1:]))
    return out[0] if len(out) == 1 else tuple(out)


def in_cloud_paths(cloud_fraction, lwp, iwp, cf_floor: float = _CF_FLOOR):
    """In-cloud water paths ``(ncol, nlev)``: GRID-MEAN path / floored ``cf``."""
    cf = jnp.maximum(jnp.asarray(cloud_fraction), cf_floor)
    return jnp.asarray(lwp) / cf, jnp.asarray(iwp) / cf


def subcolumn_paths(mask, cloud_fraction, lwp, iwp,
                    cf_floor: float = _CF_FLOOR):
    """In-cloud water paths per subcolumn, flattened to ``(n_sub*ncol, nlev)``.

    ``lwp``/``iwp`` are GRID-MEAN (= ``cf`` x in-cloud), so the in-cloud value
    is ``path / cf``; cells that are clear in a subcolumn carry zero.  Averaged
    over subcolumns this returns the grid-mean path, which is what makes the
    scheme water-conserving.
    """
    n_sub, ncol, nlev = mask.shape
    return tuple(
        jnp.reshape(jnp.where(mask, ic[None, :, :], 0.0), (n_sub * ncol, nlev))
        for ic in in_cloud_paths(cloud_fraction, lwp, iwp, cf_floor))


def average_over_subcolumns(field, n_sub: int, ncol: int):
    """Collapse a ``(n_sub*ncol, ...)`` solver output back to ``(ncol, ...)``."""
    f = jnp.asarray(field)
    return jnp.mean(jnp.reshape(f, (n_sub, ncol) + f.shape[1:]), axis=0)


#: Solver kwargs that carry a LEADING COLUMN axis and must be replicated per
#: subcolumn.  This is an EXPLICIT list, not a shape heuristic: an earlier
#: version expanded any array with ``shape[0] == ncol``, which would silently
#: mis-expand ``solar_spectral_fraction`` -- that is ``(ngpt_sw,)``, a G-POINT
#: array of order 112, so a 112-column case would have corrupted it.
PER_COLUMN_SOLVER_KEYS = frozenset({
    "T", "p_full", "p_half", "sfc_temperature", "q_v", "cos_zenith",
    "sfc_albedo", "sfc_albedo_direct", "sfc_emissivity", "o3_vmr", "o3_top_vmr",
    "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
    "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
    "cloud_fraction",
})


def expand_kwargs(kwargs: dict, n_sub: int, ncol: int,
                  keys=PER_COLUMN_SOLVER_KEYS) -> dict:
    """Replicate the PER-COLUMN entries of a solver kwargs dict to ``n_sub*ncol``.

    Only keys in ``keys`` are touched, and only when they are arrays whose
    leading dimension is ``ncol`` (a scalar ``sfc_albedo`` is left alone).
    Everything else -- ``None``, scalars, g-point arrays, GHG dicts -- passes
    through unchanged.

    A key in ``keys`` whose array has a leading dimension OTHER than ``ncol``
    is a hard error rather than a silent pass-through: that means the caller's
    layout changed and the subcolumn expansion would be wrong.
    """
    out = {}
    for k, v in kwargs.items():
        if k not in keys or v is None or isinstance(v, (int, float, bool, str)):
            out[k] = v
            continue
        a = jnp.asarray(v)
        if a.ndim == 0:
            out[k] = v
            continue
        if a.shape[0] != ncol:
            raise ValueError(
                f"per-column solver kwarg {k!r} has leading dimension "
                f"{a.shape[0]}, expected ncol={ncol}; the subcolumn expansion "
                "cannot be applied safely"
            )
        out[k] = jnp.reshape(
            jnp.broadcast_to(a[None, ...], (n_sub,) + a.shape),
            (n_sub * ncol,) + a.shape[1:])
    return out


def average_output(result, n_sub: int, ncol: int):
    """Average every per-column field of a ``RadiationOutput`` over subcolumns.

    Fluxes and heating rates are averaged because the independent-column
    approximation IS the subcolumn-weighted mean of independent solutions.
    ``toa_insolation`` is identical across subcolumns, so averaging is a no-op
    on it rather than a special case.
    """
    return type(result)(*[
        None if f is None else average_over_subcolumns(f, n_sub, ncol)
        for f in result
    ])
