"""ecRad 49r1 McICA cloud generator, IFS configuration (JAX port).

Port of ``radiation_cloud_generator.F90:cloud_generator`` with
``generate_column_exp_ran`` (the original, non-vectorizable generator:
``use_vectorizable_generator = false``), ``radiation_cloud_cover.F90:
cum_cloud_cover_exp_ran`` (Exp-Ran overlap, ``use_beta_overlap = false``) and
``radiation_pdf_sampler.F90:sample_from_pdf`` (Gamma table ``mcica_gamma.nc``),
OpenIFS 49r1 (8f6f722).  No other overlap scheme or generator is implemented.

Random numbers (user decision A16): the generator takes its uniform deviates
as explicit arrays with a static layout (vote G1), per column:

* ``rand_top (ng)``: the cloud-top draws (``rand_top`` in the oracle);
* ``rand_cloud (ng, nlev)``: slot ``k`` holds the draw deciding the transition
  across interface ``k`` (layer ``k`` -> ``k+1``, 0-based).  The oracle draws
  ``iend+1-itrigger`` numbers per g-point and reads ``iend-itrigger``; the
  surplus draw sits in slot ``iend`` and is never read;
* ``rand_inhom1/2 (ng, nlev)``: slot = layer.

Slots the oracle never draws are never read (the oracle test proves it).
:func:`draw_uniforms` fills the layout from the JAX PRNG for production.

Smoothing: ``sharpness = math.inf`` is the oracle's hard form.  Finite
sharpness smooths the McICA cloud/no-cloud tests (see the per-function notes
and the votes recorded in the port report).
"""

import functools
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.smooth import positive_part_c1, sigmoid_switch, smoothstep_cubic

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 McICA cloud generator: exponential-random cumulative cloud "
        "cover and stochastic sub-columns (one per g-point) of cloud optical "
        "depth scaling, with Gamma-distributed horizontal inhomogeneity "
        "(Raisanen et al. 2004 as modified by Hogan in ecRad)."
    ),
    "inputs": {
        "cloud_fraction": "1 (layer cloud fraction, index 0 = model top, cropped)",
        "overlap_param": "1 (Hogan-Illingworth alpha at the nlev-1 interfaces)",
        "fractional_std": "1 (fractional standard deviation of in-cloud water)",
        "rand_*": "1 (uniform deviates in [0,1], layout in the module docstring)",
    },
    "outputs": {
        "od_scaling": "1 (ng, nlev) cloud optical depth scaling, 0 = clear",
        "total_cloud_cover": "1 (0 when below the cloud fraction threshold)",
    },
    "sign_convention": "od_scaling >= 0; 0 <= total_cloud_cover <= 1.",
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) ifs-source/radiation/radiation_cloud_generator.F90, "
        "radiation_cloud_cover.F90, radiation_pdf_sampler.F90; Hogan & "
        "Illingworth (2000); Raisanen et al. (2004); Hogan et al. (2016)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_mcica_generator_oracle.py: replays the "
        "oracle's random stream and matches od_scaling and total_cloud_cover "
        "of the running 49r1 ecRad to 1e-12 (fp64)."
    ),
}

# --- radiation_cloud_cover.F90:59 MaxCloudFrac = 1 - epsilon(1.0_jprb)*10 ---
_MAX_CLOUD_FRAC = 1.0 - float(np.finfo(np.float64).eps) * 10.0

# --- seed offset of the LW generator (radiation_mcica_lw.F90:189) ---
LW_SEED_OFFSET = 997

_DATA = os.path.join(os.path.dirname(__file__), "data")
GAMMA_FILE = os.path.join(_DATA, "mcica_gamma_48r1.npz")
GAMMA_PROVENANCE_FILE = os.path.join(_DATA, "mcica_gamma_48r1.json")


@functools.lru_cache(maxsize=1)
def load_pdf_sampler():
    """(x, fsd1, inv_fsd_interval) as radiation_pdf_sampler.F90:setup_pdf_sampler.

    ``x`` is fp64 with shape (nfsd, ncdf) (Fortran ``val(ncdf, nfsd)``
    transposed); the float32 file values are upcast exactly as the Fortran read.
    """
    if not os.path.exists(GAMMA_FILE):
        raise FileNotFoundError(f"McICA gamma table missing: {GAMMA_FILE}")
    with np.load(GAMMA_FILE) as z:
        fsd = z["fsd"].astype(np.float64)
        x = z["x"].astype(np.float64)
    return x, float(fsd[0]), 1.0 / (float(fsd[1]) - float(fsd[0]))


def gamma_provenance():
    with open(GAMMA_PROVENANCE_FILE) as f:
        return json.load(f)


def _hard(sharpness):
    return sharpness == math.inf


def _lt(a, b, sharpness):
    """Indicator of the oracle's strict ``a < b``; exact at ``sharpness = inf``."""
    return 1.0 - sigmoid_switch(a - b, sharpness)


def _step_pos(x, width, sharpness):
    """``x > 0`` at inf; else C1 step, exactly 0 for x <= 0 and 1 for x >= width."""
    if _hard(sharpness):
        return (x > 0.0).astype(x.dtype)
    return smoothstep_cubic(x / width)


def _max_c1(a, b, sharpness):
    """``max(a, b)`` of two fractions in [0, 1] (vote G12, third form).

    ``(a + b + sabs(a - b)) / 2`` with the C1 Huber absolute value
    ``sabs(d) = |d|`` for ``|d| >= w`` else ``d**2/(2w) + w/2``,
    ``w = min(a + b, 2 - a - b) / sharpness``: never below the max (so the
    pair cover keeps ``(1 - pair)/(1 - f)`` in [0, 1]), at most ``w/4`` above
    it and never above 1; exact at 0, at 1 and for ``|a - b| >= w``.  The
    band formula only sees ``w`` where ``w > |d| >= 0`` (double where).
    """
    if _hard(sharpness):
        return jnp.maximum(a, b)
    d = a - b
    w = jnp.minimum(a + b, 2.0 - a - b) / sharpness
    band = jnp.abs(d) < w
    ws = jnp.where(band, w, 1.0)
    sabs = jnp.where(band, d * d / (2.0 * ws) + 0.5 * ws, jnp.abs(d))
    return 0.5 * (a + b + sabs)


def _clip01(w, sharpness):
    """``max(0, min(w, 1))`` (vote G9'): identity on [0, 1], C1 outside."""
    if _hard(sharpness):
        return jnp.clip(w, 0.0, 1.0)
    e = 1.0 / sharpness
    w = positive_part_c1(w + e, sharpness) - e
    return 1.0 - (positive_part_c1(1.0 - w + e, sharpness) - e)


def sample_from_pdf(fsd, cdf, sampler=None, sharpness=100.0):
    """radiation_pdf_sampler.F90:sample_from_pdf (bilinear table look-up).

    Integer index selection feeds a continuous interpolation (vote G14, not a
    switch); the weight clamps are inactive for cdf in [0, 1] and fsd in the
    table range and smoothed outside it (G9').
    """
    x, fsd1, inv_fsd_interval = load_pdf_sampler() if sampler is None else sampler
    nfsd, ncdf = x.shape
    wcdf = cdf * (ncdf - 1) + 1.0
    icdf = jnp.clip(jnp.trunc(wcdf).astype(jnp.int32), 1, ncdf - 1)
    wcdf = _clip01(wcdf - icdf, sharpness)
    wfsd = (fsd - fsd1) * inv_fsd_interval + 1.0
    ifsd = jnp.clip(jnp.trunc(wfsd).astype(jnp.int32), 1, nfsd - 1)
    wfsd = _clip01(wfsd - ifsd, sharpness)
    x = jnp.asarray(x)
    return ((1.0 - wcdf) * (1.0 - wfsd) * x[ifsd - 1, icdf - 1]
            + (1.0 - wcdf) * wfsd * x[ifsd, icdf - 1]
            + wcdf * (1.0 - wfsd) * x[ifsd - 1, icdf]
            + wcdf * wfsd * x[ifsd, icdf])


def cum_cloud_cover_exp_ran(frac, overlap_param, sharpness=100.0):
    """radiation_cloud_cover.F90:181-273 for one column (alpha overlap).

    Returns (cum_cloud_cover (nlev), pair_cloud_cover (nlev-1)).  The
    ``frac >= MaxCloudFrac`` branch (cum_product = 0) is a 0/0 guard whose
    value equals the limit of the other branch; it is a C1 gate of width
    (1-MaxCloudFrac)/sharpness ending exactly at MaxCloudFrac (vote G13'), so
    the safe denominator only acts where the gate weight of its branch is 0.
    """
    f1, f2 = frac[:-1], frac[1:]
    pair = (overlap_param * _max_c1(f1, f2, sharpness)
            + (1.0 - overlap_param) * (f1 + f2 - f1 * f2))

    def step(cum_product, xs):
        f, p = xs
        full = f >= _MAX_CLOUD_FRAC
        if _hard(sharpness):
            gate = full.astype(f.dtype)
        else:
            gate = smoothstep_cubic((f - _MAX_CLOUD_FRAC)
                                    / ((1.0 - _MAX_CLOUD_FRAC) / sharpness) + 1.0)
        cum_product = (1.0 - gate) * (cum_product * (1.0 - p)
                                      / jnp.where(full, 1.0, 1.0 - f))
        return cum_product, 1.0 - cum_product

    _, cum_rest = jax.lax.scan(step, 1.0 - frac[0], (f1, pair))
    return jnp.concatenate([frac[:1], cum_rest]), pair


def _generate_column(frac, overlap_param, fractional_std, rand_top, rand_cloud,
                     rand_inhom1, rand_inhom2, decorrelation_scaling, frac_threshold,
                     sampler, sharpness):
    nlev = frac.shape[0]
    cum, pair = cum_cloud_cover_exp_ran(frac, overlap_param, sharpness)
    tcc = cum[-1]
    pad = jnp.zeros((1,), frac.dtype)       # interface arrays padded to nlev
    overhang = jnp.concatenate([cum[1:] - cum[:-1], pad])
    pair = jnp.concatenate([pair, pad])
    op = jnp.concatenate([overlap_param, pad])

    # Range of cloudy layers ibegin..iend (layers with frac > 0) as masks
    # built from a per-layer presence m (vote G6'): exactly frac > 0 at inf,
    # exactly 0/1 for cropped input (frac = 0 or >= frac_threshold).
    m = _step_pos(frac, frac_threshold / sharpness, sharpness)
    at_or_above = 1.0 - jnp.cumprod(1.0 - m)                 # l >= ibegin
    none_below_incl = jnp.cumprod((1.0 - m)[::-1])[::-1]
    at_or_below = 1.0 - none_below_incl                      # l <= iend
    none_below = jnp.concatenate([none_below_incl[1:], jnp.ones((1,), frac.dtype)])  # l >= iend

    # Overlap parameter of inhomogeneities (cloud_generator lines 187-194):
    # op**(1/decorrelation_scaling) for interfaces ibegin..iend-1 where op > 0.
    use_pow = (at_or_above * jnp.concatenate([at_or_below[1:], pad])
               * _step_pos(op, 1.0 / sharpness, sharpness))
    op_pow = jnp.power(jnp.where(op > 0.0, op, 1.0), 1.0 / decorrelation_scaling)
    op_inhom = use_pow * op_pow + (1.0 - use_pow) * op

    # Quantities of the interface above each layer l (index l-1); l = 0 has
    # none and is never reached with the trigger found above it.
    km1 = jnp.maximum(jnp.arange(nlev) - 1, 0)
    f_up, pair_up, cum_up, over_up, opi_up = (
        frac[km1], pair[km1], cum[km1], overhang[km1], op_inhom[km1])

    # Cloud-top search (cloud_generator lines 222-227): stop at the first l >=
    # ibegin with trigger <= cum(l), or at iend.
    trigger = rand_top * tcc                                  # (ng,)
    stop = sigmoid_switch(cum[None, :] - trigger[:, None], sharpness) * at_or_above
    stop = stop + (1.0 - stop) * none_below
    # Smooth mode only: no soft cloud in a frac = 0 layer (vote G5').
    keep = at_or_below if _hard(sharpness) else at_or_below * m

    def step(carry, xs):
        found_prev, c_prev, u_prev = carry
        (stop_l, r_cloud, r1, r2, f_l, f_u, p_u, c_u, o_u, opi_u, fsd_l, keep_l) = xs
        first = (1.0 - found_prev) * stop_l
        # generate_column_exp_ran lines 318-334: transition probabilities
        # given cloud (c_prev) or clear sky above.
        to_cloud = _lt(r_cloud * f_u, f_l + f_u - p_u, sharpness)
        to_cloud_from_clear = _lt(r_cloud * (c_u - f_u), p_u - o_u - f_u, sharpness)
        markov = c_prev * to_cloud + (1.0 - c_prev) * to_cloud_from_clear
        c = (first + found_prev * markov) * keep_l
        # lines 350-363: within a contiguous cloud, repeat the random number
        # of the layer above with probability op_inhom.
        w = c_prev * _lt(r2, opi_u, sharpness)
        u = w * u_prev + (1.0 - w) * r1
        od = c * sample_from_pdf(fsd_l, u, sampler, sharpness)
        return (found_prev + first, c, u), od

    ng = rand_top.shape[0]
    zeros = jnp.zeros((ng,), frac.dtype)
    bcast = lambda a: jnp.broadcast_to(a[:, None], (nlev, ng))  # noqa: E731
    rc_up = jnp.concatenate([rand_cloud[:, :1], rand_cloud[:, :-1]], axis=1)  # slot l-1
    xs = (stop.T, rc_up.T, rand_inhom1.T, rand_inhom2.T,
          bcast(frac), bcast(f_up), bcast(pair_up), bcast(cum_up), bcast(over_up),
          bcast(opi_up), bcast(fractional_std), bcast(keep))
    _, od = jax.lax.scan(step, (zeros, zeros, zeros), xs)

    # cloud_generator lines 166-169: clear column below the threshold
    # (tcc = 0 if tcc < threshold); smooth width 1 % of the threshold (vote G7).
    tcc = tcc * sigmoid_switch((tcc - frac_threshold) / frac_threshold, sharpness)
    return od.T, tcc


def cloud_generator(cloud_fraction, overlap_param, fractional_std, rand_top,
                    rand_cloud, rand_inhom1, rand_inhom2, *, decorrelation_scaling,
                    frac_threshold, sharpness=100.0, sampler=None):
    """McICA optical-depth scalings for columns (leading batch dimension).

    Shapes: cloud_fraction, fractional_std (ncol, nlev); overlap_param
    (ncol, nlev-1); rand_top (ncol, ng); rand_cloud, rand_inhom1/2
    (ncol, ng, nlev).  ``decorrelation_scaling`` is the namelist
    ``cloud_inhom_decorr_scaling`` and ``frac_threshold`` the namelist
    ``cloud_fraction_threshold`` (both explicit: no defaults).  Returns
    (od_scaling (ncol, ng, nlev), total_cloud_cover (ncol,)); like the
    oracle, od_scaling carries no meaning where total_cloud_cover is 0.
    """
    sampler = load_pdf_sampler() if sampler is None else sampler
    fn = functools.partial(_generate_column, decorrelation_scaling=decorrelation_scaling,
                           frac_threshold=frac_threshold, sampler=sampler,
                           sharpness=sharpness)
    return jax.vmap(fn)(cloud_fraction, overlap_param, fractional_std, rand_top,
                        rand_cloud, rand_inhom1, rand_inhom2)


def draw_uniforms(seed, ng, nlev):
    """Production uniforms in the generator layout from the JAX PRNG (vote G2).

    ``seed`` is the oracle's seed for one column: ``iseed + LW_SEED_OFFSET``
    for LW, ``iseed`` for SW.  Returns (rand_top, rand_cloud, rand_inhom1,
    rand_inhom2).  vmap over ``seed`` for columns.
    """
    k_top, k_cloud, k_i1, k_i2 = jax.random.split(jax.random.PRNGKey(seed), 4)
    return (jax.random.uniform(k_top, (ng,)),
            jax.random.uniform(k_cloud, (ng, nlev)),
            jax.random.uniform(k_i1, (ng, nlev)),
            jax.random.uniform(k_i2, (ng, nlev)))
