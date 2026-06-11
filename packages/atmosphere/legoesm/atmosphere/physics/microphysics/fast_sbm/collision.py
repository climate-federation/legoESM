"""Bott flux-method collision-coalescence (oracle ``coal_bott`` core).

Port of WRF ``module_mp_fast_sbm.F``:

* ``courant_bott_KS`` (l. 1636) → :func:`precompute_collision_tables` —
  pair-interaction target bins ``ima`` and Courant numbers ``chucm``,
  pure functions of the mass grid (computed once, NumPy, then static).
* ``coll_xxx_lwf`` (l. 652) → :func:`bott_coalescence` — self-collection
  of one liquid spectrum. The oracle calls it for drops with liquid
  fraction ``fl ≡ 1`` (``call coll_xxx_lwf(G1, fl1=1, CWLL, ...)``), under
  which every ``*_w`` statement degenerates to the mass update (``fl`` stays
  1 identically), so the liquid port is mass-only; the LWF-tracking variant
  arrives with the snow iteration.

State variable: ``g(ln r) = 3 m² f(m)`` — mass density per unit ln(radius)
[kg m⁻³] (oracle: ``G1 = FF1R·3·x²·1e3``, mg cm⁻³). The pair kernel table is
``ck = K(m_i, m_j) · dt · dlnr`` with ``dlnr = COL`` (oracle ``Kernals_KS``:
``cwll = YWLL·dt·dlnr``, optionally × a coalescence-efficiency matrix).
The algorithm is unit-consistent in SI exactly as in CGS.

Faithfulness notes (documented deviations):
* The oracle's ``ix0/ix1`` integration-bound narrowing is a CPU shortcut —
  here every pair runs and empty pairs are masked (identical math).
* ``kp_flux_max = 44`` (1-based) is unreachable for NKR ≤ 43, so the
  halve-flux branch is omitted.
* Oracle hard-stops (``wrf_error_fatal`` on ``fl`` bounds) guard the LWF
  path only; the mass-only port has no equivalent state to corrupt.
* Sub-``gmin`` floors can create/destroy mass at the 1e-16 kg m⁻³ scale per
  pair (oracle behaviour, ``g_lim``); the box test bounds it.

Differentiable: pair updates are ``jnp.where`` selections of smooth
expressions; the sequential pass is a ``lax.scan`` over the 561 (i ≤ j)
pairs in oracle order.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import COL

__physics_contract__ = {
    "summary": (
        "Bott (1998) flux-method stochastic-collection step on the "
        "mass-doubling bin grid — liquid self-collection (oracle coal_bott/"
        "coll_xxx_lwf with fl=1)."
    ),
    "inputs": {
        "g": "kg m^-3 per unit ln(r) (mass distribution, (n_bins,))",
        "ck": "m^3 (kernel*dt*dlnr pair table, (n,n))",
        "masses": "kg (bin centres)",
        "prdkrn": "1 (kernel enhancement factor)",
    },
    "outputs": {"g": "kg m^-3 per unit ln(r) (updated spectrum)"},
    "sign_convention": (
        "Coalescence moves mass from source bins i,j to target bins k,k+1; "
        "number density never increases; total mass is conserved up to the "
        "gmin empty-bin floors (|ΔM| <~ n_pairs·gmin)."
    ),
    "conserves": ["mass"],
    "differentiable": True,
    "reference": (
        "Bott (1998) JAS 55:2284; Khain et al. (2004) JAS 61:2963; WRF "
        "module_mp_fast_sbm.F coll_xxx_lwf/courant_bott_KS/Kernals_KS"
    ),
    "idealized_test": (
        "Golovin-kernel box: N(t)/N(0)=exp(-b_m L t) and M2(t)/M2(0)="
        "exp(+2 b_m L t) analytic moment laws, mass conserved to ~1e-12; "
        "empty spectrum is a fixed point; single-bin self-collection "
        "deposits into bin k+1 (Courant c=0 on a doubling grid), with later "
        "pairs of the same Gauss-Seidel sweep advecting a small fraction "
        "onward — deposited total exactly equals the donor loss."
    ),
}

# Oracle ``g_lim = 1.0D-19*1.0D3`` [mg cm⁻³] = 1e-19 g cm⁻³ = 1e-16 kg m⁻³:
# bins at or below this mass density are treated as empty.
GMIN_DEFAULT = 1.0e-16


class CollisionTables(NamedTuple):
    """Static pair tables (oracle ``courant_bott_KS`` output, flattened).

    All fields have shape ``(n_pairs,)`` with ``n_pairs = n(n-1)/2 + ...`` —
    the (i ≤ j) pairs over bins ``0..n-2`` in oracle order (i outer
    ascending, j inner ascending). The last bin never collides as a source
    (oracle bounds both loops at ``nkr-1``) but still receives flux.
    """

    i_idx: jax.Array   # int32 source bin i
    j_idx: jax.Array   # int32 source bin j (>= i)
    k_idx: jax.Array   # int32 target bin k (oracle ``ima``, 0-based)
    c_pair: jax.Array  # float Courant number (oracle ``chucm``)


def precompute_collision_tables(masses) -> CollisionTables:
    """Build ``ima``/``chucm`` pair tables from the bin masses.

    Oracle ``courant_bott_KS``: for each pair the coalesced mass
    ``x0 = m_i + m_j`` falls between bins ``k-1`` and ``k``;
    ``chucm = ln(x0/m_{k-1})/ln 2`` (their ``/(3·dlnr)`` with
    ``dlnr = ln2/3``), snapped to the next bin when within 1e-8 of 1; the
    stored target is ``ima = min(nkr-1, k-1)`` (1-based) → 0-based
    ``min(n-2, k0-1)``. NumPy on purpose: static grid geometry, not traced.
    """
    m = np.asarray(masses, dtype=np.float64)
    n = m.shape[0]
    ii, jj, kk, cc = [], [], [], []
    for i in range(n - 1):          # oracle: do i = 1, nkr-1 (as source)
        for j in range(i, n - 1):
            x0 = m[i] + m[j]
            # On the production doubling grid every source pair lands
            # (x0 <= 2 m_j <= m_top). On FINER log grids the top source
            # pairs can overflow the grid (x0 > m_top); the oracle never
            # assigns those (its search just falls through), so they are
            # skipped here too — coalesced mass beyond the grid top is not
            # representable and the pair does not interact.
            for k in range(j, n):
                if k == 0:
                    continue
                if m[k] >= x0 and m[k - 1] < x0:
                    # Fractional log position of x0 inside [m_{k-1}, m_k].
                    # Oracle (doubling grid): ln(x0/m_{k-1})/ln2; written
                    # generally so refinement studies on finer log grids
                    # reuse the same solver.
                    c = math.log(x0 / m[k - 1]) / math.log(m[k] / m[k - 1])
                    k_t = k
                    if c > 1.0 - 1.0e-8:
                        c = 0.0
                        k_t = k + 1
                    ii.append(i)
                    jj.append(j)
                    kk.append(min(n - 2, k_t - 1))
                    cc.append(c)
                    break
    return CollisionTables(
        i_idx=jnp.asarray(ii, dtype=jnp.int32),
        j_idx=jnp.asarray(jj, dtype=jnp.int32),
        k_idx=jnp.asarray(kk, dtype=jnp.int32),
        c_pair=jnp.asarray(cc, dtype=jnp.float64),
    )


def collision_ck_matrix(kernel: jax.Array, dt: float | jax.Array) -> jax.Array:
    """Pair coefficient table ``ck = K·dt·dlnr`` (oracle ``Kernals_KS``).

    ``kernel``: gravitational/analytic collection kernel ``K(m_i, m_j)``
    [m³ s⁻¹], shape ``(n, n)``; multiply by a coalescence-efficiency matrix
    before calling if one applies (oracle ``ECOALMASSM``).
    """
    return kernel * dt * COL


def g_from_f(f: jax.Array, masses: jax.Array) -> jax.Array:
    """``g(ln r) = 3 m² f(m)`` [kg m⁻³ per unit ln r] (oracle G1↔FF1R)."""
    return 3.0 * masses**2 * f


def f_from_g(g: jax.Array, masses: jax.Array) -> jax.Array:
    """Inverse of :func:`g_from_f`."""
    return g / (3.0 * masses**2)


def mass_density_from_g(g: jax.Array) -> jax.Array:
    """LWC [kg m⁻³] = ``COL · Σ_k g_k`` (oracle ``cont_init_drop``)."""
    return COL * jnp.sum(g, axis=-1)


def number_density_from_g(g: jax.Array, masses: jax.Array) -> jax.Array:
    """Number density [m⁻³] = ``COL · Σ_k g_k/m_k``."""
    return COL * jnp.sum(g / masses, axis=-1)


def bott_coalescence(
    g: jax.Array,
    ck: jax.Array,
    masses: jax.Array,
    tables: CollisionTables,
    prdkrn: float | jax.Array = 1.0,
    gmin: float = GMIN_DEFAULT,
) -> jax.Array:
    """One collision step of the Bott flux method (oracle ``coll_xxx_lwf``).

    Sequential Gauss–Seidel pass over (i ≤ j) pairs in oracle order: each
    pair removes interacting mass from bins i and j and deposits it into
    target bin k = ``ima(i,j)`` with the exponential-flux split between k
    and k+1 controlled by the Courant number. In-place statement order is
    replicated exactly (later pairs see earlier updates; ``j == i`` and
    ``k == j`` aliasing follow the oracle's sequential array reads).

    Args:
        g: spectrum ``(n_bins,)``, mass density per ln r [kg m⁻³].
        ck: pair table from :func:`collision_ck_matrix` ``(n, n)``.
        masses: bin masses [kg] ``(n,)``.
        tables: static pair tables for this grid.
        prdkrn: scalar kernel enhancement (oracle ``MODKRN_KS`` output;
            1.0 = no temperature modification).
        gmin: empty-bin threshold [kg m⁻³] (oracle ``g_lim``; safety floor,
            not a tunable).

    Returns:
        Updated spectrum ``(n_bins,)``.
    """
    x = masses

    def body(g, pair):
        i, j, k, c_ij = pair
        kp = k + 1
        gi = g[i]
        gj = g[j]
        # Oracle loop-entry skips: empty source bins do nothing.
        active = (gi > gmin) & (gj > gmin)

        x01 = ck[i, j] * gi * gj * prdkrn
        x02 = jnp.minimum(x01, gi * x[j])
        x03 = jnp.where(j != k, jnp.minimum(x02, gj * x[i]), x02)
        gsi = x03 / x[j]
        gsj = x03 / x[i]
        gsk = gsi + gsj
        # Oracle: if (gsk <= gmin) skip BEFORE touching g.
        no_change = (~active) | (gsk <= gmin)

        gi_new = jnp.maximum(gi - gsi, 0.0)
        # j may alias i (self-collection): oracle re-reads the array after
        # the i update.
        gj_base = jnp.where(j == i, gi_new, gj)
        gj_new = gj_base - gsj
        # "new change of 23.01.11": floor at 0 only when j != k (j == k may
        # go transiently negative and is overwritten by the k write).
        gj_new = jnp.where(j != k, jnp.maximum(gj_new, 0.0), gj_new)

        # gk reads the post-update array (k can alias j; k == i impossible:
        # k >= j and k == j == i would need m_2i in bin i).
        gk_base = jnp.where(k == j, gj_new, jnp.where(k == i, gi_new, g[k]))
        gk = gk_base + gsk

        # Oracle salvage branch: g(j) < 0 with an empty target → undo the j
        # update (g(j)=0) and credit only gsi to the target, skip the flux.
        salvage = (~no_change) & (gj_new < 0.0) & (gk <= gmin)
        # Oracle: if (gk <= gmin) skip flux — i and j stay modified (mass
        # gsk is dropped; bounded by gmin per pair).
        no_flux = (~no_change) & (~salvage) & (gk <= gmin)
        full = (~no_change) & (~salvage) & (gk > gmin)

        # Exponential flux split between k and k+1 (Bott 1998 form).
        gkp = g[kp]
        gk_safe = jnp.where(full, gk, 1.0)
        x1 = jnp.log(gkp / gk_safe + 1.0e-15)
        # x1 → 0 limit (g(kp) ≈ gk): the oracle's double-precision 1e-15
        # offset keeps x1 nonzero, but in float32 log(1+1e-15) IS zero →
        # 0/0 NaN (and exploding gradients near zero in any dtype). Use the
        # analytic limit flux → gsk·c there; the threshold is far above
        # f32 roundoff and far below any physically distinct x1.
        x1_near_zero = jnp.abs(x1) < 1.0e-6
        x1_safe = jnp.where(x1_near_zero, 1.0, x1)
        flux_formula = gsk / x1_safe * (
            jnp.exp(0.5 * x1_safe) - jnp.exp(x1_safe * (0.5 - c_ij)))
        flux = jnp.where(x1_near_zero, gsk * c_ij, flux_formula)
        flux = jnp.minimum(flux, gsk)
        flux = jnp.minimum(flux, gk)
        # (oracle kp_flux_max=44 halving unreachable for NKR<=43 — omitted)

        v_i = jnp.where(no_change, gi, gi_new)
        v_j = jnp.where(no_change, gj,
                        jnp.where(salvage, 0.0, gj_new))
        salvage_base = jnp.where(k == j, 0.0, gk_base)
        v_k = jnp.where(no_change, g[k],
                        jnp.where(salvage, salvage_base + gsi,
                                  jnp.where(no_flux, gk_base,
                                            jnp.maximum(gk - flux, gmin))))
        v_kp = jnp.where(full, jnp.maximum(gkp + flux, gmin), gkp)

        # Sequential writes in oracle statement order; later writes see the
        # aliasing already resolved in the v_* values.
        g = g.at[i].set(v_i)
        g = g.at[j].set(v_j)
        g = g.at[k].set(v_k)
        g = g.at[kp].set(v_kp)
        return g, None

    pairs = (tables.i_idx, tables.j_idx, tables.k_idx,
             tables.c_pair.astype(g.dtype))
    g_out, _ = jax.lax.scan(body, g, pairs)
    return g_out
