"""Spectral remap after diffusional growth (oracle ``JERDFUN_KS``/``JERNEWF_KS``).

After one condensation substep every bin's particles have grown from
``m_k`` to ``m_k' = (m_k^{2/3} + (2/3)·B_k·∫S dt / m_k^{1/3})^{3/2}``
(exact integral of ``dm/dt = B·S`` with ``B ∝ m^{1/3}`` frozen over the
substep — oracle ``JERDFUN_KS`` drop branch). The off-grid masses are then
remapped onto the fixed grid (oracle ``JERNEWF_KS``):

1. **Kovetz–Olund 2-point split** on the packet variable ``ψ_k = f_k m_k``:
   the packet lands between grid bins I and I+1 with linear-in-mass
   weights — conserves BOTH Σψ (∝ number) and Σψ·m (∝ mass) exactly.
   Shrinking below the grid loses the sub-grid fraction (evaporation
   toward vapor); growth beyond the top lands against a sentinel at
   ``1024·m_top`` (oracle ``RRS(NRX+1)``), keeping mass in the top bin.
2. **3-point correction** (``ISIGN_3POINT = 1``, condensation only):
   anti-diffusive quadratic redistribution with the oracle's monotonicity
   ("smoothing criteria") and positivity guards; the oracle EXITs the
   whole correction loop on the first nonpositive candidate — replicated.
3. **Spurious-tail merge** (``IDROP``): inside drop bins 6..12 (1-based),
   a bin holding less than ``COEFF_REMAPING = 1/150`` of its left
   neighbour's mass content is folded into it (kills the slow large-drop
   tail created by remap diffusion).

Evaporation (``m_1' < m_1``) disables both the 3-point correction and the
tail merge (oracle ``IEvap``/``IDROP = 0``).

The oracle hard-stops on a negative remapped distribution; the port
returns diagnostics (``min_psi``) instead — the caller decides (no silent
clamp).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import COL

__physics_contract__ = {
    "summary": (
        "Condensation/evaporation bin growth (JERDFUN m^{2/3} update) and "
        "mass-grid remap (JERNEWF: Kovetz-Olund 2-point + 3-point "
        "anti-diffusive correction + drop-tail merge)."
    ),
    "inputs": {
        "f": "m^-3 kg^-1 (size distribution)",
        "masses": "kg (fixed bin grid)",
        "m_new": "kg (grown bin masses, from growth coefficients * S_int)",
    },
    "outputs": {"f_new": "m^-3 kg^-1", "min_psi": "kg m^-3 (diagnostic)"},
    "sign_convention": (
        "Condensation (m_new > masses) shifts the spectrum to larger bins; "
        "evaporation shifts it down, losing sub-grid mass to vapor. The KO "
        "split conserves spectrum number and mass exactly for in-grid "
        "remaps; the 3-point correction and tail merge redistribute but "
        "never create negative packets (oracle guards replicated)."
    ),
    "conserves": ["mass"],
    "differentiable": True,
    "reference": (
        "Kovetz & Olund (1969) JAS 26:1060; Khain et al. (2008) "
        "JAS 65:1721; WRF module_mp_fast_sbm.F JERDFUN_KS/JERNEWF_KS"
    ),
    "idealized_test": (
        "Identity when m_new == masses (exact); uniform doubling m_new = "
        "2m shifts the spectrum exactly one bin (number + mass conserved "
        "to roundoff); in-grid growth conserves Σψ and Σψ·m to 1e-12; "
        "evaporation below bin 1 loses number monotonically and never "
        "goes negative; gradients flow from f_new to S_int through the "
        "KO weights."
    ),
}

# Oracle parameters (module_mp_fast_sbm.F l. 1704-1706, 2465).
ISIGN_3POINT_DEFAULT = True
# A drop bin with < 1/150 of its left neighbour's mass content is merged
# left (oracle COEFF_REMAPING = 0.0066667).
COEFF_REMAPING = 1.0 / 150.0
# Drop-spectrum remap window, 0-based [5..11] (oracle KRDROP_REMAPING_MIN=6
# .. MAX=12, 1-based).
KRDROP_REMAP_LO = 5
KRDROP_REMAP_HI = 11
# Oracle floors negative new masses at 1e-50 g = 1e-53 kg.
_MASS_FLOOR = 1.0e-53
# Oracle exact-match shortcut |RN-RR| < 1e-16 g = 1e-19 kg.
_EXACT_MATCH_ATOL = 1.0e-19


class RemapResult(NamedTuple):
    f_new: jax.Array     # remapped distribution (n_bins,)
    min_psi: jax.Array   # min of the remapped packet variable (diagnostic;
                         # < 0 reproduces the oracle's fatal condition)


def condensation_new_masses(
    masses: jax.Array,
    growth_coefficient: jax.Array,
    s_int: jax.Array,
) -> jax.Array:
    """Grown bin masses after one substep (oracle ``JERDFUN_KS`` drop
    branch): ``m' = (m^{2/3} + (2/3)·B·S_int·m^{-1/3})^{3/2}``, floored at
    the oracle's tiny positive mass when evaporation would overshoot."""
    rate = (2.0 / 3.0) * s_int * growth_coefficient / masses ** (1.0 / 3.0)
    a = masses ** (2.0 / 3.0) + rate
    return jnp.where(a < 0.0, _MASS_FLOOR, jnp.maximum(a, 0.0) ** 1.5)


def _ko_two_point(psi_packets: jax.Array, m_new: jax.Array,
                  rrs: jax.Array, f: jax.Array, masses: jax.Array,
                  n: int) -> jax.Array:
    """Kovetz–Olund linear split of every packet ``ψ_k = f_k m_k`` at its
    new mass onto the extended grid ``rrs`` (n+1 nodes, sentinel at
    1024·m_top). Returns ``psinew`` (n+1,); order-independent scatter."""
    exact = jnp.abs(m_new - masses) < _EXACT_MATCH_ATOL
    below = m_new < rrs[0]
    # Interval I: rrs[I] <= m_new <= rrs[I+1] (oracle linear search).
    idx = jnp.clip(jnp.searchsorted(rrs, m_new, side="right") - 1, 0, n - 1)
    rr_lo = rrs[idx]
    rr_hi = rrs[idx + 1]
    w_hi = (m_new - rr_lo) / (rr_hi - rr_lo)
    w_lo = (rr_hi - m_new) / (rr_hi - rr_lo)
    # Below-grid: only the fraction m_new/rrs[0] survives into bin 0
    # (oracle GMAT2 with RRTMP=0) — the rest has evaporated off the grid.
    w_below = m_new / rrs[0]

    active = (f > 0.0) & ~exact
    p = psi_packets
    add_lo = jnp.where(active & ~below, p * w_lo, 0.0)
    add_hi = jnp.where(active & ~below, p * w_hi, 0.0)
    add_b0 = jnp.where(active & below, p * w_below, 0.0)
    add_exact = jnp.where(exact & (f > 0.0), p, 0.0)

    psinew = jnp.zeros(n + 1, dtype=psi_packets.dtype)
    psinew = psinew.at[idx].add(jnp.where(below, 0.0, add_lo))
    psinew = psinew.at[idx + 1].add(jnp.where(below, 0.0, add_hi))
    psinew = psinew.at[0].add(jnp.sum(add_b0))
    # Exact-match packets stay in place.
    psinew = psinew.at[jnp.arange(n)].add(add_exact)
    return psinew


def _three_point_correction(psi: jax.Array, psi_packets: jax.Array,
                            m_new: jax.Array, rrs: jax.Array,
                            f: jax.Array, masses: jax.Array,
                            n: int) -> jax.Array:
    """Oracle 3-point anti-diffusive pass (sequential over source bins,
    carrying the EXIT flag: the first nonpositive candidate stops ALL
    subsequent corrections, exactly like the Fortran ``EXIT``)."""
    exact = jnp.abs(m_new - masses) < _EXACT_MATCH_ATOL

    def body(carry, k):
        psi, alive = carry
        p_k = psi_packets[k]
        p_k1 = psi_packets[k + 1]
        m_n = m_new[k]
        consider = (f[k] > 0.0) & ~exact[k] & (rrs[1] < m_n)

        i = jnp.clip(jnp.searchsorted(rrs, m_n, side="right") - 1, 1, n - 2)
        rr_i, rr_p, rr_m = rrs[i], rrs[i + 1], rrs[i - 1]
        m_n2 = m_new[k + 1]
        rr_i2, rr_p2, rr_m2 = rrs[i + 1], rrs[i + 2], rrs[i]

        gn1 = (rr_p - m_n) * (rr_i - m_n) / ((rr_p - rr_m) * (rr_i - rr_m))
        gn1p = ((rr_p2 - m_n2) * (rr_i2 - m_n2)
                / ((rr_p2 - rr_m2) * (rr_i2 - rr_m2)))
        gn2 = (rr_p - m_n) * (m_n - rr_m) / ((rr_p - rr_i) * (rr_i - rr_m))
        gmat = (rr_p - m_n) / (rr_p - rr_i)
        gn3 = (rr_i - m_n) * (rr_m - m_n) / ((rr_p - rr_m) * (rr_p - rr_i))
        gmat2 = (m_n - rr_i) / (rr_p - rr_i)

        psi_im = psi[i - 1] + gn1 * p_k
        psi_i = psi[i] + gn1p * p_k1 + (gn2 - gmat) * p_k
        psi_ip = psi[i + 1] + (gn3 - gmat2) * p_k

        positive = (psi_im > 0.0) & (psi_ip > 0.0)
        # Oracle EXIT: a nonpositive candidate on a CONSIDERED bin kills
        # the rest of the pass.
        alive_next = alive & jnp.where(consider, positive, True)
        # Smoothing criteria (Fortran precedence: (A.AND.B.AND.C).OR.D),
        # only evaluated for I > 2 (1-based) i.e. i >= 2 here.
        smooth = ((psi_im > psi[i - 2]) & (psi_im < psi_i)
                  & (psi[i - 2] < psi[i])) | (psi[i - 2] >= psi[i])
        apply = consider & alive & positive & (i >= 2) & smooth

        psi = psi.at[i - 1].set(jnp.where(apply, psi_im, psi[i - 1]))
        psi = psi.at[i].set(
            jnp.where(apply, psi[i] + p_k * (gn2 - gmat), psi[i]))
        psi = psi.at[i + 1].set(jnp.where(apply, psi_ip, psi[i + 1]))
        return (psi, alive_next), None

    (psi, _), _ = jax.lax.scan(
        body, (psi, jnp.asarray(True)), jnp.arange(n - 1))
    return psi


def _drop_tail_merge(f_new: jax.Array, masses: jax.Array) -> jax.Array:
    """Oracle IDROP spurious-tail merge inside drop bins [5..11] (0-based):
    fold a bin with < COEFF_REMAPING of its left neighbour's mass content
    into that neighbour, sweeping right-to-left below the spectrum edge
    KMAX (last populated window bin). Static 7-bin window → unrolled."""
    lo, hi = KRDROP_REMAP_LO, KRDROP_REMAP_HI
    cdrop_full = 3.0 * COL * f_new * masses
    # KMAX: largest window bin with f > 0 (oracle descending search; falls
    # back to lo when the window is empty).
    kmax = lo
    for k in range(lo, hi + 1):
        kmax = jnp.where(f_new[k] > 0.0, k, kmax)
    cd = {k: cdrop_full[k] for k in range(lo, hi + 1)}
    for k in range(hi - 1, lo - 1, -1):        # K = KMAX-1 .. lo
        in_range = k <= kmax - 1
        ratio_small = (cd[k] > 0.0) & (cd[k + 1] / jnp.where(
            cd[k] > 0.0, cd[k], 1.0) < COEFF_REMAPING)
        do = in_range & ratio_small
        cd[k] = jnp.where(do, cd[k] + cd[k + 1], cd[k])
        cd[k + 1] = jnp.where(do, 0.0, cd[k + 1])
    out = f_new
    for k in range(lo, hi + 1):
        merged = cd[k] / (3.0 * COL * masses[k])
        # Oracle rewrites bins lo..KMAX only.
        out = out.at[k].set(jnp.where(k <= kmax, merged, f_new[k]))
    return out


def remap_spectrum(
    f: jax.Array,
    masses: jax.Array,
    m_new: jax.Array,
    three_point: bool = ISIGN_3POINT_DEFAULT,
    drop_tail_merge: bool = True,
) -> RemapResult:
    """Remap grown bin masses back onto the fixed grid (oracle
    ``JERNEWF_KS``). 1-D spectra; ``vmap`` over columns.

    ``three_point``/``drop_tail_merge`` are static feature gates (oracle
    ``ISIGN_3POINT``/``IDROP``); evaporation (``m_new[0] < masses[0]``)
    disables both dynamically, as in the oracle.
    """
    n = masses.shape[0]
    # Oracle pre-step: negative new mass → tiny mass, dead packet.
    dead = m_new < 0.0
    m_new = jnp.where(dead, _MASS_FLOOR, m_new)
    f = jnp.where(dead, 0.0, f)

    rrs = jnp.concatenate([masses, masses[-1:] * 1024.0])
    psi_packets = f * masses

    psinew = _ko_two_point(psi_packets, m_new, rrs, f, masses, n)
    psi = psinew[:n]

    evap = m_new[0] < masses[0]
    if three_point:
        psi_3pt = _three_point_correction(psi, psi_packets, m_new, rrs, f,
                                          masses, n)
        # Evaporation disables the correction (oracle I3POINT_CONDEVAP=0).
        psi = jnp.where(evap, psi, psi_3pt)

    # No-growth shortcut: spectra unchanged when every mass is unchanged.
    unchanged = jnp.all(jnp.abs(m_new - masses) < _EXACT_MATCH_ATOL)
    f_new = jnp.where(unchanged, f, psi / masses)

    if drop_tail_merge:
        merged = _drop_tail_merge(f_new, masses)
        # Oracle: IDROP forced 0 on evaporation.
        f_new = jnp.where(unchanged | evap, f_new, merged)

    return RemapResult(f_new=f_new, min_psi=jnp.min(f_new * masses))
