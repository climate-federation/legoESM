"""Time-averaged cloud-water budget over a pressure band, closed in the run.

WHY THIS EXISTS.  A sustained reservoir difference is a tiny residual between
large, nearly cancelling instantaneous terms, so a budget built from a single
checkpoint cannot attribute it.  Measured on the production baseline
(2026-09-23): the instantaneous band tendency at day 5 was -2.80e-01 kg/m2/day
while the true 5-day mean rate, from its own checkpoints, was -5.24e-04 -- a
factor of 535.  A 31.5 % deficit over 20 days is an imbalance of 4.2e-04
kg/m2/day, some 600x below the instantaneous terms.  Attribution therefore
needs the terms ACCUMULATED IN THE RUN across the averaging window.

WHAT IT GUARANTEES.  The accumulated terms close against the ACTUAL change in
band inventory, not merely against the instantaneous tendency:

    I(t)  = sum_band area_w * w_band * q_c * dp / g          [kg/m2]

    I(t1) - I(t0) =  (a) process terms accumulated over the steps
                   + (b) mass redistribution: the inventory change from dp and
                         from band membership, holding q_c FIXED
                   + (c) residual, which must be round-off

(b) is the piece a naive accumulator omits.  It is evaluated by re-integrating
the OLD q_c against the NEW grid, so it captures both the layer-thickness
change as surface pressure moves and the change in which layers lie in the
band.  Without it the budget appears to close only when surface pressure is
static and silently mis-attributes otherwise.

TWO EXACT CONVENTIONS, AND WHICH ONE THIS IS.  Weighting the process terms by
the NEW grid and the mass term by the OLD tracer closes exactly; so does the
mirror choice, OLD grid for the terms and NEW tracer for the mass term.  They
differ in who is charged the cross term ``dq_process * dG``: here each process
carries its own.  That is bookkeeping, not physics, and the difference is
second order per step -- but it is a CONVENTION.  On ONE synthetic window of
40 steps, with a surface pressure wandering by 60 Pa a step and a tendency
uncorrelated with it, the two conventions differed by 8.8e-4 of the process
total.  That is NOT the test's window (the test moves the surface pressure
several times harder) and it is NOT a production number: the difference is a
sum of dq*dG, so it depends on how the tendency correlates with the pressure
change, which in a real run it certainly does.  Treat 8.8e-4 as an existence
proof that the two conventions differ measurably, and measure it on the run
before reading anything into a small difference between two arms.

RANK-LOCAL UNDER MPI.  Both reductions are plain sums over the arrays handed
in.  On a sharded run they therefore give this rank's share, and a rank-local
budget closes just as well as a global one -- the closure cannot detect the
omission.  A distributed caller must reduce the closed window across ranks
(``reduce_ledger_global`` in ``process_ledger`` does this for the ledger) and
must do so for the inventory endpoints as well as the terms, or the window is
one rank's weather.  Nothing in this package does that reduction for you --
``reduce_ledger_global`` averages the columns it is handed and performs no
collective -- so the caller owns it.

WHAT THE CLOSURE DOES NOT PROVE.  The identity holds for ANY G used
consistently, so a wrong band weight, a wrong ``g`` or a wrong area
normalisation cancels exactly between the inventory, the terms and the
redistribution and leaves the residual at round-off.  The closure detects
INCONSISTENCY, not incorrectness.  The weights are validated separately, by the
range and partition-of-unity test on ``pressure_band_weight``; the area
normalisation is validated nowhere and a wrong one silently rescales every
number while still closing.  Do not read "it closes" as "the weights are right".

Band weights are FRACTIONAL (``process_ledger.pressure_band_weight``), so a
band edge need not fall on a layer interface and two arms on different vertical
grids integrate the same pressure interval.

OFF BY DEFAULT.  Nothing constructs this unless a run asks for it.
"""

from __future__ import annotations

import math

from typing import Mapping

import jax
import jax.numpy as jnp

from legoesm.diagnostics.process_ledger import pressure_band_weight


class CloudWaterBandBudget:
    """Accumulate a closing cloud-water budget over a pressure band.

    Parameters
    ----------
    p_lo_pa, p_hi_pa
        Band edges in Pa, ``p_lo_pa`` the upper (lower-pressure) edge.
    area_weights
        Per-column weights summing to 1 (cell area / total area).
    g
        Gravity, for the dp/g mass weighting.
    term_names
        The process terms supplied to :meth:`accumulate`.  Fixed at
        construction, so a term added to the model without being registered
        here surfaces as a KeyError rather than as silence.
    """

    def __init__(self, p_lo_pa: float, p_hi_pa: float, area_weights: jax.Array,
                 g: float, term_names: tuple[str, ...]):
        if p_hi_pa <= p_lo_pa:
            raise ValueError(
                f"band edges must satisfy p_lo < p_hi; got {p_lo_pa}, {p_hi_pa}")
        if not term_names:
            raise ValueError("term_names must not be empty")
        if len(set(term_names)) != len(term_names):
            raise ValueError(f"duplicate term names: {term_names}")
        self.p_lo = float(p_lo_pa)
        self.p_hi = float(p_hi_pa)
        self.area_w = jnp.asarray(area_weights)
        self.g = float(g)
        self.term_names = tuple(term_names)
        self.reset()

    @staticmethod
    def _dp(p_half):
        """Layer thickness DERIVED from the interfaces, never taken as an
        argument.  Accepting a caller's dp alongside p_half admitted a whole
        failure class: a dp inconsistent with the interfaces scales the
        inventory and the process terms by the SAME wrong factor, so the
        residual stays exactly zero and the acceptance check passes on a budget
        that is uniformly wrong (codex reproduced this with dp = 2*diff)."""
        return p_half[..., 1:] - p_half[..., :-1]

    def _inventory(self, q, p_half, w=None):
        if w is None:
            w = pressure_band_weight(p_half, self.p_lo, self.p_hi)
        return jnp.sum(self.area_w[:, None] * q * self._dp(p_half) * w) / self.g

    def reset(self) -> None:
        self.terms = {k: jnp.asarray(0.0) for k in self.term_names}
        self.mass_redistribution = jnp.asarray(0.0)
        self.elapsed_s = 0.0
        self._I0 = None
        self._open = False

    def begin_window(self, q_c, p_half) -> None:
        """Record the inventory the window's terms must account for."""
        self.reset()
        self._I0 = self._inventory(q_c, p_half)
        self._open = True

    def accumulate(self, terms: Mapping[str, jax.Array], q_c_old,
                   p_half_old, p_half_new, dt: float) -> None:
        """Add one step.

        ``terms`` are cloud-water tendencies [kg/kg/s], positive = source,
        evaluated on the ``_old`` grid.  The ``_new`` grid is the one after the
        step, so the mass-redistribution term uses the SAME q_c on both.
        """
        if not self._open:
            raise RuntimeError(
                "accumulate() before begin_window(); the opening inventory is "
                "what the closure is tested against")
        missing = set(self.term_names) - set(terms)
        if missing:
            raise KeyError(f"missing budget terms {sorted(missing)}")
        extra = set(terms) - set(self.term_names)
        if extra:
            raise KeyError(f"unregistered budget terms {sorted(extra)}")
        # The process terms are integrated against the NEW grid and the mass
        # term carries the OLD tracer.  That split is EXACT:
        #   I(t+dt) - I(t) = int (q+dq) G_new - int q G_old
        #                  = int dq G_new  +  int q (G_new - G_old)
        # with G = dp * w_band / g.  Integrating the tendency against the OLD
        # grid instead leaves a cross term int dq (G_new - G_old) unaccounted;
        # it is second order per step but accumulates, and it broke closure at
        # 9e-5 relative over 40 steps when this was first written.
        w_new = pressure_band_weight(p_half_new, self.p_lo, self.p_hi)
        aw = self.area_w[:, None]
        for k in self.term_names:
            self.terms[k] = self.terms[k] + (
                jnp.sum(aw * terms[k] * self._dp(p_half_new) * w_new)
                / self.g * dt)
        w_old = pressure_band_weight(p_half_old, self.p_lo, self.p_hi)
        self.mass_redistribution = self.mass_redistribution + (
            self._inventory(q_c_old, p_half_new, w_new)
            - self._inventory(q_c_old, p_half_old, w_old))
        self.elapsed_s = self.elapsed_s + dt

    def close_window(self, q_c, p_half, rtol: float | None = None) -> dict:
        """Close against the ACTUAL inventory change.

        The residual is the whole point: at the tolerance floor when the
        accumulation is right, a real number when a term is missing or
        mis-weighted.  Pass ``rtol`` to make a non-closing window RAISE instead
        of returning a number a caller may not inspect.
        """
        if not self._open:
            raise RuntimeError("close_window() without begin_window()")
        I1 = self._inventory(q_c, p_half)
        dI = I1 - self._I0
        total = sum(self.terms.values()) + self.mass_redistribution
        days = self.elapsed_s / 86400.0
        residual = float(dI - total)
        if rtol is not None:
            scale = float(sum(abs(v) for v in self.terms.values())
                          + abs(self.mass_redistribution))
            # NaN > tol is FALSE, so a poisoned window would sail through an
            # ordinary tolerance test.  Non-finite is checked FIRST and always.
            if not math.isfinite(residual) or not math.isfinite(scale):
                raise ValueError(
                    "cloud-water band budget is not finite: residual "
                    f"{residual}, term scale {scale}.  A column's interfaces "
                    "do not increase downward, or an input carried NaN.")
            if abs(residual) > rtol * max(scale, 1e-300):
                raise ValueError(
                    f"cloud-water band budget did not close: residual "
                    f"{residual:.6e} exceeds {rtol:g} of the term "
                    f"scale {scale:.6e}.  A term is missing, mis-weighted, or "
                    f"the arrays are rank-local rather than global.")
        out = {
            "window_days": days,
            "inventory_start": self._I0,
            "inventory_end": I1,
            "inventory_change": dI,
            "mass_redistribution": self.mass_redistribution,
            "sum_of_terms": total,
            "residual": dI - total,
            "terms": dict(self.terms),
            "term_rates_per_day": {
                k: (v / days if days > 0 else jnp.nan)
                for k, v in self.terms.items()},
            "mean_rate_per_day": (dI / days if days > 0 else jnp.nan),
        }
        self._open = False
        return out

    def get_state(self) -> dict:
        """Sidecar state, matching the CMOR accumulator persistence pattern."""
        st = {f"term.{k}": jnp.asarray(v) for k, v in self.terms.items()}
        st["mass_redistribution"] = jnp.asarray(self.mass_redistribution)
        st["elapsed_s"] = jnp.asarray(self.elapsed_s)
        if self._I0 is not None:
            st["inventory_start"] = jnp.asarray(self._I0)
        return st

    def set_state(self, state: Mapping[str, jax.Array]) -> None:
        # A truncated sidecar must NOT leave terms silently at zero: that
        # surfaces only as a large residual after the window closes, long after
        # the cause. The term set is fixed at construction, so absence is a
        # real error (GLM review, 2026-09-23).
        absent = [k for k in self.term_names if f"term.{k}" not in state]
        if absent:
            raise KeyError(
                f"sidecar is missing accumulated terms {sorted(absent)}; "
                f"restoring would silently zero them and the window would "
                f"close with an unexplained residual")
        for k in self.term_names:
            self.terms[k] = jnp.asarray(state[f"term.{k}"])
        if "mass_redistribution" in state:
            self.mass_redistribution = jnp.asarray(state["mass_redistribution"])
        if "elapsed_s" in state:
            self.elapsed_s = float(state["elapsed_s"])
        if "inventory_start" in state:
            self._I0 = jnp.asarray(state["inventory_start"])
            self._open = True
