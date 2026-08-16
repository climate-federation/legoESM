"""Is the eady_uniform/mpas_channel blow-up grid-scale noise or a resolved mode?

WHY. Centring the barotropic averaging window on t+dt (a correctness fix:
external gravity waves ran ~1.9x too slow) breaks exactly one case of 65 --
eady_uniform on mpas_channel, where the free surface reaches 5192 m against a
100 m threshold. Two explanations were proposed and BOTH are refuted: the new
window damps MORE at the coupling period, and the barotropic CFL is 0.033, a
30x margin (more substeps make it worse, not better).

What is left is the SPATIAL structure of eta just before onset, which the test
matrix does not dump at that step. This script gets it. The discriminator, on
an unstructured mesh where a Fourier spectrum is awkward:

  CHEQUERBOARD INDEX  mean over edges of -sign(eta_c1 * eta_c2), i.e. how often
      neighbouring cells carry OPPOSITE signs. A 2*dx null-space mode (the
      known hexagonal C-grid divergence chequerboard) drives this toward +1.
      A mode resolved over many cells leaves it near 0 or negative.

  CONCENTRATION      fraction of wet cells carrying 10% or more of max|eta|.
      A single hot cell or a few-cell cluster is numerical; a resolved
      baroclinic mode occupies a large fraction of the channel.

Together these separate "TRiSK/Perot noise on distorted cells" from "the Eady
mode growing faster under a correctly-phased free surface". Neither number is
a verdict on its own -- they are reported with the mesh's own reference values
so a reader can see the scale.

Run AFTER building the mesh the case uses; writes a JSON record per sampled
step so the onset can be walked rather than inferred from one frame.

MEASURED 2026-08-13 on the failing case (mpas_channel, 70 km, 504 cells,
channel lat 16-34 so mid-channel is 25 deg). Sampling every 20 steps:

    step     max|eta|      chequerboard   concentration
    19300     0.880 m        -0.623          0.720
    19400     0.881 m        -0.623          0.722
    19420     0.882 m        -0.625          0.718
    19440  5470.043 m        -0.677          0.077
    19460       NaN            --              --

  controls on THIS mesh: alternating +0.340, random -0.011, constant -1.000

WHAT THAT RULES OUT.

 * NOT a grid-scale chequerboard. The index at the blow-up frame moves AWAY
   from the alternating control (-0.677 against +0.340) -- the exploding
   field is SMOOTHER than the healthy one, not rougher. The hexagonal
   divergence null-space reading is refuted.
 * NOT a domain-filling resolved mode. Concentration collapses 0.72 -> 0.077:
   39 of 504 cells.
 * NOT the existing feature amplifying. Those 39 cells have ZERO overlap with
   the 39 largest cells one sample earlier; they sat at 0.25 m mean against a
   0.88 m domain peak, then grew 11150x in 20 steps.
 * NOT degenerate geometry. Their median cell area is 0.99 of the domain
   median and their smallest is twice the domain minimum -- they are ordinary
   cells, not the small or distorted ones.
 * NOT a wall or buffer effect. They lie at lat 25.4-29.2 in a 16-34 channel,
   i.e. from mid-channel toward the northern half, with none in the outer 10%
   latitude band.

THE CONTROLLED COMPARISON, which is the finding that matters.

Same instrument, same case, same steps, only the code differs -- edge
``max|u|`` in m/s against a 0.30 m/s jet:

    step      origin/main      with the centred window
    12000        0.416              0.471
    14000        0.426              0.781
    16000        0.425              1.364
    18000        0.430             ~2.09
    19400          --              11.91
    19428          --              45.26
    48000        0.501              (dead at 19460)

origin/main drifts 0.385 -> 0.501 across the WHOLE 200-day run and never
leaves the physical range. The centred window departs from it at about step
12000-13000 and grows monotonically from there. So the fix does not expose a
pre-existing instability -- it INTRODUCES one on this arm.

VELOCITY LEADS, THE FREE SURFACE FOLLOWS. Per-step over the onset, eta is
flat to four decimals at 0.881-0.883 while max|u| goes 11.91 -> 45.26; eta
only responds at step 19427 once u is near 40 m/s. The 5192 m free surface
that trips the blow-up detector is a SYMPTOM, roughly 30 steps downstream of
the actual failure.

IT IS NOT GEOPHYSICAL. The late growth rate is 1.589e-4 /s (e-folding 1.75
h), against the case's own Eady rate of 2.3e-7 /s and the Coriolis parameter
f = 6.16e-5 /s at 25 deg:

    observed / Eady = 691x        observed / f = 2.6x

A mode growing FASTER THAN f cannot be rotationally balanced, so this is not
the Eady instability being revealed by a correctly-phased free surface. It is
numerical.

THE MECHANISM, CONFIRMED BY A ONE-VARIABLE TEST.

The old averaging window ran the barotropic mode at roughly HALF SPEED. Its
weights have their first moment at 0.533*dt (n=30), not at dt, and because the
slow baroclinic forcing is frozen across the scan and enters linearly, that
first moment IS the multiplier on the forcing. The same weights set how far eta
advances and how large the returned transport is. Three faces of one number:

                          weight centroid   forcing applied   transport
    old (half window)        0.533 dt            53 %           53 %
    new (centred window)     1.000 dt           100 %          100 %

The centred window is CORRECT and the old one was wrong. It is not a lag, it is
a halving.

THE DISCRIMINATOR. Run the new 2n-1 substep loop with the OLD weights
zero-padded past j = n. Substep trajectory, substep count, frozen-forcing
exposure, window duration, eta diffusion and divergence damping are then all
IDENTICAL to the shipped arm; only the centroid moves.

    centroid 0.53 dt  ->  PASS, max_speed 2.6820 m/s, T_drift 3.70e-12
    centroid 1.00 dt  ->  FAIL, blow-up at step 19400

The passing arm reproduces the untouched BASELINE to five digits (baseline:
2.6820 m/s, 3.71e-12), which is the internal control that the padded weights
really do restore the old behaviour. One variable, both directions.

WHAT THIS RULES OUT, each carried by the PASSING arm and therefore dead:
 * the doubled window DURATION (~2*dt against ~1*dt),
 * the doubled total eta diffusion and divergence damping that come with it,
 * the doubled frozen-forcing extrapolation horizon,
 * the filter KERNEL shape (box and cosine share the centroid exactly).

SUBSTEP COUNT DOES NOTHING, and this retires the one fact that contradicted the
mechanism. With the count overridden at the point of use and the effective value
printed, the failure lands at the SAME step every time:

    30 substeps -> step 19400 (metric 5192.4)
    60 substeps -> step 19400 (metric nan)
   120 substeps -> step 19400 (metric nan)

The earlier "more substeps is worse" reading was an artifact: the metric goes
from finite to non-finite at one detection point, while onset never moves.

A SHORTER BAROCLINIC STEP DOES NOT RESCUE IT. Halving dt moves onset from step
19400 to 43000, which is 67.4 -> 74.7 simulated days: 11 % more physical time,
not a cure. Onset is near-invariant in PHYSICAL time, so this is not something a
timestep reduction converges away.

DIVERGENCE DAMPING IS NOT THE CAUSE and is load-bearing in the other direction:
0.05 (default) fails at 19400, 0.025 goes non-finite, 0.0 goes non-finite. Less
damping is strictly worse. A predicted per-application factor of ~0.6-0.9 said
the operator is comfortably stabilising, and the sweep agrees.

THE READING. This case was stable only because the barotropic mode was being
under-forced by half. Correcting it crosses a genuine stability boundary of the
mode split. The remedy is NOT to revert the correction and NOT to shrink the
timestep; it is to close the coupling with ONE consistent average, so the
depth-mean of the corrected 3-D velocity equals the filtered barotropic
transport by construction (SM2005 / MOM6). That the earlier velocity-only
substitution merely DELAYED blow-up (19400 -> 23500) is the signature of a
partially closed loop -- it corrected the velocity while leaving eta and the
forcing at half speed.

SUPERSEDED: an earlier revision of this file called the velocity/transport
centroid offset "contributory but not sufficient". That framing is retracted.
The offset is a symptom of the halving, not a separate mechanism.

THE FREE-SURFACE SMOOTHING IS THE DOMINANT DESTABILISER, and it is enormous.

Sweeping ``barotropic_diffusion_alpha`` on this case is the largest effect
anything has had on it, cleanly monotone over four points:

    alpha 0.10   blow-up day  61.5
    alpha 0.05   blow-up day  67.4     <- what this case ships
    alpha 0.025  blow-up day  82.3
    alpha 0.00   blow-up day 122.9     <- off

Turning it off nearly doubles survival. Not a cure (the case runs 200 days),
but no other lever has moved it at all: substep count does nothing, halving the
baroclinic step buys 11%, the divergence damping runs the other way, and two
separate velocity/transport reconciliations both stall at day 82.

WHY IT IS SO STRONG. The coefficient is applied as
``nu_dt_edge = alpha * (dt_baro/dt_ref) * areaCell``, so the implied
free-surface diffusivity is ``kappa = alpha * area / dt_ref``:

    alpha 0.05, 70 km cells, dt_ref 60 s  ->  kappa = 3.5e6 m^2/s
    alpha 0.01 (the MODEL DEFAULT)        ->  kappa = 7.1e5 m^2/s

Ocean lateral tracer diffusivity is 1e2-1e3 m^2/s. This is three to four
ORDERS of magnitude larger. As a numerical filter on the free surface that is
defensible -- it damps grid-scale eta in ~140 s -- but the number should be
stated wherever the knob is tuned, because nothing in the config says it.

AND IT MOVES MORE WATER THAN THE OCEAN DOES. Differencing two solver runs that
differ ONLY in alpha, the diffusive part of the substep mass flux against the
advective part:

    alpha 0.01 (default)   diffusive flux is   9.1x the advective transport
    alpha 0.05 (this case) diffusive flux is  42.7x the advective transport

A REFUTED FIX, recorded so it is not re-attempted. The smoothing's mass flux is
NOT carried in ``Hu_avg``, so the transport that advects layer thickness and
tracers misses it. Measured violation of ``div(Hu_avg) == (eta_old-eta_avg)/dt``:
7.1e-14 with the smoothing off, but 0.970 / 0.994 / 0.997 at alpha
0.01 / 0.05 / 0.10 -- i.e. with the shipped DEFAULT on, the transport carrying
tracers misses ~97% of the mass movement the free surface saw.

Folding the diffusive flux into the transport (``transport -= diff_flux/dt_baro``)
closes the identity exactly at every alpha (to 1e-14/1e-15) and is
stability-NEUTRAL: at the shipped alpha the case still fails at the identical
step 19400. But it is the WRONG CURE, and the ratio above is why -- it would
advect tracers with a numerical filter 9x to 43x stronger than the actual
current. The identity violation is real; laundering a 3.5e6 m^2/s filter into
tracer transport is worse than leaving it out.

The coherent alternatives, neither attempted here: make the filter weak enough
that its flux is a genuine transport, or stop it changing the volume the tracer
update integrates against. Smoothing eta at this strength while withholding its
flux is incoherent either way (GLM-5.2).

CAUTION ON THE 97% FIGURE: it is normalised by the total free-surface tendency
``max|(eta_old-eta_avg)/dt|``, so it says the omitted flux is comparable to the
whole eta change -- consistent with the 9x-43x flux ratio, which is the
dimensional statement and the one to quote.

A CAUTION THAT COST ME A WRONG READING.

The barotropic loop returns three averages and they are NOT taken over the
same effective interval. Velocity uses the symmetric filter weights centred
on t+dt; transport uses the SM2005 continuity-consistent tail-sum weights,
whose centroid sits near t+0.55*dt. Weight centroids, cosine, n=30:

                  velocity      transport      offset
    before          t+dt        ~t+0.78 dt     0.22 dt
    after           t+dt        ~t+0.55 dt     0.45 dt

The centring fix corrects the VELOCITY window and leaves the transport shape
alone, so it roughly DOUBLES a pre-existing offset. Every coupling step then
resets the 3-D barotropic mode from one average while moving mass with the
other. Two independent derivations agree (codex puts the offset at 0.451 dt
for the start-of-substep flux convention, 0.435 dt at the midpoint).

THE DENOMINATOR IS NOT THE BUG, and this constrains any fix. The transport
weight keeps the PHYSICAL n_substeps deliberately: substituting n_loop makes
the transport weights sum to 30/59 and breaks
``div(Hu_avg) == (eta_old - eta_avg)/dt`` by 49%. The tail-sum SHAPE is
load-bearing, so the mismatch cannot be removed by re-weighting transport.

MEASURED CONTRIBUTION, and the reason this is not billed as the root cause:
deriving the velocity from the transport instead (u_bar_avg = Hu_avg / H_e,
one variable, same solver) moves the blow-up from step 19400 to 23500. A 21%
delay, not a cure. The mismatch is contributory and quantified; it is not
sufficient. Note that substitution also trades one inconsistency for another
-- it reintroduces a phase error in the velocity -- so it is a mechanism
probe, not a candidate fix.

TWO MORE ARMS, one informative and one void:
 * BOX kernel instead of cosine blows up at the IDENTICAL step 19400, despite
   damping 30x harder at the coupling period (0.017 against 0.500). Kernel
   damping is not the mechanism; whatever drives this is common to both, and
   the centroid mismatch is.
 * implicit_cn also fails (step 35400) but that arm is VOID as evidence:
   barotropic_implicit_mpas.py is not in the fix's file list and never calls
   compute_filter_weights, so that solver is identical on both sides and its
   failure is pre-existing and independent. It changes two things at once.

A CAUTION THAT COST ME A WRONG READING. The matrix's reported ``max_speed``
(2.68 m/s on origin/main at day 200) is the CELL-CENTRE reconstructed speed,
not edge ``max|u|``. Comparing the two suggested the baseline had comparable
velocity growth; it does not. Name the staggering and the reduction of both
sides before comparing them.

WHAT IS LEFT. A localised, smooth, explosively growing patch in the JET
region: quiescent to 5470 m in 20 baroclinic steps after 400 steps flat to
four decimal places, in ordinary cells, starting where the flow was NOT
strongest. The onset is sharp enough that the reported blow-up step moves by
one check interval between runs (19400 vs 19500), so it sits near a threshold
rather than growing through one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def chequerboard_index(eta, cells_on_edge, wet=None):
    """Mean over interior edges of -sign(eta_a * eta_b), in [-1, 1].

    +1 means every neighbouring pair has opposite signs (a 2*dx mode); 0 means
    signs are uncorrelated across edges; negative means neighbours agree, i.e.
    a smooth large-scale field.

    Edges touching a dry cell are dropped: a land neighbour holds a constant
    and would bias the index toward whatever that constant's sign is.
    """
    coe = np.asarray(cells_on_edge)
    if coe.shape[0] != 2:                       # (nEdges, 2) -> (2, nEdges)
        coe = coe.T
    a, b = coe[0], coe[1]
    ok = (a >= 0) & (b >= 0)
    if wet is not None:
        wet = np.asarray(wet, dtype=bool)
        ok &= wet[a] & wet[b]
    if not ok.any():
        return float("nan"), 0
    ea, eb = np.asarray(eta)[a[ok]], np.asarray(eta)[b[ok]]
    good = np.isfinite(ea) & np.isfinite(eb)
    if not good.any():
        return float("nan"), 0
    return float(np.mean(-np.sign(ea[good] * eb[good]))), int(good.sum())


def concentration(eta, wet=None, frac=0.1):
    """Fraction of wet cells at or above ``frac`` of max|eta|."""
    e = np.abs(np.asarray(eta, dtype=np.float64))
    if wet is not None:
        e = e[np.asarray(wet, dtype=bool)]
    e = e[np.isfinite(e)]
    if e.size == 0 or e.max() <= 0:
        return float("nan"), 0
    return float(np.mean(e >= frac * e.max())), int(e.size)


def _controls(cells_on_edge, n_cells, wet=None, seed=0):
    """The two indices on fields whose answer is KNOWN.

    Without these the numbers above are unanchored: a chequerboard index of
    0.3 means nothing until you have seen what an actual chequerboard and an
    actual smooth field score on THIS mesh.
    """
    rng = np.random.default_rng(seed)
    out = {}
    alt = rng.standard_normal(n_cells)
    alt = np.abs(alt) * np.where(np.arange(n_cells) % 2 == 0, 1.0, -1.0)
    out["synthetic_alternating"] = chequerboard_index(alt, cells_on_edge, wet)[0]
    out["synthetic_random"] = chequerboard_index(
        rng.standard_normal(n_cells), cells_on_edge, wet)[0]
    out["synthetic_constant_sign"] = chequerboard_index(
        np.abs(rng.standard_normal(n_cells)) + 1.0, cells_on_edge, wet)[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path, required=True,
                    help="eta snapshots on the NATIVE mesh, one row per "
                         "sampled step (key 'eta', shape (n_samples, nCells))")
    ap.add_argument("--mesh-npz", type=Path, required=True,
                    help="mesh arrays: cellsOnEdge, and optionally land_mask")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    z = np.load(a.npz)
    m = np.load(a.mesh_npz)
    eta = np.asarray(z["eta"], dtype=np.float64)
    if eta.ndim == 1:
        eta = eta[None, :]
    coe = m["cellsOnEdge"]
    wet = None
    if "land_mask" in m.files:
        lm = np.asarray(m["land_mask"], dtype=np.float64)
        wet = (lm[-1] if lm.ndim == 2 else lm) > 0.5
    steps = (np.asarray(z["steps"]).tolist() if "steps" in z.files
             else list(range(eta.shape[0])))

    rec = {"controls": _controls(coe, eta.shape[1], wet), "samples": []}
    for k, s in enumerate(steps[:eta.shape[0]]):
        cb, n_edges = chequerboard_index(eta[k], coe, wet)
        cc, n_cells = concentration(eta[k], wet)
        rec["samples"].append({
            "step": int(s),
            "max_abs_eta_m": float(np.nanmax(np.abs(eta[k]))),
            "chequerboard_index": cb, "edges_used": n_edges,
            "concentration_frac": cc, "cells_used": n_cells})
        print(f"  step {int(s):>7d}  max|eta| {np.nanmax(np.abs(eta[k])):10.3f} m"
              f"   chequerboard {cb:+.3f}   concentration {cc:.4f}")

    print("\n  CONTROLS on this mesh (what the index reads for known fields):")
    for k, v in rec["controls"].items():
        print(f"    {k:26s} {v:+.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2))
    print(f"\nCOMPLETED: {a.out}")


if __name__ == "__main__":
    main()
