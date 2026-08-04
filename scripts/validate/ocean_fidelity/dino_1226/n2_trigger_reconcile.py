#!/usr/bin/env python
"""#1226 RECONCILE: does the "adiabatic" N2 no-op-equivalence claim (dino.py
~982-991, "corr 1.0000, maxdiff ~9e-7 -> ~0.1% flip") survive stratification
by |N2|, or is it a global average masking a local defect that explains the
recorded 28-40% false-convection-trigger population (2026-07-25 finding)?

Both branches are production code (packages/ocean/legoesm/ocean/eos.py):
  N2_adia = compute_buoyancy_frequency_adiabatic  (parcel displacement,
            differences two ~1027 kg/m3 numbers -> catastrophic cancellation
            when the true signal is O(1e-8))
  N2_bn2  = compute_buoyancy_frequency_nemo_bn2   (NEMO eosbn2.F90 bn2_t,
            linearized alpha/beta analytic derivative, no cancellation)

ONE state (T,S from the NEMO year-5 restart tn/sn), ONE geometry (the
bridged z_coord/jacobian, LEGOESM_NEMO_E3T=both), ONE EOS (nemo_seos, the
DINO card's own EOS) feed BOTH N2 branches -- the only thing that differs
between them is the N2 formula itself, matching production's own calling
convention (enhanced_diffusion.integration.py: adiabatic uses
compute_ocean_rho_and_pressure's "insitu" p_cell + the recipe eos_fn; nemo_bn2
uses eos.nemo_bn2_live_ladders(z_coord, eta, H_bathy) -- both wired here
exactly as make_convection_physics/enhanced_diffusion_convection call them).

A SEPARATE, clearly-labeled cross-check loads the restart's own rn2_stg
(NEMO's actual dumped N2, MY_SRC/trddump.F90:114 `rn2_stg = rn2` at the *same*
Kmm stage as tn_stg/sn_stg -- NOT tn/sn, which are a different, later time
level: tn vs tn_stg differ by max|d|=0.27 degC on this restart, so pairing
rn2_stg with tn/sn would silently substitute |T_now-T_stg| for "error", the
exact time-level trap this campaign's own ocean.fidelity.time_levels module
exists to catch). This checks NEMO's own bn2 dump reproduces from tn_stg/sn_stg
via legoESM's nemo_bn2 formula (a from-NEMO-fields self-consistency check, not
part of the main 5-section reconciliation, which stays on the ONE bridged
state per the rules above).

TWO PASSES. Pass 1 is fp64 with the precision gate ENFORCED (the sound
measurement). Pass 2 is a DELIBERATE float32 NEGATIVE CONTROL testing whether
the recorded 2026-07-25 finding predates (and is explained by) the 2026-07-28
precision gate. Findings, both from the same state, one variable = precision:

  * fp64: the two N2 forms agree almost everywhere (global flip 0.006%, 0.06%
    in the southern box x 200-1000 m). The adia-bn2 difference SCALES with
    |N2| (1e-13 -> 1.5e-7 across ascending |N2| bins) => it is a physical
    nonlinear-vs-linearized-EOS difference, NOT an additive roundoff floor.
    The subtraction consumes ~4.8 of fp64's ~16 digits: no catastrophic
    cancellation at fp64.
  * fp32: the flip rate EXPLODES (13.3% global, 59.4% southern x 200-1000 m)
    and med|adia-bn2| stops scaling, flattening to a ~1e-10..1.6e-9 floor in
    the weak bins. Mechanism, printed by section (5): the density difference
    in near-neutral southern water rounds to EXACTLY 0.0 kg/m3 in f32, so
    N2_adia == 0, which never satisfies N2 < -1e-12.
  * DIRECTION (section 1/2, ``flip_direction``): the fp32 flips are 100%
    MISSED convection (bn2 unstable, adia silent) and 0% SPURIOUS. So fp32
    makes the adiabatic trigger fire essentially NOWHERE -- the OPPOSITE sign
    from a "fires on stable water / spurious convection" report.
  * Section (6) instead characterizes the fp64 firing population: in the
    southern box 67.3% of interfaces fire, and 100% of those are MARGINAL
    (-1e-8..-1e-12) near-neutral water with ZERO strongly-unstable cells --
    and both N2 forms fire on the same cells. A large firing population on
    near-neutral water is therefore a real, fp64-robust, THRESHOLD-placement
    property of this state, not an N2-formula or a precision artifact.

Usage
-----
    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" \\
        LEGOESM_NEMO_E3T=both JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/n2_trigger_reconcile.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import netCDF4 as nc

from legoesm import constants
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    compute_buoyancy_frequency_adiabatic,
    compute_buoyancy_frequency_nemo_bn2,
    compute_ocean_rho_and_pressure,
    make_eos_fn,
    nemo_bn2_live_ladders,
)
from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.precision_gate import require_explicit_e3t_mode, require_fp64
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _nemo_native_active_3d,
)
from legoesm.ocean.vertical import compute_ocean_jacobian

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN_TRAJ = f"{DINO}/RUN_TRAJ"
REF_RESTART = f"{DINO}/RUN_Y5_REBUILD/DINO_00057600_restart.nc"

# Canonical southern boxes, verbatim from abyssal_densification.py (do not
# re-derive) -- itself sourced from southern_heat_budget.py / southern_convection_final.py.
SEDGE = slice(12, 17)
CORE = slice(1, 11)

THR = -1e-12  # nemo_dino_kamm's convection_n2_threshold (dino.py:994)
DTDZ_DEAD_TOL = 1e-5  # Claim B's "dead gradient" tolerance


def llz(a):
    """(lev, y, x) -> (y, x, lev): the legoESM interior field layout at
    nn_hls=0 (haloless NEMO 5 files); equivalent to nemo_io._to_latlon_lev at
    h=0, replicated inline per abyssal_densification.py's own `llz` (private
    symbol, not importable across modules)."""
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def flip_rate(n2_a, n2_b, wet, thr=THR):
    """Fraction of wet interfaces where (n2_a<thr) != (n2_b<thr)."""
    m = wet
    trig_a = n2_a < thr
    trig_b = n2_b < thr
    flip = (trig_a != trig_b) & m
    return float(flip.sum()) / max(int(m.sum()), 1), int(flip.sum()), int(m.sum())


def flip_direction(n2_a, n2_b, wet, thr=THR):
    """Split the flips by DIRECTION, taking ``n2_b`` (nemo_bn2) as reference.

    Claim B's assertion is directional -- the trigger firing on water that is
    GENUINELY STABLE -- so a matching flip MAGNITUDE is not yet a matching
    finding. Returns ``(n_spurious, n_missed)``:
      spurious = adia fires where bn2 says stable  (over-firing, Claim B's sign)
      missed   = bn2 says unstable, adia does not  (under-firing, opposite sign)
    """
    trig_a = (n2_a < thr) & wet
    trig_b = (n2_b < thr) & wet
    return int((trig_a & ~trig_b).sum()), int((trig_b & ~trig_a).sum())


def build_state(fp64: bool = True):
    """Bridge the NEMO year-5 restart into a legoESM state on the NEMO true
    e3t ladder (LEGOESM_NEMO_E3T=both). Mirrors eos_rab_bn2_per_element.py /
    abyssal_densification.py's bridge usage -- no new numerics.

    ``fp64=False`` is the DELIBERATE NEGATIVE CONTROL pass: it sets the global
    precision policy to float32, which is what legoESM's constructors default
    to and what silently contaminated this campaign before the 2026-07-28
    precision gate landed. It exists ONLY to test whether the recorded
    2026-07-25 "28-40% false trigger" finding is an fp32 artifact. It does NOT
    call ``require_fp64`` -- not because the gate is wrong, but because the
    gate is RIGHT and this pass is intentionally unsound by construction. The
    gate is neither weakened, patched nor deleted; the fp64 pass still calls
    it, and the printed banner marks every number from this pass as
    artifact-reproduction, not physics.
    """
    e3t_mode = require_explicit_e3t_mode(context="n2_trigger_reconcile")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be explicit; 'both' = NEMO's "
          "true 3-D ladder, what a fidelity comparison wants)")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    # NOTE: JAX_ENABLE_X64=1 only PERMITS f64; the POLICY is what the
    # constructors actually cast to. That is the trap this campaign kept
    # hitting, so the policy is set explicitly on both passes.
    set_policy(PrecisionPolicy.fp64() if fp64 else PrecisionPolicy.fp32())

    g = read_nemo_mesh_mask(f"{RUN_TRAJ}/mesh_mask.nc", nn_hls=0)
    s = read_nemo_restart(REF_RESTART, nn_hls=0)
    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    assert cfg.eos == "nemo_seos", f"expected nemo_dino_kamm eos='nemo_seos', got {cfg.eos!r}"
    assert cfg.convection_n2_mode == "adiabatic", (
        f"expected the card's own N2 mode 'adiabatic', got {cfg.convection_n2_mode!r}")
    assert cfg.convection_n2_threshold == THR, (
        f"THR={THR} must match the card's convection_n2_threshold="
        f"{cfg.convection_n2_threshold}")

    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True, omega=cfg.omega)
    z_coord = br.z_coord
    state = br.state

    if not fp64:
        # The POLICY alone is not enough to get an f32 pass: the bridge reads
        # NEMO's restart as numpy float64 and (with JAX_ENABLE_X64=1) those
        # leaves stay f64 through jnp.asarray, so only the z_coord ladder
        # actually turned f32 -- measured on the first run of this probe, whose
        # "f32" pass reproduced the f64 numbers to 4 digits because it WAS
        # still f64. Cast every inexact leaf explicitly so the arithmetic
        # really runs at f32; ints/bools (level indices, masks) are left alone.
        def _to_f32(x):
            arr = jnp.asarray(x)
            return arr.astype(jnp.float32) if jnp.issubdtype(arr.dtype, jnp.inexact) else x
        state = jax.tree_util.tree_map(_to_f32, state)
        z_coord = jax.tree_util.tree_map(_to_f32, z_coord)
    T = jnp.asarray(state.T.data)
    S = jnp.asarray(state.S.data)
    eta = jnp.asarray(state.eta.data)
    H_bathy = jnp.asarray(state.H_bathy.data)
    mask2d = np.asarray(state.land_mask.data) > 0.5

    if fp64:
        require_fp64(z_coord, T, S, eta, H_bathy, context="n2_trigger_reconcile")
    else:
        print("  [f32 pass] require_fp64 DELIBERATELY NOT CALLED -- see "
              "build_state docstring. Realized dtypes: "
              f"T={T.dtype} S={S.dtype} eta={eta.dtype} "
              f"z_coord.dz_ref={jnp.asarray(z_coord.dz_ref).dtype}")
        # A vacuously-f64 "f32 pass" is worse than no pass: it silently
        # reproduces the f64 answer and looks like a refutation of the fp32
        # hypothesis. That is exactly what the first run of this probe did.
        for nm, arr in (("T", T), ("S", S), ("eta", eta),
                        ("z_coord.dz_ref", jnp.asarray(z_coord.dz_ref))):
            if arr.dtype != jnp.float32:
                raise SystemExit(
                    f"f32 NEGATIVE CONTROL IS VACUOUS: {nm} is {arr.dtype}, not "
                    "float32 -- the pass would silently re-run fp64 arithmetic "
                    "and any 'fp32 makes no difference' conclusion from it "
                    "would be an artifact of the harness, not a measurement.")

    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)

    # --- N2_adia branch: EXACTLY enhanced_diffusion.integration.py's calling
    # convention for cfg.n2_mode == "adiabatic" ---
    eos_fn = make_eos_fn(cfg.eos, rho0=NemoSEOSConfig().rho0)
    rho, p_cell = compute_ocean_rho_and_pressure(state, z_coord, J, eos_fn=eos_fn)
    n2_adia = np.asarray(compute_buoyancy_frequency_adiabatic(
        T, S, p_cell, z_coord.dz_ref, J, eos_fn=eos_fn))

    # --- N2_bn2 branch: EXACTLY enhanced_diffusion_convection's calling
    # convention for cfg.n2_mode == "nemo_bn2" (live gdept(Kmm) ladder) ---
    gdept, gdepw_int = nemo_bn2_live_ladders(z_coord, eta, H_bathy)
    n2_bn2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(T, S, gdept, gdepw_int))

    if n2_adia.shape != n2_bn2.shape:
        raise SystemExit(f"SHAPE MISMATCH: n2_adia {n2_adia.shape} vs n2_bn2 {n2_bn2.shape}")

    # 3-D per-cell wet mask: the exact per-column bottom-level index compare
    # (z_coord.is_active), NOT a re-derived float depth compare -- same
    # production helper eos_rab_bn2_per_element.py uses (#1226 finding: the
    # float form mis-masks ~1861/10348 columns at exact-tie bottom levels).
    active_3d = np.asarray(_nemo_native_active_3d(
        jnp.asarray(mask2d), z_coord, H_bathy, T.dtype)) > 0.5
    wet_iface = active_3d[..., :-1] & active_3d[..., 1:]

    dT = np.asarray(T)[..., :-1] - np.asarray(T)[..., 1:]

    return dict(
        T=np.asarray(T), S=np.asarray(S), n2_adia=n2_adia, n2_bn2=n2_bn2,
        wet_iface=wet_iface, dT=dT, gdept=np.asarray(gdept),
        gdepw_int=np.asarray(gdepw_int),
        p_cell=np.asarray(p_cell), rho=np.asarray(rho), eos_fn=eos_fn,
        z_coord=z_coord,
    )


def section_global(st):
    print("\n" + "=" * 88)
    print("(1) GLOBAL, all wet interfaces")
    print("=" * 88)
    a, b, w = st["n2_adia"], st["n2_bn2"], st["wet_iface"]
    aw, bw = a[w], b[w]
    corr = float(np.corrcoef(aw, bw)[0, 1])
    maxdiff = float(np.max(np.abs(aw - bw)))
    rate, n_flip, n_tot = flip_rate(a, b, w)
    print(f"  n_wet_interfaces={n_tot}  corr(N2_adia,N2_bn2)={corr:.6f}  "
          f"maxdiff={maxdiff:.6e}  flip_rate={100*rate:.4f}%  ({n_flip}/{n_tot})")
    n_spur, n_miss = flip_direction(a, b, w)
    print(f"  flip DIRECTION (bn2 = reference): spurious(adia fires, bn2 stable)="
          f"{n_spur} ({100*n_spur/max(n_tot,1):.4f}%)  "
          f"missed(bn2 unstable, adia does not)={n_miss} "
          f"({100*n_miss/max(n_tot,1):.4f}%)")
    print(f"  claim-A check: flip_rate {'MATCHES' if rate < 0.005 else 'DOES NOT MATCH'} "
          f"the recorded ~0.1% ({100*rate:.4f}% vs 0.1%)")
    return dict(corr=corr, maxdiff=maxdiff, rate=rate, n_flip=n_flip, n_tot=n_tot,
                n_spurious=n_spur, n_missed=n_miss)


def section_boxes(st):
    print("\n" + "=" * 88)
    print("(2) RESTRICTED: southern box x depth 200-1000m")
    print("=" * 88)
    a, b, w = st["n2_adia"], st["n2_bn2"], st["wet_iface"]
    gdept = st["gdept"]
    gdept_bcast = np.broadcast_to(gdept, a.shape) if gdept.ndim < a.ndim else gdept
    depth_band = (gdept_bcast[..., :-1] >= 200.0) & (gdept_bcast[..., :-1] < 1000.0)

    def row_mask(rows):
        m = np.zeros(a.shape[:2], dtype=bool)
        m[rows] = True
        return m[..., None]

    combos = {
        "southern box (SEDGE+CORE rows), all depths": row_mask(
            slice(CORE.start, SEDGE.stop)) & w,
        "200-1000m, all rows": depth_band & w,
        "southern box AND 200-1000m": row_mask(slice(CORE.start, SEDGE.stop)) & depth_band & w,
    }
    out = {}
    for name, m in combos.items():
        m = m & np.ones_like(w)  # broadcast to full shape
        n_tot = int(m.sum())
        if n_tot == 0:
            print(f"  [{name}] EMPTY selection -- skipped")
            continue
        aw, bw = a[m], b[m]
        corr = float(np.corrcoef(aw, bw)[0, 1]) if aw.std() > 0 and bw.std() > 0 else float("nan")
        maxdiff = float(np.max(np.abs(aw - bw)))
        rate, n_flip, _ = flip_rate(a, b, m)
        n_spur, n_miss = flip_direction(a, b, m)
        print(f"  [{name}] n={n_tot}  corr={corr:.6f}  maxdiff={maxdiff:.6e}  "
              f"flip_rate={100*rate:.4f}%  ({n_flip}/{n_tot})  "
              f"spurious={100*n_spur/n_tot:.3f}%  missed={100*n_miss/n_tot:.3f}%")
        out[name] = dict(corr=corr, maxdiff=maxdiff, rate=rate, n_flip=n_flip,
                         n_tot=n_tot, n_spurious=n_spur, n_missed=n_miss)
    return out


def section_binned(st):
    print("\n" + "=" * 88)
    print("(3) BINNED BY |N2_bn2| (does flip_rate climb as stratification -> 0?)")
    print("=" * 88)
    a, b, w = st["n2_adia"], st["n2_bn2"], st["wet_iface"]
    absb = np.abs(b)
    edges = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, np.inf]
    labels = ["<1e-9", "1e-9..1e-8", "1e-8..1e-7", "1e-7..1e-6", "1e-6..1e-5", ">1e-5"]
    print(f"  {'bin':<14}{'count':>10}{'flip_rate%':>13}{'median|adia-bn2|':>20}")
    total_binned = 0
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = w & (absb >= lo) & (absb < hi)
        n = int(m.sum())
        total_binned += n
        if n == 0:
            print(f"  {lab:<14}{0:>10}{'--':>13}{'--':>20}")
            rows.append((lab, 0, float("nan"), float("nan")))
            continue
        rate, n_flip, _ = flip_rate(a, b, m)
        meddiff = float(np.median(np.abs(a[m] - b[m])))
        print(f"  {lab:<14}{n:>10}{100*rate:>12.3f}%{meddiff:>20.6e}")
        rows.append((lab, n, rate, meddiff))
    n_wet_total = int(w.sum())
    assert total_binned == n_wet_total, (
        f"SELF-CHECK FAILED: binned counts sum to {total_binned}, "
        f"expected total wet interfaces {n_wet_total}")
    print(f"  [self-check] bin counts sum to total wet interfaces: "
          f"{total_binned} == {n_wet_total}  OK")
    return rows


def section_dead_gradient(st):
    print("\n" + "=" * 88)
    print("(4) DEAD-GRADIENT LINK, per depth level, southern box "
          f"(rows {CORE.start}:{SEDGE.stop})")
    print("=" * 88)
    a, b, w, dT = st["n2_adia"], st["n2_bn2"], st["wet_iface"], st["dT"]
    rows = slice(CORE.start, SEDGE.stop)
    a_s, b_s, w_s, dT_s = a[rows], b[rows], w[rows], dT[rows]
    nk = a_s.shape[-1]
    print(f"  {'level':<8}{'n_wet':<9}{'frac|dT/dz|<1e-5':<20}{'frac(adia<thr)':<18}{'frac(bn2<thr)':<15}")
    fdead_all, fadia_all = [], []
    for k in range(nk):
        mk = w_s[..., k]
        n = int(mk.sum())
        if n == 0:
            print(f"  {k:<8}{0:<9}{'--':<20}{'--':<18}{'--':<15}")
            continue
        f_dead = float((np.abs(dT_s[..., k][mk]) < DTDZ_DEAD_TOL).mean())
        f_adia = float((a_s[..., k][mk] < THR).mean())
        f_bn2 = float((b_s[..., k][mk] < THR).mean())
        fdead_all.append(f_dead)
        fadia_all.append(f_adia)
        print(f"  {k:<8}{n:<9}{f_dead:<20.3f}{f_adia:<18.3f}{f_bn2:<15.3f}")
    if fdead_all:
        track = float(np.mean(np.abs(np.array(fdead_all) - np.array(fadia_all))))
        print(f"\n  mean |frac_dead - frac(adia<thr)| across levels = {track:.4f} "
              f"(claim B: 'track to 3 decimals' means this should be <~5e-4)")
        print(f"  {'TRACKS' if track < 5e-4 else 'DOES NOT TRACK (to 3 decimals)'} "
              "claim B's dead-gradient <-> adiabatic-trigger coincidence")


def section_noise_floor(st):
    print("\n" + "=" * 88)
    print("(5) THE CANCELLATION NOISE FLOOR ITSELF")
    print("=" * 88)
    # compute_buoyancy_frequency_adiabatic differences rho(T[k+1],S[k+1],p[k])
    # against rho(T[k],S[k],p[k]) = the upper cell's own in-situ rho (already
    # computed in build_state as st["rho"]). Recompute the displaced lower
    # value explicitly here (same eos_fn/p_cell as the production branch) so
    # the two operands being subtracted are visible, not just their difference.
    T, S = jnp.asarray(st["T"]), jnp.asarray(st["S"])
    p_cell = jnp.asarray(st["p_cell"])
    eos_fn = st["eos_fn"]
    rho_upper_at_p_upper = np.asarray(eos_fn(T[..., :-1], S[..., :-1], p_cell[..., :-1]))
    rho_lower_at_p_upper = np.asarray(eos_fn(T[..., 1:], S[..., 1:], p_cell[..., :-1]))
    diff = rho_lower_at_p_upper - rho_upper_at_p_upper

    w = st["wet_iface"]
    print(f"  operands being differenced (rho_lower_at_p_upper - rho_upper_at_p_upper):")
    print(f"    |rho| magnitude over wet interfaces: mean={np.mean(np.abs(rho_upper_at_p_upper[w])):.4f} kg/m^3")
    print(f"    |difference| over wet interfaces: median={np.median(np.abs(diff[w])):.4e} "
          f"mean={np.mean(np.abs(diff[w])):.4e} kg/m^3")
    # digits lost: log10(|rho| / |diff|)
    med_rho = float(np.median(np.abs(rho_upper_at_p_upper[w])))
    med_diff = float(np.median(np.abs(diff[w])))
    digits_lost = np.log10(med_rho / max(med_diff, 1e-300))
    print(f"    ~{digits_lost:.1f} decimal digits of the {med_rho:.1f} kg/m^3 magnitude "
          f"consumed by the subtraction (median case)")

    rows = slice(CORE.start, SEDGE.stop)
    gdept = np.broadcast_to(st["gdept"], st["n2_adia"].shape) if st["gdept"].ndim < st["n2_adia"].ndim else st["gdept"]
    band = (gdept[..., :-1] >= 200.0) & (gdept[..., :-1] < 1000.0)
    sel = np.zeros(w.shape, dtype=bool)
    sel[rows] = True
    sel = sel & band & w
    n_sel = int(sel.sum())
    if n_sel == 0:
        print("  southern 200-1000m band is EMPTY at this state -- cannot compare noise floor there.")
        return dict(digits_lost=float(digits_lost), ratio=float("nan"))
    noise_med = float(np.median(np.abs(diff[sel])))
    n2bn2_med_abs = float(np.median(np.abs(st["n2_bn2"][sel])))
    # Convert the density-difference noise into an N2-equivalent using the
    # SAME g/rho_ref/dz_interface arithmetic compute_buoyancy_frequency_adiabatic
    # itself uses (rho_ref default), so it is comparable to N2_bn2 directly.
    print(f"  southern-box 200-1000m band: n={n_sel} interfaces")
    print(f"    median |rho difference| there = {noise_med:.4e} kg/m^3")
    print(f"    median |N2_bn2| there = {n2bn2_med_abs:.4e} 1/s^2 "
          f"(true stratification signal)")
    n2_noise_est = float(np.median(np.abs(st["n2_adia"][sel] - st["n2_bn2"][sel])))
    print(f"    median |N2_adia - N2_bn2| there (noise, N2 units) = {n2_noise_est:.4e} 1/s^2")
    ratio = n2_noise_est / max(n2bn2_med_abs, 1e-300)
    print(f"    noise/signal ratio (N2 units) = {ratio:.2f}x")
    # Three bands, not two: at ratio ~= 1 the noise is the same SIZE as the
    # signal, which is already fatal for a SIGN test (the trigger is a sign
    # test), so "smaller than the signal" would be a misleading label there.
    if ratio > 0.5:
        print(f"  -> the noise is the SAME ORDER as the true stratification signal "
              f"({ratio:.2f}x): for a SIGN test (which the N2<thr trigger is) this "
              "means the adiabatic form is NOT reliably resolving stratification here.")
    elif ratio > 0.05:
        print(f"  -> the noise is a significant FRACTION of the signal ({ratio:.2f}x) -- "
              "marginal interfaces near the threshold are not reliably signed.")
    else:
        print(f"  -> the noise is well below the true signal ({ratio:.2f}x): "
              "the adiabatic form IS resolving stratification in this band.")
    return dict(digits_lost=float(digits_lost), ratio=float(ratio),
                n2bn2_med_abs=n2bn2_med_abs, n2_noise_est=n2_noise_est)


def section_n2_classes(st):
    """(6) CHARACTERIZE the trigger population: per depth level in the southern
    box, what FRACTION of interfaces sit in each stability class of N2_bn2?

    Purpose is descriptive only. It answers "is the 38-67% firing population
    dominated by STRONGLY unstable water (real convection) or by
    MARGINAL/near-neutral water (a threshold-sensitivity question)?" -- it does
    NOT judge whether firing is correct. The threshold question (how NEMO's
    rn_evd trigger and its MIN(rn2,rn2b) two-level hysteresis interact with
    near-neutral water) can only be posed properly once this population is
    known.
    """
    print("\n" + "=" * 88)
    print("(6) TRIGGER-POPULATION CHARACTER, southern box "
          f"(rows {CORE.start}:{SEDGE.stop}), classes of N2_bn2 [1/s^2]")
    print("=" * 88)
    b, w = st["n2_bn2"], st["wet_iface"]
    rows = slice(CORE.start, SEDGE.stop)
    b_s, w_s = b[rows], w[rows]
    # Classes span the real line with no gaps/overlaps; THR = -1e-12 sits on
    # the marginal/sub-threshold boundary by construction.
    classes = [
        ("strong_unstable", -np.inf, -1e-8),
        ("marginal", -1e-8, -1e-12),
        ("subthr_neutral", -1e-12, 1e-12),
        ("weak_stable", 1e-12, 1e-8),
        ("stratified", 1e-8, np.inf),
    ]
    hdr = "".join(f"{c[0]:>17}" for c in classes)
    print(f"  {'level':<7}{'n_wet':<8}{'frac(fire)':<12}{hdr}")
    agg = {c[0]: 0 for c in classes}
    n_fire_tot = 0
    n_tot = 0
    for k in range(b_s.shape[-1]):
        mk = w_s[..., k]
        n = int(mk.sum())
        if n == 0:
            continue
        vals = b_s[..., k][mk]
        f_fire = float((vals < THR).mean())
        cells = []
        for name, lo, hi in classes:
            cnt = int(((vals >= lo) & (vals < hi)).sum())
            agg[name] += cnt
            cells.append(cnt / n)
        n_fire_tot += int((vals < THR).sum())
        n_tot += n
        print(f"  {k:<7}{n:<8}{f_fire:<12.3f}" + "".join(f"{c:>17.3f}" for c in cells))

    print(f"\n  BOX TOTALS over {n_tot} wet interfaces "
          f"(frac firing = {n_fire_tot / max(n_tot,1):.3f}):")
    for name, _, _ in classes:
        print(f"    {name:<18} {agg[name]:>9}  ({agg[name] / max(n_tot,1):.3f})")
    # Of the FIRING population specifically, what share is strongly unstable?
    n_strong = agg["strong_unstable"]
    n_marg = agg["marginal"]
    n_fire_classes = n_strong + n_marg   # both are < THR by construction
    assert n_fire_classes == n_fire_tot, (
        f"SELF-CHECK FAILED: strong+marginal ({n_fire_classes}) must equal the "
        f"count below THR ({n_fire_tot}) -- class edges and THR are inconsistent")
    print(f"  [self-check] strong_unstable + marginal == count(N2_bn2 < THR): "
          f"{n_fire_classes} == {n_fire_tot}  OK")
    if n_fire_tot:
        share_strong = n_strong / n_fire_tot
        print(f"\n  Of the {n_fire_tot} FIRING interfaces, {share_strong:.1%} are "
              f"strongly unstable (< -1e-8) and {1 - share_strong:.1%} are marginal "
              f"(-1e-8 .. -1e-12).")
        print("  CHARACTER: the firing population is dominated by "
              + ("STRONGLY UNSTABLE water -- a deep-convecting region, so firing is "
                 "what a convection trigger is for."
                 if share_strong > 0.5 else
                 "MARGINAL / near-neutral water -- so how many cells fire is set by "
                 "where the threshold sits, making this a THRESHOLD-SENSITIVITY "
                 "question (rn_evd value, MIN(rn2,rn2b) two-level hysteresis), not a "
                 "formula-accuracy one."))
        print("  (Descriptive only -- this section does NOT judge whether firing "
              "on these cells is correct.)")
    return dict(agg=agg, n_tot=n_tot, n_fire=n_fire_tot)


def section_nemo_own_rn2(st, gdept, gdepw_int):
    """SEPARATE cross-check: does legoESM's nemo_bn2 formula reproduce NEMO's
    OWN dumped rn2_stg, fed NEMO's OWN tn_stg/sn_stg (the matched Kmm-stage
    triplet MY_SRC/trddump.F90:105-114 writes together)?  This is a
    self-consistency check on the nemo_bn2 formula against a REAL NEMO ground
    truth -- separate from the main 5-section adia-vs-bn2 reconciliation
    above, which by design stays on ONE bridged state.

    tn_stg/sn_stg/rn2_stg must NOT be paired with tn/sn (a different, later
    time level on this restart -- max|tn-tn_stg|=0.27 degC): see module
    docstring / ocean.fidelity.time_levels for why that substitution is the
    campaign's own documented trap.
    """
    print("\n" + "=" * 88)
    print("(X) CROSS-CHECK: legoESM nemo_bn2(tn_stg,sn_stg) vs NEMO's own rn2_stg dump")
    print("=" * 88)
    d = nc.Dataset(REF_RESTART)
    if "rn2_stg" not in d.variables or "tn_stg" not in d.variables:
        print("  SKIPPED: this restart does not carry rn2_stg/tn_stg/sn_stg.")
        return
    tn_stg = llz(d["tn_stg"][0])
    sn_stg = llz(d["sn_stg"][0])
    rn2_stg = llz(d["rn2_stg"][0])  # w-point indexed, level 0 = surface pad (0.0)
    print(f"  loaded tn_stg/sn_stg/rn2_stg, shapes {tn_stg.shape}")

    T_stg = jnp.asarray(tn_stg, dtype=jnp.float64)
    S_stg = jnp.asarray(sn_stg, dtype=jnp.float64)
    n2_from_stg = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        T_stg, S_stg, jnp.asarray(gdept), jnp.asarray(gdepw_int)))
    nk = n2_from_stg.shape[-1]
    # legoESM interior interface i == NEMO w-level index i+1 (matches the
    # tke_dump_rn2b indexing convention documented in eos_rab_bn2_per_element.py).
    rn2_stg_interior = rn2_stg[..., 1:1 + nk]

    w = st["wet_iface"]
    m = w & np.isfinite(n2_from_stg) & np.isfinite(rn2_stg_interior)
    a2, b2 = n2_from_stg[m], rn2_stg_interior[m]
    corr = float(np.corrcoef(a2, b2)[0, 1]) if a2.std() > 0 and b2.std() > 0 else float("nan")
    maxdiff = float(np.max(np.abs(a2 - b2)))
    rate, n_flip, n_tot = flip_rate(n2_from_stg, rn2_stg_interior, w)
    print(f"  n={int(m.sum())}  corr(legoESM nemo_bn2, NEMO rn2_stg)={corr:.6f}  "
          f"maxdiff={maxdiff:.6e}  flip_rate vs THR={100*rate:.4f}% ({n_flip}/{n_tot})")
    print("  (this validates the nemo_bn2 FORMULA against a real NEMO dump; it "
          "does not re-run the adia-vs-bn2 comparison on this state)")


def run_pass(st, label, *, with_cross_check):
    """Run the full section battery on one state and return its key numbers."""
    a, b = st["n2_adia"], st["n2_bn2"]
    print(f"\nwet-interface count (shared by both N2 branches, printed once): "
          f"n2_adia.shape={a.shape} n2_bn2.shape={b.shape} "
          f"n_wet={int(st['wet_iface'].sum())}  "
          f"dtype(n2_adia)={a.dtype} dtype(n2_bn2)={b.dtype}")

    g = section_global(st)
    boxes = section_boxes(st)
    bins = section_binned(st)
    section_dead_gradient(st)
    noise = section_noise_floor(st)
    classes = section_n2_classes(st)
    if with_cross_check:
        section_nemo_own_rn2(st, st["gdept"], st["gdepw_int"])

    # --- per-pass self-check: global flip count recomputed a second way ---
    n_flip_v2 = int(np.logical_xor(a < THR, b < THR)[st["wet_iface"]].sum())
    assert n_flip_v2 == g["n_flip"], (
        f"SELF-CHECK FAILED [{label}]: flip count via logical_xor ({n_flip_v2}) "
        f"!= flip_rate()'s count ({g['n_flip']})")
    print(f"\n  [self-check {label}] global flip count agrees via two "
          f"computation routes: {n_flip_v2}")
    assert int(st["wet_iface"].sum()) <= a.size, "wet mask larger than field size"
    return dict(g=g, boxes=boxes, bins=bins, noise=noise, classes=classes)


def main():
    banner = "#" * 88
    print(banner)
    print("PASS 1 of 2: FP64 (the SOUND pass -- precision gate ENFORCED)")
    print(banner)
    st64 = build_state(fp64=True)
    res64 = run_pass(st64, "fp64", with_cross_check=True)

    print("\n\n" + banner)
    print("PASS 2 of 2: FLOAT32 -- DELIBERATE NEGATIVE CONTROL")
    print("!! EVERY NUMBER IN THIS BLOCK IS INTENTIONALLY UNSOUND BY CONSTRUCTION. !!")
    print("!! It exists ONLY to test whether the recorded 2026-07-25 '28-40% false !!")
    print("!! trigger' finding is an fp32 artifact. It is NOT a physics result and !!")
    print("!! must never be quoted as one. require_fp64 is intentionally NOT called !!")
    print("!! here -- the gate is correct; this pass is the thing it exists to stop. !!")
    print(banner)
    st32 = build_state(fp64=False)
    res32 = run_pass(st32, "fp32", with_cross_check=False)

    # --- ONE-VARIABLE CONTROL: the two passes must differ ONLY by precision ---
    print("\n" + "=" * 88)
    print("ONE-VARIABLE CONTROL: did the f32 pass run on the SAME input state?")
    print("=" * 88)
    # (a) SAME INPUT: the f32 pass's T/S must be exactly the f64 pass's T/S
    #     rounded to f32 -- i.e. same state, only narrower.
    t_ok = bool(np.array_equal(st64["T"].astype(np.float32), st32["T"]))
    s_ok = bool(np.array_equal(st64["S"].astype(np.float32), st32["S"]))
    # (b) NOT VACUOUS: the f32 pass must really be f32. Without this, (a)
    #     passes trivially when both passes ran f64 -- the failure mode this
    #     probe actually hit on its first run.
    dt_ok = (st64["T"].dtype == np.float64) and (st32["T"].dtype == np.float32)
    print(f"  input T == f64 pass's T cast to f32: {t_ok}")
    print(f"  input S == f64 pass's S cast to f32: {s_ok}")
    print(f"  dtypes: f64 pass T={st64['T'].dtype}  f32 pass T={st32['T'].dtype}  "
          f"(distinct: {dt_ok})")
    print(f"  n2 dtypes: f64 pass={st64['n2_adia'].dtype}  f32 pass={st32['n2_adia'].dtype}")
    print(f"  wet-interface count identical: {int(st64['wet_iface'].sum())} vs "
          f"{int(st32['wet_iface'].sum())}")
    assert t_ok and s_ok, (
        "SELF-CHECK FAILED: the f32 pass did NOT run on the same input T/S as "
        "the f64 pass -- precision is then NOT the only variable and the "
        "comparison below is a confound, not a result.")
    assert dt_ok, (
        "SELF-CHECK FAILED: the two passes do not actually differ in dtype "
        f"(f64 pass T={st64['T'].dtype}, f32 pass T={st32['T'].dtype}) -- the "
        "'f32' pass is vacuous and its agreement with fp64 proves nothing.")
    assert int(st64["wet_iface"].sum()) == int(st32["wet_iface"].sum()), (
        "SELF-CHECK FAILED: wet-interface count differs between passes -- the "
        "masks are not the same population, so the flip rates are not comparable.")
    print("  [OK] same input state, genuinely different precision -- the ONLY "
          "variable between the two passes is the precision.")

    # --- side by side ---
    print("\n" + "=" * 88)
    print("SIDE BY SIDE: fp64 (sound) vs fp32 (negative control)")
    print("=" * 88)
    print(f"  {'metric':<44}{'fp64':>18}{'fp32':>18}")
    print(f"  {'global flip rate':<44}{100*res64['g']['rate']:>17.4f}%"
          f"{100*res32['g']['rate']:>17.4f}%")
    print(f"  {'global corr(adia,bn2)':<44}{res64['g']['corr']:>18.6f}"
          f"{res32['g']['corr']:>18.6f}")
    print(f"  {'global maxdiff':<44}{res64['g']['maxdiff']:>18.4e}"
          f"{res32['g']['maxdiff']:>18.4e}")
    print(f"  {'  of which spurious (adia fires, bn2 stable)':<44}"
          f"{100*res64['g']['n_spurious']/res64['g']['n_tot']:>17.4f}%"
          f"{100*res32['g']['n_spurious']/res32['g']['n_tot']:>17.4f}%")
    print(f"  {'  of which missed (bn2 unstable, adia not)':<44}"
          f"{100*res64['g']['n_missed']/res64['g']['n_tot']:>17.4f}%"
          f"{100*res32['g']['n_missed']/res32['g']['n_tot']:>17.4f}%")
    key = "southern box AND 200-1000m"
    b64, b32 = res64["boxes"].get(key, {}), res32["boxes"].get(key, {})
    r64 = b64.get("rate", float("nan"))
    r32 = b32.get("rate", float("nan"))
    print(f"  {'flip rate, southern box AND 200-1000m':<44}{100*r64:>17.4f}%{100*r32:>17.4f}%")
    print(f"  {'  spurious there':<44}"
          f"{100*b64['n_spurious']/b64['n_tot']:>17.4f}%"
          f"{100*b32['n_spurious']/b32['n_tot']:>17.4f}%")
    print(f"  {'  missed there':<44}"
          f"{100*b64['n_missed']/b64['n_tot']:>17.4f}%"
          f"{100*b32['n_missed']/b32['n_tot']:>17.4f}%")
    print(f"  {'digits consumed by the subtraction':<44}"
          f"{res64['noise']['digits_lost']:>18.1f}{res32['noise']['digits_lost']:>18.1f}")
    print(f"  {'noise/signal, southern 200-1000m':<44}"
          f"{res64['noise']['ratio']:>17.2f}x{res32['noise']['ratio']:>17.2f}x")
    # NB the bin VARIABLE is each pass's own |N2_bn2|, so membership shifts a
    # little between passes (f32 rounds the binning quantity too) -- both counts
    # are printed rather than assuming one population.
    print(f"\n  {'|N2_bn2| bin':<15}{'n f64':>8}{'n f32':>8}{'flip% f64':>12}"
          f"{'flip% f32':>12}{'med|d| f64':>14}{'med|d| f32':>14}")
    for (lab, n64, rate64, md64), (_, n32, rate32, md32) in zip(res64["bins"], res32["bins"]):
        print(f"  {lab:<15}{n64:>8}{n32:>8}{100*rate64:>11.3f}%{100*rate32:>11.3f}%"
              f"{md64:>14.4e}{md32:>14.4e}")

    print("\nDONE")


if __name__ == "__main__":
    main()
