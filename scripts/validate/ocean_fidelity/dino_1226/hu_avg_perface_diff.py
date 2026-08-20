#!/usr/bin/env python
"""#1455 sec-D JOB 1 -- the un-run measurement: per-face transport diff.

CHAIN THIS PROBE WAS BUILT ON -- READ THE HISTORY BLOCK BELOW BEFORE CITING IT:
  the per-step ~4.2e-3 m eta injection was a WIND-OFF artifact (it is 1.95e-4 m
  wind-on).  The claim that legoESM's committed eta is exactly
  ``-2dt * div(Hu_avg)`` is NOT reproduced by this probe: the per-cell identity
  leaves a wind-independent ~4.4e-4 m residual, negligible against the wind-off
  error and dominant against the wind-on one.  The ssh-divergence exoneration
  cited here came from sshnxt_operator_ab.py TEST B, which was a TAUTOLOGY
  (it re-derived its own reference); TEST B has been removed and only TEST A --
  which loads NEMO's own dumps -- supports anything.

THIS PROBE captures legoESM's ACTUAL ``Hu_avg``/``Hv_avg`` from the real bridged
one-step run (kt=230400 -> 230401, card=nemo_dino_kamm_mlf, all fidelity flags
ON) and compares it PER FACE against BOTH NEMO references:

  (i)  ENTRY transport  Hu_entry = SUM_k e2u * e3u(Nnn) * uu(Nnn)   [m^3/s]
       built from the bridged restart velocities + live e3u(Nnn) via the SAME
       r3u ssh-avg convention (domqco.F90:166) validated in sshnxt_operator_ab
       (C0).  This is what the ssh COMMIT consumes (the FIRST div_hor, BEFORE
       dyn_spg): NEMO's committed ssh(Naa)=ssh(Nbb)-2dt*div_h(Hu_entry).  The
       closure arithmetic ``-2dt*div(dH*u)`` is against THIS reference; its
       correct target is the eta ERROR d_eta_lego - d_ssh_nemo, NOT legoESM's
       total eta increment (see HISTORY -- that confusion is retraction #1).

  (ii) NEMO's barotropic time-mean advective transport dump un_adv/vn_adv:
       ``spg_dump_un_adv_final.bin`` [m^2/s] = <zhU/e2u> over the substep window
       (dynspg_ts.F90:736 ``un_adv += za2*zhU*r1_e2u``, /r1_wgt2s at :999).
       This is NEMO's SECONDARY (transport-weighted) mean -- the ``transport_avg``
       reconcile target -- so ``Hu_avg`` (legoESM's substep-mean transport) is
       the DIRECT analogue.  Map: un_adv * e2u == Hu (full transport [m^3/s]).

CONTROLS (skill Rule 3/1d/2):
  * C0 re-validation: reproduce NEMO's OWN first-guess d_ssh through the ENTRY
    transport reconstruction (the sshnxt_operator_ab C0 pattern) BEFORE trusting
    (i).  Fails -> my transport construction is wrong, no diff claim stands.
  * PLANTED per comparison: roll u,v one cell -> the per-face residual MUST blow
    up (metric is not translation-invariant).
  * day-0 bit-identity: build_replay_ic asserts max|dT|=max|d_eta|=0.
  * fp64 (run_fp64.py wraps); LEGOESM_NEMO_E3T=both; dtypes printed; card printed.

TIME LEVELS: un_adv/vn_adv time levels are taken from
``ocean.fidelity.time_levels`` (``time_level_for_dump``), not assumed here.

HISTORY -- WIND-OFF DEFECT AND THE WIND-ON RE-MEASUREMENT (2026-08-19)
==============================================================================
DEFECT.  Every number this probe recorded at commit 40b92249a was measured on
an UNFORCED ocean.  The kamm card routes the wind momentum through
``model.step(surface_forcing=sf)`` and the analytic applicator deliberately
skips its own wind when ``wind_through_step=True`` (dino.py:3756-3757), so the
old ``surface_forcing=None`` call dropped the wind from BOTH paths.  This is
the exact defect retracted at fdb5cfec6 and fixed in four sibling probes at
5e8407797; this probe and zu_frc_assembly_table.py were missed by that sweep.

CONTINUITY CONTROL first (``DINO_HU_WIND=0``): the wind-off arm reproduces the
recorded numbers EXACTLY -- err_norm 6.6191e-03, max 0.87919 m^2/s, eta
increment 4.1968e-03 m -- so the wind is the single variable that changed.

WIND-ON (fp64, LEGOESM_NEMO_E3T=both, day-0 gate 0.000e+00, tau_x in
[-0.1999, 0.1000] Pa, C0 1.39e-14 / rel 1.31e-10, planted roll 17.9 m, mapping
plant 4043x):

                            wind-off (recorded)   wind-on (this fix)   factor
  eta increment |d_eta|     4.1968e-03 m          1.9317e-04 m         21.7x down
  Hu err_norm (L2 rel)      6.6191e-03            5.7542e-05           115x down
  Hu residual max           8.7919e-01 m^2/s      5.9913e-02 m^2/s     14.7x down
  Hv residual max           6.2493e-01 m^2/s      6.3426e-02 m^2/s     9.9x down
  closure -2dt*div(dHu)     4.1658e-03 m          5.7835e-04 m

(A SECOND defect, found in adversarial review and fixed in the same pass: the
eta increment was being taken against ``st0.eta`` -- the NOW level -- while
legoESM's invariant and NEMO's own increment are both referenced to the BEFORE
level. |sshn - sshb| is 8.81e-05 m, the same order as the increment itself. The
baseline is now ``st0.eta_before``, which is why d_eta reads 1.9317e-04 rather
than the 1.9509e-04 first recorded.)

RETRACTED #1 -- "the transport diff closes the eta injection to 3 sig figs".
The recorded agreement (4.166e-3 closure vs 4.20e-3 "injection") compared the
closure against legoESM's TOTAL eta increment.  That is the wrong reference:
by linearity ``-2dt*div(Hu_avg - Hu_entry)`` equals ``d_eta_lego - d_ssh_nemo``,
i.e. the eta ERROR, and wind-off the two references coincided only because
|d_eta_lego| = 4.2e-3 dwarfed NEMO's own |d_ssh_nemo| = 1.07e-4.  Both
references are now printed, together with the PER-CELL identity residual.

THE REAL FINDING, and it is bigger than the retraction.  Because C0 pins
``d_ssh_nemo = -2dt*div(Hu_entry)`` to 1.4e-14, ``Hu_entry`` CANCELS out of the
per-cell identity, which therefore reduces to a pure legoESM-side test of
``eta(Naa) = eta(Nbb) - 2dt*div(Hu_avg)`` -- the invariant asserted in
ocean_model_latlon_cgrid.py:7716.  THAT INVARIANT DOES NOT HOLD ON THIS CARD:

  |closure - (d_eta_lego - d_ssh_nemo)|   wind-off 4.4236e-04 m, L2rel 2.39e-02
                                          wind-on  4.4641e-04 m, L2rel 5.83e-01

i.e. a residual of 231% of the wind-on eta increment's own max.  It was 2.4% of
the wind-off eta error and so passed unnoticed; wind-on it dominates.

WHAT THE RESIDUAL IS NOT (each CONFIRMED by measurement, not by argument):
  * NOT the global eta-drift fixer, or any spatially uniform writer: the
    residual is zero-mean with std/|mean| = 179, and 518 of 9920 wet cells
    carry 90% of its L2^2 -- printed by this probe.
  * NOT a metric or mask difference: legoESM's bridged dy_u/area are
    BIT-IDENTICAL to e2u/e1e2t, its face masks differ from NEMO's umask/vmask
    in ZERO faces, and the periodic seam Hu_avg[:,0]-Hu_avg[:,52] is exactly 0.
  * NOT an index-mapping error: the mapping plant control above blows the
    residual up 4043x, so the comparison can see an off-by-one and does not.

WHAT IT PLAUSIBLY IS (labelled PLAUSIBLE, one run from being settled): this
lane already recorded at bbfdb8e5d that NEMO averages its barotropic substeps
TWO ways and that legoESM commits the slower-centred average into the after
slot.  If ``eta`` and ``Hu_avg`` emerge from different substep filters the
identity CANNOT hold.  The switchable field already exists; flipping it is a
single-variable, one-run test.

M1 PRE-REGISTRATION (#1455 decisive measurement, written BEFORE the run)
==============================================================================
THE A/B.  ONE variable: the barotropic substep-averaging convention, selected
by ``DINO_RECONCILE`` (config-level override of the card's own resolved value,
no production edit).  Everything else held: fp64, LEGOESM_NEMO_E3T=both, wind
ON, the same bridged IC at kt=230400, the same single step.

NUMBER PRODUCED, per arm: the PER-CELL identity residual
``|closure - (d_eta_lego - d_ssh_nemo)|``, max [m] and L2rel.  The recorded
card-default value is 4.4641e-04 m / L2rel 5.83e-01 (231% of the wind-on eta
increment's own max).

CONFIRMS the convention owns the invariant violation: ONE arm's residual
collapses to near roundoff (max <= 1e-12 m, i.e. >= 1e8x down) while the other
stays at ~4.5e-04 m.

REFUTES it: both arms return the same residual to within 1%.

PREDICTION, recorded before the run and read off the call graph, not guessed:
REFUTED, and BIT-IDENTICALLY so.  ``_reconcile_targets``
(barotropic_latlon_cgrid.py:1113-1154) is called at :1529, AFTER the substep
loop has returned; it reads ``U_bar_avg``/``Hu_avg`` and returns ONLY the
depth-uniform target that the 3-D VELOCITY is rebuilt on.  ``eta_avg`` (:1511)
and ``Hu_avg`` (:1498) are both already fixed at that point and neither is a
function of the flag, and ``state.eta`` is set from ``eta_avg`` alone (:1546).
Within ONE step from a fixed IC the flag therefore cannot move either side of
the identity by a single bit.  If the two arms differ AT ALL, this reading of
the call graph is wrong, and THAT is the finding.

WITHDRAWN, this probe's own over-claims, both raised in review:
  * "the remainder is WIND-INDEPENDENT" was presented as suggestive.  It is
    not evidence: the wind changes |Hu_avg| by at most ~0.3% (0.88 of 272
    m^2/s), so ANY term linear in Hu_avg is constant to that tolerance across
    the flip.  The wind-off/wind-on pair cannot discriminate the candidates.
  * "the transport diff owns roughly 42% of the eta error" is withdrawn.  It
    came from 1 - L2rel (or from a ratio of two maxima at different cells);
    norms do not decompose additively into ownership shares.

RETRACTED #2 -- "WALL-concentrated (W 0.879, E 0.795, interior 0.589)".  That
structure was the missing wind stress at the zonal walls.  Wind-on the maximum
is INTERIOR (5.99e-02) and the west-wall column is the SMALL one (1.25e-02);
the coherent zonal-mean signature drops from 5.44e-01 to 3.44e-03 m^2/s.

SURVIVES -- the controls (C0 at 1.3e-14, planted roll, day-0 bit-identity) and
the secondary reading that NEMO's own un_adv and its entry transport differ
only slightly, so references (i) and (ii) give the same answer.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEQDUMP = os.environ.get(
    "DINO_NEMO_RUN_SEQDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_Y20_1R")

JPI, JPJ, JPK, HLS = 56, 203, 36, 2
NI, NJ = JPI - 2 * HLS, JPJ - 2 * HLS   # 52 x 199 interior
RN_DT = 2700.0
RDT = 2.0 * RN_DT   # MLF leap-frog


def _disposition(basename):
    """Force the NEMO TIME LEVEL of a raw dump through the shared registry.

    ``time_level_for_dump`` RAISES on an unregistered basename, so a new dump
    cannot be loaded here until someone has read its NEMO write site and
    recorded the file:line proof.

    SCOPE, precisely: this is a REGISTRATION tripwire, not a level check.  It
    stops a dump whose level nobody has established from being loaded at all.
    It does NOT verify that the caller then compares the dump against
    state at the matching level -- a registered before-level dump can still be
    differenced against now-level state, and that remains the caller's job.
    """
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    return time_level_for_dump(basename)


def _finite(a, fn):
    """NaN/Inf in an oracle dump is FATAL -- never nan-reduced away."""
    if not np.isfinite(a).all():
        raise SystemExit(f"*** {fn}: {int((~np.isfinite(a)).sum())} non-finite "
                         "values -- FATAL (a NaN here would be silently hidden "
                         "by any nan-reduction downstream)")
    return a


def _load2d_full(fn):
    """Singleton (no _kt) haloed 2-D dump -> (nj,ni) interior."""
    _disposition(fn)
    a = np.fromfile(os.path.join(SEQDUMP, fn), dtype="<f8")
    assert a.size == JPI * JPJ, (fn, a.size)
    return _finite(a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS], fn)


def _load2d_kt(base, kt):
    _disposition(base)
    a = np.fromfile(os.path.join(SEQDUMP, f"{base}_kt{kt:08d}.bin"), dtype="<f8")
    assert a.size == JPI * JPJ, (base, a.size)
    return _finite(a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS], base)


def _mesh():
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    g = lambda v: np.asarray(d.variables[v][0], dtype=np.float64)
    e1u, e2u = g("e1u"), g("e2u")
    e1v, e2v = g("e1v"), g("e2v")
    e1t, e2t = g("e1t"), g("e2t")
    e3u0, e3v0, e3t0 = g("e3u_0"), g("e3v_0"), g("e3t_0")
    umask, vmask, tmask = g("umask"), g("vmask"), g("tmask")
    d.close()
    e1e2t = e1t * e2t
    ht0 = (e3t0 * tmask).sum(0)
    hu0 = (e3u0 * umask).sum(0)
    hv0 = (e3v0 * vmask).sum(0)
    ssmask = (tmask[0] > 0.5).astype(np.float64)
    return dict(e1u=e1u, e2u=e2u, e1v=e1v, e2v=e2v, e1e2t=e1e2t, e3u0=e3u0,
                e3v0=e3v0, e3t0=e3t0, umask=umask, vmask=vmask, tmask=tmask,
                ht0=ht0, hu0=hu0, hv0=hv0, ssmask=ssmask,
                umask2=(umask[0] > 0.5), vmask2=(vmask[0] > 0.5))


def _nemo_entry_transport(u, v, sshn, m, *, roll=None):
    """(i) ENTRY per-face transport PER UNIT WIDTH  hu = SUM_k e3u(Nnn)*uu(Nnn)
    [m^2/s],  hv = SUM_k e3v(Nnn)*vv(Nnn).  u,v:(nj,ni,jpk) at faces.  Live
    e3(Nnn) via the domqco r3u/r3v area-weighted 2-cell ssh mean.  This is the
    SAME units as legoESM Hu_avg and NEMO un_adv (transport per unit width);
    the e2u/e1v width factor enters only inside div_h (see _dssh_from_transport).
    """
    u = u.copy(); v = v.copy()
    if roll is not None:
        ax, sh = roll
        u = np.roll(u, sh, axis=ax); v = np.roll(v, sh, axis=ax)
    e1e2t = m["e1e2t"]
    sshu = 0.5 * (e1e2t * sshn + np.roll(e1e2t, -1, 1) * np.roll(sshn, -1, 1))
    r3u = np.where(m["hu0"] > 0, sshu / (m["e1u"] * m["e2u"] * np.maximum(m["hu0"], 1e-30)), 0.0)
    sshv = 0.5 * (e1e2t * sshn + np.roll(e1e2t, -1, 0) * np.roll(sshn, -1, 0))
    r3v = np.where(m["hv0"] > 0, sshv / (m["e1v"] * m["e2v"] * np.maximum(m["hv0"], 1e-30)), 0.0)
    e3u = m["e3u0"] * (1.0 + r3u[None]) * m["umask"]
    e3v = m["e3v0"] * (1.0 + r3v[None]) * m["vmask"]
    up = np.moveaxis(u, -1, 0) * m["umask"]
    vp = np.moveaxis(v, -1, 0) * m["vmask"]
    hu = (e3u * up).sum(0)   # (nj,ni) [m^2/s] transport per unit width
    hv = (e3v * vp).sum(0)
    return hu, hv


def _dssh_from_transport(hu, hv, m):
    """NEMO ssh_nxt operator: d_ssh = -2dt * (1/e1e2t) * div_h(e2u*hu, e1v*hv).
    ``hu``/``hv`` are PER UNIT WIDTH [m^2/s]; the e2u/e1v width factor is applied
    HERE inside the divergence (NEMO divhor.F90: di[e2u*e3u*u]).  di=f(ji)-f(ji-1)
    (west=roll+1 in i, periodic); dj: south wall no flux.

    VALIDATED by THIS probe's own control C0 (which reproduces NEMO's dumped
    first-guess d_ssh from NEMO's own inputs): residual 1.39e-14 m, rel
    1.31e-10, with the planted roll control at 17.9 m.  The "9.3e-15" figure
    this docstring used to cite came from sshnxt_operator_ab's TEST B, which
    was a TAUTOLOGY and has been deleted (commit 8b8638cc7); it is not
    evidence and is no longer quoted."""
    Hu = m["e2u"] * hu   # -> full transport [m^3/s]
    Hv = m["e1v"] * hv
    di = Hu - np.roll(Hu, 1, axis=1)
    dj = Hv.copy()
    dj[1:, :] = Hv[1:, :] - Hv[:-1, :]
    dj[0, :] = Hv[0, :]
    return -RDT * ((di + dj) / m["e1e2t"]) * m["ssmask"]


# ---- capture legoESM's Hu_avg/Hv_avg from the real bridged one step ----------
_CAP = {}


def _hook_barotropic():
    """Monkeypatch barotropic_substeps_latlon_cgrid to stash (Hu_avg,Hv_avg).

    ``step`` is jit-compiled, so Hu_avg is a tracer here -- copy the CONCRETE
    runtime value out via io_callback (host side-effect, value unchanged)."""
    import jax
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as omod
    _orig = omod.barotropic_substeps_latlon_cgrid

    def _stash(hu, hv):
        _CAP["Hu_avg"] = np.asarray(hu)
        _CAP["Hv_avg"] = np.asarray(hv)

    def _wrapped(*a, **k):
        state_new, (Hu_avg, Hv_avg) = _orig(*a, **k)
        jax.experimental.io_callback(_stash, None, Hu_avg, Hv_avg)
        return state_new, (Hu_avg, Hv_avg)
    omod.barotropic_substeps_latlon_cgrid = _wrapped
    return _orig


def _run_one_step_capture():
    import jax
    import multistep_replay as mr
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)

    import dataclasses

    mr.IC_STEP = 230400
    jax.clear_caches()
    g, br, cfg, st0 = mr.build_replay_ic()   # asserts day-0 bit-identity
    # -- M1 A/B: the barotropic substep-averaging convention, ONE variable ----
    # ``DINO_RECONCILE`` overrides the CARD's own resolved value at the config
    # level only (no production edit).  Unset => the card decides; the resolved
    # value on the model config is printed either way, since the card default
    # is the thing under test and a probe that echoed my label instead of the
    # resolved field would be worthless as an A/B.
    _card_recon = getattr(cfg, "barotropic_reconcile_target", None)
    _recon_env = os.environ.get("DINO_RECONCILE", "")
    if _recon_env:
        if _recon_env not in ("velocity_avg", "transport_avg"):
            raise SystemExit(
                f"DINO_RECONCILE={_recon_env!r}: must be one of "
                "('velocity_avg', 'transport_avg')")
        cfg = dataclasses.replace(cfg, barotropic_reconcile_target=_recon_env)
    print(f"  card surface_tendency_placement="
          f"{getattr(cfg,'surface_tendency_placement',None)!r}")
    print(f"  RECONCILE: card default={_card_recon!r} "
          f"env override={_recon_env or None!r} "
          f"cfg now={getattr(cfg,'barotropic_reconcile_target',None)!r}")
    print(f"  leapfrog outer={getattr(cfg,'outer_integrator',None)!r}  "
          f"eta dtype={st0.eta.data.dtype} u={st0.u.data.dtype}")
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    # RESOLVED value on the object the solver actually reads (state.py
    # BarotropicConfig), not the DINOConfig the card was built from -- the two
    # are wired together at dino.py:3131 and this print is what makes the A/B
    # checkable rather than label-dependent.
    print(f"  RESOLVED barotropic.barotropic_reconcile_target="
          f"{mc.barotropic.barotropic_reconcile_target!r}  "
          f"time_filter={mc.barotropic.barotropic_time_filter!r}",
          flush=True)
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    # WIND-ON harness (commit 5e8407797's pattern, and its retraction fdb5cfec6):
    # the kamm card threads the WIND MOMENTUM through model.step(surface_forcing=
    # sf) (production run_dino.py:665-670,763-767), and the analytic applicator
    # SKIPS its eq-7 wind when wind_through_step=True (dino.py:3756-3757).  So
    # surface_forcing=None dropped the wind from BOTH paths and this probe's
    # transport was measured on an unforced ocean.  DINO_HU_WIND=0 forces the
    # wind OFF -- the continuity control that reproduces the wind-off numbers so
    # the flip is a single-variable change.
    _card_wind = bool(getattr(cfg, "wind_through_step", False))
    _control_off = os.environ.get("DINO_HU_WIND", "1") == "0"
    _wind = _card_wind and not _control_off
    sf_step = dino_step_surface_forcing(forcing) if _wind else None
    _tau = np.asarray(sf_step.tau_x) if sf_step is not None else np.zeros(1)
    _tlo, _thi = float(np.min(_tau)), float(np.max(_tau))
    _tmag = float(np.max(np.abs(_tau)))
    print(f"  FORCING: wind_through_step="
          f"{bool(getattr(cfg, 'wind_through_step', False))} applied={_wind} "
          f"surface_stress_implicit={getattr(cfg,'surface_stress_implicit',None)} "
          f"tau_x[Pa] range=[{_tlo:.4f},{_thi:.4f}] max|tau_x|={_tmag:.4f}",
          flush=True)
    # GATE INVARIANT: max|tau_x| > 0.  The defect class this guards is an
    # all-zero stress reaching the model (surface_forcing carried but empty);
    # gating on the RANGE instead would false-abort on a spatially uniform but
    # nonzero stress, and gating on nothing is how the wind-off runs got
    # recorded in the first place.
    # GATE, BOTH ARMS.  The retracted defect was surface_forcing=None reaching
    # model.step -- i.e. the _wind=False arm -- so a gate that lives only inside
    # the _wind=True branch cannot fire on it.  Wind-off is therefore reachable
    # ONLY through the explicit continuity control (DINO_HU_WIND=0); if the card
    # ever stops setting wind_through_step (it defaults False, and this probe
    # reads it through a getattr fallback), the run ABORTS instead of silently
    # reproducing the retracted numbers.
    if not _card_wind and not _control_off:
        raise SystemExit(
            "*** FORCING GATE FAILED: the card does not set "
            "wind_through_step, so this run would be WIND-OFF -- exactly the "
            "measurement retracted at fdb5cfec6. Wind-off is legitimate only "
            "as the deliberate continuity control: set DINO_HU_WIND=0.")
    if _wind and not _tmag > 0.0:
        raise SystemExit("*** FORCING GATE FAILED: wind is nominally ON but "
                         "max|tau_x| == 0 -- the probe would measure an "
                         "unforced ocean (the fdb5cfec6 defect class)")
    DT = RN_DT
    st, rate = apply_dino_lat_lon_surface_forcing(
        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    st = model.step(st, DT, surface_forcing=sf_step, external_tracer_rate=rate)
    eta = np.asarray(st.eta.data)
    # TIME LEVEL, and it is NOT the obvious one.  legoESM's leapfrog invariant
    # is eta(Naa) = eta(Nbb) - 2dt*div(Hu_avg) (docstring at
    # ocean_model_latlon_cgrid.py:7716), and the barotropic solve is seeded from
    # state.eta_before (:7902-7904) -- the BEFORE level.  NEMO's own increment
    # is likewise referenced to sshb.  Using st0.eta (the NOW level) here put
    # the two sides on DIFFERENT baselines, |sshn - sshb| = 8.81e-05 m apart --
    # the same order as the increment being measured.
    eta_b = np.asarray(st0.eta_before.data)
    # -- M1 LIVE-FLIP CONTROL (a control that perturbs a zero is not a control)
    # "the two arms are bit-identical" is worthless unless the flag is shown to
    # change SOMETHING in the same run: an inert flag and an exonerated one give
    # the identical null.  ``barotropic_reconcile_target`` selects the
    # depth-uniform target the 3-D VELOCITY is rebuilt on
    # (barotropic_latlon_cgrid.py:1529-1544), so ``st.u`` MUST move across the
    # arms while ``st.eta`` MUST NOT.  Both reductions are printed and dumped so
    # the pair can be differenced between arms without re-running.
    _u = np.asarray(st.u.data); _v = np.asarray(st.v.data)
    print(f"  LIVE-FLIP CONTROL (diff these ACROSS arms): "
          f"sum|u|={np.abs(_u).sum():.17e} sum|v|={np.abs(_v).sum():.17e} "
          f"sum|eta|={np.abs(eta).sum():.17e}", flush=True)
    _out = os.environ.get("DINO_M1_OUT", "")
    if _out:
        np.savez_compressed(_out, u=_u, v=_v, eta=eta, eta_before=eta_b,
                            reconcile=np.array(_recon_env or (_card_recon or "")))
        print(f"  [artifact] -> {_out}", flush=True)
    return eta, eta_b, st, st0


def _per_face_report(tag, lego, nemo, mask, units):
    """err_norm / max / p99.9 + boundary-vs-interior + seam col + zonal."""
    d = (lego - nemo) * mask
    a = np.abs(d)
    sel = a[mask.astype(bool)]
    ref = np.abs(nemo[mask.astype(bool)])
    enorm = float(np.sqrt((sel**2).sum() / max((ref**2).sum(), 1e-300)))
    print(f"\n  [{tag}]  ({units})")
    print(f"    |nemo| max={ref.max():.4e} p50={np.percentile(ref,50):.4e}")
    print(f"    err_norm(L2 rel)={enorm:.4e}  max={sel.max():.4e} "
          f"p99.9={np.percentile(sel,99.9):.4e} p50={np.percentile(sel,50):.4e}")
    # spatial structure (2-D: boundary vs interior, seam column, zonal)
    nj, ni = a.shape
    west2 = a[:, :2][mask[:, :2].astype(bool)]
    east2 = a[:, -2:][mask[:, -2:].astype(bool)]
    south2 = a[:2, :][mask[:2, :].astype(bool)]
    north2 = a[-2:, :][mask[-2:, :].astype(bool)]
    interior = a[2:-2, 2:-2][mask[2:-2, 2:-2].astype(bool)]
    print(f"    boundary max: W={west2.max() if west2.size else 0:.3e} "
          f"E={east2.max() if east2.size else 0:.3e} "
          f"S={south2.max() if south2.size else 0:.3e} "
          f"N={north2.max() if north2.size else 0:.3e}  "
          f"interior max={interior.max() if interior.size else 0:.3e}")
    colmax = np.where(mask.astype(bool), a, 0.0).max(0)   # per-longitude
    order = np.argsort(colmax)[::-1]
    print("    per-lon colmax top: " +
          "  ".join(f"x{int(i)}={colmax[i]:.2e}" for i in order[:5]))
    # zonal structure: is the residual longitude-uniform (barotropic seam) or localized?
    rowmean = np.where(mask.astype(bool), d, np.nan)
    with np.errstate(invalid="ignore"):
        zonal = np.nanmean(rowmean, axis=1)   # (nj,) mean over lon per lat
        zstd = float(np.nanstd(np.nanmean(rowmean, axis=0)))
    print(f"    zonal(mean-over-lon per lat) |max|={np.nanmax(np.abs(zonal)):.3e}  "
          f"lon-uniformity std(mean-over-lat per lon)={zstd:.3e}")
    return d, enorm


def main():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(f"control dtype = {get_policy().control}  "
          f"E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")

    import multistep_replay as mr
    mr.provenance("hu_avg_perface_diff")

    m = _mesh()
    print(f"mesh interior {NI}x{NJ}  e2u dtype={m['e2u'].dtype}  "
          f"e3u0 dtype={m['e3u0'].dtype}")

    kt = 230401
    mr.IC_STEP = 230400
    ns = mr.nemo_now_state_at(230400)
    nb = mr.nemo_before_state_at(230400)
    u = np.asarray(ns.u, dtype=np.float64)      # (nj,ni,jpk) faces
    v = np.asarray(ns.v, dtype=np.float64)
    sshn = np.asarray(ns.ssh, dtype=np.float64)
    sshb = np.asarray(nb.ssh, dtype=np.float64)
    print(f"un dtype={u.dtype} shape={u.shape}  sshn {sshn.shape}")

    # ================= CONTROL C0: entry transport reproduces NEMO d_ssh ======
    # Hu_e/Hv_e are PER UNIT WIDTH [m^2/s]; _dssh_from_transport applies e2u/e1v.
    Hu_e, Hv_e = _nemo_entry_transport(u, v, sshn, m)
    d_ssh_rec = _dssh_from_transport(Hu_e, Hv_e, m)
    r3t_fg = _load2d_kt("r3c_dump_r3t", kt)
    d_ssh_nemo = (r3t_fg * m["ht0"] - sshb) * m["ssmask"]
    wet = m["ssmask"] > 0.5
    err = np.abs((d_ssh_rec - d_ssh_nemo)[wet])
    scale = np.abs(d_ssh_nemo[wet])
    rel = err.max() / max(scale.max(), 1e-30)
    print(f"\n=== C0: entry transport -> NEMO first-guess d_ssh ===")
    print(f"  |d_ssh_nemo| max={scale.max():.4e} (NEMO's OWN per-step eta "
          f"increment)  "
          f"RESID max={err.max():.4e} rel={rel:.3e} "
          f"-> {'C0 CLEAN' if rel < 1e-3 else 'C0 FAIL'}")
    # C0' planted
    Hu_r, Hv_r = _nemo_entry_transport(u, v, sshn, m, roll=(1, 1))
    d_roll = _dssh_from_transport(Hu_r, Hv_r, m)
    errp = np.abs((d_roll - d_ssh_nemo)[wet]).max()
    print(f"  C0' PLANT(roll+1 i): resid max={errp:.3e} "
          f"-> {'PLANT OK' if errp > 10 * err.max() else 'PLANT FAILED'}")
    if rel >= 1e-3:
        raise SystemExit("C0 FAILED: entry-transport construction is wrong; abort.")

    # ================= capture legoESM Hu_avg/Hv_avg ==========================
    print(f"\n=== capturing legoESM Hu_avg/Hv_avg (real bridged step) ===")
    _hook_barotropic()
    eta_lego, eta_b, st, st0 = _run_one_step_capture()
    if "Hu_avg" not in _CAP:
        raise SystemExit("hook missed: Hu_avg not captured (call-site re-lookup?)")
    Hu_avg = _CAP["Hu_avg"]; Hv_avg = _CAP["Hv_avg"]
    print(f"  captured Hu_avg shape={Hu_avg.shape} dtype={Hu_avg.dtype}  "
          f"Hv_avg shape={Hv_avg.shape}")
    # day-0 gate + the per-step eta increment.  No target is asserted here:
    # the "~4.2e-3" this comment used to expect was the WIND-OFF artifact
    # (wind-on it is ~1.95e-4 m).  See HISTORY.
    d_eta = np.abs((eta_lego - eta_b)[wet])
    print(f"  eta increment |eta(Naa) - eta(Nbb)| max={d_eta.max():.4e} "
          f"p99.9={np.percentile(d_eta,99.9):.4e}")

    # legoESM Hu_avg is PER UNIT WIDTH [m^2/s] (== NEMO un_adv, == SUM_k e3u*u),
    # NOT full transport (verified: |Hu_avg|max=272 == |un_adv|max=273).  legoESM
    # u-face grid has (ni+1) faces; NEMO un_adv per-cell EAST face.  Map to NEMO
    # cell indexing: legoESM u_face[:, 1:] (east faces of cells 0..ni-1) == NEMO
    # cell (this mapping was previously credited to sshnxt_operator_ab's
    # TEST B, now deleted as a tautology; it stands on the C0 control and the
    # |Hu_avg|max == |un_adv|max magnitude check on the next line, not on TEST B).
    if Hu_avg.shape[1] == NI + 1:
        Hu_lego = Hu_avg[:, 1:]           # east faces -> NEMO cell convention
    else:
        Hu_lego = Hu_avg
    if Hv_avg.shape[0] == NJ + 1:
        Hv_lego = Hv_avg[1:, :]
    else:
        Hv_lego = Hv_avg
    print(f"  mapped Hu_lego shape={Hu_lego.shape} [m^2/s] (NEMO cell {NJ}x{NI})  "
          f"|Hu_lego|max={np.abs(Hu_lego).max():.3e}")

    # ---- PLANTED control for the INDEX MAPPING itself -----------------------
    # Deleting sshnxt_operator_ab's TEST B removed the only thing that claimed
    # to cover the lego u-face -> NEMO cell mapping (Hu_avg[:, 1:]), and it
    # never really did.  Neither C0 (which never touches legoESM's face array)
    # nor the |Hu_avg|max ~ |un_adv|max magnitude check can see an off-by-one
    # in i -- a 272-vs-273 max survives a roll.  So plant one: rolling the
    # mapped field one cell in i MUST blow the residual up.  If it does not,
    # the metric is translation-invariant and every per-face number below is
    # meaningless.
    _base = np.abs(((Hu_lego - Hu_e) * m["umask2"])[m["umask2"]]).max()
    _roll = np.abs(((np.roll(Hu_lego, 1, axis=1) - Hu_e) * m["umask2"])[m["umask2"]]).max()
    print(f"\n  MAPPING PLANT (roll Hu_lego +1 in i): base residual max="
          f"{_base:.4e} -> rolled {_roll:.4e}  ratio={_roll/max(_base,1e-300):.1f}x"
          f"  -> {'PLANT OK' if _roll > 10 * _base else '*** PLANT FAILED'}")
    if not _roll > 10 * _base:
        raise SystemExit(
            "*** MAPPING PLANT FAILED: rolling the mapped transport one cell "
            "in i did not blow up the residual, so this comparison cannot "
            "detect an index-mapping error and no per-face number stands.")

    # ================= (i) vs ENTRY transport =================================
    print("\n" + "=" * 74)
    print("(i)  legoESM Hu_avg  vs  NEMO ENTRY transport SUM_k e3u(Nnn)*u [m^2/s]")
    print("     (the transport the ssh COMMIT consumes: -2dt*div_h(e2u*this))")
    print("=" * 74)
    _per_face_report("Hu_avg - Hu_entry", Hu_lego, Hu_e, m["umask2"].astype(float), "m^2/s")
    _per_face_report("Hv_avg - Hv_entry", Hv_lego, Hv_e, m["vmask2"].astype(float), "m^2/s")
    # CLOSURE: -2dt*div(e2u*(dH)) is the arithmetic that would TIE the per-face
    # transport diff to the eta error.  Whether it does is MEASURED below by the
    # per-cell identity, not assumed -- wind-on it accounts for ~42% of the
    # error, so this is a partial attribution, not a closure.
    dHu = (Hu_lego - Hu_e) * m["umask2"]
    dHv = (Hv_lego - Hv_e) * m["vmask2"]
    d_ssh_closure = _dssh_from_transport(dHu, dHv, m)
    cl = np.abs(d_ssh_closure[wet])
    # The closure's CORRECT reference is the eta ERROR (lego increment minus
    # NEMO's own first-guess increment), NOT legoESM's total eta increment:
    #   d_eta_lego  = -2dt*div(Hu_avg)      (the ocean_model invariant)
    #   d_ssh_nemo  = -2dt*div(Hu_entry)    (C0, reproduced to 1.3e-14 above)
    # so -2dt*div(Hu_avg - Hu_entry) IS d_eta_lego - d_ssh_nemo by linearity.
    # WIND-OFF the two references coincided only because |d_eta_lego| (4.2e-3)
    # dwarfed |d_ssh_nemo| (1.07e-4) -- the "3 sig figs" agreement recorded at
    # 40b92249a was that accident, not a measurement.  Both are printed now.
    d_eta_signed = (eta_lego - eta_b) * m["ssmask"]
    d_eta_err = np.abs((d_eta_signed - d_ssh_nemo)[wet])
    print(f"\n  CLOSURE -2dt*div_h(e2u*(Hu_avg - Hu_entry)):")
    print(f"    max={cl.max():.4e} p99.9={np.percentile(cl,99.9):.4e} "
          f"p50={np.percentile(cl,50):.4e} m  (arithmetic)")
    print(f"    vs |d_eta_lego|            max={np.abs(d_eta_signed[wet]).max():.4e} m "
          f"(legoESM's TOTAL eta increment -- NOT the closure's reference)")
    print(f"    vs |d_eta_lego - d_ssh_nemo| max={d_eta_err.max():.4e} m "
          f"(the eta ERROR -- the closure's CORRECT reference)")
    # PER-CELL identity, not a ratio of two maxima taken at different cells
    # (a max-of-A / max-of-B can peak in different places and prove nothing).
    _id = np.abs((d_ssh_closure - (d_eta_signed - d_ssh_nemo))[wet])
    _ref = np.abs((d_eta_signed - d_ssh_nemo)[wet])
    print(f"    PER-CELL identity |closure - (d_eta_lego - d_ssh_nemo)|: "
          f"max={_id.max():.4e} m   L2rel="
          f"{np.sqrt((_id**2).sum()/max((_ref**2).sum(),1e-300)):.4e}")
    # LOCALIZE the identity residual instead of reporting only max and L2rel.
    # A max and a norm cannot distinguish "a uniform offset" from "a handful of
    # bad faces", and the distinction is exactly what names the owner: a
    # spatially UNIFORM residual would implicate the global eta-drift fixer,
    # while a sparse zero-mean one cannot be it.
    _r = (d_ssh_closure - (d_eta_signed - d_ssh_nemo))[wet]
    _sq = np.sort(_r**2)[::-1]
    _n90 = int(np.searchsorted(np.cumsum(_sq), 0.9 * _sq.sum()) + 1)
    print(f"    residual structure: mean={_r.mean():+.3e} std={_r.std():.3e} "
          f"m  (std/|mean|={_r.std()/max(abs(_r.mean()),1e-300):.1f})")
    print(f"    concentration: {_n90} of {_r.size} wet cells carry 90% of the "
          f"residual's L2^2")
    print(f"    -> a zero-mean, sparse residual EXCLUDES a spatially uniform "
          f"writer (e.g. the global eta-drift correction); a uniform one would "
          f"show std/|mean| ~ 0.")

    # NO VERDICT PRINTED HERE ON PURPOSE.  A probe that prints its own
    # interpretation gets that interpretation quoted back as evidence.  The
    # identity residual and its L2rel are the measurement; what they imply
    # about ownership belongs in the analysis, after the controls above pass.

    # ================= (ii) vs un_adv/vn_adv dump =============================
    print("\n" + "=" * 74)
    print("(ii)  legoESM Hu_avg  vs  NEMO un_adv/vn_adv [m^2/s] (barotropic mean)")
    print("=" * 74)
    un_adv = _load2d_full("spg_dump_un_adv_final.bin")   # [m^2/s] per unit width
    vn_adv = _load2d_full("spg_dump_vn_adv_final.bin")
    print(f"  un_adv raw |max|={np.abs(un_adv).max():.3e}  "
          f"vn_adv |max|={np.abs(vn_adv).max():.3e} [m^2/s]")
    _per_face_report("Hu_avg - un_adv", Hu_lego, un_adv, m["umask2"].astype(float), "m^2/s")
    _per_face_report("Hv_avg - vn_adv", Hv_lego, vn_adv, m["vmask2"].astype(float), "m^2/s")
    # CLOSURE via (ii): -2dt*div_h(e2u*(Hu_avg - un_adv))
    dHu2 = (Hu_lego - un_adv) * m["umask2"]
    dHv2 = (Hv_lego - vn_adv) * m["vmask2"]
    cl2 = np.abs(_dssh_from_transport(dHu2, dHv2, m)[wet])
    print(f"\n  CLOSURE -2dt*div_h(e2u*(Hu_avg - un_adv)): max={cl2.max():.4e} "
          f"p99.9={np.percentile(cl2,99.9):.4e} m")

    # which reference does the closure arithmetic REQUIRE?  The ssh COMMIT
    # consumes the ENTRY transport (i) at the FIRST div_hor -- so (i) is the
    # closure reference; (ii) is what NEMO reconciles the 3-D velocity ONTO.
    print("\n" + "=" * 74)
    print("REQUIRED CLOSURE REFERENCE = (i) ENTRY transport (the ssh commit's input).")
    print("(ii) un_adv is NEMO's barotropic substep-mean = the transport_avg target.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
