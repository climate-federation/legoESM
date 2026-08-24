#!/usr/bin/env python
"""Barotropic time-average CENTROID, NEMO vs legoESM, at the SHIPPED DINO card.

WHY THIS EXISTS.  ``dino_eta_wave_field_result.md`` measured legoESM's
barotropic impulse response as lagging NEMO's by 0.47 of a baroclinic step and
offered ONE labelled-PLAUSIBLE mechanism: that the two models place their
barotropic time-average at different CENTROIDS, so the committed free surface
carries a different effective time.  That document also named the discriminating
test and said it had not been run.  This probe is that test, and it is pure
arithmetic on the two weight vectors -- no model run, no artifact, no tolerance.

WHAT IT COMPARES.  For the free surface the committed after-level on each side
is a WEIGHTED SUM over barotropic substeps:

    NEMO      dynspg_ts.F90:991    pssh(:,:,Kaa) += wgtbtp1(jn) * ssha_e
              dynspg_ts.F90:1004   pssh(:,:,Kaa) /= SUM(wgtbtp1)
    legoESM   barotropic_latlon_cgrid.py:1016  eta_sum += w_i * eta_new
              (normalised by ``w_total``; kernel built by
              ``compute_nemo_boxcar_centred_weights``)

so the effective time of each side's committed eta is the FIRST MOMENT of its
weight vector, converted to seconds with that side's own substep length.  A
centroid difference is a time offset in the committed state; equal centroids
mean the averaging composition cannot be the owner of a timing lag, whatever
else differs.

THE BRANCH THAT RUNS.  NEMO's weights depend on ``nn_e`` / ``nn_bt_flt`` /
``ln_bt_fw``, and on this configuration ``nn_e`` is NOT the namelist value:
``ln_bt_auto=.true.`` makes NEMO compute it (namelist 30 -> computed 23).  So
the probe reads them from the RUN'S OWN ``ocean.output``, never from the
namelist and never from a hardcoded literal.  legoESM's side is read from the
DINO recipe catalog and from the shipped kernel itself, so a card edit or a
kernel edit moves this probe's answer instead of silently invalidating it.

Usage
-----
    python baro_average_centroid.py --ocean-output /path/to/ocean.output
    python baro_average_centroid.py --nn-e 23 --nn-bt-flt 2 --ln-bt-fw F
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# The DINO twin's baroclinic step (NEMO rn_Dt).  Read back from ocean.output
# when one is supplied; this is only the fallback for the explicit-args path.
DEFAULT_RN_DT_S = 2700.0
CARD = "nemo_dino_kamm_mlf"


# --------------------------------------------------------------------------
# NEMO side: verbatim port of ts_wgt
# --------------------------------------------------------------------------
def nemo_ts_wgt(nn_e: int, nn_bt_flt: int, ln_bt_fw: bool,
                ll_bt_av: bool = True):
    """Verbatim port of NEMO ``dynspg_ts.F90:1227-1294`` ``ts_wgt``.

    Returns ``(wgtbtp1, wgtbtp2, icycle)`` with the weights sliced to
    ``1..Kpit`` and returned 0-indexed (element ``i`` is NEMO's ``jn = i+1``).

    ``ll_bt_av`` is NOT a namelist variable: ``dynspg_ts.F90:202-203`` sets it
    ``.TRUE.`` unless ``nn_bt_flt==3``, so it is derived here the same way by
    the caller rather than guessed.

    The centre ``jic`` is ``nn_e`` for the forward barotropic and ``2*nn_e``
    for the centred one (``:219-221``) -- for the centred case that index is
    the new-time point ``t+Dt`` of a ``2*Dt`` integration started at ``t-Dt``,
    which is what makes the centroid directly comparable to a baroclinic step.
    """
    if nn_e < 1:
        raise ValueError(f"nn_e must be >= 1, got {nn_e!r}")
    if nn_bt_flt not in (0, 1, 2, 3):
        raise ValueError(
            f"unrecognised value for nn_bt_flt: {nn_bt_flt!r} "
            "(NEMO ts_wgt CASE DEFAULT calls ctl_stop)")
    n3 = 3 * nn_e
    w1 = np.zeros(n3 + 1, dtype=np.float64)      # 1-based, index 0 unused
    w2 = np.zeros(n3 + 1, dtype=np.float64)
    jic = nn_e if ln_bt_fw else 2 * nn_e
    kpit = jic
    if ll_bt_av:
        if nn_bt_flt == 0:                       # CASE(0): no averaging
            w1[jic] = 1.0
            kpit = jic
        elif nn_bt_flt in (1, 2):                # CASE(1)/(2): boxcar
            thr = 0.5 if nn_bt_flt == 1 else 1.0
            for jn in range(1, n3 + 1):
                if abs(jn - jic) / nn_e < thr:
                    w1[jn] = 1.0
                    kpit = jn
        else:                                    # nn_bt_flt==3 => ll_bt_av=F
            raise ValueError(
                "nn_bt_flt=3 forces ll_bt_av=.FALSE. (dynspg_ts.F90:202-203); "
                "call with ll_bt_av=False")
    else:                                        # no time averaging
        w1[jic] = 1.0
        kpit = jic
    for jn in range(1, kpit + 1):                # secondary = tail sums
        w2[jn] = w1[jn:kpit + 1].sum()
    return w1[1:kpit + 1], w2[1:kpit + 1], kpit


def centroid(w) -> float:
    """First moment of a weight vector, in NEMO's 1-based substep index."""
    w = np.asarray(w, dtype=np.float64)
    tot = w.sum()
    if tot <= 0.0:
        raise ValueError("a weight vector that sums to zero has no centroid")
    return float((np.arange(1, len(w) + 1) * w).sum() / tot)


# --------------------------------------------------------------------------
# reading the branch that actually ran
# --------------------------------------------------------------------------
def read_ocean_output(path: str) -> dict:
    """Pull nn_e / nn_bt_flt / ln_bt_fw / rn_Dt / rDt_e from a NEMO log.

    These are the values NEMO PRINTS after resolving the namelist, which is the
    only trustworthy source here: ``ln_bt_auto=.true.`` overrides the namelist's
    ``nn_e``, so quoting the namelist would quote a dead value.
    """
    txt = open(path, errors="replace").read()

    def grab(pat, cast, what):
        m = re.search(pat, txt)
        if not m:
            raise SystemExit(
                f"{path}: could not find {what} -- refusing to guess it")
        return cast(m.group(1))

    out = {
        "nn_bt_flt": grab(r"Barotropic time filter\s*=>\s*nn_bt_flt\s*=\s*(\d+)",
                          int, "nn_bt_flt"),
        "nn_e": grab(r"in iterations\s+nn_e\s*=\s*(\d+)", int, "nn_e"),
        "rDt_e_s": grab(r"Barotropic time steps => in seconds\s*=\s*([\d.E+-]+)",
                        float, "rDt_e"),
    }
    # ln_bt_fw is printed as a sentence, not a flag line.
    if re.search(r"ln_bt_fw\s*=\s*F\b.*?Centred integration", txt, re.S) or \
       re.search(r"ln_bt_fw=F\s*=>\s*Centred integration", txt):
        out["ln_bt_fw"] = False
    elif re.search(r"ln_bt_fw\s*=\s*T\b", txt) or \
            re.search(r"ln_bt_fw=T", txt):
        out["ln_bt_fw"] = True
    else:
        raise SystemExit(f"{path}: could not determine ln_bt_fw -- refusing "
                         "to guess which barotropic branch ran")
    # ts_wgt is called with ll_fw_start, NOT ln_bt_fw: dynspg_ts.F90:242-249
    # forces ll_fw_start=.TRUE. at nit000 when l_1st_euler is set, and :256-268
    # re-derives the weights at nit000+1.  On a restart continuation with
    # ln_1st_euler=F the two coincide; on a from-rest log they do NOT, and the
    # window this probe builds would be the wrong one.  Refuse rather than
    # silently model the wrong branch.
    # NEMO announces a forced Euler start in THREE different ways
    # (src/OCE/DOM/domain.F90:399, :408, :416) -- catching only one of them
    # would let the other two through silently.
    _euler_markers = (
        r"l_1st_euler\s+(?:is\s+)?forced to\s*\.true\.",   # :408 and :416
        r"forced euler first time-step",                       # :399
        r"ln_1st_euler\s*=\s*T\b",                            # namelist echo
    )
    if any(re.search(m, txt, re.I) for m in _euler_markers):
        raise SystemExit(
            f"{path}: this run takes an Euler first step (ln_1st_euler/"
            "l_1st_euler true), so ts_wgt is called with ll_fw_start=.TRUE. at "
            "nit000 and the weights are re-derived at nit000+1. This probe "
            "models the steady centred branch only -- refusing rather than "
            "reporting the wrong window.")
    out["ln_1st_euler"] = False
    m = re.search(r"rDt\s*=\s*([\d.]+)", txt)
    out["rDt_mlf_s"] = float(m.group(1)) if m else None
    # rn_Dt is the ONLY scalar that sets both time origins and legoESM's substep
    # length, so it is read too rather than left to a default (adversarial
    # review M1: the docstring promised this and main() did not do it).
    out["rn_Dt_s"] = grab(r"rn_Dt\s*=\s*([\d.E+-]+)", float, "rn_Dt")
    # Internal consistency of the three printed times: dynspg_ts.F90:1443 sets
    # rDt_e = rn_Dt/nn_e, and MLF runs rDt = 2*rn_Dt.  A log that fails either
    # is not describing the run we think it is.
    if abs(out["rDt_e_s"] * out["nn_e"] - out["rn_Dt_s"]) > 1e-6 * out["rn_Dt_s"]:
        raise SystemExit(
            f"{path}: rDt_e*nn_e = {out['rDt_e_s'] * out['nn_e']:.6f} != rn_Dt "
            f"= {out['rn_Dt_s']:.6f} -- refusing to build a clock from an "
            "inconsistent log")
    if out["rDt_mlf_s"] is not None and not out["ln_bt_fw"] and \
            abs(out["rDt_mlf_s"] - 2.0 * out["rn_Dt_s"]) > 1e-6 * out["rn_Dt_s"]:
        raise SystemExit(
            f"{path}: MLF rDt = {out['rDt_mlf_s']} != 2*rn_Dt = "
            f"{2.0 * out['rn_Dt_s']} -- the centred barotropic window assumes a "
            "2*rn_Dt leap-frog")
    return out


def lego_card_settings(card: str = CARD) -> dict:
    """Barotropic settings of the SHIPPED recipe, read from the catalog."""
    from legoesm.ocean.experiments.dino import DINO_RECIPES  # noqa: PLC0415
    if card not in DINO_RECIPES:
        raise SystemExit(f"unknown DINO recipe {card!r}")
    r = DINO_RECIPES[card]
    integ = r.get("outer_integrator", "forward_euler")
    # ocean_model_latlon_cgrid.py:8129 `_baro_scale = 2` -- the leap-frog
    # integrates the barotropic mode over rDt=2dt and scales the substep COUNT
    # so the substep LENGTH is unchanged.  Any other integrator runs scale 1.
    scale = 2 if integ in ("leapfrog", "nemo_mlf") else 1
    # This probe builds the nn_bt_flt=2 centred boxcar and NOTHING else.  If the
    # card selects a different filter its centroid is a different number, and
    # printing boxcar centroids under a "cosine" label would be a silently wrong
    # answer (adversarial review M2).
    filt = r.get("barotropic_time_filter")
    if filt not in ("nemo_boxcar_centred", "nemo_boxcar_ab3"):
        raise SystemExit(
            f"{card}: barotropic_time_filter={filt!r} is not the NEMO centred "
            "boxcar this probe models. Its centroid is NOT the one computed "
            "here -- refusing rather than mislabelling.")
    # dino.py recomputes the substep count when barotropic_auto_cmax is set;
    # the raw card field would then be a dead value.
    auto = r.get("barotropic_auto_cmax", 0.0)
    if auto:
        raise SystemExit(
            f"{card}: barotropic_auto_cmax={auto!r} makes the substep count "
            "auto-computed, so n_barotropic_substeps is not what runs.")
    return {
        "outer_integrator": integ,
        "barotropic_time_filter": r.get("barotropic_time_filter"),
        "n_barotropic_substeps": r.get("n_barotropic_substeps"),
        "substep_scale": scale,
    }


def lego_weights(n_substeps: int, substep_scale: int):
    """The SHIPPED legoESM kernel -- imported, never re-derived here."""
    import jax.numpy as jnp  # noqa: PLC0415
    from legoesm.ocean.dynamics.barotropic_common import (  # noqa: PLC0415
        compute_nemo_boxcar_centred_weights,
    )
    w, w_total, w_tr, n_loop = compute_nemo_boxcar_centred_weights(
        n_substeps, jnp.float64, substep_scale=substep_scale)
    return np.asarray(w, dtype=np.float64), np.asarray(w_tr, np.float64), int(n_loop)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ocean-output", default=None,
                   help="NEMO ocean.output of the run under test (preferred: "
                        "carries the RESOLVED nn_e, which ln_bt_auto overrides)")
    p.add_argument("--nn-e", type=int, default=None)
    p.add_argument("--nn-bt-flt", type=int, default=None)
    p.add_argument("--ln-bt-fw", choices=("T", "F"), default=None)
    p.add_argument("--rn-dt", type=float, default=DEFAULT_RN_DT_S,
                   help=f"baroclinic step [s] (default {DEFAULT_RN_DT_S})")
    p.add_argument("--card", default=CARD)
    p.add_argument("--json-out", default=None)
    a = p.parse_args(argv)

    if a.ocean_output:
        nem = read_ocean_output(a.ocean_output)
        src = a.ocean_output
    elif None not in (a.nn_e, a.nn_bt_flt, a.ln_bt_fw):
        nem = {"nn_e": a.nn_e, "nn_bt_flt": a.nn_bt_flt,
               "ln_bt_fw": a.ln_bt_fw == "T",
               "rDt_e_s": a.rn_dt / a.nn_e, "rDt_mlf_s": None}
        src = "command line"
    else:
        raise SystemExit("give --ocean-output, or all of --nn-e/--nn-bt-flt/"
                         "--ln-bt-fw; this probe will not guess which "
                         "barotropic branch ran")

    rn_dt = nem["rn_Dt_s"] if "rn_Dt_s" in nem else a.rn_dt
    nn_e, flt, fw = nem["nn_e"], nem["nn_bt_flt"], nem["ln_bt_fw"]
    ll_bt_av = flt != 3                      # dynspg_ts.F90:202-203
    w1, w2, icycle = nemo_ts_wgt(nn_e, flt, fw, ll_bt_av)
    c1, c2 = centroid(w1), centroid(w2)
    dt_e = nem["rDt_e_s"]

    # NEMO's substep jn sits at  t - Dt + jn*dt_e  for the centred barotropic
    # (loop entry is the BEFORE level), at  t + jn*dt_e  for the forward one.
    t0 = -rn_dt if not fw else 0.0
    c1_t = t0 + c1 * dt_e                     # seconds relative to t
    c2_t = t0 + c2 * dt_e

    card = lego_card_settings(a.card)
    n_sub = card["n_barotropic_substeps"] * card["substep_scale"]
    lw, ltr, n_loop = lego_weights(n_sub, card["substep_scale"])
    lc1, lc2 = centroid(lw), centroid(ltr)
    # legoESM's leap-frog seeds the barotropic from the BEFORE level too
    # (ocean_model_latlon_cgrid.py:8137-8140 `_barotropic_before_state`), and
    # dt_s = dt_mom/n_substeps with dt_mom = 2*Dt (ibid.:3952).
    l_dt_s = (2.0 * rn_dt if card["substep_scale"] == 2 else rn_dt) / n_sub
    l_t0 = -rn_dt if card["substep_scale"] == 2 else 0.0
    lc1_t = l_t0 + lc1 * l_dt_s
    lc2_t = l_t0 + lc2 * l_dt_s

    nsteps = (lc1_t - c1_t) / rn_dt

    print(f"NEMO   [{src}]  nn_e={nn_e} nn_bt_flt={flt} ln_bt_fw="
          f"{'T' if fw else 'F'} ll_bt_av={ll_bt_av}")
    print(f"  icycle = {icycle} substeps of {dt_e:.10f} s "
          f"(rn_Dt={rn_dt:g} s, rDt={nem['rDt_mlf_s']})")
    nz = np.nonzero(w1)[0]
    print(f"  primary boxcar window: substeps {nz[0]+1}..{nz[-1]+1}  "
          f"(jic = {nn_e if fw else 2*nn_e})")
    print(f"  primary   centroid = substep {c1:.6f}  ->  t {c1_t:+.4f} s "
          f"= t {c1_t/rn_dt:+.6f} steps")
    print(f"  secondary centroid = substep {c2:.6f}  ->  t {c2_t:+.4f} s "
          f"= t {c2_t/rn_dt:+.6f} steps")
    print()
    print(f"legoESM [{a.card}]  filter={card['barotropic_time_filter']!r} "
          f"n_barotropic_substeps={card['n_barotropic_substeps']} "
          f"integrator={card['outer_integrator']!r} scale={card['substep_scale']}")
    print(f"  n_loop = {n_loop} substeps of {l_dt_s:.10f} s")
    lnz = np.nonzero(lw)[0]
    print(f"  filter window: substeps {lnz[0]+1}..{lnz[-1]+1}")
    print(f"  filter    centroid = substep {lc1:.6f}  ->  t {lc1_t:+.4f} s "
          f"= t {lc1_t/rn_dt:+.6f} steps")
    print(f"  transport centroid = substep {lc2:.6f}  ->  t {lc2_t:+.4f} s "
          f"= t {lc2_t/rn_dt:+.6f} steps")
    print()
    same_len = len(lw) == len(w1)
    wmax = (float(np.abs(lw - w1 / w1.sum()).max()) if same_len else float("nan"))
    dt_diff = abs(l_dt_s - dt_e)
    print(f"  eta-kernel weights identical? n_loop {n_loop} vs icycle {icycle}; "
          f"max|w_lego - w_nemo/sum| = {wmax:.3e}")
    print(f"  substep length identical?     {l_dt_s:.10f} vs {dt_e:.10f} s  "
          f"(diff {dt_diff:.3e} s)")
    print()

    # ---- THE GATE (adversarial review B1) --------------------------------
    # The centroid offset is NOT a sufficient statistic on its own: both the
    # leap-frog window (substeps 24..68 on a clock starting at t-Dt) and the
    # forward-Euler one (substeps 1..45 on a clock starting at t) have their
    # centroid at exactly t+Dt, so a MISCONFIGURED probe reports the same
    # -0.000000 as a correct one.  Without this gate the probe had no control
    # that could make its own headline nonzero -- it could not report failure.
    # The discriminating facts are the kernel identity and the clock identity,
    # so the verdict is printed ONLY when both hold.
    problems = []
    if not same_len:
        problems.append(
            f"kernel LENGTHS differ: legoESM n_loop={n_loop} vs NEMO "
            f"icycle={icycle}")
    elif wmax > 0.0:
        problems.append(f"kernel WEIGHTS differ: max|dw| = {wmax:.3e}")
    if dt_diff > 1e-9:
        problems.append(
            f"substep LENGTHS differ: {l_dt_s:.10f} vs {dt_e:.10f} s "
            f"(diff {dt_diff:.3e} s)")
    if problems:
        for msg in problems:
            print(f"  INVALID: {msg}")
        raise SystemExit(
            "the two averaging compositions are NOT the same object, so their "
            "centroid difference is not a meaningful timing offset -- refusing "
            "to print a verdict. (A centroid match is necessary, not "
            "sufficient: different windows on different clocks can share a "
            "centroid.)")

    print(f"PREDICTED legoESM - NEMO committed-eta centroid offset = "
          f"{nsteps:+.6f} baroclinic steps = {nsteps*rn_dt/60.0:+.4f} min")
    print("  (valid: identical weight vectors on identical substep clocks)")

    res = {
        "rn_Dt_s": rn_dt,
        "nemo": {"source": src, "nn_e": nn_e, "nn_bt_flt": flt,
                 "ln_bt_fw": fw, "ll_bt_av": ll_bt_av, "icycle": icycle,
                 "rDt_e_s": dt_e, "primary_centroid_substep": c1,
                 "secondary_centroid_substep": c2,
                 "primary_centroid_steps_rel_t": c1_t / rn_dt},
        "legoesm": {"card": a.card, **card, "n_loop": n_loop,
                    "dt_s": l_dt_s, "filter_centroid_substep": lc1,
                    "transport_centroid_substep": lc2,
                    "filter_centroid_steps_rel_t": lc1_t / rn_dt},
        "max_abs_weight_diff": wmax,
        "predicted_offset_steps": nsteps,
        "predicted_offset_minutes": nsteps * rn_dt / 60.0,
    }
    if a.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
