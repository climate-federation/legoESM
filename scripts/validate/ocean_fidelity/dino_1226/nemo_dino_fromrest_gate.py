#!/usr/bin/env python
"""GATE: the standalone NEMO-faithful DINO run must START where NEMO starts.

`run_dino.py --config scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml`
builds NEMO's DINO domain, initial state and surface forcing analytically,
with no NEMO file read at run time.  This gate checks the INITIAL STATE and
the SURFACE FORCING against the oracle's own one-Euler-step-from-rest restart
(`cfgs/DINO/RUN_FROMREST_KT1`), and exits non-zero on ANY inequality.

Why that record carries the untouched initial condition: `nn_it000 = 1` with
`ln_rstart = .false.` sets `l_1st_euler = .true.` (istate.f90:114-115), and
the MLF Euler start advances Kmm/Kaa only -- so the restart's `tb`/`sb`/`sshb`
(the Kbb level) are exactly what `usr_def_istate` produced.  `utau_b` is the
stress `sbc` applied at kt=1: at nit000 with no restart NEMO sets the before
fields FROM the nit000 values (sbcmod.f90:592-599) and writes `utauU` under
that name (sbcmod.f90:611).

THE BAR IS EXACT: zero cells unequal, per field.  Every waiver is written
down with its reason, and the coverage section makes an unaccounted oracle
variable a hard failure (oracle-fidelity Rule 1).

Four sections, all fatal except where a row says DEBT:

  1. ISTATE, through run_dino.py's own path (Rule 10): T, S, eta from
     `dino_lat_lon_state` vs `tb`, `sb`, `sshb`.
  2. SBC, through run_dino.py's own path: the zonal stress, the meridional
     stress, the TKE stress modulus and E-P vs `utau_b`, `vtau_b`, `emp_b`.
  3. SBC, ORACLE SIDE ONLY -- NEMO's own kt=1 heat/salt fluxes recomputed
     from usrdef_sbc.F90 and the now-bit-exact initial state.  This certifies
     the RECORD and pins the seasonal clock; it does NOT certify legoESM,
     which expresses the same physics as a restoring timescale and has no
     `qns`-shaped array to compare.  Labelled as such on every row.
  4. COVERAGE: every initial-state / surface-forcing variable the restart
     carries is VERIFIED here or WAIVED with a reason.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/validate/ocean_fidelity/dino_1226/\\
        nemo_dino_fromrest_gate.py [--restart-glob PAT] [--plant]

`--plant` moves one wet cell of the transcribed initial temperature and one
wet U-point of the transcribed wind by 1 ulp; the gate MUST then exit
non-zero -- the non-vacuity check for the gate itself.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # scripts/validate/ocean_fidelity
from rebuild_nemo_restart import rebuild  # noqa: E402

DEFAULT_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")

# --- namusr_def, cfgs/DINO/RUN_TRAJ/namelist_cfg:22-47 --------------------
RN_PHI_MIN, RN_PHI_MAX = -70.0, 70.0
RN_ZTAU0 = 0.2
RN_TSTAR_S, RN_TSTAR_N, RN_TSTAR_EQ = -0.5, 5.0, 27.0
RN_SSTAR_S, RN_SSTAR_N, RN_SSTAR_EQ = 35.0, 35.1, 37.25
RN_TRP, RN_SRP = -40.0, -3.858e-3
RN_DT = 2700.0                       # namdom:116
NYEAR_LEN = 360.0                    # nn_leapy = 30 -> 12 x 30-day months
RPI = 3.141592653589793              # phycst.F90:25

# Rule 1 dispositions for every restart variable that is part of the INITIAL
# STATE or of the kt=1 SURFACE FORCING.  Anything else the file carries is a
# post-step diagnostic and is filtered out by _POST_STEP below, not waived
# one by one (the restart has 131 variables, 120 of them trends and
# staggered dumps written by the #1226 instrumented build).
WAIVED = {
    "tn": "Kmm level = the state AFTER the Euler step, not the IC",
    "sn": "Kmm level = the state AFTER the Euler step, not the IC",
    "sshn": "Kmm level = the state AFTER the Euler step, not the IC",
    "ub": "istate.f90:125-126 sets uu(Kbb)=0; legoESM's rest state is 0 too, "
          "but the array is U-staggered on a different index frame here, so "
          "it is checked by the twin's face-mask census, not bitwise",
    "vb": "as ub",
    "un": "Kmm level = after the Euler step",
    "vn": "Kmm level = after the Euler step",
    "en": "zdftke's TKE AFTER one step (zdftke.F90 integrates en before the "
          "restart is written), so this record cannot gate the INITIAL en; "
          "NEMO initialises en = rn_emin*wmask at nit000",
    "avt_k": "vertical diffusivity after one TKE step",
    "avm_k": "vertical viscosity after one TKE step",
    "dissl": "TKE dissipation length after one step",
    "rhd": "in-situ density anomaly diagnosed during the step",
    "emp_b": "no legoESM counterpart array: the DINO card applies no "
             "freshwater flux (rn_emp_prop=0, ln_emp_field=F). Checked "
             "ORACLE-SIDE in section 2b as a property of the record",
    "qsr_hc_b": "traqsr.F90's per-level solar heat content; legoESM applies "
                "the Jerlov column tendency directly and stores no such "
                "array",
    "sbc_hc_b": "trasbc.F90 bookkeeping of the applied surface heat content",
    "sbc_sc_b": "trasbc.F90 bookkeeping of the applied surface salt content",
    "fraqsr_1lev": "traqsr.F90's fraction of qsr absorbed in level 1; a "
                   "consequence of the Jerlov coefficients, not an input",
    "kt": "file bookkeeping", "ndastp": "file bookkeeping",
    "adatrj": "file bookkeeping", "ntime": "file bookkeeping",
    "rdt": "file bookkeeping", "time_counter": "file bookkeeping",
    "nav_lat": "coordinate copy", "nav_lon": "coordinate copy",
    "nav_lev": "coordinate copy",
}

# Post-step instrumentation from the #1226 trend build: never part of the IC.
_POST_STEP = ("trd", "acc", "_stg", "nacc_", "rn_acc_", "uslp", "vslp",
              "wslpi", "wslpj", "uu_", "vv_", "sn_", "tn_", "rn2_")


def _is_post_step(name: str) -> bool:
    return any(tok in name for tok in _POST_STEP)


def _monotone(x: np.ndarray) -> np.ndarray:
    """IEEE-754 bit pattern remapped so integer order == float order.

    The naive `a.view(int64) - b.view(int64)` is NOT a ulp distance across
    zero: -5e-11 and +5e-11 are two ulps apart in float terms but 2^63 apart
    as raw patterns, which made a SIGN FLIP look like a huge-but-negative
    distance and slip through an `abs(...) <= waiver` test.
    """
    i = np.ascontiguousarray(x, dtype=np.float64).view(np.int64)
    return np.where(i < 0, np.int64(-(2 ** 63)) - i, i)


def _ulp(a, b) -> int:
    return int(np.max(np.abs(_monotone(np.asarray(a, dtype=np.float64))
                             - _monotone(np.asarray(b, dtype=np.float64)))))


class Table:
    def __init__(self):
        self.rows, self.failed = [], False
        self.checked, self.debt = set(), set()

    def check(self, name, built, oracle, *, mask=None, oracle_name=None,
              debt=False, note=""):
        """One row.  ``mask`` restricts the claim; ``debt`` reports without
        gating (and says DEBT out loud, never 'matched')."""
        built = np.asarray(built, dtype=np.float64)
        oracle = np.asarray(oracle, dtype=np.float64)
        # A DEBT row is NOT a verified row: keeping the two sets apart stops
        # the coverage tally from reading "verified" next to a row whose own
        # verdict says DEBT.
        (self.debt if debt else self.checked).add(oracle_name or name)
        if built.shape != oracle.shape:
            self.rows.append((name, -1, float("nan"), -1,
                              f"FAIL shape {built.shape} vs {oracle.shape}"))
            self.failed = True
            return
        if mask is not None:
            n_sel = int(np.count_nonzero(mask))
            if n_sel == 0:
                # A row over zero cells would print AT BAR and claim coverage
                # while asserting nothing at all.
                self.rows.append((name, -1, float("nan"), -1,
                                  "FAIL mask selects 0 cells (vacuous row)"))
                self.failed = True
                return
            built, oracle = built[mask], oracle[mask]
        ne = int(np.sum(built != oracle))
        mx = float(np.max(np.abs(built - oracle))) if ne else 0.0
        ulp = _ulp(built, oracle) if (ne and mx < 1e-9) else 0
        if not ne:
            verdict = "AT BAR"
        elif debt:
            verdict = "DEBT"
        else:
            verdict = "FAIL"
            self.failed = True
        self.rows.append((name, ne, mx, ulp, verdict + (" " + note if note else "")))

    def report(self):
        print(f"{'field':22s} {'cells!=':>9s} {'max|d|':>12s} {'ulp':>5s}  verdict")
        for n, ne, mx, ulp, v in self.rows:
            print(f"{n:22s} {ne:9d} {mx:12.3e} {ulp:5d}  {v}")


# --------------------------------------------------------------------------
# usrdef_sbc.F90 -- the ORACLE side, for section 3 only.  These are NEMO's
# formulas, transcribed here to certify the RECORD; they are deliberately NOT
# in legoesm/, because legoESM's DINO card does not compute NEMO-shaped flux
# arrays and a second implementation there would be a fiction.
# --------------------------------------------------------------------------
def _znl_cbc(nodes_phi, nodes_val, phi):
    """usrdef_sbc.F90:595-640 -- nearest node, then the adjacent one."""
    zdphi = nodes_phi - phi
    kmin = int(np.argmin(np.abs(zdphi)))
    ks, kn = (kmin, kmin + 1) if zdphi[kmin] <= 0 else (kmin - 1, kmin)
    zs = (phi - nodes_phi[ks]) / (nodes_phi[kn] - nodes_phi[ks])
    return nodes_val[ks] + (nodes_val[kn] - nodes_val[ks]) * (3. - 2. * zs) * zs ** 2


def _oracle_wind(gphiu):
    """usrdef_sbc.F90:162-163, :221 -- the wind nodes and the profile."""
    nodes_phi = np.array([RN_PHI_MIN - 1., RN_PHI_MIN, -45., -15., 0., 15.,
                          45., RN_PHI_MAX, RN_PHI_MAX + 1.])
    nodes_val = np.array([0., 0., RN_ZTAU0, -0.1, -0.02, -0.1, 0.1, 0., 0.])
    return np.array([[_znl_cbc(nodes_phi, nodes_val, float(p)) for p in row]
                     for row in gphiu])


def _cos_sais(kt=1):
    """compute_day_of_year (usrdef_sbc.F90:528-551), ln_ann_cyc = .true."""
    ztime = float(kt) * RN_DT / (60. * 60.)          # nyear = 1 at kt = 1
    ztimemax1 = ((5. * 30.) + 21.) * 24.
    ztimemin1 = ztimemax1 + 24. * NYEAR_LEN / 2
    ztimemax2 = ((6. * 30.) + 21.) * 24.
    ztimemin2 = ztimemax2 - 24. * NYEAR_LEN / 2
    return (math.cos((ztime - ztimemax1) / (ztimemin1 - ztimemax1) * RPI),
            math.cos((ztime - ztimemax2) / (ztimemax2 - ztimemin2) * RPI))


def _oracle_heat_salt(gphit, tmask_surf, sst, sss, kt=1):
    """usrdef_sbc.F90:206-280 -- sfx, qtot, qsr_dayMean, qns at kt.

    Also returns ztstar and the solar day-mean, so the caller can invert
    the oracle's own expression for the input it implies.
    """
    c1, c2 = _cos_sais(kt)
    ts_s = RN_TSTAR_S - 0.5 * c2                     # :216-217
    ts_n = RN_TSTAR_N + 3.0 * c2
    sh = gphit.shape
    ztstar = np.empty(sh)
    zsstar = np.empty(sh)
    zqsr = np.empty(sh)
    for j in range(sh[0]):
        for i in range(sh[1]):
            p = float(gphit[j, i])
            b = ts_s if p <= 0 else ts_n              # :233-239
            ztstar[j, i] = b + (RN_TSTAR_EQ - b) * math.sin(
                RPI * (p + RN_PHI_MAX) / (RN_PHI_MAX - RN_PHI_MIN))
            sb = RN_SSTAR_S if p <= 0 else RN_SSTAR_N  # :174-182
            zsstar[j, i] = (sb + (RN_SSTAR_EQ - sb)
                            * (1 + math.cos(2 * RPI * p
                                            / (RN_PHI_MAX - RN_PHI_MIN))) / 2
                            - 1.25 * math.exp(-p ** 2 / 7.5 ** 2))
            zqsr[j, i] = max(230. * math.cos(RPI * (p - 23.5 * c1) / 180.)
                             * float(tmask_surf[j, i]), 0.)          # :268
    sfx = (RN_SRP * (sss - zsstar)) * tmask_surf                     # :208
    qtot = RN_TRP * (sst - ztstar)                                   # :257 (emp = 0)
    qns = (qtot - zqsr) * tmask_surf                                 # :279
    return sfx, qns, c1, c2, ztstar, zqsr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart-glob", default=DEFAULT_RESTART)
    ap.add_argument("--plant", action="store_true",
                    help="move one wet initial-T cell and one wet U-point of "
                         "the wind by 1 ulp; the gate MUST then exit non-zero")
    args = ap.parse_args()

    # Rule 1c: an oracle comparison runs fp64, and it says so.
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64())

    from legoesm.ocean.experiments import dino as dino_mod
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

    if args.plant:
        _true_istate = ndm.nemo_dino_istate
        _true_hgr = ndm.nemo_dino_hgr

        def _planted_istate(g, nn_initcase=4):
            T, S = _true_istate(g, nn_initcase)
            j, i, k = 100, 25, 3            # an interior wet cell
            assert g.tmask[j, i, k] > 0.5
            T[j, i, k] = np.nextafter(T[j, i, k], np.inf)
            return T, S

        def _planted_hgr(nml=ndm.NEMO_DINO_R1):
            h = dict(_true_hgr(nml))
            phi = np.asarray(h["phi_t"])
            # 1 ulp on ONE interior latitude row -- small enough that the row
            # still passes the frame-identification bound, so it has to be
            # caught HERE and not by an exception upstream.  The row is CHOSEN
            # BY MEASUREMENT rather than hand-picked: the wind profile is a
            # smooth-step with ZERO derivative at every knot, so a 1-ulp lie
            # on a row that sits on a knot moves the stress by nothing and
            # would make this control a no-op (it silently was, at row 100).
            nodes_phi = np.array([RN_PHI_MIN - 1., RN_PHI_MIN, -45., -15., 0.,
                                  15., 45., RN_PHI_MAX, RN_PHI_MAX + 1.])
            nodes_val = np.array([0., 0., RN_ZTAU0, -0.1, -0.02, -0.1, 0.1,
                                  0., 0.])
            up = np.nextafter(phi, np.inf)
            move = np.array([
                abs(_znl_cbc(nodes_phi, nodes_val, float(b))
                    - _znl_cbc(nodes_phi, nodes_val, float(a)))
                for a, b in zip(phi, up)])
            j = int(np.argmax(move))
            if move[j] == 0.0:
                raise SystemExit("PLANT would be vacuous: no row's stress "
                                 "moves under a 1-ulp latitude change")
            print(f"           wind plant: row {j} (phi {phi[j]:.6f} deg) "
                  f"+1 ulp -> stress moves {move[j]:.3e} N/m2")
            for key in ("gphit", "gphiu", "phi_t"):
                a = np.array(h[key], copy=True)
                a[j] = up[j]
                h[key] = a
            return h

        # A THIRD PLANT, for the three restoring/solar rows.  The latitude
        # plant above does NOT cover them: those rows compare the forcing
        # against the SAME transcribed ladder, so a lie in the ladder moves
        # both sides together.  Nudge the delivered T* instead -- 1 ulp, the
        # smallest lie the row must catch.
        _true_forcing = dino_mod.dino_lat_lon_surface_forcing_arrays

        def _planted_forcing(grid, cfg=None, *, wind_lat_deg=None):
            f = dict(_true_forcing(grid, cfg, wind_lat_deg=wind_lat_deg))
            import jax.numpy as _jnp
            t = np.asarray(f["T_star_2d"], dtype=np.float64).copy()
            t[100, :] = np.nextafter(t[100, :], np.inf)
            f["T_star_2d"] = _jnp.asarray(t)
            return f

        dino_mod.dino_lat_lon_surface_forcing_arrays = _planted_forcing
        ndm.nemo_dino_istate = _planted_istate
        ndm.nemo_dino_hgr = _planted_hgr
        dino_mod._nemo_faithful_dino_domain.cache_clear()
        print("PLANT ACTIVE: 1 ulp on one wet initial-T cell, 1 ulp on the "
              "source latitude row where it moves the stress most, and "
              "1 ulp on one row of the delivered T*")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    print(f"legoESM {sha}   dirty_tracked_files="
          f"{len(dirty.splitlines()) if dirty else 0}")
    print(f"restart {args.restart_glob}")

    import glob
    tiles = sorted(glob.glob(args.restart_glob))
    if not tiles:
        raise SystemExit(f"no restart tiles match {args.restart_glob}")
    h = hashlib.sha256()
    for t in tiles:
        with open(t, "rb") as fh:
            h.update(fh.read())
    print(f"{len(tiles)} tiles, concatenated sha256[:16] = "
          f"{h.hexdigest()[:16]}   precision = "
          f"{get_policy().storage.__name__}\n")

    import netCDF4 as nc
    ds = nc.Dataset(tiles[0])
    all_vars = set(ds.variables)
    ds.close()

    want = ["tb", "sb", "sshb", "utau_b", "vtau_b", "emp_b", "qns_b", "sfx_b"]
    R = rebuild(args.restart_glob, want)
    for k in want:
        if k not in R or np.isnan(R[k]).any():
            raise SystemExit(f"restart field {k} missing or has gaps")

    def O3(k):       # (z,y,x) -> (y,x,z), matching nemo_io's array order
        return np.moveaxis(R[k], 0, -1)

    # ---- the model's own path (Rule 10) ---------------------------------
    cfg = dino_mod.nemo_faithful_dino_config(
        base=dino_mod.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dino_mod.dino_lat_lon_grid(cfg)
    z = dino_mod.dino_lat_lon_vertical(grid, cfg)
    st = dino_mod.dino_lat_lon_state(grid, z, cfg)
    forcing = dino_mod.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf = dino_mod.dino_step_surface_forcing(forcing)
    g = ndm.nemo_dino_mesh()
    print(f"built {grid.n_lat}x{grid.n_lon}x{z.n_levels}; "
          f"T dtype {np.asarray(st.T.data).dtype}, "
          f"grid.lat dtype {np.asarray(grid.lat).dtype}")

    t = Table()
    print("\nSECTION 1  initial state, through run_dino.py's own path")
    wet3 = g.tmask > 0.5
    t.check("T (all cells)", st.T.data, O3("tb"), oracle_name="tb")
    t.check("T (wet only)", st.T.data, O3("tb"), mask=wet3, oracle_name="tb")
    t.check("S (all cells)", st.S.data, O3("sb"), oracle_name="sb")
    t.check("S (wet only)", st.S.data, O3("sb"), mask=wet3, oracle_name="sb")
    t.check("eta", st.eta.data, R["sshb"], oracle_name="sshb")
    t.report()

    t2 = Table()
    print("\nSECTION 2  surface forcing, through run_dino.py's own path")
    umask_s = g.umask[:, :, 0] > 0.5
    tau = np.asarray(forcing["tau_u_cell_2d"])
    t2.check("utau (wet U-points)", tau, R["utau_b"], mask=umask_s,
             oracle_name="utau_b")
    # sbcmod.f90:571-572 -- what the file actually stores is utauU, the
    # U-point stress with the (2-umask) coastline unmasking; applying that
    # to legoESM's cell stress covers the land points the mask above skips.
    utauU = (0.5 * (tau + np.roll(tau, -1, axis=1))
             * (2. - g.umask[:, :, 0])
             * np.maximum(g.tmask[:, :, 0], np.roll(g.tmask[:, :, 0], -1, axis=1)))
    rows_ns = np.ones(tau.shape, bool)
    rows_ns[0] = False    # DO_2D(0,0,0,0) never computes the closed
    rows_ns[-1] = False   # first/last global rows, so they stay zero
    t2.check("utauU (all, rows 1..n-2)", utauU, R["utau_b"], mask=rows_ns,
             oracle_name="utau_b")
    t2.check("vtau", sf.tau_y, R["vtau_b"], oracle_name="vtau_b")
    taum_oracle = np.abs(R["utau_b"]) * np.where(R["utau_b"] > 0, 1.3, 1.0)
    t2.check("taum (wet U-points)", forcing["taum_2d"], taum_oracle,
             mask=umask_s, note="(oracle taum rebuilt from utau_b per "
                                "usrdef_sbc.F90:222-223; not itself dumped)")
    # THE SIGN THE DYNAMICS ACTUALLY RECEIVES.  usrdef_sbc's utau is the stress
    # ON the ocean; legoESM's PE external-tau block takes the ATMOSPHERIC
    # convention and applies -tau, so dino_step_surface_forcing negates.  Score
    # that array, not just the dict it came from, or a sign error between the
    # two is out of the gate's reach entirely.
    t2.check("tau_x into model.step", -np.asarray(sf.tau_x), R["utau_b"],
             mask=umask_s, oracle_name="utau_b",
             note="(the applicator's OCEAN-reaction negation, undone here)")
    # THE THREE TARGETS THE FORCING FOLD ALSO MOVES.  Without these rows the
    # T*/S*/solar half of the latitude fix is UNGATED: a reviewer sign-flipped
    # T*, offset it by 111 K, reversed S* and zeroed the solar flux and the
    # gate still said PASS.  There is no NEMO array to compare them against at
    # this instant (legoESM's are the ANNUAL MEAN; NEMO's are the kt=1
    # seasonal value), so what is pinned is the thing the diff changed: that
    # all three are evaluated at NEMO's OWN gphit, bit-for-bit.
    phi_t = ndm.nemo_dino_hgr()["phi_t"]
    import jax.numpy as jnp
    for key, fn in (("T_star", dino_mod.dino_T_star_annual_mean),
                    ("S_star", dino_mod.dino_S_star),
                    ("Q_sr", dino_mod.dino_Q_sr_annual_mean)):
        t2.check(f"{key} at gphit", forcing[f"{key}_2d"][:, 0],
                 np.asarray(fn(jnp.asarray(phi_t), cfg), dtype=np.float64),
                 oracle_name=f"{key}_at_gphit",
                 note="(evaluated at NEMO's stored gphit, usrdef_sbc.F90:"
                      "174-181/:233-238/:268)")
    # tau_u_face is the array the legacy (non-wind_through_step) applicator
    # reads; it must carry the same stress as the cell-centred one.
    t2.check("tau_u_face", np.asarray(forcing["tau_u_face"])[:, 1:],
             np.broadcast_to(tau[:, :1], tau.shape)
             if tau.shape[1] == 1 else np.repeat(tau[:, :1], tau.shape[1], 1),
             oracle_name="tau_u_face",
             note="(zonally uniform: every u-face carries the row's stress)")
    t2.report()

    # emp: an ORACLE-SIDE record check, not a legoESM claim.  legoESM's DINO
    # card carries no E-P array at all (rn_emp_prop = 0, ln_emp_field = F), so
    # comparing a gate-side zeros() against it could never fail on any model
    # change -- it would book emp_b as "verified" while asserting nothing.
    if not bool(np.all(R["emp_b"] == 0.0)):
        print("\nSECTION 2b  emp_b is NOT identically zero -- the record is "
              "not the deck we transcribe")
        t2.failed = True
    else:
        print("\nSECTION 2b  emp_b identically zero in the record "
              "(ln_emp_field=F, rn_emp_prop=0), and legoESM's DINO card "
              "applies no freshwater flux -- ORACLE-SIDE, see WAIVED")

    t3 = Table()
    print("\nSECTION 3  ORACLE SIDE ONLY -- NEMO's kt=1 heat/salt fluxes "
          "recomputed from usrdef_sbc.F90.\n           Certifies the RECORD "
          "and the seasonal clock; does NOT certify legoESM.")
    sfx_o, qns_o, c1, c2, _ztstar, _zqsr = _oracle_heat_salt(
        g.gphit, g.tmask[:, :, 0], R["tb"][0], R["sb"][0])
    print(f"           kt=1 clock: ztime = 0.75 h, cos_sais1 = {c1!r}, "
          f"cos_sais2 = {c2!r}")
    t3.check("sfx (oracle-side)", sfx_o, R["sfx_b"], oracle_name="sfx_b")
    t3.check("qns (oracle-side)", qns_o, R["qns_b"], oracle_name="qns_b",
             debt=True,
             note="(DEBT: last-bit only, in the SOLAR term -- see the "
                  "inversion below)")
    t3.report()
    # WHERE THE qns DEBT LIVES, re-measured every run rather than asserted.
    # Invert the oracle's own solar expression for the input it implies
    # (Rule 1d's trick): qns_b = (qtot - zqsr)*tmask with qtot exact, so
    # zqsr_implied = qtot - qns_b, and acos(zqsr/230) gives back the latitude
    # NEMO's COS actually saw.
    #
    # TWO HYPOTHESES ARE RETRACTED HERE, both by measurement:
    #  * "the seasonal-clock cosine's last ulp" -- REFUTED: nudging
    #    zcos_sais1 by +-1..6 ulp makes the count WORSE in both directions
    #    (5304 unequal at 0 ulp; 5652 at -1, 5798 at +1), so 0 is optimal.
    #  * "glibc's vector cosine, as for ff_t/ff_f" -- REFUTED:
    #    objdump --disassemble='__usrdef_sbc_MOD_usrdef_sbc_oce' BLD/bin/
    #    nemo.exe shows 31 SCALAR cos@plt, 18 exp@plt, 3 sin@plt and NO
    #    _ZGV* vector call in this routine.
    # What survives: the implied latitude offset has median exactly 0 with a
    # SYMMETRIC +-1..3 ulp spread (p5/p95 ~ -2.9e-14 / +4.5e-14 deg, against
    # ulp(69 deg) = 1.4e-14), i.e. the ARGUMENT differs in its last bits --
    # the signature of the compiled binary contracting
    # rpi*(gphit - 23.5*zcos_sais1)/180 with an FMA, not of a wrong constant.
    # ORACLE-SIDE and last-bit, so it changes no legoESM number.
    _wetT = g.tmask[:, :, 0] > 0.5
    _qtot = RN_TRP * (R["tb"][0] - _ztstar)
    _implied = (_qtot - R["qns_b"])[_wetT]
    _mine = _zqsr[_wetT]
    _pos = _mine > 0
    if int(_pos.sum()):
        _a = np.ascontiguousarray(_mine[_pos], dtype=np.float64)
        _b = np.ascontiguousarray(_implied[_pos], dtype=np.float64)
        _ul = _a.view(np.int64) - _b.view(np.int64)
        _u, _c = np.unique(_ul, return_counts=True)
        _arg = RPI * (g.gphit[_wetT][_pos] - 23.5 * c1) / 180.
        _argi = np.arccos(np.clip(_implied[_pos] / 230., -1, 1)) * np.sign(_arg)
        _off = (_argi - _arg) * 180. / RPI
        print(f"           qns solar term: ulp offsets "
              f"{dict(zip(_u.tolist(), _c.tolist()))}")
        print(f"           implied latitude offset: median "
              f"{np.median(_off):+.3e} deg, p5 {np.percentile(_off, 5):+.3e}, "
              f"p95 {np.percentile(_off, 95):+.3e}  "
              f"(ulp(69 deg) = {np.spacing(69.0):.1e})")
    # Section 3 is oracle-side: it must not gate legoESM, but a BROKEN
    # transcription here would silently weaken the coverage claim, so the
    # bit-exact row is still fatal.
    t.failed = t.failed or t2.failed or t3.failed

    print("\nSECTION 4  coverage (oracle-fidelity Rule 1)")
    ic_sbc = {v for v in all_vars if not _is_post_step(v)}
    verified = t.checked | t2.checked | t3.checked
    debt = (t.debt | t2.debt | t3.debt) - verified
    unaccounted = sorted(ic_sbc - verified - debt - set(WAIVED))
    print(f"  {len(ic_sbc)} initial-state / surface-forcing variables in the "
          f"restart ({len(all_vars)} total, {len(all_vars) - len(ic_sbc)} "
          f"post-step trend/staggered dumps)")
    print(f"  {len(verified & ic_sbc)} verified, {len(debt & ic_sbc)} DEBT, "
          f"{len(set(WAIVED) & ic_sbc)} waived, {len(unaccounted)} unaccounted")
    for k in sorted(debt & ic_sbc):
        print(f"  DEBT    {k}: measured, NOT at bar -- see the row above")
    for k in sorted(set(WAIVED) & ic_sbc):
        print(f"  WAIVED  {k}: {WAIVED[k]}")
    if unaccounted:
        print(f"  UNACCOUNTED (hard failure): {unaccounted}")
        t.failed = True

    print("\nGATE " + ("FAIL" if t.failed else "PASS"))
    return 1 if t.failed else 0


if __name__ == "__main__":
    sys.exit(main())
