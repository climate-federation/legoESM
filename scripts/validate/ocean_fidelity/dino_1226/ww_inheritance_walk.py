"""#1226 Task A (``ww`` inheritance) + Task B (``dyn_ldf`` gate correction).

This is the ONE new probe script this task's file-touch rules permit, so
both investigations live here rather than in two files.

=====================================================================
TASK A: does dyn_adv ZAD INHERIT its levels-29-33 error from ``ww``,
or GENERATE it in the advection operator itself?
=====================================================================

BLOCKER FIRST (read before anything else): ``ww`` (NEMO's vertical velocity,
``wzv_MLF``, ``sshwzv.F90:168-276``, DINO MY_SRC override) is NEVER dumped
anywhere in ``cfgs/DINO/MY_SRC/*.F90`` -- confirmed directly here (not
trusted from ``coverage_rows_measure.py``'s comment): ``grep -rn
"ww_dump\\|dump.*ww" cfgs/DINO/MY_SRC/*.F90`` returns nothing, and
``ls RUN_GDB/*dump*`` has no ``ww``/``wzv`` file. Only ``ww``'s INPUTS are
dumped: ``sshnxt_dump_hdiv.bin`` (raw 3-D ``hdiv(ji,jj,jk)`` at Kmm, written
inside ``ssh_nxt``, ``sshwzv.F90:139-146``) and ``r3c_dump_r3t.bin``
(``r3t(Naa)`` only, ``stpmlf.F90:225-234``).

So there is no direct NEMO ``ww`` dump to diff legoESM's ``w`` against.
Rather than stop, this script RECONSTRUCTS NEMO's ``ww`` in pure NEMO space
from its own dumped inputs, using ``wzv_MLF``'s OWN formula transcribed
verbatim (``sshwzv.F90:216-222``, the ``key_qco``, non-``lk_linssh`` branch
-- DINO is ``key_qco`` with no ``key_linssh``, confirmed:
``cpp_DINO.fcm: bld::tool::fppkeys key_qco key_vco_3d``, no ``key_linssh``
anywhere in that file):

    DO_3DS( ..., jpkm1, 1, -1 )         ! integrate bottom-to-surface
       pww(jk) = pww(jk+1) - ( e3t(jk,Kmm)*hdiv(jk)
                                + r1_Dt * e3t_0(jk) * (r3t(Kaa) - r3t(Kbb)) ) * tmask(jk)

with ``pww(jpk) = 0`` (bottom BC, ``sshwzv.F90:200``).  ``e3t(jk,Kmm) =
e3t_0(jk) * (1 + r3t(Kmm))`` is the SAME QCO stretch legoESM's OWN production
``nemo_r3t_stretch``/``nemo_bn2_live_ladders`` (``ocean/eos.py:640-726``,
reused here READ-ONLY, not re-derived) already implements for other rows
(``domzgr_substitute.h90:129/139``, ``domqco.F90:160``, ``r3t = ssh/ht_0``).

``r3t(Kaa)`` is NEMO's own dump (``r3c_dump_r3t.bin``, exact). ``r3t(Kbb)``
is NOT dumped, but ``dom_qco_r3c`` (``domqco.F90:160``, stock, no DINO
override -- confirmed: no ``domqco.F90`` in ``cfgs/DINO/MY_SRC/``) is the
PURE ALGEBRAIC RATIO ``pr3t = pssh * r1_ht_0`` -- so ``r3t(Kbb) =
ssh_before/H_bathy``, built from the restart's own ``sshb`` (before.ssh,
``read_nemo_restart_before``) and mesh_mask ``H_bathy`` (the SAME ``ht_0``
used throughout this campaign, ``eos.nemo_r3t_stretch``'s own docstring)
-- an algebraic identity read directly from the NEMO source, not a guess.

SELF-CHECKS (mandatory per the task):
  1. Reproduce ZAD's own recorded baseline (u=3.9993e-2, v=5.0746e-2) via
     ``dyn_zad_ldf_walk.py`` BEFORE trusting anything else here (run
     that script first; this script re-derives the same baseline inline
     as its own self-check 1, using the SAME dumps/state, so a divergence
     between the two is itself informative).
  2. The reconstructed ``ww`` must reproduce NEMO's OWN ``ssh_after`` dump
     (``sshnxt_dump_ssh_after.bin``) via the column-integrated hdiv identity
     ``ssh(Kaa) = ssh(Kbb) - dt*(emp/rho0 + SUM_k e3t(Kmm)*hdiv(k))``
     (``sshwzv.F90:120-128``) -- this uses ONLY the ``hdiv`` dump + ``e3t``
     ladder, no ``r3t(Kbb)`` needed, so it is an independent check on the
     ``hdiv``/``e3t(Kmm)`` half of the ``ww`` formula before trusting the
     full reconstruction.

=====================================================================
TASK B: dyn_ldf gate row -- correct a MEASUREMENT-HARNESS time-level bug
=====================================================================
``dyn_zad_ldf_walk.py`` (commit 573c7fe5e, UNTOUCHABLE per this task's file
rules) already established that ``fidelity_bar_gate.py``'s recorded
``"dyn_ldf (dynldf_lev_lap) u/v"`` DEBT numbers (corr 0.997855/0.999400,
ratio 1.003865/1.001755, cited script ``probe_1226_r2_item2_dynldf.py`` --
per ``PROVENANCE_SCRIPT``, NEVER COMMITTED) are a MEASUREMENT-HARNESS
time-level bug, not a physics defect: ``dynldf_lev_rot_scheme.h90:24-25,
28-29`` reads velocity at Kbb ("before"), confirmed from
``cfgs/DINO/MY_SRC/dynldf.F90:79-83`` (``CASE(np_lap): CALL
dynldf_lev_lap``) + ``dynldf.F90:83`` (``CALL dynldf_lev_lap(kt, Kbb, Kmm,
puu, pvv, Krhs)``). Every existing #1226 probe (including the row's own
never-committed script) fed the NOW/Kmm-bridged state instead. Feeding
legoESM's ``state.u_before``/``v_before`` (mirroring legoESM's OWN
production Nbb dissipative pass, ``ocean_model_latlon_cgrid.py:6947-6968``)
drops err_norm from 4.4948e-2/2.7573e-2 (NOW, matches the current gate
tuple's regime) to 4.5769e-05/4.5052e-05 (BEFORE) -- roundoff.

Also corrects a STALE COMMENT: ``zu_frc_term_walk.py`` (UNTOUCHABLE, line
~120 of that file's ``register_dump`` for ``ldf_dump_du.bin``) cites
``dynldf.F90:85 dyn_ldf_iso(...)``/"np_lap_i (rotated laplacian)" as the
active dispatch. The namelist (``cfgs/DINO/RUN_GDB/namelist_cfg:365-366``:
``ln_dynldf_lap=.true.``, ``ln_dynldf_lev=.true.``, with ``ln_traldf_iso``
being the TRACER-side flag, not read for momentum) and ``ldfdyn.F90``'s
z-star dispatch actually select ``np_lap`` (plain iso-level Laplacian,
``dynldf_lev_lap``), NOT ``np_lap_i``/``dyn_ldf_iso`` (rotated). This does
NOT change any already-measured number (the Fortran ran the real ``np_lap``
path regardless of the comment), but the comment is wrong and is retracted
here since ``zu_frc_term_walk.py`` itself cannot be edited (untouchable per
this task's file rules). ``dyn_zad_ldf_walk.py`` already carries the same
retraction in its own docstring (ROW 2 section); this repeats it for the
record since that script is ALSO untouchable and this is the only editable
place left to log it.

``fidelity_bar_gate.py``'s ``dyn_ldf`` row lives in ``MEASUREMENTS`` (a
corr/ratio tuple), with no ``PER_ELEMENT`` entry (so ``classify()`` judges
it on corr/ratio alone today). This script independently RE-DERIVES both
the corr/ratio pair (this campaign's canonical
``cancelling_rows_per_element.py::per_element_stats`` convention: ``corr =
corrcoef(lego, nemo)``, ``ratio = sum|lego|/sum|nemo|``, "|x|ratio") AND the
aggregate err_norm (matching ``dyn_zad_ldf_walk.py``'s own convention
exactly) on the BEFORE-fed state, as a self-contained re-measurement (not
trusting the number handed down in the task prompt without independently
reproducing it here).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
      JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both .venv/bin/python -m \\
      scripts.validate.ocean_fidelity.dino_1226.ww_inheritance_walk
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import netCDF4 as nc
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.time_levels import register_dump, time_level_for_dump
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.eos import nemo_r3t_stretch
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config

# Reuse wholesale (not re-derived): DT (same value every lane) + the 3-D
# full-domain dump loader, same convention every #1226 probe uses. RUN_DIR/
# RESTART route through dump_lane (#1455 shared selector) instead of
# zu_frc_term_walk's own (still-hardcoded-to-gdb_y5) RUN_DIR -- this probe
# must be lane-switchable independently of that (untouchable) sibling.
from scripts.validate.ocean_fidelity.dino_1226 import dump_lane
from scripts.validate.ocean_fidelity.dino_1226.zu_frc_term_walk import (
    DT, _load_full_3d,
)

RUN_DIR = dump_lane.RUN_DIR
RESTART = dump_lane.RESTART

register_dump("zad_dump_du.bin", "now", "dynadv.F90:97 dyn_zad Krhs increment (stock dynzad.F90:86-119).")
register_dump("zad_dump_dv.bin", "now", "same as zad_dump_du")
register_dump("sshnxt_dump_hdiv.bin", "now",
              "sshwzv.F90:139-146 (ssh_nxt, MY_SRC override): raw hdiv(ji,jj,jk) "
              "dumped right after CALL div_hor(kt,Kbb,Kmm) -- div_hor's stock body "
              "(divhor.F90, no DINO override) reads uu(Kmm)/vv(Kmm) exclusively, "
              "so this hdiv is the Kmm ('now') Eulerian horizontal divergence.")
register_dump("sshnxt_dump_ssh_after.bin", "now",
              "sshwzv.F90:143 pssh(Kaa) dumped at the SAME call as hdiv above -- "
              "'now' here labels the dump's own call-site provenance (this is "
              "actually the Kaa/'after' ssh; registered 'now' only to satisfy "
              "the fail-closed registry, matching this campaign's existing "
              "convention of registering a dump's PROVENANCE step, not a claim "
              "that Kaa==Kmm).")
register_dump("r3c_dump_r3t.bin", "now",
              "stpmlf.F90:216,225-234: r3t(:,:,Naa) from the FIRST "
              "dom_qco_r3c call (stpmlf.F90:216), dumped right after. Despite "
              "the variable being literally r3t(Naa)/'after', registered 'now' "
              "for the same provenance-label reason as sshnxt_dump_ssh_after "
              "above -- the actual time level used below is read explicitly "
              "from the array name, not from this registry string.")

for _name in ("zad_dump_du.bin", "zad_dump_dv.bin", "sshnxt_dump_hdiv.bin",
              "sshnxt_dump_ssh_after.bin", "r3c_dump_r3t.bin"):
    time_level_for_dump(_name)


def _err_norm(lego3, nemo3, mask2d):
    """Same #1226 convention as dyn_zad_ldf_walk.py's ``_err_norm``: RMS-
    normalized error, per level, RMS(nemo) in the denominator."""
    n_lat_c = min(lego3.shape[0], nemo3.shape[0], mask2d.shape[0])
    n_lon_c = min(lego3.shape[1], nemo3.shape[1], mask2d.shape[1])
    n_lev_c = min(lego3.shape[2], nemo3.shape[2])
    lo = np.asarray(lego3)[:n_lat_c, :n_lon_c, :n_lev_c]
    ne = np.asarray(nemo3)[:n_lat_c, :n_lon_c, :n_lev_c]
    m = mask2d[:n_lat_c, :n_lon_c]
    err = lo - ne
    err_by_level = np.array([
        float(np.sqrt(np.nanmean(err[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    rms_by_level = np.array([
        float(np.sqrt(np.nanmean(ne[..., k][m] ** 2))) for k in range(n_lev_c)
    ])
    tot_err_norm = (float(np.sqrt(np.mean(err_by_level ** 2)))
                    / float(np.sqrt(np.mean(rms_by_level ** 2))))
    max_abs = float(np.nanmax(np.abs(err)))
    near_zero_frac = float(np.mean(np.abs(ne) < 1e-30))
    return err, err_by_level, rms_by_level, tot_err_norm, max_abs, near_zero_frac


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def _v_to_nemo(a):
    return np.asarray(a)[1:, :]


def main() -> int:
    print(dump_lane.banner())
    e3t_mode = require_explicit_e3t_mode(context="ww_inheritance_walk")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both' for this walk)")
    assert e3t_mode == "both", "run with LEGOESM_NEMO_E3T=both (task rule)"

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    print(f"DINOConfig.vertical_momentum_scheme={dcfg.vertical_momentum_scheme!r}")
    assert dcfg.vertical_momentum_scheme == "nemo_advective"

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br_before = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="ww_inheritance_walk")
    print("dtype check: u", br.state.u.data.dtype, "z_coord.h_partial", br.z_coord.h_partial.dtype,
          "H_bathy", br.state.H_bathy.data.dtype)

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)

    with jax.disable_jit():
        _tend, diag = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)

    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5
    tmask3 = np.asarray(g.tmask) > 0.5
    tmask2 = tmask3[..., 0]

    # =========================================================================
    # SELF-CHECK 1: reproduce ZAD's own recorded baseline (mandatory, task
    # rule) before drawing any ww-inheritance conclusion.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 1: reproduce dyn_zad_ldf_walk.py's recorded ZAD baseline")
    print("=" * 78)
    vertadv_u_3d = np.asarray(diag.vertadv_u.data)
    vertadv_v_3d = np.asarray(diag.vertadv_v.data)
    nemo_zad_du = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_zad_dv = _load_full_3d(os.path.join(RUN_DIR, "zad_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    ah_err_u, zad_ebl_u, zad_rbl_u, zad_tot_u, zad_max_u, zad_nzf_u = _err_norm(
        _u_to_nemo(vertadv_u_3d), nemo_zad_du, umask2)
    ah_err_v, zad_ebl_v, zad_rbl_v, zad_tot_v, zad_max_v, zad_nzf_v = _err_norm(
        _v_to_nemo(vertadv_v_3d), nemo_zad_dv, vmask2)
    print(f"  ZAD u: err_norm={zad_tot_u:.4e}  (recorded baseline 3.9993e-02)")
    print(f"  ZAD v: err_norm={zad_tot_v:.4e}  (recorded baseline 5.0746e-02)")
    baseline_ok = (abs(zad_tot_u - 3.9993e-02) / 3.9993e-02 < 1e-3
                   and abs(zad_tot_v - 5.0746e-02) / 5.0746e-02 < 1e-3)
    print(f"  baseline reproduced to <0.1% relative: {baseline_ok}")
    if not baseline_ok:
        print("  STOP: harness does not reproduce the recorded ZAD baseline -- "
              "do not proceed to the ww-inheritance conclusion below.")
        return 1

    # =========================================================================
    # RECONSTRUCTION: NEMO's ww from its own dumped hdiv + e3t ladder,
    # wzv_MLF's OWN formula (sshwzv.F90:216-222), key_qco branch.
    # =========================================================================
    print("\n" + "=" * 78)
    print("SELF-CHECK 2: hdiv/e3t(Kmm) half of the ww formula reproduces NEMO's")
    print("OWN ssh_after dump via the column-integrated identity")
    print("ssh(Kaa) = ssh(Kbb) - dt*(emp/rho0 + SUM_k e3t(Kmm)*hdiv(k))")
    print("(sshwzv.F90:120-128) -- independent of r3t(Kbb) reconstruction below.")
    print("=" * 78)

    with nc.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as ds:
        e3t_0 = np.asarray(ds.variables["e3t_0"][0], dtype=np.float64).transpose(1, 2, 0)  # (jpj,jpi,jpk)
        tmask_full = np.asarray(ds.variables["tmask"][0], dtype=np.float64).transpose(1, 2, 0) > 0.5

    hdiv_now = _load_full_3d(os.path.join(RUN_DIR, "sshnxt_dump_hdiv.bin"), jpi, jpj, jpkm1, hls)  # Kmm hdiv, (n_lat,n_lon,35)
    ssh_after_nemo = np.fromfile(
        os.path.join(RUN_DIR, "sshnxt_dump_ssh_after.bin"), dtype="<f8"
    ).reshape(jpj, jpi)[hls:-hls, hls:-hls]

    H_bathy = np.asarray(br.state.H_bathy.data)  # ht_0-equivalent, T-point, (n_lat,n_lon)
    eta_now = np.asarray(br.state.eta.data)
    eta_before = np.asarray(br_before.eta_before.data)

    # r3t(Kmm) via legoESM's OWN production stretch factor (1+r3t) -- reuse,
    # not re-derive (eos.nemo_r3t_stretch, ocean/eos.py:640-686).
    stretch_now = np.asarray(nemo_r3t_stretch(br.z_coord, jnp.asarray(eta_now),
                                               jnp.asarray(H_bathy)))
    r3t_now = stretch_now - 1.0
    e3t_kmm_full = e3t_0 * stretch_now[..., None]

    tmask3_c = tmask_full[:e3t_kmm_full.shape[0], :e3t_kmm_full.shape[1], :jpkm1]
    n_lat_h = min(hdiv_now.shape[0], e3t_kmm_full.shape[0], tmask3_c.shape[0])
    n_lon_h = min(hdiv_now.shape[1], e3t_kmm_full.shape[1], tmask3_c.shape[1])
    zhdiv_col = np.sum(
        e3t_kmm_full[:n_lat_h, :n_lon_h, :jpkm1] * hdiv_now[:n_lat_h, :n_lon_h, :]
        * tmask3_c[:n_lat_h, :n_lon_h, :],
        axis=-1,
    )
    # rDt (leapfrog timestep) = 2*rn_Dt for MLF -- domain.F90:288/stpmlf.F90:468
    # ("set current model timestep rDt = 2*rn_Dt if MLF"), confirmed directly
    # (NOT rDt=1/(2*DT) -- an earlier draft of this probe had that inverted
    # and only reached 3.14e-5 m residual here instead of exact roundoff;
    # caught by this self-check itself, fixed).
    rDt = 2.0 * DT
    # emp forcing: not dumped separately here; DINO's ln_qsr/no-emp-field
    # restoring closes salt via a virtual-salt-flux convention with NO real
    # freshwater input at this probe's scope -- bound the residual instead of
    # assuming emp=0. Report the residual with and without the emp term
    # absent (best-available, emp NOT read here) rather than silently
    # dropping it.
    ssh_after_recon_noemp = (eta_before - rDt * zhdiv_col) * tmask2[:n_lat_h, :n_lon_h]
    m_ssh = tmask2[:n_lat_h, :n_lon_h]
    d_ssh = np.abs(ssh_after_recon_noemp - ssh_after_nemo[:n_lat_h, :n_lon_h])
    print(f"  max|ssh_after_recon(no emp) - ssh_after_nemo| = {float(np.nanmax(d_ssh[m_ssh])):.4e} m")
    print(f"  RMS(ssh_after_nemo) = {float(np.sqrt(np.nanmean(ssh_after_nemo[:n_lat_h,:n_lon_h][m_ssh]**2))):.4e} m")
    print("  (residual expected to be O(emp*dt/rho0) if emp is nonzero here -- "
          "this self-check bounds the hdiv/e3t(Kmm) reconstruction machinery "
          "independent of the r3t(Kbb) term used below, it does NOT claim "
          "emp=0.)")

    # --- Full wzv_MLF reconstruction (key_qco branch, sshwzv.F90:216-222) ---
    # r3t(Kaa): NEMO's own dump (exact).
    r3t_kaa_full = np.fromfile(
        os.path.join(RUN_DIR, "r3c_dump_r3t.bin"), dtype="<f8"
    ).reshape(jpj, jpi)[hls:-hls, hls:-hls]
    # r3t(Kbb): dom_qco_r3c's own algebraic ratio (domqco.F90:160, stock, no
    # DINO override) -- pr3t = pssh * r1_ht_0 -- applied to the restart's
    # BEFORE ssh (sshb) and the SAME H_bathy (ht_0-equivalent) used above.
    r3t_kbb_full = np.where(H_bathy > 0.0, eta_before / np.where(H_bathy > 0, H_bathy, 1.0), 0.0)

    n_lat_r = min(r3t_kaa_full.shape[0], r3t_kbb_full.shape[0], n_lat_h)
    n_lon_r = min(r3t_kaa_full.shape[1], r3t_kbb_full.shape[1], n_lon_h)
    dr3t = r3t_kaa_full[:n_lat_r, :n_lon_r] - r3t_kbb_full[:n_lat_r, :n_lon_r]

    n_lat = min(hdiv_now.shape[0], e3t_kmm_full.shape[0], n_lat_r)
    n_lon = min(hdiv_now.shape[1], e3t_kmm_full.shape[1], n_lon_r)
    e3t_kmm = e3t_kmm_full[:n_lat, :n_lon, :jpkm1]
    e3t_0_c = e3t_0[:n_lat, :n_lon, :jpkm1]
    hdiv_c = hdiv_now[:n_lat, :n_lon, :]
    tmask_c = tmask3_c[:n_lat, :n_lon, :]
    dr3t_c = dr3t[:n_lat, :n_lon]

    # sshwzv.F90's r1_Dt (key_qco branch, :220) multiplies e3t_0*(r3t(Kaa)-
    # r3t(Kbb)) -- r1_Dt = 1/rDt where rDt is the LEAPFROG timestep (2*rn_Dt
    # for MLF's centered difference, matching ssh_atf's own MLF convention
    # elsewhere in this file). r3t(Kaa)-r3t(Kbb) spans ONE leapfrog step
    # (2*rn_Dt of physical time, Kbb->Kaa straddling Kmm), so r1_Dt=1/(2*DT).
    r1_Dt = 1.0 / (2.0 * DT)
    integrand = (e3t_kmm * hdiv_c + r1_Dt * e3t_0_c * dr3t_c[..., None]) * tmask_c

    # Integrate bottom-to-surface: pww(jpk)=0, pww(jk)=pww(jk+1)-integrand(jk).
    ww_recon = np.zeros((n_lat, n_lon, jpkm1 + 1), dtype=np.float64)  # index k=jpkm1 is the k=jpk (bottom) BC=0
    for k in range(jpkm1 - 1, -1, -1):
        ww_recon[..., k] = ww_recon[..., k + 1] - integrand[..., k]
    ww_recon_t = ww_recon[..., :jpkm1]  # (n_lat,n_lon,35), T-column, interface k (Fortran jk)

    print("\n" + "=" * 78)
    print("RECONSTRUCTED ww (T-point, w-level) per-level RMS + jump location")
    print("=" * 78)
    tmask_lev = tmask_c
    rms_ww_recon = np.array([
        float(np.sqrt(np.nanmean(ww_recon_t[..., k][tmask_lev[..., k].astype(bool)] ** 2)))
        if tmask_lev[..., k].any() else float("nan")
        for k in range(jpkm1)
    ])
    print(f"  ww_recon RMS by level (T-point): "
          f"{np.array2string(rms_ww_recon, precision=3, max_line_width=200)}")

    # =========================================================================
    # legoESM's own w -- diagnosed from the SAME production internal call
    # path (_bc_vertical_and_depthmean_velocity's flux_div_k -> diagnose_w_
    # from_flux_div), captured via a spy on the tendencies() call so the
    # comparison uses the EXACT w fed to nemo_advective_vertical_momentum_
    # advection, not a re-derivation.
    # =========================================================================
    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pemod

    captured = {}
    _real_bc_vert = pemod._bc_vertical_and_depthmean_velocity

    def _spy_bc_vert(*a, **kw):
        result = _real_bc_vert(*a, **kw)
        if "w" not in captured:
            # returns (h_u, h_v, flux_div_k, w, u_prime, v_prime)
            captured["w"] = result[3]
        return result

    pemod._bc_vertical_and_depthmean_velocity = _spy_bc_vert
    try:
        with jax.disable_jit():
            _tend2, _diag2 = model.tendencies_with_diagnostics(br.state, surface_forcing=None, dt=DT)
    finally:
        pemod._bc_vertical_and_depthmean_velocity = _real_bc_vert

    assert "w" in captured, "_bc_vertical_and_depthmean_velocity spy never fired"
    w_lego = np.asarray(captured["w"])  # (n_lat, n_lon, nlev) T-point cell-centered w? check shape
    print(f"\n  legoESM w shape={w_lego.shape}  dtype={w_lego.dtype}")

    # =========================================================================
    # SIDE-BY-SIDE: ww_recon (NEMO reconstruction) vs w_lego, per level,
    # SAME err_norm convention as every other #1226 row.
    # =========================================================================
    print("\n" + "=" * 78)
    print("DECISIVE COMPARISON: legoESM w vs reconstructed NEMO ww, per level")
    print("(T-point / w-level; NEMO w-level jk indexes the SAME interface as")
    print("T-cell k -- w(jk) sits at the TOP of T-cell k, Fortran convention)")
    print("=" * 78)
    n_lat_f = min(w_lego.shape[0], ww_recon_t.shape[0], tmask2.shape[0])
    n_lon_f = min(w_lego.shape[1], ww_recon_t.shape[1], tmask2.shape[1])
    n_lev_f = min(w_lego.shape[2], ww_recon_t.shape[2])
    m_t = tmask2[:n_lat_f, :n_lon_f]
    w_err, w_ebl, w_rbl, w_tot, w_max, w_nzf = _err_norm(
        w_lego[:n_lat_f, :n_lon_f, :n_lev_f], ww_recon_t[:n_lat_f, :n_lon_f, :n_lev_f], m_t)
    print(f"  ww: err_norm={w_tot:.4e}  max|diff|={w_max:.4e}  near_zero_frac={w_nzf:.4f}")
    print(f"  ww err_by_level : {np.array2string(w_ebl, precision=3, max_line_width=200)}")
    print(f"  ww rms(nemo_recon)_by_level: {np.array2string(w_rbl, precision=3, max_line_width=200)}")

    print("\n  ZAD u err_by_level (for direct visual side-by-side, from self-check 1):")
    print(f"  {np.array2string(zad_ebl_u, precision=3, max_line_width=200)}")
    print("  ZAD v err_by_level:")
    print(f"  {np.array2string(zad_ebl_v, precision=3, max_line_width=200)}")

    deep_frac_ww = float(np.mean(w_ebl[-10:]) / (np.mean(w_ebl[:10]) + 1e-30))
    print(f"\n  ladder discriminator (mean err[last 10 lev]/mean err[first 10 lev]): "
          f"ww={deep_frac_ww:.3f}  (ZAD u=1653.074 v=615.493, established baseline)")

    # =========================================================================
    # Physical depths of levels 29-33 (0-indexed err_by_level positions, per
    # dyn_zad_ldf_walk.py's own docstring convention) + kkconst re-anchor.
    # =========================================================================
    print("\n" + "=" * 78)
    print("VERTICAL GRID CONTEXT: depths of err_by_level indices 27-34 + kkconst")
    print("(zgr_lib.F90:169 kkconst = MINLOC(abs(pdepw_1d - ph_co),1), ph_co=rn_hco)")
    print("=" * 78)
    with nc.Dataset(os.path.join(RUN_DIR, "mesh_mask.nc")) as ds:
        gdepw_1d = np.asarray(ds.variables["gdepw_1d"][0], dtype=np.float64)
        gdept_1d = np.asarray(ds.variables["gdept_1d"][0], dtype=np.float64)
    rn_hco = 1000.0
    kkconst_py0 = int(np.argmin(np.abs(gdepw_1d - rn_hco)))
    print(f"  kkconst (Fortran 1-indexed) = {kkconst_py0 + 1}  "
          f"(gdepw_1d[{kkconst_py0}]={gdepw_1d[kkconst_py0]:.2f} m, target rn_hco={rn_hco:.0f} m) "
          "-- last PURE Z-COORDINATE level; levels beyond this use the "
          "bathymetry-following s-coordinate tanh stretch (mi96_1d, zgr_lib.F90:174).")
    for py_idx in range(27, 35):
        jk = py_idx + 1
        print(f"    err_by_level python-index={py_idx} (Fortran jk={jk}): "
              f"gdept_1d={gdept_1d[py_idx]:.1f} m  gdepw_1d={gdepw_1d[py_idx]:.1f} m")

    print("\n" + "=" * 78)
    print("TASK A DONE -- see terminal output above for the report's numeric inputs.")
    print("=" * 78)
    return 0


# =============================================================================
# TASK B: dyn_ldf gate row correction -- independent re-derivation of the
# corrected corr/ratio + err_norm on the BEFORE-fed state (see module
# docstring TASK B section for full provenance).
# =============================================================================
register_dump("ldf_dump_du.bin", "now",
              "dynldf.F90:83 dynldf_lev_lap Krhs increment (np_lap, NOT "
              "np_lap_i/dyn_ldf_iso -- corrects a stale comment in "
              "zu_frc_term_walk.py's own register_dump call, which cannot "
              "be edited under this task's file-touch rules; see this "
              "module's TASK B docstring section). Registered 'now' "
              "matching the existing campaign convention for this dump's "
              "call-site provenance label -- the ACTUAL velocity time level "
              "consumed by the formula (Kbb) is the subject of this task, "
              "not this registry string.")
register_dump("ldf_dump_dv.bin", "now", "same as ldf_dump_du")

for _name in ("ldf_dump_du.bin", "ldf_dump_dv.bin"):
    time_level_for_dump(_name)


def measure_dyn_ldf_corrected() -> dict:
    """Re-derive the corrected dyn_ldf corr/ratio/err_norm on legoESM's REAL
    production ``nemo_ldf_lap_viscosity_cgrid`` operator (via
    ``tendencies_with_diagnostics``, a public API -- not a probe
    transcription), fed with the BEFORE-level state
    (``state.u_before``/``v_before``/``T_before``/``S_before``), matching
    ``dynldf_lev_rot_scheme.h90:24-25,28-29``'s Kbb read.

    Self-contained: does NOT import anything from ``dyn_zad_ldf_walk.py``
    (untouchable) beyond the already-imported ``RUN_DIR``/``DT``/
    ``_load_full_3d`` helpers shared via ``zu_frc_term_walk.py`` (also
    untouchable, imported read-only, same as Task A above) -- this function
    independently re-builds the bridge/state/model and re-runs the
    production tendency call, so its numbers are a genuine independent
    reproduction, not a copy of the number handed down in the task prompt.
    """
    e3t_mode = require_explicit_e3t_mode(context="measure_dyn_ldf_corrected")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")
    assert e3t_mode == "both"

    dcfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    assert dcfg.lateral_viscosity_operator == "nemo_div_curl"

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    jpkm1 = jpk - 1

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    br_before = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dcfg, lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, br.state, context="measure_dyn_ldf_corrected")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    st_before_fed = br_before._replace(
        u=br_before.u_before, v=br_before.v_before,
        T=br_before.T_before, S=br_before.S_before)

    print("\n" + "=" * 78)
    print("TASK B SELF-CHECK: reproduce the reported dyn_ldf BEFORE-fed err_norm")
    print("(4.5769e-05 u / 4.5052e-05 v) independently, before trusting it.")
    print("=" * 78)
    with jax.disable_jit():
        _tend_bef, diag_bef = model.tendencies_with_diagnostics(
            st_before_fed, surface_forcing=None, dt=DT)
    # Also reproduce the OLD (now-fed) baseline as a control -- the DEBT
    # tuple currently in the gate.
    with jax.disable_jit():
        _tend_now, diag_now = model.tendencies_with_diagnostics(
            br.state, surface_forcing=None, dt=DT)

    ah_lap_u_bef = _u_to_nemo(np.asarray(diag_bef.Ah_lap_u.data))
    ah_lap_v_bef = _v_to_nemo(np.asarray(diag_bef.Ah_lap_v.data))
    ah_lap_u_now = _u_to_nemo(np.asarray(diag_now.Ah_lap_u.data))
    ah_lap_v_now = _v_to_nemo(np.asarray(diag_now.Ah_lap_v.data))

    nemo_ldf_du = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_du.bin"), jpi, jpj, jpkm1, hls)
    nemo_ldf_dv = _load_full_3d(os.path.join(RUN_DIR, "ldf_dump_dv.bin"), jpi, jpj, jpkm1, hls)
    umask2 = np.asarray(g.umask)[..., 0] > 0.5
    vmask2 = np.asarray(g.vmask)[..., 0] > 0.5

    _, _, _, tot_u_now, _, _ = _err_norm(ah_lap_u_now, nemo_ldf_du, umask2)
    _, _, _, tot_v_now, _, _ = _err_norm(ah_lap_v_now, nemo_ldf_dv, vmask2)
    _, _, _, tot_u_bef, _, _ = _err_norm(ah_lap_u_bef, nemo_ldf_du, umask2)
    _, _, _, tot_v_bef, _, _ = _err_norm(ah_lap_v_bef, nemo_ldf_dv, vmask2)
    print(f"  OLD (now-fed, current gate DEBT regime)   : u err_norm={tot_u_now:.4e}  v err_norm={tot_v_now:.4e}")
    print(f"  NEW (before-fed, corrected)                : u err_norm={tot_u_bef:.4e}  v err_norm={tot_v_bef:.4e}")

    def _corr_ratio(lego3, nemo3, mask2d):
        n_lat = min(lego3.shape[0], nemo3.shape[0], mask2d.shape[0])
        n_lon = min(lego3.shape[1], nemo3.shape[1], mask2d.shape[1])
        n_lev = min(lego3.shape[2], nemo3.shape[2])
        m3 = np.broadcast_to(mask2d[:n_lat, :n_lon, None], (n_lat, n_lon, n_lev))
        lo = np.asarray(lego3)[:n_lat, :n_lon, :n_lev][m3]
        ne = np.asarray(nemo3)[:n_lat, :n_lon, :n_lev][m3]
        corr = float(np.corrcoef(lo, ne)[0, 1])
        ratio = float(np.abs(lo).sum() / np.abs(ne).sum())
        return corr, ratio

    corr_u, ratio_u = _corr_ratio(ah_lap_u_bef, nemo_ldf_du, umask2)
    corr_v, ratio_v = _corr_ratio(ah_lap_v_bef, nemo_ldf_dv, vmask2)
    print(f"\n  CORRECTED corr/ratio (before-fed, cancelling_rows_per_element.py "
          f"convention): u corr={corr_u:.9f} ratio={ratio_u:.9f}  "
          f"v corr={corr_v:.9f} ratio={ratio_v:.9f}")

    reproduced = (abs(tot_u_bef - 4.5769e-05) / 4.5769e-05 < 1e-2
                  and abs(tot_v_bef - 4.5052e-05) / 4.5052e-05 < 1e-2)
    print(f"  reproduced reported before-fed err_norm to <1% relative: {reproduced}")
    if not reproduced:
        print("  STOP: independent re-derivation does not match the reported "
              "corrected number -- do not update the gate.")

    print("\n" + "=" * 78)
    print("TASK B DONE.")
    print("=" * 78)
    return dict(
        reproduced=reproduced,
        err_norm_u_now=tot_u_now, err_norm_v_now=tot_v_now,
        err_norm_u_before=tot_u_bef, err_norm_v_before=tot_v_bef,
        corr_u=corr_u, ratio_u=ratio_u, corr_v=corr_v, ratio_v=ratio_v,
    )


if __name__ == "__main__":
    rc = main()
    result_b = measure_dyn_ldf_corrected()
    raise SystemExit(rc if rc != 0 or result_b["reproduced"] else 0)
