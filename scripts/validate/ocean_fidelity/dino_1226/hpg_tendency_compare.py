#!/usr/bin/env python
"""#1226 term 9 (``dyn_hpg``): legoESM PGF vs NEMO's ``hpg_sco`` tendency dump.

The gate (``fidelity_bar_gate.py``) carried ``dyn_hpg`` at corr 1.000000 /
ratio 1.000045 with NO committed measurement script, so the ratio had no
provenance.  This is that script.

WHAT NEMO DUMPED (``cfgs/DINO/MY_SRC/dynhpg.F90``, the ``#1226 item 9`` block)::

    zu_before(:,:,:) = puu(:,:,:,Krhs)          ! snapshot at routine ENTRY
    ... hpg_sco body ...
    DO jk = 1, jpkm1
       WRITE(8835) ( ( puu(ji,jj,jk,Krhs) - zu_before(ji,jj,jk), ji=1,jpi ), jj=1,jpj )

so each file is the **pure ``hpg_sco`` increment** (after-minus-before, i.e. the
``zhpi + zuap`` trend and nothing else), float64, units m/s^2, written only at
``kt == nit000``, with index order ``jk`` outer / ``jj`` / ``ji`` fastest.
Shape is the RUNTIME array ``(jpkm1, jpj, jpi) = (35, 203, 56)``, which carries
NEMO's ``nn_hls=2`` halo -- the mesh_mask/domain_cfg FILES are haloless
(199 x 52), so the dump is stripped by 2 on every side to reach the compute
domain.  ``_DUMP_HALO`` below; the alignment scan re-derives it rather than
trusting it.

MEASURED (2026-07-28), NEMO's real 3-D geometry (``--e3t-mode both``), ratio =
rms(lego)/rms(nemo) over wet faces, alignment scan peaked at halo strip 2/2::

    state                                du ratio     dv ratio    corr
    Y5 developed (RUN_GDB, kt=57601)     1.000055     1.000053    0.99999999
    from-rest istate (RUN_1226_AHTU)     degenerate   1.000024    1.00000000

=> **DEBT, not at bar.**  The Y5 numbers reproduce the previously-recorded
1.000045, so that value is REAL -- not stale, and not a KE-contamination
artifact.  Measure this term on the Y5 state: at ssh=0 the qco ``(1+r3t)``
stretch and the ``zuap`` slope term are IDENTICALLY ZERO, so a from-rest run
cannot certify them (and its ``du`` is zero on both sides, i.e. degenerate).

THE GEOMETRY MODE IS DECISIVE FOR THIS TERM.  On the same Y5 state the bridge's
DEFAULT ``LEGOESM_NEMO_E3T=off`` (the known-wrong 1-D ``e3t_1d`` ladder) gives
ratio 1.007285 -- 137x the residual of ``both`` (1.000053).  Any dyn_hpg number
measured at the bridge default is dominated by that harness bias, so this
script defaults to ``both``.

WHAT IS AND IS NOT THE CAUSE (Y5, verified):
  * NOT a missing stretch/zuap: zeroing ssh while keeping developed T,S makes
    the residual 20x WORSE (1.0011), so those terms are live and load-bearing.
  * NOT the ``dy_v`` vs NEMO ``e2v`` metric: substituting NEMO's own e2v makes
    the Y5 residual WORSE (1.000053 -> 1.000066).  An earlier attribution to
    this metric was measured on the degenerate from-rest state and is RETRACTED.
  * OPEN, with a robust signature: the residual is monotonically BOTTOM-HEAVY,
    per-level ratio climbing smoothly 1.000024 (k=27) -> 1.000242 (k=34,
    deepest) with the upper ocean near-exact.  That is an ACCUMULATION
    signature -- the PGF is a cumulative depth integral, so a small per-level
    increment bias compounds downward.  Leading (UNPROVEN) hypothesis: legoESM
    feeds both the EOS depth argument and the trapezoid w-spacings the STATIC
    ``z_coord.t_depth_ref`` (ocean_pe_latlon_cgrid.py:1275-1286), whereas NEMO
    under ``key_qco`` uses the LIVE ``gdept = gdept_0*(1+r3t)``.  Pointwise
    correlation of the relative error with ``r3t`` is only -0.19 (vs +0.32 with
    depth), so this is NOT confirmed and the 1-D ladder API cannot currently
    express a per-column live depth.

WHY THE from-rest STATE MATCHES EXACTLY.  ``RUN_1226_AHTU`` has ``ln_rstart = .false.``
("Start from rest (ln_rstart=F)" in ocean.output), so the kt=1 state is the
ANALYTIC initial condition ``usr_def_istate`` CASE(4) (``nn_initcase = 4``):
a depth-only T/S profile blended linearly toward its own abyssal value from
equator to pole, with ``u = v = ssh = 0``.  That is reproduced here directly
from NEMO's own mesh (``gdept_1d``, ``gphit``, ``tmask``), so T and S are
identical by construction and the comparison isolates the PGF operator.

Two consequences of ``u = v = 0`` and ``ssh = 0`` that make this measurement
clean, and one that bounds it:

* ``u = v = 0`` -> the kinetic-energy gradient is IDENTICALLY zero, so the
  model's fused ``KE_PGF_u`` diagnostic *is* the pure pressure-gradient term.
  No KE contamination (the KE gradient is the separately-measured
  ``dyn_adv KEG``), and no need for a bespoke PGF diagnostic.
* ``ssh = 0`` -> the ``nemo_sco`` qco stretch ``(1+r3t)`` is 1 and the
  ``gdept_z0`` slope difference is a 1-D-ladder difference, i.e. zero.  So this
  measurement exercises the trapezoid ``p'`` quadrature, the EOS, ``g`` and the
  along-level gradient, but NOT the ``zuap``/stretch terms.  Those are covered
  analytically instead by ``tests/ocean/unit/test_nemo_sco_step_pgf.py``
  (rtol 1e-12 vs a literal dynhpg.F90 numpy recurrence).  Stated so the scope
  of the number this prints is not overread.

Ratio convention matches the rest of the #1226 table
(``tracer_tendency_compare.py``): ``ratio = rms(lego) / rms(nemo)`` over wet
points, ``corr = np.corrcoef``.

Run (the Y5 developed state -- the one that certifies this term)::

    G=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB
    JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu python \\
        scripts/validate/ocean_fidelity/dino_1226/hpg_tendency_compare.py \\
        --run $G --state $G/DINO_00057600_restart.nc
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np

RUN_DEFAULT = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
               "RUN_1226_AHTU")

# NEMO 5 runtime halo on the DINO dump arrays (nn_hls=2).  The alignment scan
# verifies this rather than assuming it.
_DUMP_HALO = 2


# ----------------------------------------------------------------------
# NEMO usr_def_istate CASE(4) -- transcription of
# cfgs/DINO/MY_SRC/usrdef_istate.F90:129-175
# ----------------------------------------------------------------------
def _istate_profiles_1d(gdept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The depth-only T and S profiles (usrdef_istate.F90:134-149).

    ``gdept`` is positive-down depth [m].  Returns ``(T_1d, S_1d)`` in
    (degC, PSU) -- the unmasked, horizontally-uniform profiles before the
    meridional blend.
    """
    z = gdept
    t = ((16.0 - 12.0 * np.tanh((z - 400.0) / 700.0))
         * (-np.tanh((500.0 - z) / 150.0) + 1.0) / 2.0
         + (15.0 * (1.0 - np.tanh((z - 50.0) / 1500.0))
            - 1.4 * np.tanh((z - 100.0) / 100.0)
            + 7.0 * (1500.0 - z) / 1500.0)
         * (-np.tanh((z - 500.0) / 150.0) + 1.0) / 2.0)
    s = ((36.25 - 1.13 * np.tanh((z - 305.0) / 460.0))
         * (-np.tanh((500.0 - z) / 150.0) + 1.0) / 2.0
         + (35.55 + 1.25 * (5000.0 - z) / 5000.0
            - 1.62 * np.tanh((z - 60.0) / 650.0)
            + 0.2 * np.tanh((z - 35.0) / 100.0)
            + 0.2 * np.tanh((z - 1000.0) / 5000.0))
         * (-np.tanh((z - 500.0) / 150.0) + 1.0) / 2.0)
    return t, s


def nemo_istate_case4(gdept: np.ndarray, gphit: np.ndarray,
                      tmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NEMO ``usr_def_istate`` CASE(4) T and S on a full-step (ln_zco) mesh.

    ``gdept`` is the depth NEMO ACTUALLY passes as ``pdept`` -- the 3-D
    ``gdept_0``, broadcastable to ``tmask``.  Using the 1-D ``gdept_1d`` ladder
    here is WRONG for DINO: the two T-depth ladders first part at k=26
    (``gdept_1d`` = 1225 m -- not the "~2000 m" this note used to say) and
    diverge by up to 105.0 m at k=32.  ``e3t_1d`` is the unstretched analytic
    ladder and ``e3t_0`` is stretched, but they sum to the SAME 4000.000 m over
    all 35 wet levels, so for the 75% of columns that reach the bottom level the
    difference is a redistribution of thickness rather than a change of depth.
    For the other 25% it IS a change of depth: those columns stop short, and the
    1-D ladder puts their bottom 70.4-104.2 m too deep (re-measured 2026-08-21,
    #1455).  It feeds
    straight into T, S and therefore into the PGF being measured.

    The blend anchors follow NEMO exactly (usrdef_istate.F90:153-176):

    * ``zphiMAX = MAXVAL(gphit)``
    * ``zTbot = MINVAL(T + 100*(1-tmask))`` -- the minimum over WET cells
    * ``T <- (T - zTbot) * (zphiMAX - |gphit|) / zphiMAX + zTbot``, masked

    Returns ``(T, S)`` shaped like ``tmask`` (n_lat, n_lon, nlev).
    """
    t_prof, s_prof = _istate_profiles_1d(np.asarray(gdept, dtype=np.float64))
    t = np.broadcast_to(t_prof, tmask.shape) * tmask
    s = np.broadcast_to(s_prof, tmask.shape) * tmask
    phi_max = float(np.max(gphit))
    if not phi_max > 0.0:
        # NEMO's zphiMAX is MAXVAL(gphit) (69.85 deg on DINO R1); a
        # non-positive max would divide the blend by zero and hand back a
        # silently all-NaN initial state.
        raise ValueError(
            f"MAXVAL(gphit) must be > 0 for the CASE(4) meridional blend; "
            f"got {phi_max!r}")
    # MINVAL over wet cells: the +100*(1-tmask) trick keeps dry cells out.
    t_bot = float(np.min(t + 100.0 * (1.0 - tmask)))
    s_bot = float(np.min(s + 100.0 * (1.0 - tmask)))
    blend = ((phi_max - np.abs(gphit)) / phi_max)[..., None]
    t_out = ((t - t_bot) * blend + t_bot) * tmask
    s_out = ((s - s_bot) * blend + s_bot) * tmask
    return t_out, s_out


# ----------------------------------------------------------------------
# comparison helpers
# ----------------------------------------------------------------------
def read_dump(path: str, jpk: int, jpj: int, jpi: int) -> np.ndarray:
    """Read a ``hpg_dump_d[uv].bin`` into ``(n_lat, n_lon, nlev)`` order.

    The Fortran write is ``jk`` outer / ``jj`` / ``ji`` fastest, so the raw
    buffer is ``(jpk, jpj, jpi)``; transpose to legoESM's trailing-vertical
    convention.
    """
    a = np.fromfile(path, dtype=np.float64)
    want = jpk * jpj * jpi
    if a.size != want:
        raise ValueError(
            f"{path}: {a.size} float64 values, expected {want} "
            f"(jpk={jpk}, jpj={jpj}, jpi={jpi})")
    return a.reshape(jpk, jpj, jpi).transpose(1, 2, 0)


def corr_ratio(lego: np.ndarray, nemo: np.ndarray,
               wet: np.ndarray) -> tuple[float, float, int]:
    """``(corr, ratio, n)`` over ``wet`` points; ratio = rms(lego)/rms(nemo).

    Same convention as ``tracer_tendency_compare.py`` so the number is
    comparable with the rest of the #1226 table.
    """
    lm = lego[wet]
    nm = nemo[wet]
    if lm.size < 2:
        return float("nan"), float("nan"), int(lm.size)
    corr = float(np.corrcoef(lm, nm)[0, 1])
    rms_n = float(np.sqrt(np.mean(nm ** 2)))
    rms_l = float(np.sqrt(np.mean(lm ** 2)))
    ratio = rms_l / rms_n if rms_n > 0 else float("nan")
    return corr, ratio, int(lm.size)


def alignment_scan(lego: np.ndarray, nemo_full: np.ndarray, wet: np.ndarray,
                   halo: int, span: int = 1):
    """Sweep +/-``span`` index shifts around the nominal halo strip.

    Returns a list of ``(dj, di, corr, ratio, n)`` sorted by descending corr.
    A correct alignment shows a SHARP corr peak; a flat scan means the
    comparison is not localising and the ratio must not be trusted.
    """
    n_lat, n_lon, nlev = lego.shape
    out = []
    for dj in range(-span, span + 1):
        for di in range(-span, span + 1):
            j0, i0 = halo + dj, halo + di
            if j0 < 0 or i0 < 0:
                continue
            if (j0 + n_lat > nemo_full.shape[0]
                    or i0 + n_lon > nemo_full.shape[1]):
                continue
            sub = nemo_full[j0:j0 + n_lat, i0:i0 + n_lon, :nlev]
            c, r, n = corr_ratio(lego, sub, wet)
            out.append((dj, di, c, r, n))
    out.sort(key=lambda t: (-(t[2] if np.isfinite(t[2]) else -9), abs(t[0])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=RUN_DEFAULT,
                    help="NEMO run dir holding mesh_mask.nc + hpg_dump_d[uv].bin")
    ap.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    ap.add_argument(
        "--state", default="istate",
        help="'istate' = the analytic usr_def_istate CASE(4) initial condition "
             "(for a ln_rstart=.false. run such as RUN_1226_AHTU), or a path to "
             "a NEMO restart .nc for a DEVELOPED state (e.g. "
             "RUN_GDB/DINO_00057600_restart.nc, the Y5 twin). ONLY the "
             "developed state exercises the qco (1+r3t) stretch and the zuap "
             "slope term -- both are IDENTICALLY ZERO at ssh=0, so a from-rest "
             "measurement CANNOT certify them.")
    ap.add_argument(
        "--e3t-mode", default="both",
        choices=("off", "e3t_only", "gdept_only", "both"),
        help="Which of NEMO's ACTUAL 3-D scale factors legoESM's grid is built "
             "from (nemo_state_bridge.effective_vertical_scale_factors). The "
             "bridge DEFAULT is 'off' = the known-wrong 1-D e3t_1d ladder. It "
             "was kept because NEMO's true geometry was recorded as "
             "destabilising long integrations; that did not reproduce in 2026-08 "
             "and is now an unexplained observation, not an established defect "
             "(#1455). "
             "A single-step tendency never integrates, so 'both' (NEMO's real "
             "geometry) is the apples-to-apples choice here.")
    args = ap.parse_args()

    # Set BEFORE the bridge import path reads it.
    os.environ["LEGOESM_NEMO_E3T"] = args.e3t_mode
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax.numpy as jnp
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_model_config,
    )
    from legoesm.ocean.fidelity.nemo_io import (
        NemoState, read_nemo_mesh_mask, read_nemo_restart,
    )
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )

    # --- NEMO side -----------------------------------------------------
    g = read_nemo_mesh_mask(f"{args.run}/mesh_mask.nc", nn_hls=0)
    n_lat, n_lon = g.gphit.shape
    nlev = g.tmask.shape[-1]
    print(f"mesh (haloless): n_lat={n_lat} n_lon={n_lon} nlev={nlev}")

    if args.state == "istate":
        # NEMO passes the 3-D gdept_0 as usr_def_istate's pdept; gdept_1d is a
        # DIFFERENT (unstretched) ladder and using it here biases T/S by up to
        # 105 m of depth in the abyss.
        gdept = g.gdept_0 if g.gdept_0 is not None else g.gdept_1d
        print(f"state: analytic istate CASE(4), pdept from "
              f"{'gdept_0 (3-D, NEMO actual)' if g.gdept_0 is not None else 'gdept_1d (FALLBACK)'}")
        T, S = nemo_istate_case4(gdept, g.gphit, g.tmask)
        zeros3 = np.zeros_like(T)
        st_nemo = NemoState(T=T, S=S, u=zeros3, v=zeros3,
                            ssh=np.zeros((n_lat, n_lon)), rhd=None)
    else:
        st_nemo = read_nemo_restart(args.state, nn_hls=0)
        T, S = st_nemo.T, st_nemo.S
        print(f"state: NEMO restart {args.state}")
    wetpt = g.tmask > 0
    print(f"  T [{T[wetpt].min():.4f}, {T[wetpt].max():.4f}] degC   "
          f"S [{S[wetpt].min():.4f}, {S[wetpt].max():.4f}] PSU")
    ssh_rms = float(np.sqrt(np.mean(np.asarray(st_nemo.ssh) ** 2)))
    print(f"  ssh rms = {ssh_rms:.6f} m  "
          f"(ssh==0 => qco stretch and zuap are IDENTICALLY ZERO: "
          f"{'DEGENERATE for those terms' if ssh_rms == 0.0 else 'LIVE'})")

    br = bridge_nemo_to_legoesm_topo(g, st_nemo, periodic_i=True,
                                     full_step=True)
    print(f"bridge f self-check max|df| = {br.f_match_max_abs:.3e}")

    # --- legoESM side: the PGF from the recipe's real tendency path -----
    cfg = dataclasses.replace(dino_config_for_recipe(args.recipe),
                              lon_west_deg=1.0, lon_east_deg=49.0,
                              sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    print(f"pgf_scheme={mc.pgf_scheme} pgf_quadrature={mc.pgf_quadrature} "
          f"eos={mc.eos} eos_depth={mc.eos_depth} g={mc.g!r}")

    # ISOLATE THE PURE PGF.  The model's diagnostic is fused:
    #   KE_PGF_u = -dKE_dx - dp_dx/rho_0
    # while NEMO's dump is the PGF alone.  The pressure gradient is a function
    # of (T, S, eta, geometry) ONLY -- it does not read u or v anywhere in
    # _bc_ke_and_pressure_gradients -- whereas the KE gradient is a function of
    # (u, v) ONLY.  So zeroing the velocities removes the KE term EXACTLY while
    # leaving the PGF bit-identical: this is an exact isolation, not an
    # approximation, and it does NOT make the state "from rest" (T, S and
    # crucially eta stay at their developed values, so the qco stretch and the
    # zuap slope term remain fully live).
    st = br.state
    u_dev = float(np.max(np.abs(np.asarray(st.u.data))))
    v_dev = float(np.max(np.abs(np.asarray(st.v.data))))
    st = st._replace(
        u=st.u.replace(data=jnp.zeros_like(st.u.data)),
        v=st.v.replace(data=jnp.zeros_like(st.v.data)),
    )
    print(f"KE isolation: zeroed u,v (was max|u|={u_dev:.4f} "
          f"max|v|={v_dev:.4f} m/s); PGF is u,v-independent so this removes "
          f"the KE gradient exactly and leaves eta/stretch/zuap untouched")

    _, diag = latlon_cgrid_ocean_baroclinic_tendencies(
        st, br.geometry, br.z_coord, mc, dt=cfg.dt, diagnose_momentum=True)
    pgf_u = np.asarray(diag.KE_PGF_u.data)      # (n_lat, n_lon+1, nlev)
    pgf_v = np.asarray(diag.KE_PGF_v.data)      # (n_lat+1, n_lon, nlev)

    # legoESM stores u on n_lon+1 faces with index 0 = west wall, so NEMO's
    # east-face u(ji) is lego face ji+1 (nemo_state_bridge._u_east_to_face);
    # likewise v index 0 = south wall.  The alignment scan re-derives this.
    lego_u = pgf_u[:, 1:, :]
    lego_v = pgf_v[1:, :, :]
    geom_dy_v = br.geometry.dy_v

    # --- dump ----------------------------------------------------------
    jpj = n_lat + 2 * _DUMP_HALO
    jpi = n_lon + 2 * _DUMP_HALO
    jpkm1 = nlev - 1
    du = read_dump(f"{args.run}/hpg_dump_du.bin", jpkm1, jpj, jpi)
    dv = read_dump(f"{args.run}/hpg_dump_dv.bin", jpkm1, jpj, jpi)
    print(f"dump: (jpkm1, jpj, jpi) = ({jpkm1}, {jpj}, {jpi})  "
          f"-> compare over nlev={jpkm1}")

    lego_u = lego_u[:, :, :jpkm1]
    lego_v = lego_v[:, :, :jpkm1]

    # Wet faces only.  Drop the redundant periodic-wrap u-face (bridge KNOWN
    # LIMITATION (a): lego columns 0 / n_lon are the same physical face and
    # use the closed-basin storage convention).
    umask = g.umask[..., :jpkm1] > 0.5
    vmask = g.vmask[..., :jpkm1] > 0.5
    umask = umask.copy()
    umask[:, -1, :] = False

    for tag, lego, dump, wet in (("du", lego_u, du, umask),
                                 ("dv", lego_v, dv, vmask)):
        print(f"\n=== {tag} ===")
        scan = alignment_scan(lego, dump, wet, _DUMP_HALO)
        print(f"{'dj':>3} {'di':>3} {'corr':>12} {'ratio':>12} {'n':>9}")
        for dj, di, c, r, n in scan[:5]:
            print(f"{dj:>3} {di:>3} {c:>12.6f} {r:>12.6f} {n:>9d}")
        dj, di, c, r, n = scan[0]
        second = scan[1][2] if len(scan) > 1 else float("nan")
        print(f"BEST {tag}: dj={dj} di={di} (halo strip "
              f"{_DUMP_HALO + dj}/{_DUMP_HALO + di})  "
              f"corr={c:.9f} ratio={r:.9f} n={n}")
        print(f"  peak sharpness: best {c:.6f} vs runner-up {second:.6f}")
        if np.isfinite(c) and np.isfinite(second) and c - second < 1e-3:
            print("  WARNING: flat alignment scan -- ratio NOT trustworthy")
        if not np.isfinite(c):
            print("  (degenerate: both sides identically zero on wet faces)")
            continue
        # Localisation: a PGF residual is usually bottom-heavy (deepest-level
        # BC) or surface-heavy (the k=1 half-cell term), so break the ratio
        # down by level and by latitude band.
        j0, i0 = _DUMP_HALO + dj, _DUMP_HALO + di
        sub = dump[j0:j0 + lego.shape[0], i0:i0 + lego.shape[1], :lego.shape[2]]
        print("  per-level (worst 8 by |ratio-1|):")
        rows = []
        for k in range(lego.shape[2]):
            wk = wet[:, :, k]
            if wk.sum() < 2:
                continue
            ck, rk, nk = corr_ratio(lego[:, :, k], sub[:, :, k], wk)
            rows.append((k, ck, rk, nk))
        for k, ck, rk, nk in sorted(
                rows, key=lambda t: -abs(t[2] - 1.0))[:8]:
            print(f"    k={k:>2} corr={ck:.9f} ratio={rk:.9f} n={nk}")
        print("  per-latitude-band:")
        nb = 4
        edges = np.linspace(0, lego.shape[0], nb + 1).astype(int)
        for b in range(nb):
            sl = slice(edges[b], edges[b + 1])
            cb, rb, nb_ = corr_ratio(lego[sl], sub[sl], wet[sl])
            print(f"    rows {edges[b]:>3}-{edges[b+1]:>3} "
                  f"corr={cb:.9f} ratio={rb:.9f} n={nb_}")
        # ATTRIBUTION.  The v-PGF carries the metric as -delta_j[p']/(rho0*dy).
        # The bridge RECONSTRUCTS dy from gphit/gphiv (create_latlon_geometry)
        # instead of adopting NEMO's e2v, and the two differ by tens of ppm
        # with latitude structure.  Rescaling by dy_lego/e2v_nemo therefore
        # substitutes NEMO's own metric while leaving the PGF operator's
        # output untouched: whatever ratio remains after this is the TERM,
        # and whatever it removes was the HARNESS.
        if tag == "dv" and g.e2v is not None:
            scale = (np.asarray(geom_dy_v)[1:] / np.asarray(g.e2v))[..., None]
            cm, rm, _ = corr_ratio(lego * scale[:, :, :lego.shape[2]],
                                   sub, wet)
            print(f"  metric-substituted (NEMO e2v): "
                  f"corr={cm:.9f} ratio={rm:.9f}")
            print(f"    -> harness metric accounts for "
                  f"{abs(r - 1.0) - abs(rm - 1.0):.3e} of |ratio-1|")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
