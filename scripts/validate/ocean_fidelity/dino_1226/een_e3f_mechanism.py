#!/usr/bin/env python
"""#1455 -- is the EEN F-point thickness (e3f) the owner of the wall-row
vorticity-flux mismatch?

Pre-registration: ``PREREG_een_e3f_mechanism.md``, written before any number
here existed.  Parent: ``docs/ocean/fidelity/dino_wall_ldf_alignment.md`` and
``wall_term_discriminators.py``, which measured the EEN vorticity flux at
3.10e-10 m/s2 on the four wall rows (5.25x the magnitude bar, 8.74x enriched,
2.1e-3 pointwise) and left it as the sole fix candidate.

WHAT NEMO ACTUALLY RUNS (build state verified in ``_stamp_build``, not assumed):
DINO compiles ``key_qco key_vco_3d``, so ``vor_een``'s runtime
``SELECT CASE(nn_e3f_typ)`` (dynvor.F90:722-745) is inside the ``#else`` arm and
is DEAD.  What executes is ``z1_e3f = 1/e3f_vor`` (dynvor.F90:719) with

    e3f_vor(i,j,k) = e3f_0vor(i,j,k) * ( 1 + r3f(i,j)*fe3mask(i,j,k) )

(domzgr_substitute.h90:130 + :118 + :48).  ``e3f_0vor`` is the wet-count-masked
average of the STATIC ``e3t_0`` (dynvor.F90:927-939, nn_e3f_typ=1) with zeros
overwritten by ``e3f_0`` (:950); ``r3f`` is the area-weighted UNMASKED four-point
sea-surface ratio (domqco.F90:177-181); ``fe3mask`` is ``fmask`` (dommsk.F90:198)
= the product of the four surrounding ``tmask`` (dommsk.F90:152-153), hence ZERO
at any vertex with a dry neighbour.  So NEMO applies NO free-surface stretching
to the F-point thickness at a wall vertex.  legoESM's ``een_e3f_h_vtx``
("nemo_avg") averages the LIVE thickness over the wet count, so it does.

WHAT IS PRE-REGISTERED AND WHAT IS NOT (adversarial review finding 5).
``PREREG_een_e3f_mechanism.md`` registers control A and legs 1, 1b and 2-5 --
i.e. the REFUTATION of the e3f mechanism is pre-registered. Controls B and C
and legs 6, 7 and 8 -- i.e. the identification of the OWNER -- are POST-HOC.
They follow a branch the pre-registration explicitly left open ("if the
combined residual does not go to the floor, the remainder is the triad
assembly itself, and that is the finding"), so they are a registered follow-up
rather than a fishing trip, but they are labelled EXPLORATORY wherever they
appear and no bar was set for them in advance.

CONTROLS, all of which must pass before any scored number is quoted:
  A  the transcribed ``dom_qco_r3c`` arithmetic reproduces the DUMPED r3u/r3v
     from the DUMPED after-ssh (the f-point branch is the same routine three
     lines below the u/v branch, and r3f is not dumped).
  B  the hand-assembled ``pv_flux_al81_partial_cell`` call reproduces the
     model's own ``vortcor_u`` diagnostic BIT-IDENTICALLY -- this is what
     licenses swapping one argument and attributing the change.
  C  the F-point index embedding is proven by rebuilding ``fmask`` as the
     4-tmask product in legoESM vertex indexing and matching the mesh's own
     ``fmask``; and the south vertex row's ``fe3mask`` is exactly 0, which is
     what makes the south fill of the embedded fields provably inert.

This probe prints numbers.  Every verdict string is COMPUTED from those numbers
against the bars registered in the pre-registration; none is hardcoded.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.een_e3f_mechanism
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Everything shared with the sibling probe is IMPORTED, never re-derived: the
# bridge build, the coherent (signed, thickness-weighted, zonal) reduction the
# 3.10e-10 was measured on, the row bars, the dry-face gate, the dump reader
# and the u-face convention mapping.
from acc_momentum_budget import _load_full_3d, _u_to_nemo  # noqa: E402
from wall_term_discriminators import (  # noqa: E402
    BAR_ENRICH, DINO, INTERIOR_ROWS_FAR, JPI, JPJ, JPK, JPKM1, HLS,
    RESTART, RUN, WALL_ROWS, _gate_dry_faces, build_lego, coherent_rows,
)

# The mesh the DUMPS were written from, not the trajectory run's copy.
# (Review: the two are md5-identical here, so this is provenance hygiene
# rather than a numeric change -- but the probe should read the mesh that
# belongs to the run it is scoring.)
MESH = RUN / "mesh_mask.nc"

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

# --- Bars, all from PREREG_een_e3f_mechanism.md, all pre-registered ---
MEASURED_WALL = 3.10e-10      # m/s2, the parent's coherent wall-row difference
LEG_A_FACTOR = 2.0            # "within a factor of 2" of MEASURED_WALL
BAR_R3_TRANSCRIPTION = 1e-12  # control A, relative
BAR_OWNER = 0.4               # leg 1b: residual ratio below this => OWNER
BAR_CONTRIB = 0.9             # ... above this => REFUTED as owner

# The u-face thickness in NEMO layout, for the MASS-weighted reduction.
# A one-element list so ``score`` can read it without a global statement;
# ``main`` fills it before the first score and the probe raises if it did
# not (a silent None would fall back to the level mean -- the very defect
# being corrected).
_H_U = [None]


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  run={RUN}  (1 step, kt 5761, single rank)")
    print(f"PROVENANCE  mesh={MESH}   restart={RESTART.name}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"PROVENANCE  bars: leg-a |x/{MEASURED_WALL:.2e}| in "
          f"[1/{LEG_A_FACTOR:g}, {LEG_A_FACTOR:g}], leg-b enrichment "
          f">={BAR_ENRICH}, leg-1b residual ratio <{BAR_OWNER} owner / "
          f">{BAR_CONTRIB} refuted")


def _stamp_build() -> None:
    """Quote the preprocessor + namelist state the mechanism claim rests on.

    The previous lane asserted the runtime ``nn_e3f_typ`` branch is dead in this
    build.  That is a claim about the COMPILED source, so it is re-read from the
    files here on every run rather than repeated from a document.
    """
    fcm = (DINO / "cpp_DINO.fcm").read_text().strip()
    nml = (RUN / "namelist_cfg").read_text().splitlines()

    def _grab(key):
        for ln in nml:
            s = ln.split("!")[0]
            if key in s and "=" in s:
                return s.strip()
        return "<absent>"

    print(f"BUILD  {fcm}")
    # ADVERSARIAL REVIEW FINDING 7, acted on: reading is not checking. Every
    # number in this probe is wrong if any of these changes, so each is GATED,
    # not merely echoed. ``_grab`` returning "<absent>" now fails the gate
    # instead of passing silently.
    required = {"ln_dynvor_een": ".true.", "ln_dynvor_msk": ".false.",
                "nn_e3f_typ": "1", "rn_shlat": "0.",
                "ln_dynspg_exp": ".false.", "ln_dynspg_ts": ".true."}
    bad = []
    for k, want in required.items():
        line = _grab(k)
        got = line.split("=", 1)[1].strip() if "=" in line else "<absent>"
        ok = got == want
        print(f"BUILD  namelist_cfg: {line}   [require {want}: "
              f"{'ok' if ok else 'MISMATCH'}]")
        if not ok:
            bad.append(f"{k}={got!r} (require {want!r})")
    for key in ("key_qco", "key_vco_3d"):
        if key not in fcm:
            bad.append(f"cpp key {key} absent from cpp_DINO.fcm")
    if bad:
        raise SystemExit(
            "the oracle build/namelist state this probe's arithmetic assumes "
            "is not the one on disk: " + "; ".join(bad))
    src = (Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/DYN/"
                "dynvor.F90").read_text().splitlines())
    # The guard whose arm decides whether the runtime nn_e3f_typ branch runs.
    # Located by SEARCH rather than a hardcoded line window (review finding 7),
    # and the one-line key_qco arm must be the one immediately inside it.
    hit = [i for i, ln in enumerate(src, start=1)
           if "z1_e3f(ji,jj) = 1._wp / e3f_vor" in ln]
    if not hit:
        raise SystemExit(
            "dynvor.F90 no longer contains the key_qco one-line e3f_vor arm "
            "this probe's mechanism claim is built on")
    for i in hit:
        print(f"BUILD  dynvor.F90:{i}: {src[i - 1].rstrip()}")
        print(f"BUILD  dynvor.F90:{i - 2}: {src[i - 3].rstrip()}"
              "   <- the guard whose #else arm holds the DEAD nn_e3f_typ branch")


# ---------------------------------------------------------------------------
# F-point index embedding.  NEMO's F(ji,jj) is the NE corner of T-cell (ji,jj);
# legoESM's vertex[j,i] is the SW corner of cell (j,i).  So NEMO F(jj,ji) maps
# to legoESM vertex[jj+1, ji+1].  Column 0 of the legoESM vertex grid is the
# periodic image of NEMO column n_lon-1; row 0 is the southern boundary vertex
# row, which lies south of the (entirely dry) cell row 0 and therefore has
# fe3mask == 0 -- control C asserts that, which is what makes ``south`` inert.
# ---------------------------------------------------------------------------
def _f_nemo_to_vtx(a: np.ndarray, south: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    out = np.full((a.shape[0] + 1, a.shape[1] + 1) + a.shape[2:], south,
                  dtype=np.float64)
    out[1:, 1:] = a
    out[1:, 0] = a[:, -1]
    return out


def _finite(name: str, a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise SystemExit(
            f"{name} carries {int((~np.isfinite(a)).sum())} non-finite values "
            "-- NaN is fatal in this probe, no nan-aware reduction is used")
    return a


def read_mesh_extras():
    """The mesh fields NemoGrid does not carry (e1f/e2f/e3f_0/fmask/e1v/e2u).

    Read straight from the file rather than re-derived.  ``nn_hls=0`` for this
    mesh, so the only transform is the (z,y,x) -> (y,x,z) axis move the rest of
    the fidelity harness uses; asserted against the known domain shape.
    """
    m = xr.open_dataset(str(MESH), decode_times=False)
    out = {}
    for k in ("e1f", "e2f", "e1t", "e2t", "e1u", "e2v", "e1v", "e2u",
              "ff_f"):
        out[k] = _finite(k, np.asarray(m[k].values).squeeze())
    for k in ("e3f_0", "fmask", "e3t_0", "e3u_0", "e3v_0", "tmask", "umask",
              "vmask"):
        a = _finite(k, np.asarray(m[k].values).squeeze())
        out[k] = np.moveaxis(a, 0, -1)
    ny, nx = out["e1t"].shape
    assert out["tmask"].shape == (ny, nx, JPK), out["tmask"].shape
    return out


def transcribe_r3(ssh, mx, want=("t", "u", "v", "f")):
    """``dom_qco_r3c`` (domqco.F90:155-183) transcribed, all four points.

    r3t = ssh * r1_ht_0                                            (:160)
    r3u = 0.5 *( e1e2t_i*ssh_i + e1e2t_i+1*ssh_i+1 ) *r1_hu_0*r1_e1e2u  (:166)
    r3v = 0.5 *( e1e2t_j*ssh_j + e1e2t_j+1*ssh_j+1 ) *r1_hv_0*r1_e1e2v  (:168)
    r3f = 0.25*( the four surrounding e1e2t*ssh )   *r1_hf_0*r1_e1e2f   (:177)

    The reference depths are NEMO's own (domain.F90:140-161):
    ht_0 = sum_k e3t_0*tmask, hu_0 = sum_k e3u_0*umask,
    hv_0 = sum_k e3v_0*vmask, hf_0 = sum_k e3f_0*vmask_i*vmask_i+1,
    each reciprocal taken as ssXmask/(h_0 + 1 - ssXmask) so a fully dry column
    gives exactly 0 rather than a division by zero.
    """
    ssh = _finite("ssh", ssh)
    e1e2t = mx["e1t"] * mx["e2t"]
    out = {}

    def _recip(h0, ssmask):
        return ssmask / (h0 + 1.0 - ssmask)

    ht_0 = (mx["e3t_0"] * mx["tmask"]).sum(axis=-1)
    ssmask = mx["tmask"].max(axis=-1)
    if "t" in want:
        out["t"] = ssh * _recip(ht_0, ssmask)
    if "u" in want:
        hu_0 = (mx["e3u_0"] * mx["umask"]).sum(axis=-1)
        ssumask = mx["umask"].max(axis=-1)
        out["u"] = (0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=1))
                    * _recip(hu_0, ssumask) / (mx["e1u"] * mx["e2u"]))
    if "v" in want:
        hv_0 = (mx["e3v_0"] * mx["vmask"]).sum(axis=-1)
        ssvmask = mx["vmask"].max(axis=-1)
        out["v"] = (0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=0))
                    * _recip(hv_0, ssvmask) / (mx["e1v"] * mx["e2v"]))
    if "f" in want:
        # hf_0: domain.F90:150 sums e3f_0 * vmask(i) * vmask(i+1).
        hf_0 = (mx["e3f_0"] * mx["vmask"]
                * np.roll(mx["vmask"], -1, axis=1)).sum(axis=-1)
        ssfmask = mx["fmask"].max(axis=-1)
        w = e1e2t * ssh
        quad = ((w + np.roll(w, -1, axis=1))
                + (np.roll(w, -1, axis=0) + np.roll(np.roll(w, -1, axis=0),
                                                    -1, axis=1)))
        out["f"] = 0.25 * quad * _recip(hf_0, ssfmask) / (mx["e1f"] * mx["e2f"])
    return out


def control_a(mx) -> float:
    """The transcription is untrusted code until it reproduces a dumped answer.

    r3f is not dumped; r3u and r3v are, and they come from the same routine
    three lines above the f-point branch.  Feed the DUMPED after-ssh (the ssh
    those dumps were written from) through the transcription and require the
    dumped r3u/r3v back.
    """
    print("\n=== CONTROL A -- the dom_qco_r3c transcription, against dumps ===")
    ssh_after = np.fromfile(RUN / "sshnxt_dump_ssh_after.bin",
                            dtype="<f8").reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]
    got = transcribe_r3(ssh_after, mx, want=("u", "v"))
    worst = 0.0
    for k, mask in (("u", mx["umask"].max(axis=-1) > 0.5),
                    ("v", mx["vmask"].max(axis=-1) > 0.5)):
        ref = np.fromfile(RUN / f"r3c_dump_r3{k}.bin",
                          dtype="<f8").reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]
        ref = _finite(f"r3{k} dump", ref)
        mine = _finite(f"r3{k} transcribed", got[k])
        scale = float(np.abs(ref[mask]).max())
        rel = float(np.abs(mine - ref)[mask].max()) / scale
        worst = max(worst, rel)
        print(f"  r3{k}: max|transcribed - dumped| / max|dumped| = {rel:.3e}"
              f"   (n={int(mask.sum())}, max|dumped|={scale:.4e})")
    print(f"  worst {worst:.3e} against the registered {BAR_R3_TRANSCRIPTION:.0e} bar: "
          f"{'PASS' if worst < BAR_R3_TRANSCRIPTION else 'FAIL'}")
    if worst >= BAR_R3_TRANSCRIPTION:
        raise SystemExit(
            f"the dom_qco_r3c transcription misses the dumped r3u/r3v by "
            f"{worst:.3e} -- the r3f it would produce is not trustworthy, so "
            "no e3f number is reported")
    return worst


def build_inputs(g, br, cfg, mc, state=None):
    """Reassemble exactly the arguments the model hands the EEN triad.

    Control B (below) proves this reassembly bit-identical to the model's own
    ``vortcor_u``; without that, swapping one argument would attribute a change
    to the swap that actually came from the reassembly.
    """
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        curl_vertex_cgrid, min_cell_to_uface, min_cell_to_vface,
        vertex_coriolis)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, compute_vertex_mask)
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate, compute_layer_thickness)
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx

    st = br.state if state is None else state
    u, v = st.u.data, st.v.data
    mask = st.land_mask.data
    z = br.z_coord
    # Mirrors ocean_pe_latlon_cgrid.py:3985-4000 and :1216-1219 verbatim.
    if isinstance(z, OceanPartialCellCoordinate):
        u_mask_3d, v_mask_3d = compute_face_masks_3d(z.is_active, br.geometry)
    else:
        u_mask_3d = st.u_mask.data[..., None]
        v_mask_3d = st.v_mask.data[..., None]
    eta_floor = mc.min_water_column_m - st.H_bathy.data
    eta_safe = jnp.maximum(st.eta.data, eta_floor) * mask
    h_k = compute_layer_thickness(eta_safe, st.H_bathy.data, z,
                                  min_water_column_m=mc.min_water_column_m)
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, br.geometry)
    h_vtx, _, _ = een_e3f_h_vtx(h_k, None, None, br.geometry,
                                mc.een_e3f_scheme, dz_ref=z.dz_ref)
    from legoesm.grids.latlon import ensure_geometry
    _g = ensure_geometry(br.geometry)
    # NEMO's horizontal scale factors at the u-/v-points, in legoESM's naming
    # (dx_u == e1u, dx_v == e1v, dy_u == e2u, dy_v == e2v). Carried on every
    # call so the two arms differ in EXACTLY this one argument.
    mw = (_g.dx_u, _g.dx_v, _g.dy_u, _g.dy_v)
    return dict(
        zeta=curl_vertex_cgrid(u, v, br.geometry),
        h_vtx=h_vtx, h_v=h_v, v=v, h_u=h_u, u=u,
        u_mask_3d=u_mask_3d, v_mask_3d=v_mask_3d,
        vtx_mask=compute_vertex_mask(mask, grid=br.geometry),
        f_vtx=vertex_coriolis(br.geometry),
        q_boundary=mc.een_q_boundary,
        metric_widths=(mw if getattr(mc, "een_metric_weighting", "off")
                       == "nemo" else None),
        metric_widths_nemo=mw,
        h_k=h_k, eta=np.asarray(eta_safe, dtype=np.float64), z=z,
    )


def run_triad(inp, **swap):
    """One EEN evaluation with zero or more arguments replaced."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        pv_flux_al81_partial_cell)
    a = dict(inp)
    a.update(swap)
    du, _dv = pv_flux_al81_partial_cell(
        a["zeta"], a["h_vtx"], a["h_v"], a["v"], a["h_u"], a["u"],
        a["u_mask_3d"], a["v_mask_3d"], a["vtx_mask"],
        f_vtx=a["f_vtx"], q_boundary=a["q_boundary"],
        metric_widths=a.get("metric_widths"))
    # The model publishes this diagnostic MASKED (ocean_pe_latlon_cgrid.py:4587
    # ``_mu(x) = x * u_mask_3d``); without the same mask the reassembly carries
    # land values the model never reports, which is exactly what control B
    # caught on its first run (3.0e-5 m/s2 of pure land).
    du = du * a["u_mask_3d"]
    return _u_to_nemo(np.asarray(du, dtype=np.float64))[..., :JPKM1]


def control_b(g, br, cfg, mc, inp) -> float:
    """The reassembly must reproduce the model's own diagnostic bit-for-bit."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    print("\n=== CONTROL B -- reassembly against the model's own vortcor_u ===")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    _t, diag = model.tendencies_with_diagnostics(
        br.state, surface_forcing=None, dt=float(cfg.dt))
    ref = _u_to_nemo(np.asarray(diag.vortcor_u.data, dtype=np.float64))[..., :JPKM1]
    mine = run_triad(inp)
    d = float(np.abs(mine - _finite("model vortcor_u", ref)).max())
    print(f"  max|reassembled - model diagnostic| = {d:.3e} m/s2 "
          f"(bit-identity required; {'PASS' if d == 0.0 else 'FAIL'})")
    if d != 0.0:
        raise SystemExit(
            f"the reassembled EEN call differs from the model's own "
            f"vortcor_u by {d:.3e} -- an argument swap against it would not be "
            "attributable to the swap")
    return ref


def control_c(mx, br) -> None:
    """Prove the F-point embedding, and that the south fill is inert."""
    print("\n=== CONTROL C -- the F-point index embedding ===")
    t = mx["tmask"]
    # dommsk.F90:152-153 rebuilt in legoESM VERTEX indexing: vertex[j,i] is the
    # SW corner of cell (j,i), so its four cells are (j-1,i-1),(j,i-1),
    # (j-1,i),(j,i).  Padding the south with land matches the dry cell row 0.
    tp = np.concatenate([t[:, -1:], t, t[:, :1]], axis=1)       # i = -1..n_lon
    z = np.zeros((1,) + tp.shape[1:])
    tp = np.concatenate([z, tp, z], axis=0)                     # j = -1..n_lat
    prod = tp[:-1, :-1] * tp[:-1, 1:] * tp[1:, :-1] * tp[1:, 1:]
    fm = _f_nemo_to_vtx(mx["fmask"], south=0.0)
    n_bad = int((np.abs(prod - fm) > 0).sum())
    print(f"  4-tmask product vs the mesh's own fmask, on the legoESM vertex "
          f"grid: {n_bad} of {prod.size} points disagree")
    if n_bad:
        # A disagreement is legitimate only at the bottom cap (dommsk.F90:158-170
        # zeroes fmask below mbkf); report where it is rather than swallow it.
        jj, ii, kk = np.nonzero(np.abs(prod - fm) > 0)
        print(f"    levels involved: {sorted(set(kk.tolist()))[:8]}, "
              f"rows {sorted(set(jj.tolist()))[:8]}")
    # The south vertex row of every embedded field carries a value this probe
    # INVENTED (``_f_nemo_to_vtx(..., south=...)``). Asserting that value is
    # what it was just set to proves nothing -- that circular check was
    # adversarial review finding 4 and is deleted. What licenses the fill is
    # the STENCIL: the operator's u-face row j reads vertex rows j and j+1
    # only, so vertex row 0 reaches nothing but the u-row 0 that ``umask``
    # already zeroes. That is an argument, so it is MEASURED instead --
    # ``--self-test`` (iii) sets the south row of h_vtx, f_vtx AND zeta to
    # 1e6 m and requires no scored wall number to move.
    print(f"  the south vertex row is an invented fill; its inertness is "
          f"measured by --self-test (iii) on all three embedded fields, not "
          f"asserted here")


def nemo_e3f_vtx(mx, br, mc, eta, dz_ref):
    """NEMO's ``e3f_vor`` on legoESM's vertex grid.

    ``e3f_0vor`` is built by calling legoESM's OWN production helper on the
    STATIC masked ``e3t_0`` -- the wet-count masked average is the identical
    arithmetic (dynvor.F90:927-939), so nothing is re-derived; only the field it
    is applied to changes (static reference thickness, not live thickness).
    The dry-vertex fallback ``dz_ref[k]`` is the same one the helper already
    implements for NEMO's ``e3f_0vor -> e3f_0`` overwrite (dynvor.F90:950).
    """
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
    # The dry-vertex fallback uses ``dz_ref[k]`` where NEMO overwrites with the
    # mesh's own ``e3f_0`` (dynvor.F90:950). The production code argues they are
    # the same per-level constant; the review asked for it to be checked rather
    # than argued, so it is checked here, once, directly.
    _e3f0 = mx["e3f_0"].reshape(-1, mx["e3f_0"].shape[-1])
    _spread = float(np.ptp(_e3f0, axis=0).max())
    _dz = np.asarray(dz_ref, dtype=np.float64)
    _lev = _e3f0.mean(axis=0)
    # Only the levels NEMO integrates. jpk (the last index) is never stepped --
    # vor_een loops jk = 1..jpkm1 -- and legoESM's dz_ref differs there
    # (506.37 m vs 617.46 m), which is real and inert. Reported, not swallowed.
    _gap = float(np.abs(_lev[:JPKM1] - _dz[:JPKM1]).max())
    print(f"  dry-vertex fallback: mesh e3f_0 per-level horizontal spread "
          f"{_spread:.3e} m; max|e3f_0 - dz_ref| over the {JPKM1} INTEGRATED "
          f"levels = {_gap:.3e} m; on the unused jpk level "
          f"{abs(_lev[JPKM1] - _dz[JPKM1]):.3e} m")
    if _spread > 1e-9 or _gap > 1e-9:
        raise SystemExit(
            f"dz_ref is not NEMO's e3f_0 on the integrated levels (spread "
            f"{_spread:.3e}, gap {_gap:.3e}) -- the fully-dry-vertex fallback "
            "the production helper uses is then not NEMO's, and this "
            "comparison is not clean")
    e3t0_masked = jnp.asarray(mx["e3t_0"] * mx["tmask"])
    e3f_0vor, _, _ = een_e3f_h_vtx(e3t0_masked, None, None, br.geometry,
                                   "nemo_avg", dz_ref=dz_ref)
    r3f = transcribe_r3(eta, mx, want=("f",))["f"]
    r3f_v = _f_nemo_to_vtx(r3f, south=0.0)[..., None]
    fe3 = _f_nemo_to_vtx(mx["fmask"], south=0.0)
    return jnp.asarray(np.asarray(e3f_0vor, dtype=np.float64)
                       * (1.0 + r3f_v * fe3)), np.asarray(e3f_0vor), r3f_v, fe3


def score(name, diff, wet3, tag=""):
    """The row reduction, reported under BOTH vertical weightings.

    RETRACTION, LOUD (both adversarial reviewers, 2026-08-23). Every number
    this campaign has published on the wall rows -- including the 3.10e-10
    headline -- was called "signed THICKNESS-weighted zonal mean". It is not.
    ``coherent_rows`` is called with ``h=None`` at every site in this probe and
    in the sibling (``wall_term_discriminators.py:383``), and that branch
    weights by the wet MASK, i.e. it is an unweighted mean over wet LEVELS.
    DINO's layers run 10.14 m to 545.20 m, a 53.8x range, so the published
    statistic over-weights the surface by ~54x relative to mass.

    That is not a wording defect alone. Rescored with the real ``h_u``, the
    wall-row mismatch is ~12x smaller AND CHANGES SIGN on three of the four
    wall rows -- and the sign is the leg the fix decision rests on. So both
    reductions are computed and printed for every leg from here on, the
    mass-weighted one is the one a depth-uniform geostrophic transport error
    responds to, and no verdict is taken on the level-mean alone.

    The signed number is the one the parent measured 3.10e-10 on and the one
    the registered bar is defined for -- imported from the sibling, not
    re-implemented. It is a MASK-weighted vertical mean, not a thickness-
    weighted one: ``coherent_rows`` is called with ``h=None``
    (``wall_term_discriminators.py:151-154``). Earlier wording in this probe,
    its pre-registration and its commit message said "thickness-weighted";
    that was wrong and is corrected here. The parent uses the identical call,
    so every ratio is self-consistent -- only the label was false.

    ADVERSARIAL REVIEW FINDING 1, acted on: the signed mean can cancel, and a
    factor that changes the field a great deal while alternating sign scores as
    "closes 0%". The non-cancelling ``mean|.|`` and the pointwise ``max|.|``
    are now printed beside it for EVERY leg, so an exoneration cannot rest on a
    cancellation. Leg 7 is the calibration of how much the reduction can
    suppress: 2.0e-20 pointwise against 8.2e-25 signed, five orders.
    """
    d = _finite(name, diff)

    if _H_U[0] is None:
        raise SystemExit(
            "score() was called before the mass weight was set -- it would "
            "silently fall back to the level mean, which is the defect this "
            "reduction exists to correct")

    def _red(h):
        sg, ab = coherent_rows(d, wet3, h)
        return (float(np.mean([abs(sg[j]) for j in WALL_ROWS])),
                float(np.mean([abs(sg[j]) for j in INTERIOR_ROWS_FAR])),
                float(np.mean([ab[j] for j in WALL_ROWS])), sg)

    wall, far, wall_a, signed = _red(None)
    hw, hf, hwa, hsigned = _red(_H_U[0])
    pw = float(np.abs(d[wet3]).max())
    enr = wall / far if far > 0 else float("inf")
    henr = hw / hf if hf > 0 else float("inf")
    print(f"  {name:<34s} LEVEL-mean  wall {wall:.4e} far {far:.4e} "
          f"enrich {enr:6.2f}x   |.| {wall_a:.4e}  max {pw:.4e}")
    print(f"  {'':<34s} MASS-weighted wall {hw:.4e} far {hf:.4e} "
          f"enrich {henr:6.2f}x")
    print(f"  {'':<34s} wall rows, level  "
          + " ".join(f"{signed[j]:+.2e}" for j in WALL_ROWS))
    print(f"  {'':<34s} wall rows, mass   "
          + " ".join(f"{hsigned[j]:+.2e}" for j in WALL_ROWS) + tag)
    return wall, far, enr, signed, wall_a, hw, pw, hf, hsigned


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()
    _stamp_build()

    mx = read_mesh_extras()
    g, br, cfg, mc = build_lego()
    print(f"\nresolved: vorticity_scheme={mc.vorticity_scheme!r} "
          f"een_q_boundary={mc.een_q_boundary!r} "
          f"een_e3f_scheme={mc.een_e3f_scheme!r}")
    _gate_dry_faces(br)
    control_a(mx)
    control_c(mx, br)

    inp = build_inputs(g, br, cfg, mc)
    # The MASS weight for the corrected reduction: the u-face thickness the
    # tendency actually acts on, in NEMO layout, on the integrated levels.
    _H_U[0] = _finite("h_u", _u_to_nemo(
        np.asarray(inp["h_u"], dtype=np.float64))[..., :JPKM1])
    _ = control_b(g, br, cfg, mc, inp)
    # THE SCORING BASELINE IS THE HISTORICAL OPERATOR -- the per-unit-width
    # form, with no metric weighting -- whatever the card currently selects.
    # This is what the parent measured 3.10e-10 against, so pinning it here
    # keeps the number comparable after the fix was wired onto the card.
    lego_base = run_triad(inp, metric_widths=None)
    print(f"  scoring baseline pinned to metric_widths=None (the historical "
          f"operator); the card resolves een_metric_weighting="
          f"{getattr(mc, 'een_metric_weighting', 'off')!r}")

    umask3 = np.asarray(g.umask)[..., :JPKM1] > 0.5
    d03 = _load_full_3d(str(RUN / "stp_dump_03_dynadv_du.bin"), JPI, JPJ, JPKM1, HLS)
    d04 = _load_full_3d(str(RUN / "stp_dump_04_dynvor_du.bin"), JPI, JPJ, JPKM1, HLS)
    nemo_vor = _finite("NEMO vorticity flux", d04 - d03)

    print("\n=== THE MISMATCH AS THE PARENT MEASURED IT ===")
    w0, f0, e0, s_before, a0, h0, p0, hf0, hs0 = score(
        "R_before = NEMO - legoESM", nemo_vor - lego_base, umask3)
    print(f"  parent published {MEASURED_WALL:.2e} m/s2 at the wall; this run "
          f"{w0:.4e} ({w0 / MEASURED_WALL:.3f}x)")

    # ---- the two F-point thicknesses, on the same state -------------------
    e3f_nemo, e3f_0vor, r3f_v, fe3 = nemo_e3f_vtx(
        mx, br, mc, inp["eta"], br.z_coord.dz_ref)
    h_vtx_lego = np.asarray(inp["h_vtx"], dtype=np.float64)
    e3f_n = np.asarray(e3f_nemo, dtype=np.float64)
    print("\n=== THE e3f DIFF MAP ===")
    # Only vertices the triad can reach: at least one wet surrounding face.
    live = _f_nemo_to_vtx(mx["fmask"], south=0.0)[..., :JPKM1] > 0.5
    edge = (~live) & (_f_nemo_to_vtx(
        np.maximum.reduce([mx["umask"], np.roll(mx["umask"], -1, axis=0),
                           mx["vmask"], np.roll(mx["vmask"], -1, axis=1)]),
        south=0.0)[..., :JPKM1] > 0.5)
    rel = np.abs(h_vtx_lego[..., :JPKM1] - e3f_n[..., :JPKM1]) / np.maximum(
        np.abs(e3f_n[..., :JPKM1]), 1e-30)
    for lbl, m in (("fully-wet vertices (fe3mask=1)", live),
                   ("wall vertices (fe3mask=0, a wet face)", edge)):
        if not m.any():
            print(f"  {lbl}: none")
            continue
        print(f"  {lbl}: n={int(m.sum())}  mean rel {float(rel[m].mean()):.3e}"
              f"  max rel {float(rel[m].max()):.3e}")
    wall_v = np.zeros(rel.shape[:2], bool)
    wall_v[WALL_ROWS[0]:WALL_ROWS[-1] + 2] = True
    mw = wall_v[..., None] & (live | edge)
    print(f"  the four wall rows: n={int(mw.sum())}  mean rel "
          f"{float(rel[mw].mean()):.3e}  max rel {float(rel[mw].max()):.3e}")
    print(f"  r3f at the wall rows: mean {float(r3f_v[wall_v, 0].mean()):.4e}, "
          f"max |{float(np.abs(r3f_v[wall_v, 0]).max()):.4e}|")

    # ---- LEG 1: propagate the e3f swap through the SAME triad code --------
    print("\n=== LEG 1 -- the e3f mechanism, propagated ===")
    lego_e3f = run_triad(inp, h_vtx=e3f_nemo, metric_widths=None)
    wp, fp, ep, s_pred, ap, hp, pp, hfp, hsp = score(
        "predicted difference (e3f)", lego_e3f - lego_base, umask3)
    wa, fa, ea, _s, aa, ha, pa, hfa, hsa = score(
        "R_after = NEMO - legoESM(e3f)", nemo_vor - lego_e3f, umask3)
    ratio_a = abs(wp / MEASURED_WALL)
    leg_a = (1.0 / LEG_A_FACTOR) <= ratio_a <= LEG_A_FACTOR
    leg_b = ep >= BAR_ENRICH
    # Sign: the swap must move legoESM TOWARD NEMO, i.e. the predicted
    # difference must share the sign of R_before, row by row on the wall rows.
    # Leg c is scored on the MASS-weighted signs. The registration wrote
    # "the sign that strengthens legoESM's westward lobe", which is a statement
    # about a DEPTH-MEAN acceleration; the level mean is not that quantity, and
    # under the two reductions the wall-row signs disagree. Both are printed.
    same = [int(np.sign(hsp[j]) == np.sign(hs0[j])) for j in WALL_ROWS]
    same_lvl = [int(np.sign(s_pred[j]) == np.sign(s_before[j]))
                for j in WALL_ROWS]
    leg_c = sum(same) >= 3
    ratio_1b = wa / w0 if w0 > 0 else float("inf")
    print(f"\n  leg a  magnitude {wp:.4e} vs {MEASURED_WALL:.2e} "
          f"= {ratio_a:.3f}x  -> {'PASS' if leg_a else 'FAIL'}")
    print(f"  leg b  enrichment {ep:.2f}x vs {BAR_ENRICH}  "
          f"-> {'PASS' if leg_b else 'FAIL'}")
    print(f"  leg c  sign agrees on {sum(same)}/4 wall rows {same} "
          f"(MASS-weighted, the scored one)  -> {'PASS' if leg_c else 'FAIL'}")
    print(f"         on the LEVEL mean it would be {sum(same_lvl)}/4 "
          f"{same_lvl} -- the two reductions disagree, which is the finding")
    verdict_1 = "CONFIRMED" if (leg_a and leg_b and leg_c) else "REFUTED"
    print(f"  registered legs a+b+c: {verdict_1}")
    v1b = ("OWNER" if ratio_1b < BAR_OWNER else
           "CONTRIBUTOR" if ratio_1b <= BAR_CONTRIB else "REFUTED as owner")
    print(f"  leg 1b residual ratio |R_after|/|R_before| = {wa:.4e}/{w0:.4e} "
          f"= {ratio_1b:.3f}  -> {v1b}")

    # ---- LEGS 2-5: the decomposition, whatever leg 1 did ------------------
    print("\n=== LEGS 2-5 -- the term-by-term decomposition ===")
    swaps = {}
    swaps["2 f_vtx (mesh ff_f)"] = dict(
        f_vtx=_f_nemo_to_vtx(mx["ff_f"], south=0.0))
    swaps["3 zeta (dynvor metrics)"] = dict(zeta=_nemo_zeta(mx, br, inp))
    swaps["4 mass fluxes (e3u/e3v)"] = _nemo_face_thickness(mx, br, inp)
    # ADVERSARIAL REVIEW FINDING 2, acted on: ``run_triad`` passes ELEVEN
    # arguments and only five were swapped, so "every input is exonerated" was
    # false as written. The remaining four are covered here.
    #   - q_boundary: NEMO applies NO fill (ln_dynvor_msk=.false.), and the card
    #     already selects "nemo_live". Swapping to the fill is a SENSITIVITY,
    #     not an exoneration -- it says how much the convention is worth, which
    #     is the number that matters in a WALL investigation.
    #   - vtx_mask: unused under "nemo_live" (it only drives the fill). Proved
    #     rather than assumed, by replacing it with zeros.
    #   - u_mask_3d / v_mask_3d: replaced by the mesh's own umask/vmask.
    swaps["2b q_boundary -> neumann_fill"] = dict(q_boundary="neumann_fill")
    swaps["2c vtx_mask -> all zero"] = dict(
        vtx_mask=np.zeros_like(np.asarray(inp["vtx_mask"], dtype=np.float64)))
    swaps["2d face masks (mesh u/vmask)"] = dict(
        u_mask_3d=np.concatenate(
            [mx["umask"][:, -1:], mx["umask"]], axis=1),
        v_mask_3d=np.concatenate(
            [np.zeros((1,) + mx["vmask"].shape[1:]), mx["vmask"]], axis=0))
    swaps["5 all of 1-4 together"] = dict(
        h_vtx=e3f_nemo, **swaps["2 f_vtx (mesh ff_f)"],
        **swaps["3 zeta (dynvor metrics)"], **swaps["4 mass fluxes (e3u/e3v)"])
    rows = [("1 e3f (h_vtx)", wa, ratio_1b, fa / f0 if f0 > 0 else float("inf"),
             ha / h0 if h0 > 0 else float("inf"), pa)]
    for k, sw in swaps.items():
        out = run_triad(inp, metric_widths=None, **sw)
        w, f_, _e, _s, wa_, h_, pw_, hf_, _hs = score(
            f"R_after [{k}]", nemo_vor - out, umask3)
        rows.append((k, w, w / w0 if w0 > 0 else float("inf"),
                     f_ / f0 if f0 > 0 else float("inf"),
                     h_ / h0 if h0 > 0 else float("inf"), pw_))

    # ---- LEG 6: the metric weighting on the mass flux --------------------
    # dynvor.F90:791-792 weights the meridional transport by e1v (the V-face
    # ZONAL WIDTH) and :804 divides the assembled u-tendency by e1u.  legoESM's
    # triad uses the bare h_v*v and no 1/e1u.  The two agree only where
    # e1v == e1u; on a latitude-longitude grid e1 = R*cos(phi)*dlambda, so they
    # differ by the cos-latitude ratio across a cell -- which grows as
    # dphi*tan(phi) and is therefore LARGEST at the southern wall.
    # The weighting is separable: e1v multiplies per v-face (fold into h_v) and
    # 1/e1u divides per u-face (applied to the result), so it is measurable
    # without touching the operator.
    print("\n=== LEG 6 -- the e1v / e1u metric weighting on the transport ===")
    ratio_metric = mx["e1v"] / mx["e1u"]
    print(f"  e1v/e1u: wall rows {[f'{ratio_metric[j].mean():.6f}' for j in WALL_ROWS]}"
          f"   far interior "
          f"{[f'{ratio_metric[j].mean():.6f}' for j in INTERIOR_ROWS_FAR]}")
    out6 = run_triad(inp, metric_widths=inp["metric_widths_nemo"])
    # CROSS-CHECK: the same weighting done EXTERNALLY (scale the meridional
    # flux per v-face by e1v, divide the assembled u-tendency per u-face by
    # e1u) -- which is exactly how the barotropic path has always applied it
    # (barotropic_latlon_cgrid.een_barotropic_coriolis). The measurement and
    # the shipped option must agree, or one of them is wrong.
    inp6 = dict(inp)
    inp6["h_v"] = np.asarray(inp["h_v"], dtype=np.float64) * _e1v_on_lego_vface(mx)
    out6_ext = run_triad(inp6, metric_widths=None) / mx["e1u"][..., None]
    _xc = float(np.abs(out6 - out6_ext).max())
    _rms = float(np.sqrt(np.mean(nemo_vor[umask3] ** 2)))
    print(f"  cross-check, in-operator vs external fold: max|difference| = "
          f"{_xc:.3e} m/s2, {_xc / _rms:.3e} of the term's RMS")
    # The two folds are the SAME algebra on DIFFERENT metric arrays: the option
    # uses legoESM's own geometry (a model option cannot depend on a mesh file),
    # the external fold uses NEMO's mesh. So the cross-check also MEASURES how
    # far the two geometries are apart -- which is the real content of any
    # residual here, and is reported rather than left as a loose end.
    from legoesm.grids.latlon import ensure_geometry
    _g = ensure_geometry(br.geometry)
    for nm, lego, nemo in (
            ("e1u (dx_u)", _u_to_nemo(np.asarray(_g.dx_u, dtype=np.float64)),
             mx["e1u"]),
            ("e1v (dx_v)", np.asarray(_g.dx_v, dtype=np.float64)[1:],
             mx["e1v"])):
        rel = np.abs(lego - nemo) / np.maximum(np.abs(nemo), 1e-30)
        print(f"    {nm}: max relative legoESM-vs-mesh {float(rel.max()):.3e}"
              f"  (wall rows {float(rel[WALL_ROWS[0]:WALL_ROWS[-1]+1].max()):.3e},"
              f"  argmax row {int(np.unravel_index(rel.argmax(), rel.shape)[0])})")
    w6, f6, e6, _s6, a6, h6, p6w, hf6, hs6 = score(
        "R_after [6 e1v/e1u weighting]", nemo_vor - out6, umask3)
    rows.append(("6 e1v/e1u metric weighting", w6,
                 w6 / w0 if w0 > 0 else float("inf"),
                 f6 / f0 if f0 > 0 else float("inf"),
                 h6 / h0 if h0 > 0 else float("inf"), p6w))
    _p6 = score("predicted difference (metric)", out6 - lego_base, umask3)
    same6 = [int(np.sign(_p6[8][j]) == np.sign(hs0[j])) for j in WALL_ROWS]
    print(f"  SIGN, MASS-weighted: the metric correction agrees with "
          f"R_before on {sum(same6)}/4 wall rows {same6}. Disagreement means "
          f"supplying it moves the wall rows AWAY from NEMO in the depth mean "
          f"-- the direction a depth-uniform geostrophic transport error obeys.")

    # ---- LEG 7: NEMO's assembly transcribed verbatim ---------------------
    # If this reproduces the dump to the matched-operator floor, the ledger is
    # closed: everything not explained by legs 1-6 lives in the assembly, and
    # the transcription says exactly how much.
    print("\n=== LEG 7 -- NEMO's vor_een assembly, transcribed verbatim ===")
    zua = _nemo_vor_een_verbatim(mx, inp, e3f_nemo)
    w7, f7, e7, _s7, a7, h7, p7, hf7, hs7 = score(
        "NEMO dump - NEMO transcribed", nemo_vor - zua, umask3)
    rows.append(("7 NEMO assembly (all NEMO inputs)", w7,
                 w7 / w0 if w0 > 0 else float("inf"),
                 f7 / f0 if f0 > 0 else float("inf"),
                 h7 / h0 if h0 > 0 else float("inf"), p7))
    rms = float(np.sqrt(np.mean(nemo_vor[umask3] ** 2)))
    pw = float(np.abs((zua - nemo_vor)[umask3]).max())
    print(f"  max|transcribed - dumped| = {pw:.4e} m/s2 over all wet cells; "
          f"RMS(dump) = {rms:.4e}; ratio {pw / rms:.3e}")
    w8, f8, e8, _s8, a8, h8, p8, hf8, hs8 = score(
        "NEMO transcribed - legoESM", zua - lego_base, umask3)
    print(f"  the transcription-vs-legoESM gap carries {w8 / w0:.3f} of the "
          f"measured wall mismatch")

    # ---- LEG 8: the ASSEMBLY ALONE, on legoESM's own inputs --------------
    # Review finding 2: legs 1-6 swap INPUTS and leg 7 swaps everything, so
    # "the owner is the assembly" was an inference. This measures it directly:
    # NEMO's 12-point assembly (its triad index pattern, its triad-to-flux
    # pairing, and its e1v/e1u metric) driven by legoESM's OWN potential
    # vorticity and face thickness.
    print("\n=== LEG 8 -- the assembly ALONE, on legoESM's own inputs ===")
    q_lego = (np.asarray(inp["zeta"], dtype=np.float64)
              + np.asarray(inp["f_vtx"], dtype=np.float64)[..., None]) / np.maximum(
        np.asarray(inp["h_vtx"], dtype=np.float64), 1.0e-10)
    zua8 = _nemo_vor_een_verbatim(mx, inp, e3f_nemo, lego_q=q_lego,
                                  lego_h_v=inp["h_v"])
    w9, f9, e9, _s9, a9, h9, p9, hf9, hs9 = score(
        "R_after [8 NEMO assembly, lego inputs]", nemo_vor - zua8, umask3)
    rows.append(("8 NEMO assembly on lego inputs", w9,
                 w9 / w0 if w0 > 0 else float("inf"),
                 f9 / f0 if f0 > 0 else float("inf"),
                 h9 / h0 if h0 > 0 else float("inf"), p9))
    # And the pure INDEXING residual: leg 8 against leg 6 (legoESM's operator
    # WITH the metric). If these agree, the metric is the only assembly
    # difference and the triad index pattern is identical.
    wi, fi, ei, _si, ai, hi, pi, hfi, hsi = score(
        "leg 8 - leg 6 (pure triad indexing)", zua8 - out6, umask3)
    print(f"  the residual between NEMO's assembly and legoESM's WITH the "
          f"metric is {wi / w0:.4f} of the measured mismatch -- what is left "
          f"of the assembly once the metric is supplied")
    # ADVERSARIAL REVIEW FINDING 3, acted on: the table used to be wall-only,
    # while the mechanism named (e1v/e1u ~ 1 - (dlambda/2)*sin(phi)) deviates
    # from 1 at BOTH poleward ends. Whether this is "the wall owner" or a
    # GLOBAL metric defect that happens to have been measured at the wall is
    # decided by the far-interior column, which score() already computed and
    # the table discarded. It is printed now, beside the non-cancelling |.|
    # closure and the pointwise max, so no exoneration can rest on a signed
    # cancellation.
    print(f"\n  {'factor':<36s}{'|R| wall':>13s}{'wall':>8s}{'far':>8s}"
          f"{'mass':>8s}{'max pointwise':>16s}")
    for k, w, r, rf, ra, pw in sorted(rows, key=lambda x: x[2]):
        print(f"  {k:<36s}{w:>13.4e}{r:>8.3f}{rf:>8.3f}{ra:>8.3f}{pw:>16.4e}")
    print(f"  ratios are R_after/R_before; 0.000 = closes it, 1.000 = closes "
          f"nothing")
    print(f"  columns: wall/far = LEVEL-mean ratios, mass = MASS-weighted "
          f"wall ratio (the reduction a depth-uniform transport error obeys)")
    print(f"  baseline R_before: level wall {w0:.4e} far {f0:.4e} | MASS wall "
          f"{h0:.4e} far {hf0:.4e} | max pointwise {p0:.4e} m/s2")
    print(f"  baseline R_before wall rows, MASS-weighted: "
          + " ".join(f"{hs0[j]:+.3e}" for j in WALL_ROWS))

    if args.self_test:
        return self_test(g, br, cfg, mc, mx, inp, umask3, nemo_vor, e3f_nemo)
    return 0


def _nemo_zeta(mx, br, inp):
    """dynvor.F90:757-758 relative vorticity, on NEMO's own mesh metrics."""
    import jax.numpy as jnp
    u = np.asarray(inp["u"], dtype=np.float64)
    v = np.asarray(inp["v"], dtype=np.float64)
    # NEMO: zwz(ji,jj) = ( (e2v(i+1)*v(i+1) - e2v(i)*v(i))
    #                    - (e1u(j+1)*u(j+1) - e1u(j)*u(j)) ) * r1_e1e2f
    # in NEMO F indexing; u/v here are legoESM-indexed, so map first.
    un = _u_to_nemo(u)                                   # (ny, nx, nz)
    vn = np.asarray(v)[1:]                               # v-face j -> NEMO jj
    e2v = mx["e2v"][..., None]
    e1u = mx["e1u"][..., None]
    curl = ((np.roll(e2v * vn, -1, axis=1) - e2v * vn)
            - (np.roll(e1u * un, -1, axis=0) - e1u * un))
    curl = curl / (mx["e1f"] * mx["e2f"])[..., None]
    return jnp.asarray(_f_nemo_to_vtx(curl, south=0.0))



def _e1v_on_lego_vface(mx):
    """NEMO's ``e1v`` on legoESM's v-face index.

    legoESM v-face ``j`` is the SOUTH face of cell ``j``; NEMO's V-point ``jj``
    is the NORTH face of cell ``jj``, i.e. legoESM v-face ``jj+1``.  Row 0 is
    the southern domain boundary, which is a closed wall (``v_mask == 0``), so
    the value placed there can never reach a flux -- the self-test plants a
    large value there and requires no scored number to move.
    """
    e1v = mx["e1v"]
    return np.concatenate([e1v[:1], e1v], axis=0)[..., None]


def _nemo_vor_een_verbatim(mx, inp, e3f_nemo, lego_q=None, lego_h_v=None):
    """``vor_een`` (dynvor.F90:718-810) transcribed, in NEMO (jj,ji) indexing.

    Every input is NEMO's own: ``e3f_vor`` as built above, ``ff_f`` and the
    metrics from the mesh, ``e3u``/``e3v`` from ``e3u_0*(1+r3u*umask)``.  Only
    the assembly is under test.  Rows 0 and ny-1 use a periodic roll in j and
    are therefore NOT valid; no scored row is one of them (the wall rows are
    1-4, whose j-1 is the real land row 0).
    """
    u = _u_to_nemo(np.asarray(inp["u"], dtype=np.float64))
    v = np.asarray(inp["v"], dtype=np.float64)[1:]        # -> NEMO V index
    e3f = np.asarray(e3f_nemo, dtype=np.float64)[1:, 1:]  # -> NEMO F index
    # Only e3v is needed: this transcription assembles zua, which reads zwy
    # (the meridional flux) alone. e3u/zwx belong to zva and are not scored.
    r3 = transcribe_r3(inp["eta"], mx, want=("v",))
    e3v = mx["e3v_0"] * (1.0 + r3["v"][..., None] * mx["vmask"])
    e2v, e1u = mx["e2v"][..., None], mx["e1u"][..., None]
    r1_e1e2f = (1.0 / (mx["e1f"] * mx["e2f"]))[..., None]

    def ip(a):     # ji+1, periodic in i (axis 1)
        return np.roll(a, -1, axis=1)

    def im(a):     # ji-1
        return np.roll(a, 1, axis=1)

    def jp(a):     # jj+1
        return np.roll(a, -1, axis=0)

    def jm(a):     # jj-1
        return np.roll(a, 1, axis=0)

    # zwz = ( ff_f + curl ) / e3f_vor            (dynvor.F90:771-773, :719)
    if lego_q is None:
        curl = ((ip(e2v * v) - e2v * v) - (jp(e1u * u) - e1u * u)) * r1_e1e2f
        zwz = ((mx["ff_f"][..., None] + curl) / e3f)[..., :JPKM1]
    else:
        # ASSEMBLY-ALONE arm (adversarial review finding 2): NEMO's assembly
        # driven by legoESM's OWN potential vorticity, so the only thing under
        # test is the 12-point assembly -- the triad index pattern, the
        # triad-to-flux pairing, and the e1v/e1u metric. Everything that feeds
        # it is legoESM's.
        zwz = np.asarray(lego_q, dtype=np.float64)[1:, 1:, :JPKM1]
    # horizontal fluxes                          (dynvor.F90:791-792)
    _e3v = e3v if lego_h_v is None else np.asarray(
        lego_h_v, dtype=np.float64)[1:]
    zwy = (mx["e1v"][..., None] * _e3v * v)[..., :JPKM1]
    # triads                                     (dynvor.F90:797-800)
    ztne = im(zwz) + zwz + jm(zwz)
    ztnw = jm(im(zwz)) + im(zwz) + zwz
    ztse = zwz + jm(zwz) + jm(im(zwz))
    ztsw = jm(zwz) + jm(im(zwz)) + im(zwz)
    # zua                                        (dynvor.F90:804-805)
    zua = (1.0 / 12.0) * (1.0 / mx["e1u"][..., None]) * (
        (ztne * zwy + ip(ztnw) * ip(zwy))
        + (ztse * jm(zwy) + ip(ztsw) * jm(ip(zwy))))
    return zua * (mx["umask"][..., :JPKM1] > 0.5)


def _nemo_face_thickness(mx, br, inp):
    """NEMO's live face thicknesses e3u = e3u_0*(1+r3u*umask), same for v."""
    import jax.numpy as jnp
    r3 = transcribe_r3(inp["eta"], mx, want=("u", "v"))
    e3u = mx["e3u_0"] * (1.0 + r3["u"][..., None] * mx["umask"])
    e3v = mx["e3v_0"] * (1.0 + r3["v"][..., None] * mx["vmask"])
    # NEMO u-point i is legoESM u-face i+1; wrap face 0 from the last column.
    hu = np.concatenate([e3u[:, -1:], e3u], axis=1)
    hv = np.concatenate([np.zeros((1,) + e3v.shape[1:]), e3v], axis=0)
    return dict(h_u=jnp.asarray(hu), h_v=jnp.asarray(hv))


def self_test(g, br, cfg, mc, mx, inp, umask3, nemo_vor, e3f_nemo) -> int:
    """The dry-row trap and non-vacuity, on the quantity this probe scores.

    (i)  the entirely dry southern land row IS read by the EEN flux on BOTH
         sides by design (ln_dynvor_msk=.false.), so the plant is reported with
         its size rather than forbidden -- what must hold is that it moves the
         two sides the SAME way, which is why the dry-face gate is the real
         protection and runs on the scoring path.
    (ii) planting a WET wall row must move the scored leg-1 number, or the
         measurement cannot fail.
    (iii) the south fill of the embedded F-point fields must be inert: refill
         it with a large value and require the scored number not to move.
    """
    print("\n=== SELF-TEST ===")
    base = run_triad(inp, h_vtx=e3f_nemo)
    b_w = score("baseline leg-1 arm", nemo_vor - base, umask3)[0]

    s = br.state
    uw = np.asarray(s.u.data, dtype=np.float64).copy()
    uw[0, :, :] = 1e3
    st0 = s._replace(u=s.u.replace(data=uw))
    i0 = build_inputs(g, br, cfg, mc, state=st0)
    e0, _, _, _ = nemo_e3f_vtx(mx, br, mc, i0["eta"], br.z_coord.dz_ref)
    d0 = score("dry-row plant (1e3 m/s on row 0)",
               nemo_vor - run_triad(i0, h_vtx=e0), umask3)[0]
    print(f"(i) row-0 plant moves the wall score {b_w:.4e} -> {d0:.4e}; the "
          f"coastal f-point is deliberately unmasked, so row 0 IS in the "
          f"stencil on both sides -- the dry-face gate is what makes the two "
          f"models agree about it.")

    uw2 = np.asarray(s.u.data, dtype=np.float64).copy()
    uw2[WALL_ROWS[0], :, :] += 1e-2
    st1 = s._replace(u=s.u.replace(data=uw2))
    i1 = build_inputs(g, br, cfg, mc, state=st1)
    e1, _, _, _ = nemo_e3f_vtx(mx, br, mc, i1["eta"], br.z_coord.dz_ref)
    d1 = score("wet wall-row plant (+1e-2 m/s)",
               nemo_vor - run_triad(i1, h_vtx=e1), umask3)[0]
    assert d1 != b_w, ("planting a WET wall row moved nothing -- the leg-1 "
                       "measurement cannot fail and proves nothing")
    print(f"(ii) wet wall-row plant moved the wall score "
          f"{b_w:.4e} -> {d1:.4e}")

    # (iii) THE SOUTH FILL, on EVERY embedded F-point field.
    # Adversarial review finding 4: control C's ``max|fe3mask[0]| == 0`` assert
    # is CIRCULAR -- it checks the constant ``_f_nemo_to_vtx`` had just written
    # -- and the earlier version of this plant covered only ``h_vtx``. The same
    # invented south row is injected into ``f_vtx`` (leg 2) and ``zeta`` (leg
    # 3), where it replaces genuinely NON-ZERO legoESM values, and was never
    # perturbed there. All three are perturbed now. The circular assert is
    # gone; this is the evidence that replaces it.
    import jax.numpy as jnp
    e_alt = np.asarray(e3f_nemo, dtype=np.float64).copy()
    e_alt[0] = 1.0e6
    f_alt = _f_nemo_to_vtx(mx["ff_f"], south=0.0).copy()
    f_alt[0] = 1.0e6
    z_alt = np.asarray(inp["zeta"], dtype=np.float64).copy()
    z_alt[0] = 1.0e6
    f_ref = _f_nemo_to_vtx(mx["ff_f"], south=0.0)
    z_ref = np.asarray(inp["zeta"], dtype=np.float64)
    # Each pair must differ ONLY in the south row -- an earlier version compared
    # the perturbed NEMO e3f against legoESM's OWN h_vtx, so it measured the
    # e3f swap and fired for the wrong reason.
    for label, ref_sw, sw in (
            ("h_vtx", dict(h_vtx=jnp.asarray(e3f_nemo)),
             dict(h_vtx=jnp.asarray(e_alt))),
            ("f_vtx", dict(f_vtx=f_ref), dict(f_vtx=f_alt)),
            ("zeta", dict(zeta=z_ref), dict(zeta=z_alt))):
        d2 = score(f"south row of {label} set to 1e6",
                   nemo_vor - run_triad(inp, metric_widths=None, **sw),
                   umask3)[0]
        ref = score("  (the same field, unperturbed)",
                    nemo_vor - run_triad(inp, metric_widths=None, **ref_sw),
                    umask3)[0]
        assert d2 == ref, (
            f"perturbing the south vertex row of {label} moved the wall score "
            f"{ref:.6e} -> {d2:.6e} -- the invented south fill of the embedded "
            "F-point fields is NOT inert, so every leg that uses one is "
            "contaminated by a value this probe made up")
        print(f"(iii) south row of {label} set to 1e6 moved the wall score "
              f"not at all ({ref:.6e})")
    print("SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
