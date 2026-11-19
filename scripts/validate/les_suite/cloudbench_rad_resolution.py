"""Where does the CloudBench radiation gap come from: scheme, averaging or resolution?

Reference: the LES's own stored day-5 radiation (swirl-lm ``rad_flux_lw`` +
``rad_flux_sw`` = net up - down at each node's BOTTOM face, store z = k dz, node k
centred at (k + 1/2) dz; ``rad_heat_src`` = g dF / (c_p dp) with dp the central
node-pressure difference; ``olr`` / ``asr`` at the top of its extended column).
The LES grid here is store nodes 0 .. nz-2 (faces 0 .. nz-1 all stored, top face
at the store's last z); its face pressures are halfway in ln p between the
store's ``p_ref`` nodes (ln-linear extrapolation at the ground). All LES means are
taken in float64 over all 73 day-5 frames and the plane.

Per case:
  instrument  heating rebuilt from the mean fluxes (swirl-lm formula, G, CP) vs
              the stored mean heating; the store's p_ref vs the GCM sounding's
              pressure at the LES faces; the LES condensate's ice share under
              swirl-lm's exact split (``cloudbench_radiation.liquid_fraction`` at
              width 0).
  S (scheme)  our radiation, independent column by column, on LES columns sampled
              from the store (frames every 8th; per frame 12 random and 12
              largest-LWP columns), against the LES fluxes of the same columns and
              frames (the LES radiation lags its state by up to its 240 s update).
  A (average) our radiation of the plane- and day-5-mean LES state on the same LES
              grid: clouds spread uniformly, clear sky, LES cover (McICA).
  R (resolution) our radiation on our 32 SCM layers (state as the replay builds
              it: the artifact's layer means, sounding pressure; and R_lesp_*: the
              same heights on the LES's own p_ref pressure), same variants,
              against the LES heating on our layers taken conservatively (the LES
              net flux at our interface heights, differenced over our layers).
              A's fluxes coarse-grained the same way give the resolution-free
              counterpart.

Gate numbers per comparison: net downward flux error at the column top (the
LES's last face), the ground and the LES cloud top [W/m2]; OLR and absorbed SW at
the top of each extension [W/m2] (ours stops at the GCM sounding's top, the LES's
at its own extension top; recorded, not matched); heating error per layer, max
|.| and pressure-weighted RMS [K/day], with the worst layer's index (top-first).
Numbers only, no verdict.

usage: cloudbench_rad_resolution.py OUT_JSON CASE [CASE ...]   (CASE = sSSS_mMM)
Run with JAX_ENABLE_X64=1 on a compute node.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloudbench_rad_validate import (  # noqa: E402
    ART, FORCING, NLEV, _p_of_z, day5_state, layer_mean, thermo)
from legoesm import constants  # noqa: E402
from legoesm.atmosphere.les_suite.bridge import load_artifact  # noqa: E402
from legoesm.atmosphere.les_suite.cloudbench import (  # noqa: E402
    STORE_URL, parse_case_name, read_sounding)
from legoesm.atmosphere.les_suite.cloudbench_radiation import (  # noqa: E402
    case_radiation, column_fluxes, liquid_fraction)
from legoesm.atmosphere.les_suite.cloudbench_scm import cloudbench_grid  # noqa: E402

# swirl-lm physics/constants.py: used only to rebuild the LES's own heating
SWIRL_G = 9.81
SWIRL_CP = 1.4 * 286.69 / 0.4
MEAN_VARS = ("T", "q_t", "q_c", "rho", "p_ref", "rad_flux_lw", "rad_flux_sw", "rad_heat_src")
PROFILE_VARS = ("cloud_fraction",)                 # stored already plane-mean, (t, z)
SCALAR_VARS = ("olr", "asr", "lwp", "cloud_cover")
COL_VARS = ("T", "q_t", "q_c", "rho", "rad_flux_lw", "rad_flux_sw")
FRAME_STEP = 8
N_RANDOM = 12
N_CLOUDY = 12
CLOUD_QC = 1e-5
COL_CHUNK = 24                                     # columns radiated per compiled batch


def les_store(url, seed):
    """Float64 plane/day-5 means, scalars, and sampled columns from the LES store."""
    ds = xr.open_zarr(url, consolidated=True)
    nt = ds.sizes["t"]

    def mean_frame(i):
        f = ds.isel(t=i)
        r = {v: f[v].astype("float64").mean(("x", "y")).values for v in MEAN_VARS}
        T, qc = (np.asarray(f[v].values, np.float64) for v in ("T", "q_c"))
        # the reference's phase split per cell, before any averaging
        r["q_ice"] = (qc * (1.0 - np.asarray(liquid_fraction(jnp.asarray(T), 0.0)))).mean((0, 1))
        return r
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(mean_frame, range(nt)))
    out = {v: np.stack([r[v] for r in rows]) for v in MEAN_VARS + ("q_ice",)}
    for v in PROFILE_VARS:
        if ds[v].dims != ("t", "z"):
            raise ValueError(f"{v} dims {ds[v].dims}, expected (t, z)")
        out[v] = np.asarray(ds[v].values, np.float64)
    for v in SCALAR_VARS:
        out[v] = np.asarray(ds[v].values, np.float64)
    out["z_node"] = np.asarray(ds["z"].values, np.float64)
    rng = np.random.default_rng(seed)
    cols = {v: [] for v in COL_VARS}
    kinds = []
    for i in range(0, nt, FRAME_STEP):
        f = ds.isel(t=i)
        a = {v: np.asarray(f[v].values, np.float64) for v in COL_VARS}   # (x, y, z)
        path = (a["q_c"] * a["rho"]).sum(-1) * (out["z_node"][1] - out["z_node"][0])
        pick_r = rng.choice(path.size, N_RANDOM, replace=False)
        pick_c = np.argsort(path.ravel())[::-1][:N_CLOUDY]
        for kind, pick in (("random", pick_r), ("cloudy", pick_c)):
            ix, iy = np.unravel_index(pick, path.shape)
            for v in COL_VARS:
                cols[v].append(a[v][ix, iy])
            kinds += [kind] * len(pick)
    out["cols"] = {v: np.concatenate(cols[v]) for v in COL_VARS}
    out["cols"]["kind"] = np.asarray(kinds)
    return out


def swirl_heating(flux_net, p):
    """swirl-lm two_stream.compute_heating_rate (interior nodes) [K/s]."""
    return SWIRL_G * (flux_net[2:] - flux_net[1:-1]) / (0.5 * (p[2:] - p[:-2])) / SWIRL_CP


def heating_from_netdown(net_down, p_half):
    """Layer heating [K/s] from net downward flux on top-first interfaces."""
    return constants.g / constants.c_pd * (net_down[:-1] - net_down[1:]) / np.diff(p_half)


def layer_err(ours, ref, p_half):
    dp = np.diff(p_half)
    d = 86400.0 * (ours - ref)
    return {"max_abs_K_day": float(np.max(np.abs(d))),
            "rms_K_day": float(np.sqrt((dp * d ** 2).sum() / dp.sum())),
            "k_max_topfirst": int(np.argmax(np.abs(d)))}


def onto(z_int_topfirst, net_topfirst, z):
    """Net flux of a top-first interface profile at heights ``z`` (inside it)."""
    zi, f = z_int_topfirst[::-1], net_topfirst[::-1]
    z = np.asarray(z, dtype=np.float64)
    if np.any(z < zi[0] - 1e-6) or np.any(z > zi[-1] + 1e-6):
        raise ValueError("height outside the profile's interfaces")
    return np.interp(z, zi, f)


def fluxes_fn(case, **kw):
    """Compiled our-radiation call: (T, qv, qc, pf, ph, rho) top-first -> net down
    on all column interfaces, OLR and absorbed SW at the extension top."""
    def f(T, qv, qc, pf, ph, rho):
        out, n_ext = column_fluxes(case, T, qv, qc, pf, ph, rho, **kw)
        lw_up, lw_dn, sw_up, sw_dn = (o[0] for o in (out.lw_flux_up, out.lw_flux_down,
                                                     out.sw_flux_up, out.sw_flux_down))
        net = (lw_dn + sw_dn - lw_up - sw_up)[n_ext:]
        return net, lw_up[0], sw_dn[0] - sw_up[0]
    return jax.jit(f)


def case_report(c):
    art = load_artifact(ART / f"cloudbench_{c}_amip.npz")
    par = art.les_setup["parameters"]
    snd = read_sounding(FORCING / str(int(c[1:4])) / str(int(c[6:8])) / par["experiment"]
                        / "sounding.csv")
    site = parse_case_name(art.case_name)[0]
    les = les_store(STORE_URL.format(site=site, month=int(par["month"]), exp=par["experiment"]),
                    seed=site * 100 + int(par["month"]))
    zn = les["z_node"]
    dz = zn[1] - zn[0]
    z_art = np.asarray(art.heights_m)
    if np.max(np.abs(z_art - (zn + 0.5 * dz)[1:-1])) > 1e-3:
        raise ValueError("artifact centres are not the store's nodes 1..nz-2 + dz/2")
    out = {}

    # --- LES grid: nodes 0 .. nz-2, faces 0 .. nz-1 (surface-first) ---
    m = {v: les[v].mean(0) for v in MEAN_VARS + ("q_ice",) + PROFILE_VARS}
    n = zn.size - 1
    z_face = zn.copy()
    p_node = m["p_ref"]
    lnp = np.log(p_node)
    p_face = np.exp(np.concatenate([[1.5 * lnp[0] - 0.5 * lnp[1]],
                                    0.5 * (lnp[1:] + lnp[:-1])]))
    F_les = -(m["rad_flux_lw"] + m["rad_flux_sw"])                         # net DOWN at faces
    stored = m["rad_heat_src"]
    rebuilt = np.stack([swirl_heating(les["rad_flux_lw"][i] + les["rad_flux_sw"][i],
                                      les["p_ref"][i]) for i in range(les["p_ref"].shape[0])]).mean(0)
    qc_m, T_m = m["q_c"][:n], m["T"][:n]
    dp_node = -np.diff(p_face)
    out["instrument"] = {
        "rebuilt_vs_stored_max_K_day": float(86400 * np.max(np.abs(rebuilt - stored[1:-1]))),
        "stored_min_K_day": float(86400 * stored.min()),
        "p_ref_minus_sounding_max_Pa": float(np.max(np.abs(p_face - _p_of_z(snd, z_face)))),
        "ice_share_of_condensate_path": float((m["q_ice"][:n] * dp_node).sum()
                                              / max((qc_m * dp_node).sum(), 1e-30)),
        "les_cover_max": float(m["cloud_fraction"].max()),
        "les_lwp_g_m2": float(1e3 * les["lwp"].mean())}
    out["les"] = {k: float(les[k].mean()) for k in SCALAR_VARS}
    wet = np.nonzero(qc_m > CLOUD_QC)[0]
    z_ct = float(z_face[wet[-1] + 1]) if wet.size else None
    out["cloud_top_m"] = z_ct
    z_top = float(z_face[-1])

    def gate(z_int, net, olr, asr, ref_z, ref_f):
        zt = min(z_top, float(z_int[0]))                  # the lower of the two tops
        g = {"d_top_W_m2": float(onto(z_int, net, [zt])[0] - np.interp(zt, ref_z, ref_f)),
             "d_sfc_W_m2": float(onto(z_int, net, [0.0])[0] - ref_f[0]),
             # different extension tops (ours: the GCM sounding's): recorded, not gated
             "olr_ours_W_m2": float(olr), "asr_ours_W_m2": float(asr)}
        if z_ct is not None:
            g["d_cloudtop_W_m2"] = float(onto(z_int, net, [z_ct])[0]
                                         - np.interp(z_ct, ref_z, ref_f))
        return g

    ph_t = p_face[::-1]
    pf_t = p_node[:n][::-1]
    zf_t = z_face[::-1]
    case_l = case_radiation(par, snd, float(ph_t[0]))
    H_les_fine = heating_from_netdown(F_les[::-1], ph_t)

    grid, _ = cloudbench_grid(snd, float(par["p_sfc"]), NLEV)
    p_half = np.asarray(grid.p_half_pa)
    pf_s = np.asarray(grid.p_full_pa)
    z_half = np.interp(-np.log(p_half), -np.log(np.asarray(snd["p"], np.float64)),
                       np.asarray(snd["z"], np.float64))
    z_half[-1] = 0.0
    if z_half[0] > z_top + 1e-6:
        raise ValueError(f"our column top {z_half[0]:.1f} m is above the LES top {z_top:.1f} m")
    out["column_top_m"] = float(z_half[0])
    F_a = np.interp(z_half, z_face, F_les)
    H_a = heating_from_netdown(F_a, p_half)
    out["a_les_heating_min_K_day"] = float(86400 * H_a.min())
    out["les_fine_heating_min_K_day"] = float(86400 * H_les_fine.min())

    # --- S: column by column on sampled LES columns ---
    cols = les["cols"]
    qv_c = cols["q_t"] - cols["q_c"]                     # swirl-lm's vapour (q_t - q_c)
    fS = fluxes_fn(case_l)
    pf_j, ph_j = jnp.asarray(pf_t), jnp.asarray(ph_t)
    vS = jax.jit(jax.vmap(lambda T, qv, qc, rho: fS(T, qv, qc, pf_j, ph_j, rho)))
    xs = [x[:, :n][:, ::-1] for x in (cols["T"], qv_c, cols["q_c"], cols["rho"])]
    if xs[0].shape[0] % COL_CHUNK:
        raise ValueError("sampled columns must fill whole chunks")
    parts = [vS(*(jnp.asarray(x[i:i + COL_CHUNK]) for x in xs))
             for i in range(0, xs[0].shape[0], COL_CHUNK)]
    net_c, olr_c, asr_c = (np.concatenate([np.asarray(pt[j]) for pt in parts]) for j in range(3))
    F_cols_les = -(cols["rad_flux_lw"] + cols["rad_flux_sw"])           # (ncol, nz) faces
    for kind in ("random", "cloudy"):
        sel = cols["kind"] == kind
        ours = net_c[sel].mean(0)
        ref = F_cols_les[sel].mean(0)
        g = gate(zf_t, ours, olr_c[sel].mean(), asr_c[sel].mean(), z_face, ref)
        del g["olr_ours_W_m2"], g["asr_ours_W_m2"]        # LES olr/asr are plane means
        out[f"S_{kind}"] = {
            **g, "fine": layer_err(heating_from_netdown(ours, ph_t),
                                   heating_from_netdown(ref[::-1], ph_t), ph_t),
            "scm_layers": layer_err(heating_from_netdown(onto(zf_t, ours, z_half), p_half),
                                    heating_from_netdown(np.interp(z_half, z_face, ref), p_half),
                                    p_half),
            "flux_rms_W_m2": float(np.sqrt(np.mean((ours - ref[::-1]) ** 2))),
            "n_columns": int(sel.sum())}
        # the same cohort's mean state, radiated: ours(mean state) - mean(ours per
        # column) is the averaging error on identical samples and weights
        cm = [jnp.asarray(x[sel][:, :n].mean(0)[::-1]) for x in
              (cols["T"], qv_c, cols["q_c"], cols["rho"])]
        cover = jnp.asarray((cols["q_c"][sel][:, :n] > CLOUD_QC).mean(0)[::-1])
        for vname, kw in (("uniform", {}), ("cohort_cover", {"cloud_fraction": cover})):
            net_m = np.asarray(fluxes_fn(case_l, **kw)(cm[0], cm[1], cm[2], pf_j, ph_j, cm[3])[0])
            gm = gate(zf_t, net_m, 0.0, 0.0, zf_t[::-1], ours[::-1])
            del gm["olr_ours_W_m2"], gm["asr_ours_W_m2"]
            out[f"S_{kind}"][f"averaging_{vname}"] = {
                **gm, "fine": layer_err(heating_from_netdown(net_m, ph_t),
                                        heating_from_netdown(ours, ph_t), ph_t)}

    # --- A: mean state on the LES grid ---
    a_args = tuple(jnp.asarray(x) for x in (T_m[::-1], (m["q_t"] - m["q_c"])[:n][::-1],
                                            qc_m[::-1], pf_t, ph_t, m["rho"][:n][::-1]))
    cf_t = jnp.asarray(m["cloud_fraction"][:n][::-1])
    for name, kw in (("uniform", {}), ("clear", {"clouds": False}),
                     ("les_cover", {"cloud_fraction": cf_t})):
        net, olr, asr = (np.asarray(a) for a in fluxes_fn(case_l, **kw)(*a_args))
        out[f"A_{name}"] = {
            **gate(zf_t, net, olr, asr, z_face, F_les),
            "fine": layer_err(heating_from_netdown(net, ph_t), H_les_fine, ph_t),
            "scm_layers": layer_err(heating_from_netdown(onto(zf_t, net, z_half), p_half),
                                    H_a, p_half)}

    # --- R: our 32 layers, state as the replay builds it ---
    th, qt, qc, qr, cf, _ = day5_state(art)
    zh = np.concatenate([[0.0], 0.5 * (z_art[1:] + z_art[:-1]), [z_art[-1] + 0.5 * dz]])
    if zh[-1] > float(np.max(snd["z"])):
        raise ValueError("the LES top is above the GCM sounding's top")
    ph, pfl = _p_of_z(snd, zh), _p_of_z(snd, z_art)
    ths, qts, qcs, qrs, cfs = (layer_mean(a, ph, pfl, p_half) for a in (th, qt, qc, qr, cf))
    Ts, qvs, rhos = thermo(ths, qts, qcs, qrs, pf_s)
    case_s = case_radiation(par, snd, float(p_half[0]))
    r_args = tuple(jnp.asarray(x) for x in (Ts, qvs, qcs, pf_s, p_half, rhos))
    for name, kw in (("uniform", {}), ("clear", {"clouds": False}),
                     ("les_cover", {"cloud_fraction": jnp.asarray(cfs)})):
        net, olr, asr = (np.asarray(a) for a in fluxes_fn(case_s, **kw)(*r_args))
        out[f"R_{name}"] = {**gate(z_half, net, olr, asr, z_face, F_les),
                            **layer_err(heating_from_netdown(net, p_half), H_a, p_half)}

    # --- R on the LES's own pressure: same heights, p from its p_ref (ln p in z) ---
    lnp_at = lambda z_, zz, pp: np.exp(np.interp(z_, zz, np.log(pp)))   # noqa: E731
    p_half_l = lnp_at(z_half, z_face, p_face)
    z_full = np.interp(-np.log(pf_s), -np.log(np.asarray(snd["p"], np.float64)),
                       np.asarray(snd["z"], np.float64))
    pf_l = lnp_at(z_full, zn + 0.5 * dz, p_node)        # same full-level heights
    ph_al, pfl_al = lnp_at(zh, z_face, p_face), lnp_at(z_art, zn + 0.5 * dz, p_node)
    ths, qts, qcs, qrs, cfs = (layer_mean(a, ph_al, pfl_al, p_half_l) for a in (th, qt, qc, qr, cf))
    Tl, qvl, rhol = thermo(ths, qts, qcs, qrs, pf_l)
    case_sl = case_radiation(par, snd, float(p_half_l[0]))
    H_al = heating_from_netdown(F_a, p_half_l)
    l_args = tuple(jnp.asarray(x) for x in (Tl, qvl, qcs, pf_l, p_half_l, rhol))
    for name, kw in (("uniform", {}), ("clear", {"clouds": False}),
                     ("les_cover", {"cloud_fraction": jnp.asarray(cfs)})):
        net, olr, asr = (np.asarray(a) for a in fluxes_fn(case_sl, **kw)(*l_args))
        out[f"R_lesp_{name}"] = {**gate(z_half, net, olr, asr, z_face, F_les),
                                 **layer_err(heating_from_netdown(net, p_half_l), H_al, p_half_l)}
    out["R_lesp_column_mass_change"] = float((p_half_l[-1] - p_half_l[0])
                                            / (p_half[-1] - p_half[0]) - 1.0)
    return out


def main() -> int:
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    res = {}
    for c in sys.argv[2:]:
        res[c] = case_report(c)
        print(c, json.dumps(res[c]["instrument"]), flush=True)
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
