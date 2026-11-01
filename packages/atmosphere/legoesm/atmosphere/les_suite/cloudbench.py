"""CloudBench LES (swirl-lm) as LES-suite reference truth (PI 2026-10-31).

CloudBench's full 3-D output stays online (public bucket, one zarr store per
site/month/experiment; day 5 only, every 20 min). :func:`build_artifact` streams one
case and keeps only its horizontal-mean profiles and fluxes as an
:class:`~.bridge.LESReferenceArtifact`.

What is rebuilt, because CloudBench never wrote it (its ``*_diffusive_flux_z`` and
surface fluxes are zeros): the subgrid (SGS) heat and moisture flux, from the stored
state with swirl-lm's own Smagorinsky-Lilly closure as configured for CloudBench
(swirl-lm 98f96f80: ``physics/turbulence/sgs_model.py``, ``numerics/diffusion.py``,
``physics/thermodynamics/water.py``; config c_s 0.18, Pr_t 0.3, molecular 0.1 m2/s,
``use_pr_t``). Each frame records a budget check of that rebuild against the LES's
stored ``*_les_tendency`` (:func:`budget_check`); it is recorded, not enforced
(PI 2026-10-31).

Grid (CONFIRMED in swirl-lm ``monin_obukhov_similarity_theory.py`` for a uniform
grid): collocated nodes, ``dz = L/(nz-1)``, the ground at the face between the
first fluid node and the halo, so node ``k`` sits ``(k + 1/2)·dz`` above the ground
while the store's ``z`` is ``k·dz``. Below-ground and above-top fills follow the run's
boundary conditions: u, v and scalars copy the edge node (zero-gradient), w is zero.
Stored levels are nodes 1 .. nz-2 (PI 2026-11-01): both flux faces of each are
inside the fluid.

Conversions (PI 2026-10-31):
- θ_l is OURS (:func:`~.scm_coupling.liquid_water_theta`), built cell by cell from
  the stored temperature, cloud water and reference pressure, so it is what the SCM
  scores against; swirl-lm's θ_li differs slightly (moist Exner exponent).
- Resolved fluxes are horizontal-mean covariances ``<w'φ'>`` at nodes, as in every
  other suite artifact.
- SGS fluxes: the rebuilt diffusivity applied to OUR θ_l and to q_t, kinematic per
  cell, averaged from the two neighbouring faces to the node (the suite's face to
  centre convention). KNOWN DIFFERENCE: the LES diffused its own θ_li, not θ_l.
- Frame 0 is the starting state from the GCM sounding, interpolated linearly in
  height (as swirl-lm's ``interpolate_sounding``), condensed with OUR saturation
  curve; its fluxes are zero and its radiative heating and cloud fraction are NaN
  (never scored).
- Momentum fluxes are not stored: CloudBench did not save its SGS momentum flux
  (u, v profiles only are scored, PI 2026-10-31).
- ``rad_T_tendency`` is the horizontal mean of the stored ``rad_heat_src``, a
  temperature tendency [K/s] (swirl-lm ``two_stream.py``; units PLAUSIBLE, not
  stored in the data).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.thermo import saturation_mixing_ratio

from .bridge import LESReferenceArtifact
from .scm_coupling import liquid_water_theta

STORE_URL = "https://storage.googleapis.com/cloudbench-simulation-output/{site}/{month}/{exp}/data.zarr"
SWIRL_LM_COMMIT = "98f96f801d71b9e303f22c51e615f41b29ea64c4"

# --- swirl-lm CloudBench configuration (values of that model, reproduced) ---
_CB_DOMAIN_M = 6000.0           # length {dim_0, dim_1, dim_2} in the run's pbtxt
_CB_C_S = 0.18                  # sgs_model.smagorinsky_lilly.c_s
_CB_PR_T = 0.3                  # sgs_model.smagorinsky_lilly.pr_t
_CB_D_MOL = 0.1                 # scalars {theta_li, q_t} diffusivity [m2/s]
_CB_G = 9.81                    # swirl-lm common constants.G
_CB_R_D = 286.69                # swirl-lm water.py _R_D
_CB_R_V = 461.5                 # the run's thermodynamics.water.r_v (proto default 461.89)
_CB_CP_D = 728.31 + 286.69      # cv_d + R_d (water.py cp_d)
_CB_CP_V = 1859.0               # thermodynamics.proto water.cp_v default
_CB_P00 = 1.0e5                 # thermodynamics.proto p00 default
_BUDGET_TOP_M = 1000.0          # lowest-km band of the recorded budget check

CASE_NAME = "cloudbench_s{site:03d}_m{month:02d}_{exp}"
SGS_NAME = "smagorinsky"
_FIELDS = ("u", "v", "w", "theta_li", "q_t", "q_c", "q_r", "T", "rho", "p_ref",
           "rad_heat_src", "theta_li_les_tendency", "q_t_les_tendency")


def case_name(site: int, month: int, exp: str) -> str:
    return CASE_NAME.format(site=int(site), month=int(month), exp=exp)


def _pad_z(f, bc):
    """One below-ground and one above-top plane per the run's z boundary conditions:
    ``"neumann"`` copies the edge node (swirl-lm homogeneous Neumann, order 1),
    ``"zero"`` is the Dirichlet value 0 (w)."""
    lo, hi = f[..., :1], f[..., -1:]
    if bc == "zero":
        lo, hi = np.zeros_like(lo), np.zeros_like(hi)
    elif bc != "neumann":
        raise ValueError(f"unknown z boundary {bc!r}")
    return np.concatenate([lo, f, hi], axis=-1)


def _ddz(f, dz, bc):
    p = _pad_z(f, bc)
    return (p[..., 2:] - p[..., :-2]) / (2.0 * dz)


def _ddh(f, d, axis):
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2.0 * d)


def smagorinsky_lilly_rhok(u, v, w, T, q_t, q_c, p_ref, rho, dx, dy, dz):
    """ρK at every node, as swirl-lm computes it for CloudBench: K = (c_s²/Pr_t)
    (Δ f_b)² |S| + D_mol, Δ = (dx dy dz)^(1/3), |S| = sqrt(2 S:S) of the trace-free
    strain from centred node differences, f_b = 1 for Ri <= 0 else
    max(0, 1 - Ri/Pr_t)^(1/4), Ri = (g/θ_v) ∂θ_v/∂z / |S|² (0 where |S| = 0)."""
    vel = (u, v, w)
    bcs = ("neumann", "neumann", "zero")
    g = [[_ddh(c, dx, 0), _ddh(c, dy, 1), _ddz(c, dz, bc)] for c, bc in zip(vel, bcs)]
    div = g[0][0] + g[1][1] + g[2][2]
    s2 = 0.0
    for i in range(3):
        for j in range(3):
            sij = 0.5 * (g[i][j] + g[j][i]) - (div / 3.0 if i == j else 0.0)
            s2 = s2 + sij ** 2
    smag = np.sqrt(2.0 * s2)
    eps = _CB_R_V / _CB_R_D
    r_m = _CB_R_D * (1.0 + (eps - 1.0) * q_t - eps * q_c)
    cp_m = (1.0 - q_t) * _CB_CP_D + (q_t - q_c) * _CB_CP_V
    theta_v = (r_m / _CB_R_D) * T * (p_ref / _CB_P00) ** (-r_m / cp_m)
    n2 = _CB_G / theta_v * _ddz(theta_v, dz, "neumann")
    safe = np.where(smag > 0.0, smag, 1.0)
    ri = np.where(smag > 0.0, n2 / safe ** 2, 0.0)
    fb = np.where(ri <= 0.0, 1.0, np.maximum(0.0, 1.0 - ri / _CB_PR_T) ** 0.25)
    delta = (dx * dy * dz) ** (1.0 / 3.0)
    return rho * ((_CB_C_S ** 2 / _CB_PR_T) * (delta * fb) ** 2 * smag + _CB_D_MOL)


def sgs_flux_nodes(rhok, rho, phi, dz):
    """Horizontal-mean upward kinematic SGS flux of ``phi`` at nodes 1 .. nz-2 (NaN
    at nodes 0 and nz-1): per cell at face k-1/2, -(½(ρK_k+ρK_{k-1}))(φ_k-φ_{k-1})/dz
    / ½(ρ_k+ρ_{k-1}) (swirl-lm CENTRAL_3 face flux, made kinematic), then the mean of
    the node's two faces."""
    rk = 0.5 * (rhok[..., 1:] + rhok[..., :-1])
    rf = 0.5 * (rho[..., 1:] + rho[..., :-1])
    face = (-rk * np.diff(phi, axis=-1) / dz / rf).mean(axis=(0, 1))   # faces 1/2..
    out = np.full(phi.shape[-1], np.nan)
    out[1:-1] = 0.5 * (face[:-1] + face[1:])
    return out


def quick_face_flux(m, phi):
    """swirl-lm's vertical scalar flux at face k-1/2 (index k): QUICK face values
    from the upwind side, upwinded by the face mass flux ½(m_{k-1}+m_k), Rhie-Chow
    off (``numerics/convection.py`` flux_upwinding; ``get_kernel_fn.py`` kf2). NaN
    where the stencil leaves the grid (k < 2 or k > nz-2)."""
    nz = phi.shape[-1]
    f = np.full(phi.shape, np.nan)
    k = np.arange(2, nz - 1)
    m_f = 0.5 * (m[..., k - 1] + m[..., k])
    s_pos = -0.125 * phi[..., k - 2] + 0.75 * phi[..., k - 1] + 0.375 * phi[..., k]
    s_neg = 0.375 * phi[..., k - 1] + 0.75 * phi[..., k] - 0.125 * phi[..., k + 1]
    f[..., k] = np.maximum(m_f, 0.0) * s_pos + np.minimum(m_f, 0.0) * s_neg
    return f


def budget_check(rhok, rho, w, phi, tend, dz, z_phys):
    """The LES's stored horizontal-mean tendency of ``phi`` (density-weighted)
    against minus the vertical divergence of its own advective flux plus the
    rebuilt SGS diffusion, over the nodes from 2 to the lowest-km top. Returns the
    rms residual with and without the rebuilt SGS term (the latter is the SGS
    divergence the LES needed) and their ratio."""
    nz = phi.shape[-1]
    q = quick_face_flux(rho * w, phi).mean(axis=(0, 1))
    d_adv = np.full(nz, np.nan)
    d_adv[:-1] = (q[1:] - q[:-1]) / dz
    rk = 0.5 * (rhok[..., 1:] + rhok[..., :-1])
    fd = (rk * np.diff(phi, axis=-1) / dz).mean(axis=(0, 1))          # faces 1/2..
    d_sgs = np.full(nz, np.nan)
    d_sgs[1:-1] = (fd[1:] - fd[:-1]) / dz
    t = tend.mean(axis=(0, 1))
    band = np.nonzero((np.arange(nz) >= 2) & (z_phys <= _BUDGET_TOP_M))[0]
    without = float(np.sqrt(np.mean((t + d_adv)[band] ** 2)))
    with_sgs = float(np.sqrt(np.mean((t + d_adv - d_sgs)[band] ** 2)))
    return {"rms_residual": with_sgs, "rms_residual_without_sgs": without,
            "ratio": with_sgs / without if without > 0.0 else float("nan")}


def frame_profiles(s, dx, dy, dz, z_phys):
    """Horizontal-mean profiles at all nodes of one stored frame ``s`` (3-D fields
    ``(x, y, z)`` in float64) plus its budget check."""
    exner = np.asarray(exner_function(s["p_ref"]))
    theta_l = np.asarray(liquid_water_theta(s["T"] / exner, s["q_c"], exner))
    rhok = smagorinsky_lilly_rhok(s["u"], s["v"], s["w"], s["T"], s["q_t"], s["q_c"],
                                  s["p_ref"], s["rho"], dx, dy, dz)

    def mean(f):
        return f.mean(axis=(0, 1))

    def cov(a):
        return mean(s["w"] * (a - mean(a)))

    prof = {"theta": mean(theta_l), "qt": mean(s["q_t"]), "qc": mean(s["q_c"]),
            "qr": mean(s["q_r"]), "u": mean(s["u"]), "v": mean(s["v"]),
            "wtheta_resolved": cov(theta_l), "wqt_resolved": cov(s["q_t"]),
            "wtheta_sgs": sgs_flux_nodes(rhok, s["rho"], theta_l, dz),
            "wqt_sgs": sgs_flux_nodes(rhok, s["rho"], s["q_t"], dz),
            "rad_T_tendency": mean(s["rad_heat_src"])}
    check = {name: budget_check(rhok, s["rho"], s["w"], s[name],
                                s[name + "_les_tendency"], dz, z_phys)
             for name in ("theta_li", "q_t")}
    return prof, check


def run_surface_settings(pbtxt: str) -> dict:
    """The surface roughness ``z_0`` [m] and ``surface_gustiness`` [m/s] the run's
    swirl-lm config (``_swirl_lm.pbtxt`` in the store attributes) set; each must
    appear exactly once."""
    out = {}
    for key in ("z_0", "surface_gustiness"):
        vals = re.findall(rf"^\s*{key}:\s*([-+0-9.eE]+)\s*$", pbtxt, flags=re.M)
        if len(vals) != 1:
            raise ValueError(f"run config sets {key} {len(vals)} times, expected once")
        out[key] = float(vals[0])
    return out


def read_sounding(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0]}


def initial_profiles(sounding, z_phys):
    """Frame 0 from the GCM sounding: T, q_t, u, v and p interpolated linearly in
    height onto the nodes (swirl-lm ``interpolate_sounding``: ``np.interp``, clamped);
    cloud water from OUR saturation curve, q_c = max(q_t - q_sat(T, p), 0)."""
    if not np.array_equal(sounding["T"], sounding["temperature"]):
        raise ValueError("sounding columns T and temperature differ; which one the "
                         "LES started from is undetermined")
    zs = sounding["z"]
    if not np.all(np.diff(zs) > 0.0):
        raise ValueError("sounding heights must increase")
    T, q_t, p = (np.interp(z_phys, zs, sounding[k]) for k in ("T", "q_t", "p"))
    q_c = np.maximum(q_t - np.asarray(saturation_mixing_ratio(T, p)), 0.0)
    exner = np.asarray(exner_function(p))
    return {"theta": np.asarray(liquid_water_theta(T / exner, q_c, exner)), "qt": q_t,
            "qc": q_c, "qr": np.zeros_like(q_t), "u": np.interp(z_phys, zs, sounding["u"]),
            "v": np.interp(z_phys, zs, sounding["v"])}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artifact(site, month, exp, forcing_dir, store=None, time_indices=None):
    """Stream one CloudBench case and return its artifact. ``store`` defaults to the
    public bucket URL (any zarr path works, e.g. a test store); ``forcing_dir`` holds
    ``<site>/<month>/<exp>/{sounding.csv,parameters.json}``; ``time_indices`` selects
    stored output times (default all)."""
    import xarray as xr  # noqa: PLC0415

    url = store or STORE_URL.format(site=site, month=month, exp=exp)
    fdir = Path(forcing_dir) / str(site) / str(month) / exp
    params = json.loads((fdir / "parameters.json").read_text())
    if (int(params["month"]), params["experiment"]) != (int(month), exp):
        raise ValueError(f"{fdir}: parameters.json is for month {params['month']} "
                         f"experiment {params['experiment']}")
    sounding = read_sounding(fdir / "sounding.csv")
    ds = xr.open_zarr(url, consolidated=True)
    for name in _FIELDS:
        if tuple(ds[name].dims) != ("t", "x", "y", "z"):
            raise ValueError(f"{url}: {name} dims {ds[name].dims}, expected (t, x, y, z)")
    nx, ny, nz = ds.sizes["x"], ds.sizes["y"], ds.sizes["z"]
    dx, dy, dz = (_CB_DOMAIN_M / (n - 1) for n in (nx, ny, nz))
    for ax, n, d in (("x", nx, dx), ("y", ny, dy), ("z", nz, dz)):
        c = np.asarray(ds[ax].values, dtype=np.float64)
        if not np.allclose(c, np.arange(n) * d, rtol=0.0, atol=1e-3 * d):
            raise ValueError(f"{url}: {ax} is not k*L/(n-1); the grid differs from "
                             "CloudBench's")
    z_store = np.asarray(ds["z"].values, dtype=np.float64)
    if not np.issubdtype(ds["t"].dtype, np.timedelta64):
        raise ValueError(f"{url}: t has dtype {ds['t'].dtype}, expected elapsed time")
    surface = run_surface_settings(ds.attrs["_swirl_lm.pbtxt"])
    z_phys = z_store + 0.5 * dz
    t_all = ds["t"].values.astype("timedelta64[ns]").astype(np.int64) / 1e9
    idx = list(range(len(t_all))) if time_indices is None else list(time_indices)
    keep = slice(1, nz - 1)

    init = initial_profiles(sounding, z_phys)
    series = {k: [v] for k, v in init.items()}
    for k in ("wtheta_resolved", "wqt_resolved", "wtheta_sgs", "wqt_sgs"):
        series[k] = [np.zeros(nz)]
    series["rad_T_tendency"] = [np.full(nz, np.nan)]
    series["cloud_frac"] = [np.full(nz, np.nan)]
    checks = []
    cloud = ds["cloud_fraction"]
    if tuple(cloud.dims) != ("t", "z"):
        raise ValueError(f"{url}: cloud_fraction dims {cloud.dims}, expected (t, z)")
    for ti in idx:
        s = {n: np.asarray(ds[n].isel(t=ti).values, dtype=np.float64) for n in _FIELDS}
        bad = [n for n, f in s.items() if not np.all(np.isfinite(f))]
        if bad:
            raise ValueError(f"{url}: time index {ti} has non-finite {bad}")
        prof, check = frame_profiles(s, dx, dy, dz, z_phys)
        for k, v in prof.items():
            series[k].append(v)
        series["cloud_frac"].append(np.asarray(cloud.isel(t=ti).values, np.float64))
        checks.append({"time_s": float(t_all[ti]), **check})
    arr = {k: np.stack(v)[:, keep] for k, v in series.items()}
    times = np.concatenate([[0.0], t_all[idx]])
    sst = float(params["sst"])
    setup = {
        "source": "CloudBench (data.caltech.edu hgpq9-a1861, Apache-2.0)",
        "store": url, "swirl_lm_commit": SWIRL_LM_COMMIT,
        "sounding_sha256": _sha256(fdir / "sounding.csv"),
        "parameters": params, "grid": [nx, ny, nz], "dz_m": dz, "dx_m": dx,
        "height_offset_m": 0.5 * dz, "levels": "nodes 1..nz-2",
        "z0_m": surface["z_0"], "surface_gustiness_m_s": surface["surface_gustiness"],
        "sgs": {"model": "smagorinsky_lilly", "c_s": _CB_C_S, "pr_t": _CB_PR_T,
                "d_mol_m2_s": _CB_D_MOL, "rebuilt": True},
        "budget_check": checks,
        "frame0": "GCM sounding, linear in height, our saturation; zero fluxes; "
                  "NaN radiation and cloud fraction",
        "known_differences": [
            "SGS heat flux uses the rebuilt K on our theta_l; the LES diffused theta_li",
            "no momentum fluxes: CloudBench did not store its SGS momentum flux",
            "rad_T_tendency units K/s inferred from swirl-lm two_stream.py, not stored",
        ],
    }
    return LESReferenceArtifact(
        case_name=case_name(site, month, exp), sgs=SGS_NAME,
        heights_m=z_phys[keep], times_s=times, theta=arr["theta"], u=arr["u"],
        v=arr["v"], wtheta_resolved=arr["wtheta_resolved"], wtheta_sgs=arr["wtheta_sgs"],
        qt=arr["qt"], wqt_resolved=arr["wqt_resolved"], wqt_sgs=arr["wqt_sgs"],
        qc=arr["qc"], qr=arr["qr"], cloud_frac=arr["cloud_frac"],
        rad_T_tendency=arr["rad_T_tendency"], prescribe="T_s",
        T_s=np.full(times.shape, sst), les_setup=setup)
