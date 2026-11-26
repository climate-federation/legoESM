"""Read the CLOUDSC 49r1 oracle's text inputs/outputs into the JAX port's types.

Text format (export_h5.py / driver49.F90): first line ``rank ext...``, then the
values in Fortran (column-major) order.  Inputs are read exactly as driver49
reads them (same file, same derived arrays), so the port sees bit-identical
numbers:
  * TENDENCY_TMP_* fed as the cumulative tendencies (vote V6a);
  * constants: the file's YRCST/YRTHF values (vote V7a); RA, RPI from the 49r1
    setup (printed by the driver, not in the file);
  * PLUDE/PLUDELI: what the driver wrote (``PLUDE_IN``/``PLUDELI_IN``) for the
    arm (base: zero, vote V3c; detr: vote V18a);
  * PSNDE slot 1 (rain) = 0, slot 2 (snow) = file PSNDE; PEIS = 0;
  * settings: the driver's dump of the 49r1 YDECLDP (settings_49r1.txt).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RA_49R1 = 6.37122900000000000e06     # const-ok: the 49r1 oracle's own RA (driver log, SETUP_CONSTANTS), not constants.R_earth
RPI_49R1 = 3.14159265358979312e00

_THF = ("R2ES", "R3LES", "R3IES", "R4LES", "R4IES", "R5LES", "R5IES", "R5ALVCP",
        "R5ALSCP", "RALVDCP", "RALSDCP", "RALFDCP", "RTWAT", "RTICE", "RTICECU",
        "RTWAT_RTICE_R", "RTWAT_RTICECU_R", "RKOOP1", "RKOOP2")
_CST = ("RG", "RD", "RV", "RCPD", "RETV", "RLVTT", "RLSTT", "RLMLT", "RTT")

# IFSCloudscConfig field -> YDECLDP name in the settings dump
SETTINGS = {
    "ramid": "RAMID", "rcldiff": "RCLDIFF", "rcldiff_convi": "RCLDIFF_CONVI",
    "ramin": "RAMIN", "rlmin": "RLMIN", "rssicefactor": "RSSICEFACTOR",
    "rclcrit_sea": "RCLCRIT_SEA", "rclcrit_land": "RCLCRIT_LAND",
    "rcl_kkaac": "RCL_KKAAC", "rcl_kkbac": "RCL_KKBAC", "rcl_kkaau": "RCL_KKAAU",
    "rcl_kkbauq": "RCL_KKBAUQ", "rcl_kkbaun": "RCL_KKBAUN",
    "rcl_kk_cloud_num_sea": "RCL_KK_CLOUD_NUM_SEA", "rcl_kk_cloud_num_land": "RCL_KK_CLOUD_NUM_LAND",
    "rcl_inhomogaut": "RCL_INHOMOGAUT", "rcl_inhomogacc": "RCL_INHOMOGACC",
    "rcovpmin": "RCOVPMIN", "rprecrhmax": "RPRECRHMAX", "rpecons": "RPECONS",
    "rvrfactor": "RVRFACTOR", "rcl_ka273": "RCL_KA273", "rcl_cdenom1": "RCL_CDENOM1",
    "rcl_cdenom2": "RCL_CDENOM2", "rcl_cdenom3": "RCL_CDENOM3",
    "rlcritsnow": "RLCRITSNOW", "rsnowlin1": "RSNOWLIN1", "rsnowlin2": "RSNOWLIN2",
    "rcl_overlapliqice": "RCL_OVERLAPLIQICE", "ricinit": "RICEINIT", "rcldtopcf": "RCLDTOPCF",
    "rdepliqrefrate": "RDEPLIQREFRATE", "rdepliqrefdepth": "RDEPLIQREFDEPTH",
    "rthomo": "RTHOMO", "rtaumel": "RTAUMEL", "rcl_fzrab": "RCL_FZRAB",
    "rcl_apb1": "RCL_APB1", "rcl_apb2": "RCL_APB2", "rcl_apb3": "RCL_APB3",
    "rcl_effrime": "RCL_EFFRIME", "rvice": "RVICE", "rvrain": "RVRAIN", "rvsnow": "RVSNOW",
    "rxfallspeedrain": "RXFALLSPEEDRAIN", "rdensref": "RDENSREF", "rdenswat": "RDENSWAT",
    "ncldtop": "NCLDTOP",
}
# module constants of ifs_cloudsc that must equal the dump
DERIVED = ("RCL_CONST1R", "RCL_CONST2R", "RCL_CONST3R", "RCL_CONST4R", "RCL_CONST5R",
           "RCL_CONST6R", "RCL_CONST7R", "RCL_LAM1R", "RCL_LAM2R", "RCL_DR", "RCL_X1R",
           "RCL_X2R", "RCL_CONST1S", "RCL_CONST2S", "RCL_CONST3S", "RCL_CONST4S",
           "RCL_CONST5S", "RCL_CONST6S", "RCL_CONST7S", "RCL_CONST8S", "RCL_LAM1S",
           "RCL_LAM2S", "RCL_X1S")


def read_txt(path) -> np.ndarray:
    tok = Path(path).read_text().split()
    rank = int(tok[0])
    ext = [int(t) for t in tok[1:1 + rank]]
    return np.array([float(t) for t in tok[1 + rank:]], dtype=np.float64).reshape(ext, order="F")


def read_settings(path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def raw_inputs(indir, outdir) -> dict:
    """Every float the oracle reads, as numpy arrays (the spread perturbs these)."""
    d = Path(indir)
    r = {n: read_txt(d / f"{n}.txt") for n in (
        "PT", "PQ", "PA", "PCLV", "TENDENCY_TMP_T", "TENDENCY_TMP_Q", "TENDENCY_TMP_A",
        "TENDENCY_TMP_CLD", "PVFL", "PVFI", "PAP", "PAPH", "PLSM", "PLU", "PMFU", "PMFD",
        "PSNDE", "PTSPHY")}
    r["PTSPHY"] = float(r["PTSPHY"].ravel()[0])
    o = Path(outdir)
    r["PLUDE"] = read_txt(o / "PLUDE_IN.txt")
    r["PLUDELI"] = read_txt(o / "PLUDELI_IN.txt")
    for n in ("LDCUM", "KTYPE"):
        r[n] = read_txt(d / f"{n}.txt")
    return r


def constants(indir):
    from legoesm.atmosphere.physics._ifs_thermo import IFSThermo
    d = Path(indir)
    v = {n: float(read_txt(d / f"{n}.txt").ravel()[0]) for n in _THF + _CST}
    return IFSThermo(
        RG=v["RG"], RD=v["RD"], RV=v["RV"], RCPD=v["RCPD"], RLVTT=v["RLVTT"],
        RLMLT=v["RLMLT"], RLSTT=v["RLSTT"], RTT=v["RTT"], RA=RA_49R1, RPI=RPI_49R1,
        RETV=v["RETV"], R2ES=v["R2ES"], R3LES=v["R3LES"], R4LES=v["R4LES"],
        R3IES=v["R3IES"], R4IES=v["R4IES"], R5LES=v["R5LES"], R5IES=v["R5IES"],
        R5ALVCP=v["R5ALVCP"], R5ALSCP=v["R5ALSCP"], RALVDCP=v["RALVDCP"],
        RALSDCP=v["RALSDCP"], RALFDCP=v["RALFDCP"], RTWAT=v["RTWAT"],
        RTBER=v["RTT"] - 5.0, RTBERCU=v["RTT"] - 5.0, RTICE=v["RTICE"],
        RTICECU=v["RTICECU"], RTWAT_RTICE_R=v["RTWAT_RTICE_R"],
        RTWAT_RTICECU_R=v["RTWAT_RTICECU_R"], RKOOP1=v["RKOOP1"], RKOOP2=v["RKOOP2"])


def config(settings_path):
    from legoesm.atmosphere.physics.microphysics.ifs_cloudsc import IFSCloudscConfig
    s = read_settings(settings_path)
    kw = {f: s[n] for f, n in SETTINGS.items()}
    kw["ncldtop"] = int(kw["ncldtop"])
    return IFSCloudscConfig(**kw), s


def to_inputs(r):
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.ifs_cloudsc import CloudscInputs
    ncol, nlev = r["PT"].shape
    psnde = np.zeros((ncol, nlev, 2))
    psnde[..., 1] = r["PSNDE"]          # driver49: slot 1 rain = 0, slot 2 snow = file
    f = lambda a: jnp.asarray(a, jnp.float64)
    return CloudscInputs(
        t=f(r["PT"]), q=f(r["PQ"]), a=f(r["PA"]), clv=f(r["PCLV"]),
        tend_t=f(r["TENDENCY_TMP_T"]), tend_q=f(r["TENDENCY_TMP_Q"]),
        tend_a=f(r["TENDENCY_TMP_A"]), tend_cld=f(r["TENDENCY_TMP_CLD"]),
        pvfl=f(r["PVFL"]), pvfi=f(r["PVFI"]), pap=f(r["PAP"]), paph=f(r["PAPH"]),
        plsm=f(r["PLSM"]), ldcum=jnp.asarray(r["LDCUM"] > 0),
        ktype=jnp.asarray(np.rint(r["KTYPE"]).astype(np.int32)),
        peis=jnp.zeros((ncol,), jnp.float64), plu=f(r["PLU"]), plude=f(r["PLUDE"]),
        pludeli=f(r["PLUDELI"]), psnde=f(psnde), pmfu=f(r["PMFU"]), pmfd=f(r["PMFD"]))


OUT_FIELDS = {   # port output field -> oracle file
    "tend_t": "TENDENCY_LOC_T", "tend_q": "TENDENCY_LOC_Q", "tend_a": "TENDENCY_LOC_A",
    "tend_cld": "TENDENCY_LOC_CLD", "plude": "PLUDE", "covptot": "PCOVPTOT",
    "prainfrac_toprfz": "PRAINFRAC_TOPRFZ",
    "fsqlf": "PFSQLF", "fsqif": "PFSQIF", "fcqnng": "PFCQNNG", "fcqlng": "PFCQLNG",
    "fsqrf": "PFSQRF", "fsqsf": "PFSQSF", "fcqrng": "PFCQRNG", "fcqsng": "PFCQSNG",
    "fsqltur": "PFSQLTUR", "fsqitur": "PFSQITUR", "fplsl": "PFPLSL", "fplsn": "PFPLSN",
    "fhpsl": "PFHPSL", "fhpsn": "PFHPSN",
}


def read_oracle(outdir, ncldtop) -> dict:
    """Oracle outputs with the regions CLOUDSC leaves undefined removed EXPLICITLY:
    the vapour slot of the cloud tendencies, PCOVPTOT above NCLDTOP (PFSD unused)."""
    o = Path(outdir)
    res = {k: read_txt(o / f"{v}.txt") for k, v in OUT_FIELDS.items()}
    res["tend_cld"] = res["tend_cld"][..., :4]
    res["covptot"] = res["covptot"][:, ncldtop - 1:]
    for k, v in res.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"oracle field {k} has non-finite values after masking")
    return res


def port_fields(out, ncldtop) -> dict:
    res = {k: np.asarray(getattr(out, k)) for k in OUT_FIELDS}
    res["covptot"] = res["covptot"][:, ncldtop - 1:]
    return res


def rel(a, b) -> float:
    """max|a-b| / max|b|; exact zeros on both sides -> 0."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.max(np.abs(a - b))
    s = np.max(np.abs(b))
    return 0.0 if d == 0 else float(d / s) if s > 0 else float("inf")


def read_stages(dumpdir, ncol, nrec) -> dict:
    """Stage dumps of the instrumented oracle: tag -> [nrec levels, ncol, ...] (from NCLDTOP)."""
    res = {}
    for f in sorted(Path(dumpdir).glob("*.bin")):
        raw = f.read_bytes()
        if len(raw) % nrec:
            raise ValueError(f"{f.name}: size {len(raw)} not a multiple of {nrec} records")
        n = (len(raw) // nrec - 4) // 8
        rec = np.frombuffer(raw, np.dtype([("jk", "<i4"), ("v", "<f8", (n,))]))
        if not np.array_equal(rec["jk"], rec["jk"][0] + np.arange(nrec)):
            raise ValueError(f"{f.name}: levels not consecutive")
        tail = {1: (), 5: (5,), 25: (5, 5), 4: (4,)}[n // ncol]
        res[f.stem] = np.stack([v.reshape((ncol,) + tail, order="F") for v in rec["v"]])
    return res
