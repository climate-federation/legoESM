"""Run the OpenIFS cumulus oracle (CUMASTRN, 8f6f722) and write the fixture.

usage: python make_fixture.py <columns dir> <build dir> <out.npz>

<columns dir> holds the tier-2 audit columns ``columns_{L32,L137,STRESS}.npz``
(16 L32 soundings incl. DYCOMS-II RF01, 5 L137 columns incl. two
dwarf-p-cloudsc reference columns, 3 dried-environment stress columns; see
the ``src`` entry of each column).  They carry T, q, u, v, environmental
condensate, pressure, geopotential, land flag, surface fluxes (positive
upward, W m-2) and the tendency pairs fed to CUMASTRN:
PTENT/PTENQ = ``dT_other``/``dq_other``, PTENTA/PTENQA = ``dT_adv``/``dq_adv``.
<build dir> holds ``oracle.x`` from build.sh.

Two arms, both with every process of the shipped scheme on (downdraught,
momentum transport, mid-level, detrained snow, melting, sub-cloud
evaporation, glaciation) and the stock mixed-phase fcttre functions:
  L  legoESM constants and the legoesm.thermo Tetens liquid curve
     (CONST_MODE = 1, values written into the namelist from legoesm)
  N  the IFS native constants (CONST_MODE = 2)
  L0 as L with PLITOT = 0, the configuration the legoESM wrapper
     (_ifs_faithful.py) runs; it checks the wrapper seam against the oracle
The fixture stores the CUMASTRN inputs exactly as the driver read them and
every CUMASTRN output, per arm and group, plus the constants each arm ran.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C
from legoesm import thermo

DT = 112.5        # s, dt of the committed Bechtold decks (config/amip/amip_sundqvist_l36.yaml)
DX = 1.0e5        # m; PGAW = (DX/(2 RA))**2/pi so that cumastrn's ZDX = DX
GROUPS = ("L32", "L137", "STRESS")
SWITCHES = dict(ZERO_MELT=False, ZERO_EVAP=False, LMFDD=True, LMFDUDV=True, LMFMID=True,
                LMFDSNOW=True, LMFGLAC=True, LMFSCV=True, LMFPEN=True, RTICECU_OFF=38.0)
LEGO = dict(CRG=C.g, CRD=C.R_d, CRV=C.R_v, CRCPD=C.c_pd, CRTT=C.T_freeze,
            CRLVTT=C.L_v,  # latent-ok: IFS oracle constant RLVTT
            CRA=C.R_earth, CRLMLT=C.L_f,  # latent-ok: IFS oracle constants RLVTT/RLMLT
            CR3LES=thermo.TETENS_A,
            CR4LES_OFF=thermo.TETENS_B_K, CR2ES_E0=thermo.TETENS_E0_PA)
ARMS = {"L": dict(CONST_MODE=1, **LEGO), "N": dict(CONST_MODE=2),
        "L0": dict(CONST_MODE=1, **LEGO)}
PLITOT = {"L": True, "N": True, "L0": False}
CONST_NAMES = ("RG", "RD", "RV", "RCPD", "RLVTT", "RTT", "RA", "RLMLT", "R2ES", "R3LES",
               "R4LES", "R3IES", "R4IES", "RPI")
FULL = ("T", "q", "p_full", "geo_full", "u", "v", "omega", "lit",
        "dT_other", "dq_other", "dT_adv", "dq_adv")
HALF = ("p_half", "geo_half")


def _r(x):
    """The value the driver reads back from its '%.17e' text (round trip)."""
    return float(f"{float(x):.17e}")


def inputs(d):
    """CUMASTRN inputs of one group, rounded through the driver's text."""
    ncol = d["T"].shape[0]
    out = {k: np.vectorize(_r)(d[k]).astype(np.float64) for k in FULL + HALF}
    out["land"] = np.array([float(int(x)) for x in d["land"]])
    out["ahfs"] = np.array([_r(-x) for x in d["shf_up"]])           # IFS sign
    out["qhfl"] = np.array([_r(-x / C.L_v) for x in d["lhf_up"]])  # latent-ok: IFS oracle constant (audit harness convention)
    out["pgaw"] = np.full(ncol, _r((DX / (2.0 * C.R_earth)) ** 2 / np.pi))
    out["shf_up"] = np.asarray(d["shf_up"], np.float64)
    out["lhf_up"] = np.asarray(d["lhf_up"], np.float64)
    out["names"] = np.array([str(n) for n in d["names"]])
    return out


def oracle_text(x, plitot):
    ncol, nlev = x["T"].shape
    lines = [f"{ncol} {nlev}"]
    for i in range(ncol):
        lines.append(str(x["names"][i]))
        lines.append(f"{int(x['land'][i])} {x['ahfs'][i]:.17e} {x['qhfl'][i]:.17e} "
                     f"{x['pgaw'][i]:.17e} {DT:.17e}")
        for k in FULL + HALF:
            row = x[k][i] if (k != "lit" or plitot) else np.zeros_like(x[k][i])
            lines.append(" ".join(f"{v:.17e}" for v in row))
    return "\n".join(lines) + "\n"


def parse(path):
    """-> (constants, {column: {tag: array}}) from the driver's 'out' stage."""
    toks = Path(path).read_text().split("\n")
    consts, res, col, stage, i = None, {}, None, None, 0
    while i < len(toks):
        line = toks[i].split()
        i += 1
        if not line:
            continue
        if line[0] == "K":
            vals = []
            while len(vals) < len(CONST_NAMES):
                vals.extend(toks[i].split())
                i += 1
            consts = np.array([float(v) for v in vals])
        elif line[0] == "C":
            col, stage = line[1], None
            res[col] = {}
        elif line[0] == "S":
            stage = line[1]
        elif line[0] in ("R", "I"):
            tag, n = line[1], int(line[2])
            vals = []
            while len(vals) < n:
                vals.extend(toks[i].split())
                i += 1
            if stage == "out":
                res[col][tag] = (np.array([float(v) for v in vals]) if line[0] == "R"
                                 else np.array([int(v) for v in vals]))
    return consts, res


def main(cdir, build, out):
    exe = Path(build) / "oracle.x"
    fx = {"dt": np.float64(DT), "const_names": np.array(CONST_NAMES)}
    for g in GROUPS:
        x = inputs(np.load(Path(cdir) / f"columns_{g}.npz"))
        for k, v in x.items():
            fx[f"{g}_{k}"] = v
        for arm, sw in ARMS.items():
            inf = Path(build) / f"in_{g}_lit{int(PLITOT[arm])}.txt"
            inf.write_text(oracle_text(x, PLITOT[arm]))
            outf = Path(build) / f"out_{arm}_{g}.txt"
            nml = Path(build) / f"run_{arm}_{g}.nml"
            body = ", ".join(f"{k}={'.TRUE.' if v is True else '.FALSE.' if v is False else repr(v)}"
                             for k, v in {**SWITCHES, **sw}.items())
            nml.write_text(f"&RUN {body}, INFILE='{inf}', OUTFILE='{outf}' /\n")
            r = subprocess.run([str(exe), str(nml)], capture_output=True, text=True)
            if r.returncode != 0 or "DRIVER_DONE" not in r.stdout:
                sys.exit(f"oracle failed {arm} {g}: {r.stderr[-500:]}")
            consts, res = parse(outf)
            fx[f"{arm}_const"] = consts
            for tag in res[str(x["names"][0])]:
                v = np.stack([res[str(n)][tag] for n in x["names"]])
                fx[f"{arm}_{g}_{tag}"] = v[:, 0] if v.shape[1] == 1 else v   # scalars -> (ncol,)
    np.savez_compressed(out, **fx)
    print("FIXTURE_OK", out, Path(out).stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:4])
