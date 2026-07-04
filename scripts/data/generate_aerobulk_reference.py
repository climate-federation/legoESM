"""Generate the frozen aerobulk air-sea bulk-flux oracle baseline.

Runs the AeroBulk Fortran reference implementation (Brodeau et al. 2017 —
the lineage ancestor of NEMO's ``sbcblk``) via ``aerobulk-python`` (noskin,
i.e. no cool-skin/warm-layer, matching legoESM's schemes) over a seeded
random + structured sample of the marine surface layer, and freezes inputs
+ outputs into a tiny tracked baseline:

    tests/unit/baselines/aerobulk_noskin_v1.npz
    tests/unit/baselines/aerobulk_noskin_v1.json   (provenance metadata)

Consumed by ``tests/unit/test_aerobulk_oracle.py`` — the parity test needs
only the .npz, never aerobulk itself.

Run inside a conda env with aerobulk-python (NOT the repo venv):

    mamba create -n aerobulk -c conda-forge python=3.11 aerobulk-python
    /path/to/envs/aerobulk/bin/python scripts/data/generate_aerobulk_reference.py

AeroBulk sign/unit conventions recorded in the baseline (kept verbatim,
the parity test does the convention mapping):
  ql, qh  [W/m^2]  positive DOWNWARD (ocean gains)
  taux/y  [N/m^2]  along-wind (atmospheric convention)
  evap    [kg/m^2/s] negative when the ocean loses water
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ALGOS = ("ncar", "coare3p0")
# (zt, zu): single-height 10 m, and the OMIP/JRA55-do 2 m T,q / 10 m wind split
HEIGHTS = ((10.0, 10.0), (2.0, 10.0))
NITER = 5          # matches NEMO sbcblk and legoESM compute_most_fluxes defaults
N_RANDOM = 1024
SEED = 20260703

# Rough 98%-saline q_sat envelope over SST range, ONLY to keep sampled
# humidity physically plausible (subsaturated).  Interpolated linearly.
# satcurve-ok: sampling envelope for input generation, not a physics formula;
# the frozen baseline stores the actual q values used.
_QSAT_ENVELOPE_T_K = np.array([270.0, 275.0, 280.0, 285.0, 290.0, 295.0, 300.0, 305.0])
_QSAT_ENVELOPE_Q = np.array([2.9e-3, 4.2e-3, 6.0e-3, 8.4e-3, 11.7e-3, 16.0e-3, 21.6e-3, 28.8e-3])


def sample_marine_inputs(n: int = N_RANDOM, seed: int = SEED):
    """Seeded random marine-surface-layer inputs + structured edge cases.

    Returns dict of 1-D float64 arrays: sst, t_air, q_air, u, v, slp.
    ``t_air``/``q_air`` are ABSOLUTE T [K] / specific humidity [kg/kg] at
    the (variable) zt height; the same numeric values are reused for both
    height configurations.
    """
    rng = np.random.default_rng(seed)
    sst = rng.uniform(271.5, 303.0, n)
    t_air = sst + rng.uniform(-8.0, 8.0, n)
    rh = rng.uniform(0.05, 0.95, n)
    q_air = rh * np.interp(t_air, _QSAT_ENVELOPE_T_K, _QSAT_ENVELOPE_Q)
    slp = rng.uniform(96000.0, 104000.0, n)
    u = rng.uniform(-25.0, 25.0, n)
    v = rng.uniform(-25.0, 25.0, n)

    # Structured edge cases: calm/hurricane winds, strong stable/unstable,
    # freezing and tropical SST, very dry and near-saturated air.
    edge = np.array([
        # sst,    t_air,  q_air,   u,     v,     slp
        [271.5,  269.0,  2.0e-3,  0.05,  0.05,  101000.0],   # calm, freezing
        [271.5,  279.0,  4.0e-3,  3.0,   0.0,   101000.0],   # strong stable
        [302.0,  294.0,  1.6e-2,  8.0,  -6.0,   100500.0],   # strong unstable
        [302.0,  301.5,  2.5e-2,  1.0,   1.0,   100800.0],   # moist tropics, light wind
        [290.0,  289.0,  9.0e-3, 32.0,  10.0,    97000.0],   # hurricane-force
        [285.0,  285.0,  8.0e-3,  0.0,   0.0,   101325.0],   # dead calm, neutral
        [295.0,  290.0,  2.0e-4,  6.0,   2.0,   102000.0],   # very dry air
        [278.0,  286.0,  5.5e-3,  2.0,  -1.0,   104000.0],   # warm air over cold sea
    ])
    return {
        "sst": np.concatenate([sst, edge[:, 0]]),
        "t_air": np.concatenate([t_air, edge[:, 1]]),
        "q_air": np.concatenate([q_air, edge[:, 2]]),
        "u": np.concatenate([u, edge[:, 3]]),
        "v": np.concatenate([v, edge[:, 4]]),
        "slp": np.concatenate([slp, edge[:, 5]]),
    }


def main() -> None:
    import aerobulk
    from aerobulk.flux import noskin_np

    inp = sample_marine_inputs()
    n = inp["sst"].shape[0]
    shp = (n, 1)  # aerobulk-python wants >=2-D arrays

    out: dict[str, np.ndarray] = {f"in_{k}": v for k, v in inp.items()}
    for algo in ALGOS:
        for zt, zu in HEIGHTS:
            ql, qh, taux, tauy, evap = noskin_np(
                inp["sst"].reshape(shp), inp["t_air"].reshape(shp),
                inp["q_air"].reshape(shp), inp["u"].reshape(shp),
                inp["v"].reshape(shp), inp["slp"].reshape(shp),
                algo=algo, zt=zt, zu=zu, niter=NITER,
                input_range_check=True)
            tag = f"{algo}_zt{int(zt)}_zu{int(zu)}"
            for name, arr in (("ql", ql), ("qh", qh), ("taux", taux),
                              ("tauy", tauy), ("evap", evap)):
                out[f"{tag}_{name}"] = np.asarray(arr, dtype=np.float64).ravel()

    base = Path(__file__).resolve().parents[2] / "tests" / "unit" / "baselines"
    base.mkdir(parents=True, exist_ok=True)
    npz_path = base / "aerobulk_noskin_v1.npz"
    np.savez_compressed(npz_path, **out)

    meta = {
        "generator": "scripts/data/generate_aerobulk_reference.py",
        "aerobulk_python_version": getattr(aerobulk, "__version__", "unknown"),
        "algos": list(ALGOS),
        "heights_zt_zu": [list(h) for h in HEIGHTS],
        "niter": NITER,
        "n_random": N_RANDOM,
        "n_total": n,
        "seed": SEED,
        "skin_correction": False,
        "conventions": {
            "ql_qh": "W/m^2, positive downward (ocean gains)",
            "taux_tauy": "N/m^2, along-wind (atmospheric convention)",
            "evap": "kg/m^2/s, negative when ocean loses water",
            "t_air": "ABSOLUTE temperature at zt [K]",
            "q_air": "specific humidity at zt [kg/kg]",
        },
    }
    (base / "aerobulk_noskin_v1.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {npz_path} ({npz_path.stat().st_size / 1024:.0f} KiB), "
          f"n={n}, algos={ALGOS}, heights={HEIGHTS}")


if __name__ == "__main__":
    main()
