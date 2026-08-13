#!/usr/bin/env python
"""Near-surface humidity and the bulk-evaporation driver: SCM columns vs the CRM.

The SCM-RCE convection campaign's tuned columns evaporate 1.2-1.9 mm/day while
the SAM_CRM RCE_small300 reference precipitates 2.40 mm/day.  The bulk
aerodynamic formula factors that deficit into two independent multiplicands,

    E = rho * C_E * |U| * (q_sat(T_sfc) - q_a)
        \\_______transfer_______/   \\___driver___/

so measuring the DRIVER on both sides decides which half is wrong, instead of
ranking guesses:

* driver matched, E low  -> the transfer half (exchange coefficient, wind
  speed, air density) is the defect, and the column is exonerated;
* driver small           -> the near-surface air is too moist, i.e. the defect
  is in the column the convection scheme maintains, and raising C_E would only
  paper over it.

The identity ``E_scm / E_crm == (driver ratio) * (transfer ratio)`` is exact by
construction and is printed as the instrument's own self-check: if it does not
hold to round-off, the parse is wrong and no number here may be quoted.

Conventions, both sides identical (the trap this probe exists to avoid):

* **Mixing ratio, not specific humidity.**  The RCEMIP reader takes SAM's
  native ``QV_avg``, which is a water-vapour MIXING RATIO, and legoESM's
  ``q_v`` tracer is the same quantity, so no specific-vs-mixing conversion
  enters anywhere.  The saturation value is reported in BOTH conventions
  (``r_sat`` and ``q_sat = r_sat / (1 + r_sat)``) because "saturation specific
  humidity" is the phrase in common use, but every difference and ratio below
  is formed from mixing ratios on both sides.
* **The same saturation curve the model uses.**  ``q_sat`` comes from
  ``legoesm.thermo.saturation_mixing_ratio`` (Tetens).  A re-derived Tetens in
  a diagnostic is what put a false super-saturation into CI once already
  (CLAUDE.md), so this module contains no saturation numerics of its own.
* **The same level.**  The SCM runs on the reference's own CRM-matching sigma
  grid, so index ``-1`` is the lowest full level on both sides; the probe
  asserts the level counts match rather than assuming it.

Reference evaporation is preferably the archive's own measured surface latent
heat flux (``hfls_avg``, RCEMIP 0D tier).  Absent that file it falls back to
the equilibrium identity ``E = P``, which holds for a settled RCE column and is
labelled as such in the output -- never silently.

Usage::

    python scripts/validate/check_scm_rce_surface_humidity.py \\
        results/scm_rce_convtune/arm_implicit_flux_morrison_noanchor \\
        --reference-dir results/rcemip_ref_sam300
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm import constants  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
from scripts.run.run_scm_rce_campaign import (  # noqa: E402
    FIXED_SST_K,
    SECONDS_PER_DAY,
    WING_P_SFC,
    build_reference_profiles,
)

#: Observed near-surface relative humidity over the tropical warm pool, for
#: orientation only.  Quoted as a RANGE from the literature (TOGA-COARE-era
#: surface analyses put the 10-20 m value at roughly 78-82 % over a 300 K
#: ocean); it is NOT used in any pass/fail decision here, because the
#: quantitative statement this probe makes is the SCM-vs-CRM comparison, and a
#: single literature number at a slightly different height would be a confound.
TROPICAL_SURFACE_RH_RANGE = (0.78, 0.82)


@dataclass(frozen=True)
class SurfaceHumidityRow:
    """Near-surface bulk-flux state of one column."""

    label: str
    T_air_K: float
    p_air_Pa: float
    #: Water-vapour MIXING RATIO of the lowest full level [kg/kg].
    r_air: float
    #: Saturation mixing ratio at the AIR temperature and pressure [kg/kg].
    r_sat_air: float
    relative_humidity: float
    #: Air-sea humidity difference driving evaporation [kg/kg], formed from
    #: mixing ratios: ``r_sat(SST, p_sfc) - r_air``.
    driver_kg_kg: float
    #: Air-sea temperature difference driving the sensible flux [K].
    delta_T_K: float
    evap_mm_day: float
    evap_source: str
    #: ``E / driver`` -- everything in the bulk formula except the driver,
    #: in mm/day per (kg/kg).  Proportional to ``rho * C_E * |U|``.
    transfer_mm_day_per_kg_kg: float


def surface_humidity_row(
    label: str,
    *,
    T_air_K: float,
    r_air: float,
    p_air_Pa: float,
    evap_mm_day: float,
    evap_source: str,
    sst_K: float = FIXED_SST_K,
    p_sfc_Pa: float = WING_P_SFC,
) -> SurfaceHumidityRow:
    """Bulk-flux decomposition of one column's lowest level.

    Pure: every array operation is on scalars, so this is the unit under test.
    """
    r_sat_sfc = float(saturation_mixing_ratio(np.float64(sst_K), np.float64(p_sfc_Pa)))
    r_sat_air = float(saturation_mixing_ratio(np.float64(T_air_K), np.float64(p_air_Pa)))
    driver = r_sat_sfc - float(r_air)
    transfer = float(evap_mm_day) / driver if driver != 0.0 else float("nan")
    return SurfaceHumidityRow(
        label=label,
        T_air_K=float(T_air_K),
        p_air_Pa=float(p_air_Pa),
        r_air=float(r_air),
        r_sat_air=r_sat_air,
        relative_humidity=float(r_air) / r_sat_air,
        driver_kg_kg=driver,
        delta_T_K=float(sst_K) - float(T_air_K),
        evap_mm_day=float(evap_mm_day),
        evap_source=evap_source,
        transfer_mm_day_per_kg_kg=transfer,
    )


def saturation_at_sst(
    sst_K: float = FIXED_SST_K, p_sfc_Pa: float = WING_P_SFC,
) -> tuple[float, float]:
    """Return ``(r_sat, q_sat)`` at the prescribed SST -- mixing ratio, then
    the specific-humidity form ``q = r / (1 + r)``.

    Both are returned because the two conventions differ by ~2 % at 300 K,
    which is the same order as the SCM-vs-CRM differences this probe reports;
    quoting one while comparing the other would manufacture a bias.
    """
    r_sat = float(saturation_mixing_ratio(np.float64(sst_K), np.float64(p_sfc_Pa)))
    return r_sat, r_sat / (1.0 + r_sat)


def crm_evap_from_hfls(aux_dir: Path) -> tuple[float, str] | None:
    """Reference evaporation [mm/day] from the archive's own ``hfls_avg``.

    ``None`` when the file is absent -- the caller then falls back to the
    equilibrium identity and SAYS SO.  A missing measurement must never be
    silently replaced by a derived one.
    """
    path = aux_dir / "SAM_CRM_RCE_small300_0D_hfls_avg.nc"
    if not path.exists():
        return None
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - environment-dependent
        return None
    with xr.open_dataset(path) as ds:
        name = "hfls_avg" if "hfls_avg" in ds else next(iter(ds.data_vars))
        series = np.asarray(ds[name].values, dtype=float).reshape(-1)
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return None
    # Equilibrium mean over the last quarter of the record, matching the
    # campaign's "settled window" convention rather than averaging spin-up in.
    tail = finite[-max(1, finite.size // 4):]
    hfls = float(np.mean(tail))
    return (
        hfls / constants.L_v * SECONDS_PER_DAY,
        f"measured hfls_avg={hfls:.2f} W/m^2 (last {tail.size}/{finite.size} samples)",
    )


def crm_sensible_from_hfss(aux_dir: Path) -> float | None:
    """Reference sensible heat flux [W/m^2], same windowing as ``hfls``."""
    path = aux_dir / "SAM_CRM_RCE_small300_0D_hfss_avg.nc"
    if not path.exists():
        return None
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - environment-dependent
        return None
    with xr.open_dataset(path) as ds:
        name = "hfss_avg" if "hfss_avg" in ds else next(iter(ds.data_vars))
        series = np.asarray(ds[name].values, dtype=float).reshape(-1)
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return None
    return float(np.mean(finite[-max(1, finite.size // 4):]))


def _scheme_rows(arm_dir: Path, p_air_Pa: float, condition: str) -> list[SurfaceHumidityRow]:
    rows: list[SurfaceHumidityRow] = []
    for path in sorted(arm_dir.glob("scheme_*.json")):
        with open(path) as fh:
            payload = json.load(fh)
        run = payload.get(condition)
        if not run or not run.get("qv_profile"):
            continue
        rows.append(
            surface_humidity_row(
                payload["scheme"],
                T_air_K=float(run["T_profile"][-1]),
                r_air=float(run["qv_profile"][-1]),
                p_air_Pa=p_air_Pa,
                evap_mm_day=float(run["evap_mm_day"]),
                evap_source=f"campaign applied lhflx ({condition})",
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm_dir", type=Path,
                   help="campaign arm directory holding scheme_*.json")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="CRM reference; default: read from the arm's run_meta.json")
    p.add_argument("--last-reference-files", type=int, default=5)
    p.add_argument("--condition", default="tuned", choices=("prior", "tuned"))
    p.add_argument("--aux-dir", type=Path, default=None,
                   help="directory holding the RCEMIP 0D flux files; "
                        "default <reference-dir>/aux_0D")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    arm_dir = args.arm_dir
    meta_path = arm_dir / "run_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    ref_dir = args.reference_dir or Path(str(meta.get("reference_dir", "")))
    if not ref_dir.is_dir():
        raise SystemExit(f"reference dir {ref_dir!r} not found; pass --reference-dir")
    aux_dir = args.aux_dir or (ref_dir / "aux_0D")

    ref = build_reference_profiles(ref_dir, args.last_reference_files)
    p_air = float(ref.sigma_full[-1] * WING_P_SFC)

    r_sat_sfc, q_sat_sfc = saturation_at_sst()
    print("SATURATION AT THE PRESCRIBED SST")
    print(f"  SST                     {FIXED_SST_K:.3f} K")
    print(f"  surface pressure        {WING_P_SFC / 100.0:.2f} hPa")
    print(f"  r_sat (mixing ratio)    {r_sat_sfc * 1e3:.4f} g/kg")
    print(f"  q_sat (specific)        {q_sat_sfc * 1e3:.4f} g/kg")
    print(f"  lowest full level       z = {ref.z_m[-1]:.1f} m, p = {p_air / 100.0:.2f} hPa")
    print()

    crm_e = crm_evap_from_hfls(aux_dir)
    if crm_e is None:
        crm_e = (float(ref.precip_ref_mm_day),
                 "EQUILIBRIUM IDENTITY E=P (hfls_avg absent)")
    crm_row = surface_humidity_row(
        "CRM (SAM_CRM RCE_small300)",
        T_air_K=float(ref.T_ref[-1]),
        r_air=float(ref.qv_ref[-1]),
        p_air_Pa=p_air,
        evap_mm_day=crm_e[0],
        evap_source=crm_e[1],
    )
    hfss = crm_sensible_from_hfss(aux_dir)

    rows = _scheme_rows(arm_dir, p_air, args.condition)
    if not rows:
        raise SystemExit(f"no {args.condition} checkpoints with profiles in {arm_dir}")

    header = (f"{'column':28s} {'T_air':>7s} {'r_air':>8s} {'RH':>6s} "
              f"{'driver':>8s} {'dT':>6s} {'E':>6s} {'transfer':>9s} "
              f"{'drv/CRM':>8s} {'trf/CRM':>8s} {'E/CRM':>7s}")
    print(f"NEAR-SURFACE BULK STATE ({args.condition} columns), lowest full level")
    print(f"  units: T_air K | r_air, driver g/kg | dT K | E mm/day | "
          f"transfer mm/day per (g/kg)")
    print(header)
    print("-" * len(header))

    def _line(row: SurfaceHumidityRow) -> str:
        drv = row.driver_kg_kg / crm_row.driver_kg_kg
        trf = row.transfer_mm_day_per_kg_kg / crm_row.transfer_mm_day_per_kg_kg
        ev = row.evap_mm_day / crm_row.evap_mm_day
        return (f"{row.label:28s} {row.T_air_K:7.2f} {row.r_air * 1e3:8.3f} "
                f"{row.relative_humidity:6.3f} {row.driver_kg_kg * 1e3:8.3f} "
                f"{row.delta_T_K:6.2f} {row.evap_mm_day:6.2f} "
                f"{row.transfer_mm_day_per_kg_kg / 1e3:9.4f} "
                f"{drv:8.3f} {trf:8.3f} {ev:7.3f}")

    print(_line(crm_row))
    for row in sorted(rows, key=lambda r: -r.evap_mm_day):
        print(_line(row))
    print()
    print(f"  CRM evaporation source: {crm_row.evap_source}")
    if hfss is not None:
        bowen = hfss / (crm_row.evap_mm_day / SECONDS_PER_DAY * constants.L_v)
        print(f"  CRM sensible heat flux: {hfss:.2f} W/m^2  (Bowen ratio {bowen:.3f})")
    lo, hi = TROPICAL_SURFACE_RH_RANGE
    print(f"  literature tropical near-surface RH, orientation only: {lo:.2f}-{hi:.2f}")
    print()

    print("SELF-CHECK: E_ratio == driver_ratio * transfer_ratio (exact by "
          "construction; a mismatch means the parse is wrong)")
    worst = 0.0
    for row in rows:
        lhs = row.evap_mm_day / crm_row.evap_mm_day
        rhs = ((row.driver_kg_kg / crm_row.driver_kg_kg)
               * (row.transfer_mm_day_per_kg_kg / crm_row.transfer_mm_day_per_kg_kg))
        worst = max(worst, abs(lhs - rhs))
    print(f"  max |lhs - rhs| over {len(rows)} schemes = {worst:.3e}")
    if worst > 1.0e-9:
        raise SystemExit("SELF-CHECK FAILED: the decomposition is not consistent")

    if args.json_out is not None:
        payload = {
            "sst_K": FIXED_SST_K,
            "p_sfc_Pa": WING_P_SFC,
            "r_sat_sst_kg_kg": r_sat_sfc,
            "q_sat_sst_kg_kg": q_sat_sfc,
            "condition": args.condition,
            "crm": asdict(crm_row),
            "crm_hfss_W_m2": hfss,
            "schemes": [asdict(r) for r in rows],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
