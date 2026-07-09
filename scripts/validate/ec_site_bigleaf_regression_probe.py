"""EC-site big-leaf vs two-leaf-canopy regression probe (offline, no external data).

Companion to the two-leaf EC-site runbook (``docs/land/ec_site_evaluation_runbook.md``)
for the ONE thing that runbook cannot cover on a data-free box: a controlled
BEFORE/AFTER check of the big-leaf ``SimpleSEB`` photosynthesis path after PR #897
routed it through the canonical FvCB kernels (``canopy.photosynthesis.c3/c4_assimilation``)
and retired the C3-only ``farquhar_photosynthesis``.

The runbook's model run needs the DifferBESS ``<SITE>_driver_v2.nc`` FLUXNET drivers
(100-220 MB/site, not checked in), so a real skill-vs-obs run is not reproducible offline.
What IS reproducible is a single-revision invariant probe of the big-leaf GPP response
and the two-leaf photosynthesis kernels, on site-representative forcing at the per-site
LAI recorded in the committed fixture ``scripts/validate/ec_site_regression_lai.csv``
(which documents its own provenance -- the LAI values are the median / p90 of the ``lai``
variable in the ``ec_site_example_data`` NetCDFs on branch feat/ec-site-canopy-validation,
transcribed here because those large NetCDFs are not on this branch).  Run the probe at two
git revisions via a ``PYTHONPATH`` swap and diff the JSON to get the before/after delta;
the method + the numbers behind PR #897 are recorded in
``docs/land/ec_site_bigleaf_regression.md``.

This is a BEFORE/AFTER magnitude + physical-sanity probe (finite, GPP>=0, monotone light
response, sane magnitude), NOT the runbook's NSE skill-vs-obs evaluation: forcing is
synthetic-representative, leaf temperature is set to air temperature, and each point is a
single instantaneous evaluation with no leaf energy-balance solve.

Usage::

    JAX_ENABLE_X64=1 python scripts/validate/ec_site_bigleaf_regression_probe.py \
        --out probe_head.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.core.coupling_fields import AtmToSurface  # noqa: E402
from legoesm.land.config import LandConfig  # noqa: E402
from legoesm.land.stomata_utils import compute_effective_beta  # noqa: E402
from legoesm.land.carbon.carbon_cycle import init_carbon_state  # noqa: E402
from legoesm.land.canopy.photosynthesis import (  # noqa: E402
    c3_photosynthesis,
    c4_photosynthesis,
)

_LAI_CSV = pathlib.Path(__file__).with_name("ec_site_regression_lai.csv")


def load_sites(csv_path: pathlib.Path = _LAI_CSV) -> dict:
    """Load the per-site probe inputs from the committed provenance fixture.

    Raises ``FileNotFoundError`` if the fixture is missing so a stale or absent
    input table fails loudly rather than silently probing hard-coded values.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"EC-site regression LAI fixture missing: {csv_path}. See its header for "
            "provenance; the probe will not run on unrecorded inputs.")
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(row for row in fh if not row.startswith("#"))
        sites = {}
        for r in reader:
            sites[r["site"]] = dict(
                pft=r["pft"], T_air=float(r["T_air_K"]), q=float(r["q_kgkg"]),
                lai=[float(r["lai_median"]), float(r["lai_p90"])])
    if not sites:
        raise ValueError(f"no site rows parsed from {csv_path}")
    return sites


# Site-representative summer-noon conditions; per-site LAI [median, p90] and the
# representative T_air / q come from the committed fixture (see load_sites); the values
# are held identical across the two revisions compared so only the code revision changes.
SITES = load_sites()
SW_LEVELS = [100.0, 300.0, 600.0, 900.0]      # W m-2 (morning -> noon)
BETA_LEVELS = [0.3, 0.6, 1.0]                  # soil-moisture stress dry -> wet
CO2_PPMV = 400.0
P_SFC = 1.0e5


def _forcing(sw: float, T_air: float, q: float) -> AtmToSurface:
    s = (1,)
    f = lambda v: jnp.full(s, v)
    z = jnp.zeros(s)
    return AtmToSurface(
        sw_down=f(sw), lw_down=f(350.0), precip_total=z, precip_snow=z,
        T_lowest=f(T_air), q_lowest=f(q), u_lowest=f(2.0), v_lowest=z,
        p_lowest=f(P_SFC), p_surface=f(P_SFC), rho_lowest=f(1.15),
        cos_zenith=f(0.7), co2_ppmv=jnp.array(CO2_PPMV),
        has_radiation=jnp.array(1.0), has_precipitation=jnp.array(0.0),
    )


def bigleaf_grid() -> list[dict]:
    """Big-leaf coupled-Farquhar GPP + effective beta over the site x forcing grid.

    ``land_params=None`` -> pure C3 (the documented default), so the same code path
    is exercised at any revision.
    """
    base = LandConfig()
    cfg = base._replace(
        stomata=base.stomata._replace(enabled=True),
        carbon=base.carbon._replace(scheme="differland"),
    )
    lcma = cfg.carbon.LCMA
    carb0 = init_carbon_state((1,), cfg.carbon)
    rows = []
    for site, sp in SITES.items():
        for lai in sp["lai"]:
            carb = carb0._replace(C_fol=jnp.full((1,), lai * lcma))
            for sw in SW_LEVELS:
                for beta_s in BETA_LEVELS:
                    beta, gpp, _sif = compute_effective_beta(
                        jnp.full((1,), sp["T_air"]),
                        _forcing(sw, sp["T_air"], sp["q"]),
                        jnp.full((1,), beta_s), cfg, carbon_state=carb,
                        dt=3600.0, land_params=None)
                    rows.append(dict(
                        site=site, pft=sp["pft"], lai=round(float(lai), 3),
                        sw=sw, beta_soil=beta_s,
                        gpp=float(np.asarray(gpp).reshape(-1)[0]),
                        beta_eff=float(np.asarray(beta).reshape(-1)[0]),
                    ))
    return rows


def _has_fc4() -> bool:
    """True on revisions where LandSurfaceParams carries the PR-#897 ``fC4`` field.

    The cross-revision before/after uses ``land_params=None`` (works at both revisions);
    the land_params override + C3/C4-blend coverage below only exists post-#897.
    """
    from legoesm.land.surface_params import LandSurfaceParams
    return "fC4" in LandSurfaceParams._fields


def _land_params(shape, fc4: float, vcmax: float, g1: float, lcma: float):
    """Minimal shape-matched LandSurfaceParams carrying a per-site C4 fraction, so the
    probe drives the SAME production override path SimpleSEB uses (Vc_max25/g1/LCMA/fC4)."""
    from legoesm.land.surface_params import LandSurfaceParams
    import jax.numpy as _jnp
    f = lambda v: _jnp.full(shape, v)
    return LandSurfaceParams(
        albedo_veg=f(0.15), emissivity=f(0.97), z0=f(0.1), W_max=f(200.0),
        C_soil=f(2.0e6), d_soil=f(1.0), root_depth=f(1.0), theta_wp=f(0.15),
        theta_fc=f(0.30), Vc_max25=f(vcmax), LCMA=f(lcma), g1=f(g1), fC4=f(fc4),
    )


def bigleaf_production_grid() -> list[dict]:
    """HEAD-only: drive the big leaf through the production ``land_params`` override path
    at each site for fC4 in {0 (C3), 1 (C4)}, proving the C3/C4 blend + override are live.

    Returns [] on pre-#897 revisions (no ``fC4`` field) so the probe still runs there.
    """
    if not _has_fc4():
        return []
    base = LandConfig()
    cfg = base._replace(
        stomata=base.stomata._replace(enabled=True),
        carbon=base.carbon._replace(scheme="differland"),
    )
    lcma = float(cfg.carbon.LCMA)
    vcmax = float(cfg.stomata.Vc_max25)
    g1 = float(cfg.stomata.g1_bb)
    carb0 = init_carbon_state((1,), cfg.carbon)
    rows = []
    for site, sp in SITES.items():
        lai = sp["lai"][1]                                   # p90 LAI, productive
        carb = carb0._replace(C_fol=jnp.full((1,), lai * lcma))
        for fc4 in (0.0, 1.0):
            lp = _land_params((1,), fc4, vcmax, g1, lcma)
            _beta, gpp, _sif = compute_effective_beta(
                jnp.full((1,), sp["T_air"]), _forcing(600.0, sp["T_air"], sp["q"]),
                jnp.full((1,), 0.6), cfg, carbon_state=carb, dt=3600.0, land_params=lp)
            rows.append(dict(site=site, pft=sp["pft"], fc4=fc4,
                             gpp=float(np.asarray(gpp).reshape(-1)[0])))
    return rows


def canopy_kernel_grid() -> dict[str, list[float]]:
    """Two-leaf canopy C3/C4 photosynthesis kernels over a temperature grid.

    These are the only two-leaf-canopy surfaces PR #897 touched (it factored shared
    ``c3/c4_assimilation`` kernels out of them); the full canopy flux is a pure consumer.
    Diffing this across revisions detects any kernel drift (expected: bit-identical).
    """
    Tf = jnp.array([283.0, 293.0, 298.0, 308.0])
    out: dict[str, list[float]] = {}
    for name, y in [
        ("c3", c3_photosynthesis(Tf, jnp.full(4, 280.0), jnp.full(4, 600.0),
                                 jnp.full(4, 60.0), jnp.full(4, 1.0e5),
                                 jnp.full(4, 0.8), jnp.full(4, 25.0))),
        ("c4", c4_photosynthesis(Tf, jnp.full(4, 140.0), jnp.full(4, 600.0),
                                 jnp.full(4, 40.0))),
    ]:
        out[name] = [float(v) for v in np.asarray(y).reshape(-1)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output JSON path")
    a = ap.parse_args(argv)
    result = dict(bigleaf=bigleaf_grid(),
                  bigleaf_production=bigleaf_production_grid(),
                  canopy_kernel=canopy_kernel_grid())
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {a.out}: {len(result['bigleaf'])} bigleaf rows; "
          f"canopy c3={result['canopy_kernel']['c3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
