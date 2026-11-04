"""Generate tests/atmosphere/oracle_fixtures/rad_boundary.npz.

Oracle: CESM2.1 share code cime5.6.47 (d8e9b894d7eb96fe96f8f8a6943b421147f94606),
compiled unmodified -- shr_orb_params / shr_orb_decl / shr_orb_cosz -- plus the
coupler open-ocean albedo lines seq_flux_mct.F90 L881-894 (direct albedo at the
instantaneous cosine, diffuse seq_flux_mct_albdif = 0.06).  The fixture holds the
oracle INPUTS (calendar day, latitude, longitude, orbit) and OUTPUTS (declination,
Earth-Sun distance factor, cos zenith, direct and diffuse ocean albedo).

Points (tier-1 audit set, /burg/glab/users/pg2328/oracle_audit/radiation_boundary_inputs):
7 columns x 9 simulation days x 48 half-hours.  The calendar day is the production
map ``forcing.time_utils.day_to_calendar`` (day_of_year, 1.0 = Jan 1 00Z), which is
CESM's calday origin.  The orbit is legoESM's ``earth_orbit()``; CESM's ``mvelp`` is
the longitude of perihelion minus 180 deg (shr_orb_params adds it back).

usage (compute node):
    bash build.sh <cime@cime5.6.47> <build dir>
    python make_fixture.py <build dir>/oracle_driver <cime@cime5.6.47> <out.npz>
"""
import json
import subprocess
import sys

import numpy as np

from legoesm.atmosphere.physics.radiation.solar import earth_orbit
from legoesm.forcing.time_utils import day_to_calendar

ORACLE, CIME, OUT = sys.argv[1:4]

COLS = [  # name, lat, lon [deg]
    ("tropical_deep_warmpool", 0.5, 150.0),
    ("subtrop_sc_peru", -20.0, 275.0),
    ("midlat_45N", 45.0, 0.0),
    ("polar_75N", 75.0, 20.0),
    ("polar_80S", -80.0, 300.0),
    ("edge_pole_90N", 90.0, 0.0),
    ("edge_equator_dateline", 0.0, 180.0),
]
DAYS0 = [0, 2, 79, 80, 171, 184, 265, 354, 364]   # simulation day, 0 = Jan 1 00Z
HOURS = np.arange(0.0, 24.0, 0.5)

lat, lon, sim_day = [], [], []
for _, la, lo in COLS:
    for d0 in DAYS0:
        for h in HOURS:
            lat.append(np.deg2rad(la))
            lon.append(np.deg2rad(lo))
            sim_day.append(d0 + h / 24.0)
lat, lon, sim_day = map(np.asarray, (lat, lon, sim_day))
cal = np.array([day_to_calendar(float(d)) for d in sim_day])
doy, sod = cal[:, 0], cal[:, 1]

orbit = earth_orbit()
mvelp = orbit.long_perihelion_deg - 180.0
lines = [f"{orbit.eccentricity:.17e} {orbit.obliquity_deg:.17e} {mvelp:.17e}", str(len(doy))]
lines += [f"{c:.17e} {a:.17e} {b:.17e}" for c, a, b in zip(doy, lat, lon)]
res = subprocess.run([ORACLE], input="\n".join(lines) + "\n", capture_output=True,
                     text=True, check=True)
rows = [list(map(float, ln.split())) for ln in res.stdout.strip().splitlines()]
orb_out, out = np.asarray(rows[0]), np.asarray(rows[1:])
assert out.shape == (len(doy), 5), out.shape

meta = {
    "oracle": "CESM2.1 cime5.6.47 shr_orb_mod.F90 (shr_orb_params/decl/cosz, unmodified) "
              "+ seq_flux_mct.F90 L881-894 (ocean albedo, verbatim)",
    "repo": "https://github.com/ESMCI/cime",
    "tag": subprocess.run(["git", "-C", CIME, "describe", "--tags"], capture_output=True,
                          text=True, check=True).stdout.strip(),
    "sha": subprocess.run(["git", "-C", CIME, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip(),
    "generator": "scripts/validate/oracle/rad_boundary/{build.sh,make_fixture.py}",
    "orbit": orbit._asdict(), "cesm_mvelp_deg": mvelp,
    "oracle_obliqr_lambm0_mvelpp": orb_out.tolist(),
    "columns": COLS, "sim_days": DAYS0, "hours": HOURS.tolist(),
}
np.savez(OUT, sim_day=sim_day, day_of_year=doy, seconds_of_day=sod, lat=lat, lon=lon,
         delta=out[:, 0], eccf=out[:, 1], cosz=out[:, 2], albedo_dir=out[:, 3],
         albedo_dif=out[:, 4], meta=json.dumps(meta))
print("FIXTURE_OK", OUT, out.shape, meta["tag"], meta["sha"])
