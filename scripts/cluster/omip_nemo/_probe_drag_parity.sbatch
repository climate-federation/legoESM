#!/bin/bash
#SBATCH --job-name=dragpar
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dragpar_%j.out
#
# IS OUR DRAG COEFFICIENT THE ONE NEMO USES?
#
# Both reviewers independently rank excess westward APPLIED STRESS as the top
# cause of our equatorial defect (surface current +97% too strong at 220E, the
# undercurrent collapsed to 39% and gone by 240E). Two things have since been
# excluded, so the target is now a single quantity:
#   * the WIND is common -- NEMO's applied windsp matches the CORE-II field we
#     both read to 0.05% at 220E and better than 2.2% across 180-240E
#     (ad4d26e19);
#   * the RELATIVE-WIND term is off on BOTH sides -- ORCA1 leaves rn_vfac = 0
#     (namelist_ref:319, not overridden) and our card sets neither
#     --relative-winds nor --wind-vfac, which resolves to 0.0.
# Same wind, same current-feedback setting: any applied-stress difference is
# therefore the DRAG COEFFICIENT.
#
# This needs no model run. NEMO publishes both taum and windsp, so its
# EFFECTIVE drag is Cd_eff = taum / (rho_air * windsp^2) directly from its own
# output. Ours comes from the model's OWN routine -- ncar_transfer_coefficients
# in ocean/bulk_flux_omip.py, an exact port of NEMO's turb_ncar, which is what
# the card selects (algo="ncar") -- evaluated at the SAME wind speeds. No
# hand-rolled bulk formula appears anywhere here.
#
# PRE-REGISTERED, BEFORE LOOKING, and the expected size is set by the symptom:
# the surface current is +97% too strong, so
#   * ratio ours/NEMO >= 1.6 over the defect region 180-240E -> DRAG EXPLAINS
#     IT, and the stress ranking is confirmed at its source;
#   * ratio within 0.85-1.15 -> DRAG IS EXONERATED and the question goes back
#     to continuity, which both reviewers rank second;
#   * between -> report the number and claim no direction.
# REGISTER THE STRUCTURE TOO, not just a scalar mean: the last probe's mean was
# dragged into the indeterminate band by one coastal outlier while the
# per-longitude values were decisive. The per-longitude column is the verdict.
#
# CAVEATS THAT BOUND THE READING, stated up front:
#   * taum is the modulus over the whole cell; taum_oce is the open-ocean
#     value. Both are printed. At the equator there is no ice so they should
#     agree, and if they do not, that disagreement is itself informative.
#   * rho_air is not published by NEMO, so the oracle's Cd_eff carries our
#     moist-air density evaluated from the CORE-II air temperature and
#     humidity. A density error propagates linearly into Cd_eff and would be
#     an artifact of THIS probe, not of either model -- which is why the
#     density used is printed alongside.
#   * our Cd is stability-iterated and needs air temperature, humidity and
#     SST. Air state comes from the same CORE-II files both models read; SST
#     comes from OUR snapshot, so where our SST is biased the stability
#     correction differs for a real reason rather than a coding one.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY - <<'PY'
import numpy as np
import netCDF4 as nc
from legoesm import constants
from legoesm.ocean.bulk_flux_omip import (
    ncar_transfer_coefficients, potential_air_temperature_10m,
    seawater_q_sat, rho_air_moist, exner_potential_temperature)

IN = "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS"
SBC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
       "EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_SBC.nc")
SNAP = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
        "trp_orca1faith_d30/snapshot_final.npz")
R0, R1 = 100, 120          # CORE-II 6-hourly records for days 26-30
REC = 5                    # SBC 5-day record for the same days

def _f(ds, name, sl=None):
    a = ds.variables[name][sl] if sl is not None else ds.variables[name][:]
    a = np.asarray(np.ma.filled(np.ma.masked_invalid(a), np.nan), np.float64)
    return np.where(np.abs(a) > 1e10, np.nan, a)

d = nc.Dataset(SBC)
taum = _f(d, "taum", REC)
taum_oce = _f(d, "taum_oce", REC)
wsp_n = _f(d, "windsp", REC)
nlat = np.asarray(d.variables["nav_lat"][:], np.float64)
nlon = np.asarray(d.variables["nav_lon"][:], np.float64) % 360.0
d.close()

# CORE-II air state and wind, matched window, on the source grid.
du, dv = nc.Dataset(f"{IN}/u_10.15JUNE2009.nc"), nc.Dataset(f"{IN}/v_10.15JUNE2009.nc")
dt, dq = nc.Dataset(f"{IN}/t_10.15JUNE2009.nc"), nc.Dataset(f"{IN}/q_10.15JUNE2009.nc")
un = "U_10_MOD" if "U_10_MOD" in du.variables else "U_10"
vn = "V_10_MOD" if "V_10_MOD" in dv.variables else "V_10"
tn = "T_10_MOD" if "T_10_MOD" in dt.variables else "T_10"
qn = "Q_10_MOD" if "Q_10_MOD" in dq.variables else "Q_10"
print(f"CORE-II variables used: {un} {vn} {tn} {qn}")
u = _f(du, un, slice(R0, R1)); v = _f(dv, vn, slice(R0, R1))
Ta = _f(dt, tn, slice(R0, R1)).mean(axis=0)
qa = _f(dq, qn, slice(R0, R1)).mean(axis=0)
spd = np.sqrt(u * u + v * v).mean(axis=0)
clat = np.asarray(du.variables["LAT"][:] if "LAT" in du.variables
                  else du.variables["lat"][:], np.float64)
clon = np.asarray(du.variables["LON"][:] if "LON" in du.variables
                  else du.variables["lon"][:], np.float64) % 360.0
for x in (du, dv, dt, dq):
    x.close()
if clat.ndim == 1:
    clat = clat[:, None] * np.ones((1, clon.size)); clon = np.ones((clat.shape[0], 1)) * clon[None, :]

s = np.load(SNAP)
sst_K = np.asarray(s["T"], np.float64)[..., 0] + constants.T_freeze
slat = np.asarray(s["lat_T"], np.float64)
slon = np.asarray(s["lon_T"], np.float64) % 360.0

def box(f, la, lo, lon0, dlat=1.0, dlon=2.0):
    m = ((np.abs(la) <= dlat)
         & (np.abs((lo - lon0 + 180.0) % 360.0 - 180.0) <= dlon))
    return float(np.nanmean(f[m])) if m.any() else float("nan")

print("\nEQUATORIAL DRAG COEFFICIENT, |lat|<=1 box means, matched days 26-30")
print(f"{'lon':>5} {'taum':>8} {'|U|':>7} {'rho':>6} {'Cd_NEMO':>9} "
      f"{'Cd_ours':>9} {'ratio':>7}")
ratios = []
for lon0 in (180, 200, 220, 240, 260):
    tm = box(taum, nlat, nlon, lon0)
    tmo = box(taum_oce, nlat, nlon, lon0)
    Un = box(wsp_n, nlat, nlon, lon0)
    Tb = box(Ta, clat, clon, lon0)
    qb = box(qa, clat, clon, lon0)
    Ub = box(spd, clat, clon, lon0)
    sb = box(sst_K, slat, slon, lon0)
    rho = float(rho_air_moist(Tb, qb))
    cd_n = tm / (rho * Un * Un)
    # potential_air_temperature_10m returns (theta_air, p10), NOT a scalar --
    # read the signature, do not infer it. The 10-m pressure it returns is the
    # SAME one NEMO uses for the surface Exner conversion, so it is reused
    # below instead of a hardcoded 101000 Pa.
    th_a, p10 = potential_air_temperature_10m(Tb, qb)
    th_a = float(th_a)
    th_s = float(exner_potential_temperature(sb, p10))
    q_s = float(seawater_q_sat(sb))
    cd_o = float(np.asarray(ncar_transfer_coefficients(
        th_a, qb, th_s, q_s, Ub)[0]))
    ratios.append(cd_o / cd_n)
    print(f"{lon0:5d} {tm:8.4f} {Un:7.3f} {rho:6.3f} {cd_n:9.3e} "
          f"{cd_o:9.3e} {cd_o / cd_n:7.3f}")
    if abs(tmo - tm) > 0.02 * max(abs(tm), 1e-9):
        print(f"      NOTE taum_oce {tmo:.4f} differs from taum {tm:.4f} "
              f"by {100 * (tmo - tm) / tm:+.1f}% -- ice or masking, not drag")

r = np.asarray(ratios, np.float64)
print(f"\n[drag] ratio ours/NEMO over 180-260E: "
      f"min {np.nanmin(r):.3f}  median {np.nanmedian(r):.3f}  "
      f"max {np.nanmax(r):.3f}")
print("[drag] PRE-REGISTERED: >=1.6 across 180-240E means DRAG EXPLAINS the "
      "+97% surface current and confirms the stress ranking at its source; "
      "0.85-1.15 EXONERATES drag and sends the question back to continuity; "
      "between, report and claim no direction. READ THE PER-LONGITUDE COLUMN, "
      "not just the median -- one coastal outlier dragged the last probe's "
      "scalar into the indeterminate band while the structure was decisive.")
print("[drag] rho_air is NOT published by NEMO: the oracle Cd_eff above uses "
      "OUR moist-air density from the CORE-II air state, so a density error "
      "propagates linearly and would be an artifact of this probe, not of "
      "either model. SST for our stability iteration comes from OUR snapshot.")
PY
echo DONE
