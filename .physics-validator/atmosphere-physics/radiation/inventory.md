# Radiation inventory

Scope: `src/legoesm/atmosphere/physics/radiation/`. Two backends: `gray` (Frierson) and `rrtmgp` (full correlated-k). Subdir `rrtmgp/` is large (~6500 LOC) — only top-level audited here.

## Top-level files

* `gray.py` (363 LOC): two-stream gray LW + Beer-Lambert SW.
* `solar.py` (203 LOC): solar declination, zenith angle, daily-mean and equinox insolation.
* `rrtmgp_radiation.py` (~140 LOC): wrapper over `rrtmgp/` for column-physics interface.
* `integration.py` (824 LOC): factory dispatching to gray or RRTMGP, building forcing arrays (ozone, aerosol).
* `config.py`: `RadiationConfig`, `GrayRadiationConfig`, `RRTMGPConfig`.
* `output.py`: `RadiationOutput` NamedTuple.

## RRTMGP subdir (`rrtmgp/`, deferred)

Imported from `climt`/`rrtmgp_jax` lookup tables. ~6500 LOC across `optics/`, `rte/`, `interpolation.py`, `kernel_ops.py`, `rrtmgp_common.py`. **Not deeply audited** in this pass — too large for one cycle.

## Output (`RadiationOutput`)

* `lw_flux_up`, `lw_flux_down`, `sw_flux_up`, `sw_flux_down` [W/m²], shape `(ncol, nlev+1)`.
* `heating_rate`, `lw_heating_rate`, `sw_heating_rate` [K/s], shape `(ncol, nlev)`.

## Convention

* `p_half[0]` = TOA, `p_half[-1]` = surface.
* Fluxes positive upward / downward as per name.
* `dT_dt = (g/c_pd) * (F_net_up_below - F_net_up_above) / dp` ← layer integral form.

## Constants

* `S_0 = 1361.0 W/m²` in `constants.py`. **`solar.py` defaults `S_0 = 1360.0`** ⇒ minor drift, see static analysis.
* `obliquity` default 23.45° hardcoded in solar functions; not in `constants.py`.
