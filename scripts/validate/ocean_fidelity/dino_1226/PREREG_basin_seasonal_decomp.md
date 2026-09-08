# PRE-REGISTRATION — the southern-basin gap, decomposed by season, latitude row and depth

Written BEFORE `basin_seasonal_decomp.py` produced a single number.
Lane: `fidelity/dino-basin-budget-1yr`. Parent verdict:
`docs/ocean/fidelity/dino_verdict360_result.md` (#1455, commit `a1387f1f7`).

This is the OFFLINE half of the basin-budget mission. It re-uses the verdict
run's eight member states already on disk; it launches nothing and costs no
GPU-hours. It is a NAMING lane: no fix is built here.

## The fingerprint any candidate must reproduce

From the verdict run (legoESM minus NEMO, south-of-band transport, mean
reducer):

| run day | day of year | gap [Sv] |
|---:|---:|---:|
| 10 | 190 | +0.03 |
| 90 | 270 | −0.43 |
| 180 | 0 | −0.23 |
| 270 | 90 | −0.06 |
| 300 | 120 | −0.23 |
| 330 | 150 | −0.60 |
| 360 | 180 | −0.95 |

Non-monotonic, with a near-complete recovery at day 270 and a steep final
quarter. Run day 0 is day-of-year 180 (`seasonal_t0_seconds = 15552000`),
i.e. 1 July on DINO's 360-day calendar — austral midwinter.

## Q1 — does the gap track a seasonal forcing phase?

**A fact to be CHECKED here, not assumed.** DINO's `nn_forcingtype = 4`
(`RUN_VERDICT360_M0/namelist_cfg:28`) builds the zonal wind stress from a
fixed node list with NO seasonal cosine (`usrdef_sbc.F90:162-163, 221`), and
`rn_emp_prop = 0` with `ln_emp_field = .false.` zeroes E−P. If that reading is
right, the ONLY seasonal channels are the solar flux (21-June cosine `c1`) and
the restoring target temperature (21-July cosine `c2`); a wind-phase
explanation is then structurally impossible and must not be offered.

* **Expected**: the gap is an amplitude that grows times a seasonal shape.
* **CONFIRM seasonal**: after removing a linear-in-time envelope, the residual
  gap correlates with a buoyancy-phase channel at |r| ≥ 0.7 over the 19
  matched days, with the extremum in austral winter.
* **REFUTE seasonal**: |r| < 0.3 against every available channel, or the gap
  is monotonic within the measured floor.
* Correlation over 19 samples is reported with its two-sided p-value; a
  correlation is a CORRELATION and will be labelled PLAUSIBLE, never a cause.

## Q2 — does the deficit live in the same latitude rows all year?

Per-T-row transport gap inside the south group at days 90/180/270/360.

* **CONFIRM "same rows"**: Pearson r ≥ 0.7 between the day-90 and day-360 row
  profiles, each normalised by its own sum.
* **REFUTE**: r < 0.3, i.e. the deficit migrates.

## Q3 — barotropic or baroclinic?

The south group's transport split exactly into a bottom-referenced barotropic
part and the shear above it (the split `acc_thermal_wind.bc_bt_band` performs
on the channel band, here generalised to arbitrary rows).

* **CONFIRM "barotropic-owned"**: |Δbt| ≥ 2·|Δbc| at day 360.
* **CONFIRM "baroclinic-owned"**: |Δbc| ≥ 2·|Δbt| at day 360.
* Otherwise: **shared**, and neither label is used.

## Self-checks that must pass before any number above is read

Each of these CAN fail; none is an identity.

1. The rows-general barotropic/baroclinic split reproduces
   `acc_thermal_wind.bc_bt_band` to 1e-12 Sv when handed the channel-band rows.
2. The three latitude groups sum to the full-section mean-reduced transport
   (`acc_driver_decomp`'s own assertion, at every scored day on both sides).
3. The day-90 and day-360 south-of-band gaps reproduce the verdict run's own
   recorded −0.42578 and −0.95191 Sv, and its recorded day-90 / day-360 member
   floors, to 1e-4 Sv.

   **RETRACTION, logged in place rather than silently corrected.** This line
   originally registered −0.4405 for day 90, a number taken from a code
   docstring. That value belongs to a *different* run — the 90-day gate2 twin —
   and the check as first written compared two protocols. The probe's own gate
   failed on it, which is what the gate is for, and the constant was replaced
   with the value printed by the verdict run itself. The fingerprint table
   above always said −0.43 and is unaffected.
4. The seasonal cosines evaluated here reproduce the oracle's phase: `c1`
   maximal at day-of-year 171, `c2` at 201, both to within one sample.
5. Every NEMO restart day used is present on disk for all four members; a
   missing day is fatal, never skipped.

## What this probe may NOT conclude

It measures where the gap LANDS in space, depth and season. It cannot name the
operator that puts it there — that is the accumulated stage budget, which is a
separate pre-registration. Any mechanism sentence produced from this probe
alone is PLAUSIBLE by construction.
