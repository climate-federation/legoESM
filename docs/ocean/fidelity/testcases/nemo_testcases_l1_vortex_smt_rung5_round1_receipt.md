# Receipt — SMT-RUNGS round 1: SMT-5 survey and deck proposal (T/S damping)

**Status: STOPPED_FOR_DECISION (survey only; no model code, no NEMO run).**
SMT-5 = SMT-4 + ORCA2 rung 1's module, `ln_tradmp` with `ln_tsd_dmp`. Base:
`31bd4575b`. ORCA2 pointer: every statement below is an ORCA2 rung-1
deliverable (`orca2_hierarchy/rung1`, Decision 80).

## Round 1 — NEMO's damping statement

All citations are the compiled `ppsrc` of the ORCA2 build. The SMT-4 build
holds a byte-identical `tradmp.f90` and `dtatsd.f90` (`cmp` silent on both),
so SMT-5 needs **no rebuild**: the module is already compiled in.

| item | NEMO statement | citation |
|---|---|---|
| RK3 placement | damping is added to the tracer RHS in **stage 3 only** (the only `tra_dmp` call in `stprk3_stg`); called after lateral mixing, before the implicit vertical solve | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:700`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:740` |
| same, SMT-4 build | stage 3: `tra_ldf` (526) → `tra_dmp` (529) → `tra_zdf` (538) | `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:513`, `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:526`, `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:529`, `VORTEX_SMT4_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:538` |
| formula, time level | `Krhs += resto*(T_dta − T(Kbb))`: target at step `kt`, state at **Kbb = N** (step entry), not Kmm; explicit RHS, then `tra_zdf` forms Kaa | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:190`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:194` |
| target read | `dta_tsd(kt)` returns the target at the step, stored in `tclim`/`sclim` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:181`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:184` |
| `nn_zdmp=0` | damp the whole column, shape set only by `resto` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:190` |
| `nn_zdmp=1` | skip cells with `avt > avt_c` (5 cm²/s), uses the vertical-mixing coefficient | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:200` |
| `nn_zdmp=2` | skip cells above the mixed-layer depth `hmlp` (needs `zdfmxl`) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:210` |
| `resto` source | `cn_resto` file, variable `resto` (s⁻¹, 3-D, double), read once at init | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:306` |
| data path forced | `ln_tradmp` forces `ln_tsd_dmp=T` even if the namelist says F | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:302`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:139` |
| time interpolation | `fld_read` at every step, per `sn_tem`/`sn_sal` (frequency, `ln_tint`, `clim`) | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:212` |
| masking | z/zps: target × `tmask`, no vertical interpolation; the s/mixed-s branch (`l_sco`) interpolates vertically and does not apply to the seamount z-partial-step grid | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:308` |
| raw copy before mask | `ptsd = fnow`, comment "NO mask", masked afterwards | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:261` |

`dta_tsd` also contains a hand-made ORCA2-only edit (Alboran/Red Sea
temperature and salinity patches, active for `nn_cfg == 2`, ORCA2). It is
keyed on `cn_cfg == "orca"` and `nn_cfg == 2` in the source, so it is an
ORCA2-only statement (PLAUSIBLE that VORTEX never meets it: its `cn_cfg` was
not printed this round); it belongs to the ORCA2 rung-1 walk, not the seamount.

## What ORCA2 rung 1 resolves to (read from the deck and files)

| item | value |
|---|---|
| `ln_tradmp` / `nn_zdmp` | `.true.` / `0` (rung-1 `namelist_cfg` lines 345-346) |
| `cn_resto` | not set in the cfg → reference default `'resto.nc'` |
| `ln_tsd_dmp` / `ln_tsd_init` | `.true.` / `.true.` (lines 49-50 of the cfg) |
| `sn_tem`, `sn_sal` | `data_1m_potential_temperature_nomask` / `data_1m_salinity_nomask`, freq −1 (monthly), `ln_tint=.true.`, `clim=.true.`, `'yearly'`, no weights, no mask (cfg lines 55-56) |

`resto.nc` statistics (computed from the file; 31×148×180, double, s⁻¹):

| quantity | value |
|---|---|
| global min / max | 0 / 1.1574074e-5 (= 1/86400, a 1-day timescale) |
| fraction of cells > 0 | 2.43 % |
| nonzero columns | 691, at each of levels 2-30; level 1 and level 31 are all zero |
| where | one box, j = 86-109, i = 137-179 (the Mediterranean/Red Sea/Gulf sector) |
| vertical shape | ramps from 1.1e-7 (level 2) up to 1.1574e-5 (level 17), then constant to level 30; column mean at level 17-30 is 1.1356e-5 (5 distinct values) |

So ORCA2 rung-1 damping is a **regional marginal-sea restoring**, 1 day at
depth, zero at the surface level: not a global, not a uniform field. Its
construction is the DMP_TOOLS family (files `med_red_seas.F90`, `coast_dist.F90`,
`zoom.F90`, user hook `custom.F90` under `tools/DMP_TOOLS/src`; output variable
`resto`, double, dims x,y,z, defined at utils.F90 line 111; masked by the
land-sea mask at zoom.F90 line 83).

## Round 1 — legoESM's existing restoring

Searched: `RestoringConfig`, `apply_restoring`, `sponge`, `restor` in
`packages/ocean/legoesm/ocean`. Found two mechanisms, neither selected by the
seamount card.

| mechanism | formula | time level / placement | match to NEMO |
|---|---|---|---|
| sponge relaxation (`SpongeForcing`, `apply_sponge_tracer_relaxation`; `sponge.py` lines 25-54, `ocean_tendency_common.py` lines 654-715) | `gamma*(T_ref − T)·mask`, `gamma` horizontal or full 3-D | evaluated on the step-entry `T,S` (same time level as NEMO's Kbb), added to the tendency `dT_dt` in the C-grid tendency routine (`ocean_pe_latlon_cgrid.py` around line 5948); target is a fixed array, **no time interpolation, no file path** | formula matches `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:194`; full-rank `gamma` (3-D) is the form needed for a `resto(x,y,z)` |
| surface restoring (`RestoringConfig`, `surface_forcing/config.py` line 100) | `(T* − T)/tau`, SST/SSS only, cosine/constant/array target, optional `subtract_qsr`, `implicit` | top layer only | **different statement** (surface only, no column damping) |

Gaps against the NEMO statement (all **PLAUSIBLE** until a card run measures
them; none touched this round):

1. **Card does not select either mechanism.** Caller grep: `sponge` appears in
   `nemo_testcase_recipe.py` only in a comment about AGRIF (line 1882); no
   card sets `sponge_forcing`. SMT-5 would be the first card that does.
2. **RK3 stage placement is unproven.** NEMO adds damping in stage 3 only,
   between lateral mixing and the implicit vertical solve. legoESM adds the
   sponge to the generic tracer tendency; whether the RK3 card path consumes
   that tendency once per step at the stage-3 position (not per stage, and
   before/after lateral mixing consistently with the already-certified
   ordering) must be read from the RK3 tracer update, which this round did not
   do.
3. **Implicit-vs-explicit.** NEMO's damping enters the RHS and the implicit
   `tra_zdf` solve forms Kaa from it. The existing `sponge_forcing_implicit`
   option withholds the sponge from the tendency and routes it to the implicit
   seam (Veros placement). Which of the two reproduces `Krhs` before
   `tra_zdf` is not established.
4. **Time-varying target.** No mechanism reads a monthly climatology with
   `ln_tint`/`clim` interpolation; a constant target needs none, but a 12-record
   file would be sampled differently (see option (d) below).
5. **`nn_zdmp = 1, 2`** (turbocline / mixed-layer exclusion) do not exist in
   legoESM. Not needed for `nn_zdmp = 0`.
6. **Masking.** NEMO multiplies the target by `tmask` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:308`);
   legoESM masks the tendency. Equivalent on wet cells; on partial cells and
   land both give zero tendency (land `T = 0` in both).

## Round 1 — SMT-5 deck proposal (not built)

Deck = the `VORTEX_SMT4_VEC_R8_OMIP_L1` deck (`dx = dy = 30 km`, `dz = 500 m`,
global 63×63×11, z-partial-step seamount) plus, in `namelist_cfg` only:
`ln_tradmp = .true.`, `nn_zdmp`, `cn_resto`, `ln_tsd_dmp = .true.`, `sn_tem`,
`sn_sal`. New target `VORTEX_SMT5_VEC_R8_OMIP_L1` (+ `_P3`); no existing target
is modified. Preflight performed this round (no run): the SMT-4 build already
contains `tradmp`, `dtatsd` and the stage-3 call (identical to ORCA2's, see
above); both namelist blocks exist in the SMT-4 `namelist_ref` (lines 116 and
1012); no `run.sh` was written, so `bash -n` has nothing to check and the
2-step smoke run is the operator's.

**How to produce the data files faithfully.** NEMO writes its own analytical
initial state when `nn_istate = 1`: `dia_wri_state` stores the step-entry
`votemper` and `vosaline` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/diawri.f90:134` and `ORCA2_OMIP_L4/BLD/ppsrc/nemo/diawri.f90:601`) to `output.init.nc`, global size, already masked. Using that file
as the target, copied to the `sn_tem`/`sn_sal` names, means no second
implementation of `usrdef_istate` (Rule 4) and bit-faithful targets. This is
an acquisition (operator runs a 0-step job); it is the first item of round 2.
The restoring field is a uniform or profile constant, so it needs no NEMO
tool: write it with the `resto` variable, double, dims `(z,y,x)`, matching the
global grid, in the layout of ORCA2's file; `custom.F90` is DMP_TOOLS' hook for
exactly this.

### Open deck values (all DECISIONS for the user)

| | option | what it means | pick |
|---|---|---|---|
| (a) | **a1** target = the analytical initial T/S (NEMO's `output.init.nc`) | step 1 damping is exactly zero; the vortex anomaly relaxes back to its initial shape, which is the signal the module must reproduce | **a1** |
| | a2 target = a perturbed T/S | needs a new analytical definition: a new, unreviewed scientific choice | |
| (b) | **b1** uniform `1/86400` s⁻¹ (ORCA2's deep value) on every wet level | strongest test, one day, every cell | **b1** |
| | b2 ORCA2's vertical profile (ramp, zero at the surface level) mapped to the 11 levels by depth | mimics ORCA2's shape, but the mapping to 500 m levels is a further choice | |
| | b3 same as b1 with a weaker timescale (e.g. 10 days) | slower damping, less dominated by the restoring | |
| (c) | **c1** `nn_zdmp = 0` | what ORCA2 rung 1 resolves to; no `avt`/mixed-layer inputs needed | **c1** |
| | c2 `nn_zdmp = 1` or `2` | needs turbocline or mixed-layer diagnostics, untested in the legoESM card | |

Noted, not asked: time interpolation follows ORCA2 (`-1`, `ln_tint=T`,
`clim=T`, `'yearly'`) with 12 identical monthly records, so the NEMO
time-interpolation path stays the same; linear interpolation of equal records
may differ from the record by one rounding step (about 1e-16 relative). If the
user prefers exact constancy, `ln_tint=.false.` is the alternative.

Why b1 and not ORCA2's regional mask: the 691-column mask is Mediterranean/Red
Sea geometry that has no seamount analogue (a hidden choice if invented).
Caveat on b1: a 1-day uniform restoring pulls the whole column to the target
fast, so late-time sensitivity to anything other than the damping statement
itself will be small; b3 keeps more of SMT-4's signal, at the cost of leaving
ORCA2's value.

## Choices made this round

| choice | status |
|---|---|
| citation-gate map entries for the new NEMO citations (gate file only) | ASKED (gate must stay green; no behaviour change) |
| none other; no deck, default, or library change | n/a |

UNASKED list: empty.

## Review, citations

Single review (codex): see the final section of this file's commit message;
if unavailable it reads "independent review unavailable".

Controls: **not run** — no code was added; the only code edit is the citation
map. The gate result is recorded in the commit message.

UNVERIFIED: RK3 placement of legoESM's sponge tendency (gap 2) and the
implicit/explicit question (gap 3) are read from comments and call sites, not
run; the NEMO-side statements are read from the compiled source.

**ACQUISITION_NEEDED: NONE this round.**
