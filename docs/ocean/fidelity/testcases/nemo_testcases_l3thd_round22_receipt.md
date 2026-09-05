# Lane 3b round 22 receipt — portable SI3 scalar-libm sites

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `17bf389a6dbe`

Preregistrations: `c0e42669dac7` and the inventory-triggered snowfall addendum
`94c5a68519aa`.

Implementation/test commits: `ef111aa73a4a`, `f056785ff519`, and
`2312bbc51822`.

## Outcome

Every runtime transcendental in the already-certified ORCA1 SI3 bulk and C1D
thermodynamics paths now uses the one shared precision policy when the card
selects `PrecisionPolicy.fp64(transcendentals="libm")`.  Default `native`
selection is unchanged.  The scalar-math oracle rescore is **271,560 / 271,560
bit-identical** and zero rows exceed the `1e-15` normalized bar.  The result is
CONFIRMED on the registered CPU/glibc stack.

The mandatory native-site inventory found one additional live certified power
after the first implementation: wind-blown snowfall partition.  It was
preregistered before measurement, centralized in the existing snow module, and
routed through the same policy.  Its unpoisoned result did not change a bit.
This is scope completion, not a new physics arm.

The C1D exchange remains **448,950 / 448,950 bit-identical**.  The step and
full-year thermodynamics JSONs, and the coupled-slab first-stop JSON, are
byte-identical to their Round-19/Round-21 counterparts.  The slab's first stop
therefore remains kt2 `PRE_SSM.u`, absolute/normalized
`1.1172865415493005e-7`; it is the GYRE lane's prognostic-`uu_b/vv_b` debt and
was not changed here.

## Search first and one implementation

The search found the canonical selectable implementation in
`packages/core/legoesm/core/transcendentals.py`: native JAX by default and
scalar `libm.so.6` callbacks for fp64 CPU when explicitly selected.  It found
the existing SI3 identities in `core/thermo.py`, `core/bulk_flux.py`, and
`ice/sea_ice.py`, all already guarded statement-by-statement with
`nemo_source_round`.

The post-change inventory also found the same NEMO snowfall statement repeated
in the 2-D bulk and 1-D column callers.  It is now one helper,
`ice/snow.py::nemo_si3_snowfall_fraction`, consumed by both callers.  No second
precision policy, albedo routine, bulk module, snowfall implementation, ice
model, or public Frankenstein selector was added.

## Literal source map

| NEMO 5.0.2 statement | legoESM executing statement | policy call |
|---|---|---|
| `sbc_phy.F90:674-679`, especially `:677` `LOG10(ztmp)` and `:679` `10._wp**zle` | `thermo.py:362-398`, especially `:389,:398` | shared `policy_log10`, `policy_pow` |
| `sbc_phy.F90:331-335`, `(rpref/ppa)**rgamma_dry` | `core/bulk_flux.py:1541-1598`, especially `:1572,:1591` | shared `policy_pow` |
| `icealb.F90:124-130,159-160`, the pivot, thin-break, and thickness `LOG` calls | `ice/sea_ice.py:500-550`, especially `:519,:538,:541,:549` | shared policy `log` |
| `icevar.F90:1619-1628`, `1._wp-pin**rn_snwblow`, called by `sbcblk.F90:1294` and `icethd_dh.F90:98` | `ice/snow.py:65-76`, called by `sea_ice.py:724-728` and `bitz_lipscomb.py:730-733` | shared `policy_pow` |

The `icealb.F90:167-175` EXP statements were already routed through the shared
policy.  `sbc_phy.F90:709-711` uses the compile-time-folded, bit-pinned
`constants.ln10_nemo`; it does not introduce a runtime LOG.  All new calls stay
inside the existing source-statement rounding boundaries.

## Edge semantics and row-binding controls

The binary POW domain test now includes libc `0**0`.  Eager/JIT policy output
matches live `libm.so.6` bytes and the explicitly pinned result is
`0x3ff0000000000000` (`1.0`), at
`tests/unit/test_transcendentals.py:157-166`.  The existing NaN and +Inf libc
domain cases and the second-order JVP tests remain green.

Five private, non-card hooks poison one newly routed result at a time.  Each
propagates to scored output, raises the named `GateError`, and exits nonzero:

| plant | NEMO site isolated | over-bar rows | CLI exit |
|---|---|---:|---:|
| `libm_si3_albedo_log` | `icealb` LOG interpolation | 5,271 | 1 |
| `libm_si3_exner` | surface Exner POW | 35,040 | 1 |
| `libm_si3_saturation_log10` | Goff-ice LOG10 | 87,600 | 1 |
| `libm_si3_saturation_pow` | Goff-ice POW | 87,600 | 1 |
| `libm_si3_snowfall_pow` | wind-blown snowfall POW | 28,757 | 1 |

The albedo poison is input-dependent because an additive constant would cancel
inside the log interpolation ratio.  These hooks are private probe arguments;
no partial policy combination is constructible from an ice card.

## Measured gates

Runtime: Python 3.13.0, JAX/jaxlib 0.10.0, NumPy 2.4.4, CPU, binary64,
production JIT where used, and explicit `transcendentals="libm"` on the C1D
cards.  AT_BAR and bit identity are distinct below.

| gate | result | classification |
|---|---|---|
| scalar-math C1D SI3 bulk, full year | 271,560 / 271,560 bit-identical; 0 non-bit; 0 over-bar | BIT_IDENTICAL and AT_BAR |
| C1D ice/ocean exchange, full year | 60 registered fields; 448,950 / 448,950 bit-identical; first over-bar absent | measured prefix BIT_IDENTICAL and AT_BAR |
| C1D thermodynamics kt1 | report byte-identical to prior, SHA-256 `ada00ca0058052a2c39622b9dca2a026dca8263fce19439d633c6bdeba0eb251` | no movement; closed MIXED DEBT unchanged |
| C1D thermodynamics year | report byte-identical to prior, SHA-256 `9a233a2cb6647448810c41c20b3d40a855c9f9fadf2f4a24cfbd5024453a3ae1` | no movement; closed MIXED DEBT unchanged |
| C1D coupled slab | report byte-identical; first stop kt2 `PRE_SSM.u=1.1172865415493005e-7` | existing GYRE-owned DEBT unchanged |
| LOCK faithful stage + kt1..10 | residuals unchanged; stage remains AT_BAR; trajectory first over-bar remains kt4 `u` | zero movement, not a new identity claim |
| OVERFLOW faithful stage + kt1..10 | residuals unchanged; existing stage debts; trajectory first over-bar remains kt2 `T,u` | zero movement, existing DEBT |
| GYRE production-JIT kt1..10 | all 50 compared trajectory rows unchanged; first over-bar remains kt2 `T,S,u,v` | zero movement, existing DEBT |
| ORCA2 kt1 admission | PASS; no numerical operator executes at this boundary | no SI3-transcendental row to move |

The exchange and slab were rerun after the snowfall helper was centralized;
their new reports have exactly the same hashes as the pre-addendum reports.
LOCK, OVERFLOW, and GYRE do not import or execute any of the changed SI3 sites.
Their post-main-implementation Rule-12 comparator reports zero changed residual
bits and zero worsening ULPs, so rerunning those ocean-only cards after the
snowfall-only addendum would not exercise the changed call graph.

## Rule 8/11/12 register

| boundary | moved rows | owner and disposition |
|---|---:|---|
| SI3 bulk `ice_alb` / `blk_ice_2` | 0 / 271,560 | lane 3b; portable scalar-libm identity CONFIRMED |
| C1D ice/ocean exchange | 0 / 448,950 | lane 3b; CONFIRMED unchanged |
| C1D thermodynamics step/year | 0 report bytes | prior MIXED DEBT retained; no attribution retracted |
| C1D slab through ordered kt2 stop | 0 report bytes | prognostic `uu_b/vv_b` remains GYRE-owned; slab seed untouched |
| LOCK/OVERFLOW/GYRE cross-card registers | 0 residual bytes; 0 worsening ULP | GYRE shared owner; no Rule-12 regression |

No ownership label is inferred from a nonexecuting card.  No previous finding
is retracted in this round.

## Remaining native transcendental debt

The raw post-change search is retained as
`round22_si3_libm/native_log_power_inventory.txt`.  The certified SI3 LOG,
LOG10, and fractional POW sites above have been removed from the debt list.
The remaining in-scope-looking native sites are deliberately not converted
without a measured operator gate:

| shared path | remaining native sites | status |
|---|---|---|
| ORCA2 internal-wave mixing | `internal_wave_mixing.py` LOG10/TANH and EXP identities corresponding to `zdfiwm.F90:170-205,237-242` | selected by ORCA2; full operator UNMEASURED |
| RGB shortwave | `shortwave_penetration.py` LOG10/LOG class/profile calculations; EXP was previously routed | ORCA2 entry gate does not execute it; UNMEASURED |
| NEMO EOS dynamic enthalpy | `ocean/eos.py` LOG integrations | outside this round's certified paths; UNMEASURED |
| generic Goff-water and other bulk schemes | native sites in generic saturation, ECMWF/COARE/SAM/MOST arms | non-certified arms; no portability claim |

This inventory is DEBT, not evidence that the sites differ.  No ORCA2 IWM,
RGB, EOS-enthalpy, or non-NEMO bulk implementation was changed.

## ORCA1 selector coverage

The complete selector table in the Round-21 receipt remains current.  The C1D
rung covers the ORCA1-resolved single-category thermodynamic identity:
`jpl=1`, HFN, `nlay_i=3`, `nlay_s=3`, BL99+P07, `nn_icesal=2` with
`rn_sinew=0.75`, flushing/drainage, `ln_icedA=F`, and `ln_pnd=F`.  It does not
cover dynamics/transport/ridging/rafting/landfast.  No five-category or ORCA2
SI3 work was started while the user scope decision remains pending.

## Tests and retained evidence

The final focused selection collected 82 tests and reports **82 passed in
121.36 s**.  It covers scalar transcendental values/JVPs, all SI3 bulk plants,
the rung-3.6 exchange, and the step/year thermodynamics gates.  The hardcoded-
constant ratchet was invoked on every Python source/test touched this round and
reports **8 passed in 0.84 s**.  The earlier gate-only `273.15` friction-probe
literal was replaced by `constants.T_freeze`; it is not an ice-physics change.

All runtime evidence is retained under
`/data/abyssal/dbalwada/nemo-testcases-l3/round22_si3_libm/`; no runtime output
is committed.

| artifact | SHA-256 |
|---|---|
| `si3_bulk_snowpow_final.json` | `4c34d43f5b35a024509d1f9739a45b2f784ab4c60a27c10f2d818dc5db181c7d` |
| `si3_bulk_plant_snowpow_final.log` | `51da1cc86f14c2c17d5979b7378da2e31bc2d97156b559325caac2b077e9e7c8` |
| `c1d_exchange_snowpow.json` | `aef5fb7dda64e37981904ac232ca92b5ed32d877b6b2aac23e2e6a17f2e99ca7` |
| `c1d_slab_year_snowpow.json` | `cac519825356a2230fc19a964e7e91c2566c4eefaef9532b22feb78807334f14` |
| `si3_thd_step_snowpow.json` | `ada00ca0058052a2c39622b9dca2a026dca8263fce19439d633c6bdeba0eb251` |
| `si3_column_year_snowpow.json` | `9a233a2cb6647448810c41c20b3d40a855c9f9fadf2f4a24cfbd5024453a3ae1` |
| `crosscard_rule12.json` | `d86e745b18ecc3178ef4bfd6627f43147f7b2033e09008bd7094c1fbd18e49c0` |
| `gyre_rule12.json` | `da54b41f97707ce550e0b6e1b86539ddb3c5e9cbafce2e6619ff81de61120c34` |
| `focused_pytest_snowpow_final.log` | `074df8dfcae291e8ed2473e8837d3b75855ad3b597df33ec8b4805d4b2c9a9ca` |
| `constants_ratchet_touched_final.log` | `80717ff2fb0309c115244858510312abb8897aab740670d7f408ba42f4ea79d2` |
| `native_log_power_inventory.txt` | `2f576e057f421032fa4efbb769f9f4fd3ec4ef93417ccda26490c16240c86dbf` |

The four other final poison logs are also pinned: albedo
`ea42aff97af05da279dd5f7cb9e9de85f34319f83ddd001b32a332e2fc4ef931`,
Exner `f90c2f8828b624ba849b40c47b2b1bb7c12936c02fbfdf0adad7cb6760e940aa`,
saturation LOG10
`ff9f294f666262ae8f4201030886c016cdf52900956560145f81ff1410facd7d`,
and saturation POW
`df3974327ed260af6af86723cb5c945287135a07a6dfa62be9a7932bc71ce819`.

Re-flag the retained Round-18 diagnostic trees, the Round-21 ORCA2 diagnostic
worktree, the eight `c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}`
roots, and failed `c1d_omip_l3_coupled10m_r17_oracle_a` for future deletion.
Delete nothing.

No shipped NEMO file, retained oracle, or forcing input was modified.  No GPU,
`mpirun`, synthetic input, multi-MB repository artifact, push, multi-category
ice work, or prognostic `uu_b/vv_b` work occurred.  The user-relayed review of
Round 21 was SHIP; this round creates no independent review artifact and calls
its own checks Codex-internal.

## End-of-task ASKED / UNASKED register

| choice/action | status | disposition |
|---|---|---|
| route the certified SI3 saturation, Exner, and albedo sites through scalar libm | ASKED | implemented and 271,560 / 271,560 confirmed |
| complete the certified-path inventory, including snowfall POW | ASKED-scope completion | preregistered separately; one shared helper; no bit movement |
| add libc `0**0` bytes | ASKED | exact `1.0` bytes pinned |
| add per-site poisoned-policy controls | ASKED | five row-binding nonzero exits |
| remeasure SI3 bulk, exchange, slab, thermodynamics, and nonexecuting cross-cards | ASKED | completed; no unpoisoned row moved |
| convert ORCA2 IWM, RGB, EOS enthalpy, or non-certified bulk sites | UNASKED | retained as measured-scope debt inventory |
| implement multi-category SI3 or prognostic `uu_b/vv_b` | user decision / GYRE owned | not started |
| alter/delete shipped NEMO or retained evidence, use GPU/`mpirun`, or push | forbidden | not done |
