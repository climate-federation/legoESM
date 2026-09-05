# Lane 3b round 21 receipt — scalar-libm LOG/LOG10/POW

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `17693b92997f4c70a508803876251286aa8800d7`

Preregistration: `351fb8e8139`

Implementation/test commits: `500c303ac9e`, `ace76814e68`, `ea59a7385af`,
and `e1a10dcc1df`.

## Outcome

User Decision 9 is implemented in the one shared precision-policy module.
With `PrecisionPolicy.fp64(transcendentals="libm")`, `log`, `log10`, and
binary `pow` now call scalar `libm.so.6` through `jax.pure_callback`; native
remains the default and still calls JAX.  The certified NCAR O1 result remains
**0 / 158,292 non-bit rows**, and the C1D SI3 bulk result remains
**271,560 / 271,560 bit-identical**.  These are CONFIRMED measurements on the
registered CPU/fp64 stack, not portability claims about a different libc.

No cross-card oracle-relative residual moved: the retained LOCK, OVERFLOW, and
GYRE ten-step residual arrays are bit-identical to their Round-19 arrays; the
C1D slab first registered stop has the same bits; the ORCA2 kt=1 entry report
is byte-identical to its Phase-2g report.  Therefore there is no first moved
cross-card row to assign.

## Search first and one implementation

The pre-implementation search found the canonical callback machinery in
`packages/core/legoesm/core/transcendentals.py`: scalar ctypes calls to
`libm.so.6`, `jax.pure_callback`, CPU admission, and custom JVPs for
EXP/TANH/SIN/COS.  It found the already-certified NCAR arm in the one shared
`packages/core/legoesm/core/bulk_flux.py`; no second bulk module or precision
module was created.  `precision.py` is unchanged from the parent because its
existing `transcendentals="native"|"libm"` selector is sufficient.

The extension is at `transcendentals.py:34-101,155-191,226-249`.  Unary
callbacks preserve the input floating shape/dtype.  The binary callback first
broadcasts its two operands, preserves that shape/dtype, and calls scalar C
`pow` once per element.  The JVPs are `dx/x`, `dx/(x*libm_log(10))`, and both
real `pow` partials.  No input clipping or finite-value substitution is made;
the pinned domain test observes the same NaN/+Inf bytes as libc and non-finite
tangents at those non-smooth points.  Non-CPU use raises.

## NCAR source statement map

Every live LOG/LOG10/POW call in the certified arm is now inside the existing
`nemo_source_round` boundary:

| NEMO statement | shared executing site after this round |
|---|---|
| Goff water saturation `LOG10` and `10**x`, `sbc_phy.F90:645-651` | `_nemo_goff_water_saturation_pressure`, `bulk_flux.py:1616-1647` |
| three pressure iterations and `EXP`, `sbc_phy.F90:256,272-277` | `nemo_ncar_pressure_at_height`, `:1660-1704` |
| Exner power, `sbc_phy.F90:321-337` | `nemo_ncar_theta_exner`, `:1707-1714` |
| unstable momentum logs, `sbcblk_algo_ncar.F90:312-326` | `_nemo_ncar_psi_m`, `:1787-1807` |
| unstable scalar log, `sbcblk_algo_ncar.F90:350-363` | `_nemo_ncar_psi_h`, `:1809-1822` |
| `z0_from_Cd` / `UN10_from_CD` EXP+LOG called at `sbcblk_algo_ncar.F90:176-200` | `_nemo_ncar_un10`, `:1845-1856` |
| 10 m height correction logs, `sbcblk_algo_ncar.F90:169-205` | `nemo_ncar_transfer_coefficients`, including `:1902` |

The policy call changes the transcendental provider only.  The Fortran
left-to-right association and one `nemo_source_round` per source statement
remain those certified in Round 18.

## Scalar-library tests and controls

Pinned uint64 results cover six arguments each for LOG, LOG10, and POW and are
also compared to direct ctypes calls on this host.  Eager and JIT results have
identical bytes.  `jax.test_util.check_grads(order=2)` passes for LOG, LOG10,
and both POW partials on positive finite inputs.  Native selection is tested
separately and remains unchanged.

Final focused command: 95 passed in 21.63 s.  The exact log is
`round21_libm/focused_pytest_final2.log`, SHA-256
`fb6f4beb2fef665b8def141b8fd763382a59d1e17c269d8a8320dbd34c68bae5`.
The hardcoded-constant ratchet run against each of the five touched source/test
files is **5 passed**; its log SHA-256 is
`a1dd2bdcd8a4ad1dcc379822601bbe3422909ab45fc824c4d8f77635ad88e89d`.
A repository-wide ratchet run collected 4,553 cases and reported six existing
failures in files unchanged by this round; the five Round-21 paths are not
among them.  This is not described as a green repository-wide ratchet.

The ordinary NCAR gate internally ran all 18 VERIFIED-field one-ULP plants;
every one returned `PASS_NONZERO` with exit code 1.  The new private poisoned
LOG callback is stronger than one ULP so its value survives subsequent source
rounding: it makes 70,136 / 158,292 rows non-bit and 67,221 rows over-bar; the
CLI exits 1.  The inherited SI3 `libm_return` plant also exits 1 and makes
13,468 rows over-bar (albedo, `qsr_ice`, `qsr_tot`).  These controls bind to
scored rows.

## Certified gates

Runtime: Python 3.13.0, JAX/jaxlib 0.10.0, NumPy 2.4.4, CPU, binary64,
production JIT where the gate uses it, explicit `transcendentals="libm"` on
the cards.  The NEMO scalar-math oracles remain unchanged.

| gate | measured result | bit / AT_BAR distinction |
|---|---|---|
| ORCA2 O1 NCAR bulk | 18 VERIFIED fields; `0 / 158,292` non-bit; `0 / 158,292` over-bar | BIT_IDENTICAL and AT_BAR |
| C1D SI3 bulk | `271,560 / 271,560` bit-identical; zero over-bar | BIT_IDENTICAL and AT_BAR |
| LOCK stage | 9 faithful rows, 22,880 cells, 489 non-bit; max normalized `8.674e-19`; no failed row | AT_BAR, not bit-identical |
| LOCK trajectory kt=1..10 | residual archive identical to Round 19; first over-bar remains kt4 `u` | zero movement; existing DEBT unchanged |
| OVERFLOW stage | all 9 row metrics identical to Round 19; existing four failed rows remain | zero movement; existing DEBT unchanged |
| OVERFLOW trajectory kt=1..10 | residual archive identical to Round 19; first over-bar remains kt2 `T,u` | zero movement; existing DEBT unchanged |
| GYRE production-JIT kt=1..10 | all 150 residual arrays identical to Round 19; first over-bar remains kt2 `T,S,u,v` | zero movement; existing DEBT unchanged |
| C1D coupled slab | first registered stop remains kt2 `PRE_SSM.u = 1.1172865415493005e-7` | same bits; existing prognostic-`uu_b/vv_b` DEBT unchanged |
| ORCA2 kt=1 entry | T/S/u/v each `0 / 399,600`; ssh `0 / 13,320` | byte-identical report; no operator executed |

The LOCK and OVERFLOW residual archives are compared cellwise to their
Round-19 residual archives.  GYRE is production JIT.  Row-scale ULP columns
remain in the gate JSON; “zero movement” means the before/after residual
arrays have identical bytes, not that every model-vs-oracle row is itself
bit-identical.

## Rule 8/12 register and live-call audit

| card/boundary | new LOG/LOG10/POW executed before the score? | movement | owner/disposition |
|---|---|---:|---|
| NCAR O1 bulk | yes: all statement families mapped above | 0 rows / 0 ULP | lane 3b, CONFIRMED certified |
| C1D SI3 bulk | no newly routed call; its native LOG sites happen to agree on this stack | 0 rows / 0 ULP | unchanged |
| LOCK/OVERFLOW WS-RK3 | no | 0 residual bits | GYRE shared owner; no Rule-12 regression |
| GYRE kt=1..10 | no | 0 residual bits | GYRE shared owner; no Rule-12 regression |
| C1D slab first-stop walk | no | 0 bits at the compared register | prognostic `uu_b/vv_b` remains GYRE-owned Decision 8 |
| ORCA2 kt=1 entry | no numerical operator runs; card construction uses the unchanged SIN policy | 0 report bytes | Lane 4 entry identity unchanged |

The resolved ORCA2 oracle does select NCAR (`ln_NCAR=T`), TKE
(`ln_zdftke=T`, `nn_eice=1`, `nn_etau=1`) and internal-wave mixing
(`ln_zdfiwm=T`, differential mixing true).  A later full ORCA2 step would
therefore execute NCAR LOG/LOG10/POW, TKE TANH at `zdftke.F90:255`, TKE EXP at
`:492-500`, IWM EXP at `zdfiwm.F90:170-205`, and IWM LOG10+TANH at `:237-242`.
The ORCA2 gate run here is explicitly only the kt=1 entry boundary, so none of
those operators executes and none is certified by that entry result.

## Remaining NEMO-identity transcendental debt

The post-change repository search is retained as
`round21_libm/native_jnp_log_power_sites.txt`.  Excluding generic non-NEMO and
integer-power polynomial sites, the still-native NEMO identities are:

| shared path | native site | status |
|---|---|---|
| SI3 no-pond albedo | `ice/sea_ice.py:535-546`, LOG interpolation | certified rows happen to be bit-identical on this stack; not policy-routed |
| SI3 snowfall partition | `ice/bitz_lipscomb.py:729`, `**rn_snwblow` | C1D identity measured; provider remains native |
| SI3 pressure/Exner helpers | `ice/sea_ice.py:662,715,1122` | not changed this round |
| SI3 saturation over ice | `core/thermo.py:176,383`, LOG10/`10**x` | certified rows happen to be exact; not policy-routed |
| ORCA2 internal-wave differential mixing | `ocean/.../internal_wave_mixing.py:361-364`, LOG10 and TANH; EXP at `:276,292-293` | selected by ORCA2, but full operator remains unmeasured |
| RGB shortwave class/profile | `ocean/.../shortwave_penetration.py:307,338,548`, LOG10/LOG | entry gate does not execute it; EXP was already policy-routed |
| NEMO EOS dynamic enthalpy integrations | `ocean/eos.py:2218-2219,2292,2323,2440,2462`, LOG | outside this round's certified paths |
| other shared bulk schemes | `core/bulk_flux.py:248,385-646,968-1275,1481-1491` | ECMWF/COARE/generic MOST arms, not the certified `nemo_ncar` span |

This inventory is DEBT, not a claim that every named arm differs numerically.
No opportunistic conversion was made.

## ORCA1 selector coverage versus the C1D rung

The target remains the user's ORCA1 deck, not ORCA2's five-category identity.
Source values are `/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg`;
C1D values are the resolved overlay in
`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg`.

| selector | ORCA1 | C1D resolved/executed | coverage by certified C1D rung |
|---|---:|---:|---|
| `jpl`, `ln_cat_hfn` | `1`, T (`:24,:37`) | `1`, T (`:24,:34`) | COVERED: single-category HFN |
| `nlay_i`, `nlay_s` | `3`, `3` (`:25-26`) | `3`, `3` (`:25-26`) | COVERED |
| `ln_zdf_BL99`, `ln_cndi_P07` | inherited T/T | T/T (`:98-100`) | COVERED |
| `nn_icesal`, `rn_sinew` | `2`, `0.75` (`:108,:116`) | `2`, `0.75` (`:119,:123`) | COVERED; adjacent ORCA1 comment inconsistency remains a FINDING |
| `ln_flushing`, `ln_drainage` | T/T (`:113-114`) | T/T (`:120-121`) | COVERED |
| `ln_icedH`, `ln_icedO` | inherited T/T | T/T (`:90,:92`) | COVERED |
| `ln_icedA` | F (`:87`) | F (`:91`) | COVERED only as the selected off identity; lateral-melt-on is out of scope |
| `ln_pnd` | F (`:151`) | F (`:132`) | COVERED only as the selected off identity; ponds-on is out of scope |
| `nn_qtrice` | inherited `0` | `0` (`:85`) | COVERED |
| `ln_dynALL`, aEVP | T, T (`:44`; aEVP resolved under EVP) | named T/T but skipped by `ln_c1d` | UNCOVERED dynamics |
| `ln_adv_Pra` | T (`:75`) | T but skipped by `ln_c1d` (`:73`) | UNCOVERED transport |
| `ln_ridging`, `ln_rafting` | T/T (`:61-62`) | T/T but skipped (`:56-57`) | UNCOVERED mechanical redistribution |
| `ln_landfast_L16` | T (`:46`) | inherited F and dynamics skipped | UNCOVERED |

No multi-category, 10+5-layer, salinity-option-4, pond, lateral-melt, or
landfast implementation was started while the user's SI3 scope decision is
pending.

## Provenance and retained artifacts

All runtime evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l3/round21_libm/`; no runtime JSON/NPZ
is committed.

| artifact | SHA-256 |
|---|---|
| `ncar.json` | `14a4585a514f20b2d67b7be199359fcf91fc461960a38700fc92bbdd86d36b59` |
| `ncar_plant_log.json` | `66c4d7c475d3efc0ee76c9071581d06dff7846ae88e33e9dbfad3a34bf171f61` |
| `si3_bulk.json` | `9c6040ef1824c5d1c1a9a261afd273358e55dbe07455bf17b31739e87d4e5712` |
| `si3_bulk_plant_libm.log` | `1293c678c178290fdba0744fd4277b3a42d54bfab10f0095abd3a7b416b0e6c7` |
| `lock_stage.json` | `ff05d6c0b6bed1bb7756fe976cc863178e40836aab93c70f620316b73c8f4a0a` |
| LOCK kt10 residual NPZ | `83980ff8572b497e0c22908d82cfc08fdcb4c8db8db34a5ec620ab01687cb31e` |
| `overflow_stage.json` | `81e4be4b3cccda0097b6698acaafcf5ed95e564deac595c6e56edc197a59642a` |
| OVERFLOW kt10 residual NPZ | `5ed7ea7bd8d1420bac286bc045e670efe8612d6ffcba1b4eb1c10effb9ab6b2a` |
| GYRE kt10 residual NPZ | `0ec022469c778ea05bdb3fdde36b74af7587bba92707cdded818cf70635c0c61` |
| `c1d_slab_year.json` (ordered first-stop report) | `cac519825356a2230fc19a964e7e91c2566c4eefaef9532b22feb78807334f14` |
| `orca2_entry.json` | `893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12` |

The ORCA2 diagnostic worktree `/tmp/codex-si3thd-r21-orca2` is retained and
FLAGGED FOR FUTURE DELETION.  Re-flag all Round-18 diagnostic worktrees, the
eight `c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}` roots, and
failed `c1d_omip_l3_coupled10m_r17_oracle_a`.  Delete nothing.

No shipped NEMO file, retained oracle, or input was modified.  No GPU,
`mpirun`, synthetic forcing, multi-MB repository artifact, push, BBL default
change, or prognostic `uu_b/vv_b` implementation occurred.  User Decision 10
is preserved at `bbl_adv_option=0`, `bbl_gamma_s=0.0`; there is no silent on.

## End-of-task ASKED / UNASKED register

| choice/action | status | disposition |
|---|---|---|
| extend scalar-libm to LOG/LOG10/POW | ASKED, User Decision 9 | implemented in the canonical module |
| route every live certified NCAR site | ASKED | complete; 0 / 158,292 confirmed |
| cross-card LOCK/OVERFLOW/GYRE/C1D/ORCA2 entry register | ASKED | zero movement measured |
| retain BBL defaults `0 / 0.0` | ASKED, User Decision 10 | unchanged |
| implement prognostic `uu_b/vv_b` | GYRE-owned User Decision 8 | not implemented |
| implement ORCA2 multi-category SI3 | user decision pending | not started |
| convert remaining native NEMO-identity sites | UNASKED | named as debt only |
| alter/delete shipped NEMO or retained evidence, use GPU/`mpirun`, push | forbidden | not done |

All reviews mentioned here are Codex-internal unless a branch artifact names
an independent reviewer, reviewed commit, and verdict.
