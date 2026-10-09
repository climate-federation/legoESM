# Receipt — SMT-RUNGS round 5: SMT-6 / SMT-6b build (BBL + geothermal)

**Status: STOPPED_FOR_RECORD.** Decks, NEMO-side patches, acquisition and the
legoESM arms are built and committed; the two NEMO records are the operator's
(`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smtrungs_round5_smt6/run.sh`
`--smoke`, then `--run`). Decision 110 fixes every deck value except the
SMT-6b anomaly's shape and amplitude, which are offered for amendment below.
Base `2fa05b11c`. Preregistration
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smtrungs_round5_smt6.md`
(committed before any record exists). Evidence
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round5/`.

**ORCA2 pointer.** This is ORCA2 rung 2's exact program for the BBL
(`nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl = 1000`, `rn_gambbl = 10`,
`orca2_rung2/namelist_cfg:285`, `:286`, `:287`, `:288`, `:289`) and for the
geothermal heating statement (trabbc lines 158-159). Two differences, both
named: ORCA2 reads its flux from a file (`nn_geoflx = 2`,
`orca2_rung2/namelist_cfg:274`: one more read statement and a 1e-3 unit factor,
trabbc lines 242-243) and runs TEOS-10, so ORCA2's gate takes the polynomial
expansion coefficients, not the S-EOS ones added here. The ORCA2 card's own
BBL chain is unchanged by this round (digest below).

## Round 5 — 1. Decks (namelist diffs, as the preflight prints them)

SMT-6 = SMT-5 + the one hunk `namelist_cfg_smt6_bbl_geothermal.patch`
(after SMT-5's damping block):

```
+&nambbc
+   ln_trabbc     = .true.    (ORCA2 rung-2 cfg:273)
+   nn_geoflx     = 1         (Decision 110a)
+   rn_geoflx_cst = 86.4e-3   (Decision 110a; used as W/m2)
+&nambbl
+   ln_trabbl  = .true.       (cfg:285)
+   nn_bbl_ldf = 1            (cfg:286)
+   nn_bbl_adv = 0            (cfg:287)
+   rn_ahtbbl  = 1000.        (cfg:288)
+   rn_gambbl  = 10.          (cfg:289; read, multiplies nothing at nn_bbl_adv = 0)
```

SMT-6b = SMT-6 + the additive hunk `usrdef_istate_smt6b_cold_flank.patch`,
applied after SMT-5's dump hunk and placed BEFORE the dump call, so NEMO's
damping target is the anomalous initial state:

```
+      DO jk = 1, jpkm1 ; DO jj = 1, jpj ; DO ji = 1, jpi
+         IF( ptmask(ji,jj,jk) == 1._wp .AND. ptmask(ji,jj,jk+1) == 0._wp ) THEN
+            pts(ji,jj,jk,jp_tem) = pts(ji,jj,jk,jp_tem) - 1.7_wp * REAL( jpkm1 - jk, wp )
```

**The anomaly (offered for amendment).** Shape: the bottom wet T-cell of every
column is cooled by 1.7 K per level its bottom sits above the abyssal plain
(`jpkm1 = 10`): plain columns 0, the 56 flank columns 1.7 K, the 5 summit
columns 3.4 K; nothing else moves (S stays 35). Why this shape: every sloped
face joins levels one apart, so this opens each face by the same 1.7 K.
Amplitude: SMT-6's closed margins are 1.68143 to 1.68155 K on all 48 sloped
faces (MEASURED on NEMO's dumped SMT-5 initial T). At 0.1 K resolution
**1.7 K is the smallest that opens the gate**: 1.6 K opens 0 faces, 1.7 K
opens all 24 U + 24 V faces, and the BBL trend then touches **64 bottom cells**
(36 shelf-side, 28 deep-side). Both counts are checked by the test suite on the
card.

Runs: SMT-6 is namelist-only, so all three SMT-6 arms (2-step smoke, kt = 1..10
per-stage record, 100 days daily) run the SMT-5 executables, proved by hash
against the SMT-5 smoke manifest. SMT-6b is its own build
(`VORTEX_SMT6B_VEC_R8_OMIP_L1{,_P3}`, SMT-4's stage writer unchanged); its
kt = 1..10 and 100-day arms reuse its smoke pair by hash. Admission checks, per
arm: the resolved namelist lines (exactly one of each), `SMT5_DECK_CADENCE_OK`,
the compiled anomaly line and its order before the dump (6b), and the resolved
`ocean.output` lines (`ln_trabbc = T`, `nn_geoflx = 1`,
`rn_geoflx_cst = 8.64...E-002`, `constant heat flux`, `ln_trabbl = T`,
`nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl`, `rn_gambbl`). Those output
patterns were tried on two real NEMO outputs: all of ORCA2 rung 2's
switches-on lines match, and SMT-5's switches-off lines do not.

| file (committed) | sha256 |
|---|---|
| namelist_cfg_smt6_bbl_geothermal.patch | `53e8ade3252f79f430b9c37332873722a0bf5b61af8964a319d766f4530add7e` |
| usrdef_istate_smt6b_cold_flank.patch (on the SMT-5-patched usrdef_istate.F90 `dce27ee9cb4f29e1…`) | `a35e1ab1f98b5b7aa5829e55b25081b98addcab1a0af5e2639461db883e341c8` |
| lane run.sh | `feff8a0e6fbc5cde99bd90c170eec668138e3658c6448c11c1428e736de8593d` |
| shared VORTEX driver run.sh | `8bfb2d8b5c2bef4f3509b00c4893a0e315902fb8d84b6b3137dfdea335d633fe` |

Preflight (`--preflight`, builds nothing): `SMTRUNGS_R5_SMT6_PREFLIGHT_PASS`;
six `PREFLIGHT_OK`, six `SMT6_DECK_OK`, six `GFORTRAN_PATCHED_NEMO_SYNTAX_PASS`
(the 6b ones include the anomaly), cadence OK at 2/10/3000 steps
(`round5/preflight.log`). makenemo and mpirun were not run.

## Round 5 — 2. The legoESM arms, statement by statement

All NEMO citations are the SMT-5 build's compiled source (SMT-6 runs the same
executables).

**Geothermal (tra_bbc), new card field `nemo_geothermal_qgh_wm2`.**

| NEMO statement | citation | legoESM |
|---|---|---|
| stage 3 only: after tra_ldf, before tra_bbl and tra_dmp | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:526`, `:527`, `:528`, `:529` | added to the stage-3 tracer rate after the lateral term, before the damping increment |
| `rcp = 3991.86795711963` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1947` | the card's `c_sw` (test pins it) |
| `rho0_rcp = rho0 * rcp`, `r1_rho0_rcp = 1._wp / rho0_rcp` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:2379`, `:2382` | same two operations in fp64 |
| `qgh_trd0 = r1_rho0_rcp * rn_geoflx_cst`, no unit factor | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbc.f90:226` | same |
| `Krhs(mbkt) += qgh_trd0 / (e3t_3d(mbkt)*(1+r3t(Kmm)*tmask))`, temperature only, bottom wet cell only | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbc.f90:158-159` | NEMO's `e3t_0` times `(1+r3t)` at the stage-3 level (ssh at N+1/2, the same stretch the stage weights use), one division |

The existing operator-split helper (`GeothermalConfig` and its apply routine,
which divides by `rho_0*c_sw*h` after the step) is a different statement and is
untouched; the new function sits in the same module and reuses its bottom-cell
selector (searched: `grep -rn "geotherm" packages/ocean src scripts`).

**BBL (tra_bbl, diffusive).** The existing source-ordered transcription is
reused unchanged (geometry, gate, flux; stage-3 placement); the one gap round 4
measured is closed: the gate's expansion coefficients now take the S-EOS
branch when the card's EOS is NEMO's simplified one, with the deck's own
coefficients.

| NEMO statement | citation | legoESM |
|---|---|---|
| bottom T/S at Kbb, depth `gdept_1d(mbkt)*(1+r3t(Kmm))` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:409`, `:412` | unchanged |
| `eos_rab` on the bottom cells | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:417` | S-EOS: `alpha = (a0*(1+l1*zt+mu1*zh)+nu*zs)*r1_rho0`, `beta = (b0*(1-l2*zs-mu2*zh)-nu*zt)*r1_rho0` (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1344`, `:1347`), the shared S-EOS expansion function, deck coefficients from the card; other EOS unchanged |
| `zgdrho`, `SIGN(0.5, -zgdrho*mgrhu)`, `ahu = (0.5-zsign)*ahu_0` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:427-428`, `:430`, `:431` | unchanged (Fortran SIGN zero = closed) |
| static slope sign, BBL thickness, coefficient | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:584-585`, `:594`, `:600` | unchanged; the seamount card now supplies NEMO's own `e3u_0`/`e3v_0` arrays |
| bottom-cell trend | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:258-263` | unchanged |

Advective BBL (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:464`, `:480`, `:498`) is not
selected (`nn_bbl_adv = 0`, as ORCA2 rung 2). Its legoESM arm (OVERFLOW-certified)
still evaluates the polynomial EOS only; extending it to S-EOS has no deck that
would run it (named, not built).

**Selection (no silent default).** New field `nemo_geothermal_qgh_wm2`,
default None; `bbl_diffusive_option`/`bbl_aht_m2_s` defaults unchanged.
`grep -rn "nemo_geothermal_qgh_wm2=" packages/ src/ scripts/` returns one line,
the SMT-6 card block. The card validator refuses geothermal on any other card or
at any other value, and pins the SMT-6 BBL row (`0, 0.0, 1, 1000.0`). The model
refuses the field off the WS-RK3 program. Both SMT-6 cards refuse a dumped
damping target that is not their own initial T on every cell — this is the
istate identity gate for the anomaly: once the 6b record exists, the card
either builds (NEMO's istate and the card's formula agree to the bit) or
refuses.

## Round 5 — 3. Tests (focused, fp64, CPU)

New file `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_smt6_bbl_geothermal.py`:
**23 passed**. It covers: on a synthetic two-column slope the S-EOS gate
opens exactly when NEMO's inequality holds (12 cases, deck coefficients and a
full S-EOS set, including the equal-T zero case); the deck gate is a bottom-T
sign test; missing deck EOS refused; the diffusive flux equals trabbl
258-263 bit for bit; the geothermal rate equals trabbc 158-159 with eosbn2
2379/2382 bit for bit; SMT-6 config = SMT-5 + the three fields; SMT-6b differs
from SMT-6 only on bottom cells by exactly 1.7/3.4 K; gate closed on SMT-6
(0 of 24 + 24), open on SMT-6b (24 + 24), 1.6 K opens none; validator both
directions; a target that is not the initial state is refused; the stage-3
rate carries the increment and the stage-1 PLANT fires (stage-3 identity
breaks, stage-1 tracer moves while the production stage-1 tracer equals
SMT-5's bit for bit); on the FINAL production state, geothermal on minus off
adds `rn_Dt*qgh_trd0` of column heat to each of the 3721 wet columns (rtol
1e-3, r3t) and the plant breaks it; on SMT-6 BBL on/off is bit-identical
(gate closed); on SMT-6b the BBL moves exactly the 64 touched columns and
conserves global heat to 1e-9 with the live thickness.

Also run: the SMT-5 damping file and the round-4 survey test, 16 passed;
`tests/ocean/unit/test_bbl_adv.py`, `test_geothermal.py` and the ratchets
(hard-coded constants, inline coefficients, dispatch, validate_strict, private
imports, physics contracts, parameter bounds): 6000 passed, 2 failed, both in
files this round did not touch (an old JRA55 test, already noted red in round
2, and `land/restart.py`; `git diff` silent on both).

**Closed cards bit-identical, MEASURED**: state sha256 (u, v, T, S, ssh) after
3 production steps at the parent (`2fa05b11c`, clean worktree) vs this diff on
GYRE, VORTEX, VORTEX_VEC, SMT, SMT_VEC, SMT-1..5 (SMT-5 from its real
NEMO-dumped inputs, with model time): 10/10 `CLOSED_CARDS_BIT_IDENTICAL`
(`round5/closed_cards_parent.json` `c4fbe551…`, `closed_cards_after.json`
`37e926c4…`). ORCA2 cannot step without its forcing pipeline, so the diffusive
BBL chain it selects (geometry, gate, trend, its own inputs and EOS-80) was
digested at both commits: identical (`a248aa2d…`, 227 U + 350 V faces open,
so not a zero; `round5/orca2_bbl_{parent,after}.json`).

## Round 5 — 4. Preregistered (frozen; scored next round)

SMT-6: geometry identical to SMT-5; gate closed on every sloped face at every
kt = 1..10 and every daily restart; geothermal increment on exactly the 3721
wet bottom cells, 1.215e-7 to 1.199e-6 K per step; the offline replay of the
trabbc statement on NEMO's operands equals legoESM to the bit; kt = 1 rows at
the bar; first-over-bar stays SMT-4's kt = 2 statement. SMT-6b: the card builds
from NEMO's dump (istate identity); 24 + 24 open faces at kt = 1; T trend on
64 cells, S trend 0; gate operands bit-exact, the first non-bit statement in
the BBL chain (if any) is the bottom-cell divisor and its stage-3 association
(trabbl 258-263) at ULP size. Falsifiers in the prereg.

## Choices made this round

| choice | status |
|---|---|
| deck values (nambbc constant 86.4e-3, nambbl = ORCA2 rung 2) | ASKED (Decision 110) |
| anomaly shape (bottom cell, 1.7 K per level above jpkm1) and amplitude (smallest at 0.1 K resolution) | UNASKED within D110's "smallest that opens the gate" — offered for amendment (DECISION_NEEDED) |
| anomaly applied BEFORE the SMT-5 dump (damping target = anomalous initial state, keeps the gate open over 100 days) | UNASKED — revert = move the hunk after the dump call (then damping erases the anomaly in ~1 day) |
| `bbl_gamma_s = 0` on the card although the deck reads `rn_gambbl = 10` (inert at `nn_bbl_adv = 0`; same as the ORCA2 card) | UNASKED — revert = set 10.0 (no numerical effect) |
| SMT-6 reuses the SMT-5 executables (namelist-only change) | UNASKED mechanism — revert = a fresh SMT-6 build |
| closed-card instrument extended (SMT-5, ORCA2 BBL digest); review-driven test additions | exempt (tests/instruments) |

## Review, citations

Single review (codex), verdict before fixes: **"DO NOT SHIP"**, 2 findings,
both CONFIRMED and fixed: (1) "every other card bit-identical" was unproved
for ORCA2, the one existing card on the modified BBL path — now digested at
both commits, identical; (2) the geothermal placement test read a
self-reported tuple, not the final result, and the 6b BBL test only checked
inequality — now the final-state column heat budget, the exact 64-column
footprint and global conservation. Codex reported no sign, index, staggering,
`jpkm1` mapping, `Kmm` time-level, tuple, patch-order, binary-reuse or shell
pattern defect. The second reviewer (GLM) was not run: NO GATE, single review
only, as the standing note asks.

Citation gate: from this receipt's "## Round 5 —" heading PASS, 24
citations, 0 unmapped, 0 failures (`round5/citation_gate_round5.json`); a
planted 2-line shift of the trabbc 158-159 citation exits 1. The the cumulative default run PASS (274 citations,
0 unmapped, 0 failures, 0 map entries failing audit). The insertions shifted 59
map entries citing three legoESM files; they were re-anchored by the exact
parent-to-tip line alignment (one hand-moved: the round-4 GAP line, now inside
the non-S-EOS branch), and the ORCA2 BBL-selection anchor was pinned as the
first of two occurrences.

## OPEN for round 6

1. Operator runs `run.sh --smoke` then `--run`; score P1-P7 and Q1-Q5.
2. Offline replay instrument for tra_bbc and tra_bbl from the stage-3 states
   (stage writer boundaries are `adv hpg vor ldf`; bbc/bbl/dmp follow `ldf`
   inside stage 3, so the replay starts from the post-ldf Krhs record).
3. Advective BBL under S-EOS: not built (no deck runs it).

UNVERIFIED: the NEMO side is proved by preflight and syntax compile only, not by
a run; that NEMO's istate halo loop over `jpi`/`jpj` changes nothing the card
does not see (halos are land on this closed single-rank box); the legoESM
stage-3 accumulation order `(rate + geothermal) + damping` with the BBL folded
into the content separately, versus NEMO's `((ldf + bbc) + bbl) + dmp` —
identical on SMT-6 (BBL adds exact zeros), ULP-level on SMT-6b by reading only.
