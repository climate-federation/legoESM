# Receipt — SMT-RUNGS round 6: SMT-6 / SMT-6b record admission, identity, ladders, replays, 100 days

**Status: HELD (no model change).** Both NEMO records are admitted and the
geometry and initial states are identical to what the cards read. SMT-6 (BBL
gate closed + geothermal heating) is at the bar: its ladder and its 100 days
equal SMT-5's, and the geothermal statement replays bit for bit. SMT-6b (the
cold-flank anomaly that opens the BBL gate) is **not** limited by the BBL
chain: trabbl's gate and trend replay bit for bit on NEMO's own operands. The
large SMT-6b residual (stage-3 T, 3.4e-4 K at kt=2) sits in other stage-3
statements the anomaly exercises for the first time; no cited one-variable fix
exists, so nothing is landed. Base `cac88a441`; the diff touches harness scripts,
one test, the citation map and this receipt (`git diff cac88a441..HEAD --stat
-- packages src` is empty). Evidence
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round6/`;
preregistration `docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smtrungs_round5_smt6.md`
(committed before either record existed).

**ORCA2 pointer.** ORCA2 rung 2 runs exactly these statements
(`nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl = 1000`, trabbc) on its
marginal-sea geometry. What carries to the ORCA2 rung-2 walk unchanged:
(1) the BBL gate/trend replay below is written from NEMO's own `mesh_mask`,
namelist coefficients and the entry record at Kbb; for ORCA2 it needs two
additions it does not have — the `eos_rab` TEOS-10 polynomial branch, and
NEMO's lateral-boundary exchange (it clamps the north/east neighbours, which is
exact only on this closed box; ORCA2's periodic/fold exchange of `mbku_d`,
`e3u_bbl_0` and the bottom fields is not replayed); (2) NEMO closes a
face whose `-zgdrho*mgrh` is a signed zero (measured below), so the replay and
the card use `>= 0` = closed, never a `copysign` reading; (3) the geothermal
replay (reciprocal first, `r3t = ssh*r1_ht_0` at the stage-3 level taken from
NEMO's stage-2 ssh) — ORCA2 reads its flux from a file (trabbc lines 242-243,
one more read statement and a 1e-3 factor); (4) the lateral-diffusion slope
limiter below is a statement a steep-slope ORCA2 rung can reach and a gentle
seamount stratification cannot (PLAUSIBLE; not measured on ORCA2).

## Round 6 — 1. Record admission

The shared VORTEX `check_records.py` was re-run on both decks' kt=1..10 and
100-day arms. Each parses 27 (kt) / 3067 (100-day) records from their own
headers and requires the instrumented restart byte-identical to the plain
build's; all 100 daily restarts of each deck are byte-equal to their reference
(`cmp`, 0 unequal). The header, field-name and truncation plants exit nonzero
on both decks' kt arms.

| arm | status | parsed records | restarts vs reference | admission json (sha256 prefix) |
|---|---|---:|---|---|
| SMT-6 kt=1..10 | ADMITTED | 27 | byte-identical | `eeace8a891fe0586` |
| SMT-6 100-day | ADMITTED | 3067 | 100/100 byte-identical | `4240962c36f15b45` |
| SMT-6b kt=1..10 | ADMITTED | 27 | byte-identical | `800dbf4b04149ba0` |
| SMT-6b 100-day | ADMITTED | 3067 | 100/100 byte-identical | `e0ba19b40cf752c2` |

Provenance of the dumped inputs (first 12 hex of sha256; identical across the
smoke, kt and 100-day arms of each deck):

| file | SMT-5 | SMT-6 | SMT-6b |
|---|---|---|---|
| data_1m_potential_temperature_nomask.nc | `2972e475429e` | `2972e475429e` | `df701802c29b` |
| data_1m_salinity_nomask.nc | `6734075c0744` | `6734075c0744` | `6734075c0744` |
| resto.nc | `024743b23a2d` | `024743b23a2d` | `024743b23a2d` |
| mesh_mask.nc | `de8d71e558b2` | `de8d71e558b2` | `de8d71e558b2` |

Executables: SMT-6's `nemo` binary is byte-identical to SMT-5's (`4cb228e8e26a`;
namelist-only change; **no `VORTEX_SMT6_*` build exists — the operator's list
names one; the evidence is the hash**). SMT-6b is its own build
(`VORTEX_SMT6B_VEC_R8_OMIP_L1{,_P3}`, `ab3d623a7b2f`). Both decks' `ocean.output`
resolve `ln_trabbc = T`, `nn_geoflx = 1`, `rn_geoflx_cst = 8.64E-02`,
`ln_trabbl = T`, `nn_bbl_ldf = 1`, `nn_bbl_adv = 0`, `rn_ahtbbl = 1000`,
`rn_gambbl = 10`, `ln_tradmp = T`.

## Round 6 — 2. Geometry and initial-state identity (P1, Q1, Q2)

- **Geometry gate vs NEMO's `mesh_mask.nc`: GEOMETRY IDENTICAL, 18/18 rows
  EXACT** for both cards (non-vacuity row: 1164 faces where `e3u_0 != e3t_0`).
  `mesh_mask.nc` is byte-equal to SMT-5's in every arm of both decks.
- **SMT-6**: dumped T record equals the card's analytical T on all 37144 wet
  cells (0 unequal) and equals SMT-5's dump (0 unequal); S = 35, resto equal.
- **SMT-6b (Q1)**: NEMO's dumped T record 1 equals the card's initial T —
  analytical profile plus the anomaly — on all 37144 wet cells (0 unequal; the
  card applies the anomaly analytically and refuses unless that equals the
  dump; the equality is re-read here for T, and for S: 0 unequal of 37144 on
  both decks). Against SMT-5's dump exactly 61 cells differ,
  all bottom cells: 56 by 1.7 K and 5 by 3.4 K (0 off the bottom level).
- **NEMO's own BBL criterion on NEMO's own T/S** (trabbl lines 409-431 and the
  geometry 584-600, read from NEMO's `mesh_mask` and the dumped T; the card's
  transcription gives the same counts): SMT-6: **0 open of 24 U + 24 V** sloped
  faces at the initial state, at all ten kt entries and at all 100 daily
  restarts. SMT-6b: **24 U + 24 V open at kt=1**, the BBL then touches **64
  bottom cells**. By "shelf-side = the shallower bottom of the open face" the
  split is 28 shelf-side / 36 deep-side (no cell in both roles); round 5's
  receipt printed 36/28 — the total agrees, its labels were swapped.

## Round 6 — 3. The kt=1..10 ladders (P5, P6, Q4)

Independent label (legoESM's own trajectory against NEMO's entries; `T`/`S`
in K / psu, `u`/`v` in m s-1, `ssh` in m; max abs over the scored wet cells).
Full per-kt, per-field first unequal cell / count / max abs / rms is in
`m_b_ladder.json` (`a8c68c4fcdc996f3`); the certified-format ladders are
`smt6_ladder.json` (`23350e614976f486`) and `smt6b_ladder.json`
(`f0077b8617431e45`). Each card is scored against its own NEMO run (D52: the
SMT-6b-vs-SMT-6 pair is two decks, not a candidate/control on one oracle).

| kt | field | SMT-5 | SMT-6 | SMT-6b | SMT-6b, BBL off in legoESM only |
|---:|---|---|---|---|---|
| 1 | T, S, u, v, ssh | 0, 0, 2.2e-16, 2.2e-16, 1.4e-20 | same | same | same |
| 2 | T | 1.43e-08 | 1.43e-08 | 3.42e-04 | 3.35e-04 |
| 2 | u / v | 3.2e-09 / 6.8e-10 | 3.2e-09 / 6.8e-10 | 1.4e-07 / 1.1e-07 | 1.4e-07 / 1.1e-07 |
| 2 | ssh | 3.9e-10 | 3.9e-10 | 3.9e-10 | 3.9e-10 |
| 3 | T / u | 6.0e-08 / 1.0e-08 | 5.9e-08 / 1.0e-08 | 6.6e-04 / 3.9e-05 | 6.5e-04 / 3.8e-05 |
| 10 | T / u | 6.3e-07 / 7.3e-07 | 6.9e-07 / 7.3e-07 | 2.0e-03 / 3.2e-03 | 2.0e-03 / 3.2e-03 |

- **First over the 1e-15 bar**: kt=2 on T, u, v, ssh in SMT-6 and SMT-6b alike
  (S from kt=4/5, sub-ulp of 35 psu). kt=1 is at the bar for all five fields.
- **SMT-6 = SMT-5**: same first unequal cell at kt=2 (T `[1,5,0]`, SMT-4's
  inherited statement), same max; the geothermal heating (≤1.2e-6 K per step)
  is invisible at the ladder level.
- **SMT-6b first over the bar is the same rows at the same kt as SMT-6, but
  4 orders larger in T and 44-160x in u/v**, and the error then grows by 2-3
  orders in ten steps. Switching the BBL off in legoESM changes the T maximum
  by 2 % and u/v not at all, so the BBL statement is not what sets it
  (§4 proves the BBL chain bit-exact on NEMO's operands).
- GIVEN-NEMO-ENTRY label (never mixed with the above): one legoESM step from
  NEMO's kt=2 entry gives SMT-6b T 3.37e-4 K, u 3.46e-5 m s-1 against NEMO's kt=3
  (SMT-6: 4.6e-8 K, 6.5e-9 m s-1), so the SMT-6b error is made inside each step
  from an exact NEMO state, not carried.

### 3a. Where inside kt=1 (stage records, NEMO source order)

NEMO writes the state after each RK3 stage
(`VORTEX_SMT6B_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:215` for stage 3).
legoESM tracer after stage 1 / 2 / 3 against those records (T max abs):

| deck | stage 1 | stage 2 | stage 3 |
|---|---|---|---|
| SMT-6 | 1.5e-12 | 2.2e-12 | 1.43e-08 |
| SMT-6b | 1.5e-12 | 2.2e-12 | **3.42e-04** |

Stages 1 and 2 are at the floor in both decks, so the SMT-6b residual is made
**inside stage 3**: the stage-3 tracer chain is `tra_ldf`, `tra_bbc`,
`tra_bbl`, `tra_dmp`, then `tra_zdf`
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:526`, `:527`, `:528`,
`:529`, `:538`). Starting legoESM's stage 3 from NEMO's own stage-2 state and
NEMO's barotropic handoff leaves the residual unchanged (3.42e-04 K), so it is
not inherited from stages 1-2. Term sizes in legoESM at that step (legoESM
only; NEMO is not re-run):

| term switched off or changed | SMT-6 max change in T (K) | SMT-6b max change in T (K) |
|---|---:|---:|
| BBL off | 0 | 1.2e-04 |
| geothermal off | 1.2e-06 | 1.2e-06 |
| damping off | 0 | 0 (target = state at kt=1) |
| Redi off (kappa_Redi = 0) | 5.5e-04 | 8.5e-03 |
| Redi slope limit S_max 0.01 -> 1.0 | **0** | **6.4e-04** |

The slope limiter binds only under the anomaly:
`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfslp.f90:231` and `:232` bound the
slopes by `rn_slpmax` and by `7e3/e3`, and legoESM carries both. SMT-6b's
anomaly gives the limiter its first live samples on this deck; the 6.4e-04 K
sensitivity is the same size as the 3.4e-04 K residual. **PLAUSIBLE, not
CONFIRMED**: that is a sensitivity of legoESM to a parameter, not a located
NEMO statement; NEMO's stage-3 tracer terms were not recorded.

Momentum, stage 3 (existing walk `smt4_ldf_walk`, pointed at the SMT-6/6b
records through the new `--case`): cumulative Krhs after each boundary against
NEMO's record. `hpg` and `vor` carry 6.7e-06 m s-2 in both decks (equal to four
digits: not caused by the anomaly; unexplained here, see UNVERIFIED). `adv` / `pre_ldf` / `post_ldf` are 5.5e-11 m s-2 in SMT-6b
against 4.7e-13 in SMT-6; 5.5e-11 x 2880 s = 1.6e-07 m s-1, which is the
kt=2 u residual. PLAUSIBLE: rounding-scale residual on the larger
velocities the anomaly drives; not discriminated.

## Round 6 — 4. Offline replays of the cited statements (P3, P4, Q2-Q4)

Both replays are numpy transcriptions of the cited NEMO statements, fed NEMO's
own arrays (`mesh_mask.nc`, namelist coefficients, the entry/stage records);
nothing is run inside NEMO.

**Geothermal heating, `trabbc`.** The tendency is
`qgh_trd0 = r1_rho0_rcp * rn_geoflx_cst`
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbc.f90:226`) added on the
bottom wet cell as `qgh_trd0 / (e3t_3d * (1 + r3t(Kmm)*tmask))`
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbc.f90:158-159`).

- Replay (NEMO `e3t_0`, `r3t = ssh*r1_ht_0` with NEMO's stage-2 ssh,
  `rho0`, `rcp`) vs legoESM's stage-3 increment: **3721 of 3721 bottom cells,
  0 unequal, max abs diff 0**; per-step increment 1.2151e-07 to 1.1990e-06 K.
  (A first replay that left legoESM on its own ssh differed on 2106 cells at
  1e-14 relative: the inherited ssh residual moving `1+r3t`, not the
  statement. Instrument fixed, not the model.)
- After the implicit vertical solve the increment occupies **7442 cells in
  both** NEMO and legoESM (bottom cell and the cell above). NEMO's SMT-6 entry
  minus NEMO's SMT-5 entry at kt=2 (the entries are bit-equal at kt=1, so this
  is the one namelist change) against legoESM geothermal on minus off: max
  abs difference **1.8e-15 K, at most 2 ulp, 0 cells above 2 ulp** (37144
  cells; 2609 differ by an ulp; S, u, v, ssh of the two NEMO entries equal).

**BBL, `trabbl`.** Replayed from bottom T/S at Kbb
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:409`), the reference
depth (`:412`), `eos_rab` (`:417`) with the deck's linear S-EOS
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1344`, `:1347`:
`a0 = 0.28`, every other coefficient 0, so the gate is a bottom-T sign test),
`zgdrho` and the sign (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:427-428`,
`:430`, `:431`), the slope and thickness
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:584-585`, `:594`, `:600`)
and the bottom-cell trend
(`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/trabbl.f90:258-263`).

| quantity | SMT-6 | SMT-6b |
|---|---|---|
| open U / V faces, replay vs card, at kt=1 | 0 / 0 vs 0 / 0 | 24 / 24 vs 24 / 24 |
| `ahu_bbl`/`ahv_bbl` unequal on sloped faces, 110 NEMO states (10 entries + 100 restarts) | 0 | 0 |
| trend T: nonzero cells replay / card, unequal cells, max abs | 0 / 0, 0, — | 64 / 64, **0**, 4.12e-08 K s-1 (1.19e-04 K per step) |
| trend S: nonzero cells | 0 | 0 |

So Q2 and Q3 hold, and the BBL chain (gate operands, `ahu`/`ahv`, bottom-cell
trend including the `r3t(Kmm)` divisor and the U/V association) is **bit-exact
against NEMO's own statements on NEMO's operands** — the footprint of the
SMT-6b residual is not BBL-limited.

**The signed-zero reading, measured.** On the 559 flat wet U faces with
`dT != 0` the factor `-zgdrho*REAL(mgrhu)` is a signed zero. Read with
`copysign` (gfortran's `SIGN`) half of them open (474 cells,
2.6e-06 K per step on SMT-6); read as `>= 0 = closed` none do. NEMO's SMT-6
entry minus NEMO's SMT-5 entry differs from legoESM's geothermal-only
increment by at most 1.8e-15 K on all of those 474 cells, so NEMO closes them
(`SIGN(0.5, -0.0)` did not open the face in this build: `-O3`, no `-ffast-math`).
The replay and the card use `>= 0 = closed`; the other reading is refuted by
NEMO's record, not by argument.

## Round 6 — 5. The 100-day comparison (P7, Q5)

Same harness for all three cards (the SMT-5 row is round 3's run; each
legoESM run starts with the ten-step sanity check REPRODUCED against this
round's own ladder). rms over wet cells against the card's own NEMO run:

| day | T rms (K): SMT-5 / SMT-6 / SMT-6b | u rms (m/s) | ssh rms (m) |
|---:|---|---|---|
| 1 | 9.68e-08 / 9.78e-08 / 1.65e-04 | 4.35e-08 / 4.49e-08 / 1.19e-04 | 7.4e-09 / 7.4e-09 / 1.14e-05 |
| 5 | 1.14e-06 / 1.14e-06 / 2.23e-04 | 5.17e-07 / 5.17e-07 / 1.43e-04 | 3.8e-08 / 3.8e-08 / 3.0e-05 |
| 30 | 5.29e-06 / 5.29e-06 / 2.33e-04 | 2.72e-06 / 2.72e-06 / 1.73e-04 | 4.2e-07 / 4.2e-07 / 1.03e-04 |
| 100 | 1.08e-05 / 1.08e-05 / 2.45e-04 | 5.46e-06 / 5.46e-06 / 1.95e-04 | 9.2e-07 / 9.2e-07 / 1.47e-04 |

- **SMT-6 = SMT-5 on every day**: max |T rms difference| over the 100 days is
  1.3e-08 K. First day T rms exceeds 10x its day-1 value and 1e-6 K: day 5 in
  both; peak day 70; first above 1e-5 K: day 46. P7: 1.08e-05 K, inside the
  frozen 2.2e-05 K.
- **SMT-6b never sits on a floor**: 1.65e-04 K already at day 1, flat to
  2.45e-04 K at day 100 (23x SMT-6's 1.08e-05); u 36x, ssh 160x, T max
  1.7e-02 K (SMT-6: 1.3e-03). S rms is 7e-14 psu in all three. The damping
  holds T near the anomalous target, so the T error saturates; u and ssh keep
  growing.
- **Gate over 100 days (Q5)**: NEMO's SMT-6b gate has 48 open faces at kt=1,
  36 at kt=2, 8 at kt=3, **0 at kt=4..10**, then 16 at day 1 and **12 of 48 at
  day 100** (6 U + 6 V; 12 to 16 over all 100 daily restarts). SMT-6 stays 0.

## Round 6 — 6. Round 5's frozen predictions, scored

| # | result | label |
|---|---|---|
| P1 | 18/18 EXACT both decks; mesh byte-equal; SMT-6 dumped T/S/resto byte-equal to SMT-5's | CONFIRMED |
| P2 | 0 open faces at all ten entries and 100 restarts (NEMO's criterion on NEMO's T/S); legoESM `ahu`/`ahv` 0, trend 0 | CONFIRMED |
| P3 | tendency on exactly 3721 bottom cells, 1.2151e-07 to 1.1990e-06 K per step; after the implicit solve the footprint is 7442 cells (bottom + one above) in NEMO and legoESM alike | CONFIRMED (tendency); the after-tracer wording would be refuted |
| P4 | replay 0 unequal of 3721 (with NEMO's ssh as `r3t`) | CONFIRMED |
| P5 | kt=1: T, S 0; u, v 2.22e-16; ssh 1.4e-20 — equal to SMT-5 | CONFIRMED |
| P6 | first over the bar kt=2, T first unequal `[1,5,0]`, no earlier row | CONFIRMED |
| P7 | 1.08e-05 K <= 2.2e-05 K | CONFIRMED |
| Q1 | card equals the dump on 37144/37144 wet cells | CONFIRMED |
| Q2 | 24 U + 24 V open at kt=1 | CONFIRMED |
| Q3 | 64 T cells, 0 S cells (28 shelf-side / 36 deep-side; round 5's split was swapped) | CONFIRMED |
| Q4 | gate operands bit-exact (110 states): CONFIRMED. "First non-bit statement is the bottom-cell divisor at ULP, residual <= 1e-14 K on the 64 cells": the divisor/association is bit-exact (0 unequal), and the residual on the 64 cells is 8.7e-05 K rms, 3.4e-04 K max, owned by other stage-3 statements | REFUTED (second half) |
| Q5 | 12 of 48 faces open at day 100 (< 40) | REFUTED |

Q5 is refuted for a reason that matters to the rung's design: the anomaly
margin over NEMO's criterion is 0.02 K (1.7 K against 1.68 K), and the
SMT-6b dynamics erodes it within three steps; the damping then reopens 12 to 16
faces each day. The deck exercises the BBL on about 3 steps and then
intermittently, and exercises the lateral-diffusion limiter and the pressure
gradient much harder than the BBL.

## Round 6 — 7. First non-bit statement, and what is not landed

- **SMT-6**: the first non-bit statement is SMT-4's inherited kt=2 statement
  (T first unequal `[1,5,0]`, 19447 cells, 1.43e-08 K); BBL (closed) and
  geothermal add none above the 2-ulp floor. ORCA2 rung 2's two statements are
  at the bar on the seamount.
- **SMT-6b**: the first non-bit **stage** is stage 3 (T 3.42e-04 K at kt=2,
  first unequal cell `[1,2,1]`, largest at the bottom cell of an anomaly column,
  bottom level index 8); inside it the BBL chain is excluded by replay, the damping and
  geothermal terms are 0 and 1e-6 K, and the remaining candidates are the
  stage-3 lateral-diffusion chain and the vertical solve. The Redi slope-limit
  sensitivity above favours the first (PLAUSIBLE). **No cited NEMO statement is
  identified**, so no one-variable fix exists and nothing is landed (no model
  or physics file changes; the SMT-1..5 registries, GYRE note-CH, DINO, tank and
  D96 gates apply to model changes and were not triggered).

Open for the next round (named, not built — not asked for here):

1. NEMO tracer-term records at stage 3 for SMT-6b (lateral-diffusion slope and
   flux terms, vertical-solve operands), using the existing SMT-3 internal
   record patches on the SMT-6b build, to split the 3.4e-04 K residual by
   statement. That is one acquisition by the operator.
2. The u/v error growth (3e-03 m s-1 at kt=10) is a separate item: a stage-3
   momentum boundary (`adv`) first shows the 117x increase over SMT-6.
3. A BBL certification that does not depend on the anomaly's collateral: the
   replay above already certifies the chain; a deck variant with a larger
   margin is only needed if the rung must also run clean over many days.

## Tests, instruments, review

- New `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_smtrungs_round6_measure.py`:
  4 passed (the review's four findings fixed: S identity read, `--sections`
  required, the ORCA2 caveat widened, the "built from the dump" wording
  corrected). It covers the touched-column logic, the real-record input
  identity (61 anomaly cells, all bottom), the BBL replay equalling the card on
  both decks together with the signed-zero reading predicting increments
  NEMO's record excludes, and non-vacuity: with the card's gate forced closed
  the replay reports 64 unequal cells.
- Plant: `--plant` moves one wet T cell one ulp in the independent SMT-6 kt=1
  row, exits 1 with exactly 1 unequal cell (`m_plant.json`); unplanted that
  row has 0.
- The three harnesses extended additively (default behaviour for every
  existing card unchanged): the trajectory gate and geometry gate select the
  SMT-6/6b records and read their deck inputs; the 100-day script gains tags
  `smt6`, `smt6b`; the stage-3 momentum walk gains `--case`. The measurement
  script runs each section in its own process (`--sections`): one process
  holding every section's compiled models exhausted the JIT code memory twice.
- Instrument corrections quoted: the `copysign` reading of `SIGN`, and the
  legoESM-own-ssh geothermal replay (2106 cells at 1e-14), both superseded by
  the measurements above; `m_d_replay_stages.json` still holds the superseded
  geothermal replay, `m_d2_replay.json` the corrected one.

Single review (codex): HOLD, four findings, all CONFIRMED and fixed above (input-identity gate compared T only; the default CLI ran every section in one process, which exhausts JIT memory; the replay's ORCA2 portability claim ignored lateral-boundary exchange; "card built from the dump" was reversed). The second reviewer (GLM) was not run: NO GATE, single review only, as the standing note asks.

## Choices made this round

| choice | status |
|---|---|
| scoring both labels, per-stage tracer residuals, term-size arms | the ask (items 3-4) |
| `>= 0 = closed` for the signed zero in the replay | UNASKED instrument convention, forced by NEMO's own record; the refuted reading is quoted |
| "shelf-side = the shallower bottom of the open face" | UNASKED reporting convention; totals are convention-free |
| "floor" = 10x the day-1 T rms and 1e-6 K | UNASKED reporting convention (as round 3) |
| three existing harnesses extended with default-unchanged options | UNASKED mechanism; revert = drop the options (the cards cannot be scored otherwise) |
| no acquisition script written | UNASKED omission; the next-round item 1 is the proposal |

UNVERIFIED: NEMO's stage-3 tracer terms and `ldfslp` operands for SMT-6b were
not recorded, so the residual owner is by elimination plus sensitivity only;
the signed-zero behaviour is established from NEMO's SMT-6/SMT-5 increments,
not from the compiled instruction; the S-EOS replay assumes `rn_rho0` equals
the card's `rho_0`; the 100-day comparison is one realization per card at
fp64/libm on CPU; the momentum-walk `hpg` boundary mismatch (6.7e-06 m s-2 in
both decks) is not explained here.
