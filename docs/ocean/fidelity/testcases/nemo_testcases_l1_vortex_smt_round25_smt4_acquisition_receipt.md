# Receipt — VORTEX_SMT round 25 (lane round 237): SMT-3 landing and SMT-4 acquisition

**Status: LANDED.** Decisions 95 and 96, entered in the campaign ledger after
this round's acquisition preregistration but before its production edit,
ordered the already measured round-236 closed-bottom-W-mask plus live-Kmm
tracer-LDF-divisor pair into production. The pair reproduces the round-236
production-JIT proof and 50-row registry exactly. Its disclosed two-ULP
cellwise ratchet remains red; it is registered, not hidden. The prepared
SMT-4 acquisition remains the next record request and is not used as evidence.

Base commit: `33c754c716ccf3fc6c50692ca9c30041ff7a7a64` (round 236).
Acquisition implementation commit: `93dbb587a723019d1bff616ca149fd1c131a4c18`.
Physics implementation commit: `9c1d5dc616f121f9dbe48635e392a7e0ae6c86dd`.
Preregistration commit: `114067f336dff0258f5acf4f794906e8b59e76b1`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/`.

## 1. Outcome

The shared tracer-LDF transcription now uses NEMO's closed deepest W-mask
level in the horizontal isoneutral mask sums and NEMO's live stage-3 `Kmm`
T-cell thickness in the final divergence divisor. This is exactly the
candidate state measured in round 236; the current residual artifact has the
same SHA-256, `f1fae7ba7b43bc4313db20870380f1d7b819e91259fcd4c2511a555a1045fb7a`.

The SMT-3 registry moves 34 of 50 aggregate rows: 30 toward NEMO and four
away. No row changes AT-BAR/DEBT status, every kt=1 row stays at the bar, and
first-over-bar remains kt=2. Decisions 95/96 explicitly admit the 40 disclosed
near-zero-cell two-ULP violations because the cited pair improves the target
registry in the majority without a certified-row loss.

Separately, the requested SMT-4 one-module rung is prepared mechanically:

* SMT-3 is changed only in `&namdyn_ldf` to div-rot, level Laplacian
  momentum diffusion with coefficient mode 20, `rn_Uv=0.1 m/s`,
  `rn_Lv=10 km`, and `rn_ahm_b=0`;
* new targets are `VORTEX_SMT4_VEC_R8_OMIP_L1` and
  `VORTEX_SMT4_VEC_R8_OMIP_L1_P3`; no existing target is modified;
* the 10-step arm reuses the existing self-describing stage-1/2/3 momentum
  record, including the accumulator immediately before and after `dyn_ldf`;
* the 100-day arm reuses the newly admitted SMT-4 executable by its binary
  manifest and writes the shipped 3,000-step trajectory at daily cadence;
* restart byte identity, parse-to-EOF, required names, resolved namelist
  values, `STOP 0`, and header/name/truncation plants all fail closed.

The operator command is:

```text
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smt_round25_smt4/run.sh --run
```

The acquisition script carries the clean-tree commit stamp into its manifest.
It uses shell `SECONDS`, never `/usr/bin/time`, and creates new build targets.

## 2. Preregistered predictions

| prediction | round-237 evidence | verdict |
|---|---|---|
| R25-P1 controlled one-module deck | dry application prints the inherited SMT-2 and SMT-3 hunks plus only the new `namdyn_ldf` hunk | CONFIRMED for patch construction; runtime resolution pending |
| R25-P2 resolved branch | exact runtime `ocean.output` checks are installed for the complete tuple and iso-level Laplacian dispatch | UNMEASURED |
| R25-P3 passive self-describing record | parser, restart comparison, required groups, and header/name/truncation plants are installed; synthetic controls pass | UNMEASURED on NEMO |
| R25-P4 run sanity | both arms refuse nonzero launch, absent `STOP 0`, missing restart/frame, or non-finite state | UNMEASURED |
| R25-P5 first new boundary is post-`dyn_ldf` | all three stages record pre/post-LDF momentum accumulators | UNMEASURED |
| R25-P6 acquisition-only disposition | superseded before production changes by the committed Decision-95/96 addendum | SUPERSEDED, not silently rewritten |

No failed prediction is hidden. P1 is only a construction result until the
new executable echoes the resolved card. P2--P5 remain open. R25-P6 was true
when frozen; the later, also frozen, binding addendum names its supersession
and its falsifiers before the production files changed.

## 3. Preflight and controls

The committed wrapper was run without `--run`. The complete log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/preflight.log` and
ends with:

```text
GFORTRAN_SYNTAX_PASS .../vortex_r16_stage_terms.F90
PREFLIGHT_OK  variant smt4vec: instrument and deck patches apply to the shipped sources
GFORTRAN_SYNTAX_PASS .../vortex_r16_stage_terms.F90
PREFLIGHT_OK  variant smt4vec100d: instrument and deck patches apply to the shipped sources
ROUND237_SMT4_PREFLIGHT_PASS /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/oracle_vortex_smt4
```

The checker follows the self-describing-record rule: magic plus header, then
named `(rank,n1,n2,n3,payload)` groups to EOF. It predicts no record size or
header tuple by hand. The acquisition itself runs the `header`, `field-name`,
and `truncated` plants and refuses if any exits zero.

Focused controls before finalisation:

```text
tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round237_smt4_record.py
7 passed in 0.16s

tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round224_smt3_record.py
tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round237_smt4_record.py
16 passed in 0.25s
```

`bash -n` on the shared driver and wrapper and `git diff --check` also pass.
The final citation and focused-gate results are recorded below after their
committed inputs were exercised.

## 4. Scientific gate table and moved-row registry

### 4.1 Target SMT-3 registry

The current `/round237/landing_smt3.json` reproduces the round-236 candidate
artifact and all 50 aggregate values exactly. The 34 moved rows are therefore
the immutable round-236 table, incorporated here by exact reference:
`nemo_testcases_l1_vortex_smt_round24_ldf_pair_landing_receipt.md`, section 3. Its
before/candidate values, directions, and row names are unchanged. In compact
form: kt2 T moves `1.020298116571876e-08 -> 6.957537701781045e-10`; 30 rows
move toward NEMO; kt7/8/9/10 U are the four away rows; first-over-bar remains
kt2 T/u/v/ssh; and no aggregate status changes. The cellwise comparison still
reports 40 violations and maximum worsening
`431802197.9165039` row-scale oracle ULP. This red result is the registered
Decision-95/96 exception, not a passing ratchet.

### 4.2 GYRE certified ladder and year

The full 954-row GYRE comparison has zero changed aggregate values, zero
row-status changes, and first-over-bar kt3 on T/S/u/v/ssh in both arms. Its
cellwise ratchet is red on 46 rows (maximum worsening two machine ULP and
`51122.15625` row-scale oracle ULP on a near-zero residual). This is also
registered under the binding Decision-95/96 landing; the aggregate certified
registry and first-over-bar do not regress.

The 360-day result is:

| day | before T3D rms K | after T3D rms K | change / 2e-10-K floor |
|---:|---:|---:|---:|
| 30 | 2.3432437414839976e-06 | 2.3432419318363155e-06 | -0.009048 |
| 60 | 1.4793243459436834e-05 | 1.4793244000905481e-05 | +0.002707 |
| 90 | 1.6332701526871403e-05 | 1.6332666348876788e-05 | -0.175890 |
| 120 | 1.0965908847407414e-04 | 1.0965903952280960e-04 | -0.244756 |
| 180 | 6.1153356819063488e-05 | 6.1153308005679761e-05 | -0.244067 |
| 240 | 6.5817049818294642e-05 | 6.5816987106668941e-05 | -0.313558 |
| 300 | 5.4660485988812502e-05 | 5.4660367722690732e-05 | -0.591331 |
| 360 | 5.4077372201617811e-05 | 5.4077212586815052e-05 | -0.798074 |

Seven of eight checkpoints improve; the only worsening is 0.002707 floor
units. Days 30, 240, and 360 all improve, so Decisions 43/45/59 pass. The new
certified snapshot digests are day030 `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
day240 `2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
and day360 `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`.

### 4.3 Blast radius

Twelve certified control registries were measured: LOCK_EXCHANGE, OVERFLOW,
the six flat VORTEX flux/vector cards at 30/15/10 km, SMT flux/vector, SMT-1,
and SMT-2. Every comparison is 0/50 moved with zero ratchet violations. The
generic NEMO-GYRE recipe is exercised by the focused unit gates below.

The private DINO month gate passes its fixed bar but moves from the prior
`2.053801169e-03 K` reference to `2.056821682e-03 K`, a registered worsening
of `3.020513e-06 K` (+0.147%). Its authoritative line is:

```text
PASS: DINO from-rest day-30 T RMS 2.056821682e-03 K < bar 2.244317642e-03 K
```

ORCA2 is **UNMEASURED-with-spec** on this lane. Its rung-0 isoneutral path
shares the two landed statements, so its next merge must remeasure both ORCA2
ladders; it may not infer inertness from GYRE or VORTEX.

SMT-4 itself remains unmeasured: no SMT-4 ladder, 100-day trajectory, or
first-owner claim is made before the operator-run acquisition.

## 5. Configuration choices

**UNASKED list: EMPTY.** ORCA2 rung 0 selects level Laplacian momentum
diffusion and a file-backed 3-D coefficient at
`orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:388-392`.
Decision 93 explicitly authorises the idealised-card stand-in coefficient
mode 20. The complete companion tuple comes from
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:1104-1134`. Geometry, EOS, tracer
diffusion, vertical mixing, drag, momentum advection, vorticity, pressure
gradient, barotropic program, timestep, and run length remain SMT-3's admitted
values.

## 6. Compiled-source record

For the landed statement, the record build forms NEMO's four-point mask sums
with the closed bottom W level while constructing the tensor coefficients at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`.
It accumulates the horizontal/vertical flux divergence and multiplies by the
stored horizontal-area reciprocal before dividing by the live
`e3t_3d*(1+r3t(Kmm)*tmask)` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`
and `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
The compiled stage program defines stage 3 as `Kbb=N`, `Kmm=N+1/2`,
`Kaa=N+1` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:225`
and passes that `Kmm` to `tra_ldf` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:531-532`.
Those are the statements transcribed; no stabiliser or card switch was added.

For the separate next-rung acquisition, the new SMT-4 target has not been
built, so the remainder of this section cites only the compiled SMT-3 base
whose scientific program the new deck changes.

NEMO declares and
reads `namdyn_ldf` reference-before-card at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-185`.
For z/partial-step geometry, level Laplacian resolves to `np_lap` at
`:221-276`; coefficient mode 20 computes `zUfac=0.5*rn_Uv` and calls
`ldf_c2d` at `:311-346`.

The compiled dispatcher maps `np_lap` to `dynldf_lev_lap` at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90:81-90`.
That operator forms curl and divergence with the live partial-cell
`e3f_3d`, `e3t_3d`, `e3u_3d`, and `e3v_3d` factors and adds them to the U/V
RHS at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.
The stage program calls `dyn_ldf` before the implicit vertical momentum solve
at
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:387-405`.

These citations prove the branch requested by the deck; they do not pretend
the absent SMT-4 executable has run. The next receipt must cite the new
target's own `BLD/ppsrc/nemo`, confirm its resolved `ocean.output`, and admit
its passive record before making a physics claim.

## 7. Independent review and final gates

The required separate review was attempted on the clean committed tree with:

```text
codex exec --sandbox read-only -C <writable-clone> <adversarial review prompt>
```

It exited 1 before reading the diff. Its verdict, quoted verbatim, was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

**Independent review unavailable in-sandbox.** There is no `SHIP`, `HOLD`,
or `DO NOT SHIP` verdict, and none is invented. The full log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/codex_review.log`.

The first citation-gate pass correctly refused one off-by-one endpoint and
three ambiguous repeated symbols. After correcting and committing those
pins, the final round gate reports **PASS: 8 citations, 0 failures, 0 unmapped**.
The cumulative default receipt also reports **PASS: 274 citations, 0
failures, 0 unmapped**. Shifting
`VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`
by two lines exits **1** with one citation failure, so the plant fires.

The final focused suite was launched only after the host battery census
returned zero:

```text
tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round224_smt3_record.py
tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round237_smt4_record.py
tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py
============================== 33 passed in 3.52s ==============================
```

## 8. OPEN — round 238

1. Operator runs the committed acquisition. Admit only if the plain and
   instrumented step-10 restarts are byte-identical, every named group parses
   to EOF, all three plants exit nonzero, both runs reach `STOP 0`, and the
   resolved `ocean.output` tuple is exact.
2. Read and cite the new targets' compiled source. Add the explicit
   `VORTEX_SMT4_VEC-zps` legoESM card, prove geometry and initial state against
   NEMO, and score the 50-row kt=1..10 registry plus the 100-day checkpoints.
3. Compare each stage's pre/post-`dyn_ldf` accumulator from NEMO's recorded
   entry. If post-LDF is the first non-bit boundary, walk the level-Laplacian
   div/curl operator operand by operand over partial cells. If pre-LDF is
   already non-bit, keep R25-P5 as REFUTED and walk the earlier boundary.
4. Name the first non-bit statement with the new compiled-source citation and
   write the exact ORCA2 rung-0 operand pointer. Any production candidate is a
   later round under all GYRE/year, SMT/flat VORTEX, tanks, generic GYRE,
   private DINO-month, citation, plant, and independent-review gates.

The cited shared production physics landed in round 237. No configuration
default or carried-state layout changed. SMT-4 remains acquisition-only.
