# Lane 3b round 20 receipt — OVERFLOW producer and ORCA2 SI3 admission

Status: **MIXED; ORCA2 EXACT-INPUT RUNG STOPPED AT SELECTOR ADMISSION.**
Predictions were frozen at localgit commit `0ff0b956256`; the executable
ORCA2 admission gate landed at `9e4fe6f537f`.  Measurements in this receipt
are Codex-internal.  The user-reported Round-19 review verdict is not relabelled
as a branch review artifact.

## 1. OVERFLOW non-finite producer history

The first-parent integration revision that first contains the relevant fix is
merge `c9526e58568` (`fidelity/nemo-branch-isomorphism-audit` into lane-2
GYRE).  Its second-parent ancestry contains the isolated implementation commit
`7d66b7f37e8`, whose direct parent is `8b4bcdb4af0`.  A source-level
before/after census at that exact child commit gives:

| kt1 workspace | direct parent dry nonzeros / maximum | fixed dry nonzeros / maximum | wet maximum before / after |
|---|---:|---:|---:|
| stage-1 velocity | `524 / 0.02737944122559207` | `0 / 0` | `0.06219808806021984 / 0.06219808806021984` |
| stage-2 velocity | `673 / 0.018554871083636473` | `0 / 0` | `0.07078054307233649 / 0.07078054307233649` |
| no-stage-mean private arm | `673 / 0.052952333532307376` | `0 / 0` | `0.05150200612099589 / 0.05150200612099589` |

The producer is `_replace_stage_mean`: before `7d66b7f37e8`, it applied the
2-D state U/V mask broadcast across depth after inserting the barotropic mean.
At a staircase face that is wet at some levels, the broadcast left the same
finite increment below the local seabed.  The value is constant down the dry
part of the argmax column (`ptp=0`, one unique value), which is the signature
of that broadcast.  NEMO instead multiplies the stage velocity and the
barotropic correction by three-dimensional `umask(ji,jj,jk)` at
`stprk3_stg.F90:367,375,382,444`.  Its UP3 stencil reads the neighbouring
stored value rather than skipping it (`dynadv_up3.F90:142-143,160,166-176`).

Commit `7d66b7f37e8` reuses `compute_face_masks_3d` and applies that live mask
at the producer.  LOCK is the geometry control: its live 3-D mask equals the
2-D broadcast and remains exactly zero on dry faces.  The Round-19 accepted
stage artifact then shows the merged OVERFLOW path finite.  Thus the named
root cause is **a below-seabed WS-RK3 velocity written by the wrong-rank face
mask, then consumed by UP3**.  The Round-18 `jnp.where` episode acted at the
consumer and hid the invalid workspace; it was not the producer fix and
remains reverted.

The direct census measures the invalid finite precursor, not a NaN value in
isolation.  The non-finite status belongs to the old complete stage execution;
the combination of the one-variable producer discriminator, unchanged wet
values, and first-parent incorporation identifies the producer without
claiming that the census itself reproduced every downstream NaN bit.

## 2. Rule-11 stage-clock retraction

The Round-19 receipt now retracts its stage-clock ownership label in place.
At GYRE fetched tip `b2702b318f7`, Round 21's production-JIT
post-`tra_adv_trp` `ww` rows are AT_BAR at stages 1--3:

| stage | production normalized maximum | clock-only arm |
|---:|---:|---:|
| 1 | `1.6543612251060553e-23` | `3.9555664454351783e-7`, DEBT |
| 2 | `6.107901643091556e-21` | `1.9777832227175958e-7`, DEBT |
| 3 | `2.68242819152769e-17` | `3.1361945593164585e-8`, DEBT |

The production path pairs a full-step surface-height increment with full
`dt`, giving the same stretching rate as NEMO's stage-local increment and
stage-local `rDt` (`stprk3_stg.F90:123-124,177-178,221-222`;
`sshwzv.F90:334-335`).  Changing only the denominator is not a faithful arm.

The Round-19 value `1.5711182355104825e-12` is exactly the historical GYRE
stage-2 instantaneous-u row in the GYRE receipt.  It is not among Round 22's
seven OVERFLOW trajectory rows.  The fetched branch has no Round-23 receipt or
artifact, so the honest current label is
**OVERFLOW_KT1_STAGE2_U, OWNER_UNASSIGNED_PENDING_GYRE_OVERFLOW_WALK**.

## 3. Combined bottom-plus-top drag association

The merged gate retains this source-literal statement:

```python
literal = sr(jnp.asarray(0.5) * sr(sr(bot_e + bot) + sr(top_e + top)))
```

It is NEMO's single half of the combined bottom and top neighbour sums at
`dynspg_ts.F90:1611-1612`; the implicit top-drag consumer is
`dynzdf.F90:302,478`.  On the first retained frame with non-zero top drag,
kt5 `PRE_DYN_SPG_TS.combined_drag_u` is **1 / 1 bit-identical**, non-bit
`0 / 1`, maximum absolute and normalized error `0`.  The current retained
sample does not distinguish separately halving top and bottom (zero moved
steps), so source inspection establishes survival of the nesting while the
oracle row establishes that its output remains exact.

## 4. ORCA2 SI3 exact-input admission

Candidate root:
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2q_zdf_schemafix_a_10step_np2`.
The dimension-derived schema gate validates all five present SI3 streams:
5 reassociation records, 140 thermodynamic records, 10 exchange records,
25 ZDF-input records, and 15 bulk records.  This rules out a malformed-header
stop.

The next mandatory admission check stops before model execution:

| selector | ORCA2 resolved | implemented shared SI3 identity | status |
|---|---:|---:|---|
| HFN categories (`jpl`) | 5 | 1 | UNSUPPORTED |
| ice layers (`nlay_i`) | 10 | 3 | UNSUPPORTED |
| snow layers (`nlay_s`) | 5 | 3 | UNSUPPORTED |
| conductivity | P07 | P07 | VERIFIED |
| salinity (`nn_icesal`) | 4 | 2 | UNSUPPORTED |
| new-ice salinity fraction | 0.75 | 0.75 | VERIFIED |
| drainage / flushing | T / T | T / T | VERIFIED |
| melt ponds (`ln_pnd`) | T | F | UNSUPPORTED |
| lateral melt (`ln_icedA`) | T | F | UNSUPPORTED |

These are NEMO's actual cfg-over-ref values printed in `ocean.output`:
`jpl=5`, `nlay_i=10`, `nlay_s=5` at lines 702--704; BL99/P07 at
733--738; lateral melt at 729; salinity option 4 and `rn_sinew=0.75` at
764--767; and level-ice ponds at 791--802.  The shared gate deliberately
accepts only `SI3ThermoConfig()` plus `jpl=1`; its BL99 implementation also
contains source-specific three-layer remaps and a three-plus-three conduction
solve.  Substituting ORCA2's five-category, 10+5, salinity-4, pond/lateral
program into that identity would be a Frankenstein configuration.

Accordingly the result is **STOP_SELECTOR_GAP**, `scored_physics_rows=0`,
`bit_identity=NOT_MEASURED`.  This is not reported as `0 / 0` bit-identical.
It is also not evidence against SI3: no thermodynamic operator row was
admitted.  Extending the shared SI3 identity to this full resolved deck is a
new scope decision; it was not authorized by the binding ORCA1-only scope and
is not undertaken here.

### Existing frame and time-level coverage

The retained config-local writer is SHA-256
`1a4956577681aa5a86e67f135d7382e30f280e61f5f292e7d64ce321cd088c97`.
Its execution registry is:

| stage | count | NEMO boundary / time level | coverage |
|---:|---:|---|---|
| 0 | 5 global | `ice_thd` ENTRY before frazil, `icethd.F90:117-125` | VERIFIED recorded |
| 1 | 25 packed | each category immediately POST_ZDF, `:148-159` | VERIFIED recorded |
| 2 | 25 packed | each category POST_DH, `:161-162` | VERIFIED recorded |
| 3 | 25 packed | each category POST_TEMP1, `:164-165` | VERIFIED recorded |
| 4 | 25 packed | each category POST_SAL, `:167-168` | VERIFIED recorded |
| 5 | 25 packed | each category POST_TEMP2, `:170-171` | VERIFIED recorded |
| 6 | 5 global | after per-category DA, PND, ITD remap and DO, `:176-197` | RECORDED BUT COMPOSITE |
| 7 | 5 global | after `ice_cor`, aging, LBC and velocity correction, `:199-232` | VERIFIED recorded EXIT |

The global payload contains `a_i,v_i,v_s,sv_i,oa_i,t_su,a_ip,v_ip,v_il,
e_i,e_s,szv_i` (`icethd.F90:262-266`).  The packed payload contains
`a_i_1d,h_i,h_s,t_su,e_i,e_s,sz_i` (`:275-280`).  PRE_ZDF records contain
the 13 exchange/basal operands plus every snow-layer temperature at the
category's live 1-D time level (`:284-301`).

For a future authorized five-category identity, the current composite stage 6
must be split with WRITE-only full-state frames at PRE/POST_DA per category,
PRE/POST_PND, PRE/POST_ITD_REM, PRE/POST_DO, and PRE/POST_COR.  That is the
exact ORCA2-lane handoff; it preserves current records and adds boundaries
without modifying shipped NEMO source.

### Input hashes

| input | SHA-256 |
|---|---|
| `ocean.output` resolved deck | `8cc5ad291ca57d0aa1fbee0cce57da2e8906074f5e62afe59deb171b6d731c41` |
| `oracle_si3_thd_frames.bin` | `f10e6c6a8fbee5dfbefc50afbf5ee05e6ece70821181a847f7ebb6e02acd5a6e` |
| `oracle_si3_zdf_inputs.bin` | `a41c354a18c7ebccc310f6e39477d3d6270b591e3899d9f7f4fcb80f7203b07b` |
| `oracle_si3_exchange_frames.bin` | `3d96ff548a788529676072f0f6e352b380c711a7acab6362dd1519bb3886a26c` |
| `oracle_si3_bulk_operands.bin` | `3895867017affa1c9827ae0374d50f8e69bfed78b603c70647a6516f60cc1dbc` |
| rank-0 final ice restart | `4015292b4663e27a8e44109c1c001603d1d82519339d8cc1285c6d4f268d8be2` |
| rank-1 final ice restart | `f6bea2738a8e10286653d09ee8bf1bdd18a91111c9b22c3260200975f847a229` |

## 5. Gates, controls, and hygiene

- ORCA2 header gate: PASS, all registered streams VALID.
- ORCA2 selector admission: expected nonzero exit `1`, six named unsupported
  selector rows, no physics score.
- Synthetic supported deck: admitted; one-variable `jpl=5` plant produces
  exactly one UNSUPPORTED row.  Together with the header bad-count plant and
  drag/exchange test, the focused selection is `6 passed in 2.52 s`.
- Hardcoded-constant ratchet for both newly touched Python files:
  `2 passed, 4457 deselected in 0.94 s`.
- The older whole-ORCA2 inventory gate was not used because its fixed inventory
  predates five additional retained oracle files.  Its nonzero output is a
  stale-harness result, not a SI3 verdict.
- Two broad test attempts ended without a terminal pytest summary; their
  partial logs are retained but are not cited as passes.  The completed focused
  selections above are the verification record.

No source under `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` was changed.
No ORCA2 config, oracle input, or retained runtime artifact was changed or
deleted.  No GPU, `mpirun`, push, scalar-libm LOG/LOG10/POW extension, or
prognostic `uu_b/vv_b` construction was used.

## 6. Rule 8/11/12 register

| item | boundary | result | owner/disposition |
|---|---|---|---|
| 3-D stage face mask | OVERFLOW WS-RK3 stage velocity -> UP3 | invalid dry workspace removed; wet values invariant in direct discriminator | integration source commit `7d66b7f37e8`; CONFIRMED producer |
| stage clock | post-`tra_adv_trp` `ww` | production AT_BAR stages 1--3; clock-only arm red | prior owner label RETRACTED under Rule 11 |
| historical `1.571118e-12` row | OVERFLOW kt1 stage-2 instantaneous u | exact historical GYRE row; not a Round-22 seven-row trajectory debt | OWNER_UNASSIGNED |
| combined drag association | kt5 PRE_DYN_SPG_TS U | 1 / 1 bit-identical | literal statement retained; GYRE shared owner-of-record |
| ORCA2 exact-input rung | selector admission before `ice_thd` | six unsupported selectors; zero scored rows | scope/identity decision required, no numerical owner assigned |

This was true at the Round-20 boundary but is superseded by User Decision 9
and the Round-21 scalar-libm extension; see
`nemo_testcases_l3thd_round21_receipt.md`.

## 7. ASKED / UNASKED

| choice or action | status | disposition |
|---|---|---|
| locate and name the OVERFLOW producer | ASKED | completed with first-parent incorporation plus direct source-commit discriminator |
| retract stage-clock ownership | ASKED / Rule 11 | completed; boundary retained, owner unassigned |
| cross-reference GYRE Rounds 22/23 | ASKED | Round 22 checked; Round 23 absent at fetched tip and not invented |
| verify combined drag nesting and row | ASKED | source intact; kt5 row 1 / 1 bit-identical |
| start ORCA2 SI3 exact-input rung | ASKED | headers pass; stopped non-vacuously at selector admission |
| extend SI3 to ORCA2's five-category 10+5/salinity-4/pond program | UNASKED and outside binding ORCA1 scope | not implemented; decision required |
| add a fail-closed selector admission gate | UNASKED implementation aid within requested rung | landed; no physics added |
| change shipped NEMO/config-local ORCA2 writer or delete evidence | UNASKED / forbidden | not done |
| implement prognostic `uu_b/vv_b` | GYRE-owned User Decision 8 | not implemented here |
| extend scalar-libm LOG/LOG10/POW | ASKED later, User Decision 9 | implemented in Round 21 |

## 8. FLAGGED FOR FUTURE DELETION

Delete nothing.  Retain and flag all roots already named in Round 19:
`/tmp/codex-si3thd-r18-*`, the eight
`c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}` roots,
`c1d_omip_l3_coupled10m_r17_oracle_a`, and
`/tmp/codex-si3thd-r19-c83`.  Also flag the Round-20 diagnostic trees
`/tmp/codex-si3thd-r20-mask-pre` and
`/tmp/codex-si3thd-r20-mask-post`.  Partial/non-authoritative outputs
`round20_orca2_si3/orca2_phase1_schema.*`, `focused_tests.log`,
`constants_ratchet.log`, and `round20_overflow_bisect/current_full.log` remain
retained but are not evidence.

## 9. Runtime artifacts (not committed)

| artifact | SHA-256 |
|---|---|
| `round20_overflow_bisect/pre_phantom/census.json` | `60cca559066789da62c00e5378c229274c060f3ec063ddfd7cc19e277395e601` |
| `round20_overflow_bisect/post_phantom/census.json` | `d521cbfbacabf9a7ae3dae27c01eeb455dffbe63bd725b51eafca89f3459f4c6` |
| `round20_overflow_bisect/drag_merged_kt5.json` | `855652b31f5db0e123a2a50b7e11b95877f3eb13376cfc00d5eb39fc93c3f26c` |
| `round20_orca2_si3/si3_stream_headers.json` | `741bb4b0f7bf41ead4875726c1c01ae35689bb4c95897b2960213b8576b5494d` |
| `round20_orca2_si3/orca2_si3_admission.json` | `c4bdf3141b35b565fb1b6ea2d9031158a5caa50674b8651445a23c9f906e3833` |
| `round20_orca2_si3/focused_fast_tests.log` | `a5c963535256bea8ba91436bb3ce962ef376784a8f459b9ebe3e0d2e3a5446be` |
| `round20_orca2_si3/constants_ratchet_touched.log` | `e2cbfbaefa1437545a96825a80edee2130068ab59965b242f23352cf5ed008b1` |

Result/receipt commit: `7c698bf9feaa35453763f3a504b98683f3ce8ebd`.
The following documentation-only ledger commit records this hash so the
receipt does not self-hash.
