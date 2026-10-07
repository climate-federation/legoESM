# ORCA2 round 163 — pinned GYRE merge and external-mode consumer walk

Date: 2026-10-07. Base `86b84f6f3`; preregistration `e6000fba3`; pinned
GYRE parent `ca822f33d`; merge `736cc0d32`; measurement tip `3881ac466`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round163/`.
Verdict: **LANDED** for the pinned GYRE merge and **HELD** for the private
U-cyclic/V-fold halo pair. The halo pair fails Decision 96, and no production
halo statement lands.

Rung-0 numbers below are **independent**: its own climatological T/S, zero
velocity and zero sea surface are the entry state. Rung-7 numbers are **given
NEMO's entry** under Decision 52. Those populations are kept in separate
tables and artifacts. Sea ice, all six sea-ice selectors, and the shipped
card's `unmeasured_features` tuple are unchanged.

## Pinned merge and conflict resolution

The fetched parent is exactly `ca822f33d930b9df4987561cc781cd3c1028e0a6`.
The merge produced exactly the two preregistered textual conflicts and no
package-code conflict:

1. `nemo_testcases_l2_gyre_phase3_round8_receipt.md`: kept both lanes'
   scientific prose, then mechanically re-anchored the cumulative source
   citations on the merged tree.
2. `nemo_testcase_receipt_citation_gate.py`: kept the union of both generated
   `CITATION_MAP` additions. Follow-up commits `e0ead9374`, `22c31385a`, and
   `3881ac466` correct only mechanically shifted endpoints; no entry from
   either parent was dropped.

`ocean_model_latlon_cgrid.py` auto-merged. The imported authorised SMT-3
physics is NEMO's closed deepest W mask and live stage-3 Kmm tracer-LDF
divisor, cited in the parent landing from
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259`,
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`,
and `VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331`.
The SMT-4 additions are records and held measurement tools only; no SMT-4
physics statement is in this merge.

The cumulative citation gate passes with no unmapped span. Its real compiled
line-shift plant fires. R163-P1 is **CONFIRMED**.

## GYRE predicate

The 70-row certified ladder is array-identical to round 237: zero row moves,
zero cellwise worsening, first-over-bar remains kt=3, and both residual
archives have SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.

The fresh 360-day member is also byte-identical to round 237 at all 360 daily
snapshots. Its headline T RMS distances from NEMO are:

| day | merged tree T RMS (K) | pinned T RMS (K) | snapshot SHA-256 |
|---:|---:|---:|---|
| 30 | `2.3432419318363155e-06` | `2.3432419318363155e-06` | `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b` |
| 240 | `6.5816987106668941e-05` | `6.5816987106668941e-05` | `2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe` |
| 360 | `5.4077212586815052e-05` | `5.4077212586815052e-05` | `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646` |

R163-P2 is **CONFIRMED**.

## ORCA2 ladders

### Independent rung 0

All 200 rows are unchanged from round 161, with zero exact-row loss and the
first debt unchanged at kt=1 stage-1 T. Thus the SMT-3 pair is selected by the
rung-0 tracer-LDF configuration but is numerically inactive over this
ten-step trajectory.

### Given NEMO's entry rung 7

The merge moves 183/200 rows and loses no exact row. By RMS, 108 move toward
NEMO and 75 away; by maximum, 107 move toward, 34 away and 42 are equal. The
first debt remains kt=1 stage-1 T. The complete moved-row register is
`ladder_compare.json`; the kt=10 stage-3 summary is:

| field | RMS before -> after | max before -> after | max direction |
|---|---|---|---|
| S | `0.002782408976381055 -> 0.00278240897495421` | `0.28126856934547106 -> 0.2812685693386001` | toward |
| T | `0.009425335615815026 -> 0.009425334991279432` | `1.2101222368215616 -> 1.2101222368298696` | away |
| ssh | `0.027586454634018594 -> 0.027586453643536082` | `0.3037533787093492 -> 0.30375337870920743` | toward |
| u | `0.005331082847676189 -> 0.005331083011908327` | `0.24444966796128276 -> 0.24444966796129985` | away |
| v | `0.005584658859749724 -> 0.005584658780475851` | `1.2990882244341995 -> 1.2990882239371575` | toward |

At least one certified ORCA2 boundary moves, neither loses an exact row, and
neither advances its first debt. R163-P3 is **CONFIRMED**.

## Independent rung-0 month

The unmodified round-71 protocol again first becomes non-finite at step 36,
field T, cell `[86,159,0]`; the gate reports the same boundary after clean
progress records at steps 10, 20 and 30. R163-P5 is **CONFIRMED**. No terminal
month score is claimed.

## Decision-96 halo verdict

The private U-cyclic/V-fold pair moves 195/200 independent rung-0 rows and
loses no exact row. Movement is nevertheless overwhelmingly adverse: by RMS,
10 rows move toward and 185 away; by maximum, 65 move toward, 75 away and 55
are equal. The first debt stays kt=1 stage-1 T, but the kt=10 stage-3 SSH
maximum worsens from `0.42832517646246693` to `1.0665707503278319` m. The S
maximum also worsens from `0.4156673855238111` to
`0.41567240155913865`, a `5.0160353275430225e-6` increase, and the dedicated
salinity veto fires.

The pair is not a Decision-96 net improvement. It remains private and
**HELD**. R163-P6 is **CONFIRMED**.

## First live consumer after the exact local halo

The preregistered accumulator premise was wrong. The rung-0 namelist resolves
`nn_bt_flt=3`; compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:220-223` therefore
sets `ll_bt_av=.FALSE.`. The weighted velocity/SSH accumulator at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:854-856` is dead and
cannot be the next consumer. No arm was
built for a branch this deck does not execute. R163-P7 is **REFUTED** and
retained here.

The next live source boundary is the midpoint face-depth and transport chain:
NEMO builds the midpoint U/V depths at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, forms the
unmasked V transport at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`, then takes
its directional difference and advances SSH at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:584-591`.

The admitted round-146 replay gives the following source-ordered result at
substep 2:

- associated entry U/V and the midpoint U/V values are bit-exact;
- the three V-transport operands `e1v`, `va_e`, and `zhvp2_e` are bit-exact;
- production alone multiplies by legoESM's extra compact `v_mask`, which is
  zero on 68 northern-fold cells (row 147) where NEMO's literal transport
  statement has no such factor;
- production `zhV` differs on those 68 cells by at most
  `155776.5627856178`, while the one-variable unmasked replay is bit-exact;
- materialising that exact `zhV` closes V difference, divergence, and SSH to
  zero unequal cells.

This is a measurement, not a production landing. The exact unit already
measured in rounds 153-154 comprises the raw reference face depth, no extra V
mask, the certified seven-array boundary association, and materialised `zhV`.
It must be promoted atomically under the full gates; landing only the halo
would expose the compensating transport error measured above.

## Shared blast radius and validation

The DINO month, LOCK_EXCHANGE, OVERFLOW, shared-card construction gate,
focused tests, and full fidelity battery are recorded below. Every synthetic
violation named here fires.

- DINO: PASS at day-30 wet-3D T RMS `2.056821682e-03 K` against the fixed
  `2.244317642e-03 K` bar. This is `3.020514e-06 K` (`0.147%`) farther from
  NEMO than round 160's `2.053801168e-03 K`, registered here but below the
  bar. Score SHA-256 is
  `0d67986226155d95b2641d8e8a8d350ace2cbd376f6c29bdbf92cba837ee79a1`;
  the planted `6.981690958e-03 K` value fails.
- LOCK_EXCHANGE and OVERFLOW: each compares 50/50 rows array-identically to
  round 160 with zero worsening ULPs. LOCK_EXCHANGE's first debt remains kt=8
  U; OVERFLOW's remains kt=2 T/U. Each `worsen-3ulp` plant fails.
- Shared-card census: PASS; its `card_reference` plant moves the GYRE row and
  fires.
- Focused pytest: 178/178 pass (boundary association, dead final association,
  trajectory gate, citation gate, GM/Redi unit coverage, and SMT card).
- `tests/ocean/fidelity -n 12`: the one required invocation collected 2,684
  tests and was interrupted after five minutes with no progress at 99%:
  2,654 passed, 7 skipped, 5 failed, and 18 were unfinished. Isolated reruns
  reproduce four pre-existing reds: SI3 `MY_SRC` scalar-math provenance, the
  moved certified-year-harness spread record, the allow-dirty scope ratchet,
  and the worktree-stamp ratchet. The fifth refusal is the year-owner
  acquisition's clean-tree guard seeing this uncommitted receipt; after the
  receipt commit its isolated rerun passes 1/1. No focused round-163 test
  fails.
- Independent review: **independent review unavailable in-sandbox**. The
  required separate `codex exec --sandbox read-only` attempt failed before
  reading the diff because its in-process app-server client could not create
  state on the read-only filesystem.

R163-P4 is **CONFIRMED**: DINO stays below its fixed bar; both tank registries
are array-identical; shared-card and focused gates pass; known battery reds
are retained rather than relabelled. ASKED choices are the pinned merge and
Decision-96 predicate from
Decision 97. UNASKED choices are empty. No configuration, forcing,
carried-state policy, stabiliser, sea-ice selector, or production halo
statement changed in this round.

## Prediction disposition

| prediction | disposition |
|---|---|
| R163-P1 exact conflict inventory and union | **CONFIRMED**. |
| R163-P2 GYRE pinned | **CONFIRMED**: ladder, residual archive, year values, and all daily snapshots are identical. |
| R163-P3 SMT-3 preserves both boundaries while moving ORCA2 | **CONFIRMED**: rung 0 is unchanged; rung 7 moves 183 rows without exact-row loss or earlier debt. |
| R163-P4 shared blast radius admitted | **CONFIRMED**: DINO passes its fixed bar; tank registries are exact; shared-card/focused gates pass; known battery reds are unchanged. |
| R163-P5 month boundary unchanged | **CONFIRMED**: step 36 T `[86,159,0]`. |
| R163-P6 halo still ineligible | **CONFIRMED**: 185/195 moved RMS rows go away and SSH maximum worsens. |
| R163-P7 accumulator materialisation is non-vacuous | **REFUTED**: `ll_bt_av` is false under `nn_bt_flt=3`; the statement is dead. |

## OPEN

1. Promote the complete, already-measured V-transport compensation unit
   atomically: raw reference face depth, no extra compact V mask, seven-array
   external-mode boundary association, and materialised `zhV`. Pre-register
   and run both ORCA2 ladders, the independent month, GYRE year, DINO, tanks,
   and the registry before landing.
2. Keep the local U-cyclic/V-fold halo pair separate and **HELD** until that
   consumer unit removes the exposed salinity/SSH regression under Decision
   96.
3. The independent rung-0 month remains first non-finite at step 36 T
   `[86,159,0]`; the V-transport unit is the next causal month arm.
4. The given-entry rung-7 first non-bit row remains kt=1 stage-1 T and is
   unattributed; the merged V maximum improved by `4.9704196e-10` at kt=10
   stage 3, but the 75 RMS-away rows remain registered debt.
