# NEMO testcase Lane 4 — ORCA2 Phase-2w handoff receipt

Date: 2026-09-06

Parent: `26af77048a8b28f0d4f29bc9e36699919615d0a2`

Status: **STOPPED AT THE USER-SHELL MPI BOUNDARY — Phase-2v replacement
admitted; ORCA1-ice variant twins built and staged, not executed.**

## 1. Phase-2v twin admission and V2 pin

**CONFIRMED:** both unchanged launchers completed at `time.step=10` with
`MPIRUN_RC=0` and `RUN DONE`.  The retained acquisition gate decoded the
57,266,632-byte `NEMO_L4_TKEW_1` stream from its 15-integer base header and
24-field extent table to exact EOF.  The header resolves `kt=2`, `Kbb=3`,
`Kmm=3`, binary64, `17` 3-D fields, `7` 2-D fields, and a header-derived
7,629,792-value payload.  Every unowned slot is canonical zero and the frame
is non-vacuous.

Twin A and B are raw-identical for **101 / 101** `oracle_*.bin` streams.  All
**100 / 100** inherited streams are raw-identical to the Phase-2s root.  The
ordinary-output identity control passes with dynamic counts: four restart
shards are exact bytes; eight history payloads are exact after excluding only
the global timestamp attribute; `ocean.output` is exact after the registered
WRITE-only dump notices.  Timing and launcher provenance are excluded.

Twin A is therefore pinned as the single `VARIANT_ORACLE_V2` root:

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2`

Twin B is its independent reproducibility witness.  Phase-2s and earlier roots
remain retained witnesses.  The Phase-2u twins remain
`REJECTED_NONREPRODUCIBLE_TKE_RECORD`; nothing was deleted.

The complete 105-file output/hash inventories are external evidence:

- A manifest SHA-256 `9a98cd5e97caffe54707d33e06e34a42c1c39e8cd9a875ffc187988685079047`;
- B manifest SHA-256 `3e78e3829d87ad08eda5313d44c50b42894ca1d81094ec45c1b22ce2d208500b`;
- admission JSON SHA-256 `63d77e382f951df3bf9a5ebc1243bd3126322249f85a69057226493690f61abd`;
- admitted TKE record SHA-256
  `31675493f022f71a609142f53bbe220c111b09e9a9a352926a1aff7358770a52`.

All eight binding plants exited nonzero: magic, extent, count, truncation,
trailing byte, canonical unowned slot, undefined `jpk` workspace, and twin
identity.  This is **CONFIRMED**, not inferred from the user's byte census.

## 2. Ordered TKE walk: first boundary

**CONFIRMED / production JIT, CPU, fp64 + scalar-libm:** the post-NCAR
operand-substitution boundary hands the card the oracle's own post-SBC `taum`.
The JIT hand-off is `0 / 8,794`, so no cast or reshape changes the forcing
operand.  This row certifies the supplied-input boundary, not the open NCAR
bulk operator.  The restored `nn_eice=1` attenuation is `0 / 8,794`, and the
no-Stokes `zWlc2=zcsd*taum` statement is `0 / 8,794` (`zdftke.F90:253-258,
326-333`).

The first over-bar statement is the `zpelc` potential-energy accumulation at
`zdftke.F90:339-345`: **39,290 / 242,135** defined rank-zero cells differ,
maximum absolute error `1.7763568394002505e-15`, maximum four row-scale ULP.
The first differing cell is zero-based rank-zero `[j=1,i=49,k=2]`, classified
coast-or-bottom-adjacent.  The first two values are
`0.00723748010384538` versus oracle `0.0072374801038453795`.

This is **CONFIRMED `GYRE_OWNER_SHARED_TKE`**: the executed statement is a
geometry-independent scalar recurrence in the one shared TKE implementation;
the forcing operand and preceding ORCA2 selector rows are exact.  Lane 4 did
not change it and did not walk EVD/IWM past the open TKE boundary.  Reproducer:
`nemo_testcase_l4_orca2_phase2v_tke_walk_gate.py`; its data result is
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2w/tke_walk.json`, SHA-256 to be
pinned in the final manifest.  Target-bit plants for `taum_input`,
`ice_fraction`, `zWlc2`, and `zpelc` each exited nonzero.

## 3. User Decision 12 — NEMO `nn_eice=2`

**CONFIRMED identity restoration:** the one shared
`nemo_tke_effective_ice_fraction` dispatcher now implements all four NEMO
numbers and raises otherwise: 0 is zero attenuation, 1 is
`TANH(10*fr_i)`, 2 is raw `fr_i`, and 3 is `MIN(4*fr_i,1)`
(`zdftke.F90:253-258,828-834`).  The raw-fraction behaviour that legoESM once
called mode 1 remains selectable, now under NEMO's number 2; mode 1 keeps its
restored NEMO meaning.  Defaults remain zero.  The ORCA2 identity card selects
1 and the ORCA1 CORE2 driver selects 3.

The mode-2 JIT/fp64 bit test is exact, its gradient is identically one, and the
closed dispatcher test covers 0/1/2/3 plus a raising unknown-mode control.
Targeted evidence: **93 / 93 tests passed** across the full TKE source-term,
under-ice, and tripole-vmix files; the post-change focused run passed **29 / 29**.
The ORCA1 mode-3 test plus C1D ice transport constructibility passed **9**, with
one existing skip and no failures.

**CONFIRMED Rule 12:** the GYRE, LOCK, and OVERFLOW actual NEMO kt=1 entry gate
remains `0 / n` for every T/S/u/v/ssh row; its binding LOCK-T plant exits
nonzero.  GYRE selects `eice=0`; LOCK and OVERFLOW have no TKE call path.  C1D
likewise has no ocean-TKE call path, and its ice transport tests are unchanged.
The ORCA1 CORE2 builder still resolves `eice=3`, whose established mode-3 bit
test is unchanged.  Thus only the newly selected mode-2 branch moves.

## 4. User Decision 11 — ORCA1-ice ORCA2 variant handoff

**CONFIRMED deck construction:** `ORCA2_ORCA1ICE_OMIP_L4` is a `makenemo`
copy of `ORCA2_ICE_PISCES`, built with `conda-scalarmath` and the same
`key_top key_xios` exclusion as the accepted oracle.  The comparison ocean
namelist is byte-identical to the pinned icebergs-off 10-step deck (SHA-256
`76366c96fbfe3a65af72747b1231b5bd67a8515ff71f9270f4866aa632e7bb84`),
including `ln_icebergs=F`, `nn_itend=10`, `nn_stock=10`, and `(jpni,jpnj)=(2,1)`.
The first direct filesystem copy that `makenemo` could not recognize is
retained, not deleted, at
`/tmp/nemo-orca2-phase2p/cfgs/ORCA2_ORCA1ICE_OMIP_L4_pre_makenemo`.

**CONFIRMED selector delta:** the config-local `namelist_ice_cfg` resolves the
User-Decision-11 target: `jpl=1`, `nlay_i=nlay_s=3`, `ln_dynALL=T`,
`ln_landfast_L16=T`, H79 strength with `rn_pstar=2e4` and `rn_crhg=20`,
ridging and rafting on, EVP with inherited `ln_aEVP=T`, Prather on and UMx
off, `ln_icedA=F`, `nn_icesal=2`, and `ln_pnd=F`.  These values are copied
from `/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg:24-26,
46,52-62,70,75-76,87,108,151`; the frozen lane copy is
`phase2w_orca1ice_namelist_ice_cfg`.  Everything not enumerated by the user
continues to inherit the ORCA2 variant/reference deck.  In particular,
`namelist_ice_ref:262-263` resolves the existing ORCA2 analytic ice start
(`ln_iceini=T`, `nn_iceini_file=0`), so no category-dimensioned ice input or
restart is consumed.  Whether that source path accepts `jpl=1` is
**PLAUSIBLE_PENDING_RUN**, not silently assumed; a NEMO diagnostic will stop
admission and be reported verbatim.

**CONFIRMED WRITE-only instrumentation:** Phase 2x repository hygiene replaced
the three once-committed full sources with unified patches against hash-pinned
NEMO 5.0.2 bases.  The exact applied files are retained under
`/data/abyssal/dbalwada/nemo-testcases-l4/build/phase2x_orca1ice/MY_SRC/`.
In those files, `icethd.F90:118,235,331-380` writes
thermodynamic entry/exit state, layered enthalpy/salinity, and atmosphere-ice
flux operands.  `icedyn_rhg_evp.F90:377,1197-1270` writes dynamics-entry,
strength, stresses, thickness, and Lemieux-2016 basal-stress operands before
the EVP iterations.  `icedyn_adv_pra.F90:112,491,504-578` adds a separate
self-describing Prather entry/exit stream while retaining the legacy stream.
Every model field is passed `INTENT(in)` or read directly; only local extent,
canonical-copy, header, and I/O objects are assigned.  Canonical helpers zero
halos, land, and inactive components before each write.  This is a
**CONFIRMED WRITE-only source diff**, not a physics change.

The headers are independently decodable: thermodynamics has 16 fields and a
4-by-16 extent table, dynamics 28 fields plus five scalar parameters and a
3-by-28 table, and Prather 35 fields and a 4-by-35 table.  Each carries
magic/version/kt/frame or Kmm, `jpi/jpj/jpl/nlay_i/nlay_s`, binary64 width,
field/scalar counts, and a SIZE-derived payload.  The allocation contracts are
validation only; exact EOF comes from the header.  The synthetic schema gate
accepts all three formats and its nine magic/truncation/trailing-byte plants
bind.  Real-record schema, canonicality, twin identity, ordinary-output
identity, and physics-selector admission remain **UNMEASURED_PENDING_MPI**.

The unchanged ocean cadence resolves `nn_fsbc=2` (`namelist_cfg:81`), and
`sbcmod.F90:477,604` calls SI3 only when that cadence is due.  Therefore the
first three executed ice calls are ocean `kt=1,3,5`; no SI3 operator exists at
ocean `kt=2`.  The new writers deliberately capture 1/3/5 rather than invent
a non-executed frame.

**CONFIRMED build/staging:** after correcting a writer-only rank declaration
for the rank-2 `qprec_ice`, the final scalar-math build succeeded.  Binary
SHA-256 is `2f3d162a0ce88bc429b5b511292ad1c0607c03eca402d621dcdb4739552d57da`;
`nm -D` reports zero `_ZGV*` symbols.  Build log SHA-256 is
`3631fdef41388f47c02a1087a9bcb3a21f8d9b74c3c49c860e89fc55d42f6d7a`.
Both fresh run directories contain copied deck files, symlinks to the
hash-pinned ORCA2 input tree, an absolute symlink to that binary, and a
self-contained two-rank CPU launcher.  Their common deck-manifest SHA-256 is
`ca6e31bb49ba1c1e939ad8c64c0bf672e0d46150673967666bbcda892d35274f`;
their common input-manifest SHA-256 is
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
Launcher A/B SHA-256 values are respectively
`ccc63b3bc241b01c7c0202fdef9fcbd1c59f5232c2913581c987ae9d5a2203dd`
and `3b0ecefd1e2e4894db938b565ef474d44bb83ad91178d3563354e169967a066c`.

Run these one at a time from the user shell, unchanged:

1. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2w_a_10step_np2/run.sh`
2. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2w_b_10step_np2/run.sh`

Do not pin either root until the real schema gate, 1/3/5 stream inventory,
raw twin identity, and ordinary-output identity pass.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| execute Phase-2v MPI twins | ASKED, user shell | CONFIRMED complete; launchers unchanged |
| admit and pin twin A | ASKED | CONFIRMED 101/101 twin and 100/100 inherited raw identity |
| score `taum` before TKE arithmetic | ASKED | CONFIRMED ORACLE_SUPPLIED hand-off, 0/8,794 |
| walk TKE to first non-bit statement | ASKED | CONFIRMED `zpelc`, GYRE owner; stopped fail-closed |
| alter shared `zpelc` arithmetic | Lane-4-forbidden | not done; reproducer routed to GYRE |
| keep NEMO selector numbering | ASKED, User Decision 12 | CONFIRMED 0/1/2/3 dispatcher; mode 2 is raw fraction |
| meaning of mode 1 | ASKED semantic disclosure | old legoESM: raw `fr_i`; corrected: `tanh(10*fr_i)`; old behaviour selectable as mode 2 |
| change a default or another card selector | UNASKED | not done; default remains 0, ORCA1 remains 3 |
| construct ORCA1-target ice deck | ASKED, User Decision 11 | CONFIRMED exact enumerated selector delta; ocean deck unchanged |
| preserve raw old mode-1 behavior | ASKED, User Decision 12 | CONFIRMED preserved as NEMO-numbered mode 2 |
| create synthetic SI3 kt=2 state | UNASKED / non-executed | not done; `nn_fsbc=2` produces executed frames at 1/3/5 |
| execute staged ORCA1-ice twins | ASKED, user shell only | UNMEASURED_PENDING_MPI; two launchers handed off |
| change ORCA2 analytic ice initialization | decision-gated | not done; stop on an exact NEMO refusal if one occurs |
| retain twin B and prior roots | ASKED | CONFIRMED retained and labelled |
| pin Phase-2u TKE records | forbidden | rejected records remain flagged, never scored |
| change shared arithmetic during admission | UNASKED | none |
| delete artifacts, edit shipped NEMO, commit large records, or push | forbidden | none |
