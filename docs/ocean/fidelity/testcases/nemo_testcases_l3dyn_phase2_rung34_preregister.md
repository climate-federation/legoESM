# SI3 lane 3 rung 3.4 preregistration: ICE_RHEO

Date: 2026-09-03. Tracker: climate-federation/legoESM#1699.

This document freezes the rung-3.4 oracle experiment before either copied case
is built. It is a measurement plan, not a trajectory result. Every result is
**UNMEASURED** until a committed gate reads the artifacts named below.

## Question and controlled arms

The shipped `tests/ICE_RHEO/MY_SRC/icedyn_rhg_evp.F90` is not the NEMO 5.0.2
`src/ICE` implementation. Does that stale test override alter this case after
the requested EAP-to-aEVP overlay?

Both arms are copies of the shipped case made in a writable source clone of
NEMO commit `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`:

| Arm | Copied-case path | Deliberate source difference |
|---|---|---|
| production oracle | `tests/ICE_RHEO_OMIP_L3` | `MY_SRC/icedyn_rhg_evp.F90` is excluded while copying, so `src/ICE/icedyn_rhg_evp.F90` is compiled |
| shipped control | `tests/ICE_RHEO_OMIP_L3_SHIPPED` | the shipped stale EVP override is retained |

The writable source root and run roots are under
`/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/`. The shipped tree is
read-only input: no shipped file or case is modified or deleted. The production
copy exclusion is **FLAGGED FOR FUTURE DELETION** from the upstream shipped
case because an obsolete test-local solver can shadow the released solver.
Its SHA-256 is `7efffd18b5a452d403e1a3e4e813f389a9c18a1a4f5a08e280b92272a2ed22ee`;
the released solver SHA-256 is
`f1c92a8e815f28e9178acdd8e9e307f598b8057b329e8c2e2abece627271b3fa`.
The stale EAP override is retained in both arms and is dead because
`ln_rhg_EAP=.false.`; it is not another controlled difference.

## Post-preregistration amendments (preserved history)

The paragraph above records the preregistered experiment and is intentionally
not rewritten after measurement.  The first build showed that the supposedly
selector-dead `MY_SRC/icedyn_rhg_eap.F90` is still compiled and itself fails
against the 5.0.2 module API.  The copied-case construction was therefore
amended as follows, without changing any shipped file:

* `ICE_RHEO_OMIP_L3` excludes **both** stale overrides and builds the shipped
  case against the released `src/ICE` implementations.
* `ICE_RHEO_OMIP_L3_SHIPPED` retains the stale EVP override as a build control;
  it is **UNBUILDABLE**, with 21 compiler errors recorded in the phase-1
  receipt.  Failed copies retaining the EAP override are also preserved.
* both overrides remain **FLAGGED FOR FUTURE DELETION** from the upstream test
  case; nothing in the shipped tree was modified or deleted.

On 2026-09-03 the user classified this incompatibility as an upstream NEMO
defect and explicitly selected the override-excluded copy as the only buildable
form of the shipped case and therefore the oracle for this rung.  This decision
supersedes the original stop condition below; it does not turn the unbuildable
control into a numerical comparison or manufacture an analytic substitute.

One more runtime-resolution detail was discovered only after executing the
clean copy.  With the shipped case's `ln_icethd=.false.`, SI3 overwrites
`rn_porordg` to zero and the snow/pond ridge and raft retention factors to one
at `icedyn_rdgrft.F90:1244-1247`.  The card reproduces those **executed**
values.  The ORCA1 deck's configured `0.5` values remain disclosed above; they
do not survive SI3's own thermodynamics-off resolution on this rung.

## Shipped-case dossier and resolved input

The case is a closed 2000 km by 2000 km box: `usrdef_nam.F90:73-84`; at the
case's `rn_dx=rn_dy=2000 m` (`namelist_cfg:19-20`) this resolves to a
1000 by 1000 physical domain plus NEMO halos. Coriolis is disabled by
`namelist_cfg:21` and `usrdef_hgr.F90:154-155`. The 30 s step and 720-step run
come from `namelist_cfg:29-37`. `usrdef_sbc.F90:112-117` supplies
`Rwind=-0.8`, `Umax=15 m s-1`, a 2000 km domain, 2 km cells,
`rho_air=1.22 kg m-3`, and `Cd_atm=1.4e-3`; lines 122-140 define the six-hour
wind spin-up and ice-relative stress.

The committed decks under
`scripts/validate/ocean_fidelity/testcases/configs/ice_rheo_l3_*` are exact
runtime inputs. The only cpp deviation is removal of `key_xios` from the
shipped `cpp_ICE_RHEO.fcm`; the resulting verified key set to build is
`key_si3 key_linssh key_vco_1d`, i.e. OCE+SAS+ICE, 1-D vertical coordinate.
This avoids an unavailable output-server dependency and changes no physics.

Namelist resolution is reference then configuration. Shipped values retained
are `jpl=1`, `ln_icedyn=T`, `ln_icethd=F`, `ln_dynALL=T`, Prather advection,
`ln_iceini=T`, and the case geometry/forcing. The overlay is:

| Value | Source |
|---|---|
| `nlay_i=10`, `nlay_s=5` | ORCA1 `namelist_ice_ref:25-26` |
| `rn_ishlat=2` | ORCA1 `namelist_ice_ref:57` |
| `ln_landfast_L16=F` | user pick; ORCA1 ref `:58`, explicit deviation from ORCA1 cfg `:46` |
| H79, `rn_pstar=2e4`, `rn_crhg=20`, no strength smoothing | ORCA1 cfg `:52-55` |
| exponential ridge distribution, `rn_murdg=3`, `rn_csrdg=.5` | ORCA1 cfg `:57-58`; ref `:87-88` |
| exponential participation, `rn_astar=.03` | ORCA1 cfg `:60`; ref `:92-93` |
| ridging on, `rn_hstar=25`, `rn_porordg=0`, snow/pond retention `.5/.5` | ORCA1 cfg `:61,63,65`; ref `:95,97` |
| rafting on, `rn_hraft=.75`, `rn_craft=5`, snow/pond retention `.5/.5` | ORCA1 cfg `:62,64`; ref `:100-102` |
| EVP on, EAP off, aEVP on, `rn_creepl=2e-9`, `rn_ecc=2`, `nn_nevp=100`, `rn_relast=.333`, convergence check off | ORCA1 cfg `:70`; ref `:108-116` |
| `rn_Cd_io=5e-3` | ORCA1 ref `:136` |

ORCA1 itself has `jpl=5`, thermodynamics on, and landfast on in its cfg. This
rung deliberately retains the shipped case's `jpl=1` and thermodynamics-off
scope, and the user-selected landfast-off scope. No other scheme arm is mixed.

## Write-only frames and comparison decision

Both copies receive the lane-3 write-only `icestp.F90` instrument. At every
ice-step entry, immediately before upstream `store_fields`, it writes one fp64
stream frame with header `(magic,version,kt,jpi,jpj,jpl,nlay_i,nlay_s,bits,19)`
and, in the order stated by instrument lines 164-167:

`v_i, v_s, a_i, t_su, oa_i, a_ip, v_ip, v_il, sv_i, u_ice, v_ice,
stress1_i, stress2_i, stress12_i, snwice_mass, snwice_mass_b, e_s, e_i,
szv_i`.

Time-level registry: these are the current SI3 fields at `ice_stp` entry;
`snwice_mass_b` alone explicitly names the before ledger. The final
`restart_ice.nc` is separately retained. No instrument value feeds NEMO.

The control comparison consumes every common entry frame (expected kt 1..720)
and both final ice restarts. CONFIRM means at least one finite cell differs and
the report names first step, field, index, absolute difference, relative
difference, and ULP difference. REFUTE means all compared bytes and restart
variables are equal. Missing steps, non-finite differences, or incomplete runs
are **UNMEASURED**, never REFUTE. A planted scored-row perturbation must turn
the comparison red before any verdict is accepted.

The case README documents only qualitative phenomenology: EVP has less-defined
shear-strain maxima, and longer EVP runs produce LKF intersections at a
different angle from EAP (`EXPREF/README:51-53`). This requested run changes
the shipped EAP arm to aEVP and ends at the default 720 steps, so the EAP angle
contrast is out of scope. We will report the EVP shear image/field and whether
linear maxima are visibly present, but label that result
**MEASURED-UNCLASSIFIED** because the README supplies no numeric threshold.

## legoESM pre-implementation search and proposed arm

`packages/ice/legoesm/ice/ridging.py` already contains a generic Lipscomb-2007
multi-category ridging closure (`participation_weights`,
`_ridging_column_kernel`, `apply_ridging`). It uses an e-folding-in-thickness
participation weight and a uniform overlap transfer. SI3's exact jpl=1 path in
`ice_dyn_rdgrft.F90:287-341,399-570` instead uses its `aksum` preparation,
`rn_astar` area measure, exponential ridge distribution, and its own donor /
receiver update order. Those algebras are not identical and must not be
silently shared. The ocean tracer SOM in
`packages/ocean/legoesm/ocean/advection_som.py` is likewise not ridging code.
Rafting is absent from legoESM; SI3 selects it at `ice_dyn_rdgrft.F90:475-489`
with the `rn_hraft/rn_craft` split and applies the shared redistribution ledger
later in that routine.

If implementation begins this round, a selectable `si3_orca1_jpl1` scheme arm
will be added inside the existing ice ridging module; no second model or second
ridging module will be created, and existing defaults will remain unchanged.
Geometry must equal the production oracle mesh, fp64 policy and dtypes must be
printed, kt=1 must be scored at the existing 1e-15-class bar, and the sweep must
name the first over-bar frame. Restart-carried Prather moments and EVP stresses
remain part of the restart gate.

## End-of-task choices

ASKED: harden the seven reviewed rung-3.3 harness items; create the two copied
ICE_RHEO oracle arms; measure the stale override; then add the in-module jpl=1
SI3 ridging/rafting arm and card through the cleanest honestly measured gate
boundary; commit, bundle, hash, and do not push.

UNASKED / not authorized: modify or delete the shipped NEMO tree; use MPI,
GPU, fp32, a synthetic oracle, landfast, `jpl>1`, thermodynamics, change an
existing card default, claim README phenomenology numerically without a
preregistered predicate, push, or claim a trajectory result before measuring.

## Round-8 active-ridging amendment (preregistered before measurement)

The kt=1 result does not certify finite-amplitude redistribution.  Round 8
therefore withdraws that interpretation and selects an active frame using
SI3's own operational scale rather than a post-hoc threshold.  For completed
step `k`, the `u_ice` and `v_ice` stored at entry frame `k+1` are the velocities
produced by the preceding rheology call; advection, redistribution, and
`ice_cor` do not subsequently change them (`icedyn.F90:130-135`).  The gate
will evaluate SI3's EVP closing equations exactly as written at
`icedyn_rdgrft.F90:243-252` and select the first `k` for which
`max(closing_net * rDt_ice) > epsi10`.  Here `epsi10=1e-10` is SI3's existing
source cutoff used by redistribution (`:594-595,623-624`), not a new fit.

For every step through the selected step, plus the following five steps, the
committed report will publish `max|delta_i|`, `max(opning)`,
`max(closing_net)`, and `max(closing_net*rDt_ice)`.  At the selected step it
will additionally publish the actual maximum changes made by the SI3 jpl=1
redistribution arm and the populations of its one-through-19-shift cells.
The oracle's four identically-zero fields (`oa_i`, `a_ip`, `v_ip`, `v_il`)
are preregistered **UNINFORMATIVE**: exact zero agreement cannot certify their
age or pond update channels and must never be printed as AT-BAR.

The active-frame completed-step comparison will be initialized from that
oracle entry state.  Its Prather moments must come from NEMO restarts on both
sides of the single-step window; the entry frames do not contain moments, so
zero-filled or legoESM-carried substitutes are forbidden.  The run protocol
and every selector remain byte-identical to the production oracle except for
the shorter end/restart-write clock needed to expose those two time levels.
The same 50 state rows (with the four zero rows UNINFORMATIVE) and every
available Prather moment are scored.  Missing aligned restarts makes the
active-window result UNMEASURED rather than permitting a substitute.

Prediction, frozen before the scan: the first source-significant step is
completed step **9**, and the first informative over-bar row in its completed
step is **`stress1_i`**, following the already measured stress-first boundary
at completed step 2.  A different step or owner refutes the corresponding
prediction; no owner is claimed until the gate measures it.

Round-8 choice ledger at preregistration:

| round | choice | disposition | basis |
|---|---|---|---|
| 7 | Copy and run the buildable override-excluded ICE_RHEO oracle; retain the stale-source control as UNBUILDABLE; add the in-module `jpl=1` SI3 ridge/raft arm and stop at first divergence | ASKED | Round-6/7 dispatch |
| 7 | Repair, modernize, delete, or numerically substitute for either stale shipped override | UNASKED | Explicit shipped-tree/no-substitute rules |
| 7 | Change existing ice defaults; enable landfast, thermodynamics, another rheology/ridging selector, GPU, MPI, or `jpl>1` | UNASKED | Explicit scope and CPU-only rules |
| 8 | Scan all available oracle entries, select the first source-significant closing regime, and score its aligned completed step including restart-carried moments | ASKED | Round-8 finding 1 |
| 8 | Treat the four identically-zero oracle age/pond state rows as UNINFORMATIVE | ASKED | Round-8 finding 1(c) |
| 8 | Change `ato_i` to a Prather tracer if and only if the executed NEMO source does so | ASKED | Round-8 finding 2; source inspection is controlling |
| 8 | Relabel the unperturbed rung-3.3 mechanism as PLAUSIBLE instead of running the optional one-ULP experiment | ASKED | Round-8 finding 3 expressly offered either choice |
| 8 | Publish field-relative errors beside, but do not replace, the immutable max-one normalized gate metric | ASKED | Round-8 finding 4 |
| 8 | Remove the two non-oracle final clamps, reject category-axis state, and evaluate `sishea` with NEMO's own formula | ASKED | Round-8 findings 5--7 |
| 8 | Invent a new threshold, forcing, scheme, moment family, state substitute, tolerance, default change, or shipped-NEMO edit | UNASKED | No such expansion was authorized |

## Round-9 moment-debt discrimination (preregistered before replay)

The headline active-window discrepancy is the 60/160 Prather-moment DEBT,
owned in magnitude by `sxxe_l01` at `3.492459543785742e-9` normalized and
`2.095149123314519e-8` field-relative.  This is not classified as roundoff.
The experiment starts from the same oracle entry frame 8 plus NEMO step-7
restart moments and scores the NEMO step-8 restart.  Geometry, fields, clock,
fp64 policy, and every selector remain unchanged.

Source census before running:

* a symbol search for the distinctive saved families (`sxice`, `sxxice`,
  `sxc0`, `sxxe`, `sxsi`, `sxvl`, and their y/cross partners) across all of
  `src/ICE/*.F90` finds exactly one file: `icedyn_adv_pra.F90`;
* that file changes moments only in `adv_x`/`adv_y` calls
  (`icedyn_adv_pra.F90:253-350`), halo exchange (`:432-479`), initialization or
  restart input (`:1226-1380`), and restart output (`:1383-1497`).  No
  `rdgrft`, `cor`, rheology, or thermodynamics routine writes them;
* legoESM passes moments only through `advect_si3_prather_2d`
  (`nemo_rheo_testcase_recipe.py:646-662`) and returns that tuple unchanged
  after source corrections, ridge/raft, and `ice_cor` (`:664-704`).  Therefore
  a legoESM-only rescale/re-derive/zero operation is **REFUTED by source
  census**; the one-variable comparison moves only the U/V input supplied to
  the existing transport implementation.

Preregistered arms and decisions:

| arm | only change | CONFIRM | REFUTE |
|---|---|---|---|
| baseline | recomputed legoESM aEVP U/V | reproduces the recorded 60 moment debts | recorded artifact cannot be reproduced |
| oracle-V | replace only V with oracle entry-frame-9 V | first y-sweep moment discrepancy collapses to at most 2 ULP | it remains above 2 ULP |
| oracle-U | replace only U with oracle entry-frame-9 U | x-sweep-only remainder is at most 2 ULP after y agrees | y-stage discrepancy changes materially |
| oracle-U/V | replace both velocity components | every endpoint moment closes to at most 2 ULP | any endpoint moment remains above 2 ULP |

Prediction: because completed step 8 is even, NEMO executes y then x
(`icedyn_adv_pra.F90:253,303-350`).  The two arms are identical before and
immediately after the y limiter, which has no velocity operand
(`icedyn_adv_pra.F90:757-791`).  The first baseline-versus-oracle-input moment
difference is predicted after that limiter in the y loss/merge program
(`:793-943`), with `sxxe_l01` remaining the largest endpoint row.  Whether the
first differing family is `sx`, `sy`, `sxx`, `syy`, or `sxy` is deliberately
left to the committed operand scan rather than inferred from the endpoint.
If the oracle-U/V arm does not close, the replay must descend through limiter,
loss, and receiver-merge operands and no velocity attribution is permitted.

The subsequent walk starts from the corrected/aligned step-8 state and runs
through step 720.  It records the first over-bar step independently for every
ordinary field and both normalized and field-relative errors at steps 9, 10,
50, 100, 200, 485, and 720.  Separately, the oracle closing replay evaluates
the excessive-category-removal predicate literally:
`apartf * closing_gross * rDt_ice > a_i` at
`icedyn_rdgrft.F90:600-607`.  Prediction: the first firing, if any, is after
step 13; no claim of absence is allowed until all 720 oracle entries are read.
The open-water correction at `:611-621` is reported separately and is not
silently called the category clamp.

| round | choice | disposition | basis |
|---|---|---|---|
| 9 | Bind the plant by its row transition and headline the dominant moment debt | ASKED | Review findings 1--2 |
| 9 | Census all NEMO moment writers; inspect legoESM between-advection handling; run U-only, V-only, and U/V one-variable arms | ASKED | Review finding 3 |
| 9 | Change transport before the source census and replay identify an operand | UNASKED | Violates preregistration and one-variable discipline |
| 9 | Walk ordinary fields through step 720 and scan the oracle excessive-removal predicate | ASKED | Review finding 4 |
| 9 | Promote zero age/pond receiver rows or infer ORCA1 coverage from them | UNASKED | This thermodynamics-off card cannot exercise those channels |

### Round-9 written-order operand addendum (before the transport arm)

The first four-arm measurement refuted a velocity-only owner: replacing both
U and V still leaves 60/160 moment rows in DEBT, led by `sxxe_l01` at
`2.328306381027545e-9` normalized.  Inspection of the first source operation
not reproduced literally identifies a narrower one-variable arm.  NEMO first
forms the Courant fraction and then reconstructs transported area as
`zalf*zpsm` in both directions (`icedyn_adv_pra.F90:582,805`); its negative
face pass likewise adds `zalf*psm(donor)` (`:628,851`).  legoESM instead uses
the algebraically equal but floating-point-distinct `abs(velocity)*dt`.
Because the receiver merge multiplies this area fraction by extensive
enthalpy, this reassociation is an operand-level candidate for the observed
small-moment cancellation.

The next arm changes only those four transported-area expressions to NEMO's
written ordering.  Prediction: `sxxe_l01` moves down by at least one order of
magnitude and the count of DEBT moment rows falls below 60.  Full closure to
two ULP is **not** predicted because other compiler associations may remain.
No limiter, flux-content, donor-loss, receiver-merge, velocity, state, or
selector expression changes in this arm.  A failure to move either registered
quantity refutes transported-area ordering as an owner.

## Round-10 C-grid aEVP source-rounding arm (preregistered before solver edits)

### Shared tool provenance and immutable contract

The one shared implementation is
`packages/core/legoesm/core/source_rounding.py::nemo_source_round`, imported
from SI3-thermodynamics commit `86a8eb21d189`.  Its direct identity/JIT/gradient
test is imported from GYRE commit `2a7b1f7ae258`.  The helper implementation is
not changed in this lane.  Its stale sentence claiming that
`optimization_barrier` survives to HLO will be corrected as documentation
only, as explicitly requested: the operative materialization guard is the
finite-classification `select` plus `copysign` chain.

The production C-grid arm will call this helper after each NEMO written source
statement, with no card selector or default change.  A private boolean-free
ablation helper may retain the old association only so the one-variable gate
can prove the measured move.  The A-grid EVP and mEVP arms are untouched.

### Written-operation registry for one aEVP subcycle

Each row names the NEMO assignment whose result is materialized before its next
consumer.  Parentheses and reciprocal forms remain those written by NEMO.

| stage | NEMO statements | guarded result and grid |
|---|---|---|
| fixed operands | `icedyn_rhg_evp.F90:231-247,270-317` | `ecc2`, `1/ecc2`, `1/zdtevp`; T mass/Coriolis/dt-over-mass; U/V area, mass, cross-ocean velocity, mass/dt, air/ocean drag and slope |
| F strain | `:395-397` | each U and V reciprocal-scaled difference, squared-metric product, their sum, area reciprocal product, then `fimask` product → `zds(F)` |
| T strain | `:404-421` | four weighted F-shear squares and their prescribed pair sums → `zds2`; paired U/V flux differences → `zdiv`, then `zdiv2`; paired reciprocal-metric differences → `zdt`, then `zdt2`; inner `(zdt2+zds2)`, eccentricity product, addition to `zdiv2`, square root and `zmsk` → `zdelta(T)` |
| delta floor/viscosity closure | `:423-424,427` | `zdelta+rn_creepl`, `strength/(...)`, `zmsk` product → `zp_delt(T)`, followed by periodic T halo materialization |
| wide T recomputation | `:432-440` | separately materialized duplicate `zdiv` and `zdt`, because NEMO recomputes rather than reuses them |
| adaptive alpha/beta | `:443-449,465-477` | ordered `0.5*zp_delt`, reciprocal area, `zdt_m`, square root, `pi` product, floor; `1/(alpha+1)`; separately repeated `zbeta`; four-point max → `alpha_f`; `1/(alpha_f+1)` |
| T stresses | `:458-461` | old-stress×alpha; divergence/delta branches and tension/eccentricity branch; `zp_delt` products; prescribed sums; reciprocal-alpha product; final `zmsk` on `zs1/zs2` only |
| F stress | `:483-489` | two pair sums and quarter product for `zp_delf`; shear/eccentricity product, half weight, old-stress branch, sum and reciprocal-alpha product → `zs12` (no `zmsk`) |
| stress divergence | `:495-510` | every squared-metric stress product, prescribed difference and pair sum, reciprocal metric product, cross-stress branch, outer half and reciprocal-area products → `zfU(U)`/`zfV(V)` |
| cross velocity | `:512-514` | the two parenthesized pair sums, quarter product and mask → `v_iceU(U)`/`u_iceV(V)` |
| ocean drag per component | `:534-544,585-595,640-650,692-702` | component differences, squares, pair sum, square root and drag product → `zTauO`; ocean-velocity difference and product → ocean stress; dead landfast bottom terms remain zero on this card |
| Coriolis | `:547-549,598-600,653-655,705-707` | each metric×velocity product, prescribed local pair, mass-Coriolis product, remote pair/product, outer sum, reciprocal-metric and quarter products; numerically zero here because the case sets Coriolis off, but still registered |
| velocity RHS | `:551-562,602-613,657-668,709-720` | source-ordered additions `force + air + Coriolis + slope + ocean`; beta×current plus before velocity; mass/dt product; RHS and implicit-drag additions; beta+1, mass/dt product, drag addition, denominator floor; final quotient |
| velocity masks | `:575-579,626-630,681-685,733-737` | active/low-mass branches and products, then fast-mask factor/product |
| parity and halo | `:530-634,636-741` | even subcycles write V halo-1 then U interior; odd write U halo-1 then V interior; each component result is materialized before the sequential partner and full one-rank periodic U/V halos are rebuilt after each pair, matching `lbc_lnk(...,ldfull=.true.)` |

### Prediction and discriminator

Primary prediction: under CPU production JIT and fp64, source-rounding the
registered statements makes `stress1_i`, `stress2_i`, `stress12_i`, `u_ice`,
and `v_ice` byte-exact after all 100 subcycles of completed step 1.  Because
the Round-9 oracle-U/V arm already makes every endpoint Prather moment
byte-exact, the full completed-step state and all 160 transported moments are
also predicted byte-exact.

CONFIRM means zero ULP for all five rheology outputs and every scored
completed-step field/moment.  REFUTE means any row is nonzero.  On REFUTE, the
gate records every subcycle and descends in the registry above to the first
nonzero operand; no later-stage explanation is accepted as ownership.  The
private old-association arm must reproduce a nonzero scored row, and a planted
post-solver perturbation must move a named clean scored row from bit-exact to
nonzero independently of the overall gate exit.

Only if completed step 1 is byte-exact does the experiment proceed through
steps 2--8 and the active redistribution window.  Only if that endpoint is
also byte-exact does the production-JIT trajectory rerun steps 9--720.  That
walk registers, for every field, the first step at which its numerator becomes
nonzero and prints `0 / n` for bit-exact rows rather than dividing or replacing
the denominator.  Growth checkpoints remain 9, 10, 50, 100, 200, 485 and 720.
The measured first non-bit-exact statement, if any, decides whether the old
step-720 divergence was amplification of residual arithmetic or a remaining
defect; no amplification label is assigned in advance.

### Round-10 ASKED / UNASKED choices

| choice | disposition | reason |
|---|---|---|
| Import the single shared rounding helper/test at their exact commits and record provenance | ASKED | Cross-lane one-implementation rule |
| Guard every executed written aEVP operation without a card switch; retain old association privately for ablation | ASKED | One-variable discriminator |
| Enumerate Rule-8/12 rows, bind a row-level plant, and use `0 / n` for exact rows | ASKED | Queued review items |
| Continue steps 2--720 before step 1 is byte-exact | UNASKED | Explicit dispatch stop condition |
| Change the helper implementation, A-grid defaults, schemes, bar, precision, backend, shipped NEMO, or commit artifacts | UNASKED | Outside the authorized lane |

### Round-10 secondary discriminator: H79 exponential (registered before production edit)

The WRITE-only step-8 ICE_RHEO probe made `zds` and `zdelta` byte-exact but
first separated at `zp_delt` (`icedyn_rhg_evp.F90:423-424`).  A post-hoc
one-variable host replay of H79 strength (`icedyn_rdgrft.F90:1048-1055`) found
that vector `np.exp` left 94,986 physical `zp_delt` cells nonzero with maximum
`0.00390625`, while scalar glibc `libm.exp` made every `zp_delt` cell
byte-exact.  This is a measured discriminator, not a preregistered result.

Before the production edit, the next prediction is registered: import the
existing shared scalar-libm precision policy and transcendental wrapper from
GYRE commit `61180a6776c4`, select it only in the two NEMO ice-testcase cards,
and route the one shared H79 `ice_strength` exponential through that policy.
With source-rounded subtraction, exponent product, prefactor products and the
existing `zp_delt` closure, the ICE_RHEO step-8 first two subcycles and all 100
subcycles will be byte-exact.  CONFIRM is zero ULP for `zp_delt`, `zbeta`, all
three stresses and both velocities; REFUTE is any nonzero row, followed by the
same first-operand descent.  Native transcendental policy remains the default
for every existing A-grid/user card, so this is not a global default change.

### Round-10 outcome (recorded after the arms)

The exact-input aEVP prediction is **CONFIRMED**: CPU production JIT, the
independent written-order replay, and the NEMO endpoint are byte-exact for U,
V, and all three stress carries after 100 subcycles (`0 / 9801` each).  The
private unrounded arm makes all five rows nonzero.  The final one-ULP operand
was the F-stress update: NEMO computes `1/(alpha_f+1)` and then multiplies at
`icedyn_rhg_evp.F90:476-489`; replacing that with a direct numerator division
was not source-equivalent.

The broader completed-step prediction is **REFUTED**.  Before the solver, the
testcase card's extensive `field*area/area` bridge changes `v_i` in
`164 / 10609` cells and `v_s` in `172 / 10609`, by one ULP each.  The complete
step retains three stress DEBT rows (`1.1989e-14`--`2.5963e-14` normalized).
Accordingly, the preregistered stop fired: steps 2--8 and a new 9--720 walk
were not acceptance-scored.

The H79 scalar-libm operand prediction is **CONFIRMED at `P/delta`** on the
WRITE-only ICE_RHEO step-8 subcycle dump: vector NumPy exponential is nonzero
in `94,986 / 1,008,016` cells with maximum `0.00390625`, while scalar glibc
libm is `0 / 1,008,016`.  Its broader active-window trajectory prediction is
UNMEASURED because the primary completed-step stop condition was false.  The
full measurements, controls, hashes, and ASKED/UNASKED dispositions are in the
phase-2 receipt.

### Round-11 intensive-state bridge (registered before implementation)

Executed NEMO keeps `a_i`, `v_i`, `v_s`, `v_ip`, and `v_il` as intensive
category fields across outer steps.  The one-category aggregates supplied to
rheology are direct `SUM` assignments at `icevar.F90:123-132`; dynALL then
calls rheology before advection at `icedyn.F90:130-135`.  Only inside Prather
does NEMO form the extensive work arrays as `field * e1e2t` at
`icedyn_adv_pra.F90:218-245`, and it recovers the intensive fields in the
written order `z0 * r1_e1e2t * tmask` at `:355-381`.

The preregistered repair is therefore one shared transport-boundary pair:
outer-step/card state remains intensive; immediately before the existing
Prather kernel it is packed with the literal NEMO multiplication statements;
immediately after advection and its executed source corrections it is
unpacked with NEMO's reciprocal and left-to-right products.  Each written
floating operation is passed through the canonical `nemo_source_round`.
Rheology consumes the intensive carry directly, so the initial state never
takes a fictitious `field * area / area` trip.  The existing rung-3.1/3.2
transport API and all non-testcase ice defaults remain unchanged.

Prediction: on CPU, fp64, production JIT and scalar-libm policy, the completed
rung-3.3 step 1 will make `u_ice`, `v_ice`, `stress1_i`, `stress2_i`, and
`stress12_i` byte-exact (`0 / 9801` nonzero cells each).  CONFIRM is all five
rows zero and no newly nonzero transported-field row; REFUTE is any nonzero
stress/velocity row, followed by a first-operand descent and an explicit next
owner.  Only if that endpoint is bit-exact will steps 2--8, the active
ICE_RHEO step-8 window, and the production-JIT 9--720/restart walk run.

Two review discriminators are also registered.  First, repeating the Round-9
oracle-U/V moment replay under JIT will leave all 160 moment rows byte-exact;
any nonzero row refutes the prior upstream-owner statement.  Second, the
moment plant must produce `over_two_ulp_count > 0` and a nonzero process exit;
zero is a fail-closed harness error.  If the conditional long walk runs, the
first non-bit-exact row and step are the next owner boundary; no later growth
is assigned a mechanism without a one-variable measurement.
