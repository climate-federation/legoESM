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

Round-8 choice ledger at preregistration: **ASKED** -- scan the 720 oracle
entries, bind the redistribution arm in the first source-significant regime,
score the aligned completed step including moments, and classify the four
zero oracle channels UNINFORMATIVE.  **UNASKED** -- no new physical threshold,
forcing, scheme, state substitute, tolerance, default change, or shipped-NEMO
edit; none is introduced here.
