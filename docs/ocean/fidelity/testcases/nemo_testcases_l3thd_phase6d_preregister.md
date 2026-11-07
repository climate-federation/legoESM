# Lane 3b SI3 Phase 6d snow-carrier association preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED after the corrected exact-entry pass and before
changing the snow-segment initialization associations.**

## Finding and hypothesis

Bridging NEMO's exact 1-D `h_s` is necessary but did not remove the new
zero-oracle rows.  Operand replay locates the remaining precursor in the snow
carrier construction: NEMO forms each old segment as `h_s*r1_nlay_s` and new
snow as `zsnw*sprecip*rDt_ice*r1_rhos/at_i` before its thickness-space
sublimation loop (`icethd_dh.F90:97-102,139-145,166-202`).  The candidate used
`h_s/3` and grouped `at_i*rho_snow` in a denominator.  These are algebraically
equal but not binary64-order equal at the tiny-snow transitions.

Hypothesis **H4-SNOW-CARRIER-ORDER**: extend the existing private
`_nemo_snow_sublimation_order` arm to cover the entire source-aligned carrier
construction at `:139-202`.  Confirm only if:

1. the source-order scalar replay continues to match the registered kt4242 and
   kt5734 POST_SUBLIMATION operands within two ulp;
2. kt5734 combined-arm POST_DH `e_s` improves by at least 100-fold and kt4242
   `h_i` remains bit-identical;
3. the kt5551 and kt5997 zero-oracle artifacts created by the incomplete arm
   disappear without filtering;
4. remap-only remains output-inert, while disabling the one snow-carrier hook
   plus the remap reproduces the retained Phase-5 behavior; and
5. a private plant substituting the legacy carrier result exits nonzero.

The production SI3 identity always executes the NEMO-written carrier order;
there is no NEMO switch and no public legoESM selector.  If the artifacts remain
or a new material row appears earlier, retain DEBT and name that counterexample.

## Choice register

- ASKED — resolve the kt5734 DH owner with operand-level first divergence,
  scaling, one-variable private arms, and an in-identity repair.
- ASKED — rerun all exact-entry and continuous annual evidence with controls.
- UNASKED — public carrier/remap choices, altered forcing/timestep/bar,
  mid-trajectory oracle injection, GPU/MPI, or mutation/deletion of shipped
  NEMO or earlier run roots.
