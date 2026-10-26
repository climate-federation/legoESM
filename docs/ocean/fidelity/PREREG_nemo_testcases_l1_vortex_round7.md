# PREREGISTRATION — round 191 / VORTEX round 7: switch the certified RK3 cards to NEMO's carried after-SSH slot

Frozen before changing a card, regenerating a restart, or running a trajectory.
Lane tip at the start: `c075e6fd5`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round191`.

Decision 78 authorizes exactly this switch: `GYRE-zco` and
`VORTEX_VEC-zco` state
`nemo_first_wzv_after_ssh="rk3_extrapolated_carried"`.  `ORCA2-zps`
remains `rk3_extrapolated` until its own ladder is re-measured.  No operator,
physics statement, numerical expression, library default, stabilizer, or NEMO
source changes in this round.

## Compiled NEMO statement

The carried slot was implemented and measured in round 6.  This round changes
only which certified cards select it.  NEMO writes the next step's guess after
the final time-level rotation as
`ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)` at
`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225` and
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90:222-226`.  The next step reads
that slot at `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:149,153` and
`GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:152,156`.  NEMO writes the slot
to its restart as `ssha` at
`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:184` and reads it, with the
step-entry-height initialization when it is absent, at
`VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:362-370`.

## Frozen predictions and falsifiers

1. The resolved-card census contains exactly two carried cards:
   `GYRE-zco` and `VORTEX_VEC-zco`.  `ORCA2-zps` remains uncarried;
   VORTEX-flux, both tanks, the generic non-NEMO cards, and both DINO NEMO
   recipes do not change form.  Any other moved card refuses the landing.
2. GYRE's default certified ladder reproduces round 6's carried arm on all 50
   scored rows.  It keeps `kt=1` and `kt=2` at the bar, keeps first-over-bar at
   `kt=3`, and has representative rows `kt=3 u = 4.4348e-10` and
   `kt=10 u = 1.7900e-09`.  Any row differing from the recorded round-6
   carried arm, an earlier first-over-bar, or an at-bar row leaving the bar
   falsifies the switch.
3. GYRE's default 360-day run reproduces the round-6 carried arm:
   day-30/day-240/day-360 T3D RMS
   `2.343251020612126e-06 / 6.581707093530567e-05 /
   5.407735418221895e-05 K`, subject only to the measured `~2e-10 K`
   run-to-run floor.  Every scored year row is registered.
4. VORTEX-vector's default ladder reproduces round 6's carried arm on all 50
   rows: `kt=1` and `kt=2` remain bit-identical to the prior default, while
   `kt=10 u = 4.8655e-06`.  The registered mixed T/SSH movements from round 6
   remain movements, not reclassified improvements.  VORTEX-flux stays
   byte-identical to round 6.
5. LOCK_EXCHANGE remains AT-BAR through its certified ladder.  OVERFLOW keeps
   its existing first-over-bar at `kt=2` with its published debt.  DINO's
   month gate passes its pinned day-30 bar and its resolved cards remain on
   `leapfrog_continuity`.  A changed fingerprint on an unaffected card stops
   the landing.
6. The certified configuration digests move only where the explicit card
   value moves.  The new GYRE digest is pinned only after the resolved field
   diff proves that this one value is the only configuration change.
7. A fresh format-5 restart for each switched card contains
   `eta_rk3_after`, round-trips it bit-for-bit, and a resumed step equals the
   corresponding continuous step.  Every production resume path derives the
   loader's `carries_rk3_after_ssh` requirement from the resolved model
   configuration.  A format-4 archive is still refused by the named missing-
   slot message for either switched card.  Missing the flag, silently loading
   an old archive, or a non-bit resumed step falsifies the restart work.
8. The restart control is non-vacuous: removing the derived loader flag must
   make the old-format refusal test fail, and replacing the stored carried
   slot must move the resumed next step while leaving the continuous control
   unchanged.
9. Citation mapping passes for every compiled range above; a shifted citation
   plant exits nonzero.  The final read-only Codex review must not say
   `DO NOT SHIP`.

## Gate scope

Run the GYRE `kt=1..10` ladder and the 360-day member at days 30/240/360,
both VORTEX ladders through `kt=10`, both tank ladders, the recipe-derived
card census, switched-card restart round-trip/resume controls, the DINO month
gate in a private directory, the citation gate and its plant, focused tests,
and the repository's push-gate set.  Every command runs on CPU in fp64.  The
before arm is round 6's unswitched default; its carried measurement arm is the
one-variable expected after arm.  ORCA2 remains `UNMEASURED-WITH-SPEC`: its
native ladder must be run before Decision 78 can be extended to it.

