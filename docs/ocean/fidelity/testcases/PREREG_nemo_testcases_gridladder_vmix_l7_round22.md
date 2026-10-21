# GRIDLADDER-VMIX lane — round 22 (vmix_r22): L7 river-mouth mixing

Status: **PREREGISTERED; ACQUISITION NOT YET RUN**.

Decision 141 assigns ORCA2 river-mouth mixing to this lane.  Decision 146 moved
interior damping to lane A, so the rung is L7 (there is no lane-B L6).  L7 is
the certified V5 vertical-mixing chain plus the one river-mouth diffusivity
statement; it is not built on bare G5.  No NEMO build, run, trajectory score,
or certification is permitted in this round before the operator executes and
admits the acquisition.

## Re-verified survey and frozen resolved selection

ORCA2 selects `ln_rnf_mouth=T`, `rn_hrnf=15 m`,
`rn_avt_rnf=1e-3 m2/s`, and the static `socoefr0` field in
`runoff_core_monthly.nc` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:159-171`.  `sbc_rnf_init` derives `nkrnf`
from `gdepw_1d`, then calls the mask reader at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:504-526`.  The reader opens the
configured file, reads the two-dimensional field once, fixes the five-value
vertical profile, and applies the T-grid boundary exchange at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:569-588`.

The admitted ladder input is already present; no fetch or vendoring is needed.
Its `socoefr0` variable is two-dimensional, so NEMO reports it as record 1 but
there is no time record to select.  The committed acquisition preflight will
pin the file hash, layout, record report, and value census.  The current input
has no nonzero `socoefr0` values.  ORCA2 also sets `ln_closea=F` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:32-37`, so the closed-sea mask augmentation
is not called.  This corrects round 21's unsupported wording that the file
contains an acting mouth mask; it contains the selected field, but that field
is zero on this card.

NEMO calls `zdf_phy` once at step entry, on the BEFORE state, at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:170-180`.  Inside `zdf_phy`, it copies
the TKE closure into `avt`/`avm`, adds
`2 * rn_avt_rnf * rnfmsk * wmask` to heat diffusivity only, then runs EVD,
double diffusion, and internal-wave mixing at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:332-372`.  L7 therefore composes on
certified V5 and inserts between the closure copy and EVD.  legoESM already
transcribes runoff mass/tracer forcing and V5's complete coefficient chain; a
repository search found no river-mouth diffusivity implementation to reuse, so
L7 extends the shared coefficient composition and does not duplicate runoff.

## Frozen predictions and falsifiers

P1. Only L7 selects a new `RiverMouthMixingConfig` with the three resolved deck
values above and the admitted static mask.  Every other card remains flag-off;
its normalized one-step program is identical, and every certified card's
two-step value hash is unchanged.  Any other selector, changed flag-off
program, or changed certified value refutes P1.

P2. On a traced nonzero synthetic mask, the new term changes heat diffusivity
only at NEMO W levels `jk=2..nkrnf`, by the source-associated expression
`avt + ((2 * rn_avt_rnf) * rnfmsk) * wmask`.  Salt diffusivity changes only
through NEMO's later DDM/IWM statements; momentum viscosity never receives the
river term.  A nonzero direct salt/momentum increment, a different association,
or a depth selected without the `gdepw_1d` loop refutes P2.

P3. A jitted x64 one-step V5-versus-L7 control with traced state and a planted
nonzero mask pins the source position exactly: closure, river term, EVD, DDM,
IWM.  Moving the river addition after EVD or deleting its wiring must fail the
test.  The actual admitted zero mask must keep V5 and L7 values bit-identical;
claiming that identity proves acting would refute P3.

P4. The NEMO observer writes only the defined interior cells of the four
canonical coefficient boundaries: after closure, after river addition, after
EVD/DDM, and after IWM.  Writes remain outside any XIOS-dependent false branch.
Three independent twins must be byte-identical.  Undefined-cell output, twin
disagreement, or a moved observer refuses admission.

P5. `check_acting` is a separate statement evaluated on the smoke record only.
It requires a nonzero after-minus-before river increment on a defined wet W
cell.  Given the admitted all-zero mask and `ln_closea=F`, the frozen prediction
is that this check refuses.  It must not be weakened into a switch/log check or
applied to the writer-free boundary/twins.  A passing acting claim with a zero
delta refutes P5.

P6. Acquisition is smoke first, then a writer-free boundary run to NEMO's own
stop or 100 steps, then three twins to a common even endpoint no greater than
10 steps.  The tool defaults to read-only preflight; `--run` requires explicit
operator acknowledgement, and `--admit-existing` revalidates an existing
record.  Preflight executes all seven required planted-refusal checks,
including whitespace-tolerant numeric/logical checks against a real committed
`ocean.output` excerpt and a stamped rendered deck.  Any missing refusal or
unchecked acquisition artifact refutes P6.

## Round boundary

This round builds and tests the exact dormant ORCA2 selection, then stops at
the guarded acquisition boundary.  No score is allowed.  Because P5 is
expected to refuse, L7 cannot be called acting or certified without a later
user decision changing the ORCA2 input/program; this round makes no such
choice.

