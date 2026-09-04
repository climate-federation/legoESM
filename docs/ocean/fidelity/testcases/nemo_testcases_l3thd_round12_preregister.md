# SI3 lane 3b round 12 preregistration — rung 3.6 execution

Date: 2026-09-05  
Tracker: `climate-federation/legoESM#1699`  
State at commit: **PREREGISTERED / UNMEASURED**

This execution preregistration inherits the reviewed design, source walk,
sign table, complete frame registry, coverage seed, H1--H5 private arms, 21
planted violations, and stopping rule in
`nemo_testcases_l3thd_rung36_preregister.md`.  It records the newly available
inputs and fixes the remaining construction choices before generating a
weight, building NEMO, or scoring a row.  No result in this document is a
fidelity claim.

## Immutable inputs now available

The user supplied the ORCA1 input deployment fetched on 2026-09-05.  The
deployment script names Zenodo record 14041098 directly at `deploy.sh:21-34`
and downloads the CORE-II normal-year fields at `deploy.sh:11-17`.  Local
SHA-256 verification before this commit gave:

| input/provenance object | SHA-256 | planned use |
|---|---|---|
| `INPUTS/orca1_inputs.zip` (`data_repository.zip`) | `8ba4023bcc168f34d91d4d040e20c506b5428cf3dbb0aaa5a13f4cb04cbf4241` | immutable Zenodo archive provenance |
| `.../input_fields/merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc` | `f43c5a1e8ce75e52bfc8edfa4318e68215c76c4b252cb0fb70fa00b39a40d6fe` | RGB chlorophyll source |
| `.../input_fields/weights_reg05_bilinear.nc` | `1c2f9db057676c66664d93e9b170d2abb2acbefc8ee2130cc76097b610c29562` | provenance/reference only; **not usable** on the column grid |
| `INPUTS/fetch_inputs.sh` | `5b5ecad0fa0f5f409f3b1973fbb22f1bceb11c3029e5d69bc4f96fb629db1234` | fetch provenance |
| `INPUTS/fetch_inputs.log` | `6bf81e3b9f78e382293485c17eb52ed50a84c98a52cea3c120d9acad02fda208` | fetch record |
| `ORCA1-omip/deploy.sh` | `0f881f61e9c1a14ec75f52a0503dfdb313c94e7ed3d9d372dc91a8130677d65d` | URL/deployment provenance |

The run receipt will enumerate and hash every file actually opened by the
oracle.  Files not read by the oracle will not be represented as certified
inputs.  None of the supplied inputs will be modified; any derived file will
live under the rung's new `/data/abyssal/dbalwada/nemo-testcases-l3/` root.

## Active ORCA1 choices and the column chlorophyll construction

The target deck selects `ln_traqsr=.TRUE.` at ORCA1 `namelist_cfg:118`, then
`ln_qsr_rgb=.true.`, `nn_chldta=1`, and
`sn_chl='merged_ESACCI_BIOMER4V1R1_CHL_REG05'` at `:160-170`.  It also selects
NCAR surface bulk fluxes and constant ice-air coefficients at `:129-142`.
These lines are overlays on the copied C1D EXP_PAPA deck, one deviation per
receipt row; they are not physical constants.

The supplied `weights_reg05_bilinear.nc` maps REG05 to eORCA1 and is therefore
rejected for the one-column geometry.  NEMO's shipped `tools/WEIGHTS` is the
only generator permitted here.  Its README states that the three-program
sequence `scripgrid` -> `scrip` -> `scripshape` constructs an IOF weight file
and that the final shaped file is what `fld_read` consumes (README lines
64-79, 193-215, 241-278).  The committed generator namelist will name:

- the immutable REG05 chlorophyll file as `input_file`, with its actual
  longitude/latitude coordinate names discovered from the file;
- the oracle's generated column `domain_cfg.nc` as `nemo_file`, with its
  actual T-grid longitude/latitude names;
- `map_method='bilinear'`, `normalize_opt='frac'`, one data-to-NEMO map, and
  the source cyclicity encoded by `scripshape`;
- new intermediate and final filenames inside the rung's derived-input
  directory only.

The generator binaries, source revision, namelist, target `domain_cfg.nc`,
intermediates, and final weight file will all be hash-pinned.  This is the
single user-sanctioned generated input.

**Prediction P-WEIGHT (UNMEASURED):** the first NEMO `fld_read` chlorophyll
value dumped at `POST_TRA_QSR` will equal the value obtained by replaying the
four shaped weights and source cells in the order stored by the NEMO-generated
file.  CONFIRM means bit identity, with the ordinary pointwise normalized
`1e-15` row also reported.  REFUTE means any differing bit, a missing source
cell, a non-bilinear method, or use of the eORCA1 weight file.  A refutation
stops construction before the coupled gate.

## Oracle and implementation identity

The oracle is the reviewed one-layer C1D OCE+ICE design: new copied config
`C1D_OMIP_L3_COUPLED_SM`, 8,760 one-hour ocean steps, 2,190 SI3 calls at
`nn_fsbc=4`, scalar-math arch from the first build, CPU/no `mpirun`, and zero
dynamic `_ZGV*` references.  The shipped NEMO tree remains read-only; all
namelists, MY_SRC instrumentation, build products, derived inputs, and runtime
outputs are new copies.  Every design-register frame is WRITE-only.

The pre-implementation search re-found a single legoESM exchange owner:
`packages/coupler/legoesm/coupler/ocean_forcing.py` already defines
`ice_ocean_forcing_from_ice_response`, `blend_ice_ocean_forcing`, and
`omip_sea_ice_surface_forcing`; it owns the ice freshwater/heat/stress sign
conversion and the one-time open-water/ice partition.  Existing ocean paths
already own real freshwater/SSH, RK3-WS, RGB shortwave, and shared implicit
bottom drag.  Rung 3.6 will extend these owners with a selectable `nemo_si3`
exchange card and top-drag operand; it will not create a second coupler or a
private solver.  The card will explicitly select
`PrecisionPolicy.fp64(transcendentals='libm')`, print the CPU backend and every
participating dtype, and preserve the reviewed sign map literally.

The named unmerged branch `fix/omip-gm-treguier-one-variable` is absent from
the current local refs and from `git ls-remote --heads origin` as of this
preregistration.  The visible `fix/pierre-review-triage` ref does not contain a
`lead_freeze_source` symbol.  This absence is not interpreted as a physics
choice.  The SI3 `icesbc` `zqld`/lead-heat path remains a merge-overlap watch:
if execution reaches it, this lane will transcribe the active NEMO identity in
the existing exchange owner and record the overlap, never add an independently
derived second implementation.

## Predictions, exact-entry walk, and stopping decision

The exact-entry walk follows the NEMO execution order already registered:

`INITIAL_STATE -> PRE/POST_SSM -> POST_FZP -> PRE/POST_UPDATE_FLX ->`
`POST_UPDATE_TAU -> POST_FWB -> POST_SBC_STAGGER ->`
`POST_ZDF_DRG_COEFF -> POST_TRA_SBC_RK3/POST_TRA_QSR ->`
`PRE_DYN_SPG_TS/SSH_SUBSTEP/POST_STP2D -> PRE_DYN_ZDF_SOLVE`.

For each boundary the gate reports bit counts and
`abs(legoESM-NEMO)/max(abs(NEMO),1)`, with an immutable `1e-15` bar.  Before
the first measurement, the predictions are:

- P0: generated/ingested geometry and initial state are bit-identical after
  applying the registered NEMO precision conversions;
- P1: `sbc_ssm`, `eos_fzp`, field/sign mapping, staggering, FWB, and each ocean
  boundary are at bar when driven by the oracle's exact entry frames;
- P2: any first over-bar row is localized to exactly one earliest source
  boundary by the ordered walk; no downstream agreement may hide it;
- P3: H1--H5 remain the only preregistered one-variable attribution arms.
  Scaling at 0.5/1/2 precedes owner assignment, and private arms never become
  public selectors.

CONFIRM for P0/P1 is every non-waived row at or below the pointwise bar with a
complete one-to-one coverage register and all plants red.  REFUTE is the first
over-bar or uncovered row.  If the first over-bar boundary requires a new
scientific choice rather than a source-identity transcription, this dispatch
stops there and requests the user's decision; it will not tune through it.

The 21 reviewed planted controls remain binding and each must perturb a
registered nonzero row (or its explicitly registered finite sentinel) and
exit nonzero.  A row-level plant that does not change the scored row is itself
a gate failure.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| implement the independently reviewed rung-3.6 design | ASKED | this dispatch executes it |
| use the newly fetched immutable ORCA1 inputs and hash every file read | ASKED | fixed above; receipt will give the actual-input manifest |
| generate REG05-to-column weights with NEMO `tools/WEIGHTS` and bilinear method | ASKED | sole sanctioned derived input, preregistered before generation |
| dump and check `sn_chl` as NEMO reads it | ASKED | P-WEIGHT and `POST_TRA_QSR` registry row |
| build scalar-math from the start, CPU only, zero `_ZGV*` | ASKED | hard provenance/gate condition |
| extend the existing coupler and preserve the reviewed sign map | ASKED | one implementation only |
| use libm transcendentals explicitly on the card | ASKED | hard card/provenance condition |
| pointwise bar, bit columns, full coverage, row-binding plants, ordered first-divergence | ASKED | acceptance/stopping rule |
| implement Pierre's unmerged lead-budget change independently | UNASKED | forbidden; overlap is recorded for merge |
| certify general multi-column interpolation/staggering geometry | UNASKED | one-column geometry only; off-column geometry remains WAIVED |
| change ORCA1 selectors, run non-ORCA1 SI3 arms, or relax the bar | UNASKED | outside scope |
| modify/delete shipped NEMO, delete old artifacts, push, or use GPU/MPI | UNASKED | forbidden |

## Flagged for future deletion

Nothing is deleted.  Superseded experiments or generated intermediates will
be retained under `/data` and listed in the receipt if any are created.
