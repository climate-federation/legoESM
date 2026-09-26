# Pre-registration — DINO climate-level mixed-layer-depth audit

Registered before computing any MLD field or any legoESM-versus-NEMO MLD
statistic from the four climate arms named below.  At registration time only
source, configuration, file inventories, stamps, array shapes/dtypes and grid
geometry had been inspected.

## Oracle definition, read before measurement

The executed DINO build calls `zdf_mxl` before the TKE closure and
`zdf_mxl_turb` after the closure plus EVD/DDM/IWM composition
(`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/ZDF/zdfphy.F90:280`,
`:282-338`).  They are different diagnostics:

* `hmlp` is the density-criterion mixed-layer depth.  NEMO initializes at
  `nlb10`, integrates `max(rn2b,0)*e3w`, retains the last W-level below
  `g*rho_c/rho0`, and returns that level's live `gdepw`
  (`src/OCE/ZDF/zdfmxl.F90:90-104`).  `rho_c=0.01 kg m-3` is fixed in this
  module (`:34`), and DINO resolves `g=9.80665 m s-2`, `rho0=1026 kg m-3`, so
  the integral threshold is `9.558138401559454e-05 m s-2`.
* The nominal 10 m reference is resolved by
  `zrefdep=10-0.1*min(e3w_1d)`, with `nlb10` the first W-level deeper than
  that value (`src/OCE/DOM/domzgr.F90:367-371`).  On DINO,
  `min(e3w_1d)=10.067163870644436 m`, `zrefdep=8.993283612935556 m`,
  `nlb10=2` (one-based), `gdepw_1d(nlb10)=10.13875112538517 m`, and the
  referenced T level is `gdept_1d(nla10)=5.033581935322218 m`.
* `hmld` is the turbocline depth.  Scanning from the bottom, NEMO selects the
  shallowest W-level at which the composed tracer diffusivity `avt` is below
  `avt_c=5e-4 m2 s-1`, then returns live `gdepw`
  (`src/OCE/ZDF/zdfmxl.F90:123-152`; threshold at `:35`).  This is not
  reconstructible from T/S alone.  The supplied legoESM climate snapshots do
  not carry `avt`, so `hmld` is recorded as UNMEASURED rather than silently
  replaced by a density diagnostic.

The scored diagnostic is therefore NEMO `hmlp`, using the already-committed
production transcription `_nemo_mld_from_n2_integral` that the operator-level
probe `zdf_mxl_nmln_compare.py` previously certified.  Both sides use the same
DINO S-EOS, constants, reference/live ladders, wet mask and live free-surface
stretch.  The de Boyer Montegut interpolated potential-density diagnostic is
not used: despite sharing a 0.01 kg m-3 label, it is not the `zdfmxl.F90`
operator DINO executes.

## Inputs and comparisons

Scored days: `0, 30, 60, 90`.  Each legoESM arm is compared with the control
NEMO member in `RUN_VERDICT360_M0` at the matching restart step
`kt = 5760 + 32*day`, using NOW-level `T/S/ssh` on both sides.

The four legoESM inputs are fixed by the mission:

1. `/tmp/codex-basin-rect/results/dino_1455/tcarry_basin90_legacy.npz`
2. `/tmp/codex-basin-rect/results/dino_1455/tcarry_basin90_corrected.npz`
3. `/tmp/dino_euc_mechanism/twin90_prefix/twin90_prefix.npz`
4. `/tmp/dino_euc_mechanism/twin90_fix/twin90_fix.npz`

All provenance/stability/clock/ladder/start/dtype stamps are fatal checks, not
advisory print lines.  The probe records SHA-256 for itself, every NPZ and log,
the mesh, every NEMO restart tile read, NEMO source/config/binary inputs when
available, and both repository revisions/dirty states.

## Registered map and regional reductions

For every arm and day, save the full wet-cell map
`bias = hmlp_lego - hmlp_NEMO` in metres.  Land is NaN in the saved map and is
fatal if it enters a reduction.  Also save both source MLD maps and their
integer mixed-layer base indices.

For each region report area-weighted legoESM mean, NEMO mean, signed mean bias,
and RMS difference.  Weights are the common NEMO tracer-cell horizontal area
`e1t*e2t`, zeroed by the intersection of NEMO `tmask(:,:,1)` and the arm's
surface wet mask.  No plain cell mean decides a result.

The regions reuse committed campaign definitions:

* `channel`: the recorded re-entrant band, rows `14..48` inclusive;
* `basin`: the recorded south-of-band basin, rows `0..13` (the dry wall row is
  removed by the common wet mask);
* `equator`: the structural-zero T row `99` used by the regional audit.

The full map remains the spatial result; these three reductions do not claim
to exhaust or partition the domain.

## Day-90 MLD classification

The scale is fixed from DINO's first resolved W-depth
`H1 = 10.13875112538517 m`, not from the outcome.

For an arm, day-90 MLD is **MATCHED** only if all three registered regions have

* `abs(area-weighted mean bias) <= H1/2 = 5.069375562692585 m`, and
* `area-weighted RMS difference <= H1 = 10.13875112538517 m`.

It is **DIFF** if any registered region has

* `abs(area-weighted mean bias) >= H1 = 10.13875112538517 m`, or
* `area-weighted RMS difference >= 2*H1 = 20.27750225077034 m`.

Everything in the dead band is **UNRESOLVED**.  Equality binds as written.
The classification is a registered DINO grid-scale audit bar, not an ensemble
noise floor and not a proposal to alter the campaign's five-metric acceptance
gate.

## Registered TKE-floor question

Question: **does the TKE surface-floor fix move equatorial day-90 MLD toward
NEMO?**

One variable decides: the `equator` region's area-weighted day-90 MLD RMS
difference.  Define

`delta_RMS = RMS_fix - RMS_prefix`.

* `delta_RMS <= -1.0 m`: **TOWARD_NEMO**.
* `delta_RMS >= +1.0 m`: **AWAY_FROM_NEMO**.
* otherwise: **INERT_AT_1M**.

The registered physical direction is lower RMS.  No bias, transport, shear or
global gate metric may override this one-variable answer.  The 1 m dead band
is one tenth of DINO's first resolved W-depth and prevents fp32 snapshot noise
from being promoted into a directional climate claim.

## Self-controls required before any result prints

1. A synthetic column passes a prescribed `rn2b` profile through the same
   committed NEMO criterion function and must return the hand-computed W-depth
   and base index exactly.
2. A planted threshold violation changes one synthetic contribution from just
   below to just above `g*rho_c/rho0`; the expected MLD level must change.  The
   probe must also demonstrate that a deliberately false expected level makes
   this control fail.
3. Day-0 shared-state identity is checked on wet T/S before scoring, with the
   fp32 storage quantum stated.  A mismatched arm is refused.
4. A dry-cell MLD poison must move every registered regional reduction by
   exactly zero; the same poison on a wet cell must move its region.
5. All wet MLD values and all weights must be finite float64; empty regions,
   missing days/fields/stamps, unstable arms, legacy clocks, non-bridged starts,
   non-`both` ladders and dirty producer trees are fatal.

The tool prints only these registered classifications.  It contains no stale
or superseded MLD claim to retract.
