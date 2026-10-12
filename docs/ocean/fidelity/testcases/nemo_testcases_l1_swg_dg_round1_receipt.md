# SWG double-gyre round 1 receipt

Date: 2026-10-12

Status: **LANDED; NEW NEMO LADDERS ACQUISITION_NEEDED**. The one certified SWG
card is now parameterised by a retained, resolved namelist. Its shipped default
is byte-identical, the shared acquisition tools preflight both `SWG` and
`SWG_DG`, and the CPU long-run driver's direct test passes. No NEMO acquisition
or GPU integration was run in this round.

## Provenance and case boundary

The incoming tip was `1d7fae76681ecd6e551bebca6892a1953d2e9c79`.
Preregistration is commit `b3a5aaeae10979dce1b047e87f4cd931f1ebbea5`;
the implementation is commit `a6d3b91c9b35eb8c274f49b3b4d9cc9b4b8489f3`.

The earlier 4,000 km smoke run and cancelled three-year run remain retracted:
they used the shipped fixed 2,010 km mask and therefore represented the shipped
single gyre. The replacement source case is `tests/SWG_DG`. Its `MY_SRC` tree
is byte-equal to shipped `tests/SWG/MY_SRC` except for the two upper-limit
statements in `usrdef_zgr.F90`. The committed source patch is byte-identical to
the operator's patch at SHA-256
`5021a71143943a34943497f40b72e86a43073ab7ff089646fdb3418cb029ba9e`.
The `EXP00` input files are byte-equal to shipped `EXPREF`; the case-local cpp
file is necessarily renamed and carries the sanctioned non-XIOS key set
`key_qco key_vco_3d key_RK3`. No other physics source or case input differs.

The one-line physics change is **basin extent follows `rn_domsiz`**: the lower
limits remain 10 km and both upper limits become `rn_domsiz + 10 km`. The wind
source is unchanged. The preflight additionally proves that the compiled
source contains both changed statements and that the reference executable is
newer than that compiled source.

Two independent adversarial reviews initially returned HOLD. Their findings
were dispositioned before the final PASS/PASS verdicts: resolved-namelist
provenance and every derived grid/coefficient were made binding; non-default
parameter and one-ULP grid plants were added; the operator patch became an
exact byte copy; run-stat fields and all launcher-edited namelist fields were
covered; acquisition receipts now hash the actual reference and instrumented
binaries and namelists; and long-run outputs now retain pending/completed
status plus per-day hashes.

## Parameterised card

`build_swg_card(namelist=...)` now consumes and retains the resolved SWG
namelist. The executed grid and mask take `nn_SWG`, `rn_dx`, and `rn_domsiz`;
the analytic forcing takes `rn_tau`; constants take `rn_modified_grav`; the
constant Laplacian coefficient is `rn_Uv*rn_Lv/2`; drag takes `rn_rfr`;
Coriolis arrays take `rn_beta` and `rn_f0`; and the clock takes `rn_Dt` and
`nn_itend`. Unsupported switches fail closed. Public validation recomputes the
clock, gravity, ENS selector, viscosity, drag, T/U/V/F Coriolis arrays, every
face metric, cell areas, dimensions, and wet mask from the retained namelist.

For the Maddison deck this resolves to 4,000 km, `nn_SWG=16`, 6.25 km,
644 by 644 stored points, exactly 640 by 640 wet points, one 500 m layer,
`rn_tau=0.10`, `g=9.80665`, `ahm=10`, `rn_rfr=1e-7`, `beta=2e-11`,
`f0=0.7e-4`, and `rn_Dt=50 s`. The unchanged compiled 2,000 km cosine gives
easterlies at both walls and westerlies at mid-basin.

The no-argument shipped card digest remains `f9d46bfd08f23702`. Fresh final
rescoring is fully bit-identical and AT-BAR: given-entry 338/338, chained
338/338, independent 50/50, carried state 29/29, and month 690/690. The stage
and month one-ULP controls each exit 1 with `all_at_bar=False` and
`bit_identical=False`.

## Acquisition tools and preregistered ladders

One `run.sh` and one `run_30d.sh` select `--deck shipped|maddison`. Deck choice
sets the source case, target name, committed namelist patch, and evidence path.
The Maddison source preflight requires `SWG_DG` to equal shipped `MY_SRC` plus
exactly the pinned patch. The daily writer uses `NINT(rday/rn_Dt)`; shell
preflight derives and checks `86400/rn_Dt`, the thirty-day step count, and both
daily output/restart cadences. Stage recording remains `kt <= nit000+9`.

All four real no-run preflights pass at implementation commit `a6d3b91c9`:

- shipped stage: 10 steps, target `SWG_OMIP_L1_P3`;
- shipped month: 720 steps at 24 steps/day, target `SWG_OMIP_L1_P3D`;
- Maddison stage: 10 steps at 6.25 km, target `SWG_DG_OMIP_L1_P3`;
- Maddison month: 12,960 steps at 432 steps/day, target
  `SWG_DG_OMIP_L1_P3D`.

The new ten-step ladder is preregistered at `nn_SWG=16`. Thirty stage frames,
each containing thirteen float64 644 by 644 arrays, occupy about 41.13 MiB per
frame and 1.205 GiB total before ordinary output, mesh, and restart files. The
thirty-day ladder uses the same physics at the operator-selected `nn_SWG=4`,
25 km, `rn_Dt=200 s`, 432 steps/day, and 12,960 steps. Its thirty stage-one
frames are about 2.67 MiB each and 0.078 GiB total. The coarser month is the
operator's protocol choice because the 640 by 640 single-rank reference plus
instrumented acquisition is estimated at about ten hours. The fine-grid month
remains **OPEN**.

The registered prediction for both new ladders is AT-BAR and bit-identical at
every row, with the one-ULP plants binding. If a difference appears, it must be
reported before attribution; the named first suspects are explicit `9.80665`
versus NEMO's `phycst` gravity association, the 4,000 km basin with unchanged
2,000 km wind period, and 640-cell halo handling. Missing evidence is
`ACQUISITION_NEEDED`, not confirmation or refutation.

Exact operator commands and destinations are:

```text
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1/
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run_30d.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1_30d/
```

The operator-only path writes resolved namelist lines, hashes both binaries
and namelists, requires both Maddison `run.stat` files, and preserves the tool,
source-patch, and legoESM commit hashes. This coding round did not run
`makenemo` or `mpirun`.

## Long-run driver and tests

`scripts/run/run_swg_double_gyre.py` integrates the same parameterised card
from rest for a positive integer number of model days and writes daily native
SSH/U/V NPZ files. Every output carries the clean legoESM Git SHA, full resolved
namelist, timestep, backend, precision policy, certification state, and a
SHA-256 receipt. A pending manifest is updated atomically after each day and is
promoted to `manifest.json` only after all requested days finish.

GPU mode requires physical GPU 1 and
`PrecisionPolicy.fp64(transcendentals="native")`; it is explicitly
non-certified pending the oracle ladders. The committed shell wrapper refuses
a dirty tree and pins `CUDA_VISIBLE_DEVICES=1`. CPU mode requires the CPU
backend and the certified fp64/libm policy for ladder comparison; this names
the policy, not an acquired deck certification.

The direct card/acquisition/driver battery passes 170/170. The driver test runs
two real production steps on a tiny grid and verifies daily SSH/U/V shapes,
float64 storage, provenance, pending-to-complete promotion, and the daily hash.
Shell syntax, new-file lint, and diff checks pass. No GPU run was attempted.

## Oracle citations

The compiled domain size, spacing, and stored dimensions derive from
`rn_domsiz`, `rn_dx`, and `nn_SWG` at
`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-166`. The replacement mask's
upper limits and executed U/V-point inequalities are at
`SWG_DG/BLD/ppsrc/nemo/usrdef_zgr.f90:211,213,220-221`. The single 500 m layer is
constructed at `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:158-168`. The
unchanged stress uses `rn_tau` and the fixed 2,000 km divisor at
`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:99-106`.
