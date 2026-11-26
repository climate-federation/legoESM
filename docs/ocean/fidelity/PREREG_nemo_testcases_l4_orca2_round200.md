# ORCA2 round 200 preregistration — OMT-0 restart-contract recovery

Date: 2026-10-09. Frozen base: `cd9c3d234`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round200/`.
Every state comparison is labelled **independent OMT-0**. The shipped ORCA2
card, its sea ice, six ice selectors and `unmeasured_features` remain unchanged.

Round 199's operator run is prior evidence, not a measurement made in this
round. It confirmed the two-step smoke, then NEMO stopped the twelve-step twin
at kt=11 because active `|V|max=10.24 m/s` exceeded its compiled 10 m/s bound.
The attempted run left no admissible restart payload. This round corrects the
record contract before requesting another run; it does not change OMT-0.

## Compiled contract and frozen correction

With `ln_rst_list=.true.`, NEMO initialises `nitrst` from the first list entry,
opens only the current listed target, and advances only to the next list entry
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-101`, `:116-148`, `:188-202`).
It does not add `nn_stock` or an automatic terminal restart. Therefore round
199's expected step-2, step-12 and step-96 sentinels are **RETRACTED** and the
gate must stop expecting them.

The corrected protocol is:

1. reuse and re-admit round 199's successful two-step smoke;
2. run two independent ten-step twins with `nn_itend=10` and the ten-entry
   list `1..10`, so every requested file is closed before normal `STOP 0`;
3. attempt the unchanged 96-step month with the ten-entry list
   `10,20,...,90,95`; admit its closed step-10 restart, but require NEMO's own
   step-11 safety stop and report steps 20..95 as unavailable.

NEMO forms active maxima and salinity extrema at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:176-184`, refuses velocity above
10 m/s at `:243-250`, and reports the exact offending cell before `ctl_stop`
at `:293-316`. The prior run printed kt=11 V maximum 10.24 m/s at global
`(i,j,k)=(22,84,27)`, with SSH 3.853 m, U 3.041 m/s, salinity 21.58..37.25.
The recovery gate freezes that boundary. It may not disable, raise or bypass
the check. The 96-step month is **UNMEASURED WITH SPEC** beyond step 10 if the
boundary reproduces.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R200-P1 | Round 199's smoke is valid and OMT-0-resolved. | `STOP 0`, `RUN_DONE`, exact five OFF selectors and unchanged deck hash. | Any mismatch: **REFUTED**; request a fresh smoke before longer runs. |
| R200-P2 | Ten-step twin A and B complete normally and are rank-complete. | Both reach `STOP 0`; 40 restart shards at steps 1..10; five fp64 finite fields compare array-equal in 100 field comparisons. | Missing shard, non-finite, wrong header or bit: **REFUTED**; stop without a ladder claim. |
| R200-P3 | The identical 96-step protocol reproduces NEMO's own kt=11 safety boundary. | Step-10 restart is closed on both ranks and array-equal to both twins; kt=11 values/location match the frozen report and the process exits through compiled `stp_ctl`. | Clean continuation, earlier/different stop, or changed values/location: **REFUTED**; do not classify the month until reconciled. |
| R200-P4 | No legitimate month state exists after step 10 under OMT-0. | P3 confirms; gate reports requested steps 20..95 unavailable and never fabricates a sentinel or zero field. | Any later valid restart: **REFUTED**; admit it and continue the growth table. |
| R200-P5 | The recovery changes run bookkeeping only. | OMT-0 canonical deck hash remains `9279c638a9fe50b551fcca8dc62dc1c19c0fc2c55bdcae7f3501a8afeb5ca8ac`; package/card diff empty. | Any physical or card delta: stop. |
| R200-P6 | Controls bind. | Extra deck, live module, oversized list, missing smoke, missing rank, wrong kt, non-finite, twin ULP, calibration ULP, missing month-step-10, wrong oracle stop and changed binary each refuse. | Any green plant: invalid instrument; report no record claim. |

The round stops `STOPPED_FOR_RECORD` after a preflight-ready launcher. A future
round may build and score the legoESM OMT-0 card only after P1-P3 admit. No
stabiliser, threshold change, source patch, carried-state change, sea-ice
change or NEMO rebuild is allowed.

ASKED choices: Decision 103's OMT-0 deck and record protocol.  
UNASKED choices: empty.
