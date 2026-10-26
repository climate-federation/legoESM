# ORCA2 round 339 preregistration — standalone ice-cover producer and K0 smoke

Frozen 2026-10-26. Base: `94ec2462614a`. This round builds two operator-only
tools. The agent runs their read-only preflights but does not build or execute
NEMO.

## Tool A — yearly ice-cover producer

The producer is the admitted round-5 full-SI3 ORCA2 deck, with its source,
compiler flags, inputs, initial state, two-rank layout, calendar and all
scientific selectors unchanged. The only deck edits are the Decision-181 run
endpoint and terminal restart: `nn_itend = nn_stock = 2920`. An additions-only
observer runs immediately after `ice_stp` returns, outside NEMO's loops. On
each forcing step (`kt = 1,3,...,2919`) and rank it writes the declared local
`fr_i(jpi,jpj)` as raw binary64 plus rank, local extent, global origin,
owned-range and checksum metadata.

Frozen predictions and refusals:

1. Exactly 1,460 records exist per rank, with the complete odd-`kt` sequence,
   finite values and `0 <= fr_i <= 1`; missing, duplicated, reordered,
   truncated, checksum-bad, rank-swapped or non-finite records refuse.
2. The two ranks' owned interiors cover every native `180 x 148` cell exactly
   once. Any overlap or gap refuses. Halo values are retained in the raw
   records but never averaged into the global field.
3. The binary64 mean is accumulated cell-by-cell in increasing `kt`, one
   addition per sample and one final division by 1,460. A changed summation
   order plant and a one-ULP wet-cell plant must change the output digest and
   refuse.
4. `ice_cover_clim.nc` has exactly dimensions `time_counter=1`, `y=148`,
   `x=180` and one binary64 variable `ice_cover(time_counter,y,x)`, with no
   fill value, scale or offset. Its payload must be bitwise the offline mean,
   including land. Schema and payload plants refuse.
5. The observer-free and observed builds use the round-5 production flags.
   Their per-function floating-point instruction census is nonzero and equal
   after observer calls are stripped; an injected arithmetic operation must
   fail. The generic round-328 same-program helper proves every other
   preprocessed source and object unchanged.
6. The producer must report `STOP 0`, no `E R R O R`, and the resolved
   round-5 selectors. Producer commit, executable, build, deck, inputs, raw
   records, climatology and admission report are digest-pinned. A refusal
   retains builds and moves the run into `refused/`; re-admission uses its own
   producer and binary manifest, never the current tree.

Expected operator cost: about 55 minutes stepping from round 5's measured
rate; reserve 90 minutes plus build and offline assembly.

## Tool B — K0 boot and ten-step smoke

Tool B cannot start until Tool A's admitted climatology and manifest exist. It
builds the round-5 program without an observer, retaining `key_si3` and the
same compiler flags, then changes only the user-selected K0 runtime fields:
`nn_ice=1`, `ln_drgice_imp=F`, `ln_mxl0=T`, `nn_mxlice=0`,
`ln_icebergs=F`, and `sn_ice` points to Tool A's digest-pinned file. The
round endpoint remains ten steps.

Frozen predictions and refusals:

1. The staged variables must exist in the same groups of `namelist_ref`; a
   misspelled or group-moved plant refuses before build.
2. The compiled-key echo contains `key_si3` and not `key_xios`. The resolved
   `ocean.output` contains exactly the K0 values above and no `E R R O R`.
3. Both ranks boot, complete step 10, return zero and print `STOP 0`.
4. Every numeric variable in each ocean restart is finite. A NaN plant and a
   truncated-restart plant refuse. This is a smoke result only: it certifies
   no S4 value and scores no physics.
5. The resolved namelist, `ocean.output`, executable, input climatology,
   restarts, producer commit and manifests are retained and digest-pinned.
   Refused builds and runs follow Tool A's retention and re-admission rule.

