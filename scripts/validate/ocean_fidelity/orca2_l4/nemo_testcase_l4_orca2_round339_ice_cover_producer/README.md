# ORCA2 round-339 ice-cover producer

Operator-only Tool A. Its default action is a read-only preflight and never
builds or runs NEMO:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round339_ice_cover_producer/run.sh --preflight-only
```

`--run` rebuilds the admitted round-5 full-SI3 program with one passive
post-`ice_stp` observer, stages the identical two-rank initial deck with a
2,920-step endpoint, records all 1,460 forcing-step `fr_i` arrays, and creates
`ice_cover_clim.nc` offline. It reserves 90 minutes plus build time. The raw
records, output file, build/program proof, resolved namelist, logs and digests
are retained.

A refused run is moved under this round's `refused/` evidence directory and
both the observer-free and observed builds remain in `cfgs/`. Move those two
build directories aside to unique retained names before retrying `--run`;
never delete them. Set `R339_EXISTING_RUN` to a retained run and use
`--admit-existing` to repeat admission without building or running NEMO. That
path trusts the retained manifest's producer commit and binary digest.

Expected operator sequence:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round339_ice_cover_producer/run.sh --run
```
