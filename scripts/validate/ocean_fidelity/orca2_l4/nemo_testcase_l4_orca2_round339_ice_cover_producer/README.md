# ORCA2 round-339 ice-cover producer

Operator-only Tool A. Its default action is a read-only preflight and never
builds or runs NEMO:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round339_ice_cover_producer/run.sh --preflight-only
```

`--run` builds the round-5 deck twice from NEMO's unmodified sources: an
observer-free baseline and a build whose `MY_SRC` contains only `sbcmod.F90`
plus the passive post-`ice_stp` `fr_i` writer. A whole-program proof compares
the observed build with that clean baseline after stripping the one declared
observer. It first runs the ten-step round-5 deck and
requires all four terminal restart files to equal the admitted observer-free
record byte for byte. It then stages the identical two-rank initial deck with
a 2,920-step endpoint, records all 1,460 forcing-step `fr_i` arrays, and
creates `ice_cover_clim.nc` offline. It reserves 90 minutes plus build time.
The passivity twin, raw records, output file, build/program proof, resolved
namelist, logs and digests are retained. The producer manifest is written
before the first NEMO launch and includes the complete preprocessed-source and
`MY_SRC` snapshot, so a refused retained run remains independently admissible.

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
