# ORCA2 round-339 K0 boot and ten-step smoke

Operator-only Tool B. Run it only after Tool A admits
`ice_cover_clim.nc`. Its default is a read-only preflight and does not build or
execute NEMO:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round339_k0_smoke/run.sh --preflight-only
```

`--run` performs an observer-free `key_si3` rebuild from NEMO's unmodified
sources with an empty `MY_SRC`, stages the round-5 deck,
applies only Decision 175's K0 namelist patch, pins Tool A's climatology, and
runs two ranks for ten steps. Admission requires `STOP 0`, no `E R R O R`, the
resolved K0 selectors, finite restarts, and retained resolved namelist and
`ocean.output`. The no-iceberg selector is checked through NEMO's real
`==>>>   No icebergs used` banner. The pre-run producer manifest includes the
complete preprocessed-source snapshot and proves the empty `MY_SRC`. It does
not score or certify S4 physics.

On refusal, the run moves under `round339/k0_smoke/refused/` and the K0 build
is retained. Move the retained build aside before retrying `--run`. Set
`R339_K0_EXISTING_RUN` to a retained run and use `--admit-existing` to rerun
admission against its producer and binary manifests without running NEMO.
