# globalmicro LES references — handoff (2026-06-25)

Generating warm-cloud LES reference targets (BOMEX/DYCOMS/RICO) to train the
Morrison warm-rain scale-net. Runs on Ginsburg: `ssh ginsburg`, project at
`/burg/glab/users/kl3231/Projects/globalmicro`, shared env
`/burg-archive/glab/users/jn2808/.conda/envs/legoesm`, submit via
`scripts/cluster/globalmicro/run_les_reference.sbatch {bomex|dycoms|rico} [args]`.

## Done
- **f32 under-clouds → references run f64** (sbatch default). f32 keeps only 1–2
  digits of the `q_t−q_sat` residual. Mixed-precision tested (`--mixed-micro`):
  recovers ~85% of LWP but NOT cloud cover (that's f32 advection of `q_t`), and
  on RTX 8000 costs the same as full f64. GitHub issue **#618** (with the
  experiment writeup).
- **Non-spectral `pin` core is now the default** (`--core pin`, BOMEX driver).
  Holds more cloud than spectral and `clip_q≈0` (monotone advection). Codex
  review clean. 4 deck-free unit tests in
  `tests/unit/test_run_bomex_les_pin_core.py`.
- **Time-averaging added to all 3 drivers** — shared
  `les_record.TimeMeanAccumulator`; saved reference is the hrs-`avg_from..T`
  time-mean (default T/2), with `*_snap` + `avg_from_hours`/`n_avg_samples`.
  Tests: `tests/unit/test_les_timemean_accumulator.py`. Critical fix: snapshots
  swing wildly (one BOMEX snap LWP=20.6, another 3.2) — time-mean is stable.
- **BOMEX reference regenerated** (pin f64, 64³, 6h): time-mean
  **cc=0.075, LWP=5.99 g/m²** (hrs 3–6, n=22) — LWP in GCSS band. Local data +
  figures at `results/les_bomex_pin/` (gitignored). cc 0.075 slightly below the
  0.10 floor = resolution residual at dx=100 m.

## TODO (next session)
1. **Extend pin to DYCOMS** — add Stevens longwave operator-split into the pin
   loop (pin has no radiation hook; apply externally like micro).
2. **Extend pin to RICO** — interactive bulk surface over SST is NOT a pin
   `surface` option (`free|flux|most_cooling`); inject the per-step flux
   externally (`surface="free"` + bottom-cell flux from mean state).
3. Re-run codex on the time-averaging diff (CLI timed out twice this session —
   covered by tests + inspection meanwhile).
4. Task #4 still open: wire LES npz → SCM trainer reference (regrid LES z-grid →
   SCM col grid; RICO uses `precip_mm_day` time-mean now, not `precip_accum_mm`).
5. Committed on branch `kara/microphysics-global-opt` (driver/test/doc changes).
   The `run_les_reference.sbatch` is gitignored (repo `*.sbatch` policy) so it
   stays local-only; not pushed.

## Files changed (uncommitted)
- `scripts/run/run_bomex_les.py` (--core, --mixed-micro, time-avg)
- `scripts/run/run_rico_les.py`, `scripts/run/run_dycoms_les.py` (time-avg)
- `scripts/run/les_record.py` (TimeMeanAccumulator)
- `scripts/cluster/globalmicro/run_les_reference.sbatch` (f64 default)
- `tests/unit/test_run_bomex_les_pin_core.py`, `tests/unit/test_les_timemean_accumulator.py`
