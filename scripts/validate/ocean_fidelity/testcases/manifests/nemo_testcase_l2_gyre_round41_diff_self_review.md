# Round 41 — adversarial diff self-review

Verdict: **PASS FOR OPERATOR ACQUISITION AFTER SIX CORRECTIONS; COMMIT BLOCKED
BY THE SANDBOX**.

1. The first patch attempted to write bare `e3t_0`/`e3u_0`/`e3v_0`/`e3w_0`
   objects, but this build exposes those through function-like preprocessor
   substitutions.  Corrected: every reference thickness is materialised
   elementwise through `E3*_0(i,j,k)`, exactly as the admitted round-40 writer
   does.
2. Importing the vortex-force flag from `sbcwave` unnecessarily widened the
   module dependency.  Corrected: the flag is owned by `sbc_oce`; `sbcwave` is
   cited only for the resolved initialization that sets it false.
3. The first model-input crop removed three halo rows where the campaign's
   canonical reader removes two.  A direct call through both shared model
   paths failed on incompatible `(18,29)` versus `(22,...)` geometry and
   caught it.  Corrected to `[2:-2,2:-2]`; the same diagnostic now returns
   `(22,32,30)` KEG and ZAD fields under CPU production JIT.
4. The source replay originally assumed `ntsi=4,ntei=33`; round 40's admitted
   header is `(ntsi,ntei,ntsj,ntej)=(3,34,3,24)`.  Corrected and pinned in the
   reader before any loop executes.
5. The first synthetic test fixture had a wrong V-array extent and all four
   tests failed before reaching the reader.  Corrected; the focused file is
   now 4/4 PASS, including truncation, nonzero effective-wsd, header, and
   source-replay controls.
6. Scientific KEG/ZAD plants could have returned nonzero merely because the
   unplanted residual was already red.  Corrected: each must make its own U
   row exceed `0.5` absolute after a planted unit perturbation as well as exit
   nonzero.

The final NEMO deltas apply to the exact canonical `dynadv.F90` and admitted
round-40 `stprk3_stg.F90` with zero removed lines.  The run script refuses a
dirty tree, an existing target, wrong resolved switches, a stale compiled
writer, vector-math symbols, a raw move of the deterministic kt=1 tracer
record, any consumed-field move, any source-replay/round-40 closure move, and
any plant that stays green.  It writes only the declared round-41 evidence
directory and never deletes a target.

The given-input score calls `_bc_ke_and_pressure_gradients` for C2 KEG and
`nemo_advective_vertical_momentum_advection` for ZAD: both are the shared
production implementations, not host reimplementations.  The independent
NumPy statements are admission calibration only and cannot produce a
scientific verdict.

One environmental blocker remains outside the diff: `.git/index.lock` cannot
be created because this sandbox mounts `.git` read-only.  Consequently the
requested commits cannot be made here, and `run.sh` correctly refuses until
the operator commits the explicit path set.  No measurement was taken under
an uncommitted stamp.
