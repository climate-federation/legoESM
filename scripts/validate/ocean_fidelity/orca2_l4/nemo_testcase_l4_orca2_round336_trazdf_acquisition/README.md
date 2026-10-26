# ORCA2 round-336 `tra_zdf` acquisition

This operator-only tool extends the reviewed round-333 acquisition without
changing any ORCA2 configuration value. Its default action is a preflight;
that path does not build or execute NEMO:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_trazdf_acquisition/run.sh --preflight-only
```

The extension targets kt 2, RK stage 3. `r336_begin` copies `Krhs` immediately
before `tra_zdf_imp`, because this deck assigns the same RK slot to `Krhs` and
`Kaa` and the solve overwrites it. Five calls outside `zdf_phy`'s arithmetic
loops snapshot the kt=2 boundaries after the TKE `avt_k` copy, runoff-mouth
mixing, EVD, DDM, and IWM. `r336_capture` is called once after `tra_zdf_imp`,
outside all of NEMO's arithmetic loops. It records those boundaries plus the
final `avt`, `avs`, `akz`, raw geometry and RK levels, then replays the exact
source-order matrix construction, eliminated diagonal, forward sweep and back
substitution into a passive record. In particular the RHS is evaluated as
`p2dt * (e3t_3d * (1 + r3t(Kmm) * tmask)) * Krhs`, with NEMO's parentheses;
the rejected `((p2dt * e3t_3d) * (1 + ...)) * Krhs` association is an explicit
admission plant. Admission requires the replayed
back-substitution to be bit-identical to the live post-solve `Kaa` on every
executed cell; it therefore cannot admit a reconstruction with changed
association or order.

NEMO declares `avt`, `avs`, and `avt_k` as `A2D(0),jpk`, so those fields and
all eleven phase snapshots are written and read at their real interior
`ntsi:ntei,ntsj:ntej,jpk` extent (90x148x31 for this build), never padded to
`jpi,jpj,jpk`. Other fields retain their own declared full-grid extents.
Derived metrics and work arrays preserve `-0.0` outside the exact vertical
ranges NEMO fills; the tile bounds are stored in the header and checked at
admission. Each rank writes
`oracle_r336_trazdf_kt00000002_r0000.bin` or `_r0001.bin`. The 36 fields retain
declared extents and IEEE signed zeros and carry source-order names, explicit
rank/origin/extent metadata and XOR payload checksums. The gate also re-admits
the inherited round-333 tracer and internal-LDF records and cross-checks the
new `akz`, geometry, RK stretches, `Kbb`, pre-solve `Krhs`, and live `Kaa`
against them.

Preflight applies the additions-only patch with zero fuzz to pristine source
preprocessed by the record build's own `cpp`; its nonblank output must equal
the real record-build ppsrc before the patch is applied. It audits the
resolved DDM/EVD/IWM,
mass-flux, adaptive-advection, and MSC selectors, and plants a changed value
for each. It compiles the actual record-build `trazdf.f90` with
`-O3 -funroll-all-loops -fno-tree-vectorize`; the floating-point census is
computed separately and must be nonzero for `tra_zdf` and `tra_zdf_imp`, and
must be unchanged by the observer calls while an in-function arithmetic plant
fails. `zdf_phy` receives the same function-scoped proof. A two-rank,
signed-zero fixture compiles
and runs the actual patched `trazdf` ppsrc with those production arithmetic
flags plus `-fcheck=bounds`, comparing its live state bitwise with a separately
compiled unpatched module before checking the passive record. Structural,
checksum, finite-value, wet-coverage, signed-zero, exact-RHS-association,
replay, filled-range, time-level, contribution-boundary, and live-result plants
must all fire. Preflight validates every staged `namelist_cfg` assignment
against the same group in NEMO's `namelist_ref`, with misspelled and misplaced
plants. A run containing `E R R O R` is refused after quoting NEMO's exact
matching lines. Every round-333 helper, plant,
syntax/same-program proof, deterministic rerun, admission identity check, and
refused-output retention path remains active. The reachable parent run path
checks and quotes `E R R O R` before it handles a nonzero `mpirun` status, so
the parent cannot move the run aside before that refusal is reported.

Only the operator may request the NEMO run:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_trazdf_acquisition/run.sh --run
```

The expected evidence directory is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition/orca1ice_trazdf_internal_kt2_np2`.
Set `R328_EXISTING_RUN` to a retained refusal and use `--admit-existing` to
repeat admission without rebuilding or running NEMO. Admission is pinned to
the run manifest's producer commit (which must be a local ancestor of the
admission commit) and to the manifest's exact `nemo` SHA-256 on both the
in-place and retained routes. Preflight plants each pin on both routes. Every
refusal log remains with the evidence.

If a `--run` refusal occurs after a target build was created, the build is
retained in place at
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF`.
Before re-running, move that complete directory aside to a new explicit
diagnostic name under the same `cfgs` directory; never delete it. For example:

```bash
mv /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF \
   /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF_refused_<UTC>
```

Then invoke the same `run.sh --run`. A refused run directory is already moved
under the evidence directory's `refused/`; set `R328_EXISTING_RUN` to that
retained path and use `--admit-existing` if only admission needs repeating.
