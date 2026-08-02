#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gw_mut
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gw_mut_%j.out
#
# MUTATION PROOF for the per-cadence CUMULATIVE gateway dump (PR #1440).
# Breaks each new behaviour deliberately and shows the corresponding test go
# RED.  A GREEN test proves nothing; this is what makes the suite non-vacuous.
#
# Runs in a PINNED, ISOLATED worktree ($WT) so the PR worktree is never edited
# under a running job.  Every mutation asserts its own anchor is UNIQUE, so a
# silent no-op edit cannot masquerade as either a pass or a fail.
set -uo pipefail
WT=${WT:?set WT to the pinned mutation worktree}
cd "$WT"
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
BT=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_pt_gwmut_${SLURM_JOB_ID}
mkdir -p "$BT"
echo "=== worktree $WT at $(git -C $WT rev-parse HEAD) ==="
echo "=== dirty at start: [$(git -C $WT status --porcelain --untracked-files=no | tr '\n' ' ')] ==="

SECT=$WT/packages/ocean/legoesm/ocean/diagnostics_sections.py
DRIVER=$WT/scripts/run/run_omip_core2.py
DRV_T=tests/unit/test_run_omip_core2_gateway_transports.py
SECT_T=tests/ocean/unit/test_diagnostics_sections.py

# swap <file> <old> <new> — exact single-occurrence replacement, asserted.
# A STALE ANCHOR ABORTS THE JOB: a silent no-op mutation yields a GREEN test
# that would be misread as "the mutation did not bite".
swap() {
  $PY - "$1" "$2" "$3" <<'PYEOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
if n != 1:
    sys.exit(f"ANCHOR matched {n} times in {path} (need exactly 1): {old!r}")
open(path, "w").write(src.replace(old, new))
print("  [mutation applied]")
PYEOF
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! MUTATION HARNESS ABORT: the anchor above did not apply."
    exit 1
  fi
}

# ENFORCEMENT (codex round-1 YELLOW 7).  Printing a pytest status is not a
# result: a mutation that stays GREEN is a VACUOUS test, and a job that exits 0
# regardless would report that vacuity as success.  Every outcome is recorded
# and the job FAILS at the end if any mutation survived or the control is red.
GREEN_MUTATIONS=""
NRUN=0

run() {      # run <label> <pytest args...>
  local label=$1; shift
  echo "######## MUTATION $label ########"
  $PY -m pytest "$@" -q --no-header -p no:cacheprovider --basetemp="$BT/$label" 2>&1 \
    | grep -vE "jax_plugins|xla_bridge|cuda|versions_helpers|_check_cuda|plugin_module" \
    | tail -22
  local rc=${PIPESTATUS[0]}
  NRUN=$((NRUN + 1))
  echo "=== EXIT($label) $rc   (expect NON-ZERO) ==="
  if [ "$rc" -eq 0 ]; then
    GREEN_MUTATIONS="$GREEN_MUTATIONS $label"
    echo "!!! $label SURVIVED: the test does not detect this defect (VACUOUS)."
  fi
}

restore() { git -C "$WT" checkout -- "$1"; }

# ============================================================ the row content
echo "#### M1: dump the MEAN instead of the cumulative SUM ####"
swap "$SECT" '        cells.append(repr(vol[i]))
        cells.append(repr(tr[i]))' \
             '        cells.append(repr(vol[i] / max(acc.n, 1)))   # MUTATION
        cells.append(repr(tr[i] / max(acc.n, 1)))   # MUTATION'
run M1 "$SECT_T::test_cumulative_row_writes_the_sum_not_the_mean" \
       "$DRV_T::test_differencing_two_dumps_equals_a_fresh_accumulator_over_that_window"
restore packages/ocean/legoesm/ocean/diagnostics_sections.py

echo "#### M2: quantise the row to %.6f (the transports.txt format) ####"
swap "$SECT" '        cells.append(repr(vol[i]))' \
             '        cells.append(f"{vol[i]:.6f}")   # MUTATION'
run M2 "$SECT_T::test_cumulative_row_round_trips_exactly"
restore packages/ocean/legoesm/ocean/diagnostics_sections.py

echo "#### M3: header in a FIXED order, row in the accumulator's order ####"
swap "$SECT" '    cols = list(GATEWAY_CUMULATIVE_LEAD_COLUMNS)
    for nm in names:' \
             '    cols = list(GATEWAY_CUMULATIVE_LEAD_COLUMNS)
    for nm in sorted(names):   # MUTATION'
run M3 "$SECT_T::test_cumulative_row_columns_follow_the_accumulator_name_order"
restore packages/ocean/legoesm/ocean/diagnostics_sections.py

echo "#### M4: drop the names/arrays arity check ####"
swap "$SECT" '    if len(vol) != len(acc.names) or len(tr) != len(acc.names):' \
             '    if False:   # MUTATION: arity unchecked'
run M4 "$SECT_T::test_cumulative_row_rejects_an_accumulator_whose_arrays_do_not_match_names"
restore packages/ocean/legoesm/ocean/diagnostics_sections.py

# ================================================================ the writer
echo "#### M6: drop the header/accumulator name-match guard ####"
swap "$DRIVER" '        if tuple(acc.names) != gw_csv.names:' \
               '        if False:   # MUTATION: mis-assigned columns permitted'
run M6 "$DRV_T::test_row_refuses_to_write_columns_that_would_be_mis_assigned"
restore scripts/run/run_omip_core2.py

echo "#### M7: ignore io_proc (every rank clobbers the same file) ####"
swap "$DRIVER" '    if not io_proc:
        return None
    try:
        from legoesm.ocean.diagnostics_sections import gateway_cumulative_columns' \
               '    if False:   # MUTATION: io_proc ignored
        return None
    try:
        from legoesm.ocean.diagnostics_sections import gateway_cumulative_columns'
run M7 "$DRV_T::test_open_is_a_noop_on_a_non_io_rank"
restore scripts/run/run_omip_core2.py

echo "#### M8: write the per-cadence series INTO transports.txt ####"
swap "$DRIVER" 'GATEWAY_CUMULATIVE_CSV = "gateway_transports.csv"' \
               'GATEWAY_CUMULATIVE_CSV = "transports.txt"   # MUTATION'
run M8 "$DRV_T::test_the_cumulative_dump_leaves_transports_txt_byte_identical" \
       "$DRV_T::test_the_csv_filename_is_pinned"
restore scripts/run/run_omip_core2.py

echo "#### M9: swallow a disabled accumulator by writing a zero row ####"
swap "$DRIVER" '    if gw_csv is None or acc is None or gw_csv.fh is None:
        return' \
               '    if gw_csv is None or gw_csv.fh is None:
        return
    if acc is None:   # MUTATION: fabricate a row for a disabled accumulator
        gw_csv.fh.write("0,0,0\n"); gw_csv.fh.flush(); return'
run M9 "$DRV_T::test_row_is_a_noop_when_the_accumulator_was_disabled_mid_run"
restore scripts/run/run_omip_core2.py

# ============================================================ driver wiring
echo "#### M10: delete the run-end row (last window unrecoverable) ####"
swap "$DRIVER" '    _gateway_cumulative_row(_gw_csv, _gw_acc, n_steps,
                            n_steps * dt / _SEC_PER_DAY)' \
               '    pass  # MUTATION: no run-end row'
run M10 "$DRV_T::test_driver_writes_a_run_end_row_and_closes_after_the_loop"
restore scripts/run/run_omip_core2.py

echo "#### M11: move the cadence dump OUT of the snapshot block ####"
swap "$DRIVER" '            print(f"[snapshot] day {day:.0f} saved", flush=True)
            # Same cadence as the snapshot, and AFTER this step'"'"'s
            # gateway_step, so the row'"'"'s n_steps matches the snapshot'"'"'s day.
            # `step != n_steps` above excludes the final step; the run-end row
            # below covers it, so each dump point appears exactly once.
            _gateway_cumulative_row(_gw_csv, _gw_acc, step, day)' \
               '            print(f"[snapshot] day {day:.0f} saved", flush=True)
        _gateway_cumulative_row(_gw_csv, _gw_acc, step, 0.0)  # MUTATION: every step'
run M11 "$DRV_T::test_driver_opens_the_csv_before_the_loop_and_dumps_at_the_snapshot_cadence"
restore scripts/run/run_omip_core2.py

echo "#### M12: open the CSV inside the step loop (truncates every cadence) ####"
swap "$DRIVER" '                _gw_csv = _gateway_cumulative_open(
                    out_dir, _gw_acc.names, io_proc=_is_io_proc())' \
               '                pass  # MUTATION: opened lazily in the loop instead'
swap "$DRIVER" '        if snap_every > 0 and step % snap_every == 0 and step != n_steps:
            day = step * dt / _SEC_PER_DAY
            # persistent lane: snapshot cadence' \
               '        if snap_every > 0 and step % snap_every == 0 and step != n_steps:
            day = step * dt / _SEC_PER_DAY
            _gw_csv = _gateway_cumulative_open(   # MUTATION: reopened in loop
                out_dir, _gw_acc.names, io_proc=_is_io_proc())
            # persistent lane: snapshot cadence'
run M12 "$DRV_T::test_driver_opens_the_csv_before_the_loop_and_dumps_at_the_snapshot_cadence"
restore scripts/run/run_omip_core2.py

echo "#### M13: drop the blowup-abort dump + close ####"
swap "$DRIVER" '                # Dump what the accumulator reached before the abort: the last
                # cadence row alone would understate the run, and the run-end
                # writer below is never reached on this path.
                _gateway_cumulative_row(_gw_csv, _gw_acc, step, day)
                _gateway_cumulative_close(_gw_csv)' \
               '                pass  # MUTATION: abort path drops the dump'
run M13 "$DRV_T::test_driver_dumps_and_closes_on_the_blowup_abort_path"
restore scripts/run/run_omip_core2.py

# ================================================== the tests' OWN non-vacuity
echo "#### M14: make the window fixture DEGENERATE (both windows identical) ####"
swap "$WT/$DRV_T" '    win_b = _seeded_states(model, win_a[-1], [-0.02, -0.02])' \
                  '    win_b = _seeded_states(model, win_a[-1], [0.05, 0.05])  # MUTATION'
run M14 "$DRV_T::test_differencing_two_dumps_equals_a_fresh_accumulator_over_that_window"
restore "$DRV_T"

echo "#### M15: simulate the dump WRITING BACK to the state (1e-12 on S) ####"
swap "$WT/$DRV_T" '        acc = gateway_step(acc, stack, s_on, z1, geom, source="stored")
        R._gateway_cumulative_row(csv, acc, k, k * dt / 86400.0)' \
                  '        acc = gateway_step(acc, stack, s_on, z1, geom, source="stored")
        s_on = s_on._replace(   # MUTATION: the diagnostic perturbs the state
            S=s_on.S.replace(data=s_on.S.data + jnp.asarray(1e-12)))
        R._gateway_cumulative_row(csv, acc, k, k * dt / 86400.0)'
run M15 "$DRV_T::test_accumulating_and_dumping_does_not_perturb_the_trajectory"
restore "$DRV_T"

echo "#### M16: make the digest CONSTANT (the guard on the guard) ####"
swap "$WT/$DRV_T" '    return pytree_state_digest(
        state._replace(**{k: None for k in _MASS_FLUX_SLOTS}))' \
                  '    return "constant"  # MUTATION: digest is value-blind'
run M16 "$DRV_T::test_the_digest_used_below_is_actually_value_sensitive"
restore "$DRV_T"

# ====================================== codex round-1 fixes (M17 replaces M5)
echo "#### M17: emit the STEP index as n_steps instead of the acc count ####"
echo "####      ROUND-1 HISTORY: as M5 this SURVIVED against"
echo "####      test_the_final_row_reproduces_the_transports_txt_whole_run_mean"
echo "####      because every row test had step == acc.n.  Both targets run"
echo "####      now; the first is the one that actually discriminates."
swap "$SECT" '    cells = [str(int(step)), f"{float(day):.6f}", str(int(acc.n))]' \
             '    cells = [str(int(step)), f"{float(day):.6f}", str(int(step))]   # MUTATION'
run M17 "$DRV_T::test_n_steps_is_the_accumulator_count_not_the_step_index" \
        "$DRV_T::test_the_final_row_reproduces_the_transports_txt_whole_run_mean"
restore packages/ocean/legoesm/ocean/diagnostics_sections.py

echo "#### M18: drop the duplicate-endpoint suppression ####"
swap "$DRIVER" '        if gw_csv.last_n is not None and n == gw_csv.last_n:' \
               '        if False:   # MUTATION: duplicate dump points permitted'
run M18 "$DRV_T::test_a_repeated_dump_point_is_not_written_twice"
restore scripts/run/run_omip_core2.py

echo "#### M19: retry a broken writer instead of disabling it ####"
swap "$DRIVER" '        try:
            gw_csv.fh.close()
        except Exception:  # noqa: BLE001 -- already failing; nothing to add
            pass
        gw_csv.fh = None' \
               '        pass  # MUTATION: keep the broken handle live'
run M19 "$DRV_T::test_a_write_failure_disables_the_writer_instead_of_retrying_it"
restore scripts/run/run_omip_core2.py

echo "#### M20: discard the accumulator without salvaging the tail ####"
swap "$DRIVER" '                _gateway_cumulative_row(_gw_csv, _gw_acc, step - 1,
                                        (step - 1) * dt / _SEC_PER_DAY)
                _gw_acc = None' \
               '                _gw_acc = None  # MUTATION: tail discarded'
run M20 "$DRV_T::test_driver_salvages_the_accumulated_tail_when_gateway_step_fails"
restore scripts/run/run_omip_core2.py

echo "#### M21: change the transports.txt key names (output contract) ####"
swap "$DRIVER" '                fh.write(f"gateway_{nm}_vol_Sv {vol_sv:.6f}\n")' \
               '                fh.write(f"gw_{nm}_volume_Sv {vol_sv:.6f}\n")   # MUTATION'
run M21 "$DRV_T::test_transports_txt_content_is_pinned_exactly"
restore scripts/run/run_omip_core2.py

echo "#### M22: COMMENT OUT the run-end row (source scan must not match) ####"
swap "$DRIVER" '    _gateway_cumulative_row(_gw_csv, _gw_acc, n_steps,
                            n_steps * dt / _SEC_PER_DAY)' \
               '    # MUTATION _gateway_cumulative_row(_gw_csv, _gw_acc, n_steps,
    #                         n_steps * dt / _SEC_PER_DAY)'
run M22 "$DRV_T::test_driver_writes_a_run_end_row_and_closes_after_the_loop"
restore scripts/run/run_omip_core2.py

# ==================================================================== control
echo "=== CONTROL: unmutated worktree must be GREEN ==="
DIRTY="$(git -C "$WT" status --porcelain --untracked-files=no | tr '\n' ' ')"
echo "tree dirty? [$DIRTY]"
$PY -m pytest "$DRV_T" "$SECT_T" -q --no-header -p no:cacheprovider \
  --basetemp="$BT/control" 2>&1 \
  | grep -vE "jax_plugins|xla_bridge|cuda|versions_helpers|_check_cuda|plugin_module" \
  | tail -10
CONTROL_RC=${PIPESTATUS[0]}
echo "=== EXIT(control) $CONTROL_RC   (expect ZERO) ==="

# ===================================================== the verdict, enforced
FAIL=0
echo "=== MUTATIONS RUN: $NRUN ==="
if [ -n "$GREEN_MUTATIONS" ]; then
  echo "!!! SURVIVING MUTATIONS (vacuous tests):$GREEN_MUTATIONS"
  FAIL=1
fi
if [ "$CONTROL_RC" -ne 0 ]; then
  echo "!!! CONTROL IS RED: the unmutated suite does not pass; every mutation"
  echo "!!! result above is uninterpretable."
  FAIL=1
fi
if [ -n "$DIRTY" ]; then
  echo "!!! TREE WAS ALREADY DIRTY at the control: a mutation may not have been"
  echo "!!! restored, so the control did not test the committed code."
  FAIL=1
fi
if [ "$FAIL" -eq 0 ]; then
  echo "=== MUTATION SUITE: ALL $NRUN MUTATIONS DETECTED, CONTROL GREEN ==="
else
  echo "=== MUTATION SUITE: FAILED ==="
fi
echo "=== MUTATION SUITE DONE ==="
exit $FAIL
