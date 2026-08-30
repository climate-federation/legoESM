# Preregistration: DINO standalone cold-start barotropic corrector

Date: 2026-08-30. Session: `01a053d4-8e9f-7212-bbdb-19ba2d64e140`.
Branch: `fidelity/dino-transfer-codex`. **PREREGISTERED BEFORE THE PHYSICS
CHANGE.** No GPU, NEMO, or MPI process is part of this build round.

## Frozen source ownership

The executed DINO/NEMO 5.0.2 path is the oracle, not the existing legoESM
warning. In
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/DOM/istate.F90`, a
non-restart start sets `l_1st_euler=.true.` at line 110, initializes velocity
at rest at lines 121--123, obtains the analytic DINO state at line 130, and
copies `Kbb` into `Kmm` at lines 135--137. In the DINO source that was actually
built,
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/stpmlf.F90`,
the same first step selects `rDt=rn_Dt` at lines 134--137. The call

```fortran
IF( ln_dynspg_ts ) CALL mlf_baro_corr ( kstp, Nnn, Naa, uu, vv )
```

is at line 578, after the vertical solves and before finalization. It is not
guarded by `.NOT.l_1st_euler`. The flag is cleared only at lines 685--688,
after that call. DINO has `ln_dynspg_ts=.true.`, so the first Euler step
executes the corrector. The corrector itself reduces the live `Kaa` column at
lines 752--761 and replaces its mean with `uu_b/vv_b(Kaa)` at lines 762--765.

Therefore legoESM's current no-history early return is a real one-step model
difference. A bridge hides it and is forbidden for T1.

## One change

Add `barotropic_cold_start_after_reconcile` with legal values `off` and
`nemo_mlf_baro_corr`. Default `off` preserves non-DINO cards. The faithful
oracle card `nemo_dino_kamm_mlf` and its public catalog identity
`nemo_dino_kamm_mlf_v1` both pin `nemo_mlf_baro_corr`.

The selector is fail-closed:

- an unknown value raises during model construction;
- a non-`off` cold-start selector requires a leapfrog-family outer integrator;
- it must equal `barotropic_after_reconcile`; a source pair that asks for a
  different regular-step and cold-start policy raises rather than silently
  constructing a mixed model.

On the no-history Euler bootstrap only, `_step_impl` captures the barotropic
target immediately before implicit vertical mixing, preserves the raw
pre-projection `Kaa` SSH already produced by the split-explicit solve, and
invokes the existing `_apply_after_level_reconcile` kernel immediately after
implicit mixing and before the conservation fixer. No second reconciliation
kernel may be introduced. Regular leapfrog steps keep their existing call
site and arithmetic. The forward-Euler sibling card remains `off` because it
does not implement NEMO's MLF stage layout; its registered stability bisect is
future work, not part of this change.

## Frozen controls and admission

The pre-change red state is the committed T1 runner's exact claim-length
refusal:

```text
cold-start leapfrog skips nemo_mlf_baro_corr on l_1st_euler
```

The change is admitted only if all of the following hold:

1. The claim-length admission test changes from that refusal to an empty
   blocker tuple while a truncated run remains explicitly non-claimable.
2. Both leapfrog-family no-history paths call the shared corrector exactly once
   with the selector on, after the implicit solve; the `off` plant calls it
   zero times and produces a measurably different planted column mean.
3. Unknown and mismatched selector pairs raise at construction.
4. The oracle and catalog DINO MLF configurations resolve with the selector on;
   T3 field identity remains zero-diff.
5. A CPU/fp64/no-JIT standalone smoke advances at least one step, is finite,
   emits no retired skip warning, and stamps the selected policy. This is a
   build control only, never a climate measurement.

No bar, climate statistic, ensemble floor, sampling date, or T1 member is
changed. After these controls pass, emit the six 20-year legoESM standalone
GPU arms frozen in `PREREG_dino_transfer_question.md`. Do not run them here.
