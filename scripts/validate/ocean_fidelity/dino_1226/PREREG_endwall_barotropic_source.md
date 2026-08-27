# PREREGISTRATION — end-wall split-explicit source

Frozen before any new barotropic measurement on this branch.

## Candidate and existing limit

Candidate: a difference inside legoESM's split-explicit barotropic update
versus DINO's active `dynspg_ts` path injects the sustained end-wall 2dt mode.
The existing five-state matched-state deposit maps are not an ownership test:
their two-step projection annihilates a state-constant field, and the measured
wall residual is about 95% state-constant.  They measure injection from a
freshly bridged state, not free-run amplification.

## Exact number and decision rule

The discriminating measurement is a five-day, per-step free-run A/B from the
same bridged day-180 state.  The primary number is the last-half,
per-cell-first zonal-wall 2dt amplitude ratio `A_lego/A_nemo`; the companion
number is the zonal-wall variance share.  The control is the shipped
`nemo_dino_kamm_mlf` card and must reproduce the bridged baseline ratio 2.89
within its registered interval and wall share 0.85 within 0.10 absolute.

The candidate arm replaces only the statement-level DIFF rows named by the
source alignment with literal `dynspg_ts` ordering/arithmetic.  It must not
substitute oracle state or tune a coefficient.  Before any run its diff must
show one changed mechanism, and its day-0 bridged state must be bit-identical.

- **CONFIRMS ownership:** candidate-arm ratio `<= 1.25` and wall share `<=
  0.08` while the control reproduces baseline.
- **REFUTES ownership:** candidate-arm ratio `>= 2.30` and wall share `>= 0.68`.
- Otherwise: **UNRESOLVED**.

The exact run shape, after a single alignment DIFF is selected and implemented,
is:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py \
  nemo_dino_kamm_mlf results/dino_1455/endwall_spg_candidate.npz \
  --days 5 --bridge-before --save-3d
```

Score the first 160 per-step eta samples with the existing
`eta_flicker_decay.py` per-cell-first operator.  No GPU arm is run in this
lane.  If the alignment leaves more than one live DIFF, this registration does
not authorize combining them: each receives a separate one-variable arm and
the same bars.

