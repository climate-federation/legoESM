# PREREGISTRATION — end-wall split-explicit source

Frozen before any new barotropic measurement on this branch.

## Candidate and existing limit

Candidate: a difference inside legoESM's split-explicit barotropic wall-row
update versus DINO's active `dynspg_ts` path injects the sustained end-wall 2dt
mode. Existing five-state matched-state deposit maps do not assign ownership:
their two-step projection annihilates a state-constant field, and about 95% of
the wall residual is state-constant. They measure injection from a fresh
bridge, not free-run amplification.

## Exact ownership number and frozen bars

The final ownership measurement, once one statement-level mechanism is
selected, is a five-day per-step free-run A/B from the same bridged day-180
state. The prior last-half/zonal-wall wording is retracted: its bars came from
the whole-domain first-eight ratio and first-sample aggregate-wall share. The
primary is now coherently
`regions.all.ratio_lego_over_nemo.first8` from `eta_flicker_decay.py`; this is
the mean of the first eight per-sample area-weighted wet-domain RMS amplitudes
of the **2dt-alternating eta component**. Its companion is
`wall_share.legoESM.first8.wall`. Certified shipped-card values
are 2.886305221717904 and 0.48541937969116244.

- **Control validity:** shipped-card ratio in `[2.60, 3.18]` and first-eight
  aggregate-wall share in `[0.38, 0.59]`.
- **CONFIRMS ownership:** candidate ratio `<=1.25` and share `<=0.17`.
- **REFUTES ownership:** candidate ratio `>=2.30` and share `>=0.38`.
- Otherwise: **UNRESOLVED**.

The candidate may replace only one measured statement-level DIFF with literal
`dynspg_ts` ordering/arithmetic. It must not substitute oracle state or tune a
coefficient. Its source diff must show one changed mechanism and its day-0
bridge must be bit-identical.

## Registered next discriminator — STOP before a GPU arm

The live eta candidates are (i) NEMO's per-substep ssh/velocity halo/LBC
commit versus legoESM's masked serial representation and (ii) legoESM's
post-solver uniform `fix_eta_drift` projection. The temporary NEMO Kmm
transport-mean installation is **not** a candidate: active vector `dyn_zdf`
constructs Kaa from Kbb and Krhs, tracer transport is routed separately, and
`mlf_baro_corr` later restores the momentum mean.

Cross-model first divergence is only context, not attribution. Before any GPU
arm, build two same-input counterfactual probes, reusing
`spg_substep_chain.py`/`substep_traj_compare.py` conventions with wind ON:

1. **Boundary representation.** From one identical substep state and identical
   tendencies, evaluate the first subsequent face-depth/flux,
   pressure-gradient, and Coriolis consumers twice: once with the shipped
   serial mask/periodic representation and once with a literal mapped NEMO
   `lbc_lnk` scalar/vector commit. Repeat at the final post-solver
   `finalize_lbc` boundary commit and evaluate the first next-step consumers;
   do not assume continuous masking exonerates its wall-Nyquist feedback.
   Print j=1/j=197 pointwise deltas. A planted vector-sign or halo-offset
   violation must fail. The exact primary score is
   prediction normalized error of this counterfactual delta against the
   independently dumped NEMO-minus-lego first-consumer residual. Define
   explained RMS fraction as
   `E = 1 - RMS(residual - prediction) / RMS(residual)`; correlation is the
   companion. **CONFIRMS statement ownership** only when normalized error is
   `<=0.10`, `E>=0.90`, and correlation `>=0.99`; **REFUTES** only when
   normalized error is `>=0.90`, `E<=0.10`, and correlation `<=0.20`;
   otherwise `UNRESOLVED`. NEMO's second post-loop LBC
   on normalized `un_adv/vn_adv` is tracer-transport bookkeeping and must be
   dumped as a control, but is scoped out of eta ownership unless a next-step
   eta consumer is demonstrated.
2. **Uniform eta fixer.** Capture the shipped pre-fix eta and exact scalar
   correction. From that same state, evaluate the next step's first
   face-depth/flux and pressure-gradient consumers with correction `c` versus
   zero, holding every other input fixed. Print the same prediction error,
   correlation and explained fraction against the independently dumped
   next-step residual, separately at j=1/j=197. Plant a known nonzero `c` and
   require the consumer to move. Apply the same symmetric confirmation
   (`error<=0.10`, `E>=0.90`, correlation `>=0.99`) and refutation
   (`error>=0.90`, `E<=0.10`, correlation `<=0.20`) bars.

Only a counterfactual that confirms uniquely may receive an executable
one-variable selector. Amend this preregistration with that selector and exact
invocation **before** the five-day free run, then apply the frozen ownership
bars above. If both confirm, report `UNRESOLVED` and do not combine them.

No executable barotropic candidate selector exists yet, so this registration
deliberately contains no nominal GPU command. No GPU arm runs in this lane.
