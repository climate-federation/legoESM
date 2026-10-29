# Preregistration: ORCA2 round 345 — Decision-188 year-comparison gate

Date: 2026-10-29. Base: `d3a0a329c6925786815c4684593e88ff7d58167d`.
Imported round-342 tool HEAD:
`c983dc345a0b30c3ac5287d796986a069d981289`.

Admitted arm-A pins computed from the retained files: binary SHA-256
`835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d`,
rank-0 day-1 restart SHA-256
`3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3`,
and rank-1 day-1 restart SHA-256
`a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023`.
The deterministic perturbed rank-file SHA-256 values are
`bb25179365e776ae73e6ead9d2bad42e222f2094f687c22ede3b0eb3bcae1f01` and
`0279bb4021ef1b557e178ef01b5c4cac732479102404871b557b984433fff8cf`.

This corrected registration incorporates the operator's Decision-188
clarification and supersedes the rejected float32 input-file perturbation.

## Frozen work

1. Extend the observer-free round-342 K0 year tool by two restart arms. Both
   copy admitted arm A's day-1 double-precision restart pair at `kt = 8`,
   restart NEMO at `nn_it000 = 9`, and continue through `nn_itend = 2920`.
   The control changes no restart bit. In the perturbed arm, each finite wet
   top-level `tn` value advances by exactly one float64 representable value
   toward positive infinity; that perturbed pair remains day 1 of the envelope.
2. Record the variable, dtype, wet-cell selection, direction, changed-cell
   count, and source/perturbed file digests. Re-read every variable, dimension,
   and attribute to prove that nothing except selected `tn[0, 0, :, :]` values
   changed. The cold-start temperature file remains untouched, so the T/S
   damping target remains untouched too.
3. Stage the same deck, forcing, damping files, build snapshot, and binary as
   admitted arm A. Re-hash every staged input against arm A. Pin admission to
   the retained producer manifest, producer commit, and binary digest.
4. Admit the envelope only after every control restart pair from `kt = 8`
   through `kt = 2920` is byte-identical to arm A. This proves restart
   continuation itself is exact before perturbation growth sets the KE bar.
5. Run the same observer-free real-flag census, bounds fixture, namelist,
   forcing, completion, retention, and stub checks inherited from round 342.
   The round-345 stub uses scratch below
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work`, exercises success,
   refusal, in-place `--admit-existing`, and re-admission directly from a
   retained `refused/` arm. A failed in-place admission moves both new arms
   into `refused/`, records `admit_existing.log`, and keeps them there on
   failure. A passed refused/ re-admission promotes both arms to their canonical
   run-root paths after producer-commit ancestry and manifest digests pass.
6. Apply Decision 188 month by month. Fixed bars remain SST and T-band RMS
   `0.01 K`, SSS and S-band RMS `0.002 psu`, SSH RMS `0.001 m`, PSI RMS
   `0.1 Sv` plus pattern correlation `>= 0.999`, Drake absolute monthly-mean
   difference `0.5 Sv`, and MLD RMS `1 m`. Surface KE passes when its relative
   RMS is at most `max(0.01, 3 * envelope_relative_rms)` for the matching
   month. The envelope is perturbed NEMO versus admitted arm A from day 1.

## Frozen confirmations and falsifiers

- Perturbation **CONFIRMS** only if at least one wet surface cell changes,
  every selected value is exactly one float64 ULP toward positive infinity,
  every unselected restart value and all metadata are bit-identical, and each
  source/perturbed digest differs. Otherwise it **REFUTES** the tool.
- Deck/input/binary identity **CONFIRMS** only if every forcing and damping
  digest equals admitted arm A and the real-run binary digest is exactly arm
  A's. Any mismatch **REFUTES** admission.
- A pre-existing target arm must refuse in place and remain at the same path;
  any move to `refused/` **REFUTES** the safety contract.
- A failed arm created by this invocation must retain its binary and build
  snapshot under `refused/`; a completed retained arm must admit directly via
  `--admit-existing`. Failure of either route **REFUTES** the retention tool.
- A fixed-bar plant with SST RMS `0.010000000000000002 K` must fail while an
  otherwise identical report at `0.01 K` passes.
- A KE plant at relative RMS `0.02` with envelope relative RMS `0.007` must
  pass. A candidate at `0.011` with a nonzero envelope below the `0.01` floor
  must fail, while `0.01` passes.
- A zero envelope and an envelope lacking the admitted fp64 restart
  perturbation must both refuse before any verdict.
- A zero KE envelope in any evaluated month must refuse. The gate may not use
  an envelope until the full unperturbed restart control passes.
- The ORCA2X pilot-shaped 30-day report must produce exactly one evaluated
  partial-month verdict and no invented pending months.

No model configuration, physics default, numerical bar, NEMO source, forcing,
or damping file is changed by this round. No NEMO executable is run by the
author.
