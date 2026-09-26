# Preregistration: row-10 literal Langmuir rerun

Date: 2026-08-28. Status when frozen: **production fix code present; post-fix
matched-state measurement not run**.

The accepted option is `tke_langmuir_evaluation`. `nemo_literal` is the
faithful default only on `nemo_dino_kamm` and `nemo_dino_kamm_mlf`;
`vectorized` is their explicit historical opt-in and remains the byte-identical
default everywhere else.

The literal path transcribes the active DINO no-Stokes chain at
`cfgs/DINO/MY_SRC/zdftke.F90:422-463`: left-associated `zWlc2`, a top-down
`zpelc` scan, per-column `imlc=mbkt+1` initialization, the bottom-up threshold
overwrite scan over `jk=jpkm1..2`, `zhlc`, `zus/zus3`, `zwlc`, and the TKE
source. It consumes the already-carried step-entry `rn2b`, `gdepw(Kmm)`,
`e3w(Kmm)`, `bottom_level`, and `wmask`. The literal path fails closed if a
generic external TKE source is also present; the historical path retains its
existing additive behavior.

## Frozen row-10 rerun

The dispositive number is the direct post-Langmuir `en` comparison against
`tke_dump_en_postlc.bin`, using the existing `1e-15` normalized maximum
per-column bar, all 9,920 wet columns, and the four registered southern focus
columns. CONFIRM is 0/9,920 failed columns and 4/4 focus passes. Any nonzero
failure REFUTES row closure and stops the sweep inside row 10, localized in
source order. The expected result if the registered construction is complete
is 0/9,920.

The write-only bracket, time-level registration, input/source/binary hashes,
planted perturbation, horizontal-roll, and nonfinite controls remain mandatory.
If row 10 verifies, rows 11--32 resume under the original committed call table
and bars in `PREREG_zdf_chain_sweep.md`; no later row is inferred from the
row-10 pass.

## Required production tests

- a hand-computed threshold/tie column preserving strict `>` semantics;
- unequal shallow/deep no-crossing columns proving each uses its own
  `mbkt+1` fallback, with a planted global-bottom construction that differs;
- eager/JIT and finite-gradient execution of the literal path;
- a mixed-source input that must fail closed;
- exact default-versus-explicit-`vectorized` output identity;
- scope pins proving only the two complete DINO NEMO cards select the literal
  path.

## Climate gate

Climate arms remain unauthorized until row 32 is VERIFIED. The frozen southern
day-90 MLD RMS prediction remains baseline `22.479491 m`, CONFIRM
`<=11.2397455 m`, REFUTE `>=20.2775 m`, with the previously registered
legacy-baseline, acceptance-floor, 5x pass-tally, and southern-density gates.
