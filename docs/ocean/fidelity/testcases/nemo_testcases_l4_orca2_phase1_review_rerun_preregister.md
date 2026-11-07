# ORCA2 Lane-4 Phase-1 independent-review rerun preregistration

Frozen before the replacement build and execution. Scope is corrective NEMO
oracle instrumentation only; no legoESM code or numerics are authorized.

The accepted scientific oracle definition, keys, copied deck, scalar-math
compiler policy, two-rank `jpni=2,jpnj=1` layout, and ten-step controls remain
exactly those in the Phase-1 receipt. The replacement run must use regular
copies of the accepted 19-file deck, immutable absolute symlinks to the same
40 inputs, and a symlink to one newly built scalar-math executable.

Two config-local WRITE-only implementation changes require regeneration:

1. `icethd.F90:l3rea_dump` derives its declared payload length from its write
   list as `(3 + 3*nlay_i + 2*nlay_s)*npti`. Its payload and all model arrays
   are unchanged. The gate independently mirrors that dimensional expression.
   Its ZDF check likewise mirrors the existing writer expression
   `(13+nlay_s)*npti`, not the resolved literal 18.
2. `eosbn2.F90` will allocate its 13 full-domain diagnostic scratch arrays
   only while the one-shot `l2_dump_armed` flag is true and deallocate them
   immediately after the dump. The EOS calculation and model arrays are
   untouched.

The replacement executable is accepted only if its build succeeds and
`nm -D` contains zero `_ZGV*` symbols. The replacement directory will be
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/instrumented_reviewfix_10step_np2`.
Its launcher must execute `mpirun -np 2 --oversubscribe ./nemo` from the user's
shell, using Bash `time`, and must hash-check the executable, copied deck, and
all linked inputs before launch.

After execution, Phase 1 may re-close only if the current gate passes the
frozen 90-file inventory, every schema/header/EOF/finiteness check, all nine
planted controls, the frozen restart-variable inventory, and full ordinary
output identity against `uninstrumented_10step_np2`. The six restart shards
must again be exact bytes. Any failure stops; no record may be borrowed from
the prior accepted binary.

| item | disposition |
|---|---|
| derived reassociation/ZDF header expressions | ASKED review correction |
| armed-only EOS diagnostic allocation | ASKED review correction |
| replacement scalar-math build and record regeneration | ASKED consequence |
| unchanged scientific deck and two-rank layout | REQUIRED control |
| any legoESM work | FORBIDDEN in this round |
