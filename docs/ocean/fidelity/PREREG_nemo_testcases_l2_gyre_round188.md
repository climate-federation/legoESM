# Preregistration — round 188, admit and walk developed shortwave

Committed before rerunning the corrected reader on the acquired Round-186
record or scoring any shortwave statement.  Round 187 reconciled the compiled
no-halo `qsr` extent and the step-1080 stage-3 slot rotation, but its frozen
falsifier stopped before admission.  Round 188 resumes exactly there.  Evidence
lives under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round188/`.

The acquired file is the legacy positional stream written by the single
compiled `WRITE` in the Round-186 build, not the newer named-array container.
The reader must consume its source-enumerated fields to EOF, preserve the
32-by-22 `qsr` extent, and refuse any missing, extra, non-finite, or reordered
payload.  No NEMO rebuild or run is authorized.

## Frozen predictions and falsifiers

1. The corrected header `(1,1080,3,2,1,36,26,31,64)` admits the existing
   record.  Its producer stamp matches, actual and replay increments are BIT,
   and the actual-increment one-ULP plant prints `STATUS PLANT-FIRED` and exits
   nonzero.  Any failure stops the scientific walk.
2. The record remains passive: its already-recorded step-180 and step-1080
   restart digests equal the uninstrumented Round-132/year record.  The record
   itself is not modified.
3. Walk the executed two-band statements in compiled order: coefficients,
   surface attenuation, live depth/arguments, exponentials and attenuation,
   live `ze3t`, absorbed flux/division, associated RHS update, and the
   near-surface/deep transition.  Each exposed row reports unequal cells and
   maximum absolute difference.
4. Isolated eager, isolated JIT, production eager, and production JIT are
   separate labels.  A row not exposed through the production step is
   `UNMEASURED`, never promoted from an isolated closure.
5. Frozen attribution prediction carried from Round 187: all prior operands
   are BIT and the first non-bit production-JIT statement is the first live
   attenuation expression, because the whole-column JAX construction changes
   compiled association relative to NEMO's level-ordered loop.  An earlier
   non-bit operand or a BIT attenuation row refutes this prediction; the first
   measured row wins.
6. A statement is owned only when it remains non-bit given NEMO's recorded
   operands through the production closure.  Exactness only under substituted
   NEMO operands identifies inherited error.
7. No physics lands unless a one-variable NEMO-source-exact production-JIT
   candidate passes the complete Decision-43/45/55/59 ladder, month, year,
   executing-card census, DINO, generic-card, tank, ORCA2-with-spec, plant,
   citation, and dual-review gates.  Otherwise this round is HELD with the
   first named row or STOPPED_FOR_RECORD.

The final diff receives a separate read-only Codex review.  A `DO NOT SHIP`
verdict blocks a landing.  No configuration choice is authorized or expected.
