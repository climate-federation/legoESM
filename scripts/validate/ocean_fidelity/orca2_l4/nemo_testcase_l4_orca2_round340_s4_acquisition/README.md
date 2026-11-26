# Round 340 S4 acquisition

`run.sh` is operator-only. Its default is a read-only preflight; `--run`
reuses the admitted observer-free K0 build and creates one instrumented build
from exactly that recipe. It runs independent two-rank ten-step twins and
admits every S0-F, outer surface, ice-if internal, and state-ladder frame. It
never changes a shipped NEMO deck.

Every failing run is moved to `round340/s4_acquisition/refused/` while its
build, logs, sources, records, and manifest remain intact.  Before retrying,
move the retained target build and any admitted in-place run to unique names
under that same `refused/` directory; then invoke `--run` again. Never delete
retained evidence.

If arm A completed but its wrapper refused it, point `R340_EXISTING_RUN_A` at
that retained directory and `R340_EXISTING_RUN_B` at the absent canonical arm
B, then invoke `run.sh --admit-existing`. A valid arm A is moved into its
canonical location. The next `run.sh --run` verifies arm A, reuses the retained
instrumented build, and runs arm B only.

Preflight derives the reduced writer extent from the admitted `sbc_oce`
`A2D(0)` allocation and the deck's `nn_hls=2`, links all seven patched bounds
objects, executes every one through a patched-module probe, and runs surface
kids 0--11 plus ice-interface kids 0--3. It also
runs the ten-step real CPU card and the one-step real reverse/JVP check in
separate Python processes so their XLA compilations do not accumulate.

To re-admit retained twins, set `R340_EXISTING_RUN_A` and
`R340_EXISTING_RUN_B` to either refused or in-place run directories and invoke
`run.sh --admit-existing`. Admission verifies the manifest's producing commit
and staged binary, relative-path digests, a clean stdout `STOP 0`, both logs'
absence of error markers, the complete
decoded record inventory, unchanged parent restart bytes, and byte-identical
twin record digests. The run remains under the failure trap until sources,
manifest, restart comparison, and single-run admission have all completed.
An admission failure leaves an already-retained directory in place, and any
`E R R O R` line is quoted with its source log before refusal. Every unexpected
shell failure also emits a `REFUSE` line naming its active step.
