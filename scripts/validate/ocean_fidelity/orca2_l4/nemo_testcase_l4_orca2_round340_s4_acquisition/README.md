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

To re-admit retained twins, set `R340_EXISTING_RUN_A` and
`R340_EXISTING_RUN_B` to either refused or in-place run directories and invoke
`run.sh --admit-existing`. Admission verifies the manifest's producing commit
and staged binary, relative-path digests, both clean `STOP 0` logs, the complete
decoded record inventory, unchanged parent restart bytes, and byte-identical
twin record digests. The run remains under the failure trap until sources,
manifest, restart comparison, and single-run admission have all completed.
