# Round 340 S4 acquisition

`run.sh` is operator-only. Its default is a read-only preflight; `--run`
reuses the admitted observer-free K0 build and creates one instrumented build
from exactly that recipe. It runs independent two-rank ten-step twins and
admits every S0-F, outer surface, ice-if internal, and state-ladder frame. It
never changes a shipped NEMO deck.

After a refusal, move each retained build/run to a unique name under
`round340/refused/` and invoke `--run` again. Never delete retained evidence.
`--admit-existing` verifies the retained run's producer commit, staged binary,
relative-path manifest, exact inventory, declared extents, and finite payloads.
