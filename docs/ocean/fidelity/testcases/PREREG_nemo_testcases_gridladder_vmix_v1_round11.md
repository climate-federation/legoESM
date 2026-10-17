# PREREG — vertical-mixing lane V1 round 11 (vmix_r11)

Frozen before any round-11 measurement. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Decisions 129 = 1 and 130 = 1 in force. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round11/`.

Starting state (round 10, landed `066f5c712`): kt 1-3 exact under both labels; kt 4-5 at the bar; first over-bar
row in both labels = kt-6 T, 2 cells, 1.21e-15 normalised at (34,120). Reviewer MEASURED: one step from NEMO's
kt-5 state is exact at kt 6; one step from NEMO's kt-4 state misses T/S at (34,120).

Instrument: the committed `v1_substitution_probe.py`, extended so the CLOSURE kind can start from NEMO's own
kt-(k-1) restart (entry source `nemo`) instead of the card's chain state. Identity controls as before.

- **P1 (reproduce the reviewer).** Entry from NEMO's kt-4 restart, step 5: T or S unequal in column (34,120).
  Falsifier: T and S exact there.
- **P2 (FMA class).** The same one step with FMA contraction disabled for the whole program
  (`XLA_FLAGS=--xla_cpu_max_isa=AVX`) is exact in column (34,120). Falsifier: any unequal T/S cell there
  (then the owner is not an FMA association, and the walk compares statement values offline).
- **P3 (avt hypothesis).** From NEMO's kt-4 entry, NEMO's kt-5 `avt_k` swapped in where the solves read it ->
  kt-5 T and S exact in column (34,120). PLAUSIBLE; falsifier: an unequal T/S cell there (then `tra_zdf` or
  the stretch/ratio terms are walked instead).
- **P4.** Whatever lands leaves kt 1-3 exact under both labels, and the flag-off one-step `make_jaxpr` of full
  ORCA2 and DINO is identical to base `d31991655`.
- **P5.** V1 certifies this round: PLAUSIBLE only. Falsifier: any DEBT row kt 1..10 under either label.
