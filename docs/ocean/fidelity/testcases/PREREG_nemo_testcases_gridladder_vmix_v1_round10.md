# PREREG — vertical-mixing lane V1 round 10 (vmix_r10)

Frozen before any round-10 measurement. Record: `gridladder_vmix_rounds/round1/oracle_v1_r1` (admitted round 3).
Decisions 129 = 1 and 130 = 1 in force. Codex review: paused (quota). Evidence: `gridladder_vmix_rounds/round10/`.

Starting state (round 9, landed `991cad2ba`): kt 1 bit-for-bit; kt 2 at the bar on every field (residue en 2,
avm_k 1, avt_k 3, dissl 2, T 5, S 4 cells); kt 3 first over the bar = sea level, 189 cells, 1.29e-15 normalised.

Instrument: a COMMITTED probe (`v1_substitution_probe.py`) with two substitution kinds, each arm a fresh model,
with an identity control (the card's own fields substituted) that must be bit-identical to the plain step:
- ENTRY: NEMO's kt-(k-1) restart fields replace the card's own state entering step k;
- CLOSURE: NEMO's kt-k `avm_k`/`avt_k` replace the vertical-mixing closure's output where the solves read it.

- **P1 (reproduce the reviewer).** Entry substitution at kt 3: NEMO's T/S -> kt-3 sea level 0 unequal; NEMO's
  four TKE fields -> 189 unequal; all prognostics -> kt 3 exact. Falsifier: any other count.
- **P2 (avt hypothesis).** Closure substitution at kt 2: NEMO's `avt_k` -> kt-2 T and S 0 unequal at every level.
  Falsifier: any unequal T or S cell (then the T/S owner is not avt, and `tra_zdf`/advection is walked instead).
- **P3.** The 3 unequal kt-2 `avt_k` cells lie in columns holding an unequal kt-2 `en` cell (avt inherits from en
  through `tke_avn`, `zdftke.f90:531`), so the first non-bit statement is upstream of `tke_avn` in `tke_tke`.
  PLAUSIBLE; falsifier: an unequal avt cell whose column's en, mixing length and N2 are all bit-equal to NEMO's.
- **P4.** Whatever lands (if a transcription defect) leaves kt 1 bit-identical under both labels and the flag-off
  one-step `make_jaxpr` of full ORCA2 and DINO identical to base (`e0351d089a027297`, `89418f6a0b2eabae`).
- **P5.** V1 certifies this round: PLAUSIBLE only. Falsifier: any DEBT row kt 1..10 under either label.
