# The 90-day START-MODE A/B — result (#1455)

Pre-registration: `scripts/validate/ocean_fidelity/dino_1226/PREREG_startmode_ab90.md`
(+ Amendment 1, written after adversarial review and **before any arm was scored**).
Commit: `1af36bcbb`. Seven 90-day arms, one card, one SHA, fp64, ladder `both`
(content hash `9536f62732ab8823…` identical on all seven).

## VERDICT

**Registered rule 1 — INDISTINGUISHABLE FROM ANY RECORDED CONFIG CHANGE.**
The prediction ("endpoint metrics move little") is CONFIRMED in the bounded
sense it was registered in. The start mode's day-90 endpoint cost is **bounded
at 1.81e-2 Sv of circumpolar transport**, which is:

* inside the measured config-to-config envelope (1.011e-2 … 7.042e-2 Sv),
* **5.0x below** the gate's 1x ACC floor (9.1e-2 Sv) and 25x below its 5x threshold,
* inside the mechanism reviewer's independent pre-run forecast of 1e-2 … 7e-2 Sv.

**No systematic effect is established, and none can be at n=1** — the arms are
chaotically decorrelated by day 90. This is a BOUND, not a null result.

**Consequence for the campaign: none.** No recorded gate or verdict number was
Euler-start (audited, 18 of 18 bridged), and even if one had been, the cost is
below the floor the gate scores against. No re-baselining is warranted.

## THE ARMS

| arm | start | perturbation |
|---|---|---|
| B | bridged (new default) | — |
| E | euler (`--legacy-euler-start`) | — |
| Bp / Ep | bridged / euler | `--perturb-seed 1`, 1e-14 relative on T |
| N1 / N2 / N3 | bridged | seeds 2/3/4, `--perturb-eps 1e-5` |

All seven stable, 2880 steps, ~205 s each. Day-0 gate bit-identical to the NEMO
restart on every arm.

## THE NUMBER

`D = |B − E|` at day 90, against the three bars:

| metric | **D** | matched null (N-N) | config envelope | 1x floor | 5x gate |
|---|---|---|---|---|---|
| ACC [Sv] | **1.814e-2** | 5.8e-4 … 2.4e-3 | 1.01e-2 … 7.04e-2 | 9.10e-2 | 4.55e-1 |
| upper contrast | 2.416e-6 | 5.0e-7 … 1.3e-6 | 6.6e-7 … 1.04e-5 | 1.10e-4 | 5.50e-4 |
| deep contrast | 1.366e-7 | 7.6e-8 … 2.8e-7 | 1.2e-7 … 1.91e-6 | 4.50e-5 | 2.25e-4 |
| S-band sigma MAX | 9.94e-7 | 2.3e-6 … 7.9e-6 | 1.2e-6 … 1.67e-5 | 9.50e-5 | 4.75e-4 |
| S-band sigma MEAN | 1.267e-7 | 3.1e-8 … 6.4e-8 | 1.5e-8 … 6.7e-7 | 9.50e-5 | 4.75e-4 |
| channel-band transport [Sv] | 1.977e-3 | 2.9e-4 … 3.9e-3 | (ACC floor, transferred) | 9.10e-2 | 4.55e-1 |

**Gate:** arm B `PASS 5 | FAIL 0` at 5x. Arm E scores **within 5x on all five**
and returns UNCERTIFIED (exit 3) — the start-mode criterion added in the same
commit, working as designed. The registered comparison (same pattern on both
arms, read off E's neutral markers) is **satisfied**.

**Direction (registered as weak):** 3 metrics favour B, 3 favour E. A coin flip,
which is what decorrelation predicts. NULL — no direction claim.

## RETRACTION — the matched-amplitude null is NOT a null

Registered as the secondary bar in Amendment 1; **its premise is refuted by its
own data**, so rule 2 is not read off it.

The three 1e-5 members do not scatter about arm B. They **cluster together and
sit 4.21e-2 … 4.45e-2 Sv away from B, all in the same direction**, while
scattering only 5.8e-4 … 2.4e-3 Sv among themselves — a 18x to 75x ratio. A
zero-mean random kick that produces a common systematic shift is a **rectified**
response, not decorrelation, and this campaign already named the rectifier:
convective adjustment's hard edge, measured at ~300x per step for perturbations
above ~1e-10. At 1e-5 the members are far past that threshold and share it.

So the N-N spread is a **lower bound on the null, twice over**: the kick is
T-only (the start mode also perturbs u by 2.37e-3 relative, and ACC is a
velocity integral) and its members share a rectified common mode. `D` exceeding
it on ACC, upper contrast and sigma-mean therefore does **not** establish a
systematic effect, and the MATERIAL branch is not triggered. The primary
envelope carries the verdict.

## INSTRUMENT CONTROLS, all passed

* **Non-numerical-fix control:** arm B re-run at the post-review HEAD is
  **bit-identical** to its pre-fix run (worst |diff| = 0.0 over 28 float arrays),
  confirming the review fixes are stamping/guard/prose only.
* **Floor reproduction:** the 1e-14 within-arm floor at this commit is 1.79e-5 Sv
  (ACC), consistent with the recorded 4-member value of 1.15e-5 Sv.
* **Envelope reproduction:** the config-to-config band was recomputed
  independently from the four recorded arms — 1.011e-2 / 3.521e-2 / 7.042e-2 Sv
  (min/median/max), matching the reviewer's table digit for digit.
* Ladder content hash, dtype and start mode stamped and identical/expected on
  every arm; the scorer refuses arms lacking either stamp.

## OPEN, and labelled

* **PLAUSIBLE, n=1 pair each:** the Euler arm amplifies a 1e-14 kick **23x more**
  than the bridged arm at day 90 (|E−Ep| = 4.20e-4 Sv against |B−Bp| = 1.79e-5 Sv).
  If real, the Euler start is not only offset but *noisier*. One pair each is not
  a measurement; three members per arm would settle it (~11 min).
* **A true amplitude- and channel-matched null is unbuilt.** It needs a
  perturbation matched in *both* T and u and below the convective rectifier's
  threshold — which may not exist, since matching the amplitude necessarily
  crosses it. Named, not built.
* **The 0.470 / 0.000142 impulse-lag pair remains CARRIED, not reproduced** here;
  its instrument lives on `fidelity/dino-eta-waves`.
* `D` bundles three mechanisms (half-step delay, injected perturbation, skipped
  step-1 reconciliation). It is one *flag*, not one *mechanism*; no part of the
  1.81e-2 Sv is attributed to any single term.
