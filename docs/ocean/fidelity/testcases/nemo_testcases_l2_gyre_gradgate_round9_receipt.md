# Certified GYRE reverse-mode gradient gate — round 9 receipt

Date: 2026-10-24

Status: **LANDED; GPU ACQUISITION_NEEDED**. The approved local AD boundary is
primal-identical on the certified CPU card, its one-step reverse derivative
passes the unchanged forward and finite-difference bars, its doubled-VJP plant
fires, and the production trajectory is unchanged. The requested GPU 1
multi-day agreement, device-memory, and warm-time rows are not claimed here;
the committed operator entrypoint acquires them fail closed.

## Frozen scope and implementation

The measured card remains only `GYRE-zco`, from its certified day-0 state, at
14400 s per step. CPU agreement uses fp64/libm. GPU acquisition explicitly uses
fp64/native and labels every result **GPU, XLA transcendentals — not the
certified bit-exact policy**. The four-field loss, perturbation, random
direction, central-FD step, `1e-9` reverse/forward bar, and `0.05` reverse/FD bar
are unchanged.

The local boundary takes a separately evaluated production pytree and a replay
pytree. Its primal selects production and its custom JVP selects only the replay
tangent. The full forcing plus production `_step_jitted` call is inside the
shared boundary. Input/output optimization barriers isolate the production arm
from XLA commoning with the replay graph; these barriers are AD compilation
boundaries, not physics, configuration, clipping, a guard, or `stop_gradient`.

## Preregistered predictions and verdicts

| prediction | result | verdict |
|---|---|---|
| P1: hostile one-ULP synthetic preserves eager/JIT/JVP/VJP primal and selects only replay derivative | production-arm derivative `3`, replay derivative `7`; all selected primals preserve the production bits; order-2 forward/reverse gradient check passes | CONFIRMED |
| P2: certified one-step selected VJP/JVP primals are bit-identical and derivative bars pass | final row below | CONFIRMED |
| P2 plant: doubled pullback flips the certified row without changing selected primal bits | reverse doubled; both boundary byte checks remained true | CONFIRMED |
| P2 trajectory: certified ten-step forward remains byte-identical | normalized pre/post JSON identical; residual NPZ SHA pair identical | CONFIRMED |
| P3/P4: GPU 1 one-day, 10-day, 60-day agreement and three-schedule resource rows | operator has not run the committed script | UNMEASURED |

The first implementation prediction was **REFUTED and retained**. At commit
`a747cf1a7`, the custom rule selected the production arm but an XLA
reverse-lowering optimization interaction made the VJP boundary differ on 18
leaves while the JVP boundary was bit-identical. Its derivative evidence was
already on bar, but the binding row correctly became UNMEASURED. The
discriminator added only optimization barriers around the separately executed
production arm, preventing cross-boundary optimization. At commit
`a7318c2bb`, both selected boundary outputs became bit-identical while the raw
VJP/JVP replays retained their expected 18/7-leaf last-bit differences.

## CPU agreement numbers

| row | reverse | forward | central FD | rev/fwd relative | rev/FD relative | boundary bits | status |
|---|---:|---:|---:|---:|---:|---|---|
| one step, final | `-3.884979632703884e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `1.090138208349858e-15` | `5.763743536804898e-07` | VJP/JVP true | CONFIRMED |
| one step, doubled-VJP plant | `-7.769959265407769e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `0.5000000000000006` | `0.5000002881871768` | VJP/JVP true | REFUTED |

For the final row, `sum(abs(g*d))` is
`1.8130019301617077e-04`, `norm(g)*norm(d)` is
`4.348526797535058e-04`, the absolute reverse/forward discrepancy is
`4.235164736271502e-21`, and the cancellation-scale discrepancy is
`2.3359957128637767e-17`. The binding JSON SHA-256 is
`fe77fe427fd28850b01fbd56b3c23c9596b7f0a1db992fbfadba6948e03b012c`;
the plant JSON SHA-256 is
`3ec1339f2dd99098bca4263897ccee4baf4bc4123664c1f0b48d6d0148eae350`.

The six-step certified CPU row remains deliberately skipped: the operator's
Round 8 protocol stopped multi-step CPU compilation after XLA:CPU reported
`LLVM compilation error: Cannot allocate memory`. No CPU multi-step result is
promoted. The inherited pre-boundary GPU six-step diagnostic remains REFUTED
(`reverse -8.185163817998039e-08`, `forward -8.18516302490966e-08`,
`FD -8.13448195885732e-08`, rev/fwd relative `9.689340321095861e-08`), and is
not substituted for the required new row.

## Forward identity

The incoming-tip and post-boundary ten-step trajectory gates both retain the
same inherited NEMO fidelity DEBT, the same 10 ordinary and 10 barotropic-state
rows, identical normalized scientific JSON, and residual artifact SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
Only the receipt-local artifact path and worktree commit stamp differ. Thus the
certified GYRE trajectory is byte-identical across the AD-only change.

## GPU acquisition and training deliverable

Run from the final clean clone root, after creating the evidence directory:

```bash
mkdir -p /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round9/operator_rows
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_round9_run.sh \
  /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round9/operator_rows
```

The runner pins physical GPU 1, fp64/native, disables preallocation, refuses
the platform allocator, and acquires agreement at 1, 6, 60, and 360 steps. It
then measures 6, 60, and 360 steps under `none`, `uniform`, and `binomial`.
Every resource row records the scheduled reverse and binds it to the matching
immutable per-step forward/FD row under the unchanged bars; finite-but-wrong
training gradients cannot be advertised as resource successes. It records JAX
allocator and sampled `nvidia-smi` peaks, blocked warm reverse time, and runs
ordering, time-scaling, memory-scaling, and deliberately broken controls. It
continues after classified failures and finishes with a named REFUSE/nonzero
exit if any required row is not CONFIRMED.

Until that command runs, the new 6/60/360-step reverse/forward/FD values, peak
device memory, warm reverse time, and 60-day cost are **UNMEASURED**. No
projection is substituted for them.

## Tests, evidence, and review

- Direct instrument controls: **49 passed**, including the hostile transformed
  primal, order-2 custom-JVP gradients, boundary mismatch plant, scheduled
  finite-but-wrong gradient refusal, final-status GPU vacuity, memory-scaling
  plant, runner continuation, and operator-entrypoint dispatch.
- Exact clean three-file CPU battery at executable commit `34bc5edbc`:
  **52 passed, 1 skipped in 1977.92 s**. The sole skip is the named six-step
  XLA:CPU compiler limit. Log SHA-256:
  `179387ad172a9858eace3a1698dccdba98d381ffd9e7411f952960d4c41a8cfe`.
- The initial full battery's three citation failures are retained in
  `three_file_battery.log`; they were shifted source anchors, not science
  failures. The executable citation map and Round 7/8 references were updated,
  after which all 17 citation tests and the final full battery passed.
- Two independent adversarial reviews returned SHIP after their findings were
  dispositioned. One forced the boundary to enclose state-dependent forcing
  and use genuinely distinct derivative arms; the other forced resource
  gradients to bind to forward/FD agreement and made GPU vacuity and memory
  scaling executable.
- Evidence lives under
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round9/`.

## OPEN

- **ACQUISITION_NEEDED:** run the committed Round 9 GPU entrypoint above on
  physical GPU 1; ingest all agreement, memory, and warm-time JSONs.
- The 60-step and 360-step gradients are admissible for online closure
  training only if their unchanged reverse/forward and reverse/FD bars pass.
- The SWG card retains its own closed gradient lane; this receipt makes no new
  SWG claim.

## Source citations

The certified card dispatch is `nemo_testcase_recipe.py:3351`; its production
step entry is `ocean_model_latlon_cgrid.py:13219`. The two-pytree selector is
`nemo_testcase_l2_gyre_gradient_gate.py:167`, the shared full-step observed
boundary begins at `nemo_testcase_l2_gyre_gradient_gate.py:199`, and the
production-arm compiler isolation begins at
`nemo_testcase_l2_gyre_gradient_gate.py:217`. Card construction and boundary
wiring are asserted at `nemo_testcase_l2_gyre_gradient_gate.py:231`; explicit
per-step VJP/JVP composition is
`nemo_testcase_l2_gyre_gradient_gate.py:578`; resource acquisition begins at
`nemo_testcase_l2_gyre_gradient_gate.py:1192`.
