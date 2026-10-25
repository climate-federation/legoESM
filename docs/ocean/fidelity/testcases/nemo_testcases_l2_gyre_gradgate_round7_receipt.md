# Certified GYRE reverse-mode gradient gate — round 7 receipt

Date: 2026-10-12

Status: **HELD**. The approved explicit per-production-step VJP/JVP instrument
is landed, the certified ten-step trajectory remains byte-identical, and the
current one-step CPU fp64/libm row is CONFIRMED. The mandatory six-step CPU row
is UNMEASURED: an isolated attempt compiled the reusable reverse production
step without the former LLVM section-memory abort, but its forward transform
did not produce JSON within the bounded 65-minute attempt. Consequently the
six/sixty-step resource rows, memory projections, and live certified-card plant
were not run after the CPU budget was exhausted. No physics, card setting,
timestep, objective, tolerance, stabilizer, clipping, guard, or
`stop_gradient` changed.

## Provenance and frozen claims

The incoming lane tip was `2b83ae6a986e5a67acc8eb2d4f16a36f59ed7989`.
Round 7 was preregistered at clean commit `7a1b7a593`; the external copy is
`gradgate_rounds/round7/preregistration.md`, SHA-256
`43d7289851c7fa9ef547e9b2122f7d7d28d4fcb5bc223cdf3967b845ec657c00`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round7/`.

The instrument was ported from the reviewed SWG gradient-gate commits named in
the preregistration, excluding every SWG card change and measurement. Local
port and composition commits through the measured one-step producer are
`b9d059ca4`, `115000a8a`, `ef437a35c`, `d5e2a0759`, and `46795cb96`.
Commit `b09dce0e9` adds the direct explicit-composition plant test after that
measurement without changing the producer.

The only card is the fail-closed `GYRE-zco` dispatch at
`nemo_testcase_recipe.py:3351`. The harness builds and verifies that card at
`nemo_testcase_l2_gyre_gradient_gate.py:231`. Every modeled step uses the
production JIT at `ocean_model_latlon_cgrid.py:13219`; the direct step,
Decision-49 per-step loop, and schedule entry points are
`nemo_testcase_l2_gyre_gradient_gate.py:280,301,309`.

The registered bars remain reverse/forward relative error `1e-9` and
reverse/central-FD relative error `0.05`; FD step, perturbation, direction, and
four-field normalized objective are unchanged. CPU uses the certified
fp64/libm policy. GPU carry-forward rows use fp64/native and are labeled
**GPU, XLA transcendentals — not the certified bit-exact policy**.

## Imported implementation and executable controls

The agreement path materializes the direct production states, then applies one
VJP per step in reverse order and one JVP per step in forward order. Each
transform is evaluated at the stored direct-production boundary. Transformed
replays are recorded but are never substituted for later production primals;
the implementation begins at
`nemo_testcase_l2_gyre_gradient_gate.py:578`.

This distinction is measured, not hidden. On the one-step card, VJP replay
differs on 18 leaves (largest `1.4210854715202004e-14` in salinity) and JVP
replay differs on 7 leaves (same largest difference). The direct production
zero/base/FD trajectories are nevertheless byte-identical, and all derivatives
are evaluated at those direct states. `step_vjp_replay_used_as_primal` and
`step_jvp_replay_used_as_primal` are both false in the JSON.

The deliberately doubled VJP is wired into this explicit composition. Its
direct synthetic test keeps forward mode unchanged, doubles reverse mode, and
must classify REFUTED. Resource rows build
`jax.jit(jax.value_and_grad(scalar_line))`, discard the cold compile-and-run,
and time/sample the second blocked execution at
`nemo_testcase_l2_gyre_gradient_gate.py:1192`. Rows with historical timing
stamps are executable RETRACTED outcomes, and missing/flat/non-finite sampler
data are VACUOUS at `nemo_testcase_l2_gyre_gradient_gate.py:1373`. The
constant-time scaling plant and GPU schedule-order plant are direct tests.

## Production trajectory identity

The ten-step trajectory gate at port commit `b9d059ca4` is scientifically
identical to the post-CEN2 Round 4 control. The only JSON differences are
artifact path/provenance fields (`artifacts`,
`oracle_relative_residual_artifact`, and `worktree`); every scientific key and
row is identical. Both residual NPZs have SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The Round 7 JSON SHA-256 is
`18d2e70fec9295ff73a4fde6205a186415f0e6f3e778e26c297503ce49c66832`.
No package source changed after this control; subsequent changes are confined
to the measurement script and its tests.

Preregistered R7-P1 is **CONFIRMED** for the unchanged production trajectory
and direct agreement primals. The transformed replay diagnostics are not
claimed byte-identical and do not replace those primals.

## CPU agreement and cancellation scale

| CPU fp64/libm steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | `sum(abs(g*d))` | scale-aware | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.884979632703881e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `2.180276416699718e-16` | `5.763743528083797e-07` | `1.8130019301617077e-04` | `4.671991425727554e-18` | **CONFIRMED** |
| 6 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **HELD** |
| 60 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **OPEN** |

The one-step absolute reverse/forward difference is
`8.470329472543003e-22`; `norm(g)*norm(d)` is
`4.348526797535058e-04`. Its strict JSON is from clean producer
`46795cb9668a28eeffbdb91a050cfa31965ecbcd`, verified clean again after the
measurement, and has SHA-256
`9043c1b76f1dddcea933c54164347e23d0575c640616c4691c69fefc5628ed36`.
R7-P2 is CONFIRMED for one step and UNMEASURED, not refuted, for six and sixty
steps.

## Six-step compiler attempts and executable retractions

Two early instrument attempts refused before a derivative verdict because
they incorrectly required transformed VJP replay outputs themselves to be
byte-identical. Their JSONs are retained as executable retractions:

1. `cpu_agreement_6step_explicit.json`, SHA-256
   `7ffc5d89becc354c68414a0d8e8c0430618455405a304385cfeed90dd46736f4`.
2. `cpu_agreement_6step_explicit_boundary.json`, SHA-256
   `ce3e6cff739b27141060cd644ff8a9e7d5c9b4d36ab95ce9ac2d6ce6b01f304d`.

The third attempt separated replay diagnostics from production primals. XLA
completed the reusable reverse-step compile in `10m7.827492201s`; unlike the
rounds 2/5/6 whole-window programs, its log contains no `Unable to allocate
section memory` abort. The subsequent forward transform did not finish inside
the 65-minute bounded attempt. It was interrupted with exit 130 and wrote no
JSON. Therefore it is UNMEASURED. The log SHA-256 is
`18d757c668b0a3b0a76be7b6ab5e488f5ff8d5f53150c324f262e2165b397e8c`;
the retained refusal note SHA-256 is
`b21e7a94cb0edf0c6568e5a107742f0717122d0876d53d0c736b27eaa3a6cb04`.

That attempt also exposed that its forward tangent chained transformed replay
primals. Commit `46795cb96` retracts that design in executable code: both
directions now use each stored production boundary. The corrected one-step row
above confirms the change. A corrected six-step attempt was not started after
the round's CPU use exceeded the requested approximately two-hour cap.

## CPU memory, wall time, and projections

No admissible Round 7 six-step agreement row exists, so no Round 7 six- or
sixty-step resource process was started. Peak host memory, warmed reverse wall
time, and the 60-step/360-step projections are all **UNMEASURED** for all three
schedules. Round 6's one-step RSS values are historical context only and are
not relabeled as Round 7 measurements.

| CPU fp64/libm schedule | 6-step peak/time | 60-step projection | 360-step projection |
|---|---|---|---|
| `none` | UNMEASURED | UNMEASURED | UNMEASURED |
| `uniform` | UNMEASURED | UNMEASURED | UNMEASURED |
| `binomial` | UNMEASURED | UNMEASURED | UNMEASURED |

R7-P3 is **UNMEASURED**. The compiled-timer classifier, historical-row
retraction, scaling classifier, and constant-time plant are implemented and
directly tested, but no prose claim promotes them to a measured resource row.

## GPU carry-forward and operator runner

The only GPU evidence remains the operator's Round 5 fp64/native pair:

| GPU 1 fp64/native steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.884979632702670e-06` | `-3.884979632702686e-06` | `-3.884977393500650e-06` | `4.142525191730739e-15` | `5.763741977909062e-07` | **CONFIRMED** |
| 6 | `-8.185163817998039e-08` | `-8.185163024909660e-08` | `-8.134481958857320e-08` | `9.689340321095861e-08` | `0.006191917506803739` | **REFUTED** (`>1e-9`) |

Their SHA-256 values are respectively
`e1ad50421cc6066f1dc6bdf3d90244cea29b0583d1222431a2d3485a4544e1e6`
and `fdfa79b61ebaba3a248c0354dc6d654074adc83fa15c189023f0ab54fb3d9276`.
They predate the cancellation-scale schema, so the hypothesis that the GPU
off-bar relative error is scale-blind remains untested. The bar is unchanged.

The committed Round 7 GPU runner selects physical GPU 1, native XLA
transcendentals, disables preallocation, refuses the platform allocator,
continues resource rows after an off-bar agreement row, validates every JSON's
policy/provenance, and runs its ordering plant. This sandbox cannot open a GPU;
an operator rerun is OPEN. No GPU memory or projection number is fabricated.

## Tests and gates

- Direct gradient-gate controls: **37 passed in 10.86 s**. They include the
  explicit per-step VJP/JVP equality control, its doubled-VJP plant, executable
  historical timing retraction, compiled timer no-retrace control,
  constant-time scaling plant, GPU ordering plant, memory sampler failures,
  policy dispatch, projections, provenance, and overwrite refusal.
- Certified recipe tests: **25 passed in 748.34 s**. Standalone log SHA-256:
  `454a49995beb1277344568f29c5d155fffcb274fb8d703e51c981c82f724012a`.
- Exact three-file CPU push battery: **35 passed, 1 skipped in 3890.68 s**.
  The nine legacy variants and isolated current-commit one-step GYRE row pass;
  the six-step row is the named bounded-attempt skip. The parent process exits
  normally, so no abort is hidden. Log SHA-256:
  `a64e5c5135d2e4ff95be1dbed2aa72451295e1dc427b00afe350018fd1f58b5f`.
- Receipt-specific citation gate: **PASS**, seven citations, zero unmapped
  citations, zero failures, and every planted self-test fired. JSON SHA-256:
  `df6c5de1a4acb7c27e400445fa91567531f879b9381566561d4c6acfec2f6940`.
- Independent adversarial reviews: **UNAVAILABLE, not counted**. The Codex CLI
  refused because its in-process app-server state is read-only; the Claude CLI
  returned `ENOTFOUND` for its API server. Their retained refusal files contain
  no verdict. Because this round is HELD rather than shipped, no author review
  is substituted for either required independent review.

## OPEN, retractions, and decision

OPEN:

1. Run the corrected six-step CPU agreement row in a fresh process with a
   larger explicitly approved CPU budget. A JSON is required before any
   agreement claim.
2. Only after item 1, run six/sixty-step `none`, `uniform`, and `binomial`
   resource rows, their scaling classifiers/plants, and compute the registered
   60/360-step projections from valid measurements.
3. Run
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`
   from an operator shell on physical GPU 1. Use its cancellation scale for the
   six-step off-bar decision; do not change the `1e-9` bar in this lane.
4. Run the live certified-card broken-VJP plant at the final producer. The
   direct explicit-composition plant is binding, but it is not substituted for
   the missing expensive card row.
5. Add the SWG card's own gradient gate when that certified card exists.
6. Obtain two independent adversarial review verdicts after the corrected
   six-step/resource artifacts exist; the sandbox review refusals do not
   satisfy the campaign's review rule.

RETRACTED: transformed VJP/JVP replay outputs are not bit-identical production
trajectories and are not allowed to become later-step primals. The two strict
early refusals and the chained-forward attempt remain in evidence; current code
records the replay mismatch while evaluating derivatives at stored production
boundaries.

DECISION_NEEDED: approve a fresh corrected six-step CPU run with a budget above
65 minutes; **pick: yes**, then run resources only if that agreement JSON is
CONFIRMED. There is not enough evidence to change either numerical bar.

## Choice audit

- ASKED: port the reviewed per-step composition and resource instruments,
  preserve GYRE-only dispatch, prove trajectory identity, measure one/six/sixty
  CPU rows, add cancellation scale, carry the operator GPU result, bind timing
  classifiers/plants, run gates, write this receipt, and commit.
- UNASKED: none. No physics, card, precision, numerical bar, checkpoint
  schedule, or production trajectory requirement changed. The agent stopped an
  over-budget compile and reports the resulting rows and projections as
  UNMEASURED.
