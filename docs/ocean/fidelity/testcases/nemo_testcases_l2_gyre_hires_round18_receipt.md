# GYRE high-resolution round 18 receipt

Date: 2026-10-19. Status: **HELD FOR INDEPENDENT REVIEW -- COMPLETED ENE
STENCIL EXONERATED; UPSTREAM COUPLED INPUT DEBT**. This round changes no
physical or configurable path. The measurement and controls are committed;
the scientific verdict is not promoted to LANDED without the two required
reviews.

## Finding -> disposition -> evidence

The preregistered q-only discriminator **REFUTES q as a sufficient owner**.
Against the scalar-math NEMO stage-3 record its residual is
`-4.3038944112607703e-16 m s-2`, while the scalar/vector NEMO envelope is
`4.1439741816112516e-23 m s-2`. The unchanged HPG and advection rows are
bit-exact. The admitted artifact is
`gyrehr_rounds/round17/stage3_vorticity_discriminator.json`, SHA-256
`d5034e047495d6b8ac5260c9df9cc9abc0b9b4fdc599b33df604e9cdcb34ea2c`.

The source-ordered walk then compares the statement NEMO executes -- one
float64 addition of the candidate vorticity term to scalar NEMO's recorded
pre-`dyn_vor` accumulator -- against scalar NEMO's recorded post-`dyn_vor`
accumulator. This avoids subtracting two accumulated arrays and calling that
rounded subtraction an independently observed term. The result is:

| legoESM stage-3 substitution | unequal registered surface-V words after the NEMO addition | maximum absolute residual (m s-2) | signed western-boundary mean (m s-2) |
|---|---:|---:|---:|
| live q, live `F_u` | 1,896 | `5.176321001064237e-13` | `-8.647283483104465e-16` |
| scalar-NEMO q only | 1,896 | `5.019693021706198e-13` | `-4.3038944097054443e-16` |
| scalar-NEMO `F_u` only | 1,896 | `1.402273615150676e-13` | `-4.3433892419414325e-16` |
| scalar-NEMO q + `F_u` | **0** | **0** | **0** |
| scalar-NEMO `F_v` negative control | 1,896 | `5.176321001064237e-13` | `-8.647283483104465e-16` |

Only `{q, F_u}` is exact. `F_v` leaves the entire V result bitwise unchanged,
as NEMO's V stencil requires. Every arm leaves the captured HPG and advection
rows bitwise unchanged. The scalar/vector NEMO operands themselves differ by
73 active surface q words (maximum `1.6940658945086007e-20`) and 5,099
surface `F_u` words (maximum `4.5488377509173006e-9 m3 s-1`); these are
context, not a claim that either difference alone owns legoESM's residual.

**Disposition:** the completed energy-conserving V vorticity/Coriolis stencil
is not the physics owner. Given the two NEMO formed operands, legoESM
reproduces all 1,896 registered accumulator words exactly. Neither operand is
sufficient alone, so the smallest proving set is a genuine coupled
composition boundary upstream of the completed stencil, not a localized
single-line transcription defect. Forcing either input to NEMO's value in
production would hide the still-unmeasured formation owner; no shared-path
fix is warranted in this round.

The main measurement is
`gyrehr_rounds/round18/stage3_vorticity_transport_walk.json`, SHA-256
`4bd8ea66ccbb076c243f7c0671575eb4f514f432bc4d0321bb3ccc0244045e5a`,
from clean commit `fd961f73844e3b38a11d6616656f9fec2400b829`. Both NEMO providers
are the operator-acquired commit `979d024d7` records; their admission verifies
127 manifest members, the requested arithmetic flavour, the common restart,
and same-executable passivity.

## Controls and reversion

The q plant changes native operand `[j=80,i=25,k=0]` and moves one registered
V word by a signed mean `-1.9408618718976508e-3 m s-2`. Its artifact is
`stage3_vorticity_q_plant.json`, SHA-256
`f9a02964589f82e16f5ad9ba7948a4d49198547c477a0ff1379bd5095c26c4c9`.
The `F_u` plant changes `[j=76,i=1,k=0]`, moves two registered V words, and
moves the signed mean by `-1.8182695165194448e-13 m s-2`; its artifact is
`stage3_vorticity_Fu_plant.json`, SHA-256
`e3c1ffa41baa5ddbaaec4af3e266387a8f7b81570ca9928f97fc90efaee1d1d3`.
Both are clean-commit `8f906abbc` runs. The initial first-wet-word q plant
correctly refused because that chosen coefficient was zero; the final plant
selector derives a consumed nonzero word from the executed V stencil instead
of weakening the control.

The diagnostic seam is private and default-`None`. Identical q and transports
are array-exact; a formed `F_u` override moves only V, a formed `F_v` override
moves only U, wrong shapes refuse, and public configuration cannot select the
seam. Reverting the `F_u` assignment to ignore its supplied operand makes
`test_ene_metric_transport_override_is_directional_and_active` fail at its
required V movement. Restoring it passes. Thus the zero-word row is not a
self-comparison or inert substitution.

## Inherited pins / OPEN

Inherited unchanged: the current shipped one-degree digest
`56c5171d43504b54` (the post-main-merge pin), its no-worsening ceiling, 31
levels, WS-RK3, split-explicit free surface, TKE/EVD, Redi tracers, the
`r4_smag` deck, 3,600-s step, literal 360-day clock, scalar-libm certified CPU
policy, vector-math climate reference, common day-180 restart, round-16
western-boundary wet-span reduction, and four-times twin screen. No deck,
physics, climate run, NEMO source, or operator-owned evidence was changed.

OPEN, in order:

1. At the same common state, record legoESM's live stage-3 q and `F_u`, compare
   them wordwise with scalar NEMO, then walk the source factors of the first
   differing formed quantity one at a time (curl/Coriolis/live-F thickness for
   q; metric, live U thickness, and Kmm velocity for `F_u`).
2. Only a single-factor substitution that removes the joint residual can name
   a transcription owner and authorize a shared-path edit.
3. Run both required independent reviews when their services are available.
4. Post the result to issue #1455 when network access is available.

No new acquisition is required to accept this round's negative localization;
the next walk begins with legoESM's already-computable live operands. The
operator should not launch another long run.

Choice audit: the user requested the registered common-state substitution,
an exact source walk, a fix only after localization, affected-card gates only
after such a fix, and a confirming acquisition only for a landed fix. The
measurement localizes a two-input boundary but not a single owner, so no
physical choice, special case, or acquisition was added.

## Tests and review

Focused reader, source-accumulator, seam, directionality, refusal, and plant
controls pass: `25 passed`. The deliberately reverted `F_u` seam fails the
directionality test, and the restored seam passes. `git diff --check` passes.

The high-resolution split passes `120 passed, 2 deselected`. The shared
card/operator split initially reported `176 passed, 3 deselected` plus one
failure: the default receipt still cited pre-seam legoESM line numbers. The
map audit had correctly moved those keys, so the default receipt was
re-anchored to the same statements and its complete gate suite then passed
`17 passed`. The final unique aggregate is:

`FULL_LANE_BATTERY_SPLIT_PASS: 297 passed, 5 deselected across two CPU/x64 groups`

The round-specific citation gate passes all four compiled-source citations
with zero unmapped citations, failures, or stale map entries; all nine
self-tests fire. Shifting the `vor_ene` dispatch citation by two lines returns
`SYMBOL-NOT-AT-LINE` and exits 1.

Review status is **UNREVIEWED**. The read-only Codex reviewer cannot
initialize its local app server because that service attempts a write on the
read-only host filesystem. The read-only Claude reviewer starts but terminates
with `ENOTFOUND` reaching its API. No GLM binary or endpoint is available.
These are recorded service failures, not review verdicts or waivers. No
scientific owner or physical change is landed.

## Compiled-source citations

The acquired writer brackets `dyn_vor` after HPG and before momentum advection
at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:404-430`.
The selected configuration dispatches total vorticity to `vor_ene` at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:235`.
That executed routine forms Coriolis plus relative vorticity and divides by
the live F-point thickness at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:552-572`.
It then forms `F_u=(e2u*e3u)*u` and `F_v=(e1v*e3v)*v`, and its V statement
reads q and `F_u` but not `F_v`, at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:576-591`.
