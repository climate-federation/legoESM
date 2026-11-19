# GYRE high-resolution round 19 receipt

Date: 2026-10-19. Status: **HELD FOR INDEPENDENT REVIEW -- NO
SINGLE-FACTOR TRANSCRIPTION OWNER**. This round changes no physical or
configurable path. It adds a passive write-only exposure of the exact live
stage-3 ENE q and metric transports, then applies the preregistered six-factor
walk at the common developed state.

## Finding -> disposition -> evidence

The passive trace and its independent ordinary step are byte-identical across
the complete state pytree: zero mismatched bytes. Re-forming live q and
`F_u` with legoESM's source-order builders reproduces the newly exposed arrays
exactly over 306,270 and 302,580 words respectively. Replaying those arrays
through the production ENE kernel, including the common diagnostic face mask,
reproduces the captured live surface-V term exactly over 303,780 words. These
three zero-word calibration rows precede every scientific comparison.

Against scalar-math NEMO, the live formed operands differ substantially:

| formed operand | compared words | unequal words | maximum absolute difference |
|---|---:|---:|---:|
| q | 306,270 | 300,029 | `2.5167424076469235e-5` |
| `F_u` | 302,580 | 286,719 | `1.1682881327578798e-1 m3 s-1` |

The independent factor providers show where a one-factor proof is possible.
NEMO `ff_f` and `e2u` are already bit-exact (0 unequal words). Kmm U and V
velocity differ in 286,719 and 284,741 words, with maxima
`2.711170197025603e-8` and `2.4075361593212108e-8 m s-1`. Live F and U
thickness differ in 282,000 and 285,550 words, with maxima
`1.0790017768158577e-9` and `1.083662937162444e-9 m`.

Each row below changes only the named factor, fixes the other formed operand
to scalar NEMO, and scores both the candidate formed quantity and NEMO's
executed `before + term` accumulator. The exact registered bar is zero
unequal words at both boundaries.

| substituted factor | formed unequal words | formed maximum absolute residual | source-accumulator unequal words | source maximum residual (m s-2) |
|---|---:|---:|---:|---:|
| q: Kmm U+V velocity | 288,119 | `2.5167424076469235e-5` | 1,896 | `6.172498493231537e-17` |
| q: Coriolis `ff_f` | 300,029 | `2.5167424076469235e-5` | 1,896 | `1.402273615150676e-13` |
| q: live F thickness | 300,029 | `2.5167424076469235e-5` | 1,896 | `1.4025847811741793e-13` |
| `F_u`: `e2u` metric | 286,719 | `1.1682881327578798e-1` | 1,896 | `5.019693021706198e-13` |
| `F_u`: live U thickness | 286,719 | `1.1682816699612886e-1` | 1,896 | `5.019696240431398e-13` |
| `F_u`: Kmm U velocity | 285,540 | `2.65554990619421e-6 m3 s-1` | 1,896 | `6.174531372304948e-17` |

Kmm velocity is the dominant discriminator for both formed quantities: it
reduces the source residual by roughly four orders of magnitude. It does not
clear either exact bar, however; all 1,896 registered surface-V words still
differ. Coriolis and `e2u` cannot be owners because their providers are already
bit-exact. No single velocity, metric, Coriolis, or live-thickness substitution
removes the joint residual.

**Disposition:** the registered result is
`COUPLED_FORMATION_COMPOSITION`, with an empty sufficient-factor set. This is
a boundary classification, not evidence that the remaining coupling is only
rounding and not evidence for a physics defect. It does not authorize a
shared-path arithmetic change. The round-18 result still stands: scalar q and
scalar `F_u` together make the completed NEMO accumulator exact, while this
round shows that none of their six named source factors is sufficient alone.

The main artifact is
`gyrehr_rounds/round19/stage3_vorticity_factor_walk.json`, SHA-256
`02eba672a5a5883372ba7bdda5a22eb1024ee7120a4f641bfd5a24045da4ec52`,
from clean commit `a0e617801a783208be7e3589eb5e3fd817b9ca23`. Both operator
providers are the 127-member, same-restart, same-executable-passive round-17
scalar/vector acquisitions. The inherited seam-isolation control is pinned by
SHA-256 `4bd8ea66ccbb076c243f7c0671575eb4f514f432bc4d0321bb3ccc0244045e5a`;
it proves every q/transport arm leaves HPG and advection bit-exact and that
scalar q plus scalar `F_u` gives zero source-accumulator unequal words.

## Controls and instrument disposition

The live-q plant changes consumed index `[80,25,0]` and moves one registered
V word; artifact SHA-256
`21bf608534c99b8749d29695718f0f4ee9c76fbd578d3341bbabf44b760d6578`.
The live-`F_u` plant changes `[76,1,0]` and moves two registered V words;
artifact SHA-256
`e679b7ef6eb40235d4197af4cb60379852cb0976d16ef51454bcdc05abc49c41`.
Both are clean-commit `a0e617801` runs, and both retain the exact q, `F_u`,
kernel-replay, and trace-passivity controls.

The first implementation attempted 21 distinct full-step hook compilations.
LLVM refused on memory before an artifact was written. The corrected
instrument evaluates the passive full-step trace once, calibrates one dynamic
call to the same production ENE kernel against the captured live term at zero
words, and reuses that compiled kernel for the six candidate arrays. It also
pins the previously measured full-step HPG/advection isolation control. This
changes execution cost, not the registered exact bar or the compared source
boundary.

The new exposure is gated by the existing private
`return_nemo_operator_components` path. The default return remains unchanged.
Supplying the exposed q and transports back through the existing override
seam reproduces both vorticity components array-exactly; zero q or `F_u`
still moves the V component, and unrelated HPG/advection operands remain
array-exact.

## Inherited pins / OPEN

Inherited unchanged: shipped one-degree digest `56c5171d43504b54`, its
no-worsening ceiling, 31 levels, WS-RK3, split-explicit free surface, TKE/EVD,
Redi tracers, the `r4_smag` deck, 3,600-s step, literal 360-day forcing clock,
scalar-libm certified CPU policy, vector-math climate reference, common
day-180 restart, round-16 western-boundary wet-span reduction, and the exact
source-accumulator bar. No deck, prognostic arithmetic, climate run, NEMO
source, or operator-owned evidence changed.

OPEN, in order:

1. Obtain the two required independent reviews before promoting the negative
   localization from HELD.
2. If the campaign continues upstream, preregister a minimal joint-subset walk
   at the stage-2 state boundary; this round does not infer an owner from the
   near-collapse under Kmm velocity.
3. Post the reviewed result to issue #1455 when network access is available.

No new NEMO or legoESM acquisition is required: every preregistered factor was
available from the admitted records and card mesh. No long run is warranted,
because no shared-path arithmetic changed.

Choice audit -- **ASKED**: expose and compare live stage-3 q and `F_u`; walk
curl/Kmm velocity, Coriolis, live F thickness, `e2u`, live U thickness, and
Kmm U velocity one factor at a time; name an owner only after a single-factor
exact proof; do not launch a long run; request acquisition only for an
unavailable factor. **UNASKED**: no physics edit, deck change, threshold
change, long run, or acquisition was introduced. No user decision is needed.

## Tests and review

Focused factor-reader, exact-word, classification, passive-exposure, replay,
override, ENE-kernel, momentum-closure, and NEMO-state-bridge tests pass. The
complete split CPU/x64 lane battery reports:

- high-resolution acquisition/ladder/climate/movie/instrument split:
  `122 passed, 2 deselected`;
- affected shared ENE/diagnostics/bridge/Smagorinsky split:
  `49 passed, 1 deselected`;
- shared NEMO recipe, testcase recipe, and latitude/longitude C-grid split:
  `98 passed`;
- one-degree card reconciliation and phase-3 gate split:
  `32 passed, 2 deselected`;
- citation-gate self-tests: `17 passed`.

`FULL_LANE_BATTERY_SPLIT_PASS: 318 passed, 5 deselected across five CPU/x64 groups`

The round-specific citation gate passes all five compiled-source citations at
the clean receipt commit with zero unmapped citations, failures, or stale map
entries. Shifting the new `stprk3.f90:207-215` citation by two lines returns a
failed gate and exit status 1. `git diff --check` passes.

Review status is **UNREVIEWED**. The read-only Codex reviewer cannot initialize
its local app server because that service attempts a write on the read-only
host filesystem, including from a disposable writable review clone. The
read-only Claude reviewer starts but terminates with `ENOTFOUND` reaching its
API. No third reviewer binary is installed. These are service failures, not
review verdicts or waivers. No scientific owner or physical change is landed.

## Compiled-source citations

The inherited writer emits stage 2 at Naa, swaps it into Nnn, and passes Nnn
as stage 3's Kmm input at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3.f90:207-215`.
The acquired stage-3 record brackets `dyn_vor` after HPG and before momentum
advection at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:404-430`.
The selected total-vorticity arm dispatches to `vor_ene` at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:235`.
That routine forms Coriolis plus relative curl and divides by live F thickness
at `GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:552-572`.
It forms `F_u=(e2u*e3u)*u`, then consumes q and `F_u` in the V accumulator at
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:576-591`.
