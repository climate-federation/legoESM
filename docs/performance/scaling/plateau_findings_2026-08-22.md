# Where the >100-device plateau actually is, and what is left to do about it

Levante, A100-80, 4 GPUs a node, InfiniBand HDR200, jax/jaxlib 0.10.0.
Lat-lon 2048x4096x26 in latitude bands; icosahedral subdivision 9 x 26 levels
on a space-filling-curve partition. Every number below is from a job named in
its row, with two repetitions and an interleaved or palindromic arm order.

## The short version

Two things were true at once and only one of them was being worked on.

1. **The collective channel count was starved on BOTH lanes, all campaign.**
   Sixteen channels instead of eight is 12.4% of the lat-lon step at 64 GPUs
   and 9.3% of the icosahedral step at the same count. Nothing was changed to
   get it; the campaign had simply been running the wrong value.
2. **Communication was never the whole ceiling.** With the halo switched off
   entirely the lat-lon step at 128 GPUs still takes 2.43 ms against a
   perfect-scaling 1.57. Local work alone exceeds the ideal whole step.

## Lat-lon

### The step at 128 GPUs, best settings known

| term | ms | share |
|---|---|---|
| local work under perfect scaling | 1.57 | 41% |
| local work in excess of that | 0.72 | 19% |
| communication | 1.58 | 41% |

### Communication is 94% bytes (job 27130343, 64 GPUs, 32 channels)

Doubling the halo payload costs 1.546 ms, of which 0.117 is device-side
packing, so the extra bytes cost 1.429 ms on the wire. Against a total
communication term of 1.513 ms, that leaves **0.083 ms of per-operation
cost**. CONFIRMED.

Two consequences:

* **Merging the thirteen exchanges into seven is not worth building.** It
  attacks the 0.083 ms only — at most 0.038 ms, under 1% of the step — and the
  wider halo it needs adds 14-17% payload, which costs about 0.2 ms. Net
  negative. This closes `latlon_cp_packing_roadmap.md` bucket E.
* **The remaining communication lever is fewer BYTES.** A latitude band ships
  two rows of the whole longitude circle per field however many devices there
  are; a two-dimensional tile ships a perimeter that shrinks with the tile,
  roughly six times fewer cells at 128 devices. Tiling lost 22% at 64 GPUs for
  a reason that is still not established — it is NOT layout conversions
  (refuted, job 27134078), and the same dump shows the tiled step compiling
  4895 instructions against the band step's 4118.

At 128 GPUs (job 27138082, same split, 32 channels) doubling the payload costs
1.778 ms, of which 0.022 is packing, so the extra bytes cost 1.756 ms —
**more than the entire 1.551 ms communication term**, leaving a per-operation
residual of -0.205 ms.

RETRACTED, same day, on review: I read that as "contention confirmed". It is
not confirmed. A marginal cost above the average is formally impossible for a
cost linear in bytes, so the linear split stops describing the data at 128 —
but TWO things produce that, and two points cannot separate them:

* a convex response, a shared resource saturating; or
* a DISCONTINUITY, the doubled message crossing a protocol or algorithm
  threshold in the collective library.

They predict opposite things about halving the payload — convexity says less
than half the doubling delta, a crossed threshold says possibly more — so the
extrapolation this was about to justify is not supported. What is needed is a
CURVE, at least four payload multiples at each device count, not a chord.

Against contention there is also an arithmetic check worth recording: the
halo moves roughly 11 MB per device per step, so the extra copy's 1.756 ms is
about 6 GB/s of marginal bandwidth — a quarter of one HDR200 port. If bytes
were the binding resource that figure would be near line rate. It is not.

What DOES survive: the per-operation term is unmeasurable at 128, so
message-count levers are closed there regardless of which explanation wins.

### Local work: granularity, not layout

The step runs the same ~180 kernels whatever the shard size — 179 at 32 GPUs,
181 at 128 — and their mean duration falls from 35 to 13 microseconds while
the idle share between them doubles. Fitting each kernel class to a fixed part
plus a work part puts 0.617 of the 0.707 ms per-step fixed cost in transposes,
which scale 1.38x for a 4x device increase while every other class manages
3.2-3.5x.

Those transposes convert between `(level, lat, lon)` and `(lat, lon, level)`:
the state is stored lat-leading and XLA:GPU pipelines level-leading. Most of
them act on halo edge slices of one or two latitude rows, whose shape does not
follow the shard.

CORRECTION (2026-08-22): the counts first published here — "28, of which 26
are edge slices, falling to 2" — were produced by a census whose pattern did
not allow the `ROOT ` prefix, so it silently skipped every transpose that is
the root of its computation and reported about a third of the module as the
whole. Recounted: the step holds **67** transposes, none free, and the switch
removes **60 of them, leaving 7** (instructions 4267 -> 4118). The direction
and every conclusion are unchanged and the effect is larger than reported; the
numbers were wrong. The seven survivors are six full-shard conversions at the
shard-map and scan boundaries — in both directions, including the staggered
wind's `n_lon+1` extent — and one single-row leftover.

`LEGOESM_LATLON_HALO_LEVEL_LEADING=1` serialises those payloads level-first.
**Wall clock at 128 GPUs: -4.03%** (job 27133396, 3.7934 -> 3.6406 ms, arm
spreads 0.27% and 0.43%). That is below the 5% bar written before the run, so
by that gate it is inconclusive rather than confirmed. It is bit-identical and
off by default.

The prediction was 13-15% and the lesson is that **transpose count is not
transpose time**: the sixty removed act on one- and two-row slices, the
survivors act on the full field at the shard-map and scan boundaries, and they
plausibly carry the other ~0.55 ms. Those six full-shard conversions are now
the largest single named item on this lane.

Not the lever, each refuted cheaply:

* A layout constraint on the scan carry (job 27140236): VOID, not refuted.
  The synthetic probe compiles to zero transposes in BOTH arms, so its control
  does not reproduce the thing it was built to test and it can say nothing.
  It does show the constraint is answer-preserving and costs eight extra
  instructions. Reproduce the conflict before re-running it.
* Compiler fusion flags (job 27134604): four of eight do not exist in this
  jaxlib, and the four that do leave the step at exactly 129 fusions. The
  kernel count is the program's shape, not a setting.
* A level-count cliff (job 27078027): the lat-lon core is flat to 1.12x from
  16 to 40 levels. The icosahedral core's threefold cliff is not general.
* Interior/rim overlap (job 27130389): twelve collectives hide 4-23% of their
  cost at every channel count, so the 2026-08-13 decision not to build it
  stands. Channels do cut the unhidden cost of many collectives by 71%, which
  is a separate effect.

## Icosahedral

Job 27131694, subdivision 9, 26 levels, 64 GPUs:

| channels | step | halo off | communication |
|---|---|---|---|
| 4 | 6.705 | 3.770 | 2.935 (43.8%) |
| 8 | 5.745 | 3.785 | 1.960 (34.1%) |
| 16 | 5.225 | 3.770 | 1.455 (27.8%) |
| 32 | 5.210 | 3.780 | 1.430 (27.4%) |

Monotone and saturating, the pre-registered shape meaning "channel-starved,
adopt the saturation point". Control flat to 0.40%, per-rank spread 0.007 ms —
this lane has no load imbalance at all. Whether it holds above 100 devices is
running now; on the lat-lon lane it did not.

This lane does NOT have the layout defect: its compiled step holds three
transposes and they follow the shard (job 27133706). Do not port that fix.

## How far from the limit, and what would close it

Every number in this section is measured at 128 GPUs with the best settings
known for that lane, against a floor taken from the SAME runs rather than
from a model: the halo-off arm is the time the step would take if
communication were free, and no amount of communication work can go below it.

| lane | step | of which local | communication | perfect scaling of 1 GPU | distance |
|---|---|---|---|---|---|
| lat-lon 2048x4096x26 | 3.909 ms | 2.287 | 1.551 (40%) | 1.574 ms | **2.5x** |
| icosahedral L9 x 26 | 5.000 | 2.330 | 2.670 (53%) | 1.359 | **3.7x** |

CORRECTED on review: the lat-lon row previously mixed arms, quoting a step
time measured with the level-leading halo ON beside a communication term
measured with it OFF, and the three columns did not add up. Every lat-lon
figure above now comes from ONE job (27138082, 32 channels, level-leading
off), where step minus local minus communication leaves 0.071 ms of
trajectory difference between the arms. The level-leading switch is worth a
further 4.0% and is NOT folded in, because it was measured at a different
channel count.

Perfect communication alone is worth 1.7x on the lat-lon lane and 2.1x on the
icosahedral one. Both local floors are also above perfect scaling — 1.46x and
1.71x respectively — so neither lane is granularity-free either.

The icosahedral denominator is a fresh single-GPU measurement under today's
flags rather than a borrowed one: 173.91 ms at subdivision 9 with 26 levels,
job 27141740, with the discarded warm-up arm reading 173.89 for the same
configuration. The job timed out before its second labelled repetition, so the
two figures quoted are the warm-up and the first arm; they agree to 0.01%.

What would close each gap, ranked by what the measurements support:

1. **Fewer halo bytes on the lat-lon lane** — but measure the payload CURVE
   first. Communication there is dominated by payload, and a latitude band
   ships two rows of the entire longitude circle per field however many
   devices there are. A two-dimensional tile ships a perimeter, roughly six
   times fewer cells at 128 devices. Blocked on two things: why tiling lost
   22% at 64 GPUs, which is not layout and not yet explained; and whether
   halving the payload saves anything like half, which the doubling
   measurement cannot answer.

   A REDUCED-PRECISION HALO IS NOT THAT LEVER, on two independent grounds,
   and my pricing of it was wrong twice over:

   * Arithmetic. Doubling the payload ADDS one payload; halving it REMOVES
     half of one. Even under a linear cost the downward saving is half the
     upward delta — at most 0.72 ms at 64 GPUs and 0.88 at 128, before the
     cast costs anything — and a convex response makes it smaller. I had been
     quoting the doubling delta as if it were the halving saving.
   * Numerics. The exchanged stack carries the logarithm of surface pressure,
     and bfloat16's spacing near ln(100000) is 0.0625. That is enormous for a
     quantity the step then DIFFERENCES across the band cut, and it breaks the
     deliberately matched cancellation between the geopotential gradient and
     the pressure-gradient term, leaving a force at every cut. Sixteen-bit
     floats are worse still: surface pressure overflows their range outright.
     If a reduced-precision halo is ever measured, the rounding must also
     happen at the SOURCE, with the owner using the same rounded value in its
     own interface flux — a cast on receipt leaves the two sides of every
     internal boundary computing different fluxes from different copies of one
     row, which breaks flux-form conservation in a way a short test passes.

   **The exact byte reduction that IS open: the step sends the same
   temperature rows twice.** The packed epoch carries temperature once as a
   two-row fold pad and again as a one-row wall pad, and the code says so —
   the duplication buys uniform per-field unpacking. The wall pad's rows are a
   SUBSET of the fold pad's, so the second copy can be sliced out of the first
   rather than sent. That is roughly a sixth of the epoch's payload, removed
   losslessly, and unlike every earlier packing idea it is a COPY rather than
   a re-derivation, so the one-unit-in-the-last-place mismatch that blocked
   the exchange-merging work does not apply. Codex found it; nobody has
   built it.
2. **Whatever makes icosahedral communication grow 1.83x when devices
   double.** It is not payload — halo bytes per device fall as the partition
   shrinks — and the round count is fixed at 11-14 by the edge colouring. The
   same contention signature as the lat-lon lane's 64-to-128 growth. Nothing
   measured yet separates fabric contention from per-round serialisation.
3. **Fewer, larger kernels on the lat-lon lane.** The step runs ~180 kernels
   whatever the shard size and their mean duration falls to 13 microseconds at
   128 GPUs. Not reachable by any compiler setting that exists in this build,
   so it is dycore work.
4. **The six full-shard layout conversions that survive at the shard-map and
   scan boundaries.** Plausibly ~0.55 ms of the lat-lon step. The cheap route
   — pinning the carry's device layout — has not been shown to work; the probe
   built to test it was void.

## Decisions that are yours, not taken

| # | decision | evidence |
|---|---|---|
| 1 | Sixteen collective channels — but chosen PER LANE AND PER DEVICE COUNT, not once | -12.4% lat-lon and -9.3% icosahedral at 64 GPUs; at 128 it is -9.9% on the icosahedral lane and only -1.0% on the lat-lon one |
| 2 | Thirty-two vertical levels instead of twenty-six on the icosahedral lane | 26 levels measured 3.1x more expensive per cell per level than the cheap counts; faster at every device count despite 23% more work |
| 3 | Turn the level-leading halo on for the lat-lon lane | -4.0% at 128 GPUs, bit-identical, below the 5% bar it was gated against |

Nothing in the model's defaults was changed to produce any number here.
