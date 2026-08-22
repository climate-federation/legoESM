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

PLAUSIBLE mechanism for the growth from 0.87 ms at 64 GPUs to 1.58 at 128
while payload, message count and fabric latency are all flat: per-rank
bandwidth falls as more ranks share the fabric. The discriminator is the same
payload split at 128 GPUs, queued.

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

## Decisions that are yours, not taken

| # | decision | evidence |
|---|---|---|
| 1 | Sixteen collective channels as the production default on both lanes | -12.4% lat-lon and -9.3% icosahedral at 64 GPUs, two independent lanes, controls flat |
| 2 | Thirty-two vertical levels instead of twenty-six on the icosahedral lane | 26 levels measured 3.1x more expensive per cell per level than the cheap counts; faster at every device count despite 23% more work |
| 3 | Turn the level-leading halo on for the lat-lon lane | -4.0% at 128 GPUs, bit-identical, below the 5% bar it was gated against |

Nothing in the model's defaults was changed to produce any number here.
