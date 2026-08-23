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

I first read the negative residual as "contention confirmed", and both
reviewers refused it: a marginal cost above the average kills the linear model,
but a saturating resource and a message crossing a protocol threshold both
produce that and predict opposite things about a SMALLER payload. So the
curve was measured instead of argued.

**THE CURVE, four points at 128 GPUs (job 27143144), and it is LINEAR:**

| payload | wire term | per extra copy |
|---|---|---|
| x1 | 1.564 ms | — |
| x2 | 3.329 | 1.765 |
| x4 | 6.416 | 1.544 |
| x8 | 13.181 | 1.691 |

Fitted across an eightfold range: **1.66 ms per payload copy with an intercept
of -0.10 ms.** The cost is pure payload and there is no fixed term at all —
not small, zero within the scatter. Nothing is convex and nothing crossed a
threshold; the earlier "super-linear" reading was the tiny negative intercept
plus noise, and it is retracted.

Marginal bandwidth is 6.6 GB/s per rank and stays there across the whole
range. That is about a quarter of one HDR200 port, and it does NOT degrade as
payload grows — so the fabric is not saturating; this lane simply gets a
quarter of a port's worth of bandwidth per rank, which is close to what the
measured off-node penalty predicts.

Two consequences, both now on four points rather than two:

* **Message-count levers are dead here.** A zero fixed term means removing
  exchanges buys nothing at this device count.
* **Byte reduction pays back LINEARLY and predictably.** Removing a sixth of
  the payload is worth about 0.28 ms, seven percent of the step. Halving it
  would be worth about 0.83 ms, twenty-one percent.

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
| lat-lon, both switches on | 3.504 | — | — | 1.574 | **2.2x** |
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
   rather than sent. Removed losslessly, and unlike every earlier packing idea
   it is a COPY rather than a re-derivation, so the one-unit-in-the-last-place
   mismatch that blocked the exchange-merging work does not apply. Codex found
   it; nobody has built it.

   **BUILT AND MEASURED: -5.69% of the step at 128 GPUs** (job 27145909,
   3.7899 -> 3.5741 ms, arm spreads 0.20% and 0.34%), against a bar of 1.5%
   written before the run. Bit-identical, off by default.

   I had predicted one to three percent, from counting the duplicate as 16.6%
   of that epoch's payload and guessing the epoch's share of the total. The
   measured saving is 14% of the whole communication term, so the entry epoch
   carries far more of the halo bytes than I assumed. The arithmetic
   under-called it; the A/B is why it was run.
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

## The icosahedral lane, split three ways (job 27147213, 64 GPUs)

| term | ms | share |
|---|---|---|
| local compute | 3.470 | 60% |
| device-side halo staging | 0.310 | 5% |
| communication | 1.980 | 34% |
| step | 5.760 | |

A four-point payload curve on the same job — 5.760, 7.075, 9.550 ms at one,
two and four times the payload — is LINEAR (2.88 against 3.00 for a pure
bandwidth term). Communication splits **66% payload, 34% fixed**. The lat-lon
lane's fixed term was zero, so **the two lanes need different levers**: cutting
exchanges pays here and pays nothing there.

The obvious byte lever here is REFUTED. The coloured rounds pad every message
to the widest pair in the round, shipping about twice the true halo, and an
exact-size path already exists. An older receipt measured it at -6.4%. Re-run
with the channel count set to sixteen (job 27148079, 65 steps, palindrome, two
repetitions): wide 5.210 and 5.200 ms, exact-size 5.300 and 5.270 — **+1.54%,
slower**. Sixteen channels already makes bytes cheap enough that the
exact-size collective cannot earn back its per-operation cost. The older
figure was measured at the default channel count against a 6.70 ms baseline
and does not survive the change.

That is the second pair of levers today that does not stack — the first being
the two lat-lon halo switches, which stacked at reduced value. A lever's size
is a property of the configuration it was measured in.

## What is left, and why it is not cheap any more

Both lanes are out of cheap exact byte cuts.

On the lat-lon lane a full audit of what actually goes on the wire found one
remaining exact reduction after the duplicate row: two vertical endpoints of
the mass-flux and vertical-velocity fields are literal zero by construction
and are shipped anyway. It is 8,192 elements per direction per stage against
the duplicate row's 106,496, so about 0.4% of the step — below the bar of any
A/B worth a 32-node allocation. The larger candidates on that lane, the layer
thickness and the pressure logarithm, are both reconstructible from surface
pressure but are RE-DERIVATIONS rather than copies, and this campaign has a
receipt showing a mathematically identical local reconstruction differing by
one unit in the last place where a copied operand matched exactly.

On the icosahedral lane the static surface geopotential is packed into every
exchange and its tendency is zeroed by construction — verified in the source,
not taken on report. It is one element per cell record out of twenty-eight, so
roughly 0.3% of the step. Recorded and deliberately NOT built: knowing the
size was cheaper than building it and measuring.

What remains is structural rather than contained:

* **Kernel granularity on the lat-lon lane.** The step runs about 180 kernels
  whatever the shard size, averaging 13 microseconds at 128 GPUs, and no
  compiler setting in this build moves the count. Cutting it means changing
  what the dycore asks for, not how it is compiled.
* **The on-device halo staging on the icosahedral lane**, 0.310 ms at 64 GPUs,
  measured by an arm that skips the gather, concatenate and scatter while
  changing nothing else. The coloured fill scatters into the local buffers once
  per round, thirteen times, and those receive positions are disjoint by
  construction (each halo row has one owner), so the scatters could in
  principle be deferred and merged. The free screen for this did NOT resolve:
  the per-round scatters do not appear as distinct operations in any module a
  virtual-device dump produces, and the wide-halo fill is not inside the
  per-step module. Whether merging recovers any of the 0.310 ms needs hardware,
  not a dump.
* **The fixed per-round term on the icosahedral lane**, 0.665 ms across
  thirteen coloured rounds. The colouring is already optimal for the graph, so
  the rounds cannot be cut by repartitioning; cutting them means changing the
  halo depth, which follows from the integrator's stage count.
* **A two-dimensional lat-lon decomposition**, which is the only remaining way
  to cut that lane's halo bytes substantially, and which lost 22% at 64 GPUs
  for a reason still not established.

## The icosahedral level-count table, and why the mechanism stays unknown

One GPU, subdivision 6, the same mesh throughout, two repetitions each.

| levels | ms per level | | levels | ms per level |
|---|---|---|---|---|
| 16 | 0.253 | | 26 | 0.779 |
| 20 | 0.248 | | 27 | 0.633 |
| **24** | **0.679** | | 28 | **0.665** |
| 32 | 0.266 | | 30 | 0.751 |
| 36 | 0.279 | | 31 | 0.613 |
| 40 | 0.271 | | 34 | 0.481 |
| 44 | 0.270 | | 46 | 0.450 |
| | | | 18, 21, 22, 25 | 0.371 - 0.691 |

Cheap: 16, 20, 32, 36, 40, 44. Everything else measured is two to three times
dearer per level.

Divisibility by four is NECESSARY — all fourteen counts not divisible by four
are expensive, without exception — but NOT sufficient: 24 and 28 are multiples
of four and cost 2.5x what 44 does. I claimed the divisibility rule was a
perfect separation and predicted 24 and 28 would be cheap; the prediction was
written down, tested within the hour, and refuted. The original scan had
measured twelve level counts and both counterexamples happened to be absent
from it, which is how a rule fitted to a set that excludes its own
counterexamples looks perfect.

**The mechanism is not established, and this section deliberately stops
guessing.** What IS established, and rules out a whole class of explanation:
the compiled program is structurally IDENTICAL across cheap and expensive
counts. At 24, 26, 32 and 44 levels it compiles to 2292, 2292, 2271 and 2292
instructions, 108 fusions in every case, the same twenty-three opcodes with
the same histogram, and the same shape multiset with only the extents
differing. Precisely: this is the SERIAL step, because the factory returns the
plain model step at one device — which is the right program, since the level
anomaly was measured on one GPU, and the comparison is serial against serial. So this is not different code being emitted — it is the same code
executing differently at different extents, which no dump can resolve and
which alignment of the state's column stride, tested earlier, was the wrong
place to look for.

The practical consequence is a lookup table rather than a rule, and the
decision it informs is unchanged: 26 to 24 buys 13%, 26 to 32 buys a factor
of 2.9.

## Six proposed levers, dispositioned without spending an allocation

Asked for levers this campaign had not considered, six came back. Four are now
closed on free evidence, one needs hardware, one is a decision rather than an
optimisation.

| # | lever | disposition |
|---|---|---|
| 1 | Pad the derived operator widths on the icosahedral lane, on the grounds that the level-count anomaly is a compiler-shape effect | **REFUTED.** The compiled step is structurally identical at cheap and expensive level counts — same instruction count within 1%, same 108 fusions, same opcode histogram. The shapes do not change the program, so padding them has no shape argument behind it. |
| 2 | Coalesce the icosahedral halo's thirteen per-round scatters into one | **NEEDS HARDWARE.** Semantically available — receive positions are disjoint by construction — and it attacks a measured 0.310 ms. But the free screen does not work here: the per-round scatters are not distinct operations in any module a virtual-device dump produces. |
| 3 | A one-evaluation multistep integrator, cutting the icosahedral halo depth from nine rings to three | **NOT AN OPTIMISATION.** Large upside, but it changes the numerics and the stability properties; that is a decision about the model, not a lever to pull. |
| 4 | Fuse the vertical column pipeline to avoid materialised intermediates | **REFUTED.** All eighteen reverse operations around the vertical cumulative sums are already inside fused computations; none is materialised at top level. There are no full-size intermediates to remove. |
| 5 | Wire the anchored mass correction into the sharded path so the step reduces one field instead of two | 0.02 to 0.10 ms — below the refute bar of any A/B worth an allocation. |
| 6 | The scan-folded Runge-Kutta schedule | Folding three stages into a loop compiles fewer distinct kernels but launches the same number, so it cannot move a per-kernel floor. |

## Decisions that are yours, not taken

| # | decision | evidence |
|---|---|---|
| 1 | Sixteen collective channels — but chosen PER LANE AND PER DEVICE COUNT, not once | -12.4% lat-lon and -9.3% icosahedral at 64 GPUs; at 128 it is -9.9% on the icosahedral lane and only -1.0% on the lat-lon one |
| 2 | Thirty-two vertical levels instead of twenty-six on the icosahedral lane | 26 levels measured 3.1x more expensive per cell per level than the cheap counts, on ONE GPU. Local compute is 60% of that lane's step, so this is its largest candidate — but the number at 128 GPUs with the right channel count is being measured now rather than assumed from the single-GPU table |
| 3 | Turn the level-leading halo on for the lat-lon lane | -4.0% at 128 GPUs, bit-identical, below the 5% bar it was gated against |
| 4 | Turn the duplicate-row removal on for the lat-lon lane | **-5.69% at 128 GPUs**, bit-identical, past the 1.5% bar set before the run; arm spreads 0.20% and 0.34% |
| 5 | Turn BOTH lat-lon switches on together | **-7.54% at 128 GPUs**, 3.7899 -> 3.5040 ms. They stack but do not add: the layout switch is worth 4.03% alone and 2.05% on top of the duplicate removal, because removing a row from the wire also removes the conversions that row needed. Measured, not assumed |

Nothing in the model's defaults was changed to produce any number here.

## The lat-lon lane's remaining lever, and why its one measurement failed

At 128 GPUs a latitude band ships FOUR rows of halo for every row it owns. The
band's halo is two whole circles of longitude however thin the band gets, so
adding devices thins the band and leaves the halo alone. Splitting longitude
as well turns the band into a tile whose perimeter shrinks with it: eight
longitude splits take the ratio from 4.00 to 0.62, six times less traffic.
That is the largest single lever left on this lane and it is pure geometry,
reproducible in a second with
`scripts/validate/latlon_halo_surface_ratio.py`.

The tile was measured once at 64 GPUs and LOST, 6.01 against 4.92 ms. Reading
the code says why, and the reason is messages rather than bytes: the packing
that makes a band's exchange one message per stage is latitude-only, the tile
exchanges latitude and longitude in separate rounds, every tile runs the pole
handling whether or not it touches a pole, and each east-west wrap becomes two
ring messages instead of a local copy. Off the source that is 108 messages
against 13.

**Measured now, on the compiled program at four GPUs, 128x256:** latitude
bands send 13 point-to-point messages per step, a two-by-two tile sends 28.
So 2.1 times, not eight — the source reading overstated it, because the
compiler already groups several fields into one message. The extra fifteen are
the east-west exchanges, and they are thin: one column of longitude by
sixty-four rows by twenty-six levels, about seven kilobytes each, which is
latency rather than bandwidth. Fifteen small messages is the right order to
account for the 1.09 ms the tile lost at 64 GPUs, though that arithmetic is a
coincidence until something measures it.

Two attempts at this census were discarded before this one. The first read its
counts from the layout dump, which runs with the exchange replaced by the
identity and therefore reports none. The second compared a latitude band
against a "tile" on two devices, where asking for two longitude splits leaves
a single latitude band -- two one-dimensional decompositions pointing in
different directions, not a tile. The arm now refuses to run anything labelled
a tile that is not split in both directions.

**And the fix is already written.** There is a packed two-dimensional exchange
in the code, with its own receipts, which carries every field of a stage in
one message per direction instead of one per field. Nothing calls it. The
atmosphere step asks for the packed exchange through a helper that returns
nothing at all for a two-dimensional mesh, by design, so the tiled lane has
never run packed. Wiring it is blocked on one thing: the step's exchange packs
three kinds of field at two halo depths in a single message, and the
two-dimensional body accepts only one kind at one depth, where its
one-dimensional twin accepts both.

**A caveat that applies to the transpose numbers above.** The layout-dump arm
runs with the halo exchange replaced by the identity, so that the staging work
around it can be counted on its own. Its compiled module contains no halo
messages at all. The transpose counts taken there stand -- the staging
survives, only the wire is removed -- but a MESSAGE count taken from that arm
reads zero and means nothing. One was taken and discarded before it reached
this document; the arm that replaces it leaves the communication in and
refuses to return a verdict if the band arm reads zero messages.


### The packer was wired, and it did not help. Here is where the messages are.

The tiled packer now takes what the band packer takes and the dycore calls it,
behind its own switch, bit-identical. The gate written before the run asked for
the tile's message count to fall towards the band's. It did not: 29 messages
against 28 with the packing off, i.e. one more.

The packer is not inert -- the compiled program changes, and the latitude
message grows from a ten-thousand-element buffer to a thirty-three-thousand
one, which is exactly the epoch's fields being carried together. It is simply
that the stage exchange was never where the tile's extra messages were.

Sorted by which devices they connect, the tile's 28 point-to-point messages
are 16 east-west and 12 north-south. The band's 13 are all north-south. So
every one of the tile's extra messages is a LONGITUDE exchange, and the packer
only merged the handful the stage epoch owns.

Those sixteen are the per-operator longitude wrap. On a latitude band the
whole circle of longitude is local, so wrapping a field to get its east and
west ghosts is a copy; on a tile it is two ring messages, and every zonal
gradient and interpolation in the step does its own. Cutting them means
padding longitude ONCE per stage and handing the padded field to the
operators, rather than each operator wrapping for itself -- a larger change
than packing the stage exchange, and now the one the lever depends on.

The packer stays, off by default. It is a prerequisite rather than a win: once
the wraps are hoisted, the stage exchange is the next thing that would
dominate.


## In flight: writing the icosahedral halo once instead of thirteen times

The icosahedral halo runs thirteen coloured rounds at 64 GPUs, and each round
writes its received rows into the local buffers as it lands. The rows a device
receives are disjoint across rounds, so those thirteen writes can be deferred
and issued as one. That is what the merged-scatter switch does, off by
default, and the 64-GPU arm for it is queued.

What it is aimed at is measured rather than fitted: the arm that keeps the
kernel and the local region but performs no halo staging at all runs 0.310 ms
faster than the arm that keeps the staging and deletes the communication. So
0.310 ms of a 5.760 ms step is the entire gather, concatenate and scatter
budget, and the scatters are one of those three. The gate written before
submitting asks for 1%, and flags anything larger than 0.310 ms as the knob
moving something it was not built to move.

Both reviewers pushed back on the same thing and both were right: the original
version had every round share one padding row, which made the single scatter's
index list non-unique. That is fine going forward, because the shared row is
trimmed off before anything reads it, but a repeated index has no defined
winner and its reverse-mode transpose sends that row's cotangent back to every
round at once. Each round now owns its own padding row, so every index is
unique and the scatter carries the no-duplicates promise. The disjointness the
whole thing rests on is asserted where the schedule is built, on the host,
rather than trusted in the fill.

## A defect found while gating that change, not while looking for it

The icosahedral step's reverse-mode gradient is finite on one device and NOT
finite once the model is sharded: 96 non-finite entries of the gradient on two
devices, 224 on four, on interior cells, growing with the device count. Same
initial state, same time step, same loss. Since end-to-end differentiability
is a goal of this model, that is a defect of the sharded path, and it means
training on a sharded unstructured mesh currently carries undefined gradients
on roughly one cell in a hundred.

The cause is not established and no attempt was made to establish it here.
`scripts/validate/mpas_sharded_grad_finiteness.py` reproduces the table in
under a minute on CPU.

Two smaller things worth knowing before trusting a green suite on this lane.
The halo test suites pollute one another when run in one process — a
bandwidth test that passes alone fails in company — because the switches are
read when the program is traced and a compiled program is cached. And the
compiled-program checks in these gates count operations; they do not measure
time, which is why every claim above rests on a machine arm and not on a
count.
