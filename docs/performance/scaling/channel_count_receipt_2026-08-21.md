# The collective channel count, and what it does and does not buy

2026-08-21, Levante (A100-80, 4 GPUs/node, InfiniBand HDR200, jax/jaxlib
0.10.0). One variable throughout: `NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS`,
with `NCCL_P2P_NET_CHUNKSIZE=131072` held fixed in every arm. Every sweep
carries a control in which the halo exchange is removed, and every arm is run
twice with the channel order reversed between repetitions.

## Lat-lon atmosphere, 2048 x 4096 x 26

Instrument: `scripts/cluster/scaling_levante/atm_ll_channel_shape.sbatch`,
now driven by the allocation's node count, so `sbatch --nodes=N` runs the same
sweep at 4N GPUs.

### 64 GPUs (job 27114901)

| channels | step (2 reps) | halo removed | exposed communication |
|---|---|---|---|
| 4 | 4.7358 / 4.7224 | 3.3676 / 3.3754 | 1.357 ms |
| 8 | 4.9106 / 4.9068 | 3.3772 / 3.3678 | 1.537 ms |
| 16 | 4.3076 / 4.3275 | 3.3704 / 3.3683 | 0.949 ms |
| 32 | 4.3008 / 4.3015 | 3.3729 / 3.3715 | 0.929 ms |

Eight channels — the value every run of this campaign used — is the WORST of
the four. Sixteen or thirty-two takes 12.4% off the whole step and 40% off the
communication term. The control is flat to 0.3%, so the knob reaches
communication and nothing else.

### 32 GPUs (job 27129904)

| channels | step (2 reps) | halo removed | exposed communication |
|---|---|---|---|
| 4 | 7.6635 / 7.6696 | 6.3060 / 6.3075 | 1.360 ms |
| 8 | 6.9839 / 6.9747 | 6.3057 / 6.3077 | 0.673 ms |
| 16 | 6.9788 / 6.9838 | 6.3076 / 6.3066 | 0.674 ms |
| 32 | 6.9816 / 6.9786 | 6.3054 / 6.3031 | 0.676 ms |

At half the device count the knob is already saturated at eight, and only four
channels is bad. **The effect is device-count dependent**: harmless at 32,
worth an eighth of the step at 64. The control is flat to 0.04%.

Throughout this note, "exposed communication" means the step with the halo
minus the step without it. It is the lever metric, not the cost of
communicating: work that overlaps is invisible to it.

### What that does to the campaign's one unexplained result — AT 64 GPUS ONLY

The exposed-communication term had been measured at 0.664 / 1.551 / 1.665 ms
at 32 / 64 / 128 GPUs, at eight channels throughout, while payload per rank,
message count and fabric bandwidth were all flat with device count. Read at
the best channel count instead, it is 0.673 at 32 and 0.929 at 64 — a rise of
38%, not 134%.

**Scope: this dissolves the 32-to-64 step of that growth and says nothing
about 64 to 128.** The sweep has not run at 128, and the optimum MOVES with
device count — eight channels suffices at 32 and is the worst value at 64 —
so extrapolating is not allowed. Thirty-two channels may itself be the binding
cap at 128. The claim is a 64-GPU result about a problem that lives above 100.

Two further readings worth recording. The curve is non-monotone in a way a
smooth bandwidth effect cannot produce (four channels beats eight at 64 and
loses badly at 32), which says the knob selects between NCCL regimes rather
than buying proportional bandwidth — so it will have to be re-measured
whenever resolution, halo width or the exchange count changes. And the
four-channel term reads 1.36 ms at BOTH 32 and 64 GPUs: at a fixed channel
count the growth tracks rank count, not fabric distance.

### 128 GPUs, from an earlier receipt that already held the answer (job 27104141)

Eight against sixteen channels only, two reps each, alongside the halo-off
control at each device count:

| devices | 8 channels | 16 channels | change | halo off | exposed communication at 16 |
|---|---|---|---|---|---|
| 32 | 7.030 | 7.037 | +0.1% | 6.307 | 0.730 ms |
| 64 | 4.784 | 4.244 | **-11.3%** | 3.372 | 0.872 ms |
| 128 | 3.918 | 3.877 | -1.0% | 2.298 | 1.579 ms |

**The channel win is a 64-GPU phenomenon.** At 128 it is worth one percent,
and exposed communication stays at about 1.6 ms whatever the channel count.
Whether thirty-two channels behaves differently from sixteen at 128 is the one
thing still outstanding; the sweep is queued.

So the ladder of exposed communication at the best channel count known for
each device count is 0.72 / 0.87 / 1.58 ms at 32 / 64 / 128. The 32-to-64 rise
was a tuning artifact. **The 64-to-128 rise, +81%, is not, and it is the lane's
real open question.** Payload per rank, message count, fabric latency and
fabric bandwidth are all flat across that step, so the candidates are waiting
and jitter accumulating along a 128-rank chain of latitude bands, not the wire.

## Where the rest of the step goes at 128 GPUs (job 27124764)

The whole step at 128, at the best channel count measured, is 3.877 ms:

| term | ms | share |
|---|---|---|
| local work that would exist under perfect scaling | 1.577 | 41% |
| local work in excess of perfect scaling | 0.721 | 19% |
| exposed communication | 1.579 | 41% |

Perfect scaling here means the single-GPU step (201.48 ms, job 27051113)
divided by 128. **Local work alone, 2.298 ms, already exceeds the ideal whole
step**, so communication is not the ceiling on this lane and never was the
whole story.

The excess is not a constant overhead: the halo-off arm at 32 GPUs sits ON the
single-GPU line (6.307 against 6.296, +0.2%). It appears above 32 devices and
grows superlinearly — 0.218 ms at 64, 0.721 at 128.

Where it is NOT:

* Not ghost rows. Four extra latitude rows at the single-GPU per-row cost is
  0.39 ms at EVERY device count, and there is no excess at 32 to explain.
* Not the global sum. Switching the mass fixer off moves the halo-off arm by
  0.02-0.04 ms at 32, 64 and 128 alike (same job).
* Not load imbalance. The benchmark's own rank-imbalance figure is 1.007 with
  the halo and 1.021 without it at 128.

Where it is: inside COMPUTE-KERNEL busy time, which the device trace counts
separately from collectives and from idle gaps. With the halo off, compute
kernel time falls only 2.79x for a 4x device increase, while the number of
kernels per step is unchanged — 179 at 32 devices, 181 at 128. Their mean
duration falls from 35 to 13 microseconds and the idle share between them
roughly doubles, from 2.9% to 5-7%.

**That names a lever this campaign has never touched: the step runs the same
~180 kernels however thin the shard gets.** Fewer, larger kernels is the only
thing that moves it, and it is worth more than what remains in communication
after the channel knob. The discriminator not yet run: the same 128-device
program on a four-times-larger grid has the same shard shape as this one at 64
devices; if compute per row matches that point, the cause is shard shape and
not device count.

## Does the channel count reopen the interior/rim overlap build?

No. Instrument: `scripts/cluster/scaling_levante/halo_overlap_channels.sbatch`
around `scripts/validate/collective_compute_overlap.py`, job 27130389, four
A100-80 on one node, latency-hiding scheduler on in every arm.

The August 2026 receipt found that XLA hides about 79% of ONE collective
behind independent compute and almost none of the same payload split into
twelve, and a multi-week interior/rim build was cancelled on it. That receipt
ran at NCCL's default channel count, and channels are exactly what concurrent
collectives contend for, so the conclusion was worth retesting.

Unhidden cost in ms, at zero independent work and at 3200 rounds of it:

| channels | one collective | twelve collectives |
|---|---|---|
| 4 | 0.564 -> 0.047 (92% hidden) | 1.611 -> 1.248 (23% hidden) |
| 8 | 0.326 -> 0.036 (89% hidden) | 0.879 -> 0.750 (15% hidden) |
| 16 | 0.177 -> 0.077 (56% hidden) | 0.527 -> 0.450 (15% hidden) |
| 32 | 0.138 -> 0.055 (60% hidden) | 0.470 -> 0.449 (4% hidden) |

Twelve collectives hide between 4% and 23% at every channel count, and the
compiled instruction window is `[0, 0, 8]` in every arm — only the last round
ever gets one. **The practical verdict stands: do not build the interior/rim
split on the strength of this.**

Two honest qualifications, both raised in review:

* August's stated MECHANISM is partly refuted by this table. Channels remove
  about two thirds of the twelve-collective unhidden cost, so a large part of
  what was called "the rounds serialise on the shared communicator" was
  channel starvation. The recommendation survives; the explanation does not.
* An empty scheduling window cuts both ways. It shows the compiler will not
  create overlap on its own, which is precisely the argument people use FOR
  restructuring dependencies by hand. What this microbenchmark refutes is
  overlap arriving from scheduler motion alone, not overlap in principle.
* The falling HIDDEN FRACTION as channels rise (92, 89, 56, 60% for one
  collective) is not a contradiction: there is an exposure floor of roughly
  40-80 us that no channel count removes, and it is a growing share of a
  shrinking total. Optimise the absolute unhidden cost in the many-round
  regime, never the fraction.
* The twelve collectives in the probe are mutually INDEPENDENT — each permutes
  its own slice, and only the scalar reductions that consume them are chained.
  Production's thirteen exchanges are more dependent than that, being spread
  across three Runge-Kutta stages. So the probe is an optimistic bound on
  overlap and still shows none.

### What it says about the coalescing lever — less than it first appears

The tempting reading is that one collective costs 0.138 ms unhidden while the
same payload split into twelve costs 0.470, so splitting costs 3.4x. That
ratio is an AVERAGE and prices removed rounds wrongly. The affine reading of
the same row is what matters: a per-round fixed cost of about
(0.470 - 0.138) / 11 = 30 us, on top of a payload term of about 108 us.

So cutting the lat-lon step's thirteen exchanges to seven (bucket E of
`latlon_cp_packing_roadmap.md`) is worth **at most 6 x 30 us = 0.18 ms**, on
four GPUs over NVLink where bandwidth is nearly free — not the 0.43 ms the
average ratio suggests. Against that it must pay for the wider halo it needs:
more bytes on a fabric where bytes are not free, and twice the ghost rows,
whose redundant compute at 128 devices is of the same order as the saving.
**At 128 devices this lever could be net negative.** It cannot be priced from
a single-node microbenchmark; the settling measurement is the same
split-payload probe run at 64 and 128 GPUs with the production per-exchange
byte counts and their doubled width, fitting a fixed, a per-round and a
per-byte term separately.

## Not decided here

Whether to change the production launch environment's channel count is a
configuration decision and has not been taken. Nothing in the model or in
`backend.py` was changed; the numbers above come from benchmark arms that set
the variable themselves.
