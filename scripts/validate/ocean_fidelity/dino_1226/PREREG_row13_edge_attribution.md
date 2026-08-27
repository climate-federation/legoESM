# Pre-registration — rows 13 and 49 edge attribution

Registered before this lane computes any new statistic from the verdict360
states.  The motivating observations are upstream results, not measurements of
this lane: row 13 is immediately south of the pinned channel, has 50/52 wet T
columns and 49/52 wet U columns, and carries a day-360 control-member gap of
+0.0322305 Sv; row 49 has the same wet-column counts and is immediately north
of the channel.  The channel itself is T-rows 14--48 inclusive.

Instrument to be added after this file is committed:
`row13_edge_attribution.py`.  Everything is offline from the saved verdict360
states.  No model is stepped and no GPU is used.

The exact stamp-refusing loaders and transport reducers are imported from
`regional_audit.py` at commit `102ef501a`, with SHA-256
`2f2bbc03afed792ea0148002029a6ad6481c93bd2ee37679b4bf615aab392fe7`.
The signed Pearson implementation and channel constants are imported from the
post-review `channel_rescore.py` at commit `7af72bc33`, with SHA-256
`a3b51de6292df22b90fa9bd385dcb22ab47a3049c737f2e2abf1a28eb5a18be9`.
Both hashes are checked before any state is opened.

## 1. Fixed data and signs

The target T-rows are exactly 13 and 49.  The channel is imported as
`channel_rescore.CHANNEL_ROWS`, never retyped.  The southern comparison region
for row 13 is P1 with the target removed, rows 0--12.  The northern comparison
region for row 49 is P3 with the target removed, rows 50--78.  The immediate
neighbours are fixed as follows:

| target | basin-side neighbour | channel-side neighbour | basin profile |
|---:|---:|---:|---|
| 13 | 12 | 14 | P1 rows 0--12 |
| 49 | 50 | 48 | P3 rows 50--78 |

The registered horizons are days 90, 180, 270 and 360.  The four paired
members are the verdict360 control and three 1e-14 temperature-nudge members.
Every gap is `legoESM - NEMO`.  Positive transport is eastward.  All section
reductions use the audit's identical U mask, `e3t_1d` thickness, `e2u` width,
and mean over longitudes 2 through -3.

## 2. Total trajectory, floors, and honest row verdicts

For member `m`, horizon `t`, and row `j`, the total gap is

```
G[m,t,j] = regional_audit.row_transports(lego[m,t].u, umask)[j]
           - regional_audit.row_transports(NEMO[m,t].u, umask)[j]
```

The row's own two-sided floor is

```
F[t,j] = sqrt(std_m(L[m,t,j])**2 + std_m(N[m,t,j])**2)
```

with `ddof=1` separately on each side.  No channel, band or global floor is
transferred to either row.  The output reports every member and horizon, both
per-side spreads, the RSS floor, `abs(G)/F`, and the audit's exact saturation
result and reason.  Floors have n=4 per side and are factor-of-two estimates.

Verdict vocabulary is the audit's `classify` with no storage-quantum claim:

* `abs(G) <= 2 F`: **INDISTINGUISHABLE** for that exact row/horizon statistic,
  with an unsaturated flag if applicable;
* `abs(G) > 2 F` and saturated: **GAP**;
* `abs(G) > 2 F` and unsaturated: **GAP UPPER BOUND**, never a confirmed gap.

The audit's exact `u_is_material` arithmetic is also recorded.  It may say an
unsaturated flag cannot overturn a large multiple, but it does not erase the
fact that the floor is unsaturated.

## 3. Exact bottom-reference / shear split

For each state and target row, call `acc_driver_decomp.section_bt_bc` on the
one-row slice and apply its `_avg` reducer.  This gives

```
total = bottom_reference + shear
```

exactly.  `bottom_reference = u_bottom * H` is what a depth-uniform field equal
to the deepest wet velocity would carry.  It is **not** a true depth mean and
will not be called barotropic without this qualification.  `shear` is the
transport above that reference velocity.

The output records both models, their gaps, every member and horizon, and each
component's own two-sided ensemble floor.  The split is characterization only:
because the two legs sum algebraically to the total, component dominance is not
an independent ownership verdict.  In particular, no directional claim is
issued when opposite-signed components make the absolute-margin restate the
total gap.

## 4. Thickness-weighted depth structure

The level contribution is the row transport integrand retained at each of the
36 levels:

```
Q[k] = mean_i(u[j,i,k] * wet[j,i,k] * e3t_1d[k] * e2u[j] / 1e6)
```

over the same scored longitudes.  Therefore `sum_k Q[k]` equals the exact row
total.  The artifact stores every level for both models, member and horizon.
It also reports the signed and gross-absolute gap shares in the audit's already
committed depth classes: upper <200 m, interior 200--1400 m, abyss >=1400 m.
The gross-depth centroid is

```
sum_k(abs(delta Q[k]) * gdept1d[k]) / sum_k(abs(delta Q[k]))
```

and is UNMEASURABLE if the denominator is zero.  No layer average appears
anywhere; a synthetic surface-only profile must distinguish the official
thickness-weighted reduction from an equal-layer mean.

## 5. Zonal structure and blocked columns

The per-longitude contribution divides each scored-column depth integral by
the number of scored longitudes, so its sum equals the row total.  A U column is
`blocked` when no level is wet.  Among open scored columns, `near_blocked`
means cyclic distance one on the 52-column ring from a blocked U column;
`open_interior` is the remainder.  The artifact records every column's wet
level count and contribution, blocked column indices, contiguous open segments,
and signed plus gross-absolute gap carried by `near_blocked` and
`open_interior`.  Blocked columns must contribute exactly zero.

These are descriptive partitions.  Concentration near a block does not by
itself attribute the gap to bathymetry or a wall operator.

## 6. Correlation construction for attribution

One annual cycle cannot distinguish seasonal phase from elapsed-time
evolution.  Every attribution below therefore means **trajectory
relationship**, not mechanism or seasonality.

For a comparison region `S`, form its fixed day-360 ensemble-mean row-gap
template `M_S`.  The target row is excluded from `S`.  For each paired member
and horizon, project the complete row-gap profile onto that template:

```
A_S[m,t] = dot(G[m,t,S], M_S) / dot(M_S, M_S)
```

The denominator is printed and a zero template is UNMEASURABLE.  The channel
reference uses all 35 pinned channel rows.  The basin reference uses rows 0--12
for target 13 and rows 50--78 for target 49.  This construction follows the
channel profile's changing phase without reducing it to its cancellation-prone
band sum.

For each member, correlate the target's four-horizon trajectory with the basin
projection and with the channel projection using the pinned signed Pearson
implementation.  Also correlate it with the immediate basin-side and
channel-side neighbour trajectories.  The score is the median absolute
correlation over the four paired members; signed correlations are retained and
printed.  Absolute correlation is primary because an edge lobe can be the
opposite sign of the mode it bounds.

Fixed bars:

* `R_HIGH = 0.70`: strong phase relationship;
* `R_LOW = 0.30`: weak phase relationship;
* `R_NEIGHBOUR = 0.50`: required local support;
* `R_MARGIN = 0.20`: required separation between the basin and channel profile
  scores;
* member robustness: at least 3 of 4 member profile correlations must have
  absolute value >=0.50.

What confirms each attribution:

* **CONFIRM basin-edge relationship** only if the basin profile score is
  >=0.70, the basin-neighbour score is >=0.50, at least 3/4 basin profile
  member correlations clear 0.50, and the basin profile score exceeds the
  channel profile score by >=0.20.
* **CONFIRM channel-edge relationship** only under the symmetric channel
  conditions and the same >=0.20 separation.
* **CONFIRM own object / neither relationship** only if both profile scores and
  both immediate-neighbour scores are <=0.30.
* If both relationships are high, the separation bar fails, member support
  fails, or any value lies between the bars, the verdict is **UNRESOLVED/MIXED**.
  There is no majority vote and no post-hoc threshold.

The same rules are applied independently to rows 13 and 49.  Row 49 is not
allowed a looser control bar.

## 7. Row-49 symmetry control

The symmetry comparison is across the matched topology only; it does not claim
that P1 and P3 have symmetric dynamics.  At day 360, define

```
amplitude_ratio = median_m(abs(G[m,360,49]))
                  / median_m(abs(G[m,360,13]))
```

and correlate the two rows' four-horizon trajectories member by member.

* **CONFIRM topology-symmetric response:** amplitude ratio in [0.75, 1.25],
  median absolute trajectory correlation >=0.70, and >=3/4 members clear 0.50.
* **CONFIRM row-13 amplitude asymmetry:** amplitude ratio <=0.50, row 13 clears
  2 of its own floors on the control member, and row 49 is within 2 of its own
  floors.
* Otherwise **UNRESOLVED**.

A symmetric response vetoes any statement that the magnitude is unique to the
southern boundary.  It does not override the independently scored basin versus
channel trajectory relationships.

## 8. Final verdict rule

Each row receives: its total-statistic verdict with exact floor and saturation
flag; one trajectory-attribution verdict from section 6; and, for the pair, the
symmetry verdict from section 7.  Full depth, zonal and component structures are
reported without promoting them to owners.

No row is called a confirmed physical gap while its own floor is unsaturated.
No relationship is called causal.  If a required statistic is unmeasurable, the
affected verdict is withheld rather than assigned zero.

## 9. Controls and provenance

The probe runs the regional audit and channel-rescore self-tests, then its own
controls.  Every new guard is shown to fire on a planted violation:

1. reducer SHA-256 mismatch;
2. total equals the sum over depth and over zonal contributions;
3. bottom-reference plus shear equals total;
4. equal-layer averaging differs from thickness weighting on a surface-only
   field;
5. blocked columns contribute zero and the three zonal classes partition the
   scored longitudes;
6. row-specific floors cannot be replaced by one global floor;
7. synthetic basin-owned, channel-owned, neither and mixed trajectories yield
   four distinct registered attribution outcomes;
8. a planted symmetric-control amplitude outside [0.75,1.25] cannot pass the
   symmetry verdict.

Before loading, the probe verifies that `legoesm`, `legoesm.ocean`, and the
probe itself resolve inside the current worktree.  It refuses a dirty producer
and a tree that moves while running.  The artifact records producer SHA and
cleanliness; both imported file hashes and source commits; command flags; the
exact scored paths and days; legoESM launch SHA; every candidate's
`control_dtype`, `nemo_ladder_mode`, `seasonal_t0_seconds` and start-mode stamp;
the NEMO member directories; oracle restart path and recovered elapsed clock;
legacy-clock environment before and after the audited compatibility path; and
the exact effective data-path strings used by the loaders.
