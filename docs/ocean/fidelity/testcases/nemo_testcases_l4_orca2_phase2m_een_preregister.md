# NEMO testcase Lane 4 — ORCA2 Phase-2m EEN discriminator preregistration

Date: 2026-09-06

Parent: `ce0f353e25a282b80ce4d134d1f937af24d1ed5e`

Status: **PREREGISTERED BEFORE ACQUISITION.**  This boundary extends only the
configuration-local ORCA2 oracle instrument.  It does not change a NEMO model
array, legoESM arithmetic, the icebergs-off variant, or any shipped NEMO file.
The sandbox will build and stage two runs but will not execute MPI.

## P2M-A — source boundary and frozen inventory

The GYRE-lane live-divisor arm improved all eight frozen external-mode EEN
coefficients but left every row over bar.  The existing final-coefficient
record therefore cannot distinguish the remaining source association.  The
primitive program executed by the ORCA2 variant is:

1. `dynvor.F90:918-941` constructs `e3f_0vor` from the four masked source
   `e3t_0` terms for resolved `nn_e3f_typ=0`; `:943` applies the F-point
   lateral exchange/fold and `:944-950` replaces zeros with `e3f_0`.
2. `domqco.F90:233-246` constructs the live RK3 F-point ratio `r3f` from the
   area-weighted four-T-cell SSH expression.  The executing QCO substitution
   in `domzgr_substitute.h90:125-130` makes
   `e3f_vor=e3f_0vor*(1+r3f*fe3mask)`.
3. `dynspg_ts.F90:1326-1379` selects `np_EEN`, evaluates the individual
   `q=ff_f/e3f_vor` divisions, forms four U and four V `zpvo` triads, and
   accumulates the frozen coefficients.  These are the source sites intended
   by the campaign request for the `q`/`zpvo` discriminator; `dynvor.F90` owns
   the divisor construction, while `dynspg_ts.F90` owns these actual external
   EEN consumers.

At `kt=1`, `Kmm=1`, immediately inside the executing
`dyn_cor_2D_init`, the rank-zero writer will add exactly four files:

| stream | magic | ordered binary64 payload | values |
|---|---|---|---:|
| `oracle_een_e3f0vor_kt00000001.bin` | `NEMO_L4_E3F0_1` | `e3f_0vor` | `jpi*jpj*jpk` |
| `oracle_een_e3fvor_kt00000001.bin` | `NEMO_L4_E3FV_1` | live `e3f_vor` | `jpi*jpj*jpk` |
| `oracle_een_q_kt00000001.bin` | `NEMO_L4_QEEN_1` | `ff_f/e3f_vor` | `jpi*jpj*jpk` |
| `oracle_een_zpvo_kt00000001.bin` | `NEMO_L4_ZPVO_1` | U `nw,ne,sw,se`, then V `nw,ne,sw,se` | `8*jpi*jpj*jpk` |

Each stream begins with a 16-byte blank-padded magic and ten native 32-bit
integers: `(version=1, kt, Kmm, nvor_scheme=3, jpi, jpj, jpk, nfields,
storage_bits=64, payload_values)`.  Payload arrays retain Fortran column-major
order.

Every diagnostic temporary is allocated only on rank zero while the `kt=1`
arm is live, zeroed before any copy, and deallocated after the four writes.
Owned rank-zero horizontal cells are populated; halo bands remain canonical
zero.  `e3f_0vor` is defined through `jpk`.  Live `e3f_vor` and `q` are
populated only through `jpkm1`; their unused `jpk` slot stays zero.  Each
`zpvo` field is populated only where the corresponding U or V loop executes
and only through that face's `mbku` or `mbkv`; halos, dry/no-column cells, and
levels below the face bottom stay zero.  Land-adjacent owned `q` values are
retained because a neighboring wet EEN triad can consume them; they are not
discarded merely because an F mask is zero.

The instrument may assign only writer-local temporaries and I/O metadata.  It
must not assign `e3f_0vor`, `r3f`, `e3f_vor`, `ff_f`, any `zpvo` scalar before
its ordinary use, any final EEN coefficient, or any prognostic/model array.

## P2M-B — twin and inertness admission

Two independent 10-step icebergs-off variant runs, byte-identical in recipe,
will be executed one at a time with the same two-rank `(jpni,jpnj)=(2,1)` CPU
layout.  Admission requires:

- both runs finish at step 10 with `MPIRUN_RC=0` and write all four active
  restart shards (two ocean and two SI3; TOP is compiled out);
- the four new streams pass magic/header/count/finite/canonical-slot checks;
- each of the four new streams is raw-byte identical between twins;
- every inherited V2 oracle stream is raw-byte identical between twins; and
- ordinary outputs and all six restart shards are byte-identical to the
  Phase-2j V2 control under the standing exclusion for launcher/timing text
  and NetCDF history timestamps.

The additional admission against the pre-extension Phase-2j V2 root follows
the GYRE Round-21 rule: raw identity is mandatory unless the old record schema
contains a source-undefined slot.  In this ORCA2 V2 inventory, the systematic
canonicalization already removed undefined halo, land, inactive-component,
and below-bottom bytes from all 93 inherited streams.  Therefore **all 93 are
raw-identity rows**.  There is no inherited consumed-field exception.  For
comparison, the GYRE exception was only the pre-`tra_adv_trp`
`oracle_transport_*` `zFw` slot; the ORCA2 V2 transport writers canonicalize
that slot and the WZV extension records the post-consumer field separately.

The schema gate will carry three binding plants, each required to exit
nonzero through the production checker: one header integer, one defined
payload bit, and one canonical-zero slot.  The inherited raw-identity gate's
existing one-byte plant remains mandatory.

## P2M-C — discriminator after acquisition

After the user-shell twins pass admission, Lane 4 will pin twin A as the V2
extension and the GYRE lane will walk, in order, `e3f_0vor`, live `e3f_vor`,
individual `q`, `zpvo` triads, and only then the eight final coefficients.
Each scored row is cellwise binary64 equality on defined rank-zero cells with
an explicit `0 / n` target.  The first non-bit primitive owns the handoff;
later rows remain unclaimed behind it.  No shared external-mode repair is
authorized in Lane 4.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| four EEN primitive dumps | ASKED | frozen above; config-local WRITE-only |
| canonical rank-zero/twin convention | ASKED | zero-first; two independent runs must match raw |
| user-shell MPI execution | ASKED | run directories are prepared; sandbox does not launch |
| shared EEN physics repair | UNASKED and forbidden | GYRE owner after acquisition |
| tracer/BBL/ZDF continuation | ASKED after this boundary | deferred until the acquisition returns |
| read `/tmp/codex-orca2-r21` | explicitly forbidden | not read; only committed GYRE evidence was consulted |
| shipped NEMO edits or file deletion | UNASKED and forbidden | none |
