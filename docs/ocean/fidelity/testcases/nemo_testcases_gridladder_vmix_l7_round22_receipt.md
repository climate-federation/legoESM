# GRIDLADDER-VMIX L7 round 22 receipt — ORCA2 river-mouth mixing

Status: **L7 CERTIFIED — INERT IN ORCA2, BY IDENTITY**.  The operator's
acquisition completed its smoke, boundary, and three twins, but admission
correctly refused the malformed L7 observer.  That recorder failure does not
affect the result: an exhaustive census proves every comparable NEMO L7 output
identical to V5, a direct OFF/ON restart comparison proves the selector inert,
and the production card's L7 and V5 trajectories are bit-identical on the real
input at one and ten steps.  V5 is already certified.  Codex did not build or
run NEMO during this resume.

## Oracle selection and why it is inert

The ORCA2 deck selects `ln_rnf_mouth=T`, `rn_hrnf=15 m`,
`rn_avt_rnf=1e-3 m2/s`, and `sn_cnf=runoff_core_monthly/socoefr0` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:159-171`.  NEMO's initializer computes
`nkrnf`, then calls the river-mouth mask reader inside `sbc_rnf_init` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:289-526`.  `rnf_mouth` reads the
configured `sn_cnf%clvar` into `rnfmsk` and optionally adds closed-sea inflow
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:541-588`.  The admitted
`socoefr0` has 0 nonzero values among 26,640 cells, and ORCA2 sets
`ln_closea=F` at `ORCA2_OMIP_L4/EXP00/namelist_cfg:36`; the effective mask is
therefore exactly zero.

Within the enclosing `zdf_phy` routine, NEMO copies the closure coefficients,
adds `2 * rn_avt_rnf * rnfmsk * wmask` to heat diffusivity only, then runs EVD,
double diffusion, and internal-wave mixing at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:296-372`.  This pins the term's form,
channel, and ordering.  It also explains why the selected term contributes
exactly zero in this ORCA2 configuration.

## Exhaustive NEMO identity census

The committed fail-closed census compared round 22 L7 with round 15 V5 for
the smoke, all three twins, and the writer-free/reference run.  Inventories
must match before content is read.  Raw binary files are streamed; netCDF
comparison checks every dimension, variable type and dimension list, variable
attribute, and value word, plus global attributes other than the run timestamp.

| run pair | shared standard records, raw-identical | restarts, raw-identical | history payloads, bit-identical | mesh / abort, raw-identical |
|---|---:|---:|---:|---:|
| smoke | 45 | 2 (ocean 1, ice 1) | 4 | 1 / 0 |
| twin A | 108 | 2 (ocean 1, ice 1) | 4 | 1 / 0 |
| twin B | 108 | 6 (ocean 5, ice 1) | 4 | 1 / 0 |
| twin C | 108 | 5 (ocean 4, ice 1) | 4 | 1 / 0 |
| boundary/reference | n/a | 10 ocean | 4 | 1 / 2 |
| **total** | **369** | **25 (13 in twins)** | **20** | **5 / 2** |

The 369 standard records cover every shared family: barotropic frames,
substeps, ordered/advective/drag/energy operands; step entry; RK stages and
stage operands; transports and tracer transports; RHS; slow forcing; ZDF
entry; surface input; shortwave/chlorophyll records; and SI3 dynamics,
thermodynamics, exchange, reassociation, Prather, and ZDF inputs.  All 42
variables in every ocean restart match bit-for-bit; this includes the
operator's highlighted twin-A step-10 pair.  Ice restarts contain 208 or 210
matching variables.  Each history set has 27 T-grid, 6 U-grid, 6 V-grid, and
9 W-grid variables with identical payload words.

The files that are not directly comparable are explicit:

- Four L7-only observer records (smoke and twins A/B/C) are malformed.  Their
  header says 184 x 152 x 31 = 867,008 values per field, but the fourth field
  contains the interior 180 x 148 x 31 = 825,840 values.  Each file is
  27,415,020 bytes rather than 27,744,364.  Re-reading the first interior-sized
  portion reproduces 31,143 false river-minus-closure differences and minimum
  -930,748.7023730353.  They cannot support a physics result.
- The 32 `oracle_v1_hm_i` records are V5-only historical observers; the L7
  source card has no counterpart.
- History containers differ only in the global `TimeStamp`; their complete
  scientific payloads and all other checked metadata match bit-for-bit.
- `nemo`, `namelist_cfg`, `ocean.output`, `output.namelist.dyn`,
  `run.user.log`, `timing.output`, and `timing_step.nc` identify different
  producers/decks or contain log/performance metadata, so they are not
  scientific-output identity targets.

One operator description needs correction.  The archived V5 reference and
V5 twin decks both set `ln_rnf_mouth=T`; they are not an OFF/ON pair.  The
round-22 L7 writer-free boundary deck is genuinely OFF and its twin-B deck is
ON.  Their ocean restarts at steps 2, 4, 6, 8, and 10 are raw-byte identical.
The same five OFF L7 boundary restarts are also raw-byte identical to the ON
V5 reference restarts.  Thus the evidence contains a direct measured selector
identity despite the V5-label deck nuance.

## Production-card identity and certification chain

Using the admitted V5 record forcing and `hm_i` inputs, the production model
was JIT-run twice: once as V5 with the L7 coordinate flags removed and once as
L7 with the real ORCA2 mask.  At both step 1 and step 10, every `T`, `S`, `u`,
`v`, and `eta` word is identical when viewed as binary64.  This is an actual
one-step and ten-step trajectory, not a coefficient-only surrogate.

The certification chain is therefore closed:

1. NEMO L7 and NEMO V5 have identical comparable outputs, and the genuine
   NEMO OFF/ON restart pairs are identical.
2. Our L7 and our certified V5 are bit-identical on the real inputs at steps
   1 and 10.
3. Certified V5 already matches NEMO bit-for-bit through step 10.

Consequently our L7 matches NEMO's L7 on the real ORCA2 configuration.  The
nonzero synthetic-mask controls remain required: they show that the term acts
on the expected W levels, affects heat diffusivity only, is positioned between
the closure and EVD, has finite gradients, and fails deletion and reordering
plants.  They certify the implementation's form for configurations where the
mask is nonzero; they are not presented as evidence that ORCA2's mask acts.

The round-314 audit row that counted river-mouth mixing as a rung-10 gap is
therefore reclassified: for the admitted ORCA2 input it is inert, not an
outstanding fidelity gap.

## Recorder repair and future-run guard

The observer canonicalizer now accepts the actual NEMO interior bounds but
returns a full `jpi x jpj x jpk` array, initialized to zero, and copies only
wet interior cells.  Closure, river, EVD/DDM, and IWM fields therefore share
the declared full shape with zero halo/dry cells.

Preflight no longer trusts a synthetic Python writer alone.  It generates,
compiles, and runs a minimal gfortran program containing the actual observer
write sequence, verifies that the production reader accepts the complete
record, then runs a planted version whose fourth field is short and requires
the reader to reject it.  The transformed real NEMO source also compiles in
syntax-only mode.  The clean-tree read-only preflight reported:

`L7_PREFLIGHT_OK checklist=7 plants=all_refused commit=523bea8f0b740a473eadeccd15e39e8e8fc1d6dd`

No NEMO rerun is needed: the acquired restarts and standard records are valid,
and only the diagnostic observer was malformed.  The repair makes a future
nonzero-mask acquisition record conforming.

## Landing gates

All Python/JAX tests use CPU JAX, `JAX_ENABLE_X64=1`, and the exact `land.sh`
`PYTHONPATH`.

- Focused L7 tests: **PENDING**.
- Heading-scoped receipt citation gate: **PENDING**.
- Exact default landing gate: **PENDING**.
- Independent private-import ratchet: **PENDING**.

independent review: operator (Claude)

codex review: operator

## Machine lines

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r22/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: L7_CERTIFIED — inert in ORCA2, certified by exhaustive NEMO L7/V5 identity, direct NEMO OFF/ON restart identity, production-card V5/L7 identity at steps 1 and 10, and certified V5; observer repaired for future nonzero-mask runs; codex review: operator
