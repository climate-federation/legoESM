# NEMO testcase fidelity receipt — lane 3b, scalar-math oracle V2

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
Recovery git: `/tmp/codex-si3thd-localgit`  
Preregistration commit: `5f85456c405`  
Implementation/evidence commit: `4934bb6be91`
Both-arch/metadata classification follow-up: `e949c2bb176`  
Numeric-runtime and real-vectorized-binary control follow-up: `f0bb50d9f8e9`  
Executed-input and V2-default-root follow-up: `044d29937f4a`

## Outcome

The two independently built C1D_OMIP_L3 scalar-math oracles are
**CONFIRMED / REPRODUCIBLE**.  Both completed the documented 8,760 hourly
steps on CPU without an MPI launcher; both executables contain zero `_ZGV*`
dynamic symbols.  All eight registered SI3 streams, the ASCII ice diagnostic,
both restarts, and `ocean.output` are byte-identical between V2 rebuilds and to
oracle V1.  The three annual NetCDF files differ only in their global
`TimeStamp` attribute; every NetCDF variable payload is bit-identical across
V1, V2 A, and V2 B.

This **REFUTES** the preregistered prediction that the scalar-math build would
change the bulk, exchange, thermodynamics, ZDF, or restart streams.  Therefore
there is no first differing scientific frame/field and no NEMO transcendental
call to assign as a V1-to-V2 owner.  The V1 executable does import `_ZGV*`
libmvec symbols, while V2 does not; for this forcing/case that binary change
does not change any registered scientific bit.

Rung 3.5b remains **AT-BAR, not bit-identical**.  All 227,760 registered bulk
rows satisfy the pointwise `1e-15` normalized bar, while 16,512 rows differ in
binary64 representation.  Eighteen are owned by JAX `exp` in the active
`ice_alb` path and are **AWAITING_LIBM_POLICY**; a gate-only scalar-glibc replay
is bit-identical to NEMO for all three affected outputs.  The other 16,494 are
ordinary multiplication/division/summation-order differences and remain
disclosed at bar.

Those bit counts are runtime-specific evidence, not a version-independent JAX
claim.  The accepting runtime is Python `3.13.0`, JAX `0.10.0`, jaxlib
`0.10.0`, and NumPy `2.4.4`; the committed gate stamps all four versions and
fails closed on any different tuple.  An independent review found different
bar/bit results under a newer JAX runtime, which is why an unstamped runtime is
not accepted or silently compared here.

The closed C1D thermodynamic column remains **CLOSED AS MIXED DEBT**.  Its V2
scientific gate payload is exactly equal to phase 6 after removing only the new
oracle-version/root provenance stamp.  No debt row moved, and none is explained
by vectorized math.

## Search and reuse before implementation

The search reused rather than duplicated:

- `nemo_si3_exchange_drift_gate.py` for the source-defined 36-field exchange
  layout and byte mapping;
- `nemo_si3_bulk_flux_gate.py` for all three bulk frame schemas and the existing
  legoESM implementation;
- `nemo_si3thd_phase2_gate.py` and
  `nemo_si3thd_phase2b_year_gate.py` for thermodynamics/ZDF parsing and the
  closed full-year score;
- `packages/ice/legoesm/ice/c1d_omip_l3.py` for the immutable column card; and
- the existing Goff-ice helper in `legoesm.thermo` and existing bulk/ice
  dispatchers.  No lane-local exp/tanh implementation and no second bulk or ice
  model were added.

The retained phase-2 compact ZDF gate is now explicitly pinned to its V1 root
and V1 schema hashes.  New column-card calls default to V2.  This preserves the
historical measurement instead of interpreting V2's later expanded operand
schema with the old reader.

## Oracle construction and immutable provenance

Nothing in `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` was modified.  Its
arch file was read and copied into two new retained source trees:

- `/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3bulk_scalarmath_a_src`
- `/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3bulk_scalarmath_b_src`

Each tree received a new `cfgs/C1D_OMIP_L3_SM`.  The shipped `C1D` reference
created the config; then V1's config-local `MY_SRC`, actual executed/resolved
EXP deck, and cpp-key content were copied verbatim.  The committed gate checks
all nine `MY_SRC` files, thirteen config `EXP00` inputs, all thirteen files in
each actual run root, each run root's ERA5 member, and the cpp file against V1
by SHA-256 before accepting V2.

- Creation: `makenemo -r C1D -n C1D_OMIP_L3_SM -m conda-scalarmath -j 0 -y`.
- Build: `makenemo -n C1D_OMIP_L3_SM -d 'OCE SAS ICE' -m
  conda-scalarmath -j 8 -y`.
- FC flags: `-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer
  -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
- Arch SHA-256:
  `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`.
- V2 A executable SHA-256:
  `1bb0f4c8f844b8d5755c623c18b076cda1929a5f75ab5d56e1edcc1ee10defe4`.
- V2 B executable SHA-256:
  `b52e289890126ec436d245b0e8dd80ccccf4c8c4a08fc4bb55077ad4bb307289`.
- `nm -D` `_ZGV*` count: A `0`, B `0`.

Absolute build roots make the two executable hashes different; the scientific
products are compared independently.  The two new run roots are
`c1d_omip_l3_sasice_scalarmath_v2_a` and
`c1d_omip_l3_sasice_scalarmath_v2_b` under
`/data/abyssal/dbalwada/nemo-testcases-l3/`.  Each was run with
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, and `./nemo.exe`; no `mpirun`
and no GPU were used.  Each `time.step` is `8760`, and each log closes the
step-8760 ocean and ice restarts.

Key verbatim input pins are:

| input | hash |
|---|---|
| `cpp_C1D_OMIP_L3_SM.fcm` | SHA-256 `b1ca9f56ec13250dda1d65a17ddcefcf54fd5ffdad0cf30789e38ced35c2bd72` |
| resolved `namelist_cfg` | SHA-256 `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c` |
| resolved `namelist_ice_cfg` | SHA-256 `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033` |
| `MY_SRC/icestp.F90` | SHA-256 `c2384d4d4ff96c4aa9c880ca7de07b7180355e361e2c8584f59fba84e53ec982` |
| `MY_SRC/sbcblk.F90` | SHA-256 `087e551a49e5bb0aefe93e80f1bca817365b1f68d84dc59b4178eff87c010cea` |
| `MY_SRC/icethd.F90` | SHA-256 `3afb13b7612f09b5cdb1322a55186a53f77891ff5016d3859ec9e2da58cc5490` |
| `MY_SRC/icethd_dh.F90` | SHA-256 `fb4f9f22aee0ee15a6accd2172bbb16b79f8fd958b07aa684cd8c770bcb99a2d` |
| `MY_SRC/icethd_zdf_bl99.F90` | SHA-256 `b94881987515fbc5ecde3f24f331276e56a7ba884e5a96e0c40fe2e324f55bab` |

The gate JSON contains the complete `MY_SRC` and EXP manifests.  The official
NEMO sette-input archive was rechecked: MD5
`9456e6a0a84d40630ad1804fd4061caf`, SHA-256
`54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382`.
The actual ERA5 member remains SHA-256
`e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe`.
No synthetic input was used.

## V1/V2 byte audit

| registered stream | bytes | V1 vs V2 | V2 A vs B | V2 SHA-256 |
|---|---:|---|---|---|
| bulk operands | 8,374,560 | IDENTICAL | IDENTICAL | `57868f3212646bdf6b0c4add0f48701c0082718331bc76a153050c6d1d44dfe9` |
| DH operands | 4,752 | IDENTICAL | IDENTICAL | `9efbcb9113c2748f9085c519092848596daf0f573ba8a4625d02ee3c08bc8b14` |
| DH-remap operands | 552 | IDENTICAL | IDENTICAL | `8fbeb7df70b3c66b4e7acdd7ab0df3ed8fc01bfaa6150dbc47e3b444638b40a5` |
| exchange frames | 18,571,200 | IDENTICAL | IDENTICAL | `091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4` |
| reassociation operands | 192 | IDENTICAL | IDENTICAL | `4b832b0c274d6aab032f16958224ebfb6fea603af22e1a1f1d474ff71b1e4589` |
| thermodynamics frames | 103,368,000 | IDENTICAL | IDENTICAL | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| ZDF inputs | 1,471,680 | IDENTICAL | IDENTICAL | `cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd` |
| ZDF operands | 4,600 | IDENTICAL | IDENTICAL | `aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442` |

Additional exact products are `ocean.output`
`3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430`,
ocean restart
`84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265`,
and ice restart
`b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e`.
The annual T/U/V NetCDF bytes differ beginning in header metadata, but a
variable-by-variable dtype/shape/payload comparison finds zero differing
variables and only the global `TimeStamp` attribute differs.

The active candidate paths were registered before comparison: Goff-ice
`LOG10`/real power at `sbc_phy.F90:665-679,693-711,727-790`; albedo `LOG`/`EXP`
at `icealb.F90:124-185`; BL99 radiation/conductivity `EXP`/`LOG` at the
config-local `icethd_zdf_bl99.F90:217,228-230,314-315`; and frazil `TANH` at
`icethd_do.F90:407`.  Because every registered scientific stream is identical,
none is promoted from candidate path to measured V1/V2 owner.

## Rung 3.5b bit score against V2

The existing bulk gate runs with NumPy `float64`, JAX x64 enabled, backend
`cpu`, Python `3.13.0`, JAX `0.10.0`, jaxlib `0.10.0`, NumPy `2.4.4`, and the
same 8,760-step coverage register.  Its normalized result is unchanged:
227,760 comparisons, zero over-bar rows, largest normalized error
`2.220446049250313e-16` at kt861 `POST_BLK_ICE_1.utau_ice`.  This exact runtime
tuple is part of the measurement provenance; the gate and a planted runtime
control fail closed if any member differs.

| measured first-owner group | non-bit rows | largest conventional relative error | largest row | status |
|---|---:|---:|---|---|
| JAX `exp`, active `ice_alb` and propagated solar outputs | 18 | `8.4104602317484995e-16` | kt6320 `POST_BLK_ICE_2.qsr_ice` | AWAITING_LIBM_POLICY |
| ordinary binary64 operation order; no transcendental owner | 16,494 | `2.008964945929631e-13` | kt6236 `POST_BLK_ICE_2.emp_ice` | DISCLOSED_AT_BAR_REASSOCIATION |

The first group comprises six rows each in `albedo`, `qsr_ice`, and `qsr_tot`.
A gate-only written-order replay using scalar `math.exp`/`math.log` is
bit-identical to all 8,760 NEMO rows for each of those three outputs; it closes
all 18 JAX differences.  This is the measured basis for
**AWAITING_LIBM_POLICY**.  No non-bit row is assigned to Goff real power/log or
to `tanh`.  The shared library-exact policy is external GYRE-lane work and was
not recreated here.

The ordinary-arithmetic group contains `utau_ice`, `vtau_ice`, `evap_ice`,
`devap_ice`, `emp_ice`, `emp_tot`, and `fhld`.  Its larger conventional
relative number occurs on a small nonzero `emp_ice`; every row still passes the
gate's separately defined denominator-one normalized bar.  “AT-BAR” is not
reported as bit identity.

## Closed column re-score against V2

The V2 year-gate artifact carries the full 8,760-step per-step trajectory and
70,080 boundary frames.  Its CLI exits 1 by design because the retained verdict
is DEBT; stderr is empty.  The complete scientific object equals the phase-6
object exactly after removing only `card.oracle_version` and
`card.oracle_root`.

| retained measurement | V1 | V2 | movement / status |
|---|---:|---:|---|
| exact-entry over-bar field rows | 35,282 | 35,282 | none; DEBT |
| first independent injection above `1e-12` | kt4242 `POST_DH.h_i`, `2.8315499258551526e-11` | same | none; unresolved independent injection |
| first injection above `1e-3` | kt5842 `POST_ZDF.e_s`, `0.012837520586235438` | same | none; positive-subnormal snow |
| largest positive-subnormal-snow row | kt5860 `POST_ZDF.e_s`, `0.02510099530281747` | same | none; unresolved |
| year-end `t_su` normalized error | `4.654045553508542e-6` | same | none; MIXED DEBT |
| year-end `e_i` normalized error | `1.1466756156615379e-4` | same | none; MIXED DEBT |
| year-end `h_i` normalized error | `1.045384260731163e-4` | same | none; MIXED DEBT |

The kt5860 absolute numerator is `2832427.961354971 J m-3` over the NEMO
denominator `112841260.96135496 J m-3`.  Since V1 and V2 oracle inputs and
outputs are bit-identical at every registered boundary, **none of these debts
is explained by NEMO vectorized math**.

The six measured phenomenology rows also do not move:

| quantity | NEMO V2 | legoESM fp64 | distance | retained classification |
|---|---|---|---:|---|
| minimum thickness | `0.5550952376958682 m` | `0.555144003761985 m` | `4.876606611681211e-5 m` | AT-FLOOR |
| maximum thickness | `2.445688622638321 m` | `2.445688622637688 m` | `6.328271240363392e-13 m` | AT-FLOOR |
| minimum date | `2018-09-09T00:00Z` | same | `0 h` | AT-FLOOR |
| maximum date | `2018-05-13T00:00Z` | same | `0 h` | AT-FLOOR |
| melt onset | `2018-05-14` (day 133) | same | `0 d` | AT-FLOOR |
| growth onset | `2018-09-09` (day 251) | same | `0 d` | UNMEASURED fp32 floor; user-closed phenomenology agreement |

This receipt does not upgrade the column verdict: rung 3.5 remains closed by
the user's decision, with the listed MIXED DEBT retained.

## Evidence artifacts, controls, and tests

| committed artifact | SHA-256 |
|---|---|
| scalar-math V2 provenance/byte gate JSON | `9815ee9141d69b80c61cb54b07c9e6d031591914a5d696d3a99aec7326bee030` |
| bulk V2 bit/bar gate JSON | `12784cbf13f77a993aa7f11c5dfab08cfbfe97d7db3236c7a544512a6e1178ed` |
| column V2 year-gate JSON | `91956787dcabc1080e352b0a2920c13f33a0c8794dabb9cba3261d9e2f85e5e1` |
| scalar-math V2 preregistration | `5b2d355a5cdd8de28a2498657c7a609837747597981328567874be4c2636da83` |

All seven provenance plants exit 1: payload bit change, missing stream
inventory, selecting the retained vectorized V1 executable (whose five real
`_ZGV*` imports are detected), source drift, V1 executable provenance
substituted as V2, run-root EXP drift, and run-root forcing drift.  All nine
bulk plants exit 1: `blk_ice_1`, `ice_alb`, `blk_ice_2`, `ice_flx_other`, stream
hash, coverage, selector, incomplete bit-owner register, and numeric-runtime
drift.  Plants operate in memory or on temporary files; no retained oracle
artifact is mutated.

- Final focused suite: `51 passed in 80.78s`.
- Constants ratchet on every touched SI3/card/gate/test file: `7 passed, 3384
  deselected in 0.69s`.
- Repository-wide constants ratchet: `3384 passed, 2 skipped, 5 failed`; the
  five failures are the same untouched pre-existing FV3/grid/DINO constant
  sites recorded by the prior phase-6 receipt.  No file touched here fails.
- New scripts compile with `py_compile`; the new card, scalar gate, and scalar
  gate test pass Ruff.  Remaining Ruff findings in older touched gate files are
  pre-existing style findings, not suppressed or relabeled as a clean full
  Ruff run.

Two independent adversarial reviews returned **SHIP** after their HOLD items
were corrected.  The provenance reviewer independently reproduced the scalar
JSON hash and all seven red controls, inspected both builds/runs and their
executed inputs, and confirmed the metadata-only NetCDF classification.  The
gate reviewer independently reproduced the 227,760-row bulk census, its
bit-owner split, runtime fail-closure, the V2 column identity, and the 44 core
plus 7 exchange tests.  Neither review found rung-3.6 implementation.

## Decisions

| choice | state | disposition |
|---|---|---|
| Rebuild C1D with math-call vectorization off | ASKED | Two fresh `_SM` builds/runs completed and pinned. |
| Use the supplied `arch-conda-scalarmath.fcm` | ASKED | Exact hash/flags above; zero `_ZGV*` symbols. |
| Preserve V1 and use new directories | ASKED | V1 and both V2 roots remain retained. |
| Compare every stream and report first differing owner | ASKED | All streams are identical, so no scientific first difference exists. |
| Add a per-row bulk bit score beside the normalized bar | ASKED | 16,512 non-bit rows grouped above; bar unchanged. |
| Wait for the shared library-exact exp/tanh policy | ASKED | 18 exp-owned rows labeled AWAITING_LIBM_POLICY. |
| Re-score and re-pin the closed column | ASKED | Scientific result unchanged; MIXED DEBT retained. |
| Update the exchange drift ticket for V2/two builds | ASKED | Hash remains reproducible `091395cf...`. |
| Bind bulk bit counts to the exact measured JAX runtime | UNASKED | Post-review provenance guard only; fails closed rather than changing physics. |
| Implement rung 3.6 | UNASKED | Explicitly forbidden this round; no rung-3.6 code changed. |
| Add a lane-local exp/tanh implementation | UNASKED | Forbidden; no implementation added. |
| Change the ORCA1-resolved physics scope | UNASKED | No selector or physics identity changed. |
| Modify shipped NEMO files/configs | UNASKED | Forbidden; none modified. |
| Delete historical roots or artifacts | UNASKED | Forbidden; none deleted. |
| Push the branch | UNASKED | Forbidden; no push performed. |

## Flagged for future deletion

Nothing was deleted.  Oracle V1, both V2 source/build trees, both V2 run roots,
all prior roots, and the timestamp-distinct annual NetCDF containers remain
retained evidence.  Superseded V1 pins remain named in the column-card module
for the historical compact-schema phase-2 gate rather than being removed.
