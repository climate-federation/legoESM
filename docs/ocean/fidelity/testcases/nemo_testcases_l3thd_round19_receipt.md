# Lane 3b round 19 receipt — integration reconciliation and cross-card gates

Status: **MIXED**.  Round-19 predictions were frozen in
`nemo_testcases_l3thd_round19_preregister.md` at commit `0deb1b8b340`.

## Crash recovery (2026-09-05)

The `/tmp` filesystem filled during the first full fidelity-test run.  On
resume, the worktree and `/tmp/codex-si3thd-localgit` both resolved HEAD to
`0deb1b8b340`; `MERGE_HEAD` remained
`03c6e8d96ff7f69207abdee43ec28e223d083f4e`.  The merge index was intact.
No source, receipt, preregistration, gate JSON, or oracle instrument had an
mtime in the crash interval.  Every locally patched Python file was re-read,
compiled, checked for conflict markers and checked with `git diff --check`.
No file was truncated.  The killed pytest result is discarded and is not
cited.

One pre-existing provenance inconsistency was exposed by the rerun.  Round 13
had changed three V2-A WRITE-only stream header counts from literals to
dimension-derived expressions after the V2 executable was built.  The source
hashes are now pinned and labelled `round13-derived-header-only` by
`nemo_si3_scalarmath_v2_gate.py`; V2-B remains `verbatim-v1`.  The expressions
evaluate to the byte-identical recorded counts, all eight stream hashes remain
unchanged, and the retained executable remains separately hash-bound.  This is
an instrument-source provenance correction, not a new oracle result.

## Integration merge

Merged source: `refs/remotes/origin/fidelity/nemo-testcases-l2-gyre-reconciled`
at `03c6e8d96ff7`; merge base `45db99b404ff`.  The canonical core files
`precision.py`, `transcendentals.py`, and `source_rounding.py` were already
byte-identical and did not conflict.

Resolution rule: retain the integration line's shared ocean numerics and
reapply only lane-3b's C1D card, SI3 exchange fields, one-layer RGB source
adapter, source-literal stress/drag helpers, and gates.  The round-17
barotropic Kmm seed arm was deliberately not reapplied.  NEMO carries
prognostic `uu_b/vv_b` (`dynspg_ts.F90:862,890`; `restart.F90:181,313`) and
only re-derives them as a missing-restart-field fallback
(`restart.F90:316`); the former arm re-derived them every step from the 3-D
velocity and is **SUPERSEDED** pending the integration lane's carried state.

### Conflict ledger

Every conflict hunk produced by the three-way merge is enumerated here.  “I”
means canonical integration code retained; “L3” means a lane-owned interface
was re-applied on top; “drop” means the round-17 re-derived seed behavior was
discarded.

| file | hunk(s) | overlap | resolution |
|---|---:|---|---|
| `barotropic_latlon_cgrid.py` | 1 | vertical/QCO imports | I; canonical QCO operands |
| same | 2–4 | `_nemo_literal_seed_from_card_mesh` API, face depth and return | I; drop re-derived round-17 seed |
| same | 5 | ENE block vs lane statement helpers | I + L3 helpers `nemo_literal_temporal_combination`, `nemo_literal_ssh_forward`, `nemo_flux_form_barotropic_velocity_update` |
| same | 6 | `_run_substep_loop` trace/update API | I; compiled keyed trace is canonical |
| same | 7–10 | AB3/AM4 temporal and SSH associations | I |
| same | 11–12 | U/V external velocity update | I; lane helper retained only for direct statement gate |
| same | 13 | keyed trace/return registry | I |
| `ocean_model_latlon_cgrid.py` | 1 | freshwater import/program | I; C1D uses canonical real-freshwater projection |
| same | 2 | private hook registry | I + L3 round-15/16 observation hooks; round-17 Kmm seed hook dropped |
| same | 3–5 | shared WS-RK3 tracer-stage API and source accumulation | I + L3 raw SI3 `nemo_rk3_surface` rates through the same `stage_source_rates` path |
| same | 6 | WS-RK3 constructibility validator | I + L3 complete `traadv_OFF`/`dynadv_OFF` program |
| same | 7 | quadratic bottom drag arm | I; only non-seed private diagnostic field retained |
| same | 8–10 | stress staggering, wind association, top drag | I + L3 SI3 forcing inputs and literal helper interface |
| same | 11–18 | stage state, QCO geometry, RK3 momentum/tracer ordering | I; canonical integration numerics |
| same | 19–20 | C1D raw surface source vs canonical stage-source rates | I + L3 adapter into canonical rates, no second integrator |
| same | 21 | post-stage diagnostics/return | I + L3 round-15/16 observation fields; Kmm seed drop |
| `ocean_pe_latlon_cgrid.py` | 1 | source-literal stress helper/imports | I + L3 named helper for the exchange gate |
| same | 2 | `surface_stress_faces` masks/native stress | I; canonical native-stress route |
| same | 3 | UP3 selector vs C1D OFF dispatch | I + complete static OFF program (`dynadv.F90:78-90,128-134`) |
| `nemo_testcase_recipe.py` | 1 | `PrecisionPolicy` import | union; required by C1D libm card |
| same | 2 | C1D slab builder vs GYRE builder | union; both cards retained, shared `_model_config` numerics from I |
| same | 3 | exports | union; both card families exported |
| `shortwave_penetration.py` | 1 | source-round/libm imports and NEMO RGB/2BD identities | I + L3 one-wet-layer RGB RHS adapter |
| `state.py` | 1 | native stress fields vs SI3/RK3 forcing fields | union; `tau_i_native/tau_j_native`, `rCdU_top`, `snwice_fmass`, and `NemoRK3SurfaceForcing` retained |

The merge also revealed two stale test interfaces.  The OVERFLOW trace gate
now consumes the canonical compiled `expose_barotropic_substeps` return instead
of retaining JAX tracers in a Python monkeypatch closure.  The ULP comparison
test supplies the now-mandatory durable output sidecar, and the stage gate's
expected control inventory includes the integration line's `faithful_only`
field.  These change harness plumbing only.

## NEMO OFF semantics and deck validation

`dynadv.F90:78-90,128-134` maps `ln_dynadv_OFF` to a complete no-advection
program.  The earlier validator was wrong to reject that legitimate C1D
combination.  The merged cards resolve as follows:

| card | horizontal | flux selector | vertical | status |
|---|---|---|---|---|
| LOCK | `flux_form` | `nemo_up3` | `nemo_up3` | complete NEMO UP3 |
| OVERFLOW | `flux_form` | `nemo_up3` | `nemo_up3` | complete NEMO UP3 |
| C1D slab | `off` | inert | `off` | complete NEMO OFF |

The pairing validator remains and accepts exactly those complete programs.

## Test and gate results

The post-merge ocean-fidelity selection completed as 839 passed and 7 skipped
(816+7 in the broad selection, 9 stage tests with the long test deselected,
the long planted stage test separately, and 13 OVERFLOW tests).  The focused
C1D construction/drag selection was 73/73; the facade-removal selection was
103/103; and the GYRE residual-writer selection was 19/19.  The physical
constant ratchet was also run across the tree: every file changed by this
round passed its row.  Its full-tree result retains six unrelated pre-existing
literal rows, plus five stale NCAR independent-mirror assertions exposed by
the integration merge; none is used to promote a campaign result.
The explicit touched-source ratchet rerun ends: `7 passed in 0.89s`.

The `where` stabilizers introduced by `a200b4bd4a6` are absent.  The live
stage path again uses multiplication by the 3-D velocity masks, literal to
`stprk3_stg.F90:367,375,382`.  No non-finite value was hidden.

### LOCK and OVERFLOW

All rows below are cellwise oracle-relative scores.  Their residual sidecars
use the common row-scale ULP definition
`spacing(max(max(abs(oracle row)),1))`.  AT_BAR and bit identity are distinct:
nonzero “unequal” counts below remain non-bit despite being at the bar.

| card/boundary | status | normalized L-inf | bit-unequal cells | first trajectory debt |
|---|---|---:|---:|---|
| LOCK stage 1 instantaneous u | AT_BAR | `6.505213034913027e-19` | 20 | — |
| LOCK stage 2 instantaneous u | AT_BAR | `4.336808689942018e-19` | 56 | — |
| LOCK stage 3 instantaneous u | AT_BAR | `8.673617379884035e-19` | 97 | — |
| LOCK kt1..10 | DEBT | kt10 u `1.1371472814487686e-11` | non-bit | kt4 u `2.5789383446155924e-14` |
| OVERFLOW stage 1 instantaneous u | AT_BAR | `1.3877787807814457e-17` | 174 | — |
| OVERFLOW stage 2 instantaneous u | DEBT | `1.5711182355104825e-12` | 201 | stage 2 |
| OVERFLOW stage 3 instantaneous u | DEBT | `7.069220209210414e-12` | 253 | stage 2 |
| OVERFLOW kt1..10 | DEBT, finite | kt10 u `5.422695710827208e-6`; T `1.5225922744832577e-9` | non-bit | kt2 T and u |

The preregistered LOCK prediction is CONFIRMED: the faithful stage-2 row moved
from the stale-base `9.76564494417978e-11` to
`4.336808689942018e-19`.  Its explicit private
`legacy_up3_transport_sign_selector` arm reproduces
`9.76564400092389e-11`, establishing lane-1 commit `60d0c542065a` as the
owner.  OVERFLOW no longer becomes non-finite.  Its first remaining split is
the shared WS-RK3/UP3 stage-2 boundary.  The GYRE lane owns the unresolved
stage clock: NEMO assigns `rDt=rn_Dt/3` at `stprk3_stg.F90:123-124` and passes
that value into the WZV construction at `sshwzv.F90:334-335`; this lane does
not change it.

### GYRE production-JIT Rule 8/12

The first attempt correctly failed provenance because the old default root is
V1 while the merged gate pins scalar-math V2.  The accepted execution uses
`round19_oracle_v2_external` for all three oracle-root arguments, CPU fp64,
production JIT and explicit libm.  kt1 is exact; the first over-bar register
remains kt2 T/S/u/v.  At kt10 the normalized errors are T
`0.005636634494144276`, S `0.00015483121354443772`, u
`0.056249869570541774`, v `0.011173365766925972`, and ssh
`0.000212554355858912`.

The gate now persists its 50 scored cell arrays.  A detached canonical
`c83f73c23ff8` run and the merged run produced byte-identical 12 MiB residual
sidecars (`0ec02246...c0c61`).  The shared comparison reports PASS, 0 moved
rows, 0 improved cells, 0 worsened cells, maximum worsening 0 row-scale ULP,
and unchanged first-over-bar kt2 T/S/u/v.  The row plant exited 1.

### Lane 3b gates

| boundary | result | interpretation |
|---|---|---|
| C1D exchange prefix | 448,950 / 448,950 bit-identical; 60 AT_BAR fields | CONFIRMED ice-side prefix; all 11 CLI row plants exited 1 |
| coupled C1D trajectory | first debt kt2 PRE_SSM.u `1.1172865415493005e-7`; year maximum u `1.996061119407196e-2` | round-17 shortcut dropped as required; canonical integration still lacks prognostic `uu_b/vv_b` carry, GYRE owner |
| SI3 bulk | 271,560 / 271,560 bit-identical | CONFIRMED; libm-return plant exited 1 |
| SI3 thermodynamic step gate | DEBT at kt5 POST_DH.e_s `1.21389354074819e-15` normalized | historical focused gate, not promoted to closure |
| closed SI3 year | MIXED DEBT, phenomenology retained | closure rows below unchanged |
| NCAR O1 bulk | AT_BAR and 0 / 158,292 non-bit | CONFIRMED on pinned stack |

The NCAR coverage register contains **20 fields**: 18 VERIFIED
outputs/intermediates plus two source-backed WAIVED input-only fields
(`cd_du`, `qlwn`).  All 18 VERIFIED-row plants bind; there is no plant for a
waived row.

### Closed rung 3.5 closure (User Decision, retained)

The C1D column remains **CLOSED AS MIXED DEBT**.  Its independent injection is
kt4242 h_i `2.83e-11`; positive-subnormal-snow rows remain, with the largest at
kt5860 POST_ZDF.e_s `0.0251`; year-end normalized continuous errors are t_su
`4.7e-6`, e_i `1.15e-4`, and h_i `1.05e-4`; the six phenomenology rows remain
matched at the measured resolution (melt onset day 133, growth onset day 251,
minimum thickness about 0.555 m, maximum about 2.446 m and their dates).  The
fresh V2 run reports the same classification.  None of those debt rows is
withdrawn or reassigned here.

## Rule 8/12 register

| shared move | boundary | merged-tree result | owner/disposition |
|---|---|---|---|
| ZDF input ordering before stage-3 mean imposition | WS-RK3 PRE_ZDF | canonical integration retained; GYRE 50-row cellwise comparison has 0 moved rows versus c83 | GYRE owner-of-record |
| QCO layer-thickness path | WS-RK3 tracer/momentum stage geometry | canonical integration retained; same 0-move comparison | GYRE owner-of-record |
| quadratic drag association | barotropic drag / implicit ZDF | canonical integration retained; LOCK stage 2 moves into bar, OVERFLOW is finite | GYRE owner-of-record; later OVERFLOW stage-clock debt remains |
| UP3 transport-sign selector | LOCK stage 2 | `9.765644944e-11` to `4.336808690e-19` | lane 1 commit `60d0c542065a`, CONFIRMED |
| round-17 Kmm seed | coupled C1D kt2 PRE_SSM.u | prior AT_BAR claim withdrawn; without invalid seed first debt is `1.117286542e-7` | **SUPERSEDED**; prognostic `uu_b/vv_b` carry belongs to GYRE integration |

The compensating-error rule therefore keeps the canonical shared fixes while
registering their exposed debts; no card-specific numerical switch was added.

## Portability hazard

The NCAR `0 / 158,292` result executes LOG/LOG10/POW through native XLA.  It is
pinned to Python 3.13.0 / JAX 0.10.0 / NumPy 2.4.4 on this machine and is **not
a portable bit-identity claim** until the shared precision policy covers those
functions.  User decision on extending that policy remains pending; this lane
does not implement it.

## FLAGGED FOR FUTURE DELETION

Delete nothing.  Retain and flag `/tmp/codex-si3thd-r18-*`, the eight
`c1d_omip_l3_coupled10m_r13_oracle{,_b,_c,_d,_e,_f,_g,_h}` roots, and failed
`c1d_omip_l3_coupled10m_r17_oracle_a`.  Also retain and flag the detached
`/tmp/codex-si3thd-r19-c83` verification tree.  They are not accepted inputs.

## ASKED / UNASKED

| choice | status | disposition |
|---|---|---|
| merge reconciled integration line `03c6e8d96ff7` | ASKED | performed with hunk ledger above |
| drop round-17 re-derived Kmm seed | ASKED | SUPERSEDED; not reapplied |
| revert dry-face `where` stabilizers to NEMO mask multiply | ASKED | canonical merge restored literal multiplication; verified in source and gates |
| run cross-card and lane gates | ASKED | completed; exact results and debts above |
| drop stale V1 GYRE default after provenance failure | UNASKED | not changed; accepted command names V2 roots explicitly |
| create detached c83 verification worktree | UNASKED construction aid | retained and flagged; used only for Rule 8/12 discriminator |
| delete `bulk_flux_omip.py` facade and update callers | ASKED | deleted; every caller routes directly to `core.bulk_flux` |
| extend scalar-libm to LOG/LOG10/POW | UNASKED / decision pending | not implemented; hazard recorded |
| modify shipped NEMO tree | UNASKED and forbidden | not done |
| delete retained run roots | UNASKED and forbidden this round | flagged only |

## Runtime artifacts (not committed)

Root: `/data/abyssal/dbalwada/nemo-testcases-l3/round19_cross_cards/`.

| artifact | SHA-256 |
|---|---|
| `lock_stage.json` | `ae9a275915aeb2cb2a09c758b2b2c7c4c2a8c18dd23fb2969f338ecd34fc7919` |
| `lock_trajectory_kt10.json` | `ca14b051e595ca8769135703a2342ed1856477aed6a57efecc52f521a41a9a1e` |
| `overflow_stage_faithful.json` | `1038db001470fac4694b103e03ec3e32c466a60096dace1673dde598ee947592` |
| `overflow_trajectory_kt10.json` | `fb676cc4637a45fa46b73a61da2689bcd61a3792187fed836d20dedc0eca1dcc` |
| `gyre_trajectory_kt10_residual.json` | `efc6a44a6d6e3e613bfdab3f709da202922dbb597b189d125804195c3c69e123` |
| merged/c83 GYRE residual sidecar (each) | `0ec022469c778ea05bdb3fdde36b74af7587bba92707cdded818cf70635c0c61` |
| `gyre_c83_vs_merged_rule12.json` | `619d696d323afeb8babe69921f9ed01c77f369ceb7b6d48340754247f89fb90f` |
| `c1d_exchange.json` | `aef5fb7dda64e37981904ac232ca92b5ed32d877b6b2aac23e2e6a17f2e99ca7` |
| `c1d_slab_year_rerun.json` | `6a188616438e3e2e3bdab0920546ff953642405ec01bd2135f74b991cc1ead45` |
| `si3_bulk.json` | `9c6040ef1824c5d1c1a9a261afd273358e55dbe07455bf17b31739e87d4e5712` |
| `si3_thd_step.json` | `ada00ca0058052a2c39622b9dca2a026dca8263fce19439d633c6bdeba0eb251` |
| `si3_column_year.json` | `9a233a2cb6647448810c41c20b3d40a855c9f9fadf2f4a24cfbd5024453a3ae1` |
| `ncar.json` | `30fe378e3e15f07f7eb0e79e67b8f8e717e2fc38d21384859bafc73c1c1c6dca` |

Implementation commits after preregistration: merge `6ce33d2df36`, C1D
reconciliation `8388053b3b5`, facade removal `f80e080e6f0`, and durable GYRE
cell residuals `6f4d83509ff`.  The result/receipt commit is `8c001a25740`;
the following documentation-only ledger commit records that hash.
