# NEMO testcase Lane 4 — ORCA2 Phase-2u handoff receipt

Date: 2026-09-06

Parent: `ccdc57c3c29c`

Status: **STOP FOR USER-SHELL MPI.**  **CONFIRMED:** the RK3 shear selector is
now named for its actual Nbb whole-step-entry level, the ORCA2 card selects
NEMO's `nn_eice=1` attenuation, and that boundary is exact at **0 / 8,794**.
The ordered TKE walk stops fail-closed at the next statement,
`zdftke.F90:332 zWlc2=zcsd*taum`, because the admitted V2 record has its input
but no oracle result.  One canonical WRITE-only TKE stream and two
hash-guarded `(jpni,jpnj)=(2,1)` run directories are ready.  No MPI/NEMO run
was executed in the sandbox; EVD, IWM, SI3, and shared TKE arithmetic were not
entered or changed.

## 1. Nbb selector name and time-level registry

**CONFIRMED / identity-preserving rename:**
`nemo_face_native_now2` became `nemo_face_native_nbb2`; no compatibility alias
exists.  The spatial formula, `tke_shear_avm_weighting='nemo_face'`,
`tke_shear_evaluation_stage='step_entry'`, and
`tke_shear_metric_source='nemo_qco_live_face'` are unchanged.  The ORCA2 card
builder selects the tuple at
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:117-130`, and
its structural guard is at `:1381-1403`.

**CONFIRMED / time-level proof:** the live key-RK3 call is
`zdf_phy(kstp,Nbb,Nbb,Nrhs)` at NEMO `stprk3.F90:165`; the preceding
`zdf_phy(kstp,Nbb,Nnn,Nrhs)` is commented out at `:164`.  The latter tuple is
the live MLF call at `stpmlf.F90:190`, not the ORCA2 RK3 arm.  The formal
Kbb/Kmm expression in `zdfsh2.F90:80-100` therefore receives Nbb twice under
this executable.  The registered level is **Nbb / whole-step entry** in both
factors; this receipt never calls that slot “now”.

**CONFIRMED:** the kt=2 ZDF record reports `Kbb=3, Kmm=3, Krhs=1` and the
renamed production and source-literal SH2 paths remain **0 / 218,024** on all
record-complete rank-zero cells.  The 4,396 outer-edge stencils whose halo
operands were deliberately canonicalized remain
`UNINFORMATIVE_RECORD_INCOMPLETE`, not physics debt.  The binding SH2 plant
exits 1.

**CONFIRMED / record semantics:** `NEMO_L4_ZSH2_2` carries all 25 extent
triples in its own header.  The decoder walks the payload to exact EOF from
that header alone.  Its separately encoded allocation table is used only to
validate the header's allocation claims, never to supply decoding side
knowledge.

Evidence: `nemo_testcases_l4_orca2_phase2u_sh2.json`, SHA-256
`24d4db06387bebbb07be8cd141cea66248a254ae4fda1a7830bb2613a7a43211`.

## 2. `nn_eice=1` identity restoration

**CONFIRMED / resolved selector:** the ORCA2 deck selects TKE at
`namelist_cfg:387-396`, overrides `nn_mxl=3` and `nn_etau=1` at `:404-415`,
and inherits `nn_eice=1` from `namelist_ref:1255-1259`.  NEMO executes
`zice_fra=TANH(fr_i*10._wp)` at `zdftke.F90:253-258`; the operand first enters
the Langmuir source at `:358-367`.  The card is restored from `eice=0` to
`eice=1`.  This is a NEMO-identity restoration, not a user physics choice,
and the global default remains zero.

**CONFIRMED / one implementation:**
`nemo_tke_effective_ice_fraction` at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2002-2028` is the
single shared TKE mapping.  Mode 1 materializes `10*fr_i` with
`nemo_source_round`, evaluates `precision_tanh` under the scalar-libm policy,
then materializes the result.  Both the C-grid caller
(`k_profiles.py:883-901`) and MPAS caller (`mpas_integration.py:834-843`) use
that function.  Mode 3 retains the existing `min(4*fi,1)` branch.  KPP's
separately documented analogue is unchanged.

**CONFIRMED / production score:** the CPU, production-JIT, fp64,
scalar-libm gate constructs an independent Python-scalar-libm target from the
recorded `fr_i` and scores the actual production helper:

| boundary | unequal / n | disposition |
|---|---:|---|
| `zdftke.F90:255`, `TANH(10*fr_i)` | **0 / 8,794** | AT-BAR |

The target-bit and wrong-selector arms each exit 1 through the same scorer.
The previously exact surface, bottom, and record-complete SH2 rows remain
respectively **0 / 8,794**, **0 / 8,794**, and **0 / 218,024**.

Evidence: `nemo_testcases_l4_orca2_phase2u_tke_entry.json`, SHA-256
`329b43f6ae81e4b875e766c1853f8614f83596d5f4ce55fdbef895a2e9768c2e`.

## 3. Rule-12 and constructibility controls

**CONFIRMED:** the ORCA2 kt=1 entry state did not move: T/S/u/v are each
**0 / 399,600**, and SSH is **0 / 13,320**.  Its binding T plant exits 1 with
`T (1/399600)`.

**CONFIRMED / cross-card scope:** the cross-card gate reconstructs the actual
NEMO kt=1 Nbb entry records; it does not substitute constructibility for a
numerical row.

| card | T | S | u | v | SSH | total |
|---|---:|---:|---:|---:|---:|---:|
| GYRE | 0 / 18,000 | 0 / 18,000 | 0 / 17,400 | 0 / 17,100 | 0 / 600 | **0 / 71,100** |
| LOCK_EXCHANGE | 0 / 2,560 | 0 / 2,560 | 0 / 2,540 | 0 / 7,800 | 0 / 128 | **0 / 15,588** |
| OVERFLOW | 0 / 17,000 | 0 / 17,000 | 0 / 16,900 | 0 / 60,600 | 0 / 200 | **0 / 111,700** |

The LOCK one-bit T plant exits 1.  GYRE also inherits NEMO `nn_eice=1`, but
its legoESM card remains at mode zero in this Lane-4 work unit; the new helper
is statically unselected there.  LOCK and OVERFLOW do not execute TKE.
**PLAUSIBLE WITH EXPLICIT SCOPE:** C1D remains a constructibility proxy because
the ice fidelity card is not present on this branch; no C1D numerical identity
claim is made.

Evidence: `nemo_testcases_l4_orca2_phase2u_selector_crosscard.json`, SHA-256
`a6f5af1f201b5a79f431e21523689b7b4661ccf6d06fafc094f64e1ec5da7479`,
and `nemo_testcases_l4_orca2_phase2u_orca2_entry.json`, SHA-256
`893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12`.

Focused production/config tests: **15 passed, 132 deselected**.  They cover
scalar-libm mode 1 and its gradient, ORCA2 selector validation, C-grid and
MPAS call sites, bottom-TKE selection, constructibility, and CLI acceptance of
the Nbb-named selector.

## 4. Ordered TKE walk and first unmeasured statement

**CONFIRMED:** NEMO reports `ln_wave=F` at `ocean.output:519` and “No surface
waves” at `:959`.  Thus `cpl_sdrftx=F`; the executed Langmuir velocity arm is
the no-Stokes statement `zWlc2=zcsd*taum` at `zdftke.F90:332`, not the coupled
Stokes expression at `:310-325`.

The current ordered register is:

| executed boundary | result | owner/disposition |
|---|---:|---|
| surface Dirichlet, `:264-269` | 0 / 8,794 | CONFIRMED shared input statement |
| bottom friction/floor, `:279-288,841-844` | 0 / 8,794 | CONFIRMED Lane-4 selector restored in Phase 2t |
| ice attenuation, `:255` | 0 / 8,794 | CONFIRMED Lane-4 forcing selector restored |
| SH2 precursor, `zdfsh2.F90:80-100` | 0 / 218,024 | CONFIRMED shared precursor |
| no-Stokes `zWlc2=zcsd*taum`, `zdftke.F90:332` | — | **UNMEASURED_NEEDS_WRITE_ONLY_INTERNAL_FRAME**, `GYRE_OWNER_SHARED_TKE` |
| LC depth/source, `:339-367` | — | UNMEASURED after first stop |
| Prandtl/RHS/matrix, `:381-420` | — | UNMEASURED after first stop |
| forward/back solve, `:451-469` | — | UNMEASURED after first stop |
| `nn_etau=1`, `:492-495` | — | UNMEASURED after first stop |
| `nn_mxl=3`, `tke_avn:651-723` | — | UNMEASURED after first stop |
| avm/avt/dissl assembly, `tke_avn:711-724` | — | UNMEASURED after first stop |

**CONFIRMED / fail-closed:** the admitted kt=2 frame supplies `taum`, SH2,
buoyancy, geometry, carried en/avm/avt, and masks, but it has no oracle
`zWlc2` or later internal target.  Therefore no first non-bit arithmetic claim
is possible at line 332.  The gate reports
`STOP_TKE_INTERNAL_FRAME_REQUIRED`; it does not compare a derived target to
itself.  Per the preregistration, EVD and IWM are not entered because TKE has
not closed.

## 5. WRITE-only TKE acquisition and build

**CONFIRMED / source audit:** the config-local `zdftke.F90` override adds only
writer-owned allocatables, copies from model operands/results into those
temporaries, and stream output.  It never assigns a model array.  It is armed
only when `lwp .AND. .NOT.ln_tile` and `kt=nit000+1`; storage is allocated only
while armed, zeroed before every copy, restricted to owned wet water columns,
then deallocated.  The reproducible patch is
`scripts/validate/ocean_fidelity/orca2_l4/phase2u_tke_walk_writer.patch`,
SHA-256 `fd4895df20a07a6d43d8a3bc659c3fc05b7a93172b8bdf5b755f4e152320ea25`.
The resulting MY_SRC file SHA-256 is
`993886a619880108bf7c963ff96ef9508f8a887ed7bcc250e64e9112056a431b`.

**CONFIRMED / frozen record:** `oracle_tke_walk_kt00000002.bin` uses magic
`NEMO_L4_TKEW_1`.  Its base header records version, kt, Kbb/Kmm, local domain,
precision, field counts, a derived payload count, and the four resolved
selectors.  Each field has a header extent triple.  Payload order is:

- 3-D: `zpelc`, `en_post_lc`, `pdlr`, `zdiag_pre_solve`,
  `zd_lw_pre_solve`, `zd_up_pre_solve`, `en_rhs_pre_solve`,
  `zdiag_after_forward`, `zd_lw_after_forward`, `en_post_solve`,
  `en_post_etau`, `mxlm`, `mxld`, `avm_post`, `avt_post`, `dissl_post`;
- 2-D: `zice_fra`, `zWlc2`, `imlc_real`, `zhlc`, `zus3`.

The admission gate walks the real record to exact EOF from those headers,
requires non-vacuous Langmuir/solve/closure targets, requires **101 / 101** raw
twin identity and **100 / 100** inherited record identity against the Phase-2s
V2 root, and calls the Phase-1 ordinary-output identity gate.  Magic, extent,
payload-count, truncation, trailing-byte, canonical-halo, and twin-difference
plants are binding and preregistered.  They remain
`UNMEASURED_PENDING_USER_MPI` until the files exist.

**CONFIRMED / build:** `makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath`
completed in the retained writable NEMO copy `/tmp/nemo-orca2-phase2p`; the
shipped checkout was not modified.  The binary is 55,635,824 bytes, SHA-256
`d8122aa75a528dc1b52fc03a1a2312528c65ed9e0a80464fac29217727044d3a`.
`nm -D` reports **0 `_ZGV*` symbols**.  The build log is
`/data/abyssal/dbalwada/nemo-testcases-l4/build_phase2u_tke_walk.log`, SHA-256
`01a7018c62221ab19e08a577438eec772b5ce244bc3cc5717b00b3b4f8e49530`.

## 6. Prepared user-shell twins

Both directories contain copied, hash-identical icebergs-off 10-step deck
files and 40 absolute symlinks to the immutable input deck.  `nemo` is a
read-only symlink to the one binary.  Each launcher validates the binary,
deck manifest, every input, and its exact directory; refuses pre-existing
outputs; sets single-thread CPU libraries; uses Bash `time`; and runs
`mpirun -np 2 --oversubscribe ./nemo` with pipe-status propagation.

| twin | directory | `run.sh` SHA-256 |
|---|---|---|
| A | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2u_tke_a_10step_np2` | `ec19336b2af663a3c7525ad8c815599a8610531e7e5554338b6d777d5ca0c332` |
| B | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2u_tke_b_10step_np2` | `08cfd5bb71c20a3c8917dca74886f81b85bde2a3619e7f0f96ef78d73218c84e` |

The deck manifest SHA-256 is
`e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059`;
the input manifest SHA-256 is
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
No launcher has been run in the sandbox.

## 7. Rule 8 / 11 / 12 and ownership

| rule | disposition |
|---|---|
| Rule 8 | **CONFIRMED:** the eice statement runs through the one shared production helper; selector/target plants fail; the acquisition override is WRITE-only |
| Rule 11 | **CONFIRMED:** the obsolete NOW-named selector is rejected and no current receipt describes Nbb as “now”; the walk stops instead of inferring missing targets |
| Rule 12 | **CONFIRMED:** ORCA2 entry and GYRE/LOCK/OVERFLOW kt=1 rows remain 0 / n after the shared helper change |

**CONFIRMED:** Lane 4 changed only the ORCA2 card selector and the missing
NEMO-named ice-forcing transformation.  The first pending arithmetic at line
332 is registered to `GYRE_OWNER_SHARED_TKE` with this acquisition as its
reproducer.  **PLAUSIBLE PENDING RECORDS:** the new stream will let the next
round distinguish the operand, LC recurrence, RHS, solve, penetration, and
closure statements in order without another writer round.

## 8. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| rename selector to Nbb | ASKED review fix | complete; old name invalid |
| retain `step_entry` | ASKED | unchanged and structurally checked |
| add RK3/MLF proof to registry | ASKED | complete with `stprk3:164-165` and `stpmlf:190` |
| restore ORCA2 `eice: 0 -> 1` | ASKED identity restoration | complete; no choice/default change |
| implement missing mode-1 mapping | ASKED | one shared source-literal helper; C-grid and MPAS use it |
| alter GYRE's currently zero legoESM eice selector | UNASKED / other-lane card work | not done |
| continue TKE to first non-bit/unmeasured boundary | ASKED | stopped fail-closed at line 332 |
| acquire targets needed by that stop | normal in-scope instrumentation | built and staged as twin user-shell runs |
| enter EVD/IWM before TKE closes | forbidden by requested order | not done |
| execute MPI/NEMO in sandbox | forbidden | not done; user-shell launchers handed off |
| alter shared TKE arithmetic after eice | Lane-4-forbidden | not done; routed to GYRE owner |
| enter SI3, edit shipped NEMO, delete, add multi-MB git data, or push | forbidden | none done |

## 9. Stop / next action

Run twin A and twin B, one at a time, with their unchanged `run.sh`.  On
resume, admission must prove the new stream's schema to exact EOF, every plant
nonzero, 101/101 twin record identity, 100/100 inherited V2 identity, and
ordinary-output identity.  Only then may the TKE statement walk resume at
`zdftke.F90:332`.  EVD and IWM remain downstream and unentered.

Session ID: `01a06d99-f562-7b11-bc63-e9b112877f54`.
