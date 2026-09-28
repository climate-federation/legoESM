# NEMO testcase Lane 4 — ORCA2 Phase-2s handoff receipt

Date: 2026-09-06

Starting parent: `65dcf0adc25e396758cc018128c1334e8d9c8547`

Status: **STOP FOR USER-SHELL MPI.**  **CONFIRMED:** the ORCA2 card now selects
NEMO's resolved RK3 `zdf_sh2` identity, the scalar-math acquisition binary is
built with a self-describing kt=1/kt=2 ZDF stream and the four EEN streams,
and two hash-guarded `(jpni,jpnj)=(2,1)` twin directories are ready.  No NEMO
or MPI process was run in the sandbox.  Shared SH2/TKE arithmetic was not
changed.

## 1. ORCA2 card selector

**CONFIRMED:** the ORCA2 deck selects `ln_zdftke=.true.` at
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:387-402`.  NEMO consequently
selects `np_TKE` and sets `l_zdfsh2=.TRUE.` at `zdfphy.F90:207-223`, then
executes `zdf_sh2` before `zdf_tke` at `zdfphy.F90:264-286`.  The no-Stokes
arm is the face-native formula at `zdfsh2.F90:78-100`: formal Kmm and Kbb face
differences, face `avm_k` sums, the corresponding `e3uw/e3vw` factors, face
masks and coast weights.  Under the resolved RK3 call at
`stprk3.F90:164-165`, the commented MLF-form call uses Nbb/Nnn while the live
call passes Nbb/Nbb.  Thus this card requires the Nbb whole-step-entry slot in
both factors and the same live-QCO face metric in both factors.

**CONFIRMED / NEMO-identity restoration, not a choice:** only the ORCA2
specialization in `nemo_testcase_recipe.py` changes:

| selector | prior | corrected |
|---|---|---|
| `tke_shear_production` | `squared_centered` | `nemo_face_native_nbb2` |
| `tke_shear_avm_weighting` | `tpoint` | `nemo_face` |
| `tke_shear_evaluation_stage` | `step_entry` | `step_entry` |
| `tke_shear_metric_source` | `tpoint_jacobian` | `nemo_qco_live_face` |

**CONFIRMED:** the structural validator at
`nemo_testcase_recipe.py:1369-1386` rejects the former tuple.  The old-tuple
plant uses that real validator and passes only by raising.  The ORCA2 entry
gate remains T/S/u/v `0 / 399,600` each and SSH `0 / 13,320`; its JSON is
byte-identical to Phase 2r (SHA-256
`893f38dfa12c0ee7054e4ebfb2a23f23b16fbfe1f96c42f17137112210228d12`).

**CONFIRMED / RULE 12:** the committed cross-card gate reads actual NEMO kt=1
Nbb entry records and scores the unchanged constructed cards:

| card | T | S | u | v | SSH | total |
|---|---:|---:|---:|---:|---:|---:|
| GYRE | 0 / 18,000 | 0 / 18,000 | 0 / 17,400 | 0 / 17,100 | 0 / 600 | 0 / 71,100 |
| LOCK_EXCHANGE | 0 / 2,560 | 0 / 2,560 | 0 / 2,540 | 0 / 7,800 | 0 / 128 | 0 / 15,588 |
| OVERFLOW | 0 / 17,000 | 0 / 17,000 | 0 / 16,900 | 0 / 60,600 | 0 / 200 | 0 / 111,700 |

The GYRE production trajectory gate independently reports `AT-BAR` at kt=1.
The LOCK production trajectory gate independently reports the same kt=1 rows
at bar.  **CONFIRMED:** the direct entry gate is the reduced-scope OVERFLOW
route: it avoids recompiling an ocean step while scoring all 101 levels.  Its
one-bit LOCK T plant exits 1 with `FAIL: LOCK_EXCHANGE-zco kt=1 T moved`.

**CONFIRMED WITH SCOPE:** the branch does not yet contain the Lane-3 C1D
fidelity card, so a literal C1D card row cannot be manufactured here.  The
available C-grid ice path is constructible and its focused suite passes; this
is recorded as `CONSTRUCTIBILITY_PROXY_ONLY_PENDING_ICE_MERGE`, not a C1D
numerical claim.  The combined selector/recipe/ice suite reports 53 passed and
one pre-existing environment skip.  Source isolation plus those numerical
rows proves that no merged GYRE/LOCK/OVERFLOW card moved.

## 2. WRITE-only ZDF/EEN acquisition

**CONFIRMED:** the config-local `zdfphy.F90` block is after `zdf_sh2` and
before `zdf_tke`.  It is armed at `kt=nit000` and `kt=nit000+1`, guarded by
`lwp .AND. .NOT.ln_tile`, allocates only writer-local storage, fills every
view from zero, and assigns no model array.  The tracked
`phase2s_zdf_een_writer.patch` is the reproducible source artifact; the
writable build source hashes are:

| MY_SRC override | SHA-256 |
|---|---|
| `zdfphy.F90` | `121e5b1b8283c39e1cc81c6d873c8f4ac3fe6f825956728a79aeecbd21f83d4e` |
| `dynspg_ts.F90` | `3430c945df0d8e2dbdb7fe6f23599bcf24af19f81d4c2a16555c10c7453456c1` |

**CONFIRMED:** `NEMO_L4_ZSH2_2` is self-describing.  Its header has the
existing 13 int32 values followed by 25 `(n1,n2,n3)` int32 allocation triples
in payload order.  Each extent comes from `SHAPE` or `SIZE` in the writer;
the payload count is `SUM(PRODUCT(extent(:,field)))`.  The gate uses those
real extents to walk all 25 arrays to exact EOF and rejects altered magic,
extent, derived count, truncation, trailing data, a non-canonical slot, or a
vacuous kt=2 velocity-gradient frame.  These plants are preregistered but
cannot execute until real files exist.

**CONFIRMED:** the Phase-2m EEN block was restored byte-for-byte to
`dynspg_ts.F90`.  The one binary therefore writes `e3f_0vor`, live `e3f_vor`,
`q`, and eight `zpvo` arrays at kt=1, in addition to both ZDF frames.  Phase-2m
will become a witness after admission; until then its pin remains authoritative.

### Frozen ZDF inventory and time levels

**CONFIRMED:** both ZDF frames carry this exact frozen inventory.  The actual
Kbb/Kmm/Krhs integers and extents will be accepted only from the real header.
The reader decodes the record from that header alone; the separately written
allocation table is used only to validate the header's allocation claims.

| arrays | allocation/grid | registered level at the pre-closure boundary |
|---|---|---|
| `sh2` | A2D W | output of `zdf_sh2(Kbb,Kmm)` |
| `avm_k_pre`, `avt_k_pre`, `en_pre` | full/reduced W | carried pre-`zdf_tke` closure state |
| `rn2`, `rn2b` | reduced W | RK3 Nbb/Nbb buoyancy operands at step entry |
| `u_Kbb`, `u_Kmm` | full U | explicit header Kbb and Kmm |
| `v_Kbb`, `v_Kmm` | full V | explicit header Kbb and Kmm |
| `e3uw_Kbb`, `e3uw_Kmm` | full WU | explicit header Kbb and Kmm live-QCO metrics |
| `e3vw_Kbb`, `e3vw_Kmm` | full WV | explicit header Kbb and Kmm live-QCO metrics |
| `umask`, `vmask`, `wumask`, `wvmask` | full U/V/WU/WV | static resolved masks |
| `gdepw_Kmm`, `e3t_Kmm`, `e3w_Kmm` | full W/T/W | explicit header Kmm live geometry |
| `taum`, `fr_i`, `rCdU_bot` | A2D/full T | current post-SBC / drag inputs entering TKE |
| `mbkt_real` | full T | static bottom index, exact integer encoded as binary64 |

**PLAUSIBLE PENDING REAL FRAME:** kt=2 will contain nonzero Kbb and Kmm
vertical face gradients and will discriminate `zdfsh2.F90:80-100`, including
the live-QCO divisor and coast weighting.  The gate refuses admission if this
prediction is false.

## 3. Build and staged twins

**CONFIRMED:** `makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath` completed from
the writable copy `/tmp/nemo-orca2-phase2p`; the shipped NEMO checkout was not
modified.  The binary is 55,610,040 bytes, SHA-256
`4d9fd12ecdee900ad5ee3eedc49d2a1abe4aa3fe803d5ec3237265f9869b6f65`,
and `nm -D` reports `ZGV_SYMBOLS=0`.  The retained build log SHA-256 is
`752b03efbb200622fe2ac117de19bb1c2e4f2080ef5c908f1d379c3e23c7bb4e`.

**CONFIRMED:** both run roots use the same read-only binary symlink, copied
icebergs-off variant namelists (`nn_itend=10`, `nn_stock=10`), and 40 absolute
input symlinks.  Both deck manifests hash to
`e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059`;
both input manifests hash to
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
The launchers validate every hash, refuse pre-existing outputs, set single
thread CPU libraries, run `mpirun -np 2 --oversubscribe ./nemo`, retain Bash
wall/user/sys timing, and propagate MPI/tee status.

Run directories and launcher hashes:

| twin | directory | `run.sh` SHA-256 |
|---|---|---|
| A | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2s_zdf_een_a_10step_np2` | `66f9515a04d13c2d1020f2e0e87ec896a823dea75d1849c726221b48bae1d9f0` |
| B | `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2s_zdf_een_b_10step_np2` | `d16daa0729267a1f188ab6328b19475ea4ce45b36b7089d2c5d411c8ddc38abd` |

**CONFIRMED / admission contract:** twins must be raw-identical for every
record.  Relative to the Phase-2q V2 root, all 94 inherited records other than
the deliberately versioned kt=1 ZDF stream must remain raw-identical; kt=2 ZDF
and four EEN streams are additions.  The ordinary restart/history identity
gate also applies.  Any consumed or ordinary-output difference voids the
WRITE-only claim.

## 4. TKE entry pre-walk

**CONFIRMED / source-input row:** using the admitted kt=1 operands, the
production surface Dirichlet statement is exact: `0 / 8,794` wet columns,
including 8,765 columns above the `rn_emin0=1e-4` floor.  This scores
`zbbrau=rn_ebb/rho0` and `en(:,:,1)=MAX(rn_emin0,zbbrau*taum)` at
`zdftke.F90:238,264-269` under production JIT, fp64 and scalar-libm.  Its
one-bit target plant exits 1 through the same scorer.

**CONFIRMED / FIRST TKE DEPARTURE:** NEMO next enters the bottom-boundary
branch because `ln_drg_OFF=.false.` (`zdftke.F90:279-288`).  The current ORCA2
card has `bottom_tke_bc=False`.  At cold start, all Kbb velocities are zero,
but NEMO still writes the IWM-forced `rn_emin=1e-10` at every one of 8,794 wet
bottom boundaries (`zdftke.F90:841-844` sets that floor); the card leaves the
corresponding pre-closure boundary at zero.  Result: **8,794 / 8,794**.  Owner
is `LANE4_ORCA2_CARD_SELECTOR`; shared bottom-boundary arithmetic is not
changed in this phase.

**CONFIRMED / NEXT REGISTERED DEPARTURE:** NEMO resolves `nn_eice=1` and forms
`TANH(10*fr_i)` at `zdftke.F90:246,253-258`, first consumed in the Langmuir
source at `:305-370`.  The card has `eice=0`; the operand differs in 1,779 of
8,794 wet columns.  This is a Lane-4 card selector followed by
`GYRE_OWNER_SHARED_TKE` source-order arithmetic.  It is not repaired or
claimed here.  `nn_etau=1` versus card `etau_mode='none'` and the card's
vectorized Langmuir evaluation are also registered downstream as UNMEASURED.
The existing frame does not contain `nmld/hmlp` or post-TKE mixing lengths, so
the `zdf_mxl`/`tke_avn` length rows are explicitly
`UNMEASURED_NO_LENGTH_TARGET`; no zero or proxy is substituted.

**CONFIRMED / RULE 11:** kt=1 has zero nonzero values in all four recorded
Kbb/Kmm U/V operands, so the SH2 `0 / 224,547` row remains vacuous.  The kt=2
frame will discriminate face gradients, face `avm_k`, live metrics, coast
weighting, `p_sh2`, and its first use in the matrix/RHS statements at
`zdftke.F90:381-420`.  No shared SH2/TKE certification is inferred before the
twins are admitted.

## 5. Coverage, rules and ownership

| boundary | disposition | owner |
|---|---|---|
| ORCA2 resolved SH2 selector tuple | **CONFIRMED corrected** | `LANE4_ORCA2_CARD_SELECTOR` |
| kt=1 cold-start SH2 | **PLAUSIBLE / UNMEASURED_ZERO_GRADIENT** | GYRE shared SH2 |
| kt=2 nonzero SH2 frame | **UNMEASURED_PENDING_USER_MPI** | GYRE shared SH2 after operand admission |
| TKE surface Dirichlet | **CONFIRMED 0 / 8,794** | shared TKE input statement |
| TKE bottom selector | **CONFIRMED DEBT 8,794 / 8,794** | Lane 4 card selector |
| TKE under-ice attenuation | **CONFIRMED selector debt 1,779 / 8,794** | Lane 4 selector, then GYRE arithmetic |
| `zdf_mxl` / post-TKE mixing lengths | **UNMEASURED_NO_LENGTH_TARGET** | shared TKE after acquisition |
| TKE matrix, solve, `nn_etau`, EVD/DDM/IWM | **UNMEASURED** | shared or adjudicated after first departure |
| SI3 dynamics/thermodynamics | **UNMEASURED_PENDING_ICE_MERGE** | ice lanes; exchanges remain oracle-supplied |

**CONFIRMED / Rule 8:** the only production physics edit is a selector in the
single shared ORCA2 card builder; no operator was forked.  **CONFIRMED / Rule
12:** GYRE, LOCK and OVERFLOW entry rows are all exact, and their builders
remain constructible.  **PLAUSIBLE:** C1D will remain unchanged because no ice
code changed, but this branch cannot make a C1D fidelity-card numerical claim
before the scheduled ice merge.

## 6. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| correct ORCA2 SH2 tuple | ASKED NEMO-identity restoration | corrected from old tuple to exact resolved tuple |
| keep BBL defaults 0 / 0.0 | ASKED, Decision 10 | resolved: unchanged; no silent on |
| GYRE prognostic `uu_b/vv_b` | Decision 8 assigns GYRE | not changed here; their cross-card rerun remains pending |
| scalar-libm `log/log10/pow` | Decision 9 assigns ice-thermo | not changed here |
| choose kt=`nit000+1` | ASKED minimum kt>=2 frame | selected; actual non-vacuity is admission-gated |
| choose extent triples | ASKED implementation option | selected over allocation hash; self-describing and source-derived |
| restore EEN streams in same binary | ASKED | restored unchanged from Phase 2m |
| execute MPI/NEMO | forbidden in sandbox | not done; two user-shell launchers staged |
| change shared SH2/TKE arithmetic | forbidden in Lane 4 | not done; debts registered |
| correct TKE selectors during the pre-walk | UNASKED this phase | not done; first boundary registered for next decision/work unit |
| claim C1D numerical identity without merged card | forbidden by Rule 11 | not done; constructibility proxy labelled |
| delete, edit shipped NEMO, add multi-MB git data, or push | forbidden | none done |

**CONFIRMED:** no Phase-2r review findings were relayed during this phase; its
review remains pending and therefore no unreceived finding is represented as
closed.

## 7. Stop and next action

**CONFIRMED:** run the two launchers above, one at a time, unchanged.  On
resume, Lane 4 must validate both real schemas to exact EOF, prove all plants
exit nonzero, require raw twin identity and ordinary-output identity, then pin
one combined V2 root only if every admission condition passes.  The first
nonzero SH2 statement and the registered TKE selector boundary are the next
ordered ladder work.  Stop remains before all legoESM SI3 work.

Session ID: `01a06d99-f562-7b11-bc63-e9b112877f54`.
