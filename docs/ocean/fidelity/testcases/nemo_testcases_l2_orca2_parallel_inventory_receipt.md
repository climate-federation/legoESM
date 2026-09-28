# NEMO-testcases L2 ORCA2 parallel inventory receipt

Date: 2026-09-17. Status: **HELD AT PRE-KT1 ADMISSION; TKE BOUNDARY FRAME
ADMITTED**. The exact certified ORCA2 legoESM testcase assembly is still
absent. The existing Phase-2v A/B record remains reusable for entry/stage
comparison. The native post-boundary/pre-Langmuir TKE candidate is now
admitted: its frame, marker, and inherited-stream byte-passivity all pass.
The earlier refusal was a defect in the gate, corrected in section 3.
(`orca2_inventory.json:195-202`, `orca2_boundary_admission.json:5-47`)

## 1. Governing TKE contract and corrected record audit

Round 101 rejects source reconstruction and requires five native statement
states: `en_entry`, `en_after_boundaries`, `en_after_langmuir`,
`rhs_pre_sweep`, and `en_post_sweep`. In particular,
`en_after_boundaries` must be captured after the surface/bottom assignments and
before the active `ln_lc` arm. (`round101_prereg.md:44-56`,
`round101_prereg.md:60-70`)

Both Phase-2v runs completed kt=10, contain 101 native streams, have identical
stream inventories, and match on executable, deck-manifest, and input-manifest
producer hashes. The probe parsed all ten entry frames and all thirty RK3 stage
frames, required the `T,S,u,v,ssh` schema and exact EOF, and found every A/B
entry/stage twin exact. (`orca2_inventory.json:1276-1307`,
`orca2_inventory.json:203-230`)

The Phase-2v TKE stream is a valid 17-three-dimensional/7-two-dimensional-field
record through exact EOF and pinned SHA-256. It contains `en_post_lc`, not
`en_after_boundaries`; the historical writer copies `en` only after the
Langmuir block. (`orca2_inventory.json:141-193`,
`phase2v_tke_walk_writer.patch:83-99`)

The complete semantic map is therefore:

| Round-101 field | Phase-2v native source | disposition |
|---|---|---|
| `en_entry` | kt=2 ZDF `en_pre` | present |
| `en_after_boundaries` | none | **missing; reconstruction forbidden** |
| `en_after_langmuir` | TKE `en_post_lc` | present |
| `rhs_pre_sweep` | TKE `en_rhs_pre_solve` | present |
| `en_post_sweep` | TKE `en_post_solve` | present |

The inventory names every Phase-2v TKE field, classifies the non-contract fields
as supplemental operands, and lists only `en_after_boundaries` as missing.
(`orca2_inventory.json:1308-1345`, `orca2_inventory.json:1345-1467`)

## 2. Complete current-tree inventory

The current tree does have an execution guard: `validate_nemo_testcase_card` is
called by all three current testcase builders. The current dispatch still
contains only LOCK_EXCHANGE, OVERFLOW, and GYRE; it has no exact `ORCA2-zps`
builder, and the scan found no ORCA2 asset under `scripts/experiment/`.
(`nemo_testcase_recipe.py:669`, `nemo_testcase_recipe.py:759`,
`nemo_testcase_recipe.py:877`, `nemo_testcase_recipe.py:919-925`,
`orca2_inventory.json:8-16`, `orca2_inventory.json:31-40`)

The following generic assets are present and usable:

| asset | current-tree evidence |
|---|---|
| named ocean recipes | `legoesm_nemo_like_v1` and `omip_nemo_match_tripole_v1` (`recipes.py:104-130`, `recipes.py:148-173`) |
| NEMO/TKE configuration and rest constructor | `_nemo_tke_config`, shared NEMO physics, `build_nemo_rest_recipe`, and dispatcher (`nemo_recipe.py:220-318`, `nemo_recipe.py:469-500`, `nemo_recipe.py:1242-1258`) |
| arbitrary external tripole mesh | `--tripole-mesh` and `create_tripole_grid` loading (`run_omip.py:535-561`, `run_omip.py:1931-1967`) |
| NEMO partial cells | mesh-native vertical ladder plus `nemo_tpoint` bottom rule (`run_omip.py:5273-5325`, `vertical.py:1184-1210`) |
| CORE-II | native-file cache builder and runtime loader (`build_core2_nyf_zarr.py:1-37`, `core2.py:106-147`) |
| JRA55-do | noleap cache preparation calling `build_jra55_cache` (`prepare_omip_forcing.py:1-31`, `prepare_omip_forcing.py:183-236`) |
| eORCA1 native fields | monthly SSS, T/S, and SI3 ice readers (`nemo_native_fields.py:1-20`, `nemo_native_fields.py:110-178`, `nemo_native_fields.py:297-319`) |
| QCO/partial-cell arithmetic | source-associated live-face QCO geometry and explicit partial-cell constructor (`vertical.py:535-583`, `vertical.py:1184-1210`) |

These assets support a generic from-rest tripolar run over an externally
supplied ORCA2 mesh. They do not provide the exact certified `ORCA2-zps`
testcase card or its monthly ORCA2 deck reader. The precise conclusion is:
**exact certified ORCA2 testcase assembly is absent; a generic from-rest
tripolar run over an external ORCA2 mesh is constructible from current assets**.
(`orca2_inventory.json:17-40`)

The read-only NEMO oracle target remains built but not runnable directly from
its template: its `EXP00` lacks the named domain and initial T/S files. Its
compiled card does resolve RK3, QCO/VCO, SI3, and TKE. The inventory probe
itself ran no NEMO integration; the later operator acquisition is recorded
below. (`orca2_inventory.json:90-126`)

## 3. Required WRITE-only acquisition

**ACQUISITION_COMPLETE; ADMISSION PASS:**
`COMMIT cf192ae78270:scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2_tke_boundary_acquisition/run.sh`

The committed evidence chain is `COMMIT 0204649679d9`, `COMMIT f507cf48230e`,
and the operator follow-up `COMMIT cf192ae78270`. Its external evidence files
are under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2/`,
including `orca2_inventory.json`, `acq2_preflight.log`, `acquisition2.log`,
`finalize_fix4.log`, `orca2_finalize_admitted.log`, and
`orca2_passivity_plant.txt`, plus the candidate run's admission and
reference/digest manifests; no temporary checkout is an evidence pointer.

The user-executed acquisition cloned `ORCA2_ICE_PISCES`, copied the current
`ORCA2_OMIP_L4` source card file by file, and built the new target
`ORCA2_OMIP_L4_P2VBND_R2`. The original `ORCA2_OMIP_L4_P2VBND` name contains
only the abandoned reference-clone build from the failed first acquisition and
was not reused. The retry pinned the historical Phase-2v TKE patch, A-run deck
and input manifests, ten-step resolved configuration, and np2 layout. The
post-run inventory shows that this candidate does not reproduce the full
101-stream cumulative inventory: seven of those streams come from writer
instrumentation that no longer exists in the current source card. The card must
therefore not be described as an exact Phase-2v source clone, even though the
build itself is passive. (`orca2_boundary_run.sh:38-131`,
`orca2_boundary_run.sh:573-629`, `orca2_boundary_run.sh:762-827`,
`orca2_boundary_admission.json:15-33`)

The new patch is additive: it initializes a write-only buffer, copies native
`en` immediately after the boundary assignments and immediately before
`IF(ln_lc)`, then writes the frame after `tke_tke` returns. It changes no
namelist or model field, and the model TKE dummy argument is `INTENT(in)`.
(`orca2_boundary.patch:1-30`, `orca2_boundary_writer.F90:1-17`,
`orca2_boundary_writer.F90:23-57`, `orca2_boundary_writer.F90:60-83`)

Preflight reconstructs the registered Phase-2v `zdftke` patch, rejects any
removed/replaced line, checks the capture location, and proves both writer and
patched `zdftke` with `gfortran -fsyntax-only`. Every unexpected or explicit
nonzero path emits a named `REFUSE:` line. (`orca2_boundary_run.sh:4-10`,
`orca2_boundary_run.sh:658-747`)

The new `--finalize` mode branches before the clean-tree, source, build, and
launch path. It verifies the registered target directory, built and copied
binary digests, record size, exact schema/EOF, STOP/run status, and time step;
it scans every `ocean.output`, `ocean.output.*`, or `ocean.output_*` rank file
for the native marker and requires exactly one occurrence at the registered
`kt=2`. Before validation it removes prior derived finalization artifacts, so
an early refusal cannot leave a stale admission. It pins the 101 baseline
streams to a registered manifest digest, byte-compares the comparable ones,
writes the record stamp and admission JSON, and makes both those baseline files
and the external built executable directly checkable through the final digest
manifest. (`orca2_boundary_run.sh:193-306`, `orca2_boundary_run.sh:307-341`,
`orca2_boundary_run.sh:343-390`)

Finalization cleared the original logging defect: one marker was found in the
one `ocean.output` file at `kt=2`, and the 3,543,512-byte frame passed exact
schema/EOF with SHA-256
`9a634be35ac9e1714e1f3e5ab4ab31461b0b36726455bf2a6bfa43ce5ab6634e`.

Admission first refused with exit 69 at `expected=101 compared=94
identical=93 missing=7 differing=1`. Both halves of that refusal were defects
in the gate rather than in the build, and the gate has been corrected.
(`finalize_fix4.log:1-19`)

The seven absent streams -- the BBL diffusive record, four EEN records, and both
ZDF-SH2 operand records -- come from writer instrumentation that the current
`ORCA2_OMIP_L4` source card no longer contains. The reference executable holds
each of those writer format strings and the candidate executable holds none of
them, so no passive rebuild of the current card can produce them, and the card
that built the reference has since been overwritten. They are now pinned by
name and counted, so a stream disappearing for any other reason still refuses.

The one differing stream was `oracle_transport_kt00000001_s1.bin`, and the
difference is not a passivity signal at all. See the defect finding below. The
comparison now stops before that stream's third field, leaving its header and
both real transports compared, and the corrected gate admits the record:
marker **PASS**, record **PASS**, passivity **PASS** at 94 compared, 94
identical, 0 missing, 7 registered-absent, 0 differing, 95 of 95 target
streams, exit 0. (`orca2_finalize_admitted.log:1-16`,
`orca2_boundary_admission.json:5-47`)

### FINDING: the reference model dumps an uninitialised array (defect, upstream)

**This is a defect in the NEMO reference model's own instrumentation, not in
legoESM and not in this acquisition.** It should be reported upstream.

The stage-1 transport instrument writes three fields -- `zFu`, `zFv`, `zFw` --
in one statement at line 350 of the source card's `stprk3_stg.F90`. `zFw` is
allocated at line 115 of that file and is assigned only inside the flux-form
branch of the advection-form test that begins at line 322. This deck runs
vector-invariant momentum advection, resolved in `ocean.output` as
`ln_dynadv_vec = T`, so that branch never executes and `zFw` reaches the write
statement having never been assigned.
(`ORCA2_OMIP_L4/MY_SRC/stprk3_stg.F90:115`,
`ORCA2_OMIP_L4/MY_SRC/stprk3_stg.F90:322-350`,
`phase2v_a_ocean.output:1315`)

The consequence is general, and wider than this one acquisition: **every
stage-1 transport stream recorded this way under vector-invariant momentum
advection contains garbage in its third field.** The bytes are leftover heap
and track the executable's memory layout, not the model state. In this pair the
reference happens to show zeros across the whole field while the candidate
shows 1,583 values at the 270 K sea-ice initialisation temperature
(`rn_tsu_ini`, `rn_tmi_ini`, `rn_tms_ini`), and other historical builds show the
same 270s. The header and the first two fields are bit-identical between the
two runs, so no real transport moved.

Any past or future comparison that treats a stage-1 transport stream's third
field as model output is comparing uninitialised memory. The first two fields
remain fully usable.

## 4. Controls and disposition

The missing-boundary plant starts from a hypothetically complete contract,
removes only `en_after_boundaries`, flips admission to the binding missing-frame
verdict, exits 2, and writes no artifact. The provenance-mismatch plant changes
only A/B producer agreement, flips entry/stage reuse to false, exits 2, and
writes no artifact. (`orca2_inventory_missing_boundary_plant.txt:1-4`,
`orca2_inventory_provenance_mismatch_plant.txt:1-4`)

The passivity plant corrupts one byte of an ordinary inherited stream, one byte
of the transport stream's first field, one byte of its second field, and the
transport stream's length both short and padded; all five still refuse with
exit 69, which shows the narrowed comparison is not vacuous. Each plant is
required to produce the line for the specific condition it targets, not merely
any refusal: the three byte plants must name the stream they caught on the
`ORCA2_TKE_BOUNDARY_DIFFERS` line, and the two length plants must produce the
separate refusal that a transport stream is not the registered 10,630,320
bytes, so the length requirement is shown to be what fires rather than an
ordinary byte difference. The sixth control perturbs the excluded third field
and is admitted, which is the documented exclusion behaving as described rather
than an accident. Each plant backs the stream up beside the run, outside the
temporary directory the cleanup trap removes, and restores it afterwards; a
backup left behind by an interrupted attempt is put back, verified against its
recorded digest, before the next attempt plants anything, so restoring is
idempotent and an interrupted plant is repaired by running the script again.
The run is returned to its admitted state afterwards.
(`orca2_boundary_plant.sh:1-25`, `orca2_passivity_plant.txt:1-10`)

The finalizer control first admits a synthetic twin holding the 94 comparable
streams plus the record, with the seven registered-absent streams present only
in the baseline. One of those comparable streams is a synthetic stage-1
transport stream carrying the full three-field geometry, and the twin's two
copies of it disagree in the excluded third field exactly as the reference and
the candidate do, so every admission in this control happens with that
disagreement in place. On that stream the control plants a corrupted header, a
corrupted first field, a corrupted second field and a corruption at the last
compared byte, all of which refuse and name the stream; a corruption at the
first excluded byte and at the last byte of the stream, both of which are
admitted; and a short and a padded copy, both of which refuse under the
separate wrong-length condition. It then plants an externally changed executable, a duplicate
valid marker, a changed candidate stream, an unregistered stream going missing,
a registered-absent stream reappearing, a changed baseline with a matching
candidate, the wrong marker step, a multiline binary manifest, a changed
candidate executable, a symlinked run directory, and a truncated record. Every
planted violation fires; early refusals also prove the prior
stamp/admission/manifests were removed. The two count plants are what keep the
registered-absent list from becoming a blanket waiver.
(`test_nemo_testcase_l2_orca2_parallel_inventory.py:396-628`)

The inventory's held conclusion has two independent parts:

1. Reuse the existing Phase-2v entry/stage frames only after their current
   reader/admission bridge is installed; their schemas, counts, twins, and
   producer provenance pass this audit.
2. The boundary frame is acquired and admitted; use it directly for the TKE
   statement-boundary comparison. Do not reconstruct `en_after_boundaries`, and
   do not copy the seven registered-absent baseline streams into the candidate
   to make the counts agree.

No file under `packages/ocean/legoesm` was changed by this fix round.

## 5. OPEN

1. The uninitialised-`zFw` defect above is recorded but not yet reported to the
   NEMO developers. Report it upstream against the write statement at line 350
   of `cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3_stg.F90`.
   (`ORCA2_OMIP_L4/MY_SRC/stprk3_stg.F90:344-350`)
2. The seven registered-absent streams are absent because the source card that
   built the Phase-2v reference was overwritten. If those operands are wanted
   again, the instrumentation has to be reinstated in the current card and the
   registered-absent list shrunk to match; the list is deliberately pinned by
   name so that shrinking it is a visible edit.
