# Vertical-mixing lane — V1 round 1 (vmix_r1)

Status: **STOPPED_FOR_RECORD.** No V1 record exists; nothing was measured
against NEMO. Survey, deck, card, acquisition and preregistration are
committed.

Preregistration `docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_vmix_v1_round1.md`
(`0a18f041d`; line-number corrections only in `ef16c0e91`, predictions
unchanged). Commits `0a18f041d` (card, deck, acquisition, tests),
`76aa462e8` (citation re-anchor), `ef16c0e91` (review fixes). Evidence
`gridladder_vmix_rounds/round1/`: `v1_smoke_kt1.json` 21fd0520c293d79f,
`v1_smoke_kt1.py` e3d956934cbe67d5, `codex_review.log` a33e3f33545d24e2,
`codex_review_2.log` 7f2b51b1764cebdf, `reanchor.py` 5b03c522b12d0209.

## 1. Survey (full table in the preregistration, section 1)

G5 and ORCA2 resolve differently on six vertical-mixing switches (TKE, EVD,
DDM, IWM, `ln_zdfcst`, `nn_havtb`) and on `nn_mxlice`. The NEMO code adds two
couplings the deck hides: IWM forces TKE's floors (`rn_emin`, `rmxl_min`) and
resets the backgrounds to molecular and uniform, so in ORCA2 `nn_havtb=1`
only shapes the initial `avt_k`. TKE needs no companion and no cpp key.
Proposed order: V1 TKE, V2 EVD, V3 DDM, V4 IWM, V5 `nn_havtb`.

## 2. V1 and its expected boundary

V1 = G5 + `ln_zdftke=T`, `ln_zdfcst=F`, ORCA2's four `&namzdf_tke` lines.
Deck SHA-256 `5b5dd21ee452d98797a822320e2cdfcf8fa21b35c4c94b5d2a9ff7fcc814ba2c`.
With IWM off, TKE takes its own floors (1e-6 m²/s², 1 cm), not ORCA2's
(1e-10, 1 mm): DECISION_NEEDED below. Expected boundary: NEMO stops at kt 11
as in G5 (P1, PLAUSIBLE). The acquisition records `min(10, completed)` steps
whatever the boundary turns out to be.

## 3. Card — reused, not rebuilt

Searched `packages/ocean` for `zdftke`, `tke`, `mxlice`, `etau`, `hm_i`.
Found and reused: NEMO TKE in `physics/vertical_mixing/tke.py`; the ORCA2
identity's TKE selectors, already carried inert in G5's config; the DINO
card's `nn_etau=1` arm; the full ORCA2 card's TKE seed pattern. Against G5,
the config moves on exactly four fields: the scheme, the background mode,
`etau_mode` and `etau_htau_mode` (test-pinned). The state moves only on the
five TKE seeds. Found missing: `nn_mxlice=2` (zdftke.f90:619-623). Its absence
is declared in the card's unmeasured list, with
`tke_nn_mxlice2_under_ice_mixing_length`. G1–G5 are unchanged (codex finding
7). The two lane-A tests that pin whole rung tables now compare only the G
rungs; that is a likely merge touchpoint.

Execution smoke (not a NEMO comparison): one V1 card step on G5's kt-1
post-SI3 forcing is finite, with tke max 0.068 m²/s². ssh equals G5's at
kt 1, which is consistent with mixing being purely vertical.

## 4. Acquisition

`ACQUISITION_NEEDED`:
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_v1_orca2tke/run.sh`.
Preflight at `ef16c0e91` prints `V1_PREFLIGHT_OK`.

- **Writer-free reference.** G5's pinned writer-free binary, reused without
  rebuilding. `zdftke.f90` and `zdfphy.f90` are byte-identical to
  ORCA2_OMIP_L4's, and preflight checks this.
- **Record build.** `ORCA2V1_OMIP_L4_R1FULLENTRY` is G5's pinned record
  MY_SRC set plus one write-only `hm_i` writer. It is placed after the reused
  surface writer and before `zdf_phy`.
  - The patch is additive only, and a checked guard enforces this.
  - `gfortran -fsyntax-only` against G5's modules passes. A planted
    undefined symbol fails it.
- **TKE state.** It sits in NEMO's restart (`en`, `avt_k`, `avm_k`, `dissl`).
- **Run order.** Two-step smoke, then the 100-step boundary run, then twins
  A/B/C. Restarts are written at every step from 2 to N, never step 1.
  Surface and `hm_i` frames must be byte-identical across all three twins.
- **Not covered by `run.sh`.** Passivity (writer-free reference vs twin
  restart identity) is next round's admission step, as in G5.

## 5. Tests and gates (Rule 2: gate named per claim)

All suites ran on CPU in fp64 (`JAX_ENABLE_X64=1`):

- **New V1 file:** 10 passed. Plants that fail as designed:
  - a TKE line on the wrong rung, and a changed `nn_mxlice`;
  - five planted TKE selectors and two vertical selectors;
  - a moved anchor in `stprk3`, and a removed line in the additive guard;
  - truncated, NaN, negative or wrong-step `hm_i` frames.
- **Inherited suites:** G5 + G4 + G2-ladder + G3 gave 60 passed (13
  deselected). G2 card + G1 tools + G5 top-drag consumers gave 46 passed.
- **Citation gate and private imports:** 23 passed after a mechanical
  re-anchor of 7 recipe citations.
- **Not run:** the G1 ladder fast set, which did not fit the budget in G5
  round 4 either. G1's card is unchanged.

Single review (codex). The first pass timed out with no verdict. The re-run
returned **"VERDICT: DO NOT SHIP"** with three CONFIRMED defects:

1. **Initial `dissl` is wet-masked; NEMO sets it unmasked
   (zdftke.f90:916).** Checked, then kept. The dry value only meets `wmask`,
   or a dry `en` of 0, at :420/:425, and :711 overwrites it at kt 1, so the
   difference cannot be observed. The shared model's own NEMO init masks it
   the same way. The reason is now written next to the constant.
2. **namelist_ref line numbers were wrong.** I had read them off a
   comment-stripped listing. Fixed in the card, deck and prereg, and the deck
   SHA was re-pinned.
3. **The additive-patch guard was vacuous under `pipefail`.** Fixed, and a
   plant test was added.

It confirmed everything else: TKE values, writer placement and passivity,
fail-closed order, and no G1–G5 change.

## OPEN

1. The operator runs `run.sh`.
2. Round 2 admits the record (passivity plus twins). It then extends the
   ladder instrument to V1, and tests P2 by replaying `tke_avn` offline with
   NEMO's `hm_i`.
3. DECISION_NEEDED: the IWM-coupled TKE floors.

ASKED: survey, rung order, V1 deck, card, acquisition, preregistration,
focused tests, one review.
UNASKED, offered for revert:
- the one-step execution smoke;
- narrowing the two lane-A rung-table tests to the G rungs.

UNVERIFIED: NEMO's V1 boundary; whether legoESM's `nn_etau=1`/`nn_htau=1`,
Langmuir and `nn_eice=1` arms are exact on this card (first measured next
round).
