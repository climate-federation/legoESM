# DINO / NEMO fidelity debt register (#1226)

Written 2026-07-27 in response to: *"how many more 'good enough's have you left
in there?"* — a full audit of every term where I declared PASS / MATCHED /
CLOSED / WAIVED while the measurement was not actually exact.

**The bar** (standing user directive): NEMO-faithful, corr 1.0 **and** ratio 1.0.
Not "close", not "within tolerance", not "attributed to X" on the strength of a
single black-box analysis.

**Method required to clear an item** (learned the hard way): dump the routine's
**internals** from NEMO, compare **stage by stage**, and name the first
deviating stage. Output-only comparisons hide compensating errors — that is how
`dyn_adv` passed for a day with a perfect KEG masking a broken ZAD, and how
"threshold chatter"/"irreducible amplification" got recorded as verdicts when
they were only ever hypotheses.

---

## A. Revoked acceptances (already reopened)

| # | term | measured | label I gave it | why it is debt |
|---|---|---|---|---|
| 8 | `traadv_fct` limiter | tendency 0.9923 | "FAITHFUL" | `nonosc` is deterministic; a true transcription must hit roundoff. Prime suspect for spurious diapycnal mixing. **IN PROGRESS** |
| 4 | `zdf_mxl` | 99.88% levels (12 cols) | "ACCEPTED" | "below input precision" never proven by comparing internals |
| 10 | `dyn_vor` EEN | 0.9999; bottom levels 1.04–1.07 | "ACCEPTED" | two fix attempts inert ⇒ mechanism still unknown |
| 7 | eiv transport | u 0.9985 / v 0.9954 | "near, accepted-for-now" | "diffuse, no lead" = not investigated to internals |
| 11 | `zdftke` composite | 0.9976 (257 cells) | "CLOSED" | "EVD threshold chatter" is a hypothesis, not a verdict |
| 5 | `ldf_slp` bottom row | \|x\| 1.0255 | "irreducible amplification" | asserted from a substitution test, not from internals |

## B. Called PASS/MATCHED with a ratio that is NOT 1 (never revoked — new debt)

| term | corr | ratio / error | what I said |
|---|---|---|---|
| `eos_rab` α | 1.000000 | median 4.7e-6 (β bit-exact) | "PASS" |
| `bn2` | 1.0000000000 | median \|rel\| 6.96e-6 | "PASS" |
| `ldf_eiv` κ | 1.000000 | \|x\| **1.000608** | "CLOSED" |
| `dyn_hpg` | 1.000000 | \|x\| **1.000045** | "MACHINE-EXACT" |
| `dyn_spg_ts` outputs | 0.9996–0.99999 | `puu_b` \|x\| **0.9871**, `un_adv` **1.0067** | "VERIFIED" — a 1.3% ratio gap, never explained |
| ATF filters | u 0.999969 | u \|x\| **0.9955** | "MATCHED" — 0.45% |
| `dyn_ldf` | 0.9979 / 0.9994 | \|x\| **1.0039** | "MATCHED" — 0.4% |
| `traadv_fct` fluxes | 0.99994 | \|x\| 1.0001 | "faithful" |
| slopes (interior) | ≥0.9988 | \|x\| 1.0011 | "CLOSED" |

## C. Never verified at all (waived, deferred, or structurally skipped)

| item | status | note |
|---|---|---|
| `mlf_baro_corr` | algebra-verified only | inlined at -O3; needs a `_step_impl` diagnostics hook. NEMO dumps 8883-8886 already exist |
| `dom_qco_r3c` **r3u/r3v** | never compared | `nemo_io` reader lacks `hu_0`/`hv_0`; only the T-point `r3t` was checked |
| `lbc_lnk` sign convention | deferred | needs a different harness |
| `zdf_mxl_turb` | UNVERIFIED | found by the runtime trace; turbocline depth, consumers unchecked |
| `zdf_drg_nonlin` + `dyn_drg_init` | "interface-covered" | bottom drag, never term-isolated |
| `dyn_cor_2d` (69×/step) | "interface-covered" | per-substep barotropic Coriolis, never term-isolated |
| tracer advection **salinity** | never compared | only `jp_tem` was dumped/compared; S assumed to follow |
| `traadv_fct` clean tendency | worked around | `trd` dumps are Krhs-contaminated; comparison used flux reconstruction, not a byte-level tendency |

## D. Unexplained anomalies (no owner, no hypothesis under test)

1. **legoESM recovers only 68% of its own ACC from its own density field**, while NEMO is thermal-wind self-consistent (1.46 by the same measure, i.e. consistent within the method's bias). A genuine momentum/reference-velocity residual, never chased.
2. **The 90-day twin's `u_surf` got WORSE** (0.9797 → 0.9691) after the momentum RHS was made machine-exact. Never explained; implies error downstream of the explicit RHS.
3. `dyn_spg_ts` `puu_b` \|x\| 0.987 — the entry-seed hypothesis was falsified (fix inert); no replacement hypothesis.
4. NEMO's `grid_T` history output is **numerically wrong** (votemper ×9.6 surface → ×290 deep; restarts are fine). Cause not diagnosed — an XIOS thickness-weighting/normalisation issue is suspected. Any past analysis reading NEMO T/S from `grid_T` is invalid.

---

## Honest count

- **6** acceptances revoked
- **9** terms labelled PASS/MATCHED with a non-unit ratio
- **8** never verified at all
- **4** unexplained anomalies

The step-by-step methodology was created precisely to prevent this, and then I
applied it with a tolerance it does not have. Nothing in A–C is known to be
wrong; the point is that **none of it is known to be right** at the stated bar,
and the campaign has now produced enough retractions to make that distinction
load-bearing.
