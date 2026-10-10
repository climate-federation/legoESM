# ORCA2 round 233 — atomic stage-transport geometry refutation

Date: 2026-10-10. Frozen base: `71c8b189f`. Preregistration commit:
`e88189315`. Diagnostic measurement commit: `902d9ec47`. Diagnostic-restoration
and classification commit: `edf36726b`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round233/`.

The final tree changes no package model file, card, deck, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` tuple. The temporary
private hook used only the pre-existing resolved raw-mesh builder and was
removed after measurement; `git diff 71c8b189f..edf36726b -- packages/` is
empty. Every binding number below is kept separately as **independent** and
**given NEMO's entry**. No NEMO acquisition was run.

## Source unit and instrument

The executing record build reads the four raw reference face thicknesses at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188`, constructs the
U/V/F masks from `tmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-213`, applies their
sign-`+1` boundary association at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:228-232`, and executes the
V-point T-pivot fold rule at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:973-980`. It builds the
barotropic correction at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279` and the
stage-1 transport at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.

The private arm combined the complete round-217 vector unit with
`nemo_qco_resolved_mesh_operands`; it did not add a second geometry builder.
The round-230 self-described operand record SHA-256 is
`5cd9ef3f875cf54f79ec1e3896cd3d738505046d4175c2a4bb51fd25116afb79`.
Both labels ran production JIT on CPU with fp64/libm. Every completed state
slot is array-identical between the ordinary and traced executions.

## Binding result — geometry closes, root ownership does not

The two labels give the same numeric endpoint scores:

| label | operand or endpoint | unequal / support | max absolute | argmax |
|---|---|---:|---:|---|
| independent | live `e3v` | 0 / 226,637 | 0 m | — |
| independent | `vmask` | 0 / 399,600 | 0 | — |
| independent | `r1_hv` | 0 / 8,589 | 0 | — |
| independent | `vn_adv` | 8,589 / 8,589 | 24.341577728515905 | `[147,49]` |
| independent | `zvb` | 8,589 / 8,589 | 0.00919996399945781 m/s | `[147,49]` |
| independent | `zFv` | 226,637 / 226,637 | 339003.55099822127 | `[147,49,25]` |
| independent | fold-band S | 4,164 / 16,200 | 3.283356343139289 PSU | retained in JSON |
| independent | fold-band T | 4,164 / 16,200 | 0.17733430832081432 K | retained in JSON |
| given NEMO's entry | live `e3v` | 0 / 226,637 | 0 m | — |
| given NEMO's entry | `vmask` | 0 / 399,600 | 0 | — |
| given NEMO's entry | `r1_hv` | 0 / 8,589 | 0 | — |
| given NEMO's entry | `vn_adv` | 8,589 / 8,589 | 24.341577728515905 | `[147,49]` |
| given NEMO's entry | `zvb` | 8,589 / 8,589 | 0.00919996399945781 m/s | `[147,49]` |
| given NEMO's entry | `zFv` | 226,637 / 226,637 | 339003.55099822127 | `[147,49,25]` |
| given NEMO's entry | fold-band S | 4,164 / 16,200 | 3.283356343139289 PSU | retained in JSON |
| given NEMO's entry | fold-band T | 14,995 / 16,200 | 0.17733430832081432 K | retained in JSON |

The raw registry also closes exactly: `tmask`, `umask`, `vmask`, and `fmask`
are each 0 unequal against the admitted NEMO bundle under both labels. The
different T unequal counts are signed-zero/bit counts; their binding maximum
is identical and far above the floor.

R233-P1 is **CONFIRMED**: the instrument is passive and the selected raw
geometry is exact. R233-P2 is **REFUTED**: `vn_adv` remains 8,589/8,589
unequal rather than collapsing to at most 35, and `zFv` remains globally
unequal. R233-P3 is independently **REFUTED**: fold S remains 3.283 PSU and T
remains 0.177 K rather than falling to the preregistered thresholds. Thus the
raw geometry defect is real and independently exact, but it is not the root of
the external-mode correction debt that the complete unit exposes.

R233-P4 and P5 are **NOT ACTIVATED** by their frozen prerequisites. The
expensive OMT/rung-0/rung-10 ladders, OMT-4 month, GYRE year, DINO month, and
tank gates were therefore not run and no landing is claimed. This is a failed
prediction retained mechanically, not a post-hoc reinterpretation.

## Controls and validation

The independent/given-entry JSON SHA-256 values are
`60f0cacdaef066925cf02772efd974b3d56da5b3388bcb6cd60f56f9f8e906c2` and
`8035372d4bcacd33ccbef2957c4d4a1d3404d388a6c5fd869717165a9d234895`.
Their log SHA-256 values are
`893715153f9227dc011826fd575fe3bf46ef728bc560be1d035725613f17324c` and
`99a70513bbd4b02a08a20148400764b584004c78d4962aec0437f0e169516d73`.
The classification SHA-256 is
`c4517f20f26873a68e1c2380553bd328bf02490785b81d7ffb161ed2b92fa166`.

The focused round-232/233 gate battery passes 8/8, including label-coverage,
geometry-closure, and false-root plants (log SHA-256
`0e712fbd64eeecef1c415a672fc835407b5cfd0eed22f6bf7a5b7cdfdd29e035`).
The round receipt citation gate passes 6/6 citations and the cumulative
default gate passes 274/274, both with zero failures or unmapped citations.
The rigid two-line-shift plant on the `domzgr.f90:184-188` citation refuses.
Round/default/plant log SHA-256 values are
`6a29dd6348f3acc785c26f080a10a6df7ca05300dcf7510937c18764c1c97ce9`,
`4136317d58965de5347339a30380431b81a6e1e53243292e134f85bdfdb9bdfc`,
and `f1201febbbf03c76475c49348ab47598bc8234c8eed98842f6b22a5de55a0919`.

Independent review was attempted with a separate `codex exec --sandbox
read-only` process. It exited before reading the diff with `failed to
initialize in-process app-server client: Read-only file system (os error 30)`.
Independent review unavailable in-sandbox. Log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The one prescribed `tests/ocean/fidelity -n 12` battery collected 3,142
tests. At the bounded 99% compiler-limit stall it had emitted 3,113 PASS,
seven SKIP, and four FAIL results; 18 tests were unclassified rather than
called passed. The four failures are exactly the registered pre-existing
reds: the GYRE round-129 record stamp, allow-dirty scope, worktree-stamp
ratchet, and SI3 scalar-math provenance. No round-233 test is red. Battery
log SHA-256:
`6e716bfe6a7f0ea00275f205155f078c800c1299debf06ea6f6ad1770ed959e6`.

## Preregistered predictions

| ID | disposition |
|---|---|
| R233-P1 | **CONFIRMED**: passive trace; raw face thickness and all four masks exact. |
| R233-P2 | **REFUTED**: `vn_adv` remains 8,589/8,589 unequal and `zFv` remains global. |
| R233-P3 | **REFUTED**: fold S/T maxima are unchanged and the landing endpoint is absent. |
| R233-P4 | **NOT ACTIVATED**: P3 is false. |
| R233-P5 | **NOT ACTIVATED**: no candidate qualifies for landing gates. |
| R233-P6 | **CONFIRMED for diagnostic restoration**: final package diff is empty; remaining documentary gates are reported below. |

## OPEN

The user must choose between landing the independently exact raw
face-thickness/mask geometry as its own cited correction, or retaining it
privately until the separate external-mode `vn_adv` debt closes. **Pick: land
the independently exact geometry unit** in a subsequent preregistered round,
because its operands are directly sourced and bit-gated even though they
expose an independent compensating error. This round does not take that choice.

After that decision, walk `vn_adv` from the already-passive completed-stage
states and the admitted operand record. No new NEMO acquisition is presently
required.

ASKED choices: Decisions 103, 109, 113, standing Decision 96, and the round-233
geometry-unit disposition above. UNASKED choices: empty.
