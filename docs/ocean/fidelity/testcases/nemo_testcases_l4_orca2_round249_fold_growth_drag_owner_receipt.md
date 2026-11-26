# ORCA2 round 249 — private-unit growth starts in the slow-V drag correction

Date: 2026-10-12. Frozen base: `5ebe69712`. Preregistration commit:
`ad385514f`. Measurement commit: `32a56f2d7`. Gate update: `dd82b322a`.
Candidate retraction: `ddc0e20d9`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round249/`.

This was a measurement-only round. It changes no effective package/model file,
deck, selector, carried state, stabiliser, sea-ice field, or
`unmeasured_features` entry. The complete round-217 private unit plus its raw
surface-V-mask/T-pivot and resolved stage-mask arms was applied for the
measurement and then retracted. Results are separately labelled
**independent** and **given NEMO's entry**; every reported value is identical
between the two labels. No acquisition was needed.

## Compiled source order

The executing R230 record build first depth-averages the three-dimensional
momentum RHS and then calls the drag and wind consumers at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:194-230`. The drag
consumer forms the bottom residual and adds its live-depth-scaled correction
at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463`.
Only after that does the external solver copy the completed slow forcing and
remove the two-dimensional Coriolis tendency at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:286-324`.

The committed round-249 gate reads passive completed stage states and the
admitted self-describing slow-forcing/substep streams. It never feeds an
offline replay back into the executable. All four kt=1--4 observed transitions
are array-identical to their unobserved executions for every state field.

## Frozen predictions

1. **R249-P1 — CONFIRMED.** Both labelled stage observers are passive, and all
   kt=1--4 state fields remain finite.
2. **R249-P2 — PARTLY CONFIRMED, MAGNITUDE CLAIM REFUTED.** The first greater-
   than-10x boundary is kt=1 stage 1 as predicted, but T, S, and SSH are exact
   there. The predicted O(0.1--1) tracer error is false. The remaining error is
   V on the northern fold.
3. **R249-P3 — REFUTED.** The first recorded substep debt is the incoming
   slow-V forcing on 35 fold faces, before the external recurrence. Within that
   forcing, the depth-mean output is exact and the first non-bit output is the
   baroclinic bottom-drag correction. The held Decision-116 boundary
   association is therefore not this boundary's owner.
4. **R249-P4 — CONFIRMED.** Nothing lands. The final package diff against the
   frozen base is empty.
5. **R249-P5 — CONFIRMED.** All nine named classifier plants refuse, and the
   focused gate tests pass 10/10.

## First-boundary table

The required kt=1 stage-1 table comes first. Entries are fold-band
`max/rms`; the NEMO fold-identity residual is zero for every field.

| field | unit OFF | complete unit | unit fold-identity unequal |
|---|---:|---:|---:|
| SSH | 0.113622392 / 0.028542002 | 0 / 0 | 0 |
| U | 0.0604770345 / 0.004805518 | 3.469e-18 / 2.915e-19 | 1,680 signed-zero-only |
| V | 0.0272271361 / 0.003562965 | 0.0272271361 / 0.002202566 | 4,771 (3,452 signed-zero-only) |
| T | 0.0013611108 / 5.415e-5 | 0 / 0 | 0 |
| S | 0.0009604808 / 4.552e-5 | 0 / 0 | 0 |

Thus the old 3.28 PSU / 0.177 K kt=1 framing is superseded once the exact
transport operands are used: the first boundary has exact tracers.

## kt=1--4 growth table

Each cell is `fold max/rms; interior max/rms`. Stage 0 is the step entry.
Both labels give this same table.

| kt/stage | SSH | U | V | T | S |
|---|---:|---:|---:|---:|---:|
| 1/0 | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 | 0/0; 0/0 |
| 1/1 | 0/0; 0/0 | 3.469e-18/2.915e-19; 1.388e-17/5.181e-19 | 0.02723/0.002203; 1.388e-17/6.680e-19 | 0/0; 3.553e-15/1.892e-17 | 0/0; 7.105e-15/5.578e-17 |
| 1/2 | 0/0; 0/0 | 0.003380/0.000270; 6.198e-5/3.196e-7 | 0.02902/0.002150; 6.765e-5/2.560e-7 | 0.3682/0.01106; 7.105e-15/1.040e-16 | 6.818/0.2012; 1.421e-14/2.527e-16 |
| 1/3 | 0/0; 0/0 | 0.01831/0.001155; 0.05655/0.0001968 | 0.03952/0.002924; 0.02567/0.0001497 | 0.7220/0.02166; 4.545e-5/1.746e-7 | 13.35/0.3922; 2.570e-6/1.815e-8 |
| 2/0 | 8.327e-17/1.808e-17; 6.661e-16/7.364e-17 | 0.01831/0.001155; 0.05655/0.0001968 | 0.03952/0.002924; 0.02567/0.0001497 | 0.7220/0.02166; 4.545e-5/1.746e-7 | 13.35/0.3922; 2.570e-6/1.815e-8 |
| 2/1 | 0.05572/0.01301; 0.1155/0.003715 | 0.05416/0.005059; 0.05195/0.0005259 | 0.05374/0.005931; 0.03526/0.0005753 | 1.014/0.02921; 0.002336/1.245e-5 | 18.75/0.5293; 0.001140/9.821e-6 |
| 2/2 | 0.08357/0.01952; 0.1732/0.005573 | 0.07128/0.005340; 0.05004/0.0005315 | 0.06075/0.006011; 0.03576/0.0005813 | 0.7198/0.02243; 0.003485/1.849e-5 | 13.38/0.4069; 0.001709/1.455e-5 |
| 2/3 | 0.1671/0.03904; 0.3465/0.01115 | 0.2103/0.006480; 0.1831/0.0007309 | 0.1030/0.006077; 0.2839/0.0008004 | 1.337/0.02723; 0.004054/3.393e-5 | 24.71/0.4928; 0.003849/2.949e-5 |
| 3/0 | 0.1671/0.03904; 0.3465/0.01115 | 0.2103/0.006480; 0.1831/0.0007309 | 0.1030/0.006077; 0.2839/0.0008004 | 1.337/0.02723; 0.004054/3.393e-5 | 24.71/0.4928; 0.003849/2.949e-5 |
| 3/1 | 0.1471/0.03360; 0.1896/0.009338 | 0.1826/0.01411; 0.1720/0.0009229 | 0.1442/0.01350; 0.2806/0.001114 | 2.304/0.03579; 0.01149/5.676e-5 | 42.61/0.6507; 0.004808/4.145e-5 |
| 3/2 | 0.1468/0.03242; 0.1554/0.008814 | 0.3000/0.01425; 0.1771/0.0009146 | 0.2715/0.01360; 0.2884/0.001120 | 6.152/0.08667; 0.01617/7.375e-5 | 113.96/1.590; 0.006053/5.374e-5 |
| 3/3 | 0.2244/0.03598; 0.1759/0.009234 | 0.3019/0.01581; 0.4493/0.001142 | 0.6453/0.01828; 0.7194/0.001521 | 8.749/0.1519; 0.05917/0.000141 | 161.6/2.763; 0.01235/9.659e-5 |
| 4/0 | 0.2244/0.03598; 0.1759/0.009234 | 0.3019/0.01581; 0.4493/0.001142 | 0.6453/0.01828; 0.7194/0.001521 | 8.749/0.1519; 0.05917/0.000141 | 161.6/2.763; 0.01235/9.659e-5 |
| 4/1 | 0.5183/0.04493; 0.1270/0.008679 | 0.9697/0.06514; 0.4267/0.001455 | 1.229/0.06814; 0.7145/0.001858 | 21.62/0.2883; 0.08332/0.0001837 | 399.5/5.284; 0.01200/0.0001029 |
| 4/2 | 0.7392/0.06232; 0.1085/0.008739 | 5.375/0.09856; 0.4379/0.001451 | 5.722/0.1036; 0.7348/0.001863 | 181.5/1.912; 0.09637/0.0002090 | 3,342/35.21; 0.01223/0.0001062 |
| 4/3 | 1.402/0.1245; 0.1616/0.01017 | 72,020/1,040; 0.6326/0.001626 | 78,500/892.9; 1.020/0.002169 | 1,947/19.80; 0.1837/0.0003395 | 36,010/367.8; 0.01969/0.0001220 |

The first greater-than-10x boundary is kt=1 stage 1: fold maximum grows from
zero to 0.027227136128767482 in V. A second greater-than-10x boundary at kt=1
stage 2 is downstream and is not attributed as an origin.

## Source-ordered replay and owner

The admitted external-substep table first differs at substep 1's incoming
slow-V forcing: 35/13,320 faces, maximum 1.6557659420864476e-6 at
`[j=147,i=54]`, with 0/13,050 interior faces unequal. Its source-ordered
operand table, identically under both labels, is:

| boundary | unequal on 35-face owner support | max absolute |
|---|---:|---:|
| completed V RHS | 0/1,050 | 0 |
| depth-mean V output | 0/35 | 0 |
| post-bottom-drag V | 35/35 | 1.6557659420864476e-6 |
| post-wind / external incoming V | 35/35 | 1.6557659420864476e-6 |
| Coriolis subtraction | 0/35 | 0 |
| surface V mask | 0/35 | 0 |
| final slow V | 35/35 | 1.6557659420864476e-6 |

Therefore the first non-bit executable output is NEMO's baroclinic bottom-
drag correction, specifically the V statement in `dyn_drg_init`; no external
substep recurrence or boundary association has executed yet. The correction
is not landed because this round did not split its coefficient, bottom-
velocity residual, and live reciprocal as a cancelling unit.

Evidence hashes: `classification.json`
`9285559facf289ab169362e7d2050978b82be6685c4fa3d31a69a83b186ab21d`;
`on_independent.json`
`6130a67cd97727421a7f4853165f55003ffbd0d8245be957b9bcb7adb9c6c8c2`;
`on_given.json`
`5cacf0ed5ad47f876fee8dbb2be0a7fd9ac6a022585b2b32b638b092bbcb7dab`.

## Landing disposition

Nothing lands. The private candidate is retracted, and the package tree is
identical to the frozen base. GYRE, DINO, tanks, OMT, rung 0, rung 10, and
month trajectories therefore retain their base behavior by construction; no
trajectory claim is made from an unrun landing gate. Decision 116 remains
pending with the user and is not re-emitted.

## Controls, tests, citations, and review

The focused round-249 tests pass 10/10. The classifier passes both labels and
all nine plants exit 2 with `PLANT-FIRED`. The round receipt citation gate
passes 3/3 with zero failures and zero unmapped citations; the cumulative
default receipt gate also passes with zero failures, zero failing map entries,
and zero unmapped citations. Shifting the drag citation by two lines makes the
gate exit 1 with `FAIL`, proving the citation control fires.

The one mandatory `tests/ocean/fidelity -n 12` invocation reached 99% with
3,229 passes and 7 skips, then stopped reporting with one worker outstanding.
Before the hang it printed exactly four reds. Each reproduced serially with
`-p no:xdist`; they are the registered pre-existing failures: the moved GYRE
certified-year harness stamp, SI3 `A MY_SRC is not verbatim`, 13 unscoped
allow-dirty drivers, and 13 unstamped report emitters. The isolated result was
4 failed in 7.59 s. No round-249 test failed.

A separate `codex exec --sandbox read-only` review was attempted. Its
in-process app-server could not initialise on the read-only filesystem, before
it read the diff. Verdict: **independent review unavailable in-sandbox**.

## OPEN

Round 250 splits the compiled `dyn_drg_init` V correction offline, in written
order, on the same 35 faces: stored/live V reciprocal, bottom drag coefficient,
bottom V residual, then addition/rounding. The last exact depth-mean output is
the replay seed. A source-exact arm is re-scored atomically with the complete
private unit only after its own output closes; no boundary-association work or
downstream stabiliser precedes that split.
