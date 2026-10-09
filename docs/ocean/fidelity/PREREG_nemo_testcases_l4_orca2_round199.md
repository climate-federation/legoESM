# ORCA2 round 199 preregistration — OMT-0 acquisition

Date: 2026-10-09. Incoming tip: `d9f8c6950`. Claim label for every future
score from this record: **independent OMT-0**, unless a later receipt names a
separate given-NEMO-entry bridge. No score is measured in this acquisition
round.

Decision 103 fixes OMT-0 as hierarchy rung 0 with five modules removed and no
other physics change. The admitted source deck is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/rung0_namelist_cfg`,
SHA-256 `b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91`.
The scalar-math binary stays the admitted rung-0 binary, SHA-256
`c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343`.

## Frozen deck delta

| module | rung 0 | OMT-0 | compiled selection |
|---|---|---|---|
| momentum advection | `ln_dynadv_vec=.true.` | `ln_dynadv_vec=.false.`, `ln_dynadv_OFF=.true.` | `dynadv.f90:162-190` |
| lateral momentum diffusion | `ln_dynldf_lap=.true.` | `ln_dynldf_lap=.false.`, `ln_dynldf_OFF=.true.` | `ldfdyn.f90:177-228` |
| tracer advection | `ln_traadv_fct=.true.` | `ln_traadv_fct=.false.`, `ln_traadv_OFF=.true.` | `traadv.f90:587-633` |
| lateral tracer diffusion | `ln_traldf_lap=.true.` | `ln_traldf_lap=.false.`, `ln_traldf_OFF=.true.` | `ldftra.f90:230-268` |
| bottom drag | `ln_lin=.true.` | `ln_lin=.false.`, `ln_drg_OFF=.true.` | `zdfdrg.f90:371-401` |

All citations above name the compiled source under
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/`.
The compiled one-of checks prove that both halves of each pair are required;
leaving the old selector true would make the deck invalid rather than define
OMT-0. Every other rung-0 assignment, the CPP keys, mesh and inputs are frozen.
In particular EEN, split-explicit free surface, partial-cell HPG, constant
vertical mixing and NEMO-replacement EVD stay live. Ice stays off as hierarchy
rung 0 already requires; the shipped ORCA2 card and its ice feature registry
are not edited.

## Frozen run protocol

The operator-run launcher must perform, in order:

1. one two-step smoke run (`nn_itend=2`, `nn_stock=2`, explicit restart list
   containing step 1 only);
2. two independent twelve-step executions whose admitted payload is the
   rank-complete restarts at steps 1 through 10 (ten-entry list, terminal
   sentinel at step 12);
3. one 96-step independent execution with a ten-entry restart list at
   `10,20,...,90,95` and a terminal step-96 sentinel.

The physical deck is identical in all four runs. Only `nn_itend`, `nn_stock`,
`ln_rst_list` and `nn_stocklist` may differ. `nn_itend` is always a multiple of
the unchanged `nn_fsbc=2`; no list exceeds NEMO's compiled capacity of ten.
The launcher never invokes `/usr/bin/time`.

## Frozen predictions and falsifiers

**R199-P1 — exact deck delta.** The rendered OMT-0 deck has exactly five
changed assignments and five additions, the pairs in the table, and no other
assignment difference from the admitted rung-0 deck. CONFIRM: exact inventory
and resolved OFF selectors. REFUTE: any extra/missing assignment, changed CPP,
binary, input manifest or rung-0 retained-module value.

**R199-P2 — executable smoke.** The two-step smoke reaches `STOP 0` and its
resolved log prints all five OFF choices. CONFIRM: clean stop plus exact
resolved selection. REFUTE: startup refusal, nonzero exit, missing OFF row, or
any old live selector still true.

**R199-P3 — rank-complete ten-step identity.** Both record executions produce
40 self-describing NetCDF restart shards: steps 1 through 10, ranks 0000 and
0001. `kt`, dtype, dimensions and payload length are read from each file. The
five fields `sshn/un/vn/tn/sn` are finite and `np.array_equal` between twins.
CONFIRM: 200 field comparisons exact. REFUTE: missing shard/field, wrong header,
non-finite value or one differing bit.

**R199-P4 — month calibration and growth record.** The 96-step run produces
rank-complete finite restarts at `10,20,...,90,95` plus the terminal sentinel
at 96. Its step-10 state is array-identical to both ten-step twins. CONFIRM: 20
growth shards plus two sentinel shards, 100 finite field payloads, and 20 exact
step-10 calibration comparisons. REFUTE: missing/wrong-step payload,
non-finite field or any step-10 bit difference.

**R199-P5 — record-only round.** No model/card score and no first-non-bit
statement is claimed before P1-P4 admit. CONFIRM: `git diff -- packages` and
the recipe diff are empty. REFUTE: any production model/card edit or any
trajectory claim made from a refused/incomplete record.

**R199-P6 — controls bind.** Plants for an extra deck delta, a live old module,
an over-capacity restart list, missing smoke completion, missing rank, wrong
`kt`, non-finite payload, twin ULP, step-10 calibration ULP, missing month step
and changed binary must each refuse. CONFIRM: every plant exits nonzero with
`STATUS PLANT-FIRED`. REFUTE: any planted violation stays green.

## Stop conditions

Any P1 deck mismatch is `STOPPED_FOR_DECISION` because it would mean Decision
103 does not uniquely specify a valid NEMO deck. Any missing or refused output
after a valid preflight is `STOPPED_FOR_RECORD` with the same committed
launcher reported as `ACQUISITION_NEEDED`. No stabiliser, tolerance change,
source patch, carried-state change, sea-ice change or NEMO rebuild is allowed.
