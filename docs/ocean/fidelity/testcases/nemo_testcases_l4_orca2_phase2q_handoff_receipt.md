# NEMO testcase Lane 4 — ORCA2 Phase-2q stop receipt

Date: 2026-09-06

Starting parent: `124d9f5cea63`

Status: **STOP — ZDF HEADER DEFECT FOUND; REPLACEMENT TWINS PREPARED.**
The Phase-2p ZDF twins are successful and reproducible, but their new stream
cannot be admitted because its declared payload count is false.  The writer,
validator, scalar-math binary, and two hash-guarded launchers are corrected.
No MPI/NEMO run was made in the sandbox.  SH2 scoring and the BBL review
measurements remain behind this fail-closed admission boundary.

## 1. Phase-2p twin evidence and rejected admission

**CONFIRMED:** both reported run directories contain 95 records; every one is
raw byte-identical between twins.  The 94 records shared with the Phase-2n BBL
root are raw byte-identical.  The ordinary-output control independently passes
with four restart shards exact and eight history payloads exact.  The retained
measurement is `/tmp/orca2_phase2q_pre_admission_identity.log`, SHA-256
`f9af454aefc0f4517e3465040bf6440497220d290265e9faabb2fc9e31bd36c2`.

**CONFIRMED / REJECTED:**
`oracle_zdf_sh2_operands_kt00000001.bin` is 74,141,252 bytes and identical in
both twins (SHA-256
`8ebfac84d19bc36c5b0b1cfc8381fd225afa0111d09e3d1869efab35ef83d618`),
but the header says 9,358,640 binary64 payload values while EOF contains
9,267,648.  The binding validator exits 1 with the exact header in
`/tmp/orca2_phase2q_rejected_header.log`, SHA-256
`ca09682aad71366c746b3925daea949ef719c22a208fd52b21a0bf440afe87a6`.
The root is preserved and flagged `REJECTED_MALFORMED_ZDF_HEADER`; it is not a
`VARIANT_ORACLE_V2` extension.

The mismatch is the writer's allocation assumption:

| written value | allocation actually passed to `WRITE` |
|---|---|
| `sh2`, `avt_k`, `en` | `A2D(0),jpk` = `90*148*31` each |
| `avm_k` and the other 17 3-D values | full `jpi*jpj*jpk` |
| `taum` | `A2D(0)` = `90*148` |
| the other three 2-D values | full `jpi*jpj` |

Therefore the false excess is exactly
`3*(94*152-90*148)*31 + (94*152-90*148) = 90,992` values.  The Phase-2p
receipt and admission tool now retract the earlier uniform-full-domain count.

## 2. EEN-stream disposition

**CONFIRMED:** the four Phase-2m EEN streams are absent from the Phase-2p ZDF
build because that source assembly applied the SH2 patch without the separate
EEN writer patch.  This was not the Phase-2p preregistered intention.  It is a
WRITE-only acquisition omission, not a physics difference: all 94 shared
records and ordinary outputs pass identity.  The EEN discriminator remains
pinned on
`variant_icebergs_off_phase2m_een_a_10step_np2`; no synthetic combined record
root is claimed.

## 3. WRITE-only correction and replacement build

**CONFIRMED:** the config-local `zdfphy.F90` writer now derives its payload
header by summing `SIZE(...)` for the exact 25 expressions in the subsequent
`WRITE`.  It retains the existing rank-zero, non-tiled, first-step guards and
canonical zero-first views; it assigns no model array.  The admission parser
mirrors the mixed full/A2D allocations and validates exact EOF, finiteness,
and grid-specific canonical slots.  It also walks the inherited ZDF-entry and
SI3-ZDF schemas and routes header, canonical-slot, owned-payload, and identity
plants through the real validators.

The replacement build was made in the disposable config copy
`/tmp/nemo-orca2-phase2p`, not the shipped checkout, with:

```text
env PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:... \
  ./makenemo -n ORCA2_OMIP_L4 -m conda-scalarmath
```

The build log is
`/data/abyssal/dbalwada/nemo-testcases-l4/build_phase2q_zdf_schemafix.log`
(SHA-256
`c3e724ff6f4a5588d1bf65d773cd5420bc4c74cba350ec105dc4db7abe1d1f45`).
The 55,585,472-byte binary is
`/data/abyssal/dbalwada/nemo-testcases-l4/binaries/nemo_ORCA2_OMIP_L4_phase2q_zdf_schemafix.exe`
(SHA-256
`1f3bdea8fa63ea73ee3232666835d1c531fbc655e651dbae0f4c76e160f25fe2`);
`nm -D` finds **zero `_ZGV*` symbols**.

The shipped NEMO checkout already contains the campaign's pre-existing dirty
copies/configurations; Lane 4 made no shipped-tree edit in this phase.  No file
or prior run was deleted.

## 4. Prepared user-shell twins

**CONFIRMED:** each directory contains the byte-copied accepted icebergs-off
deck, the same 40 absolute input symlinks, the new absolute binary symlink,
unchanged `(jpni,jpnj)=(2,1)` CPU layout, and a self-contained Bash-time
launcher.  Both deck and input manifests pass before handoff.

Run these one at a time, unchanged:

- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2q_zdf_schemafix_a_10step_np2/run.sh`
- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2q_zdf_schemafix_b_10step_np2/run.sh`

Launcher SHA-256 values are respectively
`f1f030b108c250f0eee1fe87499ebdfbda0cbc5512460d4a6748419e82515213`
and
`419c925746bf4c1806916157eca10bb009bb78e66a3a78208781b8e86959275d`.

Expected admission after resume is 95/95 raw twin identity, 94/94 raw
inherited identity, exact ordinary outputs, a header-derived 9,267,648-value
SH2 payload, three complete ZDF schema walks, and every plant nonzero.  Any
failure stops again.

## 5. Deferred Phase-2q work

**PLAUSIBLE / UNMEASURED:** no `zdf_sh2` numerical score is reported because
the only new target record failed schema admission.  The first non-bit
statement/cell class and GYRE handoff remain pending the replacement twins.

**PLAUSIBLE / UNMEASURED:** the `_nemo_bbl_scalar_divide` review arm, second
execution of the three BBL gates/plants, and any required HLO/gradient evidence
remain pending.  They were not run out of order after the admission failure.
The requested BBL default decision row has nevertheless been added to the
Phase-2o ASKED table; defaults remain `0` / `0.0` and the decision remains with
the user.

## 6. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| validate reported ZDF twins | ASKED | 95/95 twin and 94/94 inherited exact, then rejected at header |
| accept a reproducible but malformed record | forbidden by oracle rules | not accepted |
| fix header from exact write expressions | ASKED by fail-closed schema rule | completed WRITE-only |
| rebuild and prepare replacement twins | ASKED / necessary rerun | completed; user-shell paths above |
| call EEN omission intended | UNASKED / false | explicitly recorded as unintended patch-series omission |
| score malformed SH2 target | forbidden | deferred |
| change shared SH2 or BBL numerics before admission | out of order | deferred |
| alter BBL defaults | ASKED decision pending | no change |
| sandbox MPI/NEMO | forbidden | none |
| shipped edit, deletion, push | forbidden | none |

Session ID: `01a06d99-f562-7b11-bc63-e9b112877f54`.
