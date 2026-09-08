# Preregistration: row-10 step-entry N2 bundle and ordered tail

Date: 2026-08-28. Status at commit: **implementation is in progress; no
round-6 matched-state measurement has run**.

This round resumes at row 10. The round-5 artifact localized the first dumped
failure to the Langmuir `rn2b` operand: 9,920/9,920 wet columns failed the
pointwise `1e-15` bar, including all four southern-basin focus columns. Rows
11--32 remain unmeasured and may be scored only after row 10 passes.

## Production selection and scope

`tke_n2_evaluation_stage` has two values. `step_entry` constructs one
trace-local `(rn2, rn2b, gdepw_Kmm, e3w_Kmm)` bundle from the physical
step-entry state and routes it to every TKE consumer of those fields.
`implicit_solve_state` retains the historical reconstruction at the implicit
solve. The generic defaults remain `implicit_solve_state`; only
`nemo_dino_kamm` and inherited `nemo_dino_kamm_mlf` select `step_entry`.
The MLF card constructs genuine Nbb `rn2b`; the forward-Euler card retains its
documented ceiling `rn2b == rn2` because it carries no Nbb state.

The option changes no prognostic-state schema. The omitted/default legacy
selector and explicit `implicit_solve_state` must be exactly array-identical
for every unchanged card. The legacy selector must reject a supplied bundle;
the step-entry selector must reject a missing or shape-incompatible bundle.

The bundle covers only the registered four operands. NEMO also uses live
`e3t(Kmm)` in the row-12 matrix and row-20 length sweeps. That operand remains
ordered-unmeasured and is not silently folded into this fix: if it is the next
failure, the sweep stops there with a separately registered design.

## Frozen bars and stop rule

Row 10 is VERIFIED only when its independently dumped `rn2b` operand has
0/9,920 failing wet columns, maximum normalized column error `<=1e-15`, all
four southern focus columns pass, and perturbation, horizontal-roll, and
nonfinite controls fire. The undumped full Langmuir source remains explicitly
non-dispositive.

On a row-10 pass, rows 11--32 resume in the exact order, classes, bars, and
focus registry already committed in `PREREG_zdf_chain_sweep.md`. The probe
stops at the first DIVERGED row. A row is not promoted from an offline
reconstruction when its registered NEMO checkpoint is absent; it is WAIVED
with a written execution-state reason or remains UNMEASURED pending a dump.

For row 11, the existing `tke_dump_{zri,pdlr,avm_in,sh2,rn2b}.bin` streams
are primary. The literal branch and arithmetic follow
`MY_SRC/zdftke.F90:477-496`; pointwise bar `1e-15`, 0/9,920 and 4/4 focus pass.
Rows 12 onward retain their canonical preregistered bars. In particular,
row 12 walks carried `avm`, live `e3t(Kmm)`, live `e3w(Kmm)`, mask and
association in source order before the matrix composite is dispositioned.

## Climate prediction receipt

Climate arms remain unauthorized until the ordered chain is clean through
row 32. The prediction is unchanged: baseline southern-basin MLD RMS
`22.479491 m`; CONFIRM `<=11.2397455 m`; REFUTE `>=20.2775 m`, with the
previously registered acceptance-floor, pass-tally, legacy-baseline, and
southern-density conditions unchanged.

The eventual faithful arm remains option-free. The historical control gains
the new opt-out:

```text
--tke-n2-evaluation-stage implicit_solve_state
```

in addition to the four round-5 opt-outs. No GPU command is authorized by
this preregistration.

## Dated amendment: row-11 literal arithmetic fix

Date: 2026-08-28. Written after the first row-11 measurement and before the
production change or post-fix rerun. Row 10 passed with 0/9,920 failing wet
columns. Row 11 then failed in 153/9,920 columns at the `1e-15` bar even
though its dumped `rn2b`, carried `p_avm`, and frozen `p_sh2` operands each
passed in all 9,920 columns. All four focus columns passed; that does not
waive the whole-domain failure.

The registered fix is a literal transcription of the two evaluated NEMO
expressions, preserving their source association:

```text
zri    = (rn2b * p_avm) / zdiv
p_pdlr = max(0.1, ri_cri / max(ri_cri, zri))
Pr     = 1 / p_pdlr
```

The exact-zero `zdiv` arm substitutes `rn_bshear` exactly as NEMO does. The
old multiplication by `1/zdiv` and algebraic `clamp((1/ri_cri)*zri,1,10)`
route become the planted red alternatives. An oracle-input diagnostic
reproduced `tke_dump_zri.bin` and `tke_dump_pdlr.bin` bit-for-bit with the
registered expressions; this diagnostic is targeting only and is not the
post-fix score.

Scope is tied to the already registered `tke_n2_evaluation_stage` selector:
the two complete DINO NEMO cards select `step_entry` and therefore literal
NEMO arithmetic; `implicit_solve_state` retains the historical association
byte-for-byte. Thus no additional card changes and the legacy climate control
continues to opt out with the same single N2-stage flag.

The post-fix row-11 bar remains unchanged: 0/9,920 failing columns, maximum
normalized column error `<=1e-15`, 4/4 focus columns passing, and every
control firing. Only after that pass may row 12 be measured.
