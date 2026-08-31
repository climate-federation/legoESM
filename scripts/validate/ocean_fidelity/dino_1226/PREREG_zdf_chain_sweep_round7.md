# Preregistration: row-11 Richardson N2 operand chain

Date: 2026-08-28. Status at commit: **no row-11 intermediate dump has been
compiled or measured**.

Round 6 closed row 10 at 0/9,920 failing columns and stopped at row 11: the
dumped Richardson composite fails in 86/9,920 columns at the pointwise
`1e-15` bar, while all four southern-basin focus columns pass. Substituting
NEMO's independently dumped `rn2b` makes the composite bit-exact. This round
therefore instruments the producing `bn2` call before changing production
arithmetic.

## Ordered checkpoints and bars

The checkpoints follow the live S-EOS arm of
`src/OCE/TRA/eosbn2.F90:1459-1468`, in this exact order:

1. `zrw = (gdepw(Kmm)-gdept(Kmm))/(gdept(k-1,Kmm)-gdept(k,Kmm))`;
2. `zaw`, the thermal-expansion interpolation using that `zrw`;
3. `zbw`, the saline-contraction interpolation using that `zrw`;
4. the complete pre-divisor numerator
   `grav*(zaw*dT-zbw*dS)`;
5. the assigned `/e3w(Kmm)*wmask` result (`rn2b`).

Every checkpoint is scored over the same 9,920 wet DINO columns and all wet
interfaces used by row 11. VERIFIED requires 0/9,920 failing columns,
maximum normalized per-column error `<=1e-15`, and 4/4 registered southern
focus columns passing. The first failing checkpoint is DIVERGED and later
checkpoints are not used to move attribution upstream. Perturbation,
horizontal-roll, and nonfinite controls must each fail. Retractions change
the probe output, not merely prose.

The NEMO instrument writes only the first `bn2` call in
`stpmlf.F90:207-208` (Nbb tracers, `rab_b`, `Nnn` geometry), selected by a
debug-only optional argument whose absent value preserves every other call.
It emits full one-rank bracket streams named
`bn2_dump_{zrw,zaw,zbw,numerator,result}.bin`. The result stream must be
bit-identical to the independently existing `tke_dump_rn2b.bin` on the scored
slice; otherwise the call/time-level registration fails and no attribution is
allowed.

Before the intermediate dumps are admissible, one certified and one
instrumented CPU step must start from identical restart inputs and finish with
every numeric variable in the restart bit-identical. A planted comparison
against the wrong restart step/field must fail. File-container bytes and
metadata are not the state identity statistic.

## Fix decision

If one checkpoint is first to fail, the production design is the smallest
literal transcription that gives that operand the same source association as
NEMO while retaining JAX autodiff/JIT purity. It is selected only through the
already registered `tke_n2_evaluation_stage="step_entry"` path, which is the
faithful default on `nemo_dino_kamm` and inherited
`nemo_dino_kamm_mlf`. The `implicit_solve_state` opt-in and every other card
remain array-identical. A hand-computed source-order case and a planted old
association are required before the post-fix row-11 score.

After row 11 reaches 0/9,920 and 4/4, rows 12--32 resume in the previously
committed order. The sweep stops at the next DIVERGED row. An absent required
checkpoint is not a pass; it is UNMEASURED unless its execution state is
provably inapplicable and written as WAIVED.

Climate arms remain unauthorized. The frozen prediction and eventual command
pair are unchanged until an end-to-end row-32 pass.

## Dated amendment: measured owner and registered production fix

Date: 2026-08-28. Written after the first instrumented row-11 run and before
the production change or post-fix rerun.

The first numeric divergence is `zrw`: 9,920/9,920 wet columns fail its
`1e-15` bar (maximum normalized column error `6.470553e-15`), including all
four focus columns. Against `eosbn2.F90:1459-1460`, NEMO expands the qco macros
literally as four independently rounded `raw_depth*(1+r3t)` operands. Its
`r3t` is itself `pssh*r1_ht_0` (`domqco.F90:160`), where `r1_ht_0` was stored
earlier as the reciprocal in `domain.F90:158`. legoESM instead formed
`eta/H_bathy`, then passed preassembled live ladders into the ratio. The
mathematically cancelling stretch therefore had a different fp64 association.
An offline source-order control using `eta*(1/H_bathy)` and the four raw-mesh
multiplications reproduces the new NEMO `zrw` stream bit-for-bit; the old
static/cancelled ratio differs in every wet column.

The registered fix is confined to the already selected
`tke_n2_evaluation_stage="step_entry"` bundle. It constructs `r3t` in NEMO's
reciprocal-then-multiply order, carries the raw `gdept_0/gdepw_0` operands,
and evaluates `zrw` from four separately materialized
`raw_depth*(1+r3t)` values before `zaw/zbw`. JAX optimization barriers may be
used only at those four rounding boundaries to prevent algebraic cancellation;
JIT and gradients must remain live. `implicit_solve_state`, all non-oracle
cards, and all other `nemo_r3t_stretch` consumers retain their old quotient
and preassembled-ladder path byte-for-byte. The selector is not widened.

Red tests require a hand-computed column where reciprocal/multiply differs
from quotient, the old cancelled-`zrw` alternative to fail, the NEMO source
order to pass, JIT/gradient finiteness, and unchanged-card array identity.
The post-fix row-11 target remains 0/9,920 at `1e-15` and 4/4 focus; the exact
NEMO `zrw` stream is additionally expected bit-identical. Only then may row
12 be measured.

**LOUD RETRACTION of one instrumentation-control clause.** The initial text
required every numeric restart variable to match an older certified binary.
That binary has 95 numeric restart variables; this sandbox's same-source
dump-on/off builds have 131 because they already include the campaign budget
accumulators, and their pre-existing literal-tau work changes only
`utrd_tau`. The older binary is therefore not the bracket for this patch.
The valid controlled pair is identical-source dump-on versus dump-off:
131/131 variables present and 0 differing. The committed probe prints the
certified comparison as `RETRACTED` and lists `utrd_tau`; it cannot print the
withdrawn all-numeric claim. The planted wrong-field comparison fires.
