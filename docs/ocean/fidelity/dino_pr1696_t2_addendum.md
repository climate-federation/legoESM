# PR #1696 description addendum — T2 FE admission correction

Ready-to-paste addendum, 2026-08-30. Session
`01a053d4-8e9f-7212-bbdb-19ba2d64e140`.

> **Post-sweep transfer admission correction.** The climate re-battery in this
> PR certifies only `nemo_dino_kamm_mlf`. The permanent-forward-Euler sibling
> `nemo_dino_kamm` cannot currently enter the same developed-state battery: the
> current card becomes unphysical and reaches the checked
> `raw-mesh e3w_int must contain only finite values > 0` error at step 36--37.
> This is an FE-stabilization debt and no T2 climate verdict was issued.
>
> A preregistered CPU bisection then restored every literal evaluation selector
> promoted by the sweep—barotropic, TKE/ZDF, Redi, wind, and V-metric—while
> retaining the unrelated FE damping fence. That all-restored arm still
> exploded and raised the same check at step 37. More decisively, the clean
> pre-#1696 target commit `b794c0618e287ebf1d364a8713c3c504ac2eb01c`
> itself becomes fully nonfinite at its first day/step-32 checkpoint under the
> same developed-state, before-level, TKE, fp64 CPU reproduction. Therefore the
> proposed “#1696 introduced the FE explosion” mechanism is **refuted**: no
> promoted selector passes the paired activation/restoration causality gate,
> and withdrawing another selector would be an unmeasured patch rather than a
> regression fix.
>
> The already-established frame fences remain: FE keeps
> `barotropic_diffusion_alpha=0.01` and generic `zad_qco_evaluation` /
> `wzv_call2_evaluation`; the zero-alpha and coupled Nbb/Kaa literal forms are
> MLF-only. No additional default is scoped in this addendum. Faithful FE
> equivalents and a stable permanent-FE composition remain registered future
> work. The honest PR claim is unchanged but narrower in emphasis: #1696 ships
> and climate-certifies the MLF card, exposes faithful selectors on the sibling
> where guarded, and does **not** establish FE developed-state runnability or
> climate fidelity.
>
> The requested long-window green FE regression test cannot be added honestly:
> both sides of the alleged before/after boundary fail. A five-step test would
> be vacuous—the registered failure is at step 32 or later. The committed
> reproducer and preregistration instead fail closed and preserve the exact FE
> stabilization block for a distinct round.

Evidence hashes:

- all-sweep-selectors-restored JSON:
  `6392216940040b02184f1c2d4a3da99adab1dfda55a7f833f7bc3a0318cbe4d1`;
- all-restored log:
  `a70c489bae288b657113401ee974928d66a6fd78e7bfaf0c320b07e5c9fa6fed`;
- clean pre-#1696 two-day artifact:
  `9f97a8ba82800672116a7f60c1cb1043dfc1aab7b51ab230311bced1284f2839`;
- clean pre-#1696 log:
  `4f5f128c7ea83ebfe3d81a41da4b29146b4d744e0862a573b3c9f04605218464`.
