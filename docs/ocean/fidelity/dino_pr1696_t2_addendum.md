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
> MLF-only. The authorized continuation has now admitted one additional FE
> default: `barotropic_time_filter=nemo_boxcar_ab3`. NEMO's first and only
> Euler bootstrap still runs the nn_bt_flt=2 AB3 velocity predictor and
> `ts_bck_interp` SSH interpolation; repeating that first-step composition is
> the minimal source-grounded stabilization for legoESM's perpetual-Euler card.
> It passes the registered 40-step discriminator and 64-step clean-card gate.
> This does not make perpetual Euler oracle-faithful—NEMO switches to MLF after
> one step—and does not establish FE climate fidelity.
>
> The committed developed-state reproducer is the non-vacuous regression gate:
> its planted old `nemo_boxcar_centred` arm fails at step 36, the one-field
> AB3/AM4 arm completes 40 steps, and the shipped default completes 64. A
> short synthetic five-step CI test would remain vacuous, so the full clean
> CPU receipt—not a fabricated short green test—carries the runnability claim.

Evidence hashes:

- all-sweep-selectors-restored JSON:
  `6392216940040b02184f1c2d4a3da99adab1dfda55a7f833f7bc3a0318cbe4d1`;
- all-restored log:
  `a70c489bae288b657113401ee974928d66a6fd78e7bfaf0c320b07e5c9fa6fed`;
- clean pre-#1696 two-day artifact:
  `9f97a8ba82800672116a7f60c1cb1043dfc1aab7b51ab230311bced1284f2839`;
- clean pre-#1696 log:
  `4f5f128c7ea83ebfe3d81a41da4b29146b4d744e0862a573b3c9f04605218464`;
- live fast-term plain-filter control JSON/log:
  `c2623575caf21283f0abe820873f94611f5d401cd3f32e611fd2cad967dcf26c` /
  `15936968c8cb9197b6ec8568d369c64bbb8bf4d004c425c05096292274790a38`;
- one-field AB3/AM4 40-step JSON/log:
  `eb7a378a8eb5c85294e0591db17927bfc1f52fb12c6e7fbf3700aebef794c92e` /
  `7c91c5c77732b0f979a05de477f3ff744aaecab03226187c9a8e0902c90c7a2d`;
- promoted-default 64-step JSON/log:
  `1d2da9994bc18246a05de5ac20ad5d333ee1e42b83447e2cbac18afdb735ef38` /
  `1d5649ffe67344ace9125be863e5baf2f8447b21bb866d8c562a9fe2cbd5e724`.
