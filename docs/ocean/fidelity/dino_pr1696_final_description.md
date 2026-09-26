# PR #1696 final description (ready to paste)

Suggested title: **Complete DINO/NEMO literal step chain: ZDF, split-explicit
momentum, and Redi tracer tail**

> **Status: COMPLETE. Merge dependency: #1695 must land first.** This branch
> consumes #1695's faithful T-point reconstruction of NEMO's carried U/V-face
> restart stress; merge #1695 first, then rebase #1696 and drop any
> patch-equivalent carry commits. Do not reverse that order.
>
> This PR closes the registered, NEMO-ordered DINO production walk from the
> ZDF/TKE state through split-explicit momentum and the Redi/ZDF/Asselin tracer
> tail. It makes the two DINO NEMO cards select the faithful paths while all
> non-DINO recipes retain byte-pinned historical defaults. The implementation
> covers the carried step-entry TKE/N2/shear bundle and literal TKE matrix,
> recurrence, Langmuir and implicit ZDF arithmetic; NEMO's barotropic seed,
> EEN coefficient, V-face metric, continuity, PGF and transport association;
> the coupled live-QCO W/thickness vertical-advection and second-WZV/MLF carry;
> the tracer-velocity write; and live-Kmm Redi face thickness, horizontal,
> vertical-skew, W-slope and A33 associations. Selector validators reject
> incoherent combinations, and red controls prove the legacy paths remain
> discriminating.
>
> The ordered receipts close the split-explicit loop at every one of its 68
> substeps (no strict SSH/U/V failure; final U/V errors
> `4.8e-16/4.4e-16`, below the registered `1.4e-14` bound), close row-4
> vertical advection with the coupled QCO path, and advance the tracer tail
> through T and S Redi fluxes and the implicit/Asselin end. The final S `zfw`
> arithmetic floor is explicitly cleared under Rule 1b after all registered
> association alternatives were exhausted; it is not relabelled bit-exact.
>
> The fetched master `stp_MLF` registry gate SHA-admits every lane result
> document. Final strict active coverage is **34/37 = 91.891892%**; accounting
> is **116/116**, with **0 UNCOVERED**, 79 explicit inactive/nonphysics/
> instrumentation waivers, and only three visible unmeasured rows:
> `ldf_dyn`, `tra_sbc`, and `tra_qsr`. Those three are follow-up measurement
> debt, not hidden omissions from this physics chain.
>
> **Current-default climate re-battery: all three registered bars pass.**
> Bit-identical duplicate arms pass every provenance and planted
> control. Southern-basin day-90 MLD RMS is `0.000103797 m` (`0.103797 mm`,
> **CONFIRM**); the day-360 basin gap is `-0.0231838 Sv` against the historical
> `-0.9519122 Sv` (`+0.9287285 Sv`, `15.043F`, NEMO frame `10.0160712 Sv`,
> **CONFIRMED**); and the five-day wall ratio/share is
> `1.1997321/0.1091188` (**CONFIRMED**). These replace the stale pre-tail
> climate epochs for the combined `nemo_dino_kamm_mlf` faithful default; they
> do not attribute the closure among individual fixes. The same selectors
> default on `nemo_dino_kamm`, but this battery did not climate-test that card.
> Scope remains the state-initialized twin with NEMO before/T-stress/TKE
> bridging, not yet a standalone-initialization or cross-recipe transfer
> claim.
>
> Validation includes the focused selector/physics suites, red-capable scorer
> controls, deterministic NEMO stream brackets, all ordered round result
> receipts, the full-step synthetic-unaccounted-call plant, and the current
> climate-rebattery classifier/default-card tests, and the completed battery
> artifact SHA-256
> `db966e97b214151789e203cc6058d0c0a3564071cc92e301b5ac63aa8b7f693e`.
> No bar constant was relaxed. The only residual measurement debt is
> `ldf_dyn`/`tra_sbc`/`tra_qsr`; separately, salinity `zfw` retains its
> explicit Rule-1b arithmetic-floor qualification. Two independent closing
> reviews are **PASS/PASS** after their recipe-scope, scored-storage,
> plant/gate-vocabulary, residual-debt and exact-verdict-label findings were
> corrected.

This final description supersedes the earlier round-16 and wall-bisect
boundary snapshots only because the same branch was subsequently extended by
explicit campaign direction through registry end. Those historical documents
remain unchanged because the coverage gate hash-admits them as receipts.
