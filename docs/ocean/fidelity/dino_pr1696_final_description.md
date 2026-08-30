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
> **Climate qualification:** this PR makes no present-day MLD, basin-transport,
> or wall-flicker improvement claim. The earlier MLD refutation, day-360 basin
> floor, and wall-epoch attribution all predate later tracer-velocity,
> momentum-couple, Redi and A33 fixes. A fresh duplicate-controlled re-battery
> against the current faithful defaults is committed and preregistered; its
> MLD-band, day-360 basin and five-day wall verdicts must replace those stale
> epochs only after the GPU arms pass their identity and provenance gates.
>
> Validation includes the focused selector/physics suites, red-capable scorer
> controls, deterministic NEMO stream brackets, all ordered round result
> receipts, the full-step synthetic-unaccounted-call plant, and the current
> climate-rebattery classifier/default-card tests. No bar constant was relaxed.

This final description supersedes the earlier round-16 and wall-bisect
boundary snapshots only because the same branch was subsequently extended by
explicit campaign direction through registry end. Those historical documents
remain unchanged because the coverage gate hash-admits them as receipts.
