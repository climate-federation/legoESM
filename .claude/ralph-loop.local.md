---
active: true
iteration: 88
session_id: 
max_iterations: 500
completion_promise: DONE
started_at: "2026-04-15T05:09:25Z"
---

Make FV3 cubed-sphere implementation faithful to the original Fortran implementation in ../atmos_cubed_sphere-symmetryclean. Work through docs/fv3_fortran_fidelity_review_20260414.md systematically. Required evaluations: cosine bell, Williamson 2, Williamson 5, ocean rest state. Required: no visible edge artifacts, conservation within numerical errors, implementation matches Fortran oracle.
