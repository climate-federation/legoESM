#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_wv
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_wv_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of ONE new instrument in the FV3 duo-grid JAX port:
`git show HEAD` -- scripts/validate/fv3_native/compare_wind_vectors.py
plus scripts/cluster/fv3_native/wind_vector_parity.sbatch. Cite file:line,
classify BLOCKER / MAJOR / MINOR, give the smallest fix, and say explicitly
where you found NOTHING. This is a measurement instrument: a bug here
manufactures a false parity certificate, so treat a wrong number the same as
a wrong result.

ROUND 2. Round 1 raised five findings, now addressed at HEAD: (1) es/ew sign is
DERIVED from the op (edge_expected_sign) and the data's best-fit sign is required
to agree, so a negated port fails; (2) the self-controls and interior gate run on
the scored interior mask (the oracle zero-fills unwritten slots, so finite is no
longer treated as written); (3) the required interior must be finite on both
sides -- no NaN masked to zero; (4) gate tightened to 1e-13; (5) the runner
refuses a dirty tree. VERIFY each fix is real and non-vacuous, then keep
attacking -- do not assume round 1 was exhaustive.

WHAT IT DOES. It scores the port's compute_fv3_native_wind_vectors
(packages/core/legoesm/grids/fv3_native_metrics.py:693) -- the physics-coupling
unit vectors vlon/vlat (cell-centre east/north) and es1/ew2 (D-grid edge
tangents) -- against the pinned Fortran gridstruct dumped by the extchain
oracle driver as M_VLON{1,2,3} / M_VLAT{1,2,3} / M_ES{1,2,3} / M_EW{1,2,3}
(three Cartesian components each). It reuses load_oracle / derive_face_map /
op_scalar from compare_extchain_oracle.py, and the face map is the searched
bijection the metric certificate already uses.

Attack these specific points; do not stop at the first finding:

1. WINDOWING (crop_to_oracle). The port builds vectors on the FULL data domain
   (ng=3 halo), verified by tests/grids/test_fv3_wind_vectors.py (asserts finite
   at [NG:NG+N, NG:NG+N+1]). vlon/vlat are dumped by the oracle only on
   is-2:ie+2 (halo 2), es/ew on the full isd:ied (halo 3). crop_to_oracle trims
   (port-oracle)/2 rings symmetrically. Is the symmetric trim the CORRECT
   physical alignment -- i.e. after the trim does oracle index 0 correspond to
   the same physical cell as port index (trim)? Prove it for vlon (54->52) and
   for the staggered es/ew (no trim) separately. A plausible-but-off-by-one
   crop gives an O(1) residual, but could any crop give a SMALL wrong residual?

2. THE CARTESIAN COMPONENT AXIS. op_scalar is applied per component and the
   code asserts the 3-vector is Earth-fixed so the component axis is never
   transformed (unlike op_vector, which sign-flips covariant components under a
   reversal). Is that correct for vlon/vlat (absolute east/north) AND for es/ew
   (edge tangents whose DIRECTION is tied to grid orientation)? If a reversal
   op physically reverses an edge tangent, is that captured by the single global
   sign resolve_sign_vec returns (which flips all three components together), or
   could the true transform need a PER-COMPONENT sign or a component permutation
   that a single global sign silently cannot represent -- passing at floor for
   the wrong reason, or failing when the vectors are in fact correct?

3. THE es<->ew SWAP. Under a transposing op the code maps port es1 to oracle
   M_EW (SWAP_PARTNER) and crops to its OWN family's oracle shape first
   (_oshape_for). Is crop-then-op-then-compare-to-partner the right order, and
   is the partner's shape what op_scalar actually produces? Check a transposing
   face concretely.

4. THE GATE. WIND_VECTOR_PARITY_OK requires per-family interior <= PARITY_FLOOR
   (1e-11) AND no determined -1 sign on vlon/vlat. There is a liveness guard
   (raise if a face has no live interior cell, and if total live < 6*(n-2)^2/2).
   Can the gate still pass vacuously or on a masked-away defect? Is the
   sentinel/NaN handling (np.where(live,...)) hiding a real disagreement rather
   than an unwritten cell? Is PARITY_FLOOR defensible for a float64 primitive
   built from mid_pt3 cross products, or is it loose enough to admit a real gap?

5. THE ORACLE SELF-CONTROLS (|v|-1, vlon.vlat). Are they non-vacuous -- would
   they actually fail on a corrupted dump or a wrong reader, or can they pass
   while the scored comparison is meaningless?

6. Anything in the sbatch: does it read the committed tree at a known SHA, and
   would a partial/failed run report OK? Reuse of the run-9470130 dumps -- are
   those the SAME grid config (C48/C12, ng=3, grid_type 0, do_schmidt identity)
   the port builds?
PROMPT
)" 2>&1

echo "CODEX_WV_REVIEW_DONE"
