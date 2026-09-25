#!/bin/bash
#SBATCH --job-name=cdx_frztke3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_frztke3_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_frztke3_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Re-review commit 6ce01f873 (git show 6ce01f873) of the frozen-column TKE twin,
scripts/validate/ocean_fidelity/frozen_column_tke_twin.py + its sbatch. Your prior
review (v1) returned NO-GO with 10 findings; this rewrite claims to fix them by
switching the NEMO side to the day-30 INSTANTANEOUS restart (ORCA1_00000720_
restart_oce_*.nc, 32 tiles) and feeding the closure NEMO's OWN en for a single
evaluation, comparing to NEMO avm_k/avt_k (the pure TKE-closure diffusivities per
field_def_nemo-oce.xml). VERIFY each prior finding is actually resolved AND audit
the NEW code the rewrite introduced. Numbered, severity-tagged:
A. Prior #6 (5-day-mean): is the NEMO side now a genuine instant? Does closure_K
   feed NEMO's restart en as tke_old and return K CONSISTENT WITH THAT en, or does
   one dt=3600 step evolve en so far that the returned K reflects a different en?
   If the closure updates en internally before computing K, the 'single evaluation
   with NEMO's en' premise is violated — check the tke_vertical_mixing internal
   order (production/dissipation then K) and whether n_iterations/dt is chosen so K
   reflects the INPUT en.
B. Restart stitcher: is the 32-tile reassembly by DOMAIN_position_first/last
   correct, including NEMO's eliminated all-land tiles and halo overlaps? Could a
   tile's global placement be off-by-one, or a halo double-count a column in the
   equatorial box? Verify un/vn centering (west/east face average) matches NEMO's
   C-grid convention and that avm_k/avt_k are read at the right vertical level
   (W-points) with the right sign/units.
C. Prior #1 interface ladder, #2 staggering, #3 t_depth/w_depth, #5 reduction
   parity, #7 eos-from-manifest: confirm each is genuinely fixed in the code, not
   just in comments. For #5 confirm ONE reduction function runs on ours, stored,
   AND NEMO with identical masking/order.
D. Non-vacuity (prior #9): STEP 2 now does N2 x4 (T,S deviation x2) expecting band
   K x0.5 vs control_tol 1.5. Is K ~ 1/sqrt(N2) the right sensitivity in the
   65-105 m FLOORED-TKE band, and does x0.5 reliably fall outside [0.667,1.5] for
   the plausible baseline ratio? Could the baseline control ratio itself sit near
   the tol edge so the unperturbed STEP 1 spuriously fails?
E. Any NEW correctness bug from the near-rewrite (527+/598- lines): shapes, dtype,
   a silent fallback, a masked-NaN leak, a gate that cannot fail, or a defaulted
   physical input that slipped back in.
GO/NO-GO on trusting the NEMO twin number if STEP 1 and STEP 2 gates pass." > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
