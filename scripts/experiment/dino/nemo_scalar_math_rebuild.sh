#!/usr/bin/env bash
# Rebuild NEMO's DINO with SCALAR transcendentals, and regenerate mesh_mask.nc.
#
# WHY.  legoESM's analytic DINO mesh
# (packages/ocean/legoesm/ocean/fidelity/nemo_dino_mesh.py) reproduces every
# field of NEMO's mesh_mask BIT-EXACTLY except ff_t/ff_f, which differ by <=3
# ulp.  The cause is READ OFF THE SHIPPED BINARY, not inferred:
#
#   objdump --disassemble='__usrdef_hgr_MOD_usr_def_hgr' \
#       $NEMO/cfgs/DINO/BLD/bin/nemo.exe | grep -c _ZGVbN2v_sin@plt
#   -> 14
#
# `-O3` vectorised usr_def_hgr.F90:152-153's two whole-array Coriolis
# assignments into glibc's 2-wide VECTOR sine (accurate to 4 ulp), while the
# elementwise cos/asin/tanh in the DO_2D loop stayed scalar and correctly
# rounded -- which is exactly why every other mesh field is bit-exact and only
# these two are not.  Rebuilding without the vectoriser should make ff_t/ff_f
# bit-exact too, and would let the gate's FF_ULP_WAIVER be deleted.
#
# WHAT THIS SCRIPT DOES NOT DO.  It does not touch cfgs/DINO, its arch file, or
# any existing RUN_* directory: it creates a SEPARATE arch and a SEPARATE
# configuration, so every certified artefact keeps its provenance.  Run it
# yourself; nothing in legoESM invokes makenemo or mpirun.
#
# PRE-REGISTERED OUTCOME, so this is falsifiable rather than hopeful:
#   CONFIRMS the diagnosis -> step 4 prints 0 vector-sine calls AND step 6's
#                             gate prints ff_t/ff_f with 0 cells unequal.
#   REFUTES  it             -> step 4 prints 0 but ff_t/ff_f still differ; then
#                             the residual is NOT the vector sine and the
#                             waiver's stated cause is wrong -- say so.
#   INCONCLUSIVE            -> step 4 still prints a nonzero count: the flag
#                             below did not disable the vectorised call, and no
#                             conclusion about ff_t can be drawn from step 6.
set -euo pipefail

NEMO=${NEMO:-$HOME/oracle-builds/nemo5/nemo_5.0.2}
CFG=${CFG:-DINO_SCALARMATH}
SRC_CFG=${SRC_CFG:-DINO}
ARCH=${ARCH:-conda-scalarmath}
JOBS=${JOBS:-8}

cd "$NEMO"

# 1. A separate arch file: same toolchain, vectoriser and FP contraction off.
#    -fno-tree-vectorize is the blunt instrument on purpose -- it is the one
#    flag that certainly removes the libmvec call; -ffp-contract=off removes
#    FMA as a second possible source of last-bit disagreement.
sed -e 's/^%FCFLAGS .*/%FCFLAGS             -fdefault-real-8 -O2 -fno-tree-vectorize -ffp-contract=off -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch/' \
    -e 's/^%FFLAGS .*/%FFLAGS              %FCFLAGS/' \
    arch/arch-conda.fcm > "arch/arch-${ARCH}.fcm"
grep -E '^%FC(FLAGS|)' "arch/arch-${ARCH}.fcm"

# 2. Copy the configuration (MY_SRC + cpp keys) under a new name.
./makenemo -n "$CFG" -r "$SRC_CFG" -m "$ARCH" -j 0
cp -f "cfgs/${SRC_CFG}/cpp_${SRC_CFG}.fcm" "cfgs/${CFG}/cpp_${CFG}.fcm"
sed -i "s/${SRC_CFG}/${CFG}/g" "cfgs/${CFG}/cpp_${CFG}.fcm" || true

# 3. Build.
./makenemo -n "$CFG" -m "$ARCH" -j "$JOBS"

# 4. THE CHECK THAT DECIDES WHETHER STEP 6 MEANS ANYTHING.
echo "vector-sine calls in usr_def_hgr (want 0):"
objdump --disassemble='__usrdef_hgr_MOD_usr_def_hgr' "cfgs/${CFG}/BLD/bin/nemo.exe" \
  | grep -c '_ZGVbN2v_sin@plt' || true

# 5. One step, only to make it write mesh_mask.nc, with RUN_TRAJ's namelist.
RUN="cfgs/${CFG}/RUN_MESH"
mkdir -p "$RUN" && cd "$RUN"
cp -f "../../${SRC_CFG}/RUN_TRAJ/namelist_cfg" .
cp -f "../../${SRC_CFG}/EXP00/namelist_ref" . 2>/dev/null || true
sed -i 's/^\( *nn_itend *= *\).*/\11/' namelist_cfg
ln -sf ../BLD/bin/nemo.exe nemo
mpirun -np 1 ./nemo
cd "$NEMO"

# 6. Re-run legoESM's gate against the regenerated mesh.
cat <<EOF

Now, from the legoESM checkout:

  JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python \\
    scripts/validate/ocean_fidelity/dino_1226/nemo_dino_mesh_gate.py \\
    --mesh-mask ${NEMO}/${RUN}/mesh_mask.nc

Read the ff_t / ff_f rows against the pre-registration at the top of this file.
If they are EXACT, delete FF_ULP_WAIVER from the gate and record the rebuild's
mesh_mask sha256 next to the numbers.
EOF
