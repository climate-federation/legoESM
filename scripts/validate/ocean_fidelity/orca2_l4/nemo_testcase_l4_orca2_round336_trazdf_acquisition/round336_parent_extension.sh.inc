# Expanded into the imported round-333 wrapper immediately before it seals
# its inner driver. These commands edit only the inherited reviewed hooks.
sed -i 's|[^ ]*nemo_testcase_l4_orca2_round333_traldf_gate.py|@GATE@|' \
  "$ADAPTED"
if ! grep -F '"$PY" @GATE@ --same-program' "$ADAPTED" >/dev/null; then
  grep -n -- '--same-program' "$ADAPTED" >&2 || true
  refuse 'round-336 same-program gate substitution missed'
fi
sed -i 's|--observer-patch "traldf_iso=\$scratch/traldf_combined.patch" \\|--observer-patch "traldf_iso=\$scratch/traldf_combined.patch" --observer-patch "trazdf=@PATCH@" \\|' \
  "$ADAPTED"
sed -i 's|--observer-patch "zdfphy=\$TKE_PHY_PATCH"|--observer-patch "zdfphy=\$scratch/zdfphy_combined.patch"|' \
  "$ADAPTED"

sed -i "/for unit in dynzdf stprk3_stg; do/i\\
cp @WRITER@ \"\$dry/round336_trazdf_writer.F90\"\\
patch --batch -s --fuzz=0 -p0 -d \"\$dry\" <@ZDFPHY_PATCH@\\
check_r336_zdfphy_landing \"\$dry/zdfphy.F90\" || refuse 'round-336 zdf_phy placement moved in syntax source'" \
  "$ADAPTED"
sed -i 's|syntax_units=(dynzdf_round328_writer tke_round331_writer round332_writer round333_traldf_writer \\|syntax_units=(dynzdf_round328_writer tke_round331_writer round332_writer round333_traldf_writer round336_trazdf_writer \\|' \
  "$ADAPTED"

sed -i "/# Same-program preflight:/i\\
if diff -u --label zdfphy.F90 --label zdfphy.F90 \"\$SOURCE_ZDFPHY\" \"\$dry/zdfphy.F90\" >\"\$scratch/zdfphy_combined.patch\"; then\\
  refuse 'round-336 combined zdf_phy observer patch is empty'\\
else\\
  [[ \$? -eq 1 ]] || refuse 'could not construct combined zdf_phy observer patch'\\
fi" "$ADAPTED"

sed -i "/same_program \"\$dry_bld\"/i\\
fpp @TRAZDF_SOURCE@ \"\$scratch/trazdf_r336_cpp.f90\"\\
awk 'NF' \"\$scratch/trazdf_r336_cpp.f90\" >\"\$scratch/trazdf_r336_cpp.nonblank\"\\
awk 'NF' @SOURCE_PPSRC@ >\"\$scratch/trazdf_r336_record.nonblank\"\\
cmp -s \"\$scratch/trazdf_r336_cpp.nonblank\" \"\$scratch/trazdf_r336_record.nonblank\" || refuse 'build-cpp trazdf differs from record ppsrc'\\
sed '0,/SUBROUTINE tra_zdf(/s//SUBROUTINE tra_zdf_planted(/' \"\$scratch/trazdf_r336_cpp.nonblank\" >\"\$scratch/trazdf_r336_cpp.plant.nonblank\"\\
plant_must_refuse r336-build-cpp-same-program cmp -s \"\$scratch/trazdf_r336_cpp.plant.nonblank\" \"\$scratch/trazdf_r336_record.nonblank\"\\
rm -f \"\$dry_bld/ppsrc/nemo/trazdf.f90\" \"\$dry_bld/ppsrc/nemo/trazdf.F90\"\\
cp \"\$scratch/trazdf_r336_cpp.f90\" \"\$dry_bld/ppsrc/nemo/trazdf.F90\"\\
patch --batch -s --fuzz=0 -p0 -d \"\$dry_bld/ppsrc/nemo\" <@PATCH@\\
mv -- \"\$dry_bld/ppsrc/nemo/trazdf.F90\" \"\$dry_bld/ppsrc/nemo/trazdf.f90\"\\
cp @WRITER@ \"\$dry_bld/ppsrc/nemo/round336_trazdf_writer.f90\"\\
check_r336_landing \"\$dry_bld/ppsrc/nemo/trazdf.f90\" || refuse 'round-336 placement moved in dry ppsrc/nemo/trazdf.f90'\\
: >\"\$dry_bld/obj/round336_trazdf_writer.o\"" "$ADAPTED"

sed -i "/# makenemo rewrites shared work_cfgs.txt/i\\
sha256sum @PATCH@ @ZDFPHY_PATCH@ @ZDFPHY_PPSRC_PATCH@ @WRITER@ @NAMELIST_GATE@ @GATE@ @TEST@ >>\"\$manifest/toolchain.sha256\"" \
  "$ADAPTED"

sed -i "/touch \"\$TARGET_ROOT\/MY_SRC\/\"\*\.F90/i\\
cp @TRAZDF_SOURCE@ \"\$TARGET_ROOT/MY_SRC/trazdf.F90\"\\
cp @WRITER@ \"\$TARGET_ROOT/MY_SRC/round336_trazdf_writer.F90\"\\
patch --batch -s --fuzz=0 -p0 -d \"\$TARGET_ROOT/MY_SRC\" <@PATCH@\\
patch --batch -s --fuzz=0 -p0 -d \"\$TARGET_ROOT/MY_SRC\" <@ZDFPHY_PATCH@" \
  "$ADAPTED"

sed -i "/diff <(grep -v/i\\
grep 'NEMO_L4_R336TZ' \"\$TARGET_ROOT/BLD/ppsrc/nemo/round336_trazdf_writer.f90\" >/dev/null || refuse 'compiled round-336 tra_zdf writer is absent'\\
check_r336_landing \"\$TARGET_ROOT/BLD/ppsrc/nemo/trazdf.f90\" || refuse 'round-336 tra_zdf placement moved under TARGET_ROOT'\\
check_r336_zdfphy_landing \"\$TARGET_ROOT/BLD/ppsrc/nemo/zdfphy.f90\" || refuse 'round-336 zdf_phy placement moved under TARGET_ROOT'" \
  "$ADAPTED"

# Test-only redirection. Production invocations never set the variable.
sed -i '/^TARGET_RUN=\$DEFAULT_TARGET_RUN$/i\
if [[ -n ${R336_STUB_FLOW_ROOT:-} && ${R328_ADMISSION_IDENTITY_ONLY:-0} != 1 ]]; then\
  [[ -d $R336_STUB_FLOW_ROOT && $R336_STUB_FLOW_ROOT == /* ]] || { printf '\''REFUSE: invalid stub flow root\\n'\'' >&2; exit 64; }\
  EVIDENCE=$R336_STUB_FLOW_ROOT/evidence\
  DEFAULT_TARGET_RUN=$EVIDENCE/$(basename "$DEFAULT_TARGET_RUN")\
fi' "$ADAPTED"
sed -i '/^pin() {/i\
if [[ -n ${R336_STUB_FLOW_ROOT:-} ]]; then\
  source @FLOW_LIB@\
fi\
' "$ADAPTED"

# The flow exercise runs this generated driver itself.  It redirects only the
# writable build/run roots and substitutes makenemo/mpirun/NEMO at their real
# call sites; the inherited mode, cleanup, retention, and admission branches
# remain the branches under test.
sed -i '/^readonly TARGET_ROOT=\$NEMO_ROOT\/cfgs\/\$TARGET_CFG$/c\
TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG\
if [[ -n ${R336_STUB_FLOW_ROOT:-} ]]; then TARGET_ROOT=$R336_STUB_FLOW_ROOT/build/$TARGET_CFG; fi\
readonly TARGET_ROOT' "$ADAPTED"
sed -i '/^export PATH=.*nemo-build\/bin/a\
if [[ -n ${R336_STUB_FLOW_ROOT:-} ]]; then PATH=$R336_STUB_BIN:$PATH; export PATH; fi' \
  "$ADAPTED"
sed -i '/^admit() {$/a\
  if [[ -n ${R336_STUB_FLOW_ROOT:-} ]]; then r336_stub_admit "$TARGET_RUN"; return 0; fi' \
  "$ADAPTED"
sed -i 's|^\./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key '\''key_xios'\''$|if [[ -n ${R336_STUB_FLOW_ROOT:-} ]]; then r336_stub_prepare_target "$TARGET_ROOT"; else ./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key '\''key_xios'\''; fi|' \
  "$ADAPTED"
sed -i 's|^\./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key '\''key_xios'\''$|if [[ -n ${R336_STUB_FLOW_ROOT:-} ]]; then r336_stub_finish_target "$TARGET_ROOT" "$dry_bld"; else ./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key '\''key_xios'\''; fi|' \
  "$ADAPTED"

source @FLOW_LIB@
bash -n "$ADAPTED" || refuse 'generated round-336 inner driver has invalid shell syntax'
r336_check_driver_body "$ADAPTED" inner \
  || refuse 'generated round-336 inner driver body is incomplete'
cp -- "$ADAPTED" "$driver_dir/run.syntax-plant.sh"
printf '\nif then\n' >>"$driver_dir/run.syntax-plant.sh"
plant_must_refuse r336-inner-driver-syntax bash -n "$driver_dir/run.syntax-plant.sh"
printf 'R336_GENERATED_DRIVER_SYNTAX_PASS role=inner path=%s\n' "$ADAPTED"
if [[ -n ${R336_GENERATED_INNER_COPY:-} ]]; then
  cp -- "$ADAPTED" "$R336_GENERATED_INNER_COPY"
  cp -- "$COMBINED_WRITER" "$R336_GENERATED_INNER_COPY.round332_writer.F90"
  sed -i "s|^readonly R332_WRITER=.*|readonly R332_WRITER=$R336_GENERATED_INNER_COPY.round332_writer.F90|" \
    "$R336_GENERATED_INNER_COPY"
  chmod 700 "$R336_GENERATED_INNER_COPY"
fi
