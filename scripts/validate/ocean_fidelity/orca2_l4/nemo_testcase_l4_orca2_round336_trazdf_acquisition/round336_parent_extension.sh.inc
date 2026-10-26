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

sed -i "/same_program \"\$dry_bld\"/i\\
rm -f \"\$dry_bld/ppsrc/nemo/trazdf.f90\" \"\$dry_bld/ppsrc/nemo/trazdf.F90\"\\
cp @SOURCE_PPSRC@ \"\$dry_bld/ppsrc/nemo/trazdf.F90\"\\
patch --batch -s --fuzz=0 -p0 -d \"\$dry_bld/ppsrc/nemo\" <@PATCH@\\
mv -- \"\$dry_bld/ppsrc/nemo/trazdf.F90\" \"\$dry_bld/ppsrc/nemo/trazdf.f90\"\\
cp @WRITER@ \"\$dry_bld/ppsrc/nemo/round336_trazdf_writer.f90\"\\
check_r336_landing \"\$dry_bld/ppsrc/nemo/trazdf.f90\" || refuse 'round-336 placement moved in dry ppsrc/nemo/trazdf.f90'\\
: >\"\$dry_bld/obj/round336_trazdf_writer.o\"" "$ADAPTED"

sed -i "/# makenemo rewrites shared work_cfgs.txt/i\\
sha256sum @PATCH@ @WRITER@ @GATE@ @TEST@ >>\"\$manifest/toolchain.sha256\"" \
  "$ADAPTED"

sed -i "/touch \"\$TARGET_ROOT\/MY_SRC\/\"\*\.F90/i\\
cp @TRAZDF_SOURCE@ \"\$TARGET_ROOT/MY_SRC/trazdf.F90\"\\
cp @WRITER@ \"\$TARGET_ROOT/MY_SRC/round336_trazdf_writer.F90\"\\
patch --batch -s --fuzz=0 -p0 -d \"\$TARGET_ROOT/MY_SRC\" <@PATCH@" \
  "$ADAPTED"

sed -i "/diff <(grep -v/i\\
grep 'NEMO_L4_R336TZ' \"\$TARGET_ROOT/BLD/ppsrc/nemo/round336_trazdf_writer.f90\" >/dev/null \\
  || refuse 'compiled round-336 tra_zdf writer is absent'\\
check_r336_landing \"\$TARGET_ROOT/BLD/ppsrc/nemo/trazdf.f90\" || refuse 'round-336 tra_zdf placement moved under TARGET_ROOT'" \
  "$ADAPTED"
