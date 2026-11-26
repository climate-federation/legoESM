#!/bin/bash
# Instrument self-check build: the UNCHANGED CLOUDSC dwarf Fortran program
# (dwarf_cloudsc + cloudsc_driver_mod + its cloudsc.F90 + common modules),
# with only hdf5_file_mod swapped for textio_hdf5_file_mod.F90, so it reads
# the text export of input.h5 / reference.h5 and runs its own validation.
# usage: build_dwarf.sh <dwarf-p-cloudsc checkout> <build dir>
set -euo pipefail
DW=$1; B=$2; HERE=$(cd "$(dirname "$0")" && pwd)
M=$DW/src/common/module; F=$DW/src/cloudsc_fortran
mkdir -p "$B"; cd "$B"
FF="-O1 -ffp-contract=off -fno-fast-math -cpp -ffree-line-length-none -fopenmp -DHAVE_HDF5 -g -fbacktrace"
INC="-I$DW/src/common/include"
for f in $M/parkind1.F90 $M/abor1.F90 $M/oml_mod.F90 $M/cloudsc_mpi_mod.F90 $M/timer_mod.F90 \
         $M/ec_pmon_mod.F90 $HERE/textio_hdf5_file_mod.F90 $M/file_io_mod.F90 $M/yomcst.F90 \
         $M/yoethf.F90 $M/yoecldp.F90 $M/yoephli.F90 $M/yomphyder.F90 $M/expand_mod.F90 \
         $M/validate_mod.F90 $M/cloudsc_global_state_mod.F90 $F/cloudsc.F90 \
         $F/cloudsc_driver_mod.F90 $F/dwarf_cloudsc.F90; do
  gfortran $FF $INC -c "$f"
done
gcc -O1 -c $M/mycpu.c
gfortran $FF -o dwarf.x *.o
gfortran --version | head -1
echo BUILD_OK
