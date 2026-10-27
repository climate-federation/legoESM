#!/bin/bash
# Build the CAM6 MG2 oracle drivers against UNMODIFIED CAM sources.
# usage: build.sh <CAM checkout (cam_cesm2_1_rel_60)> <build dir>
# CAM's own wv_sat_methods.F90 (Goff-Gratch, initialised by the drivers with
# legoESM's epsilon), micro_mg_utils.F90 and micro_mg2_0.F90 compile as
# shipped.  gfortran 8.5.0 was used for the
# committed fixture (tests/atmosphere/microphysics/oracle_fixtures/).
set -euo pipefail
CAM=$1/src/physics/cam
B=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
cd "$B"
FF="-O0 -g -ffree-line-length-none -DHAVE_GAMMA_INTRINSICS -fbacktrace"
gfortran $FF -c "$CAM/wv_sat_methods.F90" "$CAM/micro_mg_utils.F90" "$CAM/micro_mg2_0.F90"
gfortran $FF -o mg2_driver "$HERE/mg2_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_procs "$HERE/mg2_procs_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_rime "$HERE/mg2_rime_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
gfortran $FF -o mg2_ice "$HERE/mg2_ice_driver.F90" wv_sat_methods.o micro_mg_utils.o micro_mg2_0.o
# Probe build (S10.2/S10.3 stage oracle): a COPY of micro_mg2_0.F90 that only
# records values in public module arrays (nctend right after the activation
# re-apply, micro_mg2_0.F90:2022, and after sedimentation, before :2501; the
# droplet sedimentation sub-step count); no MG2 arithmetic changes.  Built in tap/ so the unmodified
# objects above stay untouched.
mkdir -p tap
# S10.4b adds tap_snow1 (state after the snow-melt pre-apply, M:1178), tap_snow2
# (pools, Tp, tends before the post-sedimentation snow melt, M:2535) and tap_snow3
# (tends after it, before M:2569); line-addressed where the text repeats.
# S10.4 adds tap_pre (pools, Tp and the tends just before the ice melt, M:2616)
# and tap_post (tends after homogeneous freezing, before M:2687).
# S10.3 adds nctend right before M:2501 (after all sedimentation) and the
# droplet sub-step count of the cloud sedimentation loop.
sed -e '0,/^contains$/s//real(r8), public :: nctend_stage(1,512) = 0._r8\nreal(r8), public :: nctend_sed(1,512) = 0._r8\ninteger, public :: nstepc_tap(1) = 0\nreal(r8), public :: tap_pre(1,512,10) = 0._r8\nreal(r8), public :: tap_post(1,512,5) = 0._r8\nreal(r8), public :: tap_snow1(1,512,5) = 0._r8\nreal(r8), public :: tap_snow2(1,512,8) = 0._r8\nreal(r8), public :: tap_snow3(1,512,5) = 0._r8\nreal(r8), public :: tap_ilim1(1,512,25) = 0._r8\nreal(r8), public :: tap_ilim2(1,512,5) = 0._r8\nreal(r8), public :: tap_ilim3(1,512,8) = 0._r8\nreal(r8), public :: tap_frz(1,512,5) = 0._r8\ncontains/' \
    -e 's/^  nctend = nctend + npccn$/&\n  nctend_stage = 0._r8\n  nctend_stage(1:mgncol,1:nlev) = nctend/' \
    -e 's/^  ! get new update for variables that includes sedimentation tendency$/  nctend_sed = 0._r8\n  nctend_sed(1:mgncol,1:nlev) = nctend\n&/' \
    -e 's/^        faloutc  = fc(i,:)  \* dumc(i,:)$/        nstepc_tap(i) = nstep\n&/' \
    -e '1178s/^  end do $/&\n  tap_snow1 = 0._r8\n  tap_snow1(1:mgncol,1:nlev,1) = qs\n  tap_snow1(1:mgncol,1:nlev,2) = ns\n  tap_snow1(1:mgncol,1:nlev,3) = qr\n  tap_snow1(1:mgncol,1:nlev,4) = nr\n  tap_snow1(1:mgncol,1:nlev,5) = tlat/' \
    -e 's/^  ! calculate instantaneous processes (melting, homogeneous freezing)$/  tap_snow2 = 0._r8\n  tap_snow2(1:mgncol,1:nlev,1) = dums\n  tap_snow2(1:mgncol,1:nlev,2) = dumns\n  tap_snow2(1:mgncol,1:nlev,3) = t + tlat\/cpp*deltat\n  tap_snow2(1:mgncol,1:nlev,4) = qstend\n  tap_snow2(1:mgncol,1:nlev,5) = nstend\n  tap_snow2(1:mgncol,1:nlev,6) = qrtend\n  tap_snow2(1:mgncol,1:nlev,7) = nrtend\n  tap_snow2(1:mgncol,1:nlev,8) = tlat\n&/' \
    -e '2569s/^   do k=1,nlev$/  tap_snow3 = 0._r8\n  tap_snow3(1:mgncol,1:nlev,1) = qstend\n  tap_snow3(1:mgncol,1:nlev,2) = nstend\n  tap_snow3(1:mgncol,1:nlev,3) = qrtend\n  tap_snow3(1:mgncol,1:nlev,4) = nrtend\n  tap_snow3(1:mgncol,1:nlev,5) = tlat\n&/' \
    -e 's/^   if (do_cldice) then$/  tap_pre = 0._r8\n  tap_pre(1:mgncol,1:nlev,1) = dumc\n  tap_pre(1:mgncol,1:nlev,2) = dumnc\n  tap_pre(1:mgncol,1:nlev,3) = dumi\n  tap_pre(1:mgncol,1:nlev,4) = dumni\n  tap_pre(1:mgncol,1:nlev,5) = t + tlat\/cpp*deltat\n  tap_pre(1:mgncol,1:nlev,6) = qctend\n  tap_pre(1:mgncol,1:nlev,7) = nctend\n  tap_pre(1:mgncol,1:nlev,8) = qitend\n  tap_pre(1:mgncol,1:nlev,9) = nitend\n  tap_pre(1:mgncol,1:nlev,10) = tlat\n&/' \
    -e 's/^     ! remove any excess over-saturation, which is possible due to non-linearity when adding$/  tap_post = 0._r8\n  tap_post(1:mgncol,1:nlev,1) = qctend\n  tap_post(1:mgncol,1:nlev,2) = nctend\n  tap_post(1:mgncol,1:nlev,3) = qitend\n  tap_post(1:mgncol,1:nlev,4) = nitend\n  tap_post(1:mgncol,1:nlev,5) = tlat\n&/' \
    -e 's/^        !PMC 12\/3\/12: ratio is also frac of step w\/ liquid.$/        tap_ilim1(i,k,:) = [vap_dep(i,k), ice_sublim(i,k), mnuccd(i,k), berg(i,k), msacwi(i,k), nsacwi(i,k), prci(i,k), nprci(i,k), bergs(i,k), psacws(i,k), prds(i,k), qcrat(i,k), q(i,k), t(i,k), qi(i,k), ni(i,k), qs(i,k), qc(i,k), qvi(i,k), precip_frac(i,k), icldm(i,k), lcldm(i,k), nnuccd(i,k), prai(i,k), nprai(i,k)]\n&/' \
    -e 's/^        mnuccri(i,k)=0._r8$/        tap_frz(i,k,:) = [mnuccr(i,k), nnuccr(i,k), lamr(i,k), qric(i,k), nric(i,k)]\n&/' \
    -e 's/^        if (pre(i,k) < 0._r8) then$/        tap_ilim2(i,k,:) = [pre(i,k), mnuccr(i,k), mnuccri(i,k), pracs(i,k), nnuccri(i,k)]\n&/' \
    -e 's/^     ! Big "administration" loop enforces conservation, updates variables$/     tap_ilim3(1:mgncol,k,1) = vap_dep(:,k)\n     tap_ilim3(1:mgncol,k,2) = ice_sublim(:,k)\n     tap_ilim3(1:mgncol,k,3) = mnuccd(:,k)\n     tap_ilim3(1:mgncol,k,4) = prci(:,k)\n     tap_ilim3(1:mgncol,k,5) = nprci(:,k)\n     tap_ilim3(1:mgncol,k,6) = prds(:,k)\n     tap_ilim3(1:mgncol,k,7) = pre(:,k)\n     tap_ilim3(1:mgncol,k,8) = nnuccd(:,k)\n&/' \
    "$CAM/micro_mg2_0.F90" > tap/micro_mg2_0.F90
test "$(grep -c nctend_stage tap/micro_mg2_0.F90)" -eq 3
test "$(grep -c nctend_sed tap/micro_mg2_0.F90)" -eq 3
test "$(grep -c nstepc_tap tap/micro_mg2_0.F90)" -eq 2
test "$(grep -c tap_pre tap/micro_mg2_0.F90)" -eq 12
test "$(grep -c tap_post tap/micro_mg2_0.F90)" -eq 7
test "$(grep -c tap_snow1 tap/micro_mg2_0.F90)" -eq 7
test "$(grep -c tap_snow2 tap/micro_mg2_0.F90)" -eq 10
test "$(grep -c tap_snow3 tap/micro_mg2_0.F90)" -eq 7
# S10.6 adds tap_ilim1 (limiter inputs after the qc limiter, before M:1588), tap_ilim2
# (pre, mnuccr, mnuccri after the rain limiter, before M:1687) and tap_ilim3 (limited
# vap_dep, ice_sublim, mnuccd, prci, nprci, prds, pre, nnuccd before M:1855).
test "$(grep -c tap_ilim1 tap/micro_mg2_0.F90)" -eq 2
test "$(grep -c tap_ilim2 tap/micro_mg2_0.F90)" -eq 2
test "$(grep -c tap_ilim3 tap/micro_mg2_0.F90)" -eq 9
# 2026-10-22 adds tap_frz (Bigg mnuccr, nnuccr and lamr, qric, nric before the
# frozen-rain ice/snow split, M:1645).
test "$(grep -c tap_frz tap/micro_mg2_0.F90)" -eq 2
test "$(diff "$CAM/micro_mg2_0.F90" tap/micro_mg2_0.F90 | grep -c '^>')" -eq 66
# S10.4 claim: between the homogeneous-freezing block and the final number
# clear, nctend is written only by the nccons branch (off in the P driver), the
# droplet re-adjust and the clear; qctend only under allow_sed_supersat (off).
test "$(sed -n '2686,2913p' "$CAM/micro_mg2_0.F90" | grep -c 'nctend(i,k) *=')" -eq 3
test "$(sed -n '2686,2913p' "$CAM/micro_mg2_0.F90" | grep -c 'qctend(i,k) *=')" -eq 1
(cd tap && gfortran $FF -I.. -c "$CAM/micro_mg_utils.F90" micro_mg2_0.F90 \
   && gfortran $FF -DMG2_TAP -I. -I.. -o ../mg2_tap "$HERE/mg2_driver.F90" ../wv_sat_methods.o \
      micro_mg_utils.o micro_mg2_0.o)
echo BUILD_OK
