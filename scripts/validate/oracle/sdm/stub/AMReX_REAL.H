// HARNESS STUB (not AMReX): minimal definitions so ERF 26.09 header-only SDM
// physics compiles standalone in double precision on the host.
#pragma once
#include <cmath>
#include <cstdio>
#include <cassert>
#include <array>
namespace amrex { using Real = double; using ParticleReal = double;
  template <class T, int N> struct GpuArray { T arr[N]; T& operator[](int i){return arr[i];} const T& operator[](int i) const {return arr[i];} };
  inline void Abort(const char* m){ std::fprintf(stderr,"ABORT %s\n",m); std::abort(); }
  template<class... A> inline void ignore_unused(const A&...) {}
}
#define AMREX_GPU_HOST_DEVICE
#define AMREX_GPU_DEVICE
#define AMREX_GPU_HOST
#define AMREX_FORCE_INLINE inline
#define AMREX_INLINE inline
#define AMREX_DEVICE_PRINTF(...) std::printf(__VA_ARGS__)
#define AMREX_ALWAYS_ASSERT(x) assert(x)
#define AMREX_ASSERT(x) assert(x)
#define AMREX_ENUM(NAME, ...) enum class NAME { __VA_ARGS__ }
#ifndef AMREX_SPACEDIM
#define AMREX_SPACEDIM 3
#endif
