// CPU shim for the oracle's #include <curand_kernel.h>: only the types/symbols the
// deterministic device functions under test reference. The RNG is NOT exercised:
// uniforms are supplied by the harness.
#pragma once
struct curandState { int dummy; };
struct curandStateScrambledSobol32_t { int dummy; };
typedef unsigned int curandDirectionVectors32_t[32];
inline void curand_init(unsigned long long, unsigned long long, unsigned long long, curandState*) {}
inline double curand_uniform_double(curandState*) { return 0.5; }
inline float curand_uniform(curandState*) { return 0.5f; }
