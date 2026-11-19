// HARNESS STUB (not AMReX): declarations ERF_MaterialProperties.H needs to
// compile; the MultiFab methods are never called by the oracle driver.
#pragma once
#include "AMReX_REAL.H"
namespace amrex {
  struct MultiFab {};
  struct PrintStub { template <class T> PrintStub& operator<<(const T&) { return *this; } };
  inline PrintStub Print() { return {}; }
  template <class E> inline const char* getEnumNameString(E) { return ""; }
}
