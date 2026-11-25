# SI3 Rung O1 Preregistration Amendment 1

This amendment preserves the original preregistration unchanged and records a scope correction discovered by source inspection before acquisition.

NEMO 5.0.2 calls `adv_pra_rst('WRITE', kt)` only inside `ice_dyn_adv_pra` (`src/ICE/icedyn_adv_pra.F90:487`), reached only when `ln_icedyn = .true.` (`src/ICE/icestp.F90:167-168`). O1 fixes `ln_icedyn = .false.`, so its live NetCDF restart cannot contain Prather moments. O1 still records and scores the allocated in-memory moments in the binary observer. Live NEMO NetCDF Prather restart certification moves to O4, the first rung that enables Prather dynamics. This is a scope correction, not a changed fidelity result.
