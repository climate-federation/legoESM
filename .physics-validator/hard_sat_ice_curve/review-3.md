Finding remains: the cap does **not** exactly match Morrison’s conversion at low density. Morrison uses `N_i_nuc_max / clip(rho, 0.1)`, while the helper uses `p/(R_d T)` floored only at `1e-6`. Above roughly 300 hPa this can seed up to `0.1/rho` more number than Morrison’s Cooper target (≈3.9× at a true 15 hPa / 205 K; far larger near the 200 Pa top).

The new test also labels `1.5e4 Pa` as 15 hPa; it is 150 hPa, so it misses this regime. Use Morrison’s `0.1 kg m⁻³` density floor (ideally its same density diagnostic) and test genuine 1500 Pa.

`OVERALL VERDICT: findings remain: F3 cap is not fully Morrison-consistent at low density due to mismatched rho floor.`
