**Verdict: OVERSTATED.** The measurements strongly support a *PCG-iteration-correlated, latency-dominated distributed overhead*, but do not yet uniquely establish “~95 µs/iter of exposed dependent synchronisation.”

(a) `111/4` is an ideal-scaling assumption, not a measured compute share. At these tile sizes, launch floors, reduced occupancy, cache effects, halo work, and different fusion/kernel choices can make per-device compute sub- or super-linear. The inferred 95 µs is therefore an estimate, not a separation.

(b) The slope difference is valid evidence of *additional nd4 marginal cost*, but not a clean sync/compute decomposition: each PCG iteration changes local compute, halo/reduction communication, launch count, and possible compilation choices together. It identifies the difference only if local per-iteration compute is independently shown to scale as 1/4.

(c) A truly fixed per-step cost cannot create the observed 60→N slope: it appears in the fitted intercept. But it can still account for some of the 4.28 ms roofline residual at N=60. Report the regression intercept and its confidence interval, ideally including N=0 or a non-PCG equivalent, before attributing the whole residual to 60 iterations.

(d) 5.70 ms predicted versus 6.47 ms observed differs by 0.77 ms (13.5%). That is plausible agreement only if the combined uncertainty/model error is at least about that size; no error bars are given. With latency calibration, run-to-run variance, and the unvalidated 1/4 scaling assumption, it is supportive rather than confirmatory.

The byte census and scheduler nulls make the communication plausibly latency-bound and poorly overlappable. Strengthen the claim by measuring nd4 local PCG kernel time (or a communication-disabled surrogate) across iteration counts, fitting `T(N)=intercept+N·slope` with CIs, and showing the local-compute slope separately.