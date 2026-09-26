# PREREGISTRATION CORRECTION — end-wall one-cell participation

Status: frozen after the first round-2 execution refused the domain
participation *name*, before the corrected execution.

The first round-2 preregistration used the conventional inverse-participation
formula `(sum(delta**2)**2)/sum(delta**4)`. Cross-family review's quoted
"domain error is about 24 faces" instead uses peak-equivalent faces:

`N_peak = sum(delta**2) / max(delta**2)`.

Calling 54.66 the review's participation number would therefore be wrong.
That domain label is **RETRACTED**; no threshold or source array failed, and
the already registered `j=1` result remains numerically one under either
definition. The corrected probe prints both formulas under unambiguous names.
`N_peak` is primary for the review question; inverse participation is
descriptive only.

For direct `dynzdf` and `F_slow`, print `N_peak` on the 49 wet `j=1` faces and
the 9,758 wet domain faces. The expected signature supplied before either run
is `N_peak_j1=1.0000` and `N_peak_domain` approximately 24, with the same
dominant coordinate/support in both operands. **CONFIRMS the one-cell row** iff
`N_peak_j1<=1.01`, exactly one wet row face has `abs(lego/NEMO-1)>0.01`, and
the row-scaled maximum is `>=0.20`; **REFUTES** iff `N_peak_j1>=2.0`, at least
two faces exceed one percent, or the row-scaled maximum is `<=0.01`; otherwise
`UNRESOLVED`. A planted second peak-sized row residual must make `N_peak>=1.9`
and increment the outlier count.

For the coastal-unmask hypothesis, the earlier `128*eps*max(1,...)` support is
also retracted as a material-support definition: it admitted ordinary
non-bit-identical wet differences. The corrected, scale-aware full-plane set is
`abs(delta) >= 0.01*max(abs(delta))`, fixed here. Report its overlap with the
exact `sbcmod.F90:543-544` coastal-unmask set (`umask=0` and multiplier 2), but
never merge dry points into wet scores. **CONFIRMS** iff overlap is at least
90% and the wet `j=1` argmax is itself coastal-unmasked; **REFUTES** iff overlap
is at most 10% or that argmax is not coastal-unmasked; otherwise `UNRESOLVED`.
A planted coastal-flag flip at a synthetic confirming argmax must force REFUTE.

The corrected run reuses the committed bridge/loaders and same kt=5761 arrays.
It must print the original 2.0966766111 and Pearson-gate retractions, this
participation-definition retraction, Git SHA, this preregistration commit,
inputs/flags/hashes, and all controls. No GPU runs and no existing GPU artifact
is rescored here.
