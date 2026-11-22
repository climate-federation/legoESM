# T24a pressure-coordinate tracer repair preregistration

Scope authorized by claude_phase2a.md: all phase1 leaking atmosphere sigma and
spectral PE hybrid routes; MPAS SB selector; spectral T31 production SB.
Height-coordinate time coupling and all ocean implementation remain excluded.
Baseline ed2c8ead3a (implementation parent 6a8314d738); phase1 census 94e936281c.

Before any measurement, predict: every repaired native vertical route, both SB
and van Leer, has maximum per-column paired instantaneous residual <=1e-12
using the unchanged phase1 gross-tendency denominator and closed lids.
CONFIRM requires all fixed routes and both schemes; any violation REFUTES.
Height and ocean census results remain report-only with their prior limitations.
Known conserving and planted/leaking controls must retain their classifications.
Sigma equivalence target: A=0 hybrid on the same sigma geometry and ps, with
smoothing widths in kg/kg and Pa/s, roundoff <=1e-12 relative gross tendency.
Preserve MPAS hybrid van Leer and latlon hybrid SB fixture step outputs BITWISE
against ed2c8ead3a. New default paths need finite, nonzero fp32 tracer gradients.
Conservation/selector regression tests must fail under isolated baseline revert.
All execution PBS CPU develop@desched1 UCLB0017; local commits only, no push.

Detail ballot phase2a_vote_prompt.md items1-6: Codex AGREE, GLM AGREE (glm-5.3),
2/2 each. No disagreements. Approved reuse/extraction, inherited widths,
roundoff gate, fixtures, guards, selector placement and SB standalone default.
Full-diff independent Codex and GLM review required before implementation commit.
User-selected production default: spectral SB; FV3 conserves so retains null.
No new time-coupling algorithm or prognostic field selected.
