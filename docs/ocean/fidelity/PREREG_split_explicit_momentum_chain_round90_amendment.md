# Round-90 control amendment: admit certified inert coefficients

Date: 2026-08-30. Frozen after the first round-90 score returned `INVALID` and
before rescoring. The ownership bars, arms, population, and measurements are
unchanged.

The original `each_substitution_noninert` control was structurally wrong:
round 90 explicitly includes already-certified ahtu/ahtv, so exact raw
coefficient operands must make their substitutions inert. Replace it with two
controls: ahtu/ahtv raw operands pass and their single substitutions are
bit-inert; wslpi/wslpj substitutions are non-inert. This amendment changes
only validity classification and is bound alongside the original preregistration.
