# Preregistration: W-slope stage discriminator, round 91

Date: 2026-08-30. Frozen before measurement. The admitted round-90 receipt is
SHA-256 `32a79283ab3ed45fbf909ffc03ed8a8afc5ccb9b82e99a5f5de6f22f9ae1dc05`.

Use the two native-slope tuples production already computes: the Redi tuple
from step-entry Kmm geometry and the bolus tuple from the Naa geometry. Expose
the latter's wslpi/wslpj diagnostically and run a 2^2 offline substitution
against the held NEMO skew component. Keep Kmm uslp/vslp, coefficients,
gradients, metrics, masks, and source association fixed. No NEMO run is
permitted.

The Naa W pair owns only if its joint arm passes the unchanged `1e-15` bar
and removes at least 90% of baseline error. Record both main effects and the
interaction. Each raw W field is scored separately, and the current baseline
must recompose production. Identity, finite, four-arm, wet-point, roll, sign,
and non-inert controls must pass.

If owned, production forms a mixed Redi native tuple: Kmm uslp/vslp plus Naa
wslpi/wslpj. The through-FCT bolus remains wholly Naa, preserving row 8.8;
the horizontal Redi flux retains Kmm uslp/vslp, preserving 78.T.1/T.2. Only
the two DINO NEMO cards select the mixed carry; generic cards remain byte-pinned.
If the Naa pair is not exact, stop for the existing row-30 raw-slope operand
ladder rather than adding held instrumentation.
