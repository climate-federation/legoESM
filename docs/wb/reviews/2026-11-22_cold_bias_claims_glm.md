**Verification caveat:** I read the cited code paths (config.py:1424, multilayer_land.py:913-930, canopy/radiative_transfer.py:630-650, diagnostics.py ~875, integration.py ~641/2294) and reference CAM/CLM sources; probe outputs I take on your word.

**C1 – CONFIRMED (as mechanism), PLAUSIBLE (as attribution).** Default False at config.py:1424 is verified; multilayer_land.py:913-930 does export snowy-ground albedo uniformly. But "main driver" is not yet proven: your TOA albedo excess is +0.10-0.15 over Siberia/Canada, consistent with forest unmasking (CLM5 Δ~0.2-0.4 over taiga). A 1-D column test (toggle masking, same forcing) would nail attribution. Action: agree, but gate on C2 — masking ON with fcansno=0 is *darker* than CLM5, so naive activation may overcorrect.

**C2 – CONFIRMED, and this is the trap.** Verified fcansno hardwired 0 (radiative_transfer.py:630-650); your probe numbers (0.14 vs CLM5 0.16-0.30) show masking-ON bracket *below* CLM5. CLM5 0.34 computes fcansno = fraction of canopy covered by snow = min(1, CanopySnw/((elai+esai)·mw), snow buries short canopy via Esnow_scaling); see CanopyHydrologyMod (CanopySnw = CanopyWater above fresno threshold) and SurfaceAlbedoMod. Our land does carry canopy-intercepted snow if two-leaf multilayer includes a canopy water store — verify; if it's snow-blind (liquid only), the port is bigger than proposed. Challenge the action: don't ship masking without fcansno, or evergreen winter albedo will be too *dark* → warm bias / early melt.

**C3 – PLAUSIBLE.** 5000 K activation is effectively "never ages while cold" — verified as config semantics; but 0.735 after melt onset vs 0.51 is a probe, not in-code. Note confound: tau and masking interact; recalibrate tau *after* C2 lands, or you'll calibrate to compensate for the albedo bug.

**C4 – CONFIRMED (code).** integration.py passes only q_c/q_i; CAM radiation.F90 ~888-986 uses cldfsnow/cldfprime — verified in reference. But impact is mostly over Arctic Ocean/polar night, not the Siberian/Canada *land* annual-mean signal; snow cloud LW warms surface year-round weakly at high sun. Medium priority, don't expect it to fix NH land tas.

**C5 – CONFIRMED as bias direction, but unverified magnitude.** Monin-Obukhov ocean drag at 60 m → 2 m in stable BL underwarms — yes, diagnostic issue. Challenge: if it underreads, true cold bias is *worse* than -1.12 K; fix this first so C1-C4 effects are measurable. Reorder the action.

**C6 – CONFIRMED, low priority, agree.**

**Ranking (annual land tas):** C1+C2 (0.5-1.0 K, NH land) > C3 (0.2-0.5 K) > C5 (not a fix, but inflates apparent bias) > C4 (mostly ocean) > C6.

**What you may have missed:** (a) the -0.24/-0.43 mm/d evap/precip deficit — a dry bias reduces boundary-layer moisture and cloud LW downwelling over land independently of C4; check land soil moisture initialization/spinup. (b) Antarctica +40% cloud — if that's in the multilayer land/turbulent closure (too-weak stability, excess condensation), it's a separate bug, not CMIP6-inherited. (c) No snow-density/thermal-conductivity aging → frozen-soil heat flux error in winter. Evap deficit is the biggest candidate for >1 K you haven't tied down.
