Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fc673-e158-7bf1-9d41-8d8f3e8718d4
--------
user
Round-21 confirm: r20 items — (1) numeric python scalars now included in assert_pytree_bytes_equal's leaf filter; (2) _agree_ocean_spmd_call gained has_aux+aux_schema payload entries (fixed tuple extended) and sharded_step passes aux=aux (the _global call site uses the default aux=None — payload width still fixed); (4) band_fingerprint 0-D ValueError + five new names in __all__. Item 3 (atm lane broadcast) explicitly TRACKED in the campaign doc as a follow-up with file:line and an unverified-at->=96 warning. 167 tests + 2-proc repro green. Read the diff (git diff HEAD~1). End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll audit the last commit’s diff against each stated Round-21 item, then run focused checks only if the change leaves a material uncertainty.
exec
/usr/bin/bash -lc 'git diff --check HEAD~1 HEAD && git diff --find-renames --stat HEAD~1 HEAD && git diff --find-renames HEAD~1 HEAD' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 2 in 366ms:
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:27: trailing whitespace.
+    
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1137: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1139: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1166: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1170: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1171: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1276: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1277: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1315: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1321: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1380: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1381: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1401: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1410: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1428: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1429: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1447: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1449: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1460: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1467: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1469: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1484: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1488: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1489: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1635: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1650: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1773: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1777: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1808: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1819: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1843: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1850: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1854: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1855: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3191: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3200: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3202: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3222: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3232: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3240: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3250: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3255: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3261: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3274: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3284: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3290: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3291: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3304: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3305: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3326: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3327: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3331: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3337: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3345: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3368: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3369: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3473: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3483: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3500: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3501: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3539: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3545: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3604: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3605: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3608: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3615: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3634: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3643: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3650: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3669: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3696: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3697: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3701: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3716: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3723: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3725: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3726: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3731: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3740: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3746: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3758: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3760: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3773: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3775: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3776: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3780: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3793: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3794: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3799: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3803: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3820: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3823: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3826: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3847: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3848: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3969: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3976: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3986: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4001: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4013: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4016: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4032: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4039: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4052: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4175: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4179: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4184: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4195: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4207: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4214: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4239: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4244: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4248: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4259: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4262: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4287: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4332: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4386: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4391: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4392: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4397: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4406: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4416: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4418: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4437: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4445: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4454: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4461: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4464: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4465: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4474: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4475: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4480: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4481: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4487: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4488: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4493: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4494: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4502: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4503: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4509: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4510: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4519: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4520: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5566: trailing whitespace.
+   640	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5567: trailing whitespace.
+   641	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5572: trailing whitespace.
+   646	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5579: trailing whitespace.
+   653	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5589: trailing whitespace.
+   663	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5601: trailing whitespace.
+   675	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5613: trailing whitespace.
+   687	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5616: trailing whitespace.
+   690	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5632: trailing whitespace.
+   706	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5638: trailing whitespace.
+   712	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5649: trailing whitespace.
+   723	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5662: trailing whitespace.
+   736	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5680: trailing whitespace.
+   871	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5714: trailing whitespace.
+   905	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5768: trailing whitespace.
+   959	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5773: trailing whitespace.
+   964	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5774: trailing whitespace.
+   965	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5779: trailing whitespace.
+   970	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5780: trailing whitespace.
+   610	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5781: trailing whitespace.
+   611	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5786: trailing whitespace.
+   616	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5793: trailing whitespace.
+   623	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5803: trailing whitespace.
+   633	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5815: trailing whitespace.
+   645	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5824: trailing whitespace.
+   654	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5827: trailing whitespace.
+   657	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5843: trailing whitespace.
+   673	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5849: trailing whitespace.
+   679	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5860: trailing whitespace.
+   690	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5888: trailing whitespace.
+   718	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5942: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6101: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6111: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6119: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6130: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6135: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6141: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6153: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6156: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6163: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6182: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6191: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6198: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6217: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6241: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6242: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6246: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6261: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6268: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6270: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6271: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6276: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6285: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6291: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6303: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6318: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6320: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6321: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6325: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6338: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6339: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6350: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6356: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6367: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6393: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6412: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6416: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6442: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6477: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6512: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6517: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6518: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6523: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6532: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6542: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6544: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:6563: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7032: trailing whitespace.
+   214	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7042: trailing whitespace.
+   224	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7059: trailing whitespace.
+   241	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7060: trailing whitespace.
+   242	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7067: trailing whitespace.
+   249	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7068: trailing whitespace.
+   250	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7083: trailing whitespace.
+   265	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7084: trailing whitespace.
+   266	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7087: trailing whitespace.
+   269	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7107: trailing whitespace.
+   289	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7108: trailing whitespace.
+   290	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7111: trailing whitespace.
+   293	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7118: trailing whitespace.
+   300	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7137: trailing whitespace.
+   319	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7146: trailing whitespace.
+   328	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7152: trailing whitespace.
+   334	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7170: trailing whitespace.
+   352	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7193: trailing whitespace.
+   375	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7194: trailing whitespace.
+   376	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7198: trailing whitespace.
+   380	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7213: trailing whitespace.
+   395	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7219: trailing whitespace.
+   401	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7221: trailing whitespace.
+   403	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7222: trailing whitespace.
+   404	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7227: trailing whitespace.
+   409	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7236: trailing whitespace.
+   418	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7242: trailing whitespace.
+   424	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7254: trailing whitespace.
+   436	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7256: trailing whitespace.
+   438	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7268: trailing whitespace.
+   450	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7272: trailing whitespace.
+   533	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7273: trailing whitespace.
+   534	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7276: trailing whitespace.
+   537	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7284: trailing whitespace.
+   545	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7287: trailing whitespace.
+   548	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7295: trailing whitespace.
+   556	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7307: trailing whitespace.
+   568	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7313: trailing whitespace.
+   574	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7343: trailing whitespace.
+   604	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7344: trailing whitespace.
+   605	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7351: trailing whitespace.
+   612	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7352: trailing whitespace.
+   613	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7355: trailing whitespace.
+   616	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7363: trailing whitespace.
+   624	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9199: trailing whitespace.
+  1028	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9207: trailing whitespace.
+  1036	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9212: trailing whitespace.
+  1041	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9218: trailing whitespace.
+  1047	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9224: trailing whitespace.
+  1053	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9228: trailing whitespace.
+  1057	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9232: trailing whitespace.
+  1061	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9253: trailing whitespace.
+  1082	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9257: trailing whitespace.
+  1086	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9281: trailing whitespace.
+  1799	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9285: trailing whitespace.
+  1803	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9289: trailing whitespace.
+  1807	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9292: trailing whitespace.
+  1810	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9301: trailing whitespace.
+  1819	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9308: trailing whitespace.
+  1826	

exec
/usr/bin/bash -lc 'git status --short && git diff --name-status HEAD~1 HEAD && git diff --numstat HEAD~1 HEAD' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 547ms:
?? .physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md
A	.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md
M	docs/performance/scaling/levante_campaign_2026-07-24.md
M	packages/core/legoesm/parallel/geometry_consistency.py
M	packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
10735	0	.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md
20	0	docs/performance/scaling/levante_campaign_2026-07-24.md
12	3	packages/core/legoesm/parallel/geometry_consistency.py
8	2	packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py

exec
/usr/bin/bash -lc 'git diff --find-renames HEAD~1 HEAD -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py docs/performance/scaling/levante_campaign_2026-07-24.md && rg -n -C 4 "assert_pytree_bytes_equal|band_fingerprint|_agree_ocean_spmd_call|sharded_step|global_step|__all__" packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 291ms:
diff --git a/docs/performance/scaling/levante_campaign_2026-07-24.md b/docs/performance/scaling/levante_campaign_2026-07-24.md
index 8dbe52d0c..e78fdec15 100644
--- a/docs/performance/scaling/levante_campaign_2026-07-24.md
+++ b/docs/performance/scaling/levante_campaign_2026-07-24.md
@@ -2086,3 +2086,23 @@ fixed lane (l50081+l50100 excluded; l50081 exclusion harmless-superset):
   **eff 0.84** — a healthy-tile strong leg on the ocean lane.
 * This closes the user directive's ocean-hundreds gap: both lat-lon
   lanes (atm + ocean) now hold receipts at 96-128 GPUs, MPAS at 128.
+
+### Merge-port of #1362 (geometry_consistency) — shared-module fixes + one tracked follow-up
+
+The three multicontroller fixes now live in
+`legoesm.parallel.geometry_consistency` (checked_shard_put /
+addressable_shard_put / assert_pytree_bytes_equal / band_fingerprint —
+the ONE implementation, #1362 doctrine); the ocean lane calls them, and
+#1362's entry gates gained aux coverage + numeric-scalar leaves in the
+digest gate (codex r20). 167 gate/parity tests + the 2-proc repro
+(15.11 ms) green post-fix.
+
+**TRACKED FOLLOW-UP (codex r20 item 3): the ATMOSPHERE lat-lon lane
+still routes its band/tile geometry stacks through `broadcast_checked`
+-> `broadcast_one_to_all` (sharded_atm_latlon_step.py:492/1586) — wall
+1 preserved there** (an [n_processes, stack] psum program). The atm
+128-GPU receipts predate #1362, so the current main atm lane at >=96
+processes is UNVERIFIED and plausibly walled exactly as ocean was.
+Port = same checked_shard_put swap + aux threading; needs its own
+parity run + a 2-proc repro before any atm hundreds rerun on merged
+main.
diff --git a/packages/core/legoesm/parallel/geometry_consistency.py b/packages/core/legoesm/parallel/geometry_consistency.py
index 0a131ca18..9633437b7 100644
--- a/packages/core/legoesm/parallel/geometry_consistency.py
+++ b/packages/core/legoesm/parallel/geometry_consistency.py
@@ -44,6 +44,11 @@ import jax
 import numpy as np
 
 __all__ = [
+    "addressable_shard_put",
+    "assert_pytree_bytes_equal",
+    "band_fingerprint",
+    "band_fingerprints_agree",
+    "checked_shard_put",
     "content_hash48",
     "name_digest48",
     "schema_fingerprint",
@@ -630,9 +635,10 @@ def band_fingerprint(host, n_bands):
     fixed per-band count, pass the float gate.
     """
     host = np.asarray(host)
-    if host.shape[0] != n_bands:
+    if host.ndim == 0 or host.shape[0] != n_bands:
         raise ValueError(
-            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
+            f"band_fingerprint: leading axis "
+            f"{host.shape[0] if host.ndim else '<0-d>'} != n_bands "
             f"{n_bands}")
     is_exact = host.dtype.kind in "biu"
     struct = [float(host.ndim), *map(float, host.shape),
@@ -707,8 +713,11 @@ def assert_pytree_bytes_equal(tree, what):
         return
     from jax.experimental import multihost_utils
 
+    # Numeric python scalars included (codex r20 item 1): the scatter
+    # paths jnp.asarray + put them, so a rank-divergent scalar must not
+    # bypass the gate. Non-numeric leaves (None, strings) stay excluded.
     leaves = [x for x in jax.tree_util.tree_leaves(tree)
-              if hasattr(x, "ndim")]
+              if hasattr(x, "ndim") or isinstance(x, (int, float, complex))]
     vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
                     dtype=np.float64)
     g = multihost_utils.process_allgather(vals)
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index f9e7ecb7e..03e4f77a9 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -608,10 +608,12 @@ def _agree_ocean_spmd_entry(model, mesh, *, where: str) -> None:
 _OCEAN_CALL_ENTRY_FLAGS = (
     "has_mesh", "n_dev", "axis_names", "axis_sizes",
     "state_schema", "has_forcing", "forcing_schema",
+    "has_aux", "aux_schema",
 )
 
 
-def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
+def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str,
+                           aux=None) -> None:
     """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
 
     #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
@@ -636,6 +638,10 @@ def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
         float(forcing is not None),
         (tree_schema_digest48(forcing) if forcing is not None
          else FLAG_ABSENT),
+        # aux (codex r20 item 2): a rank-local None-vs-provided or schema
+        # mismatch must fail HERE, not desynchronize the jit call below.
+        float(aux is not None),
+        (tree_schema_digest48(aux) if aux is not None else FLAG_ABSENT),
     ), context=where)
 
 
@@ -883,7 +889,7 @@ def make_sharded_ocean_step(model, mesh):
         # structure key below distinguishes every None<->array combination.
         _agree_ocean_spmd_call(
             mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
-            where="make_sharded_ocean_step.step")
+            where="make_sharded_ocean_step.step", aux=aux)
         forcing = (freshwater, surface_forcing, sponge, t_seconds)
         _validate_forcing_layout((freshwater, surface_forcing, sponge))
         # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-6-operators are element-wise / local-stencil and the in-step halo exchange routes
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-7-through the SPMD band body once ``activate_latlon_spmd_halo(mesh)`` is armed; the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-8-barotropic PCG reductions already dispatch to ``batch_psum_spmd`` on ``"lat"``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-9-(``barotropic_common._global_dot_batch``). This is the ocean analogue of the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:10:cubed-sphere ``parallel.sharded_dynamics.make_sharded_step``.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-11-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-12-Architecture (the two non-trivial pieces — see ``omip-multinode-spmd-scope``):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-13-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-14-* GRID = **replicated-stacked, indexed** (Structure A). The ``N`` band
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-46-import jax.numpy as jnp
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-47-import numpy as np
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-48-from legoesm.parallel.geometry_consistency import (
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-49-    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-51-    coerce_bool, coerce_count, config_digest48, name_digest48,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-52-    tree_schema_digest48)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-53-from jax.sharding import NamedSharding, PartitionSpec as P
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-54-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-311-    # v-face row (regular pole wall OR tripole seam/cap row) must be
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-312-    # wall-masked — the carrier drops it and reconstructs it as zero, which
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-313-    # would silently delete a LIVE seam row.  Host-side check on the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-314-    # concrete state (this fn runs outside jit).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:315:    assert_pytree_bytes_equal(state, "shard_state_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-316-    vm = getattr(state, "v_mask", None)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-317-    if vm is not None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-318-        import numpy as _np
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-319-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-390-    # blocked in one (#1362 round 4, blocker 6).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-391-    _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-392-    if forcing is None or mesh is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-393-        return forcing
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-395-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-396-    def _put(leaf):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-397-        if leaf is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-398-            return None
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-433-                            where="shard_forcing_stack_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-434-    if mesh is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-435-        return stack
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-436-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-438-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-439-    def _put(leaf):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-440-        if leaf is None or not hasattr(leaf, "ndim"):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-441-            return leaf
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-611-    "has_aux", "aux_schema",
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-612-)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-613-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-614-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:615:def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-616-                           aux=None) -> None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-617-    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-618-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:619:    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-620-    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-621-    forcing layout that is invalid on one rank only makes that rank raise while
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-622-    its peers enter the collective program -- a hang.  The state + forcing leaf
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-623-    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-805-        # Fused v-carrier reconstruction (scaling-M4): under the SAME
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-806-        # trace-time switch as the pad aggregation, pack the boundary-row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-807-        # ppermutes of ALL staggered carriers (v + v_mask) into one
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-808-        # collective per dtype group — value-identical (a bit-copy
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:809:        # exchange; the flag flip re-keys the sharded_step cache below so
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-810-        # a reused step object rebuilds).  Default OFF = the historical
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-811-        # per-field ppermutes, byte-identical.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-812-        import os as _os
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-813-        _fused_v = _os.environ.get(
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-874-                    f"has leading dim {leaf.shape[0]} != n_lat "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-875-                    f"({n_lat_global}); forcing must be cell-centered "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-876-                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-877-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:878:    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-879-                     sponge=None, t_seconds=None, aux=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-880-        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-881-        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-882-        # inlines the inner call), concrete closure arrays become
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-886-        # jit boundary as an ARGUMENT and pass it back here.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-887-        # ONE forcing operand: None fields drop out of the pytree structure,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-888-        # so specs derived by tree.map skip them automatically and the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-889-        # structure key below distinguishes every None<->array combination.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:890:        _agree_ocean_spmd_call(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-891-            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-892-            where="make_sharded_ocean_step.step", aux=aux)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-893-        forcing = (freshwater, surface_forcing, sponge, t_seconds)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-894-        _validate_forcing_layout((freshwater, surface_forcing, sponge))
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-964-            set_halo_backend(_prev_backend, _prev_topo)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-965-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-966-    # Expose the stacks so outer-jit callers can pass them as arguments
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-967-    # (see the ``aux`` note in the signature).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:968:    sharded_step.aux = (geom_stacks, vmask_stack)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:969:    return sharded_step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-970-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-971-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-972-def make_sharded_ocean_step_global(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-973-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-994-                       surface_forcing=surface_forcing))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-995-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-996-    inner = make_sharded_ocean_step(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-997-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:998:    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:999:        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1000-                               where="make_sharded_ocean_step_global.step")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1001-        # Scatter the global state AND forcing to the band layout
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1002-        # explicitly (the old comment claimed inner sharded the forcing;
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1003-        # it forwarded it global and relied on implicit JIT input
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1008-                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1009-                   freshwater=shard_forcing_latlon(freshwater, mesh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1010-        return gather_state_latlon(ss, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1011-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1012:    return sharded_step_global
--
packages/core/legoesm/parallel/geometry_consistency.py-42-
packages/core/legoesm/parallel/geometry_consistency.py-43-import jax
packages/core/legoesm/parallel/geometry_consistency.py-44-import numpy as np
packages/core/legoesm/parallel/geometry_consistency.py-45-
packages/core/legoesm/parallel/geometry_consistency.py:46:__all__ = [
packages/core/legoesm/parallel/geometry_consistency.py-47-    "addressable_shard_put",
packages/core/legoesm/parallel/geometry_consistency.py:48:    "assert_pytree_bytes_equal",
packages/core/legoesm/parallel/geometry_consistency.py:49:    "band_fingerprint",
packages/core/legoesm/parallel/geometry_consistency.py:50:    "band_fingerprints_agree",
packages/core/legoesm/parallel/geometry_consistency.py-51-    "checked_shard_put",
packages/core/legoesm/parallel/geometry_consistency.py-52-    "content_hash48",
packages/core/legoesm/parallel/geometry_consistency.py-53-    "name_digest48",
packages/core/legoesm/parallel/geometry_consistency.py-54-    "schema_fingerprint",
--
packages/core/legoesm/parallel/geometry_consistency.py-615-# jit) becomes an MLIR constant whose value cannot be fetched for
packages/core/legoesm/parallel/geometry_consistency.py-616-# non-addressable arrays. The helpers below remove (1) and (2) — (3) is the
packages/core/legoesm/parallel/geometry_consistency.py-617-# callers' aux-threading contract, see make_sharded_ocean_step.
packages/core/legoesm/parallel/geometry_consistency.py-618-
packages/core/legoesm/parallel/geometry_consistency.py:619:def band_fingerprint(host, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py-620-    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).
packages/core/legoesm/parallel/geometry_consistency.py-621-
packages/core/legoesm/parallel/geometry_consistency.py-622-    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
packages/core/legoesm/parallel/geometry_consistency.py-623-    already be schema-gated across processes (:func:`assert_schema_agrees`)
--
packages/core/legoesm/parallel/geometry_consistency.py-636-    """
packages/core/legoesm/parallel/geometry_consistency.py-637-    host = np.asarray(host)
packages/core/legoesm/parallel/geometry_consistency.py-638-    if host.ndim == 0 or host.shape[0] != n_bands:
packages/core/legoesm/parallel/geometry_consistency.py-639-        raise ValueError(
packages/core/legoesm/parallel/geometry_consistency.py:640:            f"band_fingerprint: leading axis "
packages/core/legoesm/parallel/geometry_consistency.py-641-            f"{host.shape[0] if host.ndim else '<0-d>'} != n_bands "
packages/core/legoesm/parallel/geometry_consistency.py-642-            f"{n_bands}")
packages/core/legoesm/parallel/geometry_consistency.py-643-    is_exact = host.dtype.kind in "biu"
packages/core/legoesm/parallel/geometry_consistency.py-644-    struct = [float(host.ndim), *map(float, host.shape),
--
packages/core/legoesm/parallel/geometry_consistency.py-661-        vals = np.array(per_band, dtype=np.float64)
packages/core/legoesm/parallel/geometry_consistency.py-662-    return np.array(struct, dtype=np.float64), vals, is_exact
packages/core/legoesm/parallel/geometry_consistency.py-663-
packages/core/legoesm/parallel/geometry_consistency.py-664-
packages/core/legoesm/parallel/geometry_consistency.py:665:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
packages/core/legoesm/parallel/geometry_consistency.py:666:    """True iff every process's :func:`band_fingerprint` matches process 0's."""
packages/core/legoesm/parallel/geometry_consistency.py-667-    if rtol is None:
packages/core/legoesm/parallel/geometry_consistency.py-668-        rtol = _FLOAT_RTOL
packages/core/legoesm/parallel/geometry_consistency.py-669-    struct_ok = bool(np.all(g_struct == g_struct[0]))
packages/core/legoesm/parallel/geometry_consistency.py-670-    if is_exact:
--
packages/core/legoesm/parallel/geometry_consistency.py-689-        return jax.device_put(arr, sharding)
packages/core/legoesm/parallel/geometry_consistency.py-690-    from jax.experimental import multihost_utils
packages/core/legoesm/parallel/geometry_consistency.py-691-
packages/core/legoesm/parallel/geometry_consistency.py-692-    host = np.asarray(arr)
packages/core/legoesm/parallel/geometry_consistency.py:693:    struct, vals, is_exact = band_fingerprint(host, n_bands)
packages/core/legoesm/parallel/geometry_consistency.py-694-    g_struct = multihost_utils.process_allgather(struct)
packages/core/legoesm/parallel/geometry_consistency.py-695-    g_vals = multihost_utils.process_allgather(vals)
packages/core/legoesm/parallel/geometry_consistency.py:696:    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/core/legoesm/parallel/geometry_consistency.py-697-        raise RuntimeError(
packages/core/legoesm/parallel/geometry_consistency.py-698-            f"{context}: band-stacked field {name!r} DIVERGES across "
packages/core/legoesm/parallel/geometry_consistency.py-699-            f"processes (exact_dtype={is_exact}, "
packages/core/legoesm/parallel/geometry_consistency.py-700-            f"gathered={g_vals.tolist()}) — a real config/grid "
--
packages/core/legoesm/parallel/geometry_consistency.py-702-    return jax.make_array_from_callback(
packages/core/legoesm/parallel/geometry_consistency.py-703-        host.shape, sharding, lambda idx: host[idx])
packages/core/legoesm/parallel/geometry_consistency.py-704-
packages/core/legoesm/parallel/geometry_consistency.py-705-
packages/core/legoesm/parallel/geometry_consistency.py:706:def assert_pytree_bytes_equal(tree, what):
packages/core/legoesm/parallel/geometry_consistency.py-707-    """Cheap multi-process replacement for the per-leaf assert_equal that
packages/core/legoesm/parallel/geometry_consistency.py-708-    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
packages/core/legoesm/parallel/geometry_consistency.py-709-    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
packages/core/legoesm/parallel/geometry_consistency.py-710-    symmetric raise on mismatch. No-op single-process.
--
packages/core/legoesm/parallel/geometry_consistency.py-733-
packages/core/legoesm/parallel/geometry_consistency.py-734-def addressable_shard_put(arr, sharding):
packages/core/legoesm/parallel/geometry_consistency.py-735-    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
packages/core/legoesm/parallel/geometry_consistency.py-736-    consistency the CALLER has already gated (state/forcing pytrees via
packages/core/legoesm/parallel/geometry_consistency.py:737:    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
packages/core/legoesm/parallel/geometry_consistency.py-738-    if jax.process_count() <= 1:
packages/core/legoesm/parallel/geometry_consistency.py-739-        return jax.device_put(arr, sharding)
packages/core/legoesm/parallel/geometry_consistency.py-740-    host = np.asarray(arr)
packages/core/legoesm/parallel/geometry_consistency.py-741-    return jax.make_array_from_callback(

exec
/usr/bin/bash -lc "sed -n '560,750p' packages/core/legoesm/parallel/geometry_consistency.py && sed -n '570,660p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && sed -n '850,1020p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && rg -n \"assert_pytree_bytes_equal|band_fingerprint|_agree_ocean_spmd_call|has_aux|aux_schema\" --glob '"'!*.md'"' ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1626ms:

    host = np.asarray(arr)
    flat = host.ravel()
    is_exact = host.dtype.kind in "biu"
    # FIXED-WIDTH payloads (see _STRUCT_WIDTH/_VALS_WIDTH): the gathered shape
    # must never depend on rank-local data, or two processes can enter this
    # collective with different shapes and hang. Shape is folded in as a
    # digest rather than splatted, so an ndim difference cannot change the
    # length either.
    struct = np.zeros(_STRUCT_WIDTH, dtype=np.float64)
    struct[0] = float(host.ndim)
    struct[1] = float(np.dtype(host.dtype).num)
    struct[2] = float(1.0 if is_exact else 0.0)
    struct[3] = float(host.size)
    struct[4] = name_digest48([str(d) for d in host.shape])
    vals = np.zeros(_VALS_WIDTH, dtype=np.float64)
    if is_exact:
        vals[0] = content_hash48(host)
    else:
        finite = flat[np.isfinite(flat)]
        f64 = finite.astype(np.float64)
        # Non-finite COUNT is structural: a NaN appearing on one process only
        # must not be averaged away by the moment compare below.
        struct[5] = float(flat.size - finite.size)
        vals[0] = float(f64.sum()) if f64.size else 0.0
        vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
        vals[2] = float(np.abs(f64).max()) if f64.size else 0.0

    g_struct = multihost_utils.process_allgather(struct)
    g_vals = multihost_utils.process_allgather(vals)
    struct_ok = bool(np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(np.allclose(g_vals, g_vals[0],
                                   rtol=_FLOAT_RTOL, atol=0.0))
    if not (struct_ok and vals_ok):
        raise RuntimeError(
            f"{context}: geometry field {name!r} DIVERGES across processes "
            f"(struct_ok={struct_ok}, vals_ok={vals_ok}, "
            f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
            f"config/grid inconsistency, not autotune noise; refusing to "
            f"broadcast process 0 over it.")
    return np.asarray(multihost_utils.broadcast_one_to_all(host))


# --- assert-free sharded puts + per-band gates (2026-08-03, ocean walls) ----
# Three stacked multicontroller walls were found on the ocean lane (codex
# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
# [n_processes, stack] psum program (nd x 849 MB at LL2304 L20 — 81.5 GB at
# 96 procs); (2) jax.device_put of a NUMPY array onto an all-process
# sharding internally runs multihost_utils.assert_equal on the FULL array
# ([n_proc, field] landing on ONE device: fits under an 80 GB A100 up to
# ~64 procs, dies at 96 — jax _src/dispatch.py::_device_put_sharding_impl);
# (3) a concrete sharded-global array captured by an OUTER trace (jit-of-
# jit) becomes an MLIR constant whose value cannot be fetched for
# non-addressable arrays. The helpers below remove (1) and (2) — (3) is the
# callers' aux-threading contract, see make_sharded_ocean_step.

def band_fingerprint(host, n_bands):
    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).

    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
    already be schema-gated across processes (:func:`assert_schema_agrees`)
    — the payload widths depend on it, and mismatched widths would hang the
    allgather rather than raise.

    Exact dtypes (int/bool/uint): one positional 48-bit byte digest per
    band. Floats: per-band ``[sum, sum_of_squares, absmax]`` of finite
    entries plus per-band non-finite counts folded into ``struct``.
    Per-band (not whole-array) because each process's OWN bytes become the
    live inputs for the bands it owns under the assert-free put: a
    band-local drift must not hide in a whole-array sum (codex r14).
    DOCUMENTED RESIDUALS: a within-band float change preserving all three
    moments to rtol, and non-finite entries changing position/kind at a
    fixed per-band count, pass the float gate.
    """
    host = np.asarray(host)
    if host.ndim == 0 or host.shape[0] != n_bands:
        raise ValueError(
            f"band_fingerprint: leading axis "
            f"{host.shape[0] if host.ndim else '<0-d>'} != n_bands "
            f"{n_bands}")
    is_exact = host.dtype.kind in "biu"
    struct = [float(host.ndim), *map(float, host.shape),
              float(np.dtype(host.dtype).num)]
    if is_exact:
        vals = np.array([content_hash48(host[b]) for b in range(n_bands)],
                        dtype=np.float64)
    else:
        per_band = []
        for b in range(n_bands):
            flat = host[b].ravel()
            finite = flat[np.isfinite(flat)]
            f64 = finite.astype(np.float64)
            struct.append(float(flat.size - finite.size))
            per_band.extend([
                float(f64.sum()) if f64.size else 0.0,
                float((f64 * f64).sum()) if f64.size else 0.0,
                float(np.abs(f64).max()) if f64.size else 0.0,
            ])
        vals = np.array(per_band, dtype=np.float64)
    return np.array(struct, dtype=np.float64), vals, is_exact


def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
    """True iff every process's :func:`band_fingerprint` matches process 0's."""
    if rtol is None:
        rtol = _FLOAT_RTOL
    struct_ok = bool(np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
    return struct_ok and vals_ok


def checked_shard_put(arr, name, sharding, *, context, n_bands):
    """Gate a band-stacked field per band, then put WITHOUT broadcast or
    jax's whole-array device_put assert (walls 1+2 above).

    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
    round trip. Multi-process: per-band fingerprint gate (symmetric raise
    on real divergence), then ``jax.make_array_from_callback`` hands each
    process exactly its addressable slabs. Cross-process byte-identity of
    NON-owned bands is not required — owned bands are the only bytes that
    reach any device, and their drift is bounded by the gate.
    """
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    from jax.experimental import multihost_utils

    host = np.asarray(arr)
    struct, vals, is_exact = band_fingerprint(host, n_bands)
    g_struct = multihost_utils.process_allgather(struct)
    g_vals = multihost_utils.process_allgather(vals)
    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
        raise RuntimeError(
            f"{context}: band-stacked field {name!r} DIVERGES across "
            f"processes (exact_dtype={is_exact}, "
            f"gathered={g_vals.tolist()}) — a real config/grid "
            f"inconsistency, not autotune noise; refusing to shard it.")
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])


def assert_pytree_bytes_equal(tree, what):
    """Cheap multi-process replacement for the per-leaf assert_equal that
    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
    symmetric raise on mismatch. No-op single-process.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    # Numeric python scalars included (codex r20 item 1): the scatter
    # paths jnp.asarray + put them, so a rank-divergent scalar must not
    # bypass the gate. Non-numeric leaves (None, strings) stay excluded.
    leaves = [x for x in jax.tree_util.tree_leaves(tree)
              if hasattr(x, "ndim") or isinstance(x, (int, float, complex))]
    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
                    dtype=np.float64)
    g = multihost_utils.process_allgather(vals)
    if not bool(np.all(g == g[0])):
        bad = [i for i in range(len(leaves))
               if not bool(np.all(g[:, i] == g[0, i]))]
        raise RuntimeError(
            f"{what}: array leaves {bad} differ across processes (48-bit "
            f"byte digests disagree) — the per-process inputs are NOT "
            f"identical, which jax's device_put assert would have refused. "
            f"Fix the per-process build before sharding.")


def addressable_shard_put(arr, sharding):
    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
    consistency the CALLER has already gated (state/forcing pytrees via
    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    host = np.asarray(arr)
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])
        payload, problem = coerce_count(value, absent=absent)
        if problem is not None:
            problems.append((label, problem))
        return payload

    flags = (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        float(len(names)),
        name_digest48(names),
        name_digest48(sizes),
        _count(getattr(grid, "n_lat", None), "grid.n_lat", absent=0.0),
        _count(getattr(grid, "n_lon", None), "grid.n_lon", absent=0.0),
        # The geometry array fields that `build_band_grids` slices and
        # `_replicated_put` broadcasts, by dtype + shape.
        tree_schema_digest48(grid),
        float(bool(fold is not None and getattr(fold, "is_active", False))),
        # ONE digest over EVERY static scalar of the ocean config instead of a
        # hand-picked few: the step body branches on `outer_integrator`, the
        # tracer integrator, the polar filter, the freeze floor and the EW
        # overlap, and NONE of them were agreed (codex round-4, blocker 3).
        # A valid/invalid or euler/ab2 split makes one rank raise during
        # tracing while its peer compiles a different program.
        config_digest48(getattr(model, "config", None)),
        # `_build_band_vertex_masks` RAISES when this cache is unprimed, and
        # it runs before the schema collective -- so its presence must be
        # agreed first or an unprimed rank dies while its peer blocks
        # (codex round-4, blocker 4).
        float(getattr(model, "_vertex_mask", None) is not None),
    )
    assert_flags_agree(_OCEAN_SPMD_ENTRY_FLAGS, flags, context=where)
    # AFTER the collective only: symmetric on every rank (see atm twin).
    for label, problem in problems:
        raise ValueError(f"{where}: {label} {problem}")


# Ordered flag names for the PER-INVOCATION gate on the returned ocean SPMD
# callable. STATIC tuple: fixed width, never rank-local.
_OCEAN_CALL_ENTRY_FLAGS = (
    "has_mesh", "n_dev", "axis_names", "axis_sizes",
    "state_schema", "has_forcing", "forcing_schema",
    "has_aux", "aux_schema",
)


def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str,
                           aux=None) -> None:
    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.

    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
    forcing layout that is invalid on one rank only makes that rank raise while
    its peers enter the collective program -- a hang.  The state + forcing leaf
    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
    it, so two processes with different optional fields compile different
    programs.

    Cost: one small allgather per CALL, and an exact no-op under a single
    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
    on cache misses is NOT a valid optimisation.
    """
    names, sizes = _ocean_mesh_axis_terms(mesh)
    assert_flags_agree(_OCEAN_CALL_ENTRY_FLAGS, (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        name_digest48(names),
        name_digest48(sizes),
        tree_schema_digest48(state),
        float(forcing is not None),
        (tree_schema_digest48(forcing) if forcing is not None
         else FLAG_ABSENT),
        # aux (codex r20 item 2): a rank-local None-vs-provided or schema
        # mismatch must fail HERE, not desynchronize the jit call below.
        float(aux is not None),
        (tree_schema_digest48(aux) if aux is not None else FLAG_ABSENT),
    ), context=where)


def make_sharded_ocean_step(model, mesh):
    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
    sponge=None, t_seconds=None) -> state`` running ``model.step``
    lat-band-SPMD.

    The forcing channels mirror ``model.step``'s keyword surface: pass
    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
    dynamics-only program; each distinct None<->populated combination
    compiles (and caches) its own executable.

    Parameters
    def _validate_forcing_layout(forcing):
        """Loudly refuse forcing leaves the lat-band shard cannot split.

        Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])``
        — there are no v-face ``(n_lat+1, …)`` forcing arrays, so no
        ``v_lower`` handling.  A wrong-leading-dim leaf would otherwise die
        inside shard_map with an opaque divisibility error.
        """
        leaves, _ = jax.tree_util.tree_flatten_with_path(forcing)
        for path, leaf in leaves:
            nd = int(getattr(leaf, "ndim", np.ndim(leaf)))
            if nd == 1:
                # No OMIP forcing field is 1-D; _lat_spec would shard a
                # profile/staggered vector over "lat" silently-wrong
                # (codex r1 #2).
                raise ValueError(
                    f"sharded ocean step: forcing leaf "
                    f"{jax.tree_util.keystr(path)} is 1-D "
                    f"(shape {tuple(leaf.shape)}); forcing must be "
                    f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
                    f"scalars.")
            if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
                raise ValueError(
                    f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
                    f"has leading dim {leaf.shape[0]} != n_lat "
                    f"({n_lat_global}); forcing must be cell-centered "
                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")

    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
                     sponge=None, t_seconds=None, aux=None):
        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
        # inlines the inner call), concrete closure arrays become
        # OUTER-trace constants whose value the MLIR handler cannot fetch
        # for non-addressable arrays (broken since #1370-iii sharded the
        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
        # jit boundary as an ARGUMENT and pass it back here.
        # ONE forcing operand: None fields drop out of the pytree structure,
        # so specs derived by tree.map skip them automatically and the
        # structure key below distinguishes every None<->array combination.
        _agree_ocean_spmd_call(
            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
            where="make_sharded_ocean_step.step", aux=aux)
        forcing = (freshwater, surface_forcing, sponge, t_seconds)
        _validate_forcing_layout((freshwater, surface_forcing, sponge))
        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
        # forcing leaves' RANKS: in_specs/out_specs are derived from them,
        # so a later call with a different structure (an optional field
        # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
        # the restoring lane passing no forcing at all) OR a same-field
        # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
        # OR 3-D full-rank — same structure, different _lat_spec; codex r1
        # #1) must rebuild the shard_map rather than reuse stale specs.
        forcing_ndims = tuple(
            int(getattr(leaf, "ndim", np.ndim(leaf)))
            for leaf in jax.tree.leaves(forcing))
        # The SPMD fused-halo switch is read at TRACE time inside the pad
        # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
        # step object must rebuild the shard_map, not reuse a stale jaxpr
        # (codex, audit item 7).
        import os as _os

        _fused_halo = _os.environ.get(
            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
        key = (jax.tree.structure(state), jax.tree.structure(forcing),
               forcing_ndims, _fused_halo)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(_lat_spec, state)
            # Forcing leaves are cell-centered -> plain lat-band specs;
            # scalars (t_seconds) replicate, exactly like dt.
            forcing_spec = jax.tree.map(_lat_spec, forcing)
            # Stage (iii): the stacks are banded on their leading axis, so
            # the shard_map spec matches their P("lat") placement (the body
            # indexes its local (1, ...) slab at [0]).
            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
            vmask_spec = P("lat")
            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
            # replication check); the band halo reads neighbour-rank data so
            # disable it (same as the validated PCG / halo-parity shard_maps).
            fn = jax.jit(shard_map(
                _body,
                mesh=mesh,
                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
                out_specs=in_spec,
                check_vma=False,
            ))
            _cache[key] = fn
        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
        # previous backend (codex finding): leaving it globally armed makes a
        # later serial/full-domain ocean call take SPMD-only branches
        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
        # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
        # the backend armed (baking the SPMD halo/reduction ops into the
        # compiled program); later calls reuse the cached compile, and the
        # arm/restore keeps any interleaved serial path untouched.
        # Save+restore the FULL backend state (backend + MPI topology + SPMD
        # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
        # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
        # topology (codex).
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh,
        )
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        activate_latlon_spmd_halo(mesh)
        try:
            _geom, _vmask = aux if aux is not None else (geom_stacks,
                                                        vmask_stack)
            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    # Expose the stacks so outer-jit callers can pass them as arguments
    # (see the ``aux`` note in the signature).
    sharded_step.aux = (geom_stacks, vmask_stack)
    return sharded_step


def make_sharded_ocean_step_global(model, mesh):
    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
    that takes a GLOBAL (single-device-layout) state + forcing and returns a
    GLOBAL state — the minimal-diff driver entry point.

    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
    the OMIP host loop keep operating on a normal full-domain state — the per-step
    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
    should use :func:`make_sharded_ocean_step` directly to stay sharded).

    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
    """
    # FIRST statement: agree every rank-local input before ANY
    # rank-local check can raise or return (codex round-2).
    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step_global")
    if mesh is None:                   # single-device: plain step
        return lambda state, dt, surface_forcing=None, freshwater=None: (
            model.step(state, dt, freshwater=freshwater,
                       surface_forcing=surface_forcing))

    inner = make_sharded_ocean_step(model, mesh)

    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
                               where="make_sharded_ocean_step_global.step")
        # Scatter the global state AND forcing to the band layout
        # explicitly (the old comment claimed inner sharded the forcing;
        # it forwarded it global and relied on implicit JIT input
        # placement — jax's whole-array device_put assert under
        # multicontroller, the nd-linear wall this module removes).
        ss = shard_state_latlon(state, mesh)
        ss = inner(ss, dt,
                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
                   freshwater=shard_forcing_latlon(freshwater, mesh))
        return gather_state_latlon(ss, mesh)

    return sharded_step_global
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:315:    assert_pytree_bytes_equal(state, "shard_state_latlon")
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:611:    "has_aux", "aux_schema",
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:615:def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str,
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:890:        _agree_ocean_spmd_call(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:999:        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
./packages/core/legoesm/parallel/geometry_consistency.py:48:    "assert_pytree_bytes_equal",
./packages/core/legoesm/parallel/geometry_consistency.py:49:    "band_fingerprint",
./packages/core/legoesm/parallel/geometry_consistency.py:50:    "band_fingerprints_agree",
./packages/core/legoesm/parallel/geometry_consistency.py:619:def band_fingerprint(host, n_bands):
./packages/core/legoesm/parallel/geometry_consistency.py:640:            f"band_fingerprint: leading axis "
./packages/core/legoesm/parallel/geometry_consistency.py:665:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
./packages/core/legoesm/parallel/geometry_consistency.py:666:    """True iff every process's :func:`band_fingerprint` matches process 0's."""
./packages/core/legoesm/parallel/geometry_consistency.py:693:    struct, vals, is_exact = band_fingerprint(host, n_bands)
./packages/core/legoesm/parallel/geometry_consistency.py:696:    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
./packages/core/legoesm/parallel/geometry_consistency.py:706:def assert_pytree_bytes_equal(tree, what):
./packages/core/legoesm/parallel/geometry_consistency.py:737:    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
./tests/ocean/unit/test_isoneutral_slope_density.py:514:        (total, K33), grad = jax.value_and_grad(k33_sum, has_aux=True)(T)
./tests/ocean/unit/test_sharded_geom_fingerprint.py:13:    band_fingerprint as geom_band_fingerprint,
./tests/ocean/unit/test_sharded_geom_fingerprint.py:14:    band_fingerprints_agree,
./tests/ocean/unit/test_sharded_geom_fingerprint.py:23:    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
./tests/ocean/unit/test_sharded_geom_fingerprint.py:34:    assert band_fingerprints_agree(*_gather(a, a.copy()))
./tests/ocean/unit/test_sharded_geom_fingerprint.py:41:    assert band_fingerprints_agree(*_gather(a, b))
./tests/ocean/unit/test_sharded_geom_fingerprint.py:63:    assert not band_fingerprints_agree(*_gather(a, b))
./tests/ocean/unit/test_sharded_geom_fingerprint.py:73:    assert not band_fingerprints_agree(*_gather(a, b))
./tests/ocean/unit/test_sharded_geom_fingerprint.py:82:    assert not band_fingerprints_agree(*_gather(a, b))
./tests/ocean/unit/test_sharded_geom_fingerprint.py:89:    assert not band_fingerprints_agree(*_gather(a, b))
./tests/ocean/unit/test_sharded_geom_fingerprint.py:95:    fa = geom_band_fingerprint(a, N_BANDS)
./tests/ocean/unit/test_sharded_geom_fingerprint.py:96:    fb = geom_band_fingerprint(b, N_BANDS)
./tests/ocean/unit/test_sharded_geom_fingerprint.py:104:        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
./packages/ml/legoesm/training/neural_gcm_spectral.py:1395:        scalar suitable for ``has_aux=True`` accumulation.
./packages/ml/legoesm/training/neural_gcm_spectral.py:1590:    directly and use ``has_aux=True`` in
./packages/ml/legoesm/training/neural_gcm_spectral.py:2702:                loss_fn, has_aux=True,
./packages/ml/legoesm/training/neural_gcm_spectral.py:2729:                loss_fn, has_aux=True,
./packages/ml/legoesm/training/neural_gcm_spectral.py:3838:                loss_fn, has_aux=True,
./scripts/run/train_land_params_era5.py:174:    vg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
./scripts/run/run_aimip_amip_finetune.py:429:                lambda p: _sample_loss(p, sample, _n), has_aux=True,
./tests/unit/test_geometry_consistency.py:1019:         "_agree_ocean_spmd_call"),
./tests/unit/test_geometry_consistency.py:1021:         "_agree_ocean_spmd_call"),
./scripts/run/train_multilayer_land_era5.py:657:    vg = jax.jit(jax.value_and_grad(loss_ml, has_aux=True))
./tests/unit/test_diff_atmosphere_convection_micro.py:522:        value_and_grad = jax.jit(jax.value_and_grad(mask_sum, has_aux=True))

exec
/usr/bin/bash -lc "sed -n '960,1060p' tests/unit/test_geometry_consistency.py && sed -n '1,130p' tests/ocean/unit/test_sharded_geom_fingerprint.py && git show --format=fuller --stat HEAD && git log --oneline -5 && git diff --check HEAD~1 HEAD -- ':"'!.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 323ms:
                        and sub.func.id == "NamedSharding"):
                    hits.append(node)
                    break
        return hits

    @pytest.mark.parametrize("lane,mod_fn,gated,ungated", _LANES,
                             ids=[l[0] for l in _LANES])
    def test_exempt_entries_contain_no_mesh_device_put(self, lane, mod_fn,
                                                       gated, ungated):
        fns = _top_level_functions(_module_tree(mod_fn()))
        for name in ungated:
            fn = fns.get(name)
            if fn is None:                    # a module alias, not a def
                continue
            hits = self._mesh_device_put_calls(fn)
            assert not hits, (
                f"{lane}.{name} is on the UNGATED allow-list but contains a "
                f"jax.device_put(..., NamedSharding(...)): under multi-process "
                f"that is serviced by an ALL-GATHER (see "
                f"latlon_spmd.shard_leaf), so the exemption reason is FALSE. "
                f"Gate it or migrate it to shard_leaf.")

    def test_the_detector_is_not_vacuous(self):
        """It must FIND the pattern it claims to reject."""
        fn = _parse_fn("""
            def f(state, mesh):
                return jax.device_put(state, NamedSharding(mesh, P()))
        """)
        assert self._mesh_device_put_calls(fn), (
            "fixture: the detector must match the hazard shape, else "
            "test_exempt_entries_contain_no_mesh_device_put passes vacuously")
        clean = _parse_fn("""
            def g(state):
                return state + 1
        """)
        assert not self._mesh_device_put_calls(clean)


class TestReturnedClosuresAreGated:
    """codex round-4, blocker 7 + MAJOR.

    The factories are gated, but the CLOSURES they return are public entry
    points too — a user holds ``step = make_...(...)`` and calls it — and they
    ran rank-local refusals (``refuse_unthreaded_stateful_physics``, the 2-D
    ``NotImplementedError``, ``_validate_forcing_layout``) BEFORE entering
    their collective program.  The top-level enumerator could not see them.
    """

    # (module, factory, returned-closure name, gate)
    CLOSURES = [
        ("atm", "make_sharded_atm_latlon_step", "sharded_step",
         "_agree_spmd_call"),
        ("atm", "make_sharded_atm_latlon_step_2d", "sharded_step",
         "_agree_spmd_call"),
        ("atm", "make_sharded_atm_latlon_segment", "segment",
         "_agree_spmd_call"),
        ("atm", "make_sharded_atm_latlon_segment_2d", "segment",
         "_agree_spmd_call"),
        ("ocean", "make_sharded_ocean_step", "sharded_step",
         "_agree_ocean_spmd_call"),
        ("ocean", "make_sharded_ocean_step_global", "sharded_step_global",
         "_agree_ocean_spmd_call"),
        ("opsplit", "make_sharded_operator_split_step", "sharded_split_step",
         "_agree_opsplit_mesh_entry"),
    ]

    _MODULES = {"atm": _atm_module, "ocean": _ocean_module,
                "opsplit": _opsplit_module}

    @pytest.mark.parametrize("lane,factory,closure,gate", CLOSURES,
                             ids=[f"{c[0]}.{c[1]}.{c[2]}" for c in CLOSURES])
    def test_closure_gate_is_its_first_direct_statement(self, lane, factory,
                                                        closure, gate):
        import ast
        fns = _top_level_functions(_module_tree(self._MODULES[lane]()))
        outer = fns[factory]
        inner = [n for n in ast.walk(outer)
                 if isinstance(n, ast.FunctionDef) and n.name == closure]
        assert inner, f"{factory} defines no closure named {closure}"
        # The SHARDED closure is the one that is returned; take the last
        # definition, which is the multi-device body (an early serial
        # fallback may share the name in a `mesh is None` branch).
        idx = _direct_gate_index(inner[-1], gate)
        assert idx == 0, (
            f"{lane}.{factory}.{closure}: {gate}(...) must be its FIRST "
            f"DIRECT statement (got index {idx}). The rank-local refusals "
            f"below it — stateful-physics refusal, forcing-layout validation, "
            f"the 2-D carry NotImplementedError — otherwise raise on one rank "
            f"while a peer enters the shard_map: a HANG.")

    @pytest.mark.parametrize("lane,factory,closure,gate", CLOSURES,
                             ids=[f"{c[0]}.{c[1]}.{c[2]}" for c in CLOSURES])
    def test_closure_is_actually_returned(self, lane, factory, closure, gate):
        """Guards against the gate being attached to a dead nested def."""
        import ast
        fns = _top_level_functions(_module_tree(self._MODULES[lane]()))
        returned = {n.value.id for n in ast.walk(fns[factory])
                    if isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Name)}
        assert closure in returned, (
            f"{factory} does not return {closure}; the gate would be on a "
"""Unit tests for the band-geometry cross-process fingerprint gate.

The gate decides whether ``make_sharded_ocean_step`` accepts per-process
band-geometry stacks without the (removed, nd-linear-cost) process-0
broadcast — see the 2026-08-03 fix note at the sharded put. These tests
pin the gate's discrimination properties single-process (the
multicontroller allgather wiring is exercised by the distributed suite).
"""
import numpy as np
import pytest

from legoesm.parallel.geometry_consistency import (
    band_fingerprint as geom_band_fingerprint,
    band_fingerprints_agree,
)

N_BANDS = 4
SHAPE = (N_BANDS, 6, 8)


def _gather(*hosts):
    """Simulate process_allgather over per-process fingerprints."""
    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
    exacts = {fp[2] for fp in fps}
    assert len(exacts) == 1
    g_struct = np.stack([fp[0] for fp in fps])
    g_vals = np.stack([fp[1] for fp in fps])
    return g_struct, g_vals, fps[0][2]


def test_identical_float_stacks_agree():
    rng = np.random.default_rng(0)
    a = rng.normal(size=SHAPE).astype(np.float32)
    assert band_fingerprints_agree(*_gather(a, a.copy()))


def test_ulp_scale_drift_agrees():
    rng = np.random.default_rng(1)
    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    b = np.nextafter(a, np.inf)  # a TRUE 1-ULP elementwise drift
    assert band_fingerprints_agree(*_gather(a, b))


def test_band_local_drift_refused_where_global_gate_passed():
    # THE r14 discrimination case: a band-local drift SMALL enough that the
    # old whole-array gate (sum/sumsq/absmax at rtol 1e-5) accepts it, with
    # the global absmax held by an UNAFFECTED band — the per-band gate must
    # still refuse.
    rng = np.random.default_rng(2)
    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    a[0, 0, 0] = 500.0          # absmax lives in band 0
    b = a.copy()
    b[2] *= 1.0 + 3e-5          # one band drifts; global sums move ~7e-6 rel

    def _global_moments(x):
        f = x.ravel().astype(np.float64)
        return np.array([f.sum(), (f * f).sum(), np.abs(f).max()])

    # the OLD global gate would have ACCEPTED this pair...
    assert np.allclose(_global_moments(a), _global_moments(b),
                       rtol=1e-5, atol=0.0)
    # ...the per-band gate refuses it.
    assert not band_fingerprints_agree(*_gather(a, b))


def test_exact_dtype_permutation_refused():
    # Moment fingerprints are blind to permutations; the positional
    # per-band byte digest must not be.
    a = np.zeros(SHAPE, dtype=np.int32)
    a[1, 2, 3] = 1
    b = np.zeros_like(a)
    b[1, 3, 2] = 1  # same count, different position, same band
    assert not band_fingerprints_agree(*_gather(a, b))


def test_bool_mask_two_cell_flip_refused():
    a = np.zeros(SHAPE, dtype=bool)
    a[0, 0, 0] = True
    b = a.copy()
    b[0, 0, 0] = False
    b[0, 5, 7] = True  # true-count preserved
    assert not band_fingerprints_agree(*_gather(a, b))


def test_nonfinite_count_mismatch_refused():
    a = np.ones(SHAPE, dtype=np.float32)
    b = a.copy()
    b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
    assert not band_fingerprints_agree(*_gather(a, b))


def test_shape_mismatch_refused():
    a = np.ones(SHAPE, dtype=np.float32)
    b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
    fa = geom_band_fingerprint(a, N_BANDS)
    fb = geom_band_fingerprint(b, N_BANDS)
    # Different shapes -> different struct lengths; the agree helper is
    # only called on stackable gathers, so assert the structs differ.
    assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])


def test_wrong_leading_axis_raises():
    with pytest.raises(ValueError):
        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
commit b336bb123463368550efa493a015ac56cd245faa
Author:     Pierre Gentine <b381103@levante2.lvt.dkrz.de>
AuthorDate: Mon Aug 3 09:08:26 2026 +0200
Commit:     Pierre Gentine <b381103@levante2.lvt.dkrz.de>
CommitDate: Mon Aug 3 09:08:26 2026 +0200

    fix(parallel): r20 hardening — scalar leaves in digest gate, aux in ocean call gate, ValueError+__all__; atm broadcast tracked
    
    Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_016q4upjC5GViwutba4AFQUC

 .../scaling_campaign/codex_merge_r20_2026-08-03.md | 10735 +++++++++++++++++++
 .../scaling/levante_campaign_2026-07-24.md         |    20 +
 .../core/legoesm/parallel/geometry_consistency.py  |    15 +-
 .../legoesm/ocean/dynamics/sharded_ocean_step.py   |    10 +-
 4 files changed, 10775 insertions(+), 5 deletions(-)
b336bb123 fix(parallel): r20 hardening — scalar leaves in digest gate, aux in ocean call gate, ValueError+__all__; atm broadcast tracked
e13de3afe Merge remote-tracking branch 'origin/main' into worktree-scaling-campaign
1b71f5dc3 fix(plot): repair literal-newline in oc panel note
ac4289131 docs(scaling): OCEAN AT HUNDREDS — LL2304 @96/@128 receipts (11.63/13.00 GC/s, eff 0.84)
625b50f9d fix(ocean-spmd): three stacked multicontroller walls — assert-free puts, exact-hash gate, aux threading

exec
/usr/bin/bash -lc "sed -n '270,455p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && rg -n \"def content_hash48|def tree_schema_digest48|def assert_flags_agree|FLAG_ABSENT|def name_digest48\" packages/core/legoesm/parallel/geometry_consistency.py && sed -n '1,220p' packages/core/legoesm/parallel/geometry_consistency.py && git show HEAD~1:packages/core/legoesm/parallel/geometry_consistency.py | sed -n '615,740p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 280ms:
    #1362 round 4, blockers 5-6.  Both directions are collective here: the
    gather's ``replicate_leaf`` compiles a jit identity with replicated
    ``out_shardings``, and the scatter's DIRECT ``device_put`` of a full
    global array onto a cross-process ``NamedSharding`` falls back to an
    all-gather (documented on ``latlon_spmd.shard_leaf``) -- so the earlier
    "SCATTER, therefore no collective" exemption was FALSE for this lane.
    One collective runs PER LEAF, so the leaf schedule (optional fields,
    dtypes, shapes) is rank-local data and is folded into one digest.
    """
    names, sizes = _ocean_mesh_axis_terms(mesh)
    assert_flags_agree(_OCEAN_MESH_ENTRY_FLAGS, (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        float(len(names)),
        name_digest48(names),
        name_digest48(sizes),
        float(tree is not None),
        tree_schema_digest48(tree) if tree is not None else FLAG_ABSENT,
    ), context=where)


def shard_state_latlon(state, mesh):
    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.

    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
    staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
    ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
    ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
    by the band halo.  ``None`` fields pass through.  The inverse is
    :func:`gather_state_latlon`.

    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
    """
    # FIRST statement: a DIRECT ``device_put`` of full global arrays onto a
    # cross-process ``NamedSharding`` is serviced by an ALL-GATHER, and one
    # runs per leaf -- so both the mesh and the leaf SCHEDULE are rank-local
    # inputs to a collective (#1362 round 4, blockers 5-6).
    _agree_ocean_mesh_entry(mesh, state, where="shard_state_latlon")
    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
    # v-face row (regular pole wall OR tripole seam/cap row) must be
    # wall-masked — the carrier drops it and reconstructs it as zero, which
    # would silently delete a LIVE seam row.  Host-side check on the
    # concrete state (this fn runs outside jit).
    assert_pytree_bytes_equal(state, "shard_state_latlon")
    vm = getattr(state, "v_mask", None)
    if vm is not None:
        import numpy as _np

        if _np.asarray(vm.data)[-1].any():
            raise ValueError(
                "shard_state_latlon: the state's TOP v-face row is LIVE "
                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
                "drops that row and reconstructs it as the pole/cap wall "
                "zero, which would silently delete seam velocities. "
                "Mask the cap row (the tripole cap convention) or extend "
                "the carrier before sharding this state.")

    def _shard_cell(field):
        if field is None:
            return None
        sh = NamedSharding(mesh, _lat_spec(field.data))
        return field.replace(data=addressable_shard_put(field.data, sh))

    def _shard_v(field):
        if field is None:
            return None
        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
        # input's top v-row is already zero (a valid masked regular-grid state;
        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
        # bit round-trip of that one row.  NOT for a tripole north fold (raises
        # in make_sharded_ocean_step) where the top row is a live fold partner.
        nlat1 = field.data.shape[0]
        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
        sh = NamedSharding(mesh, _lat_spec(v_lower))
        return field.replace(data=addressable_shard_put(v_lower, sh))

    updates = {}
    for name in _V_STAGGERED_STATE_FIELDS:
        updates[name] = _shard_v(getattr(state, name))
    # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
    # v-staggered set and not None get the cell sharding; None stays None.
    for name in state._fields:
        if name in _V_STAGGERED_STATE_FIELDS:
            continue
        val = getattr(state, name)
        if val is None:
            updates[name] = None
        elif hasattr(val, "data"):                 # a Field
            updates[name] = _shard_cell(val)
        else:
            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
            arr = jnp.asarray(val)
            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
            # scalars replicate.
            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
    return state._replace(**updates)


def shard_forcing_latlon(forcing, mesh):
    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.

    Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
    cell->face wind-stress interpolation happens INSIDE the step through the
    SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
    no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
    untouched (they vanish from the pytree structure, matching the specs the
    step derives).  ``forcing=None`` returns ``None``.
    """
    # FIRST statement: the `forcing is None or mesh is None` return below
    # SKIPS every per-leaf put, so a rank with no forcing would leave a peer
    # blocked in one (#1362 round 4, blocker 6).
    _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
    if forcing is None or mesh is None:
        return forcing
    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")

    def _put(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))

    return jax.tree.map(_put, forcing)


def shard_forcing_stack_latlon(stack, mesh):
    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
    block-scan (the ``run_omip`` JRA55 lanes; see
    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).

    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
    block builders stack ``N`` steps / raw records along a LEADING axis,
    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
    time index stays shard-local so the in-scan interpolation needs no
    cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
    1-D metadata / scalars replicate; ``None`` and non-array leaves pass
    through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).

    CONTRACT: rank is the ONLY signal used, so a rank-2 leaf is assumed to
    be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
    metadata that is genuinely rank-2 but NOT lat-major (e.g. a
    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
    metadata 1-D (or replicate it explicitly) before it reaches this helper.

    Keeping this next to :func:`shard_state_latlon` means the driver and
    the parity tests share ONE layout definition — the block-scan forcing
    stack must be laid out consistently with the state the sharded step
    carries, and a second copy would drift.
    """
    # FIRST statement: same rank-local early-return + per-leaf put as
    # shard_forcing_latlon (#1362 round 4, blocker 6).
    _agree_ocean_mesh_entry(mesh, stack,
                            where="shard_forcing_stack_latlon")
    if mesh is None:
        return stack

    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")

    def _put(leaf):
        if leaf is None or not hasattr(leaf, "ndim"):
            return leaf
        arr = jnp.asarray(leaf)
        if arr.ndim >= 3:
            spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
        elif arr.ndim == 2:
            spec = P("lat", None)
        else:
            spec = P()
        return addressable_shard_put(arr, NamedSharding(mesh, spec))

    return jax.tree.map(_put, stack)


def append_vface_wall_row(v_lower):
    """Rebuild the full ``(n_lat+1, ...)`` staggered v-array from the lat-band
63:    "FLAG_ABSENT",
81:# can take.  They must also not collide with each other: ``FLAG_ABSENT`` used
86:FLAG_ABSENT = -6.0e15
116:def coerce_count(value, *, absent: float = FLAG_ABSENT):
143:    ``None`` maps to ``absent`` (default :data:`FLAG_ABSENT`, itself outside
175:def coerce_bool(value, *, absent: float = FLAG_ABSENT):
264:def tree_schema_digest48(tree) -> float:
343:def content_hash48(arr) -> float:
358:def name_digest48(names) -> float:
446:def assert_flags_agree(names, values, *, context: str) -> None:
"""Cross-process agreement checks for per-process-recomputed SPMD geometry.

Every multi-controller SPMD lane faces the same hazard: each process rebuilds
the band/tile geometry from the same config, then hands it to a REPLICATED
``device_put``.  A ``P()`` (fully-replicated) put ASSERTS the value is
bit-identical on every process, and per-process XLA autotuning on
device-derived grid fields makes the last ULPs differ at larger sizes (job
26450848: LL576 np=4, area-scale fields differing at 1e-7 relative), which
trips that assert.

The remedy is to broadcast process 0's bytes — but broadcasting BLINDLY would
silently paper over a REAL cross-process inconsistency (a different wet
domain, a different field list, a mixed ``jax_enable_x64``), turning a loud
crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
fingerprint must agree first, and a disagreement RAISES.

This module is the ONE implementation of that protocol.  It was extracted
from ``ocean.dynamics.sharded_ocean_step`` (where it was developed and
hardened over five rounds of adversarial review) so the atmosphere lat-lon
lane — which had the identical defect (#1362) — reuses it instead of growing
a second, drifting copy.  Per legoESM's no-duplicated-numerics rule, new SPMD
lanes MUST call these helpers rather than re-derive the fingerprints.

Sequencing contract, in this order:

1. :func:`assert_schema_agrees` ONCE, before any per-field work — a single
   fixed-shape collective that every process reaches.  A process-dependent
   field selection (e.g. an optional mask present on some ranks only) would
   otherwise DESYNCHRONIZE the per-field gathers below instead of failing
   with a clear message.
2. :func:`broadcast_checked` per field, in an order identical on every
   process.

NO DEADLOCK RISK: every process fingerprints the same fields in the same
order and derives its verdict from the SAME gathered array, so the refusal is
symmetric — all raise or none.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np

__all__ = [
    "addressable_shard_put",
    "assert_pytree_bytes_equal",
    "band_fingerprint",
    "band_fingerprints_agree",
    "checked_shard_put",
    "content_hash48",
    "name_digest48",
    "schema_fingerprint",
    "assert_schema_agrees",
    "assert_flags_agree",
    "broadcast_checked",
    "coerce_count",
    "coerce_bool",
    "config_digest48",
    "tree_schema_digest48",
    "safe_repr",
    "FLAG_ABSENT",
    "FLAG_UNCOERCIBLE",
    "FLAG_OUT_OF_RANGE",
    "FLAG_NEGATIVE",
    "FLAG_MAX_EXACT",
    "FLAG_DIGEST_FAILED",
]

# --- entry-gate payload sentinels -------------------------------------------
# An entry gate turns rank-local scalars (n_steps, segment_steps, grid dims)
# into a fixed-width float payload.  Building that payload must NEVER raise:
# a rank that dies in `int(n_steps)` while its peers block in
# `process_allgather` is a HANG, which is strictly worse than the bug the gate
# exists to fix (codex 2026-07-29 round-3, blocker 3).  So an unusable value is
# mapped to a SENTINEL that travels through the collective; every rank then
# sees it in the gathered payload and the raise that follows is symmetric.
#
# The sentinels are large-magnitude NEGATIVE values that NO legitimate count
# can take.  They must also not collide with each other: ``FLAG_ABSENT`` used
# to be ``-1.0``, so a rank passing ``segment_steps=None`` and a peer passing
# ``-1`` produced the SAME payload entry, agreed, and then diverged downstream
# (codex round-4, blocker 1).  Counts are validated non-negative, so every
# sentinel is unreachable from valid data AND distinct from every other.
FLAG_ABSENT = -6.0e15
FLAG_UNCOERCIBLE = -8.0e15
FLAG_OUT_OF_RANGE = -7.0e15
FLAG_NEGATIVE = -5.0e15
FLAG_DIGEST_FAILED = -4.0e15
# 2**53 is the largest integer whose successor is exactly representable in
# float64.  Above it two DIFFERENT counts alias to the same payload entry, so
# the gate would pass a real divergence (codex round-3, minor 2).  Values past
# the bound are refused rather than silently compared.
FLAG_MAX_EXACT = 2.0 ** 53
_MAX_EXACT_INT = 2 ** 53


def safe_repr(value, limit: int = 120) -> str:
    """``repr(value)`` that cannot raise and cannot blow up the message.

    A user object whose ``__repr__`` raises would otherwise propagate out of
    the payload build — the very pre-collective throw the gates exist to
    remove (codex round-4, blocker 1).
    """
    try:
        text = repr(value)
    except Exception:                       # pragma: no cover - defensive
        try:
            text = f"<unrepresentable {type(value).__name__}>"
        except Exception:                   # pragma: no cover - defensive
            text = "<unrepresentable>"
    return text if len(text) <= limit else text[:limit] + "..."


def coerce_count(value, *, absent: float = FLAG_ABSENT):
    """Map a rank-local COUNT to an exactly-comparable entry-gate payload float.

    Returns ``(payload, problem)``.  ``problem`` is ``None`` when the value is
    usable; otherwise it is a human-readable clause naming the offending value,
    which the caller must raise AFTER its collective so the refusal is
    symmetric across processes.

    This function NEVER raises.  That is the whole point: it is called while
    ASSEMBLING a collective payload, upstream of the collective itself, where a
    raise deadlocks the peers (codex round-3, blocker 3).

    STRICT by type, not by coercibility (codex round-4, blocker 1).  Only a
    real non-negative Python/NumPy integer is accepted:

    * ``3.5`` is REJECTED.  ``int(3.5) == 3`` made a rank carrying ``3.5``
      indistinguishable from a peer carrying ``3``; the payloads agreed and
      then ``range(3.5)`` blew up on one rank alone while its peer entered the
      step collective.
    * ``bool`` is REJECTED.  ``True`` is not a step count, and silently
      encoding it as ``1`` hides a caller bug.
    * Arrays (even size-1) are REJECTED: ``int(arr)`` succeeds for size 1 and
      raises for size > 1, so accepting them makes the gate's behaviour depend
      on rank-local shape.
    * NEGATIVE integers get their OWN sentinel, so they can never collide with
      the "absent" encoding.

    ``None`` maps to ``absent`` (default :data:`FLAG_ABSENT`, itself outside
    the valid range) so a call site that does not carry the value still emits a
    FIXED-WIDTH payload.
    """
    if value is None:
        return float(absent), None
    # `bool` is a subclass of `int`, so it must be excluded FIRST.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return FLAG_UNCOERCIBLE, (
            "must be a non-negative Python/NumPy integer (got type "
            f"{type(value).__name__}: {safe_repr(value)}); every process must "
            "be launched with the same value")
    try:
        as_int = int(value)
    except Exception:                       # pragma: no cover - defensive
        return FLAG_UNCOERCIBLE, (
            f"could not be read as an integer ({safe_repr(value)})")
    if as_int < 0:
        return FLAG_NEGATIVE, (
            f"must be non-negative (got {safe_repr(value)})")
    # Range check on the INTEGER: converting to float first would already have
    # collapsed 2**53+1 onto 2**53, so the very aliasing this guards against
    # would be invisible to the guard.
    if as_int > _MAX_EXACT_INT:
        return FLAG_OUT_OF_RANGE, (
            f"is outside the exactly-comparable range n <= 2**53 (got "
            f"{safe_repr(value)}); beyond that bound two different counts "
            f"alias to the same float64 payload entry and the cross-process "
            f"agreement check would pass a real divergence")
    return float(as_int), None


def coerce_bool(value, *, absent: float = FLAG_ABSENT):
    """Strict, NON-THROWING tri-state encoder for a rank-local BOOLEAN flag.

    Returns ``(payload, problem)`` exactly like :func:`coerce_count`.
    ``None`` -> ``absent`` ("not applicable at this call site").

    Only a real ``bool`` / ``np.bool_`` is accepted.  ``bool(value)`` on an
    arbitrary object RAISES for a multi-element array ("truth value of an array
    is ambiguous") — inside a gate that is a pre-collective throw, i.e. a hang
    (codex round-4, blocker 2).  Anything else becomes a sentinel that travels
    through the collective and is refused symmetrically afterwards.
    """
    if value is None:
        return float(absent), None
    if isinstance(value, (bool, np.bool_)):
        return (1.0 if value else 0.0), None
    return FLAG_UNCOERCIBLE, (
        f"must be a bool (got type {type(value).__name__}: "
        f"{safe_repr(value)})")


def _canonical_config_terms(obj, prefix: str = "", depth: int = 0,
                            out=None, seen=None):
    """Flatten a config object into ORDERED ``"path=value"`` strings.

    Covers the STATIC scalars that select a compiled program: scheme literals,
    integrator names, and every feature-gating bool (``fix_mass``,
    ``fix_moisture``, ``use_polar_filter``, ...).  Arrays contribute only
    ``dtype`` + ``shape`` — comparing their VALUES is the job of
    :func:`broadcast_checked`, not of a cheap fixed-width entry gate.

    Never raises: any unreadable field becomes a ``<unreadable>`` term, which
    still participates in the comparison.
    """
    if out is None:
        out, seen = [], set()
    if depth > 4 or len(out) > 512:          # bounded work, bounded payload
        return out
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    fields = getattr(obj, "_fields", None)   # NamedTuple
    if fields is None:
        dc = getattr(obj, "__dataclass_fields__", None)
        fields = tuple(dc) if dc else None
    if fields is None:
    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).

    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
    already be schema-gated across processes (:func:`assert_schema_agrees`)
    — the payload widths depend on it, and mismatched widths would hang the
    allgather rather than raise.

    Exact dtypes (int/bool/uint): one positional 48-bit byte digest per
    band. Floats: per-band ``[sum, sum_of_squares, absmax]`` of finite
    entries plus per-band non-finite counts folded into ``struct``.
    Per-band (not whole-array) because each process's OWN bytes become the
    live inputs for the bands it owns under the assert-free put: a
    band-local drift must not hide in a whole-array sum (codex r14).
    DOCUMENTED RESIDUALS: a within-band float change preserving all three
    moments to rtol, and non-finite entries changing position/kind at a
    fixed per-band count, pass the float gate.
    """
    host = np.asarray(host)
    if host.shape[0] != n_bands:
        raise ValueError(
            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
            f"{n_bands}")
    is_exact = host.dtype.kind in "biu"
    struct = [float(host.ndim), *map(float, host.shape),
              float(np.dtype(host.dtype).num)]
    if is_exact:
        vals = np.array([content_hash48(host[b]) for b in range(n_bands)],
                        dtype=np.float64)
    else:
        per_band = []
        for b in range(n_bands):
            flat = host[b].ravel()
            finite = flat[np.isfinite(flat)]
            f64 = finite.astype(np.float64)
            struct.append(float(flat.size - finite.size))
            per_band.extend([
                float(f64.sum()) if f64.size else 0.0,
                float((f64 * f64).sum()) if f64.size else 0.0,
                float(np.abs(f64).max()) if f64.size else 0.0,
            ])
        vals = np.array(per_band, dtype=np.float64)
    return np.array(struct, dtype=np.float64), vals, is_exact


def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
    """True iff every process's :func:`band_fingerprint` matches process 0's."""
    if rtol is None:
        rtol = _FLOAT_RTOL
    struct_ok = bool(np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
    return struct_ok and vals_ok


def checked_shard_put(arr, name, sharding, *, context, n_bands):
    """Gate a band-stacked field per band, then put WITHOUT broadcast or
    jax's whole-array device_put assert (walls 1+2 above).

    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
    round trip. Multi-process: per-band fingerprint gate (symmetric raise
    on real divergence), then ``jax.make_array_from_callback`` hands each
    process exactly its addressable slabs. Cross-process byte-identity of
    NON-owned bands is not required — owned bands are the only bytes that
    reach any device, and their drift is bounded by the gate.
    """
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    from jax.experimental import multihost_utils

    host = np.asarray(arr)
    struct, vals, is_exact = band_fingerprint(host, n_bands)
    g_struct = multihost_utils.process_allgather(struct)
    g_vals = multihost_utils.process_allgather(vals)
    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
        raise RuntimeError(
            f"{context}: band-stacked field {name!r} DIVERGES across "
            f"processes (exact_dtype={is_exact}, "
            f"gathered={g_vals.tolist()}) — a real config/grid "
            f"inconsistency, not autotune noise; refusing to shard it.")
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])


def assert_pytree_bytes_equal(tree, what):
    """Cheap multi-process replacement for the per-leaf assert_equal that
    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
    symmetric raise on mismatch. No-op single-process.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    leaves = [x for x in jax.tree_util.tree_leaves(tree)
              if hasattr(x, "ndim")]
    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
                    dtype=np.float64)
    g = multihost_utils.process_allgather(vals)
    if not bool(np.all(g == g[0])):
        bad = [i for i in range(len(leaves))
               if not bool(np.all(g[:, i] == g[0, i]))]
        raise RuntimeError(
            f"{what}: array leaves {bad} differ across processes (48-bit "
            f"byte digests disagree) — the per-process inputs are NOT "
            f"identical, which jax's device_put assert would have refused. "
            f"Fix the per-process build before sharding.")


def addressable_shard_put(arr, sharding):
    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
    consistency the CALLER has already gated (state/forcing pytrees via
    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    host = np.asarray(arr)
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])

exec
/usr/bin/bash -lc "sed -n '240,380p' packages/core/legoesm/parallel/geometry_consistency.py && sed -n '430,500p' packages/core/legoesm/parallel/geometry_consistency.py && rg -n \"def _lat_spec|def shard_pytree|addressable_shard_put|assert_pytree_bytes_equal\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py packages/ocean/legoesm/ocean/dynamics/*.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 278ms:
    return out


def config_digest48(obj) -> float:
    """One fixed-width, order-sensitive digest of a config's STATIC scalars.

    Why a digest instead of a hand-listed set of flags: an entry gate that
    enumerates ``fold``/``anchor``/``polar`` by hand agrees only the fields
    somebody remembered.  ``fix_mass`` gates a global-area psum,
    ``outer_integrator`` selects a different program, ``fix_moisture`` adds a
    reduction — each was MISSING from the hand-written list (codex round-4,
    blocker 3).  Digesting every static scalar closes the class instead of the
    three instances, and costs ONE payload entry.

    Never raises; an internal failure returns :data:`FLAG_DIGEST_FAILED`,
    which still compares equal across ranks that fail identically and unequal
    against a rank that succeeded.
    """
    try:
        return name_digest48(_canonical_config_terms(obj))
    except Exception:                        # pragma: no cover - defensive
        return FLAG_DIGEST_FAILED


def tree_schema_digest48(tree) -> float:
    """Digest of a pytree's LEAF SCHEMA: ordered path, dtype and full shape.

    The gather/scatter entry points run one cross-process replication PER
    NON-``None`` LEAF, so the NUMBER and ORDER of those collectives is
    rank-local data: a state whose tracer dict differs across processes (extra
    species, different insertion order, different shape) produces mismatched
    schedules and hangs (codex round-4, blocker 5).  Folding the whole leaf
    schema into ONE fixed-width float makes that a clean symmetric raise.

    ``jax.tree_util`` key paths give a canonical, ORDER-SENSITIVE description
    (dict keys are sorted by ``tree_flatten_with_path``, so an insertion-order
    difference alone does not false-positive, while a KEY-SET difference does
    move the digest).  Never raises.
    """
    try:
        from jax.tree_util import tree_flatten_with_path, keystr
        leaves, _ = tree_flatten_with_path(tree)
        terms = []
        for path, leaf in leaves:
            dtype = getattr(leaf, "dtype", None)
            shape = getattr(leaf, "shape", None)
            terms.append(
                f"{keystr(path)}:{safe_repr(dtype, 32)}:"
                f"{safe_repr(tuple(shape) if shape is not None else None, 64)}")
        return name_digest48(terms)
    except Exception:                        # pragma: no cover - defensive
        return FLAG_DIGEST_FAILED


def _dtype_kind_and_ndim(a):
    """``(kind, ndim)`` for the schema digest, tolerant of a plain scalar.

    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
    both without materialising, so no device sync).  A plain Python scalar or
    list has neither; falling through to ``np.asarray`` there is FREE (it is
    already host data) and, critically, keeps this function from dying with a
    bare ``AttributeError`` BEFORE :func:`assert_schema_agrees` reaches its
    collective — a rank-local raise ahead of a collective is a HANG, so
    "fail explicitly" here must NOT mean "raise here" (codex round-3, minor 3).

    An unsupported dtype class (object/str) is reported as ``unsupported:<k>``
    rather than being silently bucketed with the float fields, so a
    disagreement about it is visible in the digest and a same-on-all-ranks
    unsupported field fails later in :func:`broadcast_checked` with its own
    message instead of here.
    """
    dtype = getattr(a, "dtype", None)
    if dtype is None:
        host = np.asarray(a)
        dtype, ndim = host.dtype, host.ndim
    else:
        ndim = int(getattr(a, "ndim", np.ndim(a)))
    k = np.dtype(dtype).kind
    if k in "biu":
        kind = "exact"
    elif k in "fc":
        kind = "inexact"
    else:
        kind = f"unsupported:{k}"
    return kind, int(ndim)


# Every per-field collective payload is padded to these FIXED widths.  A
# payload whose LENGTH depends on rank-local data (dtype class, ndim,
# non-finite count) would let two processes enter `process_allgather` with
# different shapes and DEADLOCK -- the exact failure this module exists to
# turn into a clean symmetric raise (codex 2026-07-29, blocker 2; the flaw was
# inherited from the pre-extraction ocean implementation, so fixing it here
# fixes BOTH lanes).
_STRUCT_WIDTH = 8
_VALS_WIDTH = 3

# Relative tolerance for FLOAT geometry fields. Only ULP-scale autotune drift
# is expected there; quantize-then-assert-equal false-positived on a rounding
# boundary (job 26453240), so compare with a tolerance instead.
_FLOAT_RTOL = 1e-5


def content_hash48(arr) -> float:
    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.

    Used to compare EXACT-dtype arrays (masks, index tables) across
    processes: unlike moment fingerprints, a byte digest is positional, so a
    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
    under 2**53 so it survives the float64 ``process_allgather`` payload
    exactly. Not cryptographic — collision-resistance at 2**-48 is far
    beyond the ~10 setup-time comparisons this guard makes.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(a.tobytes(), digest_size=6)
    return float(int.from_bytes(h.digest(), "big"))


def name_digest48(names) -> float:
    """Order-sensitive, UNAMBIGUOUS digest of a sequence of names.

    Uses a NUL separator, which cannot occur in a Python identifier or any
    legoESM field name, so ``["a,b", "c"]`` and ``["a", "b,c"]`` cannot
    collide.  A plain ``",".join`` COULD (codex 2026-07-29, minor 5): those
    two lists have the same length, so a count check does not separate them
    either.
    """
    joined = "\x00".join(names).encode()
    return float(int.from_bytes(
        hashlib.blake2b(joined, digest_size=6).digest(), "big"))


def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
    """Fixed-shape schema digest gathered ONCE before the per-field loop.

    Covers the field-name list (order-sensitive), the count, the x64 flag,
    ``n_dev``, and -- critically -- the per-field DTYPE CLASS and NDIM.

    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
    exact dtypes to a 1-value digest and float dtypes to a 3-moment
    fingerprint, and its struct entry depends on ndim.  If the schema gate

    ``jax.core.trace_state_clean`` was removed from the public ``jax.core`` in
    jax 0.7 and survives only as ``jax._src.core``, so this reads the private
    module.  The ``except`` returns False — i.e. the gate RUNS and the traced
    lane crashes loudly again — deliberately: for a correctness gate a loud
    crash beats a silent skip.  ``tests/unit/test_geometry_consistency_trace_
    gate.py`` asserts this returns True inside ``jax.jit`` AND inside
    ``lax.scan``, so a JAX version that moves the symbol turns CI red first.
    """
    try:
        from jax._src import core as _jax_core
        return not _jax_core.trace_state_clean()
    except Exception:  # pragma: no cover - JAX internal moved; test goes red
        return False


def assert_flags_agree(names, values, *, context: str) -> None:
    """Raise unless every process agrees on a tuple of rank-local CONFIG flags.

    Call this BEFORE any rank-local ``raise`` that inspects per-process
    config.  Otherwise one process can reject its config and exit while its
    peers proceed into a collective and block forever — a collective-ORDER
    violation whose symptom (hang vs backend error) is backend-dependent
    (codex 2026-07-29, blocker 1).

    ``names`` and ``values`` must be STATIC tuples written at the call site,
    so the payload length is fixed by the code path rather than by data.

    No-op under a JAX trace — see :func:`in_jax_trace` (#1405).
    """
    if jax.process_count() <= 1 or in_jax_trace():
        return
    from jax.experimental import multihost_utils

    payload = np.array(
        [float(len(values)), name_digest48(names),
         *(float(v) for v in values)], dtype=np.float64)
    gathered = multihost_utils.process_allgather(payload)
    if not bool(np.all(gathered == gathered[0])):
        raise RuntimeError(
            f"{context}: per-process CONFIG differs across processes "
            f"(flags {list(names)} -> gathered {gathered.tolist()}). Every "
            f"process must be built from the same config; refusing before "
            f"any rank-local rejection so the failure is symmetric rather "
            f"than a hang.")


def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
    """Raise unless every process agrees on the geometry field SCHEMA.

    ``names`` must be an ORDERED sequence — the per-field
    :func:`broadcast_checked` calls that follow are matched positionally
    across processes, so a reordering is itself a divergence worth catching.

    Pass ``arrays`` (the per-name arrays, same order) so the gate also covers
    each field's DTYPE CLASS and NDIM.  Those decide the per-field payload
    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
    bool-vs-float disagreement slip past this gate and deadlock in the
    per-field gather instead of raising here.

    No-op when ``jax.process_count() == 1``.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    names = list(names)
    if arrays is None:
        kinds, ndims = (), ()
    else:
        # Read dtype/ndim from array METADATA, never via np.asarray: a jax
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:49:    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:110:def _lat_spec(x):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:315:    assert_pytree_bytes_equal(state, "shard_state_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:333:        return field.replace(data=addressable_shard_put(field.data, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:351:        return field.replace(data=addressable_shard_put(v_lower, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:373:            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:400:        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:449:        return addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:49:    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:110:def _lat_spec(x):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:315:    assert_pytree_bytes_equal(state, "shard_state_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:333:        return field.replace(data=addressable_shard_put(field.data, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:351:        return field.replace(data=addressable_shard_put(v_lower, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:373:            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:400:        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:449:        return addressable_shard_put(arr, NamedSharding(mesh, spec))

codex
The requested code paths are present. I found a boundary case in the new scalar gate and am validating it against the schema/digest behavior before issuing the verdict.
exec
/usr/bin/bash -lc "PYTHONPATH=packages/core python - <<'PY'
import numpy as np
from legoesm.parallel.geometry_consistency import content_hash48, tree_schema_digest48
for x in (0, 0.0):
    a = np.asarray(x)
    print(type(x).__name__, a.dtype, a.shape, a.tobytes().hex(), content_hash48(a), tree_schema_digest48((x,)))
PY
pytest -q tests/ocean/unit/test_sharded_geom_fingerprint.py tests/unit/test_geometry_consistency.py -q" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 317ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system
/usr/bin/bash: line 7: pytest: command not found

exec
/usr/bin/bash -lc 'PYTHONPATH=packages/core python -c "import numpy as np; from legoesm.parallel.geometry_consistency import content_hash48, tree_schema_digest48; [(lambda a, x: print(type(x).__name__, a.dtype, a.shape, a.tobytes().hex(), content_hash48(a), tree_schema_digest48((x,))))(np.asarray(x), x) for x in (0, 0.0)]"' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 271ms:
/usr/bin/bash: python: command not found

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/geometry_consistency.py | sed -n '258,286p;337,354p;706,729p' && nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '352,376p;388,405p;432,452p' && git diff --check HEAD~1 HEAD 2>&1 | sed -n '1,8p' && git diff --check HEAD~1 HEAD 2>&1 | wc -l" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 411ms:
   258	    try:
   259	        return name_digest48(_canonical_config_terms(obj))
   260	    except Exception:                        # pragma: no cover - defensive
   261	        return FLAG_DIGEST_FAILED
   262	
   263	
   264	def tree_schema_digest48(tree) -> float:
   265	    """Digest of a pytree's LEAF SCHEMA: ordered path, dtype and full shape.
   266	
   267	    The gather/scatter entry points run one cross-process replication PER
   268	    NON-``None`` LEAF, so the NUMBER and ORDER of those collectives is
   269	    rank-local data: a state whose tracer dict differs across processes (extra
   270	    species, different insertion order, different shape) produces mismatched
   271	    schedules and hangs (codex round-4, blocker 5).  Folding the whole leaf
   272	    schema into ONE fixed-width float makes that a clean symmetric raise.
   273	
   274	    ``jax.tree_util`` key paths give a canonical, ORDER-SENSITIVE description
   275	    (dict keys are sorted by ``tree_flatten_with_path``, so an insertion-order
   276	    difference alone does not false-positive, while a KEY-SET difference does
   277	    move the digest).  Never raises.
   278	    """
   279	    try:
   280	        from jax.tree_util import tree_flatten_with_path, keystr
   281	        leaves, _ = tree_flatten_with_path(tree)
   282	        terms = []
   283	        for path, leaf in leaves:
   284	            dtype = getattr(leaf, "dtype", None)
   285	            shape = getattr(leaf, "shape", None)
   286	            terms.append(
   337	# Relative tolerance for FLOAT geometry fields. Only ULP-scale autotune drift
   338	# is expected there; quantize-then-assert-equal false-positived on a rounding
   339	# boundary (job 26453240), so compare with a tolerance instead.
   340	_FLOAT_RTOL = 1e-5
   341	
   342	
   343	def content_hash48(arr) -> float:
   344	    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
   345	
   346	    Used to compare EXACT-dtype arrays (masks, index tables) across
   347	    processes: unlike moment fingerprints, a byte digest is positional, so a
   348	    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
   349	    under 2**53 so it survives the float64 ``process_allgather`` payload
   350	    exactly. Not cryptographic — collision-resistance at 2**-48 is far
   351	    beyond the ~10 setup-time comparisons this guard makes.
   352	    """
   353	    a = np.ascontiguousarray(arr)
   354	    h = hashlib.blake2b(a.tobytes(), digest_size=6)
   706	def assert_pytree_bytes_equal(tree, what):
   707	    """Cheap multi-process replacement for the per-leaf assert_equal that
   708	    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
   709	    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
   710	    symmetric raise on mismatch. No-op single-process.
   711	    """
   712	    if jax.process_count() <= 1:
   713	        return
   714	    from jax.experimental import multihost_utils
   715	
   716	    # Numeric python scalars included (codex r20 item 1): the scatter
   717	    # paths jnp.asarray + put them, so a rank-divergent scalar must not
   718	    # bypass the gate. Non-numeric leaves (None, strings) stay excluded.
   719	    leaves = [x for x in jax.tree_util.tree_leaves(tree)
   720	              if hasattr(x, "ndim") or isinstance(x, (int, float, complex))]
   721	    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
   722	                    dtype=np.float64)
   723	    g = multihost_utils.process_allgather(vals)
   724	    if not bool(np.all(g == g[0])):
   725	        bad = [i for i in range(len(leaves))
   726	               if not bool(np.all(g[:, i] == g[0, i]))]
   727	        raise RuntimeError(
   728	            f"{what}: array leaves {bad} differ across processes (48-bit "
   729	            f"byte digests disagree) — the per-process inputs are NOT "
   352	
   353	    updates = {}
   354	    for name in _V_STAGGERED_STATE_FIELDS:
   355	        updates[name] = _shard_v(getattr(state, name))
   356	    # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
   357	    # v-staggered set and not None get the cell sharding; None stays None.
   358	    for name in state._fields:
   359	        if name in _V_STAGGERED_STATE_FIELDS:
   360	            continue
   361	        val = getattr(state, name)
   362	        if val is None:
   363	            updates[name] = None
   364	        elif hasattr(val, "data"):                 # a Field
   365	            updates[name] = _shard_cell(val)
   366	        else:
   367	            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
   368	            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
   369	            arr = jnp.asarray(val)
   370	            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
   371	            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
   372	            # scalars replicate.
   373	            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
   374	    return state._replace(**updates)
   375	
   376	
   388	    # FIRST statement: the `forcing is None or mesh is None` return below
   389	    # SKIPS every per-leaf put, so a rank with no forcing would leave a peer
   390	    # blocked in one (#1362 round 4, blocker 6).
   391	    _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
   392	    if forcing is None or mesh is None:
   393	        return forcing
   394	    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
   395	
   396	    def _put(leaf):
   397	        if leaf is None:
   398	            return None
   399	        arr = jnp.asarray(leaf)
   400	        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
   401	
   402	    return jax.tree.map(_put, forcing)
   403	
   404	
   405	def shard_forcing_stack_latlon(stack, mesh):
   432	    _agree_ocean_mesh_entry(mesh, stack,
   433	                            where="shard_forcing_stack_latlon")
   434	    if mesh is None:
   435	        return stack
   436	
   437	    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
   438	
   439	    def _put(leaf):
   440	        if leaf is None or not hasattr(leaf, "ndim"):
   441	            return leaf
   442	        arr = jnp.asarray(leaf)
   443	        if arr.ndim >= 3:
   444	            spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
   445	        elif arr.ndim == 2:
   446	            spec = P("lat", None)
   447	        else:
   448	            spec = P()
   449	        return addressable_shard_put(arr, NamedSharding(mesh, spec))
   450	
   451	    return jax.tree.map(_put, stack)
   452	
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:27: trailing whitespace.
+    
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1137: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1139: trailing whitespace.
+ 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1166: trailing whitespace.
+ 
560

exec
/usr/bin/bash -lc "rg -n -i -C 2 \"167|2-proc|15\\.11|has_aux|0-d|python scalar|item 3|unverified|>=96\" .physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md .physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md docs/performance/scaling/levante_campaign_2026-07-24.md | tail -n 220" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 279ms:
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1133-)
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1134-
--
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1347-./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1348-./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1349:./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:611:    "has_aux", "aux_schema",
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1350-./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:615:def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str,
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1351-./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:890:        _agree_ocean_spmd_call(
--
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1362-./packages/core/legoesm/parallel/geometry_consistency.py:706:def assert_pytree_bytes_equal(tree, what):
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1363-./packages/core/legoesm/parallel/geometry_consistency.py:737:    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1364:./tests/ocean/unit/test_isoneutral_slope_density.py:514:        (total, K33), grad = jax.value_and_grad(k33_sum, has_aux=True)(T)
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1365-./tests/ocean/unit/test_sharded_geom_fingerprint.py:13:    band_fingerprint as geom_band_fingerprint,
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1366-./tests/ocean/unit/test_sharded_geom_fingerprint.py:14:    band_fingerprints_agree,
--
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1375-./tests/ocean/unit/test_sharded_geom_fingerprint.py:96:    fb = geom_band_fingerprint(b, N_BANDS)
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1376-./tests/ocean/unit/test_sharded_geom_fingerprint.py:104:        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1377:./packages/ml/legoesm/training/neural_gcm_spectral.py:1395:        scalar suitable for ``has_aux=True`` accumulation.
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1378:./packages/ml/legoesm/training/neural_gcm_spectral.py:1590:    directly and use ``has_aux=True`` in
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1379:./packages/ml/legoesm/training/neural_gcm_spectral.py:2702:                loss_fn, has_aux=True,
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1380:./packages/ml/legoesm/training/neural_gcm_spectral.py:2729:                loss_fn, has_aux=True,
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1381:./packages/ml/legoesm/training/neural_gcm_spectral.py:3838:                loss_fn, has_aux=True,
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1382:./scripts/run/train_land_params_era5.py:174:    vg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1383:./scripts/run/run_aimip_amip_finetune.py:429:                lambda p: _sample_loss(p, sample, _n), has_aux=True,
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1384-./tests/unit/test_geometry_consistency.py:1019:         "_agree_ocean_spmd_call"),
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1385-./tests/unit/test_geometry_consistency.py:1021:         "_agree_ocean_spmd_call"),
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1386:./scripts/run/train_multilayer_land_era5.py:657:    vg = jax.jit(jax.value_and_grad(loss_ml, has_aux=True))
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:1387:./tests/unit/test_diff_atmosphere_convection_micro.py:522:        value_and_grad = jax.jit(jax.value_and_grad(mask_sum, has_aux=True))
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1388-
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-1389-exec
--
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2217-
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2218-    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:2219:    both without materialising, so no device sync).  A plain Python scalar or
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2220-    list has neither; falling through to ``np.asarray`` there is FREE (it is
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2221-    already host data) and, critically, keeps this function from dying with a
--
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2474-   714	    from jax.experimental import multihost_utils
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2475-   715	
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:2476:   716	    # Numeric python scalars included (codex r20 item 1): the scatter
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2477-   717	    # paths jnp.asarray + put them, so a rank-divergent scalar must not
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2478-   718	    # bypass the gate. Non-numeric leaves (None, strings) stay excluded.
--
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2563-
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md-2564-exec
.physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md:2565:/usr/bin/bash -lc "rg -n -i -C 2 \"167|2-proc|15\\.11|has_aux|0-d|python scalar|item 3|unverified|>=96\" .physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md .physics-validator/scaling_campaign/codex_merge_r21_2026-08-03.md docs/performance/scaling/levante_campaign_2026-07-24.md | tail -n 220" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-12---------
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-13-user
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:14:Round-20: review the merge-port (git show e13de3afe --stat; git diff e13de3afe^1 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py). Context: origin/main gained #1362 (shared geometry_consistency module + entry gates; its broadcast_checked ends in broadcast_one_to_all and its ocean _replicated_put re-adds device_put — centralizing walls 1+2). The port: (a) NEW shared fns in geometry_consistency: band_fingerprint (per-band, ValueError on wrong leading axis, schema-gate prerequisite documented), band_fingerprints_agree (rtol defaults _FLOAT_RTOL), checked_shard_put (per-band gate + make_array_from_callback; single-process plain device_put), assert_pytree_bytes_equal (per-leaf 48-bit digests), addressable_shard_put (ungated, caller-gated inputs); wall history comment block. (b) ocean: _replicated_put now calls checked_shard_put(n_bands=n_dev) AFTER their assert_schema_agrees(arrays=...) gate; 5 state/forcing puts -> addressable_shard_put with assert_pytree_bytes_equal gates; aux threading on sharded_step + step.aux; _global shards forcing explicitly; their entry gates + schema-arrays gate kept verbatim. (c) my unit test retargeted to the shared module (ValueError test). Receipts: 181 gate/parity + 48 run_omip tests + 2-proc multicontroller repro (15.39 ms) green post-merge.
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-15-Attack: (1) does anything still call broadcast_checked with a BAND STACK (grep callers — atm lane #1362 sites!) — if the atm lane broadcasts global-sized stacks it carries wall 1 at high nd; is that a this-PR fix or a flagged follow-up? (2) checked_shard_put's n_bands prerequisite — is the ocean call ordering (schema gate before puts) actually enforced in code order? (3) duplicate/conflicting fingerprint machinery left over (their fixed-width whole-array vs my per-band — both used where?); (4) import cycles (core/parallel importing jax fine; ocean importing core ok per layering?); (5) anything from their entry gates broken by my aux param (\_agree_ocean_spmd_call signature vs new aux arg?). End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-16-codex
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-689-+
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-690-+    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:691:+    both without materialising, so no device sync).  A plain Python scalar or
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-692-+    list has neither; falling through to ``np.asarray`` there is FREE (it is
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-693-+    already host data) and, critically, keeps this function from dying with a
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-899-+        # the single real materialisation later.
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-900-+        #
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:901:+        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-902-+        # bare `a.dtype` read did not (codex round-3, minor 3): no production
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-903-+        # caller passes one today, but an AttributeError HERE would be a
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1657--        # assert (job 26450848). With the #1370-iii P("lat") sharding each
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1658--        # process's devices consume ONLY its own band rows, so the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1659:-        # broadcast became both unnecessary and, at nd>=96, fatal (its
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1660--        # psum program is nd x the stack — see the note at the put below).
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1661--        # GUARD (codex round-3): process 0 must not silently mask REAL
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1778-     def sharded_step(state, dt, freshwater=None, surface_forcing=None,
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1779-                      sponge=None, t_seconds=None, aux=None):
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1780:-        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1781--        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1782--        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1820-     def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1821--        # Scatter the global state AND forcing to the band layout explicitly
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:1822:-        # (codex r17 item 3: the old comment claimed inner sharded the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1823--        # forcing; it forwarded it global and relied on implicit JIT input
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-1824--        # placement — which under multicontroller pays jax's whole-array
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2738-+
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2739-+    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:2740:+    both without materialising, so no device sync).  A plain Python scalar or
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2741-+    list has neither; falling through to ``np.asarray`` there is FREE (it is
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2742-+    already host data) and, critically, keeps this function from dying with a
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2948-+        # the single real materialisation later.
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2949-+        #
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:2950:+        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2951-+        # bare `a.dtype` read did not (codex round-3, minor 3): no production
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-2952-+        # caller passes one today, but an AttributeError HERE would be a
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-3273- # pure slicers import without requiring mpi4py.  See omip-multinode-spmd-scope.
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-3274- 
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:3275: # The LatLonCGridGeometry fields that MUST stay static python scalars (they gate
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-3276- # trace-time ``if``-s + ``jnp.zeros`` shape builds): never stacked/indexed.  The
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-3277- # ``fold`` is a FoldDescriptor whose ``is_active``/``fold_j``/``cap_j`` are python
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4059--        # assert (job 26450848). With the #1370-iii P("lat") sharding each
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4060--        # process's devices consume ONLY its own band rows, so the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4061:-        # broadcast became both unnecessary and, at nd>=96, fatal (its
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4062--        # psum program is nd x the stack — see the note at the put below).
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4063--        # GUARD (codex round-3): process 0 must not silently mask REAL
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4288-     def sharded_step(state, dt, freshwater=None, surface_forcing=None,
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4289-                      sponge=None, t_seconds=None, aux=None):
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4290:-        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4291--        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4292--        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4419-     def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4420--        # Scatter the global state AND forcing to the band layout explicitly
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:4421:-        # (codex r17 item 3: the old comment claimed inner sharded the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4422--        # forcing; it forwarded it global and relied on implicit JIT input
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-4423--        # placement — which under multicontroller pays jax's whole-array
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-5867-   697	        # assert (job 26450848). With the #1370-iii P("lat") sharding each
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-5868-   698	        # process's devices consume ONLY its own band rows, so the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:5869:   699	        # broadcast became both unnecessary and, at nd>=96, fatal (its
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-5870-   700	        # psum program is nd x the stack — see the note at the put below).
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-5871-   701	        # GUARD (codex round-3): process 0 must not silently mask REAL
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7583-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-165-    The #1100 invariant for multi-process runs: **neither global builds nor
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7584-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-166-    ``device_put`` replication** — every global-shaped leaf is created with
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7585:packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:167:    ``jax.make_array_from_callback``, whose callback is invoked only for the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7586-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-168-    row slices owned by THIS process's addressable devices (documented JAX
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7587-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-169-    semantics: per-addressable-shard callbacks with GLOBAL index slices).  No
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7830---
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7831-tests/unit/test_land_ml_checkpoint_roundtrip.py-29-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7832:tests/unit/test_land_ml_checkpoint_roundtrip.py-30-def test_land_ml_survives_carry_aux_npz_roundtrip(tmp_path):
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7833-tests/unit/test_land_ml_checkpoint_roundtrip.py-31-    saved = _state(1.0)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7834-tests/unit/test_land_ml_checkpoint_roundtrip.py-32-    src = SimpleNamespace(
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7928-tests/ocean/unit/test_isoneutral_slope_density.py-512-            return jnp.sum(K33), K33
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7929-tests/ocean/unit/test_isoneutral_slope_density.py-513-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7930:tests/ocean/unit/test_isoneutral_slope_density.py:514:        (total, K33), grad = jax.value_and_grad(k33_sum, has_aux=True)(T)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7931-tests/ocean/unit/test_isoneutral_slope_density.py-515-        assert jnp.all(jnp.isfinite(K33))
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7932-tests/ocean/unit/test_isoneutral_slope_density.py-516-        assert float(jnp.min(K33)) >= 0.0
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7979-tests/unit/test_double_moment_checkpoint.py-55-        assert f"dmtr_{k}" in aux, k
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7980---
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:7981:tests/unit/test_double_moment_checkpoint.py-60-def test_warm_rain_checkpoint_carry_aux_is_unchanged():
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7982-tests/unit/test_double_moment_checkpoint.py-61-    aux_in = {"held_dT_rad": jnp.zeros((4,))}
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-7983-tests/unit/test_double_moment_checkpoint.py-62-    src = _Stub({"q_v": jnp.ones((4,)), "q_c": jnp.zeros((4,)),
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8274-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1547-    grid = model.grid
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8275---
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:8276:packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1679-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8277-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1680-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8278-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1681-            "follow-up.)  Use p_lon == 1 or disable the filter.")
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8484-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1547-    grid = model.grid
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8485---
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:8486:packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1679-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8487-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1680-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8488-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1681-            "follow-up.)  Use p_lon == 1 or disable the filter.")
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8787-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1018-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8788-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1019-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:8789:packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1020-def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8790-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1021-                                    physics_fn=None, *,
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8791-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1022:                                    shard_geometry: bool = True):
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8905-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1623-            **{name: stacks_local[name][gi, gj]
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8906---
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:8907:packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1679-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8908-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1680-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-8909-packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1681-            "follow-up.)  Use p_lon == 1 or disable the filter.")
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9236-  1065	    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9237-  1066	    first stepS (``p_s`` first, ``u/v/T`` next via the promoted ``p_s`` —
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9238:  1067	    observed jobs 8916406/8916740) exactly as the per-step Python loop
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9239-  1068	    absorbs silently; an already-stable state unrolls NOTHING and scans all
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9240-  1069	    ``n_steps``.  Zero extra casts, zero numerical difference vs the
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9954-tests/ocean/unit/test_tidal_forcing.py-338-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9955-tests/ocean/unit/test_tidal_forcing.py-339-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:9956:tests/ocean/unit/test_tidal_forcing.py-340-def test_apply_no_time_is_byte_identical_noop():
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9957-tests/ocean/unit/test_tidal_forcing.py:341:    """t_seconds=None cannot evaluate an equilibrium tide; the wrapper returns
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-9958-tests/ocean/unit/test_tidal_forcing.py-342-    the inputs unchanged rather than silently substituting a frozen t=0 tide.
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10329-packages/core/legoesm/core/field.py-165-# Register Field as a JAX pytree
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10330-packages/core/legoesm/core/field.py:166:jax.tree_util.register_pytree_node(
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10331:packages/core/legoesm/core/field.py-167-    Field,
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10332-packages/core/legoesm/core/field.py:168:    lambda f: f.tree_flatten(),
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10333-packages/core/legoesm/core/field.py:169:    lambda aux, children: Field.tree_unflatten(aux, children),
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10399-packages/core/legoesm/core/field.py-165-# Register Field as a JAX pytree
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10400-packages/core/legoesm/core/field.py:166:jax.tree_util.register_pytree_node(
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10401:packages/core/legoesm/core/field.py-167-    Field,
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10402-packages/core/legoesm/core/field.py:168:    lambda f: f.tree_flatten(),
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10403-packages/core/legoesm/core/field.py:169:    lambda aux, children: Field.tree_unflatten(aux, children),
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10706-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10707-codex
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10708:1. `assert_pytree_bytes_equal` skips Python scalar leaves, but the scatter paths explicitly replicate scalars via `jnp.asarray` + `addressable_shard_put`. Rank-divergent scalar inputs therefore bypass the replacement for `device_put`’s equality assert. Include numeric scalars in the digest gate. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:710) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:396)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10709-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10710:2. `aux` is a new per-call input but is absent from `_agree_ocean_spmd_call`; a rank-local `None`/provided or schema mismatch can get past the entry gate before the JIT call. Add `has_aux` and `tree_schema_digest48(aux)` to the fixed payload and pass `aux` to the gate. [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:608) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:872)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10711-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10712-3. Atmosphere still sends band/tile stacks through `broadcast_checked`, which ends in `broadcast_one_to_all`; both the 1-D segment default and 2-D default use sharded geometry. This preserves wall 1 for high process counts. It is outside the ocean file port, but cannot be described as centrally eliminating wall 1. Migrate the sharded-geometry paths to an appropriate checked callback put, or explicitly track it as a scaling follow-up. [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492) [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10713-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10714:4. Minor API/doc gaps: `band_fingerprint` raises `IndexError`, not the promised `ValueError`, for a 0-D input; and the four new shared helpers are missing from `__all__`. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:23) [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:633)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10715-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10716-The ocean schema gate is correctly before every `checked_shard_put`; it gates `n_dev`, dtype class, and rank before the per-field collectives. No ocean-local duplicate fingerprint code remains: whole-array fixed-width fingerprints serve broadcast callers (atmosphere/coupler), while per-band fingerprints serve the assert-free ocean put. The import direction is valid: core imports only JAX/NumPy, and ocean depends on core.
--
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10721-tokens used
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10722-183,212
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10723:1. `assert_pytree_bytes_equal` skips Python scalar leaves, but the scatter paths explicitly replicate scalars via `jnp.asarray` + `addressable_shard_put`. Rank-divergent scalar inputs therefore bypass the replacement for `device_put`’s equality assert. Include numeric scalars in the digest gate. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:710) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:396)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10724-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10725:2. `aux` is a new per-call input but is absent from `_agree_ocean_spmd_call`; a rank-local `None`/provided or schema mismatch can get past the entry gate before the JIT call. Add `has_aux` and `tree_schema_digest48(aux)` to the fixed payload and pass `aux` to the gate. [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:608) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:872)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10726-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10727-3. Atmosphere still sends band/tile stacks through `broadcast_checked`, which ends in `broadcast_one_to_all`; both the 1-D segment default and 2-D default use sharded geometry. This preserves wall 1 for high process counts. It is outside the ocean file port, but cannot be described as centrally eliminating wall 1. Migrate the sharded-geometry paths to an appropriate checked callback put, or explicitly track it as a scaling follow-up. [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492) [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10728-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md:10729:4. Minor API/doc gaps: `band_fingerprint` raises `IndexError`, not the promised `ValueError`, for a 0-D input; and the four new shared helpers are missing from `__all__`. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:23) [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:633)
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10730-
.physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md-10731-The ocean schema gate is correctly before every `checked_shard_put`; it gates `n_dev`, dtype class, and rank before the per-field collectives. No ocean-local duplicate fingerprint code remains: whole-array fixed-width fingerprints serve broadcast callers (atmosphere/coupler), while per-band fingerprints serve the assert-free ocean put. The import direction is valid: core imports only JAX/NumPy, and ocean depends on core.

exec
/usr/bin/bash -lc "nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '2086,2112p' && nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '604,648p;886,895p;996,1002p' && nl -ba packages/core/legoesm/parallel/geometry_consistency.py | sed -n '44,57p;632,644p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 269ms:
  2086	  **eff 0.84** — a healthy-tile strong leg on the ocean lane.
  2087	* This closes the user directive's ocean-hundreds gap: both lat-lon
  2088	  lanes (atm + ocean) now hold receipts at 96-128 GPUs, MPAS at 128.
  2089	
  2090	### Merge-port of #1362 (geometry_consistency) — shared-module fixes + one tracked follow-up
  2091	
  2092	The three multicontroller fixes now live in
  2093	`legoesm.parallel.geometry_consistency` (checked_shard_put /
  2094	addressable_shard_put / assert_pytree_bytes_equal / band_fingerprint —
  2095	the ONE implementation, #1362 doctrine); the ocean lane calls them, and
  2096	#1362's entry gates gained aux coverage + numeric-scalar leaves in the
  2097	digest gate (codex r20). 167 gate/parity tests + the 2-proc repro
  2098	(15.11 ms) green post-fix.
  2099	
  2100	**TRACKED FOLLOW-UP (codex r20 item 3): the ATMOSPHERE lat-lon lane
  2101	still routes its band/tile geometry stacks through `broadcast_checked`
  2102	-> `broadcast_one_to_all` (sharded_atm_latlon_step.py:492/1586) — wall
  2103	1 preserved there** (an [n_processes, stack] psum program). The atm
  2104	128-GPU receipts predate #1362, so the current main atm lane at >=96
  2105	processes is UNVERIFIED and plausibly walled exactly as ocean was.
  2106	Port = same checked_shard_put swap + aux threading; needs its own
  2107	parity run + a 2-proc repro before any atm hundreds rerun on merged
  2108	main.
   604	
   605	
   606	# Ordered flag names for the PER-INVOCATION gate on the returned ocean SPMD
   607	# callable. STATIC tuple: fixed width, never rank-local.
   608	_OCEAN_CALL_ENTRY_FLAGS = (
   609	    "has_mesh", "n_dev", "axis_names", "axis_sizes",
   610	    "state_schema", "has_forcing", "forcing_schema",
   611	    "has_aux", "aux_schema",
   612	)
   613	
   614	
   615	def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str,
   616	                           aux=None) -> None:
   617	    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
   618	
   619	    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
   620	    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
   621	    forcing layout that is invalid on one rank only makes that rank raise while
   622	    its peers enter the collective program -- a hang.  The state + forcing leaf
   623	    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
   624	    it, so two processes with different optional fields compile different
   625	    programs.
   626	
   627	    Cost: one small allgather per CALL, and an exact no-op under a single
   628	    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
   629	    on cache misses is NOT a valid optimisation.
   630	    """
   631	    names, sizes = _ocean_mesh_axis_terms(mesh)
   632	    assert_flags_agree(_OCEAN_CALL_ENTRY_FLAGS, (
   633	        float(mesh is not None),
   634	        float(mesh.devices.size if mesh is not None else 0),
   635	        name_digest48(names),
   636	        name_digest48(sizes),
   637	        tree_schema_digest48(state),
   638	        float(forcing is not None),
   639	        (tree_schema_digest48(forcing) if forcing is not None
   640	         else FLAG_ABSENT),
   641	        # aux (codex r20 item 2): a rank-local None-vs-provided or schema
   642	        # mismatch must fail HERE, not desynchronize the jit call below.
   643	        float(aux is not None),
   644	        (tree_schema_digest48(aux) if aux is not None else FLAG_ABSENT),
   645	    ), context=where)
   646	
   647	
   648	def make_sharded_ocean_step(model, mesh):
   886	        # jit boundary as an ARGUMENT and pass it back here.
   887	        # ONE forcing operand: None fields drop out of the pytree structure,
   888	        # so specs derived by tree.map skip them automatically and the
   889	        # structure key below distinguishes every None<->array combination.
   890	        _agree_ocean_spmd_call(
   891	            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
   892	            where="make_sharded_ocean_step.step", aux=aux)
   893	        forcing = (freshwater, surface_forcing, sponge, t_seconds)
   894	        _validate_forcing_layout((freshwater, surface_forcing, sponge))
   895	        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
   996	    inner = make_sharded_ocean_step(model, mesh)
   997	
   998	    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
   999	        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
  1000	                               where="make_sharded_ocean_step_global.step")
  1001	        # Scatter the global state AND forcing to the band layout
  1002	        # explicitly (the old comment claimed inner sharded the forcing;
    44	import numpy as np
    45	
    46	__all__ = [
    47	    "addressable_shard_put",
    48	    "assert_pytree_bytes_equal",
    49	    "band_fingerprint",
    50	    "band_fingerprints_agree",
    51	    "checked_shard_put",
    52	    "content_hash48",
    53	    "name_digest48",
    54	    "schema_fingerprint",
    55	    "assert_schema_agrees",
    56	    "assert_flags_agree",
    57	    "broadcast_checked",
   632	    band-local drift must not hide in a whole-array sum (codex r14).
   633	    DOCUMENTED RESIDUALS: a within-band float change preserving all three
   634	    moments to rtol, and non-finite entries changing position/kind at a
   635	    fixed per-band count, pass the float gate.
   636	    """
   637	    host = np.asarray(host)
   638	    if host.ndim == 0 or host.shape[0] != n_bands:
   639	        raise ValueError(
   640	            f"band_fingerprint: leading axis "
   641	            f"{host.shape[0] if host.ndim else '<0-d>'} != n_bands "
   642	            f"{n_bands}")
   643	    is_exact = host.dtype.kind in "biu"
   644	    struct = [float(host.ndim), *map(float, host.shape),

codex
The requested aux, 0-D, exports, and ATM follow-up changes are present. The campaign entry records the stated 167 tests and 2-proc repro.

1. The new scalar gate still misses a real cross-rank type divergence: Python `0` and `0.0` have no `.dtype`/`.shape` for the schema digest, and on 64-bit hosts both serialize to eight zero bytes. [`content_hash48`](file...) hashes only those bytes, so [`assert_pytree_bytes_equal`](...) accepts them although `jnp.asarray` creates distinct integer/float inputs. Hash scalar dtype/shape too (or canonicalize the schema) and add a 2-proc regression.

2. Non-blocking hygiene: the committed 10,735-line campaign transcript introduces 560 `git diff --check` trailing-whitespace errors.

VERDICT: FIX-FIRST
tokens used
73,277
The requested aux, 0-D, exports, and ATM follow-up changes are present. The campaign entry records the stated 167 tests and 2-proc repro.

1. The new scalar gate still misses a real cross-rank type divergence: Python `0` and `0.0` have no `.dtype`/`.shape` for the schema digest, and on 64-bit hosts both serialize to eight zero bytes. [`content_hash48`](file...) hashes only those bytes, so [`assert_pytree_bytes_equal`](...) accepts them although `jnp.asarray` creates distinct integer/float inputs. Hash scalar dtype/shape too (or canonicalize the schema) and add a 2-proc regression.

2. Non-blocking hygiene: the committed 10,735-line campaign transcript introduces 560 `git diff --check` trailing-whitespace errors.

VERDICT: FIX-FIRST
