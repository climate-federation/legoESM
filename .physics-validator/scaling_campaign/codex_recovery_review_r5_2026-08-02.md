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
session id: 019fc238-56a1-75b1-b93f-8220e8fecfaf
--------
user
Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
exec
/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 2031ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
./uv.lock-2170-sdist = { url = "https://files.pythonhosted.org/packages/0e/4a/c27b42ed9b1c7d13d9ba8b6905dece787d6259152f2309338aed29b2447b/ml_dtypes-0.5.4.tar.gz", hash = "sha256:8ab06a50fb9bf9666dd0fe5dfb4676fa2b0ac0f31ecff72a6c3af8e22c063453", size = 692314, upload-time = "2025-11-17T22:32:31.031Z" }
./uv.lock-2171-wheels = [
./uv.lock-2172-    { url = "https://files.pythonhosted.org/packages/c6/5e/712092cfe7e5eb667b8ad9ca7c54442f21ed7ca8979745f1000e24cf8737/ml_dtypes-0.5.4-cp311-cp311-macosx_10_9_universal2.whl", hash = "sha256:6c7ecb74c4bd71db68a6bea1edf8da8c34f3d9fe218f038814fd1d310ac76c90", size = 679734, upload-time = "2025-11-17T22:31:39.223Z" },
./uv.lock:2173:    { url = "https://files.pythonhosted.org/packages/4f/cf/912146dfd4b5c0eea956836c01dcd2fce6c9c844b2691f5152aca196ce4f/ml_dtypes-0.5.4-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl", hash = "sha256:bc11d7e8c44a65115d05e2ab9989d1e045125d7be8e05a071a48bc76eb6d6040", size = 5056165, upload-time = "2025-11-17T22:31:41.071Z" },
./uv.lock-2174-    { url = "https://files.pythonhosted.org/packages/a9/80/19189ea605017473660e43762dc853d2797984b3c7bf30ce656099add30c/ml_dtypes-0.5.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", hash = "sha256:19b9a53598f21e453ea2fbda8aa783c20faff8e1eeb0d7ab899309a0053f1483", size = 5034975, upload-time = "2025-11-17T22:31:42.758Z" },
./uv.lock-2175-    { url = "https://files.pythonhosted.org/packages/b4/24/70bd59276883fdd91600ca20040b41efd4902a923283c4d6edcb1de128d2/ml_dtypes-0.5.4-cp311-cp311-win_amd64.whl", hash = "sha256:7c23c54a00ae43edf48d44066a7ec31e05fdc2eee0be2b8b50dd1903a1db94bb", size = 210742, upload-time = "2025-11-17T22:31:44.068Z" },
./uv.lock-2176-    { url = "https://files.pythonhosted.org/packages/a0/c9/64230ef14e40aa3f1cb254ef623bf812735e6bec7772848d19131111ac0d/ml_dtypes-0.5.4-cp311-cp311-win_arm64.whl", hash = "sha256:557a31a390b7e9439056644cb80ed0735a6e3e3bb09d67fd5687e4b04238d1de", size = 160709, upload-time = "2025-11-17T22:31:46.557Z" },
--
./uv.lock-2660-    { url = "https://files.pythonhosted.org/packages/c4/d3/b7da1d5d7dbdc5ef52ed7debd2b484313b832982266905315dad5a0bf0b1/pandas-3.0.2-cp311-cp311-macosx_11_0_arm64.whl", hash = "sha256:dbbd4aa20ca51e63b53bbde6a0fa4254b1aaabb74d2f542df7a7959feb1d760c", size = 9926987, upload-time = "2026-03-31T06:46:11.724Z" },
./uv.lock-2661-    { url = "https://files.pythonhosted.org/packages/52/77/9b1c2d6070b5dbe239a7bc889e21bfa58720793fb902d1e070695d87c6d0/pandas-3.0.2-cp311-cp311-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl", hash = "sha256:339dda302bd8369dedeae979cb750e484d549b563c3f54f3922cb8ff4978c5eb", size = 10757067, upload-time = "2026-03-31T06:46:14.903Z" },
./uv.lock-2662-    { url = "https://files.pythonhosted.org/packages/20/17/ec40d981705654853726e7ac9aea9ddbb4a5d9cf54d8472222f4f3de06c2/pandas-3.0.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl", hash = "sha256:61c2fd96d72b983a9891b2598f286befd4ad262161a609c92dc1652544b46b76", size = 11258787, upload-time = "2026-03-31T06:46:17.683Z" },
./uv.lock:2663:    { url = "https://files.pythonhosted.org/packages/90/e3/3f1126d43d3702ca8773871a81c9f15122a1f412342cc56284ffda5b1f70/pandas-3.0.2-cp311-cp311-musllinux_1_2_aarch64.whl", hash = "sha256:c934008c733b8bbea273ea308b73b3156f0181e5b72960790b09c18a2794fe1e", size = 11771616, upload-time = "2026-03-31T06:46:20.532Z" },
./uv.lock-2664-    { url = "https://files.pythonhosted.org/packages/2e/cf/0f4e268e1f5062e44a6bda9f925806721cd4c95c2b808a4c82ebe914f96b/pandas-3.0.2-cp311-cp311-musllinux_1_2_x86_64.whl", hash = "sha256:60a80bb4feacbef5e1447a3f82c33209c8b7e07f28d805cfd1fb951e5cb443aa", size = 12337623, upload-time = "2026-03-31T06:46:23.754Z" },
./uv.lock-2665-    { url = "https://files.pythonhosted.org/packages/44/a0/97a6339859d4acb2536efb24feb6708e82f7d33b2ed7e036f2983fcced82/pandas-3.0.2-cp311-cp311-win_amd64.whl", hash = "sha256:ed72cb3f45190874eb579c64fa92d9df74e98fd63e2be7f62bce5ace0ade61df", size = 9897372, upload-time = "2026-03-31T06:46:26.703Z" },
./uv.lock-2666-    { url = "https://files.pythonhosted.org/packages/8f/eb/781516b808a99ddf288143cec46b342b3016c3414d137da1fdc3290d8860/pandas-3.0.2-cp311-cp311-win_arm64.whl", hash = "sha256:f12b1a9e332c01e09510586f8ca9b108fd631fd656af82e452d7315ef6df5f9f", size = 9154922, upload-time = "2026-03-31T06:46:30.284Z" },
--
./uv.lock-2779-    { url = "https://files.pythonhosted.org/packages/5e/26/d325f9f56c7e039034897e7380e9cc202b1e368bfd04d4cbe6a441f02885/pillow-12.2.0-cp314-cp314-musllinux_1_2_aarch64.whl", hash = "sha256:9aba9a17b623ef750a4d11b742cbafffeb48a869821252b30ee21b5e91392c50", size = 6507628, upload-time = "2026-04-01T14:45:12.378Z" },
./uv.lock-2780-    { url = "https://files.pythonhosted.org/packages/5f/f7/769d5632ffb0988f1c5e7660b3e731e30f7f8ec4318e94d0a5d674eb65a4/pillow-12.2.0-cp314-cp314-musllinux_1_2_x86_64.whl", hash = "sha256:deede7c263feb25dba4e82ea23058a235dcc2fe1f6021025dc71f2b618e26104", size = 7209321, upload-time = "2026-04-01T14:45:15.122Z" },
./uv.lock-2781-    { url = "https://files.pythonhosted.org/packages/6a/7a/c253e3c645cd47f1aceea6a8bacdba9991bf45bb7dfe927f7c893e89c93c/pillow-12.2.0-cp314-cp314-win32.whl", hash = "sha256:632ff19b2778e43162304d50da0181ce24ac5bb8180122cbe1bf4673428328c7", size = 6479723, upload-time = "2026-04-01T14:45:17.797Z" },
./uv.lock:2782:    { url = "https://files.pythonhosted.org/packages/cd/8b/601e6566b957ca50e28725cb6c355c59c2c8609751efbecd980db44e0349/pillow-12.2.0-cp314-cp314-win_amd64.whl", hash = "sha256:4e6c62e9d237e9b65fac06857d511e90d8461a32adcc1b9065ea0c0fa3a28150", size = 7217400, upload-time = "2026-04-01T14:45:20.529Z" },
./uv.lock-2783-    { url = "https://files.pythonhosted.org/packages/d6/94/220e46c73065c3e2951bb91c11a1fb636c8c9ad427ac3ce7d7f3359b9b2f/pillow-12.2.0-cp314-cp314-win_arm64.whl", hash = "sha256:b1c1fbd8a5a1af3412a0810d060a78b5136ec0836c8a4ef9aa11807f2a22f4e1", size = 2554835, upload-time = "2026-04-01T14:45:23.162Z" },
./uv.lock-2784-    { url = "https://files.pythonhosted.org/packages/b6/ab/1b426a3974cb0e7da5c29ccff4807871d48110933a57207b5a676cccc155/pillow-12.2.0-cp314-cp314t-macosx_10_15_x86_64.whl", hash = "sha256:57850958fe9c751670e49b2cecf6294acc99e562531f4bd317fa5ddee2068463", size = 5314225, upload-time = "2026-04-01T14:45:25.637Z" },
./uv.lock-2785-    { url = "https://files.pythonhosted.org/packages/19/1e/dce46f371be2438eecfee2a1960ee2a243bbe5e961890146d2dee1ff0f12/pillow-12.2.0-cp314-cp314t-macosx_11_0_arm64.whl", hash = "sha256:d5d38f1411c0ed9f97bcb49b7bd59b6b7c314e0e27420e34d99d844b9ce3b6f3", size = 4698541, upload-time = "2026-04-01T14:45:28.355Z" },
--
./uv.lock-2864-    { url = "https://files.pythonhosted.org/packages/9e/f8/91c27b22ccda1dbc7967f921c42825564fa5336a01ecd72eb78a9f4f53c2/propcache-0.4.1-cp311-cp311-musllinux_1_2_armv7l.whl", hash = "sha256:67fad6162281e80e882fb3ec355398cf72864a54069d060321f6cd0ade95fe85", size = 202064, upload-time = "2025-10-08T19:46:36.993Z" },
./uv.lock-2865-    { url = "https://files.pythonhosted.org/packages/f2/26/7f00bd6bd1adba5aafe5f4a66390f243acab58eab24ff1a08bebb2ef9d40/propcache-0.4.1-cp311-cp311-musllinux_1_2_ppc64le.whl", hash = "sha256:f10207adf04d08bec185bae14d9606a1444715bc99180f9331c9c02093e1959e", size = 212429, upload-time = "2025-10-08T19:46:38.398Z" },
./uv.lock-2866-    { url = "https://files.pythonhosted.org/packages/84/89/fd108ba7815c1117ddca79c228f3f8a15fc82a73bca8b142eb5de13b2785/propcache-0.4.1-cp311-cp311-musllinux_1_2_s390x.whl", hash = "sha256:e9b0d8d0845bbc4cfcdcbcdbf5086886bc8157aa963c31c777ceff7846c77757", size = 216727, upload-time = "2025-10-08T19:46:39.732Z" },
./uv.lock:2867:    { url = "https://files.pythonhosted.org/packages/79/37/3ec3f7e3173e73f1d600495d8b545b53802cbf35506e5732dd8578db3724/propcache-0.4.1-cp311-cp311-musllinux_1_2_x86_64.whl", hash = "sha256:981333cb2f4c1896a12f4ab92a9cc8f09ea664e9b7dbdc4eff74627af3a11c0f", size = 205097, upload-time = "2025-10-08T19:46:41.025Z" },
./uv.lock-2868-    { url = "https://files.pythonhosted.org/packages/61/b0/b2631c19793f869d35f47d5a3a56fb19e9160d3c119f15ac7344fc3ccae7/propcache-0.4.1-cp311-cp311-win32.whl", hash = "sha256:f1d2f90aeec838a52f1c1a32fe9a619fefd5e411721a9117fbf82aea638fe8a1", size = 38084, upload-time = "2025-10-08T19:46:42.693Z" },
./uv.lock-2869-    { url = "https://files.pythonhosted.org/packages/f4/78/6cce448e2098e9f3bfc91bb877f06aa24b6ccace872e39c53b2f707c4648/propcache-0.4.1-cp311-cp311-win_amd64.whl", hash = "sha256:364426a62660f3f699949ac8c621aad6977be7126c5807ce48c0aeb8e7333ea6", size = 41637, upload-time = "2025-10-08T19:46:43.778Z" },
./uv.lock-2870-    { url = "https://files.pythonhosted.org/packages/9c/e9/754f180cccd7f51a39913782c74717c581b9cc8177ad0e949f4d51812383/propcache-0.4.1-cp311-cp311-win_arm64.whl", hash = "sha256:e53f3a38d3510c11953f3e6a33f205c6d1b001129f972805ca9b42fc308bc239", size = 38064, upload-time = "2025-10-08T19:46:44.872Z" },
--
./uv.lock-2906-    { url = "https://files.pythonhosted.org/packages/36/1d/fc272a63c8d3bbad6878c336c7a7dea15e8f2d23a544bda43205dfa83ada/propcache-0.4.1-cp313-cp313t-manylinux2014_s390x.manylinux_2_17_s390x.manylinux_2_28_s390x.whl", hash = "sha256:af223b406d6d000830c6f65f1e6431783fc3f713ba3e6cc8c024d5ee96170a4b", size = 280420, upload-time = "2025-10-08T19:47:36.338Z" },
./uv.lock-2907-    { url = "https://files.pythonhosted.org/packages/07/0c/01f2219d39f7e53d52e5173bcb09c976609ba30209912a0680adfb8c593a/propcache-0.4.1-cp313-cp313t-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", hash = "sha256:a78372c932c90ee474559c5ddfffd718238e8673c340dc21fe45c5b8b54559a0", size = 263254, upload-time = "2025-10-08T19:47:37.692Z" },
./uv.lock-2908-    { url = "https://files.pythonhosted.org/packages/2d/18/cd28081658ce597898f0c4d174d4d0f3c5b6d4dc27ffafeef835c95eb359/propcache-0.4.1-cp313-cp313t-musllinux_1_2_aarch64.whl", hash = "sha256:564d9f0d4d9509e1a870c920a89b2fec951b44bf5ba7d537a9e7c1ccec2c18af", size = 261205, upload-time = "2025-10-08T19:47:39.659Z" },
./uv.lock:2909:    { url = "https://files.pythonhosted.org/packages/7a/71/1f9e22eb8b8316701c2a19fa1f388c8a3185082607da8e406a803c9b954e/propcache-0.4.1-cp313-cp313t-musllinux_1_2_armv7l.whl", hash = "sha256:17612831fda0138059cc5546f4d12a2aacfb9e47068c06af35c400ba58ba7393", size = 247873, upload-time = "2025-10-08T19:47:41.084Z" },
./uv.lock-2910-    { url = "https://files.pythonhosted.org/packages/4a/65/3d4b61f36af2b4eddba9def857959f1016a51066b4f1ce348e0cf7881f58/propcache-0.4.1-cp313-cp313t-musllinux_1_2_ppc64le.whl", hash = "sha256:41a89040cb10bd345b3c1a873b2bf36413d48da1def52f268a055f7398514874", size = 262739, upload-time = "2025-10-08T19:47:42.51Z" },
./uv.lock-2911-    { url = "https://files.pythonhosted.org/packages/2a/42/26746ab087faa77c1c68079b228810436ccd9a5ce9ac85e2b7307195fd06/propcache-0.4.1-cp313-cp313t-musllinux_1_2_s390x.whl", hash = "sha256:e35b88984e7fa64aacecea39236cee32dd9bd8c55f57ba8a75cf2399553f9bd7", size = 263514, upload-time = "2025-10-08T19:47:43.927Z" },
./uv.lock-2912-    { url = "https://files.pythonhosted.org/packages/94/13/630690fe201f5502d2403dd3cfd451ed8858fe3c738ee88d095ad2ff407b/propcache-0.4.1-cp313-cp313t-musllinux_1_2_x86_64.whl", hash = "sha256:6f8b465489f927b0df505cbe26ffbeed4d6d8a2bbc61ce90eb074ff129ef0ab1", size = 257781, upload-time = "2025-10-08T19:47:45.448Z" },
--
./uv.lock-3085-    { url = "https://files.pythonhosted.org/packages/f8/85/c2b1706e51942de19076eff082f8495e57d5151364e78b5bef4af4a1d94a/pyproj-3.7.2-cp313-cp313-manylinux_2_28_x86_64.whl", hash = "sha256:5141a538ffdbe4bfd157421828bb2e07123a90a7a2d6f30fa1462abcfb5ce681", size = 9514269, upload-time = "2025-08-14T12:04:34.599Z" },
./uv.lock-3086-    { url = "https://files.pythonhosted.org/packages/34/38/07a9b89ae7467872f9a476883a5bad9e4f4d1219d31060f0f2b282276cbe/pyproj-3.7.2-cp313-cp313-musllinux_1_2_aarch64.whl", hash = "sha256:f000841e98ea99acbb7b8ca168d67773b0191de95187228a16110245c5d954d5", size = 10808437, upload-time = "2025-08-14T12:04:36.485Z" },
./uv.lock-3087-    { url = "https://files.pythonhosted.org/packages/12/56/fda1daeabbd39dec5b07f67233d09f31facb762587b498e6fc4572be9837/pyproj-3.7.2-cp313-cp313-musllinux_1_2_x86_64.whl", hash = "sha256:8115faf2597f281a42ab608ceac346b4eb1383d3b45ab474fd37341c4bf82a67", size = 10745540, upload-time = "2025-08-14T12:04:38.568Z" },
./uv.lock:3088:    { url = "https://files.pythonhosted.org/packages/0d/90/c793182cbba65a39a11db2ac6b479fe76c59e6509ae75e5744c344a0da9d/pyproj-3.7.2-cp313-cp313-win32.whl", hash = "sha256:f18c0579dd6be00b970cb1a6719197fceecc407515bab37da0066f0184aafdf3", size = 5896506, upload-time = "2025-08-14T12:04:41.059Z" },
./uv.lock-3089-    { url = "https://files.pythonhosted.org/packages/be/0f/747974129cf0d800906f81cd25efd098c96509026e454d4b66868779ab04/pyproj-3.7.2-cp313-cp313-win_amd64.whl", hash = "sha256:bb41c29d5f60854b1075853fe80c58950b398d4ebb404eb532536ac8d2834ed7", size = 6310195, upload-time = "2025-08-14T12:04:42.974Z" },
./uv.lock-3090-    { url = "https://files.pythonhosted.org/packages/82/64/fc7598a53172c4931ec6edf5228280663063150625d3f6423b4c20f9daff/pyproj-3.7.2-cp313-cp313-win_arm64.whl", hash = "sha256:2b617d573be4118c11cd96b8891a0b7f65778fa7733ed8ecdb297a447d439100", size = 6230748, upload-time = "2025-08-14T12:04:44.491Z" },
./uv.lock-3091-    { url = "https://files.pythonhosted.org/packages/aa/f0/611dd5cddb0d277f94b7af12981f56e1441bf8d22695065d4f0df5218498/pyproj-3.7.2-cp313-cp313t-macosx_13_0_x86_64.whl", hash = "sha256:d27b48f0e81beeaa2b4d60c516c3a1cfbb0c7ff6ef71256d8e9c07792f735279", size = 6241729, upload-time = "2025-08-14T12:04:46.274Z" },
--
./scripts/bench/bench_mpas_spmd_scaling.py-103-    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
./scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
./scripts/bench/bench_mpas_spmd_scaling.py-105-    device count of THIS run's mesh/model (the two differ for the
./scripts/bench/bench_mpas_spmd_scaling.py:106:    single-device reference leg of a ladder, via ``--reorder-for``).
./scripts/bench/bench_mpas_spmd_scaling.py-107-    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
./scripts/bench/bench_mpas_spmd_scaling.py-108-    wave) so the sharded step's packed tracer halo exchange + RK tracer
./scripts/bench/bench_mpas_spmd_scaling.py-109-    advection sit on the timed/gated path.
--
./scripts/bench/bench_mpas_spmd_scaling.py-125-        raise SystemExit(
./scripts/bench/bench_mpas_spmd_scaling.py-126-            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
./scripts/bench/bench_mpas_spmd_scaling.py-127-            f"divisible by --n-devices {run_nd}; use a ladder where every "
./scripts/bench/bench_mpas_spmd_scaling.py:128:            f"count divides --reorder-for ({reorder_target}).")
./scripts/bench/bench_mpas_spmd_scaling.py-129-    sigma = create_sigma_coordinate(nlev)
./scripts/bench/bench_mpas_spmd_scaling.py-130-    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
./scripts/bench/bench_mpas_spmd_scaling.py-131-    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
--
./scripts/bench/bench_mpas_spmd_scaling.py-186-                        "mesh (scaling receipts only, never physics — "
./scripts/bench/bench_mpas_spmd_scaling.py-187-                        "must match the prewarmed cache key at subdiv>=9).")
./scripts/bench/bench_mpas_spmd_scaling.py-188-    p.add_argument("--n-devices", type=int, required=True)
./scripts/bench/bench_mpas_spmd_scaling.py:189:    p.add_argument("--reorder-for", type=int, default=None,
./scripts/bench/bench_mpas_spmd_scaling.py-190-                   help="partition/reorder the mesh for THIS device count "
./scripts/bench/bench_mpas_spmd_scaling.py-191-                        "(default: --n-devices). Pin it to the ladder's "
./scripts/bench/bench_mpas_spmd_scaling.py-192-                        "max so single-device reference runs time the "
--
./scripts/bench/bench_mpas_spmd_scaling.py-308-    reorder_for = args.reorder_for if args.reorder_for is not None else nd
./scripts/bench/bench_mpas_spmd_scaling.py-309-    if reorder_for < nd:
./scripts/bench/bench_mpas_spmd_scaling.py-310-        raise SystemExit(
./scripts/bench/bench_mpas_spmd_scaling.py:311:            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
./scripts/bench/bench_mpas_spmd_scaling.py-312-            f"the ghost padding only guarantees divisibility for the "
./scripts/bench/bench_mpas_spmd_scaling.py-313-            f"partition target.")
./scripts/bench/bench_mpas_spmd_scaling.py-314-    mesh, model, s0, dev_config = build_model_and_state(
--
./scripts/run/run_omip_core2.py-1237-                                     # basin past its 47N/54E geographic edge
./scripts/run/run_omip_core2.py-1238-                                     # (ignition at 48.5N/51E, job 8486227).)
./scripts/run/run_omip_core2.py-1239-        (43.0, 48.0, 57.0, 62.0),    # Aral Sea (endorheic)
./scripts/run/run_omip_core2.py:1240:        (41.0, 49.0, 268.0, 285.0),  # Great Lakes (inland; no-op if WOA-land)
./scripts/run/run_omip_core2.py-1241-    ]
./scripts/run/run_omip_core2.py-1242-    n_before = int(out.sum())
./scripts/run/run_omip_core2.py-1243-    for lat0, lat1, lon0, lon1 in boxes:
--
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-11-#SBATCH --output=mpas_s8_l0.%j.log
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-12-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:18:# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-21-#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
--
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:49:      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:53:echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-54-for NP in 8 16 32; do
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-55-  F="$OUTDIR/np${NP}.jsonl"
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-56-  "$PY" -c "
--
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-61-RUN_MPAS="${RUN_MPAS:-1}"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-62-# Lat-band lanes C/D task count (8 = 2 nodes; 16 = sbatch --nodes=4).
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-63-NP_LL="${NP_LL:-8}"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:64:# Lane E device count; >6 pads the mesh via --reorder-for (even split).
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-65-NP_MPAS="${NP_MPAS:-6}"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-66-# Slurm job STEPS do not inherit the job's GPU allocation on all Slurm
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-67-# versions/configs — without an explicit step gres some tasks see zero
--
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-198-if [ "$RUN_MPAS" = "1" ]; then
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-199-    echo "=== LANE E: icosahedral MPAS multicontroller np=$NP_MPAS (L$ICO_LEVEL) ==="
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-200-    # >6 devices: 10*4^L+2 = 2*odd has no even split beyond 6 — pad the mesh
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:201:    # to the target count via --reorder-for (identical padded mesh per ladder).
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-202-    E_TPN=3; [ $((NP_MPAS % 4)) -eq 0 ] && E_TPN=4
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-203-    E_REORDER=""
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:204:    [ "$NP_MPAS" -gt 6 ] && E_REORDER="--reorder-for $NP_MPAS"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-205-    # shellcheck disable=SC2086
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-206-    srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-207-        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
--
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-34-#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:36:# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
--
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-68-    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-69-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:71:      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:72:      --partition-method sfc --reorder-for 128 \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-75-  echo "[$1] exit=$s epoch=$(date +%s.%N)"
--
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-9-#SBATCH --output=prewarm_s10.%j.log
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-10-# Prewarm subdiv-10 lloyd=0 (10.5M cells, ~7 GB npz) into the shared
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-11-# mesh cache — unlocks MPAS at 128-224 GPUs ABOVE the ~30k tile floor
./scripts/cluster/scaling_levante/prewarm_s10.sbatch:12:# (np128 = 81.9k, np224 = 46.8k cells/GPU).  s9 (2.62M) built in 67 min;
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-13-# s10 estimated ~4-5 h.
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-14-set -uo pipefail
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-15-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
--
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-19-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-20-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-21-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
./scripts/cluster/scaling_levante/prewarm_s10.sbatch:22:"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0
--
./data/les_cases/LBA/snd-28-11330.0000  230.1000  345.6043    0.1231   -4.2600    2.6600
./data/les_cases/LBA/snd-29-11835.9990  213.2000  347.0636    0.0781   -7.5200    4.7900
./data/les_cases/LBA/snd-30-12342.0000  197.0000  347.6107    0.0500   -8.8800    3.4000
./data/les_cases/LBA/snd:31:12841.0000  182.3000  348.0988    0.0326   -9.0000    3.1400
./data/les_cases/LBA/snd-32-13348.0000  167.9000  348.9892    0.0175   -7.7700    3.9300
./data/les_cases/LBA/snd:33:13841.0000  154.9000  350.1689    0.0092   -5.3700    7.5700
./data/les_cases/LBA/snd-34-14313.0000  143.0000  352.8745    0.0055   -3.8800    2.5800
./data/les_cases/LBA/snd-35-14826.0000  131.1000  355.2420    0.0030   -1.1500    2.5000
./data/les_cases/LBA/snd-36-15328.0000  119.7000  358.9141    0.0017   -2.3600    6.4400
./data/les_cases/LBA/snd-37-15888.9990  108.9000  362.8484    0.0011   -9.2000    6.8400
./data/les_cases/LBA/snd-38-16361.0000  100.1000  371.5976    0.0011   -8.0100    0.1900
./data/les_cases/LBA/snd:39:16831.0020   92.1000  381.9160    0.0012   -5.6800   -2.2000
./data/les_cases/LBA/snd-40-17312.0000   84.6000  388.5070    0.0011   -8.8300   -3.6000
./data/les_cases/LBA/snd-41-17804.0000   77.5000  398.8011    0.0011  -14.5100    0.5600
./data/les_cases/LBA/snd-42-18267.0000   71.4000  414.3179    0.0016  -15.5500    6.6800
--
./data/les_cases/LBA/snd-76-11330.0000  230.1000  345.6043    0.1231   -4.2600    2.6600
./data/les_cases/LBA/snd-77-11835.9990  213.2000  347.0636    0.0781   -7.5200    4.7900
./data/les_cases/LBA/snd-78-12342.0000  197.0000  347.6107    0.0500   -8.8800    3.4000
./data/les_cases/LBA/snd:79:12841.0000  182.3000  348.0988    0.0326   -9.0000    3.1400
./data/les_cases/LBA/snd-80-13348.0000  167.9000  348.9892    0.0175   -7.7700    3.9300
./data/les_cases/LBA/snd:81:13841.0000  154.9000  350.1689    0.0092   -5.3700    7.5700
./data/les_cases/LBA/snd-82-14313.0000  143.0000  352.8745    0.0055   -3.8800    2.5800
./data/les_cases/LBA/snd-83-14826.0000  131.1000  355.2420    0.0030   -1.1500    2.5000
./data/les_cases/LBA/snd-84-15328.0000  119.7000  358.9141    0.0017   -2.3600    6.4400
./data/les_cases/LBA/snd-85-15888.9990  108.9000  362.8484    0.0011   -9.2000    6.8400
./data/les_cases/LBA/snd-86-16361.0000  100.1000  371.5976    0.0011   -8.0100    0.1900
./data/les_cases/LBA/snd:87:16831.0020   92.1000  381.9160    0.0012   -5.6800   -2.2000
./data/les_cases/LBA/snd-88-17312.0000   84.6000  388.5070    0.0011   -8.8300   -3.6000
./data/les_cases/LBA/snd-89-17804.0000   77.5000  398.8011    0.0011  -14.5100    0.5600
./data/les_cases/LBA/snd-90-18267.0000   71.4000  414.3179    0.0016  -15.5500    6.6800
--
./data/les_cases/BOMEX/grd_74-21-   1503.486              21   136.6805    
./data/les_cases/BOMEX/grd_74-22-   1653.834              22   150.3486    
./data/les_cases/BOMEX/grd_74-23-   1819.218              23   165.3835    
./data/les_cases/BOMEX/grd_74:24:   2001.140              24   181.9218    
./data/les_cases/BOMEX/grd_74-25-   2201.311              25   200.1709    
./data/les_cases/BOMEX/grd_74-26-   2431.507              26   230.1966    
./data/les_cases/BOMEX/grd_74-27-   2696.233              27   264.7261    
--
./packages/ocean/legoesm/ocean/bathymetry.py-50-    ("Hormuz",                    26.5,  56.5,  40.0,  100.0),
./packages/ocean/legoesm/ocean/bathymetry.py-51-    ("Malacca",                    2.5, 101.5,  40.0,   50.0),
./packages/ocean/legoesm/ocean/bathymetry.py-52-    ("Indonesian Throughflow",    -3.0, 120.0, 200.0, 1500.0),
./packages/ocean/legoesm/ocean/bathymetry.py:53:    ("Mozambique Channel",       -17.0,  41.0, 200.0, 2500.0),
./packages/ocean/legoesm/ocean/bathymetry.py-54-    ("Denmark Strait",            66.0, -27.0, 150.0,  600.0),
./packages/ocean/legoesm/ocean/bathymetry.py-55-    ("Faroe Bank Channel",        61.5,  -8.5, 100.0,  800.0),
./packages/ocean/legoesm/ocean/bathymetry.py-56-    ("Bering Strait",             65.8,-169.0,  60.0,   40.0),
--
./packages/ocean/legoesm/ocean/bathymetry.py-59-    ("Taiwan Strait",             24.0, 119.5,  80.0,   60.0),
./packages/ocean/legoesm/ocean/bathymetry.py-60-    ("Windward Passage",          20.0, -73.5,  60.0, 1500.0),
./packages/ocean/legoesm/ocean/bathymetry.py-61-    ("Florida Strait",            25.5, -79.5, 100.0,  800.0),
./packages/ocean/legoesm/ocean/bathymetry.py:62:    ("Luzon Strait",              20.5, 121.5, 100.0, 2000.0),
./packages/ocean/legoesm/ocean/bathymetry.py-63-]
./packages/ocean/legoesm/ocean/bathymetry.py-64-
./packages/ocean/legoesm/ocean/bathymetry.py-65-
--
./scripts/validate/compare_mie_vs_microhh.py-89-  print(f"     {'band':>4} {'r_eff':>6} {'g_mc':>7} {'g_ref':>7} "
./scripts/validate/compare_mie_vs_microhh.py-90-        f"{'<ang>_mc':>9} {'<ang>_ref':>10} {'L1':>6}")
./scripts/validate/compare_mie_vs_microhh.py-91-  for b in (1, 7, 13):
./scripts/validate/compare_mie_vs_microhh.py:92:    for r_eff in (3.5, 10.5, 20.5):
./scripts/validate/compare_mie_vs_microhh.py-93-      us = jax.random.uniform(jax.random.PRNGKey(b * 100 + int(r_eff)),
./scripts/validate/compare_mie_vs_microhh.py-94-                              (args.n_sample,))
./scripts/validate/compare_mie_vs_microhh.py-95-      cos = np.asarray(jax.vmap(lambda u: mie_sample_cos(
--
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py-241-
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py-242-# RGB class-index formula (NEMO trc_oce.F90): itab = NINT(offset + slope*log10(Chl)).
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py-243-# Fixed published constants — not tunable.
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:244:_RGB_CLASS_INDEX_OFFSET: float = 41.0
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py-245-_RGB_CLASS_INDEX_SLOPE: float = 20.0
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py-246-
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py-247-# --- Morel & Berthon (1989) analytical vertical Chl profile coefficients ----
--
./data/les_cases/ARM9707/snd-162-  3840.068   615.000   318.372     4.032     7.865    -1.874
./data/les_cases/ARM9707/snd-163-  4522.320   565.000   319.943     3.087     7.105    -3.441
./data/les_cases/ARM9707/snd-164-  5252.780   515.000   321.914     2.182     6.609    -3.637
./data/les_cases/ARM9707/snd:165:  6041.001   465.000   324.546     1.410     6.303    -3.339
./data/les_cases/ARM9707/snd-166-  6899.272   415.000   327.462     0.854     5.739    -2.781
./data/les_cases/ARM9707/snd-167-  7842.888   365.000   330.264     0.514     4.743    -2.501
./data/les_cases/ARM9707/snd-168-  8892.266   315.000   332.831     0.277     3.450    -2.192
--
./data/les_cases/ARM9707/snd-793-  6897.684   415.000   331.204     1.350     5.047     7.994
./data/les_cases/ARM9707/snd-794-  7853.778   365.000   335.098     0.712     4.638     8.563
./data/les_cases/ARM9707/snd-795-  8919.749   315.000   338.447     0.324     4.225     8.827
./data/les_cases/ARM9707/snd:796: 10124.940   265.000   341.085     0.144     4.021     7.848
./data/les_cases/ARM9707/snd-797- 11517.335   215.000   344.296     0.056     3.826     6.282
./data/les_cases/ARM9707/snd-798- 13197.342   165.000   354.860     0.021     2.670     4.493
./data/les_cases/ARM9707/snd-799- 15414.795   115.000   385.769     0.012    -0.676     2.746
--
./data/les_cases/ARM9707/snd-1172-  6022.636   465.000   324.700     1.901     7.307    -0.277
./data/les_cases/ARM9707/snd-1173-  6882.374   415.000   328.263     1.211     8.064     0.532
./data/les_cases/ARM9707/snd-1174-  7828.857   365.000   331.350     0.736     9.190     1.891
./data/les_cases/ARM9707/snd:1175:  8881.953   315.000   334.034     0.364     9.633     3.338
./data/les_cases/ARM9707/snd-1176- 10072.731   265.000   337.379     0.156    11.196     4.857
./data/les_cases/ARM9707/snd-1177- 11456.920   215.000   344.065     0.055    12.064     5.216
./data/les_cases/ARM9707/snd-1178- 13141.479   165.000   357.077     0.018    11.914     3.690
--
./data/les_cases/ARM9707/snd-1301-  3207.081   665.000   311.898     3.478     4.937    -0.410
./data/les_cases/ARM9707/snd-1302-  3838.034   615.000   314.053     2.187     4.940    -0.732
./data/les_cases/ARM9707/snd-1303-  4511.786   565.000   317.026     1.297     4.744    -0.692
./data/les_cases/ARM9707/snd:1304:  5236.637   515.000   320.516     0.919     4.348    -0.673
./data/les_cases/ARM9707/snd-1305-  6021.961   465.000   323.947     0.808     4.162    -0.734
./data/les_cases/ARM9707/snd-1306-  6878.799   415.000   327.155     0.559     4.827    -1.034
./data/les_cases/ARM9707/snd-1307-  7821.968   365.000   330.364     0.333     6.853    -1.367
--
./data/les_cases/ARM9707/snd-1328- 10068.659   265.000   337.385     0.086    14.822    -3.341
./data/les_cases/ARM9707/snd-1329- 11454.805   215.000   345.072     0.033    17.030    -2.916
./data/les_cases/ARM9707/snd-1330- 13143.924   165.000   357.969     0.008    13.855    -1.667
./data/les_cases/ARM9707/snd:1331: 15381.994   115.000   389.572     0.004     6.246    -0.047
./data/les_cases/ARM9707/snd-1332-    178.7290              18   970.4337     day,levels,pres0
./data/les_cases/ARM9707/snd-1333-    49.856   965.000   303.882    14.896     2.802     3.592
./data/les_cases/ARM9707/snd-1334-   519.461   915.000   305.128    13.152     3.625     4.861
--
./data/les_cases/ARM9707/snd-1338-  2619.317   715.000   310.244     6.018     4.902     2.142
./data/les_cases/ARM9707/snd-1339-  3214.646   665.000   312.088     4.401     5.610     1.378
./data/les_cases/ARM9707/snd-1340-  3846.277   615.000   314.202     3.007     6.081     0.899
./data/les_cases/ARM9707/snd:1341:  4520.503   565.000   317.066     1.793     5.998     0.706
./data/les_cases/ARM9707/snd:1342:  5245.625   515.000   320.575     1.140     5.524     0.591
./data/les_cases/ARM9707/snd-1343-  6031.453   465.000   324.271     0.759     4.777     0.478
./data/les_cases/ARM9707/snd-1344-  6889.572   415.000   327.827     0.525     4.983     0.359
./data/les_cases/ARM9707/snd-1345-  7834.637   365.000   331.019     0.332     6.622    -0.099
--
./data/les_cases/ARM9707/snd-1610-  6860.663   415.000   327.533     0.236     6.557    -2.892
./data/les_cases/ARM9707/snd-1611-  7805.235   365.000   331.079     0.113     9.889    -3.875
./data/les_cases/ARM9707/snd-1612-  8859.880   315.000   335.507     0.071    13.992    -4.925
./data/les_cases/ARM9707/snd:1613: 10059.479   265.000   341.016     0.043    17.993    -6.459
./data/les_cases/ARM9707/snd-1614- 11459.198   215.000   348.116     0.019    19.568    -6.238
./data/les_cases/ARM9707/snd-1615- 13158.888   165.000   359.258     0.007    16.493    -4.056
./data/les_cases/ARM9707/snd-1616- 15399.229   115.000   388.941     0.004     7.743    -0.732
--
./data/les_cases/ARM9707/snd-1646-  5229.778   515.000   321.222     1.467     9.389    -2.077
./data/les_cases/ARM9707/snd-1647-  6016.336   465.000   324.159     0.698     9.443    -1.266
./data/les_cases/ARM9707/snd-1648-  6874.268   415.000   327.855     0.297    10.179    -1.590
./data/les_cases/ARM9707/snd:1649:  7820.524   365.000   331.914     0.183    11.307    -2.532
./data/les_cases/ARM9707/snd-1650-  8878.138   315.000   336.523     0.142    12.950    -3.518
./data/les_cases/ARM9707/snd-1651- 10080.378   265.000   341.454     0.074    16.340    -5.766
./data/les_cases/ARM9707/snd-1652- 11480.167   215.000   347.677     0.027    18.397    -6.197
--
./data/les_cases/ARM9707/snd-1823- 11554.488   215.000   347.553     0.014    16.128    -5.491
./data/les_cases/ARM9707/snd-1824- 13244.364   165.000   355.625     0.007    14.543    -4.779
./data/les_cases/ARM9707/snd-1825- 15462.211   115.000   385.064     0.007     8.228    -2.236
./data/les_cases/ARM9707/snd:1826:    181.9790              18   964.3576     day,levels,pres0
./data/les_cases/ARM9707/snd-1827-    -5.982   965.000   306.593    17.456     0.298     7.070
./data/les_cases/ARM9707/snd-1828-   468.361   915.000   307.673    15.360     1.668     9.426
./data/les_cases/ARM9707/snd-1829-   963.106   865.000   309.368    12.704     3.508     9.976
--
./data/les_cases/ARM9707/snd-2004-  3208.461   665.000   319.447     3.900    11.052     2.072
./data/les_cases/ARM9707/snd-2005-  3854.215   615.000   320.990     2.742    10.862     0.430
./data/les_cases/ARM9707/snd-2006-  4542.633   565.000   323.683     1.445    11.965    -1.696
./data/les_cases/ARM9707/snd:2007:  5281.994   515.000   326.666     0.500    13.498    -3.464
./data/les_cases/ARM9707/snd-2008-  6081.006   465.000   329.215     0.236    15.006    -4.901
./data/les_cases/ARM9707/snd-2009-  6950.068   415.000   331.355     0.154    15.066    -4.722
./data/les_cases/ARM9707/snd-2010-  7904.505   365.000   334.140     0.095    14.765    -2.897
--
./data/les_cases/ARM9707/snd-2061-  3213.958   665.000   318.439     3.915     7.236     3.760
./data/les_cases/ARM9707/snd-2062-  3857.907   615.000   320.229     2.623     7.689     2.734
./data/les_cases/ARM9707/snd-2063-  4544.558   565.000   322.822     1.393     9.072     0.546
./data/les_cases/ARM9707/snd:2064:  5281.946   515.000   325.801     0.510    12.063    -1.579
./data/les_cases/ARM9707/snd-2065-  6078.696   465.000   328.215     0.249    14.437    -3.274
./data/les_cases/ARM9707/snd-2066-  6945.053   415.000   330.288     0.190    16.318    -2.796
./data/les_cases/ARM9707/snd-2067-  7896.095   365.000   332.821     0.133    17.183    -0.818
./data/les_cases/ARM9707/snd-2068-  8955.298   315.000   336.629     0.069    18.832     1.594
./data/les_cases/ARM9707/snd:2069: 10157.033   265.000   341.076     0.032    20.369     2.290
./data/les_cases/ARM9707/snd-2070- 11553.142   215.000   346.223     0.015    21.392     1.591
./data/les_cases/ARM9707/snd-2071- 13237.146   165.000   354.522     0.007    19.403     0.599
./data/les_cases/ARM9707/snd-2072- 15449.357   115.000   384.307     0.005    11.284     0.470
--
./data/les_cases/ARM9707/snd-2085-  6950.265   415.000   329.998     0.162    15.916    -2.605
./data/les_cases/ARM9707/snd-2086-  7900.333   365.000   332.441     0.112    17.184    -1.016
./data/les_cases/ARM9707/snd-2087-  8958.318   315.000   336.245     0.062    19.005     0.729
./data/les_cases/ARM9707/snd:2088: 10159.048   265.000   340.900     0.029    20.533     1.507
./data/les_cases/ARM9707/snd-2089- 11554.698   215.000   346.178     0.014    21.435     1.383
./data/les_cases/ARM9707/snd-2090- 13238.920   165.000   354.664     0.006    19.987     0.671
./data/les_cases/ARM9707/snd-2091- 15452.260   115.000   384.547     0.004    11.779     0.674
--
./data/les_cases/ARM9707/snd-2157-  3858.994   615.000   321.608     0.995    10.107     0.455
./data/les_cases/ARM9707/snd-2158-  4547.625   565.000   323.793     0.475    10.767     0.538
./data/les_cases/ARM9707/snd-2159-  5285.932   515.000   325.852     0.294    11.743     1.113
./data/les_cases/ARM9707/snd:2160:  6081.915   465.000   327.578     0.198    12.700     2.035
./data/les_cases/ARM9707/snd-2161-  6946.197   415.000   329.366     0.127    14.037     2.716
./data/les_cases/ARM9707/snd-2162-  7894.601   365.000   331.927     0.081    15.925     3.232
./data/les_cases/ARM9707/snd-2163-  8950.931   315.000   335.720     0.055    18.418     3.829
--
./data/les_cases/ARM9707/snd-2310-  4504.557   565.000   320.196     1.909    14.044    -0.646
./data/les_cases/ARM9707/snd-2311-  5235.366   515.000   322.425     1.041    15.239     1.121
./data/les_cases/ARM9707/snd-2312-  6023.753   465.000   324.560     0.547    16.559     3.866
./data/les_cases/ARM9707/snd:2313:  6881.932   415.000   327.656     0.334    17.504     5.391
./data/les_cases/ARM9707/snd-2314-  7826.723   365.000   331.058     0.255    19.779     5.702
./data/les_cases/ARM9707/snd-2315-  8880.768   315.000   335.094     0.157    22.654     4.677
./data/les_cases/ARM9707/snd-2316- 10078.290   265.000   340.225     0.075    26.630     3.636
--
./data/les_cases/ARM9707/snd-2768-  5997.891   465.000   321.691     0.742     5.789    -6.753
./data/les_cases/ARM9707/snd-2769-  6847.880   415.000   324.242     0.395     7.032    -6.210
./data/les_cases/ARM9707/snd-2770-  7781.691   365.000   326.803     0.186     9.131    -5.578
./data/les_cases/ARM9707/snd:2771:  8820.515   315.000   329.734     0.095    12.265    -5.555
./data/les_cases/ARM9707/snd-2772-  9999.126   265.000   334.941     0.059    16.546    -5.215
./data/les_cases/ARM9707/snd-2773- 11378.414   215.000   344.193     0.030    19.295    -3.526
./data/les_cases/ARM9707/snd-2774- 13072.068   165.000   360.877     0.014    18.326    -3.039
--
./data/les_cases/ARM9707/snd-2945- 13136.476   165.000   358.831     0.016    13.493    -2.326
./data/les_cases/ARM9707/snd-2946- 15384.560   115.000   392.131     0.011     6.523    -1.860
./data/les_cases/ARM9707/snd-2947-    189.3540              18   969.5034     day,levels,pres0
./data/les_cases/ARM9707/snd:2948:    41.041   965.000   301.390    16.302     1.589     6.695
./data/les_cases/ARM9707/snd-2949-   507.699   915.000   303.220    14.936     6.325     9.398
./data/les_cases/ARM9707/snd-2950-   996.697   865.000   306.721    12.858     9.450     8.418
./data/les_cases/ARM9707/snd-2951-  1510.833   815.000   309.430    10.531     9.578     4.624
--
./data/les_cases/ARM9707/snd-3030-  3234.745   665.000   314.964     4.040     6.028     1.017
./data/les_cases/ARM9707/snd-3031-  3871.407   615.000   316.399     2.893     5.356    -0.517
./data/les_cases/ARM9707/snd-3032-  4549.066   565.000   318.050     2.001     5.118    -1.832
./data/les_cases/ARM9707/snd:3033:  5275.310   515.000   320.509     1.211     5.813    -2.180
./data/les_cases/ARM9707/snd-3034-  6060.332   465.000   323.675     0.650     6.660    -1.833
./data/les_cases/ARM9707/snd-3035-  6916.659   415.000   327.110     0.382     8.028    -0.951
./data/les_cases/ARM9707/snd-3036-  7859.601   365.000   330.299     0.267     9.874    -0.178
--
./data/les_cases/ARM9707/snd-3087-  3226.369   665.000   314.561     4.006     2.887     0.698
./data/les_cases/ARM9707/snd-3088-  3862.258   615.000   316.012     3.046     2.187     0.540
./data/les_cases/ARM9707/snd-3089-  4539.352   565.000   317.847     2.184     2.618     0.355
./data/les_cases/ARM9707/snd:3090:  5265.442   515.000   320.500     1.436     4.205     0.247
./data/les_cases/ARM9707/snd-3091-  6051.360   465.000   324.331     0.925     6.476     0.002
./data/les_cases/ARM9707/snd-3092-  6910.409   415.000   328.416     0.692     9.011    -0.173
./data/les_cases/ARM9707/snd-3093-  7857.850   365.000   332.027     0.501    11.262    -0.632
--
./data/les_cases/ARM9707/snd-3266- 10113.098   265.000   338.578     0.154    12.605    -0.115
./data/les_cases/ARM9707/snd-3267- 11499.005   215.000   343.663     0.063    13.697    -0.152
./data/les_cases/ARM9707/snd-3268- 13183.059   165.000   357.290     0.023    11.318     0.003
./data/les_cases/ARM9707/snd:3269: 15420.564   115.000   390.121     0.013     3.595    -0.200
./data/les_cases/ARM9707/snd-3270-    191.4790              18   970.1393     day,levels,pres0
./data/les_cases/ARM9707/snd-3271-    46.725   965.000   300.878    15.917     1.404     5.073
./data/les_cases/ARM9707/snd-3272-   513.311   915.000   303.713    14.958     3.629     7.857
--
./data/les_cases/ARM9707/snd-3309-    48.853   965.000   302.821    16.551    -1.108     4.356
./data/les_cases/ARM9707/snd-3310-   517.991   915.000   304.926    15.109     0.647     6.695
./data/les_cases/ARM9707/snd-3311-  1008.135   865.000   306.229    13.753     1.942     7.070
./data/les_cases/ARM9707/snd:3312:  1520.589   815.000   307.468    11.923     2.508     6.869
./data/les_cases/ARM9707/snd-3313-  2057.723   765.000   308.935     9.748     2.898     6.358
./data/les_cases/ARM9707/snd-3314-  2622.891   715.000   310.891     7.427     3.490     5.762
./data/les_cases/ARM9707/snd:3315:  3220.500   665.000   313.360     5.468     3.830     5.059
./data/les_cases/ARM9707/snd-3316-  3855.753   615.000   316.118     4.089     3.820     4.705
./data/les_cases/ARM9707/snd-3317-  4534.873   565.000   319.293     2.998     4.012     4.565
./data/les_cases/ARM9707/snd-3318-  5265.649   515.000   322.858     2.300     4.729     4.418
--
./data/les_cases/ARM9707/snd-3551- 10177.257   265.000   341.770     0.151     4.977     9.761
./data/les_cases/ARM9707/snd-3552- 11572.461   215.000   344.991     0.062     4.775     9.898
./data/les_cases/ARM9707/snd-3553- 13256.791   165.000   355.976     0.024     3.297     7.912
./data/les_cases/ARM9707/snd:3554: 15481.975   115.000   387.247     0.016    -1.106     4.003
./data/les_cases/ARM9707/snd-3555-    193.3540              18   969.8561     day,levels,pres0
./data/les_cases/ARM9707/snd-3556-    44.122   965.000   300.742    15.324     2.002     9.748
./data/les_cases/ARM9707/snd-3557-   510.593   915.000   304.038    13.712     5.101    14.158
--
./data/les_cases/ARM9707/snd-3675-  2628.511   715.000   314.199     5.714     6.087     3.744
./data/les_cases/ARM9707/snd-3676-  3230.858   665.000   315.514     4.439     5.432     2.203
./data/les_cases/ARM9707/snd-3677-  3869.343   615.000   317.521     3.226     5.210     1.494
./data/les_cases/ARM9707/snd:3678:  4550.893   565.000   320.520     2.015     5.527     1.076
./data/les_cases/ARM9707/snd-3679-  5284.148   515.000   324.255     1.020     5.989     0.665
./data/les_cases/ARM9707/snd-3680-  6078.750   465.000   327.877     0.437     6.401     0.634
./data/les_cases/ARM9707/snd-3681-  6945.668   415.000   331.025     0.224     6.493     1.042
--
./data/les_cases/ARM9707/snd-3808-  2629.728   715.000   315.232     7.137     7.345     0.181
./data/les_cases/ARM9707/snd-3809-  3234.556   665.000   316.613     5.421     6.154    -1.610
./data/les_cases/ARM9707/snd-3810-  3875.228   615.000   318.266     3.902     5.181    -3.253
./data/les_cases/ARM9707/snd:3811:  4557.889   565.000   320.577     2.523     5.186    -4.266
./data/les_cases/ARM9707/snd-3812-  5290.628   515.000   323.556     1.419     5.924    -4.560
./data/les_cases/ARM9707/snd-3813-  6083.288   465.000   326.855     0.651     6.813    -4.034
./data/les_cases/ARM9707/snd-3814-  6947.456   415.000   329.897     0.300     6.853    -2.607
--
./data/les_cases/ARM9707/snd-4010- 15434.975   115.000   388.334     0.019     3.155    -2.423
./data/les_cases/ARM9707/snd-4011-    196.3540              18   972.7925     day,levels,pres0
./data/les_cases/ARM9707/snd-4012-    71.257   965.000   302.947    16.261    -2.526     1.426
./data/les_cases/ARM9707/snd:4013:   541.058   915.000   305.860    14.347    -1.767     3.070
./data/les_cases/ARM9707/snd-4014-  1033.456   865.000   308.484    12.527    -0.073     2.993
./data/les_cases/ARM9707/snd-4015-  1549.661   815.000   310.222    10.409     1.041     1.752
./data/les_cases/ARM9707/snd-4016-  2091.185   765.000   311.863     7.854     1.416     0.481
--
./packages/core/legoesm/parallel/voronoi_partition.py-900-
./packages/core/legoesm/parallel/voronoi_partition.py-901-
./packages/core/legoesm/parallel/voronoi_partition.py-902-# ============================================================================
./packages/core/legoesm/parallel/voronoi_partition.py:903:# Mesh padding for even sharding
./packages/core/legoesm/parallel/voronoi_partition.py-904-# ============================================================================
./packages/core/legoesm/parallel/voronoi_partition.py-905-
./packages/core/legoesm/parallel/voronoi_partition.py-906-def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
--
./data/les_cases/ARM9707/lsf-162-  3840.068   615.000   0.17505E-04   0.16769E-07     7.865    -1.874     0.000
./data/les_cases/ARM9707/lsf-163-  4522.320   565.000  -0.49052E-05   0.20045E-07     7.105    -3.441     0.000
./data/les_cases/ARM9707/lsf-164-  5252.780   515.000  -0.19318E-04   0.17776E-07     6.609    -3.637     0.000
./data/les_cases/ARM9707/lsf:165:  6041.001   465.000  -0.21186E-04   0.12559E-07     6.303    -3.339     0.000
./data/les_cases/ARM9707/lsf-166-  6899.272   415.000  -0.12827E-04   0.80458E-08     5.739    -2.781     0.000
./data/les_cases/ARM9707/lsf-167-  7842.888   365.000  -0.15395E-05   0.37178E-08     4.743    -2.501     0.000
./data/les_cases/ARM9707/lsf-168-  8892.266   315.000   0.19792E-05   0.54860E-09     3.450    -2.192     0.000
--
./data/les_cases/ARM9707/lsf-1172-  6022.636   465.000  -0.17378E-03   0.43490E-07     7.307    -0.277     0.000
./data/les_cases/ARM9707/lsf-1173-  6882.374   415.000  -0.15754E-03   0.39554E-07     8.064     0.532     0.000
./data/les_cases/ARM9707/lsf-1174-  7828.857   365.000  -0.12953E-03   0.26901E-07     9.190     1.891     0.000
./data/les_cases/ARM9707/lsf:1175:  8881.953   315.000  -0.10504E-03   0.13475E-07     9.633     3.338     0.000
./data/les_cases/ARM9707/lsf-1176- 10072.731   265.000  -0.76808E-04   0.39722E-08    11.196     4.857     0.000
./data/les_cases/ARM9707/lsf-1177- 11456.920   215.000  -0.39609E-04   0.26621E-09    12.064     5.216     0.000
./data/les_cases/ARM9707/lsf-1178- 13141.479   165.000  -0.73919E-05  -0.97373E-10    11.914     3.690     0.000
--
./data/les_cases/ARM9707/lsf-1328- 10068.659   265.000  -0.58348E-05   0.14472E-08    14.822    -3.341     0.000
./data/les_cases/ARM9707/lsf-1329- 11454.805   215.000  -0.37857E-05   0.63864E-09    17.030    -2.916     0.000
./data/les_cases/ARM9707/lsf-1330- 13143.924   165.000  -0.37287E-05   0.83059E-10    13.855    -1.667     0.000
./data/les_cases/ARM9707/lsf:1331: 15381.994   115.000  -0.35812E-05   0.73822E-11     6.246    -0.047     0.000
./data/les_cases/ARM9707/lsf-1332-    178.7290              18   970.4337       day,levels, pres0
./data/les_cases/ARM9707/lsf-1333-    49.856   965.000   0.21777E-04  -0.10939E-08     2.802     3.592     0.000
./data/les_cases/ARM9707/lsf-1334-   519.461   915.000   0.26037E-04  -0.42581E-08     3.625     4.861     0.000
--
./data/les_cases/ARM9707/lsf-1338-  2619.317   715.000   0.15854E-04   0.88060E-08     4.902     2.142     0.000
./data/les_cases/ARM9707/lsf-1339-  3214.646   665.000   0.60831E-05   0.16486E-07     5.610     1.378     0.000
./data/les_cases/ARM9707/lsf-1340-  3846.277   615.000  -0.11562E-04   0.28294E-07     6.081     0.899     0.000
./data/les_cases/ARM9707/lsf:1341:  4520.503   565.000  -0.30475E-04   0.28034E-07     5.998     0.706     0.000
./data/les_cases/ARM9707/lsf-1342-  5245.625   515.000  -0.40834E-04   0.16894E-07     5.524     0.591     0.000
./data/les_cases/ARM9707/lsf-1343-  6031.453   465.000  -0.28151E-04   0.58485E-08     4.777     0.478     0.000
./data/les_cases/ARM9707/lsf-1344-  6889.572   415.000  -0.70689E-05   0.10709E-08     4.983     0.359     0.000
--
./data/les_cases/ARM9707/lsf-1646-  5229.778   515.000  -0.53570E-04   0.31550E-07     9.389    -2.077     0.000
./data/les_cases/ARM9707/lsf-1647-  6016.336   465.000  -0.73160E-04   0.19204E-07     9.443    -1.266     0.000
./data/les_cases/ARM9707/lsf-1648-  6874.268   415.000  -0.88930E-04   0.73046E-08    10.179    -1.590     0.000
./data/les_cases/ARM9707/lsf:1649:  7820.524   365.000  -0.99093E-04   0.16534E-08    11.307    -2.532     0.000
./data/les_cases/ARM9707/lsf-1650-  8878.138   315.000  -0.85697E-04   0.61233E-09    12.950    -3.518     0.000
./data/les_cases/ARM9707/lsf-1651- 10080.378   265.000  -0.58528E-04   0.99643E-09    16.340    -5.766     0.000
./data/les_cases/ARM9707/lsf-1652- 11480.167   215.000  -0.35555E-04   0.31992E-09    18.397    -6.197     0.000
--
./data/les_cases/ARM9707/lsf-1823- 11554.488   215.000   0.25970E-06  -0.13301E-09    16.128    -5.491     0.000
./data/les_cases/ARM9707/lsf-1824- 13244.364   165.000   0.94166E-05  -0.12935E-10    14.543    -4.779     0.000
./data/les_cases/ARM9707/lsf-1825- 15462.211   115.000   0.12150E-04   0.13643E-10     8.228    -2.236     0.000
./data/les_cases/ARM9707/lsf:1826:    181.9790              18   964.3576       day,levels, pres0
./data/les_cases/ARM9707/lsf-1827-    -5.982   965.000   0.30229E-04  -0.27148E-07     0.298     7.070     0.000
./data/les_cases/ARM9707/lsf-1828-   468.361   915.000   0.36064E-04  -0.44640E-07     1.668     9.426     0.000
./data/les_cases/ARM9707/lsf-1829-   963.106   865.000   0.73619E-04  -0.82710E-07     3.508     9.976     0.000
--
./data/les_cases/ARM9707/lsf-2004-  3208.461   665.000   0.49641E-04  -0.20867E-07    11.052     2.072     0.000
./data/les_cases/ARM9707/lsf-2005-  3854.215   615.000   0.44105E-04  -0.21929E-07    10.862     0.430     0.000
./data/les_cases/ARM9707/lsf-2006-  4542.633   565.000   0.43574E-04  -0.23786E-07    11.965    -1.696     0.000
./data/les_cases/ARM9707/lsf:2007:  5281.994   515.000   0.19378E-04  -0.77024E-08    13.498    -3.464     0.000
./data/les_cases/ARM9707/lsf-2008-  6081.006   465.000  -0.10581E-04   0.31097E-09    15.006    -4.901     0.000
./data/les_cases/ARM9707/lsf-2009-  6950.068   415.000  -0.15278E-04   0.16607E-08    15.066    -4.722     0.000
./data/les_cases/ARM9707/lsf-2010-  7904.505   365.000  -0.13341E-04   0.13418E-08    14.765    -2.897     0.000
--
./data/les_cases/ARM9707/lsf-2061-  3213.958   665.000   0.52443E-05  -0.17871E-08     7.236     3.760     0.000
./data/les_cases/ARM9707/lsf-2062-  3857.907   615.000   0.51372E-05   0.10625E-08     7.689     2.734     0.000
./data/les_cases/ARM9707/lsf-2063-  4544.558   565.000   0.14192E-04  -0.11782E-07     9.072     0.546     0.000
./data/les_cases/ARM9707/lsf:2064:  5281.946   515.000  -0.31685E-05  -0.67960E-08    12.063    -1.579     0.000
./data/les_cases/ARM9707/lsf-2065-  6078.696   465.000  -0.30209E-04  -0.24804E-08    14.437    -3.274     0.000
./data/les_cases/ARM9707/lsf-2066-  6945.053   415.000  -0.40477E-04  -0.12672E-08    16.318    -2.796     0.000
./data/les_cases/ARM9707/lsf-2067-  7896.095   365.000  -0.66011E-04   0.68774E-09    17.183    -0.818     0.000
--
./data/les_cases/ARM9707/lsf-2085-  6950.265   415.000  -0.37539E-04  -0.15667E-08    15.916    -2.605     0.000
./data/les_cases/ARM9707/lsf-2086-  7900.333   365.000  -0.65028E-04   0.18650E-09    17.184    -1.016     0.000
./data/les_cases/ARM9707/lsf-2087-  8958.318   315.000  -0.98925E-04   0.93584E-09    19.005     0.729     0.000
./data/les_cases/ARM9707/lsf:2088: 10159.048   265.000  -0.82800E-04   0.91504E-09    20.533     1.507     0.000
./data/les_cases/ARM9707/lsf-2089- 11554.698   215.000  -0.37566E-04   0.37662E-09    21.435     1.383     0.000
./data/les_cases/ARM9707/lsf-2090- 13238.920   165.000  -0.44174E-05   0.61533E-10    19.987     0.671     0.000
./data/les_cases/ARM9707/lsf-2091- 15452.260   115.000   0.17915E-05   0.13945E-10    11.779     0.674     0.000
--
./data/les_cases/ARM9707/lsf-2157-  3858.994   615.000   0.56751E-04  -0.21818E-07    10.107     0.455     0.000
./data/les_cases/ARM9707/lsf-2158-  4547.625   565.000   0.39198E-04  -0.93472E-08    10.767     0.538     0.000
./data/les_cases/ARM9707/lsf-2159-  5285.932   515.000   0.13484E-04  -0.18605E-08    11.743     1.113     0.000
./data/les_cases/ARM9707/lsf:2160:  6081.915   465.000  -0.13514E-04  -0.10071E-09    12.700     2.035     0.000
./data/les_cases/ARM9707/lsf-2161-  6946.197   415.000  -0.34263E-04   0.82492E-09    14.037     2.716     0.000
./data/les_cases/ARM9707/lsf-2162-  7894.601   365.000  -0.49367E-04   0.10952E-08    15.925     3.232     0.000
./data/les_cases/ARM9707/lsf-2163-  8950.931   315.000  -0.45175E-04   0.10340E-08    18.418     3.829     0.000
--
./data/les_cases/ARM9707/lsf-2310-  4504.557   565.000  -0.90698E-04   0.34672E-07    14.044    -0.646     0.000
./data/les_cases/ARM9707/lsf-2311-  5235.366   515.000  -0.12319E-03   0.29208E-07    15.239     1.121     0.000
./data/les_cases/ARM9707/lsf-2312-  6023.753   465.000  -0.15519E-03   0.16375E-07    16.559     3.866     0.000
./data/les_cases/ARM9707/lsf:2313:  6881.932   415.000  -0.17790E-03   0.84304E-08    17.504     5.391     0.000
./data/les_cases/ARM9707/lsf-2314-  7826.723   365.000  -0.19451E-03   0.24035E-08    19.779     5.702     0.000
./data/les_cases/ARM9707/lsf-2315-  8880.768   315.000  -0.21225E-03   0.18837E-08    22.654     4.677     0.000
./data/les_cases/ARM9707/lsf-2316- 10078.290   265.000  -0.16233E-03  -0.36539E-09    26.630     3.636     0.000
--
./data/les_cases/ARM9707/lsf-2768-  5997.891   465.000   0.21453E-04   0.14914E-08     5.789    -6.753     0.000
./data/les_cases/ARM9707/lsf-2769-  6847.880   415.000   0.25277E-04  -0.13811E-08     7.032    -6.210     0.000
./data/les_cases/ARM9707/lsf-2770-  7781.691   365.000   0.28036E-04  -0.14659E-08     9.131    -5.578     0.000
./data/les_cases/ARM9707/lsf:2771:  8820.515   315.000   0.39124E-04   0.68858E-09    12.265    -5.555     0.000
./data/les_cases/ARM9707/lsf-2772-  9999.126   265.000   0.60675E-04   0.10370E-08    16.546    -5.215     0.000
./data/les_cases/ARM9707/lsf-2773- 11378.414   215.000   0.80608E-04   0.68806E-09    19.295    -3.526     0.000
./data/les_cases/ARM9707/lsf-2774- 13072.068   165.000   0.56205E-04   0.25745E-09    18.326    -3.039     0.000
--
./data/les_cases/ARM9707/lsf-2945- 13136.476   165.000   0.16728E-04   0.22031E-09    13.493    -2.326     0.000
./data/les_cases/ARM9707/lsf-2946- 15384.560   115.000   0.12543E-05   0.24479E-10     6.523    -1.860     0.000
./data/les_cases/ARM9707/lsf-2947-    189.3540              18   969.5034       day,levels, pres0
./data/les_cases/ARM9707/lsf:2948:    41.041   965.000   0.21079E-04   0.19251E-07     1.589     6.695     0.000
./data/les_cases/ARM9707/lsf-2949-   507.699   915.000   0.56206E-04   0.24817E-07     6.325     9.398     0.000
./data/les_cases/ARM9707/lsf-2950-   996.697   865.000   0.79984E-04   0.13042E-07     9.450     8.418     0.000
./data/les_cases/ARM9707/lsf-2951-  1510.833   815.000   0.74893E-04   0.98097E-08     9.578     4.624     0.000
--
./data/les_cases/ARM9707/lsf-3266- 10113.098   265.000  -0.10932E-04   0.18217E-08    12.605    -0.115     0.000
./data/les_cases/ARM9707/lsf-3267- 11499.005   215.000  -0.69073E-05   0.54227E-09    13.697    -0.152     0.000
./data/les_cases/ARM9707/lsf-3268- 13183.059   165.000  -0.12874E-04   0.10488E-10    11.318     0.003     0.000
./data/les_cases/ARM9707/lsf:3269: 15420.564   115.000  -0.10393E-04  -0.14650E-10     3.595    -0.200     0.000
./data/les_cases/ARM9707/lsf-3270-    191.4790              18   970.1393       day,levels, pres0
./data/les_cases/ARM9707/lsf-3271-    46.725   965.000  -0.10197E-04   0.34842E-09     1.404     5.073     0.000
./data/les_cases/ARM9707/lsf-3272-   513.311   915.000  -0.16337E-04   0.31401E-09     3.629     7.857     0.000
--
./data/les_cases/ARM9707/lsf-3309-    48.853   965.000  -0.34246E-04  -0.79887E-08    -1.108     4.356     0.000
./data/les_cases/ARM9707/lsf-3310-   517.991   915.000  -0.23849E-04  -0.78387E-08     0.647     6.695     0.000
./data/les_cases/ARM9707/lsf-3311-  1008.135   865.000  -0.14154E-04   0.13310E-07     1.942     7.070     0.000
./data/les_cases/ARM9707/lsf:3312:  1520.589   815.000  -0.17148E-04   0.55123E-07     2.508     6.869     0.000
./data/les_cases/ARM9707/lsf-3313-  2057.723   765.000  -0.35628E-04   0.89956E-07     2.898     6.358     0.000
./data/les_cases/ARM9707/lsf-3314-  2622.891   715.000  -0.56215E-04   0.93281E-07     3.490     5.762     0.000
./data/les_cases/ARM9707/lsf:3315:  3220.500   665.000  -0.69830E-04   0.70161E-07     3.830     5.059     0.000
./data/les_cases/ARM9707/lsf-3316-  3855.753   615.000  -0.71686E-04   0.41178E-07     3.820     4.705     0.000
./data/les_cases/ARM9707/lsf-3317-  4534.873   565.000  -0.75526E-04   0.27190E-07     4.012     4.565     0.000
./data/les_cases/ARM9707/lsf-3318-  5265.649   515.000  -0.78347E-04   0.18160E-07     4.729     4.418     0.000
--
./data/les_cases/ARM9707/lsf-3551- 10177.257   265.000   0.17152E-05  -0.15756E-08     4.977     9.761     0.000
./data/les_cases/ARM9707/lsf-3552- 11572.461   215.000   0.31187E-06  -0.58830E-09     4.775     9.898     0.000
./data/les_cases/ARM9707/lsf-3553- 13256.791   165.000  -0.80239E-05  -0.17461E-09     3.297     7.912     0.000
./data/les_cases/ARM9707/lsf:3554: 15481.975   115.000  -0.58005E-05  -0.23485E-10    -1.106     4.003     0.000
./data/les_cases/ARM9707/lsf-3555-    193.3540              18   969.8561       day,levels, pres0
./data/les_cases/ARM9707/lsf-3556-    44.122   965.000   0.44440E-05  -0.11063E-07     2.002     9.748     0.000
./data/les_cases/ARM9707/lsf-3557-   510.593   915.000  -0.26910E-05   0.71432E-08     5.101    14.158     0.000
--
./data/les_cases/ARM9707/lsf-4010- 15434.975   115.000   0.12603E-04   0.27647E-10     3.155    -2.423     0.000
./data/les_cases/ARM9707/lsf-4011-    196.3540              18   972.7925       day,levels, pres0
./data/les_cases/ARM9707/lsf-4012-    71.257   965.000  -0.12035E-05   0.11179E-07    -2.526     1.426     0.000
./data/les_cases/ARM9707/lsf:4013:   541.058   915.000   0.81880E-05   0.30918E-07    -1.767     3.070     0.000
./data/les_cases/ARM9707/lsf-4014-  1033.456   865.000   0.32506E-05   0.27374E-07    -0.073     2.993     0.000
./data/les_cases/ARM9707/lsf-4015-  1549.661   815.000  -0.16740E-06   0.11312E-07     1.041     1.752     0.000
./data/les_cases/ARM9707/lsf-4016-  2091.185   765.000  -0.14163E-04   0.30453E-07     1.416     0.481     0.000
--
./packages/core/legoesm/parallel/device_config.py-77-    "a100": 40.0,
./packages/core/legoesm/parallel/device_config.py-78-    "a100-80": 80.0,
./packages/core/legoesm/parallel/device_config.py-79-    "h100": 80.0,
./packages/core/legoesm/parallel/device_config.py:80:    "h200": 141.0,
./packages/core/legoesm/parallel/device_config.py-81-    "v100": 16.0,
./packages/core/legoesm/parallel/device_config.py-82-    "t4": 16.0,
./packages/core/legoesm/parallel/device_config.py-83-    "l4": 24.0,
--
./data/les_cases/ARM9707/sfc-72-   178.729    302.65        81.271       271.236     0.000
./data/les_cases/ARM9707/sfc-73-   178.854    303.91        63.405       269.895     0.000
./data/les_cases/ARM9707/sfc-74-   178.979    302.13        32.639       174.997     0.000
./data/les_cases/ARM9707/sfc:75:   179.104    299.16        14.174        81.964     0.000
./data/les_cases/ARM9707/sfc-76-   179.229    296.60        13.231        35.050     0.000
./data/les_cases/ARM9707/sfc-77-   179.354    295.45        12.106        24.001     0.000
./data/les_cases/ARM9707/sfc-78-   179.479    296.17        22.630        49.696     0.000
--
./data/les_cases/ARM9707/sfc-95-   181.604    297.43        64.271       150.865     0.000
./data/les_cases/ARM9707/sfc-96-   181.729    301.73        88.213       260.596     0.000
./data/les_cases/ARM9707/sfc-97-   181.854    304.52        82.504       283.591     0.000
./data/les_cases/ARM9707/sfc:98:   181.979    304.15        39.525       193.141     0.000
./data/les_cases/ARM9707/sfc-99-   182.104    302.14      -0.80039        82.104     0.000
./data/les_cases/ARM9707/sfc-100-   182.229    299.61       -24.587        19.198     0.000
./data/les_cases/ARM9707/sfc-101-   182.354    298.14       -28.339         9.564     0.000
--
./data/les_cases/ARM9707/sfc-166-   190.479    296.26        15.999        50.196     0.000
./data/les_cases/ARM9707/sfc-167-   190.604    298.91        64.168       145.519     0.000
./data/les_cases/ARM9707/sfc-168-   190.729    302.84        99.236       235.966     0.000
./data/les_cases/ARM9707/sfc:169:   190.854    305.23        85.048       241.046     0.000
./data/les_cases/ARM9707/sfc-170-   190.979    304.17        37.843       161.047     0.000
./data/les_cases/ARM9707/sfc-171-   191.104    301.53       -2.8682        72.849     0.000
./data/les_cases/ARM9707/sfc-172-   191.229    298.43       -19.885        30.577     0.000
--
./data/les_cases/ARM9707/rad-652-    648.760       0.992175E-05
./data/les_cases/ARM9707/rad-653-    605.305       0.923667E-05
./data/les_cases/ARM9707/rad-654-    562.485       0.783845E-05
./data/les_cases/ARM9707/rad:655:    520.524       0.922620E-05
./data/les_cases/ARM9707/rad-656-    479.591       0.100775E-04
./data/les_cases/ARM9707/rad-657-    439.819       0.125019E-04
./data/les_cases/ARM9707/rad-658-    401.326       0.121473E-04
--
./data/les_cases/ARM9707/rad-2309-    925.667      -0.517122E-05
./data/les_cases/ARM9707/rad-2310-    895.406      -0.602701E-05
./data/les_cases/ARM9707/rad-2311-    859.841      -0.612660E-05
./data/les_cases/ARM9707/rad:2312:    820.597      -0.670542E-05
./data/les_cases/ARM9707/rad-2313-    778.953      -0.104638E-04
./data/les_cases/ARM9707/rad-2314-    735.903      -0.144898E-04
./data/les_cases/ARM9707/rad-2315-    692.203      -0.154081E-04
--
./data/les_cases/ARM9707/rad-2348-    648.760      -0.272645E-04
./data/les_cases/ARM9707/rad-2349-    605.304      -0.214954E-04
./data/les_cases/ARM9707/rad-2350-    562.485      -0.193643E-04
./data/les_cases/ARM9707/rad:2351:    520.524      -0.187118E-04
./data/les_cases/ARM9707/rad-2352-    479.591      -0.159490E-04
./data/les_cases/ARM9707/rad-2353-    439.819      -0.145592E-04
./data/les_cases/ARM9707/rad-2354-    401.326      -0.143085E-04
--
./data/les_cases/ARM9707/rad-2509-    607.616      -0.299395E-04
./data/les_cases/ARM9707/rad-2510-    564.797      -0.282557E-04
./data/les_cases/ARM9707/rad-2511-    522.836      -0.268134E-04
./data/les_cases/ARM9707/rad:2512:    481.902      -0.238228E-04
./data/les_cases/ARM9707/rad-2513-    442.130      -0.220927E-04
./data/les_cases/ARM9707/rad-2514-    403.638      -0.223480E-04
./data/les_cases/ARM9707/rad-2515-    366.535      -0.214103E-04
--
./data/les_cases/ARM9707/rad-2742-    267.613      -0.680565E-05
./data/les_cases/ARM9707/rad-2743-    237.172      -0.297118E-05
./data/les_cases/ARM9707/rad-2744-    208.625       0.175862E-05
./data/les_cases/ARM9707/rad:2745:    181.999       0.537477E-05
./data/les_cases/ARM9707/rad-2746-    157.252       0.931738E-05
./data/les_cases/ARM9707/rad-2747-    134.244       0.118194E-04
./data/les_cases/ARM9707/rad-2748-    112.703       0.135953E-04
--
./data/les_cases/ARM9707/rad-3030-    267.519       0.111345E-04
./data/les_cases/ARM9707/rad-3031-    237.078       0.165866E-04
./data/les_cases/ARM9707/rad-3032-    208.531       0.176065E-04
./data/les_cases/ARM9707/rad:3033:    181.905       0.207284E-04
./data/les_cases/ARM9707/rad-3034-    157.158       0.259447E-04
./data/les_cases/ARM9707/rad-3035-    134.150       0.209656E-04
./data/les_cases/ARM9707/rad-3036-    112.609       0.206172E-04
--
./data/les_cases/ARM9707/rad-3239-    864.942      -0.290261E-04
./data/les_cases/ARM9707/rad-3240-    825.698      -0.288810E-04
./data/les_cases/ARM9707/rad-3241-    784.054      -0.277687E-04
./data/les_cases/ARM9707/rad:3242:    741.004      -0.295210E-04
./data/les_cases/ARM9707/rad-3243-    697.304      -0.290551E-04
./data/les_cases/ARM9707/rad-3244-    653.518      -0.301515E-04
./data/les_cases/ARM9707/rad-3245-    610.062      -0.249298E-04
--
./data/les_cases/ARM9707/rad-3414-    267.551      -0.116691E-04
./data/les_cases/ARM9707/rad-3415-    237.110      -0.625598E-05
./data/les_cases/ARM9707/rad-3416-    208.563      -0.122050E-05
./data/les_cases/ARM9707/rad:3417:    181.937       0.267069E-05
./data/les_cases/ARM9707/rad-3418-    157.190       0.762718E-05
./data/les_cases/ARM9707/rad-3419-    134.182       0.104591E-04
./data/les_cases/ARM9707/rad-3420-    112.641       0.116820E-04
--
./data/les_cases/ARM9707/rad-3542-    267.578      -0.318898E-05
./data/les_cases/ARM9707/rad-3543-    237.137      -0.990155E-07
./data/les_cases/ARM9707/rad-3544-    208.590       0.430149E-05
./data/les_cases/ARM9707/rad:3545:    181.964       0.741450E-05
./data/les_cases/ARM9707/rad-3546-    157.217       0.107873E-04
./data/les_cases/ARM9707/rad-3547-    134.209       0.129942E-04
./data/les_cases/ARM9707/rad-3548-    112.668       0.150130E-04
--
./data/les_cases/ARM9707/rad-3853-    607.706      -0.591957E-05
./data/les_cases/ARM9707/rad-3854-    564.887      -0.702651E-05
./data/les_cases/ARM9707/rad-3855-    522.926      -0.723073E-05
./data/les_cases/ARM9707/rad:3856:    481.992      -0.505495E-05
./data/les_cases/ARM9707/rad-3857-    442.220      -0.239989E-05
./data/les_cases/ARM9707/rad-3858-    403.728      -0.171788E-05
./data/les_cases/ARM9707/rad-3859-    366.625      -0.334984E-05
--
./data/les_cases/ARM9707/rad-5590-    267.596      -0.147840E-04
./data/les_cases/ARM9707/rad-5591-    237.155      -0.113387E-04
./data/les_cases/ARM9707/rad-5592-    208.608      -0.714148E-05
./data/les_cases/ARM9707/rad:5593:    181.982      -0.128787E-05
./data/les_cases/ARM9707/rad-5594-    157.235       0.290112E-05
./data/les_cases/ARM9707/rad-5595-    134.227       0.587965E-05
./data/les_cases/ARM9707/rad-5596-    112.686       0.692960E-05
--
./data/les_cases/ARM9707/rad-6710-    267.563       0.123273E-05
./data/les_cases/ARM9707/rad-6711-    237.122       0.478705E-05
./data/les_cases/ARM9707/rad-6712-    208.576       0.731151E-05
./data/les_cases/ARM9707/rad:6713:    181.949       0.122005E-04
./data/les_cases/ARM9707/rad-6714-    157.202       0.150230E-04
./data/les_cases/ARM9707/rad-6715-    134.194       0.135030E-04
./data/les_cases/ARM9707/rad-6716-    112.653       0.160325E-04
--
./data/les_cases/ARM9707/rad-6727-    865.032       0.137901E-04
./data/les_cases/ARM9707/rad-6728-    825.788       0.136008E-04
./data/les_cases/ARM9707/rad-6729-    784.144       0.133342E-04
./data/les_cases/ARM9707/rad:6730:    741.094       0.130099E-04
./data/les_cases/ARM9707/rad-6731-    697.394       0.114475E-04
./data/les_cases/ARM9707/rad-6732-    653.608       0.114078E-04
./data/les_cases/ARM9707/rad-6733-    610.152       0.988671E-05
--
./data/les_cases/ARM9707/rad-6950-    898.368      -0.119571E-04
./data/les_cases/ARM9707/rad-6951-    862.804      -0.119106E-04
./data/les_cases/ARM9707/rad-6952-    823.560      -0.129452E-04
./data/les_cases/ARM9707/rad:6953:    781.916      -0.164470E-04
./data/les_cases/ARM9707/rad-6954-    738.866      -0.219518E-04
./data/les_cases/ARM9707/rad-6955-    695.165      -0.204682E-04
./data/les_cases/ARM9707/rad-6956-    651.380      -0.234704E-04
--
./data/les_cases/ARM9707/rad-7014-    898.361      -0.191285E-04
./data/les_cases/ARM9707/rad-7015-    862.796      -0.187612E-04
./data/les_cases/ARM9707/rad-7016-    823.552      -0.190103E-04
./data/les_cases/ARM9707/rad:7017:    781.908      -0.219219E-04
./data/les_cases/ARM9707/rad-7018-    738.858      -0.277873E-04
./data/les_cases/ARM9707/rad-7019-    695.158      -0.254392E-04
./data/les_cases/ARM9707/rad-7020-    651.372      -0.294572E-04
--
./data/les_cases/ARM9707/rad-7046-    898.373      -0.311032E-04
./data/les_cases/ARM9707/rad-7047-    862.808      -0.272020E-04
./data/les_cases/ARM9707/rad-7048-    823.564      -0.270769E-04
./data/les_cases/ARM9707/rad:7049:    781.920      -0.258723E-04
./data/les_cases/ARM9707/rad-7050-    738.870      -0.277522E-04
./data/les_cases/ARM9707/rad-7051-    695.170      -0.272416E-04
./data/les_cases/ARM9707/rad-7052-    651.384      -0.284310E-04
--
./data/les_cases/ARM9707/rad-7078-    898.373      -0.321893E-04
./data/les_cases/ARM9707/rad-7079-    862.808      -0.284772E-04
./data/les_cases/ARM9707/rad-7080-    823.564      -0.283293E-04
./data/les_cases/ARM9707/rad:7081:    781.920      -0.271958E-04
./data/les_cases/ARM9707/rad-7082-    738.870      -0.289815E-04
./data/les_cases/ARM9707/rad-7083-    695.170      -0.285068E-04
./data/les_cases/ARM9707/rad-7084-    651.384      -0.296240E-04
--
./data/les_cases/ARM9707/rad-7110-    898.357      -0.333246E-04
./data/les_cases/ARM9707/rad-7111-    862.792      -0.297437E-04
./data/les_cases/ARM9707/rad-7112-    823.548      -0.296220E-04
./data/les_cases/ARM9707/rad:7113:    781.904      -0.285236E-04
./data/les_cases/ARM9707/rad-7114-    738.854      -0.302473E-04
./data/les_cases/ARM9707/rad-7115-    695.154      -0.297754E-04
./data/les_cases/ARM9707/rad-7116-    651.368      -0.308731E-04
--
./data/les_cases/ARM9707/rad-7245-    607.696      -0.252965E-04
./data/les_cases/ARM9707/rad-7246-    564.877      -0.237714E-04
./data/les_cases/ARM9707/rad-7247-    522.917      -0.220184E-04
./data/les_cases/ARM9707/rad:7248:    481.983      -0.208050E-04
./data/les_cases/ARM9707/rad-7249-    442.211      -0.200445E-04
./data/les_cases/ARM9707/rad-7250-    403.718      -0.174390E-04
./data/les_cases/ARM9707/rad-7251-    366.616      -0.167932E-04
--
./data/les_cases/ARM9707/rad-7277-    607.642      -0.225973E-04
./data/les_cases/ARM9707/rad-7278-    564.823      -0.209596E-04
./data/les_cases/ARM9707/rad-7279-    522.862      -0.190562E-04
./data/les_cases/ARM9707/rad:7280:    481.928      -0.177441E-04
./data/les_cases/ARM9707/rad-7281-    442.156      -0.169307E-04
./data/les_cases/ARM9707/rad-7282-    403.663      -0.141003E-04
./data/les_cases/ARM9707/rad-7283-    366.561      -0.134054E-04
--
./data/les_cases/ARM9707/rad-7309-    607.616      -0.208463E-04
./data/les_cases/ARM9707/rad-7310-    564.797      -0.191017E-04
./data/les_cases/ARM9707/rad-7311-    522.836      -0.171574E-04
./data/les_cases/ARM9707/rad:7312:    481.902      -0.158608E-04
./data/les_cases/ARM9707/rad-7313-    442.130      -0.150035E-04
./data/les_cases/ARM9707/rad-7314-    403.638      -0.122643E-04
./data/les_cases/ARM9707/rad-7315-    366.535      -0.115118E-04
--
./data/les_cases/ARM9707/rad-7341-    607.627      -0.140850E-04
./data/les_cases/ARM9707/rad-7342-    564.807      -0.140112E-04
./data/les_cases/ARM9707/rad-7343-    522.846      -0.144801E-04
./data/les_cases/ARM9707/rad:7344:    481.913      -0.131021E-04
./data/les_cases/ARM9707/rad-7345-    442.141      -0.117486E-04
./data/les_cases/ARM9707/rad-7346-    403.648      -0.101241E-04
./data/les_cases/ARM9707/rad-7347-    366.546      -0.842294E-05
--
./data/les_cases/ARM9707/rad-7373-    607.633      -0.819892E-05
./data/les_cases/ARM9707/rad-7374-    564.814      -0.856586E-05
./data/les_cases/ARM9707/rad-7375-    522.853      -0.939671E-05
./data/les_cases/ARM9707/rad:7376:    481.919      -0.872497E-05
./data/les_cases/ARM9707/rad-7377-    442.147      -0.779431E-05
./data/les_cases/ARM9707/rad-7378-    403.655      -0.657955E-05
./data/les_cases/ARM9707/rad-7379-    366.553      -0.520208E-05
--
./data/les_cases/ARM9707/rad-7845-    925.647      -0.420709E-04
./data/les_cases/ARM9707/rad-7846-    895.386      -0.360121E-04
./data/les_cases/ARM9707/rad-7847-    859.821      -0.328525E-04
./data/les_cases/ARM9707/rad:7848:    820.577      -0.327266E-04
./data/les_cases/ARM9707/rad-7849-    778.933      -0.317619E-04
./data/les_cases/ARM9707/rad-7850-    735.883      -0.332818E-04
./data/les_cases/ARM9707/rad-7851-    692.183      -0.328777E-04
--
./data/les_cases/ARM9707/rad-8229-    925.667       0.207107E-04
./data/les_cases/ARM9707/rad-8230-    895.406       0.972889E-05
./data/les_cases/ARM9707/rad-8231-    859.841       0.707053E-05
./data/les_cases/ARM9707/rad:8232:    820.597       0.698076E-05
./data/les_cases/ARM9707/rad-8233-    778.953       0.690301E-05
./data/les_cases/ARM9707/rad-8234-    735.903       0.660639E-05
./data/les_cases/ARM9707/rad-8235-    692.203       0.520072E-05
--
./data/les_cases/ARM9707/rad-8645-    925.590      -0.390407E-04
./data/les_cases/ARM9707/rad-8646-    895.329      -0.318468E-04
./data/les_cases/ARM9707/rad-8647-    859.764      -0.280502E-04
./data/les_cases/ARM9707/rad:8648:    820.520      -0.279212E-04
./data/les_cases/ARM9707/rad-8649-    778.876      -0.267567E-04
./data/les_cases/ARM9707/rad-8650-    735.826      -0.285842E-04
./data/les_cases/ARM9707/rad-8651-    692.126      -0.280838E-04
--
./data/les_cases/ARM9707/rad-9183-    46.1015       0.148222E-04
./data/les_cases/ARM9707/rad-9184-    26.1015       0.214303E-04
./data/les_cases/ARM9707/rad-9185-    6.10150       0.799708E-04
./data/les_cases/ARM9707/rad:9186: 181.9375,  31  day,levels
./data/les_cases/ARM9707/rad-9187-    960.998       0.189688E-05
./data/les_cases/ARM9707/rad-9188-    947.911      -0.912337E-07
./data/les_cases/ARM9707/rad-9189-    924.986      -0.155303E-05
--
./data/les_cases/ARM9707/rad-9215-    45.8549       0.104296E-04
./data/les_cases/ARM9707/rad-9216-    25.8549       0.141672E-04
./data/les_cases/ARM9707/rad-9217-    5.85486       0.538116E-04
./data/les_cases/ARM9707/rad:9218: 181.979172,  31  day,levels
./data/les_cases/ARM9707/rad-9219-    960.795      -0.302702E-05
./data/les_cases/ARM9707/rad-9220-    947.707      -0.525875E-05
./data/les_cases/ARM9707/rad-9221-    924.782      -0.680368E-05
--
./data/les_cases/ARM9707/rad-10509-    607.709      -0.169196E-05
./data/les_cases/ARM9707/rad-10510-    564.890      -0.286931E-05
./data/les_cases/ARM9707/rad-10511-    522.929      -0.223190E-05
./data/les_cases/ARM9707/rad:10512:    481.995      -0.265178E-05
./data/les_cases/ARM9707/rad-10513-    442.223      -0.167124E-05
./data/les_cases/ARM9707/rad-10514-    403.730      -0.260579E-05
./data/les_cases/ARM9707/rad-10515-    366.628      -0.382314E-05
--
./data/les_cases/ARM9707/rad-10534-    898.365       0.828354E-05
./data/les_cases/ARM9707/rad-10535-    862.800       0.568202E-05
./data/les_cases/ARM9707/rad-10536-    823.556       0.559418E-05
./data/les_cases/ARM9707/rad:10537:    781.912       0.551809E-05
./data/les_cases/ARM9707/rad-10538-    738.862       0.522781E-05
./data/les_cases/ARM9707/rad-10539-    695.162       0.385219E-05
./data/les_cases/ARM9707/rad-10540-    651.376       0.395042E-05
--
./data/les_cases/ARM9707/rad-10566-    898.430       0.132933E-04
./data/les_cases/ARM9707/rad-10567-    862.865       0.105031E-04
./data/les_cases/ARM9707/rad-10568-    823.621       0.103226E-04
./data/les_cases/ARM9707/rad:10569:    781.977       0.100682E-04
./data/les_cases/ARM9707/rad-10570-    738.927       0.975875E-05
./data/les_cases/ARM9707/rad-10571-    695.227       0.826802E-05
./data/les_cases/ARM9707/rad-10572-    651.441       0.823010E-05
--
./data/les_cases/ARM9707/rad-10637-    607.706       0.436887E-05
./data/les_cases/ARM9707/rad-10638-    564.888       0.307207E-05
./data/les_cases/ARM9707/rad-10639-    522.927       0.435916E-05
./data/les_cases/ARM9707/rad:10640:    481.993       0.514868E-05
./data/les_cases/ARM9707/rad-10641-    442.221       0.739721E-05
./data/les_cases/ARM9707/rad-10642-    403.728       0.706838E-05
./data/les_cases/ARM9707/rad-10643-    366.626       0.521456E-05
--
./data/les_cases/ARM9707/rad-11246-    563.691      -0.511882E-05
./data/les_cases/ARM9707/rad-11247-    521.730      -0.619628E-05
./data/les_cases/ARM9707/rad-11248-    480.796      -0.617665E-05
./data/les_cases/ARM9707/rad:11249:    441.024      -0.558287E-05
./data/les_cases/ARM9707/rad-11250-    402.532      -0.475805E-05
./data/les_cases/ARM9707/rad-11251-    365.430      -0.366849E-05
./data/les_cases/ARM9707/rad-11252-    329.830      -0.416528E-05
--
./data/les_cases/ARM9707/rad-12756-    337.776      -0.577463E-05
./data/les_cases/ARM9707/rad-12757-    303.790      -0.590967E-05
./data/les_cases/ARM9707/rad-12758-    271.528      -0.518768E-05
./data/les_cases/ARM9707/rad:12759:    241.087      -0.221065E-05
./data/les_cases/ARM9707/rad-12760-    212.540       0.202900E-05
./data/les_cases/ARM9707/rad-12761-    185.914       0.502823E-05
./data/les_cases/ARM9707/rad-12762-    161.167       0.827778E-05
--
./data/les_cases/ARM9707/rad-14038-    267.548      -0.164897E-04
./data/les_cases/ARM9707/rad-14039-    237.107      -0.131809E-04
./data/les_cases/ARM9707/rad-14040-    208.560      -0.914982E-05
./data/les_cases/ARM9707/rad:14041:    181.934      -0.352795E-05
./data/les_cases/ARM9707/rad-14042-    157.187       0.495193E-06
./data/les_cases/ARM9707/rad-14043-    134.179       0.335581E-05
./data/les_cases/ARM9707/rad-14044-    112.638       0.436419E-05
--
./data/les_cases/ARM9707/rad-15734-    267.594      -0.925190E-05
./data/les_cases/ARM9707/rad-15735-    237.153      -0.355016E-05
./data/les_cases/ARM9707/rad-15736-    208.606       0.174947E-05
./data/les_cases/ARM9707/rad:15737:    181.980       0.584467E-05
./data/les_cases/ARM9707/rad-15738-    157.233       0.110518E-04
./data/les_cases/ARM9707/rad-15739-    134.225       0.140413E-04
./data/les_cases/ARM9707/rad-15740-    112.684       0.153110E-04
--
./data/les_cases/ARM9707/rad-16583-    865.010      -0.163135E-04
./data/les_cases/ARM9707/rad-16584-    825.766      -0.139928E-04
./data/les_cases/ARM9707/rad-16585-    784.122      -0.135609E-04
./data/les_cases/ARM9707/rad:16586:    741.072      -0.133637E-04
./data/les_cases/ARM9707/rad-16587-    697.372      -0.111479E-04
./data/les_cases/ARM9707/rad-16588-    653.586      -0.103756E-04
./data/les_cases/ARM9707/rad-16589-    610.130      -0.100564E-04
--
./data/les_cases/ARM9707/rad-17255-    864.976      -0.289482E-04
./data/les_cases/ARM9707/rad-17256-    825.732      -0.270972E-04
./data/les_cases/ARM9707/rad-17257-    784.088      -0.280462E-04
./data/les_cases/ARM9707/rad:17258:    741.038      -0.260415E-04
./data/les_cases/ARM9707/rad-17259-    697.337      -0.241492E-04
./data/les_cases/ARM9707/rad-17260-    653.552      -0.234038E-04
./data/les_cases/ARM9707/rad-17261-    610.096      -0.215368E-04
--
./data/les_cases/ARM9707/rad-18438-    898.449      -0.578992E-05
./data/les_cases/ARM9707/rad-18439-    862.884      -0.588996E-05
./data/les_cases/ARM9707/rad-18440-    823.640      -0.647143E-05
./data/les_cases/ARM9707/rad:18441:    781.996      -0.102470E-04
./data/les_cases/ARM9707/rad-18442-    738.946      -0.142914E-04
./data/les_cases/ARM9707/rad-18443-    695.246      -0.152139E-04
./data/les_cases/ARM9707/rad-18444-    651.460      -0.132080E-04
--
./data/les_cases/ARM9707/rad-18470-    898.357      -0.170847E-04
./data/les_cases/ARM9707/rad-18471-    862.792      -0.170434E-04
./data/les_cases/ARM9707/rad-18472-    823.549      -0.179624E-04
./data/les_cases/ARM9707/rad:18473:    781.905      -0.210732E-04
./data/les_cases/ARM9707/rad-18474-    738.855      -0.259632E-04
./data/les_cases/ARM9707/rad-18475-    695.154      -0.246453E-04
./data/les_cases/ARM9707/rad-18476-    651.368      -0.273122E-04
--
./data/les_cases/ARM9707/rad-18598-    898.373      -0.365845E-04
./data/les_cases/ARM9707/rad-18599-    862.808      -0.335077E-04
./data/les_cases/ARM9707/rad-18600-    823.564      -0.333851E-04
./data/les_cases/ARM9707/rad:18601:    781.920      -0.324456E-04
./data/les_cases/ARM9707/rad-18602-    738.870      -0.339257E-04
./data/les_cases/ARM9707/rad-18603-    695.170      -0.335322E-04
./data/les_cases/ARM9707/rad-18604-    651.384      -0.344582E-04
--
./data/les_cases/ARM9707/rad-18630-    898.406      -0.375812E-04
./data/les_cases/ARM9707/rad-18631-    862.841      -0.346212E-04
./data/les_cases/ARM9707/rad-18632-    823.597      -0.345206E-04
./data/les_cases/ARM9707/rad:18633:    781.953      -0.336127E-04
./data/les_cases/ARM9707/rad-18634-    738.903      -0.350375E-04
./data/les_cases/ARM9707/rad-18635-    695.203      -0.346474E-04
./data/les_cases/ARM9707/rad-18636-    651.417      -0.355548E-04
--
./data/les_cases/ARM9707/rad-18662-    898.438      -0.399824E-04
./data/les_cases/ARM9707/rad-18663-    862.873      -0.366711E-04
./data/les_cases/ARM9707/rad-18664-    823.629      -0.371886E-04
./data/les_cases/ARM9707/rad:18665:    781.985      -0.351784E-04
./data/les_cases/ARM9707/rad-18666-    738.935      -0.354079E-04
./data/les_cases/ARM9707/rad-18667-    695.235      -0.346881E-04
./data/les_cases/ARM9707/rad-18668-    651.449      -0.331774E-04
--
./data/les_cases/ARM9707/rad-18790-    898.445      -0.373895E-04
./data/les_cases/ARM9707/rad-18791-    862.880      -0.329860E-04
./data/les_cases/ARM9707/rad-18792-    823.636      -0.313935E-04
./data/les_cases/ARM9707/rad:18793:    781.992      -0.322099E-04
./data/les_cases/ARM9707/rad-18794-    738.942      -0.304851E-04
./data/les_cases/ARM9707/rad-18795-    695.242      -0.288571E-04
./data/les_cases/ARM9707/rad-18796-    651.456      -0.282157E-04
--
./data/les_cases/ARM9707/rad-18822-    898.435      -0.343831E-04
./data/les_cases/ARM9707/rad-18823-    862.870      -0.299602E-04
./data/les_cases/ARM9707/rad-18824-    823.626      -0.284590E-04
./data/les_cases/ARM9707/rad:18825:    781.982      -0.294480E-04
./data/les_cases/ARM9707/rad-18826-    738.932      -0.278934E-04
./data/les_cases/ARM9707/rad-18827-    695.232      -0.264400E-04
./data/les_cases/ARM9707/rad-18828-    651.446      -0.258742E-04
--
./data/les_cases/ARM9707/rad-18854-    898.439      -0.287332E-04
./data/les_cases/ARM9707/rad-18855-    862.875      -0.237841E-04
./data/les_cases/ARM9707/rad-18856-    823.630      -0.212115E-04
./data/les_cases/ARM9707/rad:18857:    781.987      -0.208129E-04
./data/les_cases/ARM9707/rad-18858-    738.937      -0.209151E-04
./data/les_cases/ARM9707/rad-18859-    695.236      -0.186904E-04
./data/les_cases/ARM9707/rad-18860-    651.450      -0.178483E-04
--
./data/les_cases/ARM9707/rad-18886-    898.438      -0.211492E-04
./data/les_cases/ARM9707/rad-18887-    862.873      -0.170170E-04
./data/les_cases/ARM9707/rad-18888-    823.629      -0.147355E-04
./data/les_cases/ARM9707/rad:18889:    781.985      -0.143110E-04
./data/les_cases/ARM9707/rad-18890-    738.935      -0.141171E-04
./data/les_cases/ARM9707/rad-18891-    695.235      -0.119388E-04
./data/les_cases/ARM9707/rad-18892-    651.449      -0.111796E-04
--
./data/les_cases/ARM9707/rad-18918-    898.404      -0.139878E-04
./data/les_cases/ARM9707/rad-18919-    862.839      -0.105112E-04
./data/les_cases/ARM9707/rad-18920-    823.595      -0.849812E-05
./data/les_cases/ARM9707/rad:18921:    781.951      -0.800148E-05
./data/les_cases/ARM9707/rad-18922-    738.901      -0.757887E-05
./data/les_cases/ARM9707/rad-18923-    695.201      -0.538594E-05
./data/les_cases/ARM9707/rad-18924-    651.415      -0.472137E-05
--
./data/les_cases/ARM9707/rad-19278-    563.696      -0.215290E-04
./data/les_cases/ARM9707/rad-19279-    521.735      -0.205621E-04
./data/les_cases/ARM9707/rad-19280-    480.801      -0.164448E-04
./data/les_cases/ARM9707/rad:19281:    441.029      -0.146417E-04
./data/les_cases/ARM9707/rad-19282-    402.537      -0.145394E-04
./data/les_cases/ARM9707/rad-19283-    365.435      -0.136920E-04
./data/les_cases/ARM9707/rad-19284-    329.835      -0.116355E-04
--
./packages/core/legoesm/core/weno.py-653-        fp2 = jnp.roll(f, -2, axis=axis)
./packages/core/legoesm/core/weno.py-654-        fm3 = jnp.roll(f, 3, axis=axis)
./packages/core/legoesm/core/weno.py-655-        fp3 = jnp.roll(f, -3, axis=axis)
./packages/core/legoesm/core/weno.py:656:        return ((215641.0 / 241920.0) * f
./packages/core/legoesm/core/weno.py-657-                + (6361.0 / 107520.0) * (fm1 + fp1)
./packages/core/legoesm/core/weno.py-658-                - (281.0 / 53760.0) * (fm2 + fp2)
./packages/core/legoesm/core/weno.py-659-                + (367.0 / 967680.0) * (fm3 + fp3))
--
./data/les_cases/TOGA/lsf-1283-  -999.000   200.000   0.28620E-04  -0.88334E-10  -11.6515    2.0381    0.0232
./data/les_cases/TOGA/lsf-1284-  -999.000   175.000   0.26305E-05  -0.15819E-09  -14.3227    2.3932    0.0112
./data/les_cases/TOGA/lsf-1285-  -999.000   150.000  -0.60562E-05  -0.44793E-10  -17.1294    2.3461   -0.0031
./data/les_cases/TOGA/lsf:1286:  -999.000   125.000   0.14428E-05  -0.25787E-11  -20.5010    0.5188   -0.0114
./data/les_cases/TOGA/lsf-1287-  -999.000   100.000  -0.56925E-05  -0.42730E-12  -20.1669    0.5793   -0.0077
./data/les_cases/TOGA/lsf-1288-  -999.000    75.000    0.0000        0.0000       -5.3221    1.2521    0.0000
./data/les_cases/TOGA/lsf-1289- 362.,  38,  1007.09369
--
./data/les_cases/TOGA/lsf-1517-  -999.000   200.000  -0.24429E-05  -0.44737E-10   -5.3795    1.5569   -0.0056
./data/les_cases/TOGA/lsf-1518-  -999.000   175.000   0.52958E-05   0.15956E-09   -8.8434    2.4950   -0.0126
./data/les_cases/TOGA/lsf-1519-  -999.000   150.000   0.15458E-04   0.79419E-10  -13.2340    4.0403   -0.0143
./data/les_cases/TOGA/lsf:1520:  -999.000   125.000   0.15070E-04   0.54142E-11  -20.5281    5.1000   -0.0105
./data/les_cases/TOGA/lsf-1521-  -999.000   100.000   0.45861E-05  -0.12620E-12  -18.8998    0.8755   -0.0045
./data/les_cases/TOGA/lsf-1522-  -999.000    75.000    0.0000        0.0000       -4.1138   -2.1844    0.0000
./data/les_cases/TOGA/lsf-1523- 363.5,  38,  1007.16992
--
./data/les_cases/TOGA/lsf-2314-  -999.000   750.000  -0.19586E-04  -0.58837E-08   18.2743    0.5664   -0.0055
./data/les_cases/TOGA/lsf-2315-  -999.000   725.000  -0.19219E-04   0.60030E-09   18.7840    0.9311   -0.0051
./data/les_cases/TOGA/lsf-2316-  -999.000   700.000  -0.18101E-04  -0.12036E-07   19.1156    0.4653   -0.0047
./data/les_cases/TOGA/lsf:2317:  -999.000   675.000  -0.14113E-04   0.40664E-08   20.5641    0.8317   -0.0047
./data/les_cases/TOGA/lsf-2318-  -999.000   650.000  -0.64352E-05   0.68683E-08   20.3894    1.5162   -0.0050
./data/les_cases/TOGA/lsf-2319-  -999.000   625.000  -0.13664E-04   0.14394E-07   19.4837    1.9052   -0.0056
./data/les_cases/TOGA/lsf-2320-  -999.000   600.000  -0.10105E-04   0.54604E-08   19.5263    2.0893   -0.0062
--
./data/les_cases/TOGA/snd-55-  -999.000000   675.000000   316.010010     6.602194    -2.477360     2.202260
./data/les_cases/TOGA/snd-56-  -999.000000   650.000000   317.283081     5.974324    -4.010030     0.583906
./data/les_cases/TOGA/snd-57-  -999.000000   625.000000   318.951294     5.818423    -5.356420    -1.326130
./data/les_cases/TOGA/snd:58:  -999.000000   600.000000   320.532166     5.576106    -6.670140    -2.872650
./data/les_cases/TOGA/snd-59-  -999.000000   575.000000   321.880859     5.156816    -8.036410    -3.496050
./data/les_cases/TOGA/snd-60-  -999.000000   550.000000   324.041901     4.160646    -9.205650    -3.528520
./data/les_cases/TOGA/snd-61-  -999.000000   525.000000   326.718323     2.838579    -9.808200    -3.584900
--
./data/les_cases/TOGA/snd-796-  -999.000000   675.000000   315.203278     8.025856     9.632900     2.421540
./data/les_cases/TOGA/snd-797-  -999.000000   650.000000   317.054596     7.444191     8.980450     1.795250
./data/les_cases/TOGA/snd-798-  -999.000000   625.000000   318.900970     6.776553     7.644810     1.177080
./data/les_cases/TOGA/snd:799:  -999.000000   600.000000   320.539093     5.873714     5.760500     0.334134
./data/les_cases/TOGA/snd-800-  -999.000000   575.000000   322.412628     4.783368     3.169250    -0.472453
./data/les_cases/TOGA/snd-801-  -999.000000   550.000000   324.549652     4.119228     0.835018    -1.501840
./data/les_cases/TOGA/snd-802-  -999.000000   525.000000   326.273438     3.389917    -1.370900    -2.747570
--
./data/les_cases/TOGA/snd-835-  -999.000000   675.000000   315.303955     8.336493     7.624240     2.863990
./data/les_cases/TOGA/snd-836-  -999.000000   650.000000   317.047791     7.673318     7.314840     2.893590
./data/les_cases/TOGA/snd-837-  -999.000000   625.000000   318.730560     6.896424     6.209900     2.048570
./data/les_cases/TOGA/snd:838:  -999.000000   600.000000   320.510162     5.962461     4.504930     1.055470
./data/les_cases/TOGA/snd-839-  -999.000000   575.000000   322.519226     4.748746     2.445170     0.240076
./data/les_cases/TOGA/snd-840-  -999.000000   550.000000   324.811829     4.060318    -0.148111    -1.219900
./data/les_cases/TOGA/snd-841-  -999.000000   525.000000   326.429749     3.187868    -3.237210    -3.033230
--
./data/les_cases/TOGA/snd-874-  -999.000000   675.000000   315.343140     8.261939     8.498760     1.062080
./data/les_cases/TOGA/snd-875-  -999.000000   650.000000   316.948303     7.492365     7.611340     1.411640
./data/les_cases/TOGA/snd-876-  -999.000000   625.000000   318.866638     6.879334     6.876490     1.352540
./data/les_cases/TOGA/snd:877:  -999.000000   600.000000   320.593475     6.410060     5.423610     0.948934
./data/les_cases/TOGA/snd-878-  -999.000000   575.000000   322.467682     5.614388     3.873870    -0.260560
./data/les_cases/TOGA/snd-879-  -999.000000   550.000000   324.550812     4.817226     1.507520    -0.934396
./data/les_cases/TOGA/snd-880-  -999.000000   525.000000   326.429749     3.480176    -1.015560    -2.145720
--
./data/les_cases/TOGA/snd-1080-  -999.000000   400.000000   335.065216     2.422656     1.801480     3.742000
./data/les_cases/TOGA/snd-1081-  -999.000000   375.000000   337.175720     1.955799     2.313430     4.157880
./data/les_cases/TOGA/snd-1082-  -999.000000   350.000000   339.182770     1.496535     2.704810     4.696090
./data/les_cases/TOGA/snd:1083:  -999.000000   325.000000   341.096008     1.089514     2.294090     4.906670
./data/les_cases/TOGA/snd-1084-  -999.000000   300.000000   342.726501     0.764662     1.920030     5.162970
./data/les_cases/TOGA/snd-1085-  -999.000000   275.000000   344.236176     0.486548     1.882980     5.803450
./data/les_cases/TOGA/snd-1086-  -999.000000   250.000000   345.521729     0.281210     0.860812     6.527910
--
./data/les_cases/TOGA/snd-1236-  -999.000000   400.000000   335.492706     1.992183     6.192840     5.726880
./data/les_cases/TOGA/snd-1237-  -999.000000   375.000000   337.251160     1.640365     4.573410     4.795920
./data/les_cases/TOGA/snd-1238-  -999.000000   350.000000   338.992401     1.284735     2.849630     3.983700
./data/les_cases/TOGA/snd:1239:  -999.000000   325.000000   341.039490     0.890943     0.686806     3.687910
./data/les_cases/TOGA/snd-1240-  -999.000000   300.000000   342.857697     0.613103    -1.657610     3.875270
./data/les_cases/TOGA/snd-1241-  -999.000000   275.000000   344.348969     0.421807    -4.195300     4.170920
./data/les_cases/TOGA/snd-1242-  -999.000000   250.000000   345.674805     0.253516    -6.658720     4.035020
--
./data/les_cases/TOGA/snd-1275-  -999.000000   400.000000   335.113281     2.041161     3.537080     6.581560
./data/les_cases/TOGA/snd-1276-  -999.000000   375.000000   337.317352     1.740726     2.006520     5.928990
./data/les_cases/TOGA/snd-1277-  -999.000000   350.000000   339.278625     1.361720     1.217470     5.595150
./data/les_cases/TOGA/snd:1278:  -999.000000   325.000000   341.067047     0.969755     0.248613     5.212860
./data/les_cases/TOGA/snd-1279-  -999.000000   300.000000   342.709564     0.678110    -1.583360     4.616830
./data/les_cases/TOGA/snd-1280-  -999.000000   275.000000   344.252106     0.451053    -2.318750     4.185380
./data/les_cases/TOGA/snd-1281-  -999.000000   250.000000   345.585632     0.248318    -4.781030     3.704500
--
./data/les_cases/TOGA/snd-1283-  -999.000000   200.000000   347.615570     0.054830   -11.651500     2.038070
./data/les_cases/TOGA/snd-1284-  -999.000000   175.000000   348.988068     0.023514   -14.322700     2.393210
./data/les_cases/TOGA/snd-1285-  -999.000000   150.000000   350.460144     0.008612   -17.129400     2.346110
./data/les_cases/TOGA/snd:1286:  -999.000000   125.000000   355.842712     0.003083   -20.500999     0.518827
./data/les_cases/TOGA/snd-1287-  -999.000000   100.000000   369.195038     0.001279   -20.166901     0.579342
./data/les_cases/TOGA/snd-1288-  -999.000000    75.000000   414.001495     0.007179    -5.322130     1.252050
./data/les_cases/TOGA/snd-1289- 362.,  38,  1007.09369
--
./data/les_cases/TOGA/snd-1517-  -999.000000   200.000000   349.193420     0.041322    -5.379530     1.556910
./data/les_cases/TOGA/snd-1518-  -999.000000   175.000000   350.780396     0.022267    -8.843380     2.494970
./data/les_cases/TOGA/snd-1519-  -999.000000   150.000000   352.868134     0.009292   -13.234000     4.040270
./data/les_cases/TOGA/snd:1520:  -999.000000   125.000000   357.069427     0.003013   -20.528099     5.100010
./data/les_cases/TOGA/snd-1521-  -999.000000   100.000000   372.990143     0.001709   -18.899799     0.875480
./data/les_cases/TOGA/snd-1522-  -999.000000    75.000000   413.368225     0.004122    -4.113780    -2.184350
./data/les_cases/TOGA/snd-1523- 363.5,  38,  1007.16992
--
./data/les_cases/TOGA/snd-1771-  -999.000000   675.000000   316.020081     6.441426    13.063300     4.367750
./data/les_cases/TOGA/snd-1772-  -999.000000   650.000000   317.389374     5.770089    12.085900     3.762780
./data/les_cases/TOGA/snd-1773-  -999.000000   625.000000   319.029083     5.215204    11.747500     3.657760
./data/les_cases/TOGA/snd:1774:  -999.000000   600.000000   320.593475     4.749952    12.338100     4.209410
./data/les_cases/TOGA/snd-1775-  -999.000000   575.000000   321.975708     4.353061    12.820100     4.409140
./data/les_cases/TOGA/snd-1776-  -999.000000   550.000000   323.632599     3.993867    12.205700     3.325900
./data/les_cases/TOGA/snd-1777-  -999.000000   525.000000   325.330902     3.515193    10.600300     2.162730
--
./data/les_cases/TOGA/snd-1860-  -999.000000   400.000000   334.253082     1.134559     4.138260     2.199500
./data/les_cases/TOGA/snd-1861-  -999.000000   375.000000   336.445099     1.167082    -0.445529     0.743690
./data/les_cases/TOGA/snd-1862-  -999.000000   350.000000   338.788574     0.956863    -2.038520    -0.769306
./data/les_cases/TOGA/snd:1863:  -999.000000   325.000000   341.020172     0.633860    -2.178300    -0.378745
./data/les_cases/TOGA/snd-1864-  -999.000000   300.000000   342.952209     0.419869    -3.332590     0.823656
./data/les_cases/TOGA/snd-1865-  -999.000000   275.000000   344.820496     0.225645    -5.748060     2.475280
./data/les_cases/TOGA/snd-1866-  -999.000000   250.000000   346.689941     0.130131    -8.231820     3.175030
--
./data/les_cases/TOGA/snd-1938-  -999.000000   400.000000   335.152252     1.231792     1.844460    -0.301163
./data/les_cases/TOGA/snd-1939-  -999.000000   375.000000   337.274994     1.139393    -0.459455    -0.695547
./data/les_cases/TOGA/snd-1940-  -999.000000   350.000000   339.338013     0.913772    -1.267910    -0.605478
./data/les_cases/TOGA/snd:1941:  -999.000000   325.000000   341.057404     0.664807    -0.848717     0.670172
./data/les_cases/TOGA/snd-1942-  -999.000000   300.000000   342.699677     0.467287     0.075097     2.096080
./data/les_cases/TOGA/snd-1943-  -999.000000   275.000000   344.610779     0.309999    -0.874683     3.831320
./data/les_cases/TOGA/snd-1944-  -999.000000   250.000000   346.104340     0.172856    -4.413410     4.128310
--
./data/les_cases/TOGA/snd-2279-  -999.000000   650.000000   316.801239     6.387422    17.786699    -0.757101
./data/les_cases/TOGA/snd-2280-  -999.000000   625.000000   317.905853     6.156203    17.941099    -0.168624
./data/les_cases/TOGA/snd-2281-  -999.000000   600.000000   319.028931     5.707921    18.087900     0.016040
./data/les_cases/TOGA/snd:2282:  -999.000000   575.000000   320.530212     5.042333    17.805300    -0.007193
./data/les_cases/TOGA/snd-2283-  -999.000000   550.000000   322.383331     4.214692    16.624901    -0.039037
./data/les_cases/TOGA/snd-2284-  -999.000000   525.000000   324.771851     3.180867    14.719400    -0.922233
./data/les_cases/TOGA/snd-2285-  -999.000000   500.000000   327.358154     2.395553    13.268100    -2.117500
--
./data/les_cases/TOGA/snd-2314-  -999.000000   750.000000   311.046661     7.889376    18.274300     0.566417
./data/les_cases/TOGA/snd-2315-  -999.000000   725.000000   312.539795     7.499120    18.784000     0.931057
./data/les_cases/TOGA/snd-2316-  -999.000000   700.000000   313.852478     7.149276    19.115601     0.465265
./data/les_cases/TOGA/snd:2317:  -999.000000   675.000000   315.213348     6.895981    20.564100     0.831715
./data/les_cases/TOGA/snd-2318-  -999.000000   650.000000   316.619141     6.571471    20.389400     1.516180
./data/les_cases/TOGA/snd-2319-  -999.000000   625.000000   317.870392     6.300162    19.483700     1.905150
./data/les_cases/TOGA/snd-2320-  -999.000000   600.000000   319.234924     5.678164    19.526300     2.089340
--
./data/les_cases/DYCOMS_RF02/grd_orig-56-  763.8
./data/les_cases/DYCOMS_RF02/grd_orig-57-  770.2
./data/les_cases/DYCOMS_RF02/grd_orig-58-  776.2
./data/les_cases/DYCOMS_RF02/grd_orig:59:  781.9
./data/les_cases/DYCOMS_RF02/grd_orig-60-  787.3
./data/les_cases/DYCOMS_RF02/grd_orig-61-  792.4
./data/les_cases/DYCOMS_RF02/grd_orig-62-  797.5
--
./scripts/plot/plot_scaling_paper_figure.py-35-                  "LL2048@64 26502539, LL2048@128 26534060",
./scripts/plot/plot_scaling_paper_figure.py-36-    "atm_cube": "26452894/26453782",
./scripts/plot/plot_scaling_paper_figure.py-37-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
./scripts/plot/plot_scaling_paper_figure.py:38:                "s8 np32-128 26549646/26538474, s9 26600095, "
./scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076",
./scripts/plot/plot_scaling_paper_figure.py-40-    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
./scripts/plot/plot_scaling_paper_figure.py-41-                   "lat-lon 2-D r512 26628073",
./scripts/plot/plot_scaling_paper_figure.py-42-    "oc_latlon": "26460444-501/26460365/26493592",
--
./scripts/plot/plot_scaling_paper_figure.py-66-                                        (16, 7.10), (32, 8.13), (64, 5.27),
./scripts/plot/plot_scaling_paper_figure.py-67-                                        (128, 6.47)]),
./scripts/plot/plot_scaling_paper_figure.py-68-                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
./scripts/plot/plot_scaling_paper_figure.py:69:                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
./scripts/plot/plot_scaling_paper_figure.py-70-                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
./scripts/plot/plot_scaling_paper_figure.py:71:        note="weak eff 0.53–0.67 at\nmatched tile (lloyd-0 pairs)",
./scripts/plot/plot_scaling_paper_figure.py-72-    ),
./scripts/plot/plot_scaling_paper_figure.py-73-    dict(
./scripts/plot/plot_scaling_paper_figure.py-74-        key="atm_ico_cpu", title="ico + lat-lon 2-D", sub="subdiv-7 / r512 L26 · Milan CPU–MPI",
--
./scripts/plot/plot_scaling_paper_figure.py-77-                             (64, 131.58)]),
./scripts/plot/plot_scaling_paper_figure.py-78-                ("float64", [(1, 9987.81), (2, 4727.46), (4, 2349.06),
./scripts/plot/plot_scaling_paper_figure.py-79-                             (8, 1161.98), (16, 624.06), (32, 350.89),
./scripts/plot/plot_scaling_paper_figure.py:80:                             (64, 220.57), (128, 92.4), (256, 56.2),
./scripts/plot/plot_scaling_paper_figure.py-81-                             (512, 66.7)]),
./scripts/plot/plot_scaling_paper_figure.py-82-                ("f64 lat-lon 2-D (r512)", [(64, 297.57), (128, 161.03),
./scripts/plot/plot_scaling_paper_figure.py-83-                                            (256, 72.06), (512, 44.78)])],
--
./scripts/plot/plot_scaling_paper_figure.py-115-          "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
./scripts/plot/plot_scaling_paper_figure.py-116-          "float32 (LL2048)": "#009E73",
./scripts/plot/plot_scaling_paper_figure.py-117-          "f64 lat-lon 2-D (r512)": "#CC79A7",
./scripts/plot/plot_scaling_paper_figure.py:118:          "f32 (s8 lloyd-0)": "#009E73",
./scripts/plot/plot_scaling_paper_figure.py-119-          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
./scripts/plot/plot_scaling_paper_figure.py-120-          "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
./scripts/plot/plot_scaling_paper_figure.py-121-MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
--
./scripts/plot/plot_scaling_paper_figure.py-124-           "float32 (C768)": "o", "float32 (C384)": "^",
./scripts/plot/plot_scaling_paper_figure.py-125-           "float32 (LL2048)": "^",
./scripts/plot/plot_scaling_paper_figure.py-126-           "f64 lat-lon 2-D (r512)": "D",
./scripts/plot/plot_scaling_paper_figure.py:127:           "f32 (s8 lloyd-0)": "v",
./scripts/plot/plot_scaling_paper_figure.py-128-           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
./scripts/plot/plot_scaling_paper_figure.py-129-           "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
./scripts/plot/plot_scaling_paper_figure.py-130-
--
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py-462-    221.1120,
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py-463-    28.9300,
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py-464-    117.1498,
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py:465:    320.5095,
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py-466-    262.3602,
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py-467-    336.2148,
./packages/land/legoesm/land/canopy/clm_ml_backend/clm_share/shr_orb_mod.py-468-    233.0046,
--
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py-516-        221.1120,
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py-517-        28.9300,
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py-518-        117.1498,
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py:519:        320.5095,
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py-520-        262.3602,
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py-521-        336.2148,
./packages/land/legoesm/land/canopy/clm_ml_backend/cime_src_share_util/shr_orb_mod.py-522-        233.0046,
--
./scripts/data/generate_amip_forcing.py-215-    1985: dict(co2=346.04, ch4=1645.0, n2o=304.0, cfc11=220.0, cfc12=400.0),
./scripts/data/generate_amip_forcing.py-216-    1990: dict(co2=354.39, ch4=1714.0, n2o=308.0, cfc11=259.0, cfc12=478.0),
./scripts/data/generate_amip_forcing.py-217-    1995: dict(co2=360.91, ch4=1748.0, n2o=311.5, cfc11=265.0, cfc12=531.0),
./scripts/data/generate_amip_forcing.py:218:    2000: dict(co2=369.55, ch4=1773.0, n2o=316.0, cfc11=261.0, cfc12=541.0),
./scripts/data/generate_amip_forcing.py-219-    2005: dict(co2=379.80, ch4=1774.0, n2o=319.0, cfc11=251.0, cfc12=540.0),
./scripts/data/generate_amip_forcing.py-220-    2010: dict(co2=389.85, ch4=1798.0, n2o=323.0, cfc11=240.0, cfc12=530.0),
./scripts/data/generate_amip_forcing.py-221-    2014: dict(co2=397.55, ch4=1830.0, n2o=327.1, cfc11=233.0, cfc12=518.0),
--
./scripts/data/prewarm_voronoi_mesh.py-8-
./scripts/data/prewarm_voronoi_mesh.py-9-Mesh flavours
./scripts/data/prewarm_voronoi_mesh.py-10--------------
./scripts/data/prewarm_voronoi_mesh.py:11:``--lloyd 50`` (default) is the production SCVT. ``--lloyd 0`` is the
./scripts/data/prewarm_voronoi_mesh.py-12-LABELLED SYNTHETIC SCALING MESH — a bisected icosahedron whose Voronoi
./scripts/data/prewarm_voronoi_mesh.py-13-dual is valid for TRiSK but under-relaxed: measured at subdiv-6, area CV
./scripts/data/prewarm_voronoi_mesh.py-14-0.084 vs 0.061 and 128-part imbalance 1.148 vs 1.095 against lloyd=50
--
./scripts/data/prewarm_voronoi_mesh.py-21-Usage
./scripts/data/prewarm_voronoi_mesh.py-22------
./scripts/data/prewarm_voronoi_mesh.py-23-    LEGOESM_MESH_CACHE_DIR=/work/.../mesh_cache \\
./scripts/data/prewarm_voronoi_mesh.py:24:    python scripts/data/prewarm_voronoi_mesh.py --level 9 --lloyd 0
./scripts/data/prewarm_voronoi_mesh.py-25-"""
./scripts/data/prewarm_voronoi_mesh.py-26-from __future__ import annotations
./scripts/data/prewarm_voronoi_mesh.py-27-
--
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-186-    vcmaxpft = vcmaxpft.at[1].set(62.5)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-187-    vcmaxpft = vcmaxpft.at[2].set(62.5)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-188-    vcmaxpft = vcmaxpft.at[3].set(39.1)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py:189:    vcmaxpft = vcmaxpft.at[4].set(41.0)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-190-    vcmaxpft = vcmaxpft.at[5].set(61.4)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py:191:    vcmaxpft = vcmaxpft.at[6].set(41.0)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-192-    vcmaxpft = vcmaxpft.at[7].set(57.7)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-193-    vcmaxpft = vcmaxpft.at[8].set(57.7)
./packages/land/legoesm/land/canopy/clm_ml_backend/multilayer_canopy/MLpftconMod.py-194-    vcmaxpft = vcmaxpft.at[9].set(61.7)
--
./packages/land/legoesm/land/canopy/config.py-35-    "ENF":    [ 62.5,  62.5,  62.6],  # [CLM45] NET Temperate/Boreal
./packages/land/legoesm/land/canopy/config.py-36-    "EBF":    [ 55.0,  61.5,  61.5],  # [CLM45] BET Tropical/Temperate
./packages/land/legoesm/land/canopy/config.py-37-    "DNF":    [ 39.1,  39.1,  39.1],  # [CLM45] NDT Boreal (only DNF entry)
./packages/land/legoesm/land/canopy/config.py:38:    "DBF":    [ 41.0,  57.7,  57.7],  # [CLM45] BDT Tropical/Temperate/Boreal
./packages/land/legoesm/land/canopy/config.py-39-    "MF":     [ 54.0,  62.0,  63.0],  # [JR] Mixed forest (no CLM MF PFT)
./packages/land/legoesm/land/canopy/config.py-40-    "SHR":    [ 54.0,  54.0,  54.0],  # [CLM45] BDS / [JR] shrub (OSH+CSH)
./packages/land/legoesm/land/canopy/config.py-41-    "SAV":    [ 90.0, 120.0, 120.0],  # [JR] savanna (WSA+SAV; no CLM PFT)
--
./docs/dev-notes/ocean_faithfulness_nemo.md-306-| **ACC@Drake** (section) | **143.6 Sv** | n/a (no diag) | 159 | ~137 |
./docs/dev-notes/ocean_faithfulness_nemo.md-307-| **SST RMSE / corr** | **1.87 / 0.985** | 2.27 / 0.976 | — | — |
./docs/dev-notes/ocean_faithfulness_nemo.md-308-| SSS RMSE / bias | 3.97 / −1.53 | 7.92 / −4.17 | — | — |
./docs/dev-notes/ocean_faithfulness_nemo.md:309:- **AMOC overshoot FIXED**: 45→20.5 Sv, now NEAR NEMO 17.7 / obs ~17. Two compounding fixes — the
./docs/dev-notes/ocean_faithfulness_nemo.md-310-  min-rule AMOC edge thickness (removed the partial-cell overcount) AND the corrected SSS (no fresh
./docs/dev-notes/ocean_faithfulness_nemo.md-311-  drift → physical stratification → physical overturning). Multi-decade-gated transport now lands in-band
./docs/dev-notes/ocean_faithfulness_nemo.md-312-  at year 5.
--
./packages/land/legoesm/land/surface_params.py-259-    # broadleaf evergreen temperate
./packages/land/legoesm/land/surface_params.py-260-    [0.15, 0.98, 1.50, 220.0, 2.0e6, 1.0, 1.8, 0.12, 0.30, 61.0, 50.0, 9.0],
./packages/land/legoesm/land/surface_params.py-261-    # broadleaf deciduous tropical
./packages/land/legoesm/land/surface_params.py:262:    [0.16, 0.98, 2.00, 200.0, 2.0e6, 1.0, 1.5, 0.12, 0.30, 41.0, 40.0, 9.0],
./packages/land/legoesm/land/surface_params.py-263-    # broadleaf deciduous temperate
./packages/land/legoesm/land/surface_params.py-264-    [0.17, 0.97, 1.00, 200.0, 2.0e6, 1.0, 1.5, 0.12, 0.30, 58.0, 45.0, 9.0],
./packages/land/legoesm/land/surface_params.py-265-    # broadleaf deciduous boreal
--
./docs/dev-notes/fv3_fortran_fidelity_review.md-667-| config            | step survival | |u_max| (m/s) | h range (m) |
./docs/dev-notes/fv3_fortran_fidelity_review.md-668-|-------------------|--------------:|--------------:|-------------|
./docs/dev-notes/fv3_fortran_fidelity_review.md-669-| default (no Smag) |    288/288    |        31.30  | [2069, 18007] |
./docs/dev-notes/fv3_fortran_fidelity_review.md:670:| Smag iter-963     |    288/288    |        41.03  | [2726, 16660] |
./docs/dev-notes/fv3_fortran_fidelity_review.md-671-
./docs/dev-notes/fv3_fortran_fidelity_review.md-672-W5 analytical |u| = 20 m/s.  Default damping gives |u|=31.3 m/s
./docs/dev-notes/fv3_fortran_fidelity_review.md-673-(closer to analytical); Smagorinsky tuning gives |u|=41 m/s
--
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-87-| **Dissipation deficit** | KE budget at rough & laminar states | dissipation is *large* (`D=-1.1e9`, exceeds wind input); balances wind exactly at the laminar state |
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-88-| **Coriolis** | instantaneous power `∫u·(f k×u)` (uniform grid → exact) | **machine-zero** (`|P|/(f·KE)=4.5e-8`); legoESM's Sadourny Coriolis already conserves energy |
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-89-| **Centered advection** | instantaneous power `∫u·adv` | **machine-zero** (`P=1e-13`); legoESM's centered flux-form is energy-neutral |
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md:90:| **Time integration** | inviscid `E(T)/E0` vs `dt` (1200→150 s) | growth **converges** to +20.5% as `dt→0` → spatial, not a time-truncation error |
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-91-
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-92-## UPDATE 2026-06-18 (iteration 3): the leak is OPERATOR-LEVEL, not the split structure
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-93-
--
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-165-A **total mechanical-energy** budget (`E = KE + ½g∫η²`; PE is only ~0.1 % of KE here) shows a
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-166-genuine spurious source under inviscid + unforced dynamics:
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-167-
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md:168:- at the **rough/turbulent** state: `dE/dt > 0`, inviscid-unforced growth **+20.5 %/5.6 days**
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-169-  (`S/W ≈ 3.2`, i.e. the dynamics inject ~3× the wind work);
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-170-- at MITgcm's **smooth laminar** state: `S ≈ 0` (`|S|/|W| = 0.06`) and legoESM is *stable*.
./docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md-171-
--
./tests/bench/test_bench_mpas_spmd_gates.py-56-    src = Path(_BENCH).read_text()
./tests/bench/test_bench_mpas_spmd_gates.py-57-    for flag in ("--parity-gate", "--check-conservation", "--mass-rtol",
./tests/bench/test_bench_mpas_spmd_gates.py-58-                 "--multicontroller", "--coordinator", "--partition-method",
./tests/bench/test_bench_mpas_spmd_gates.py:59:                 "--reorder-for", "--lloyd"):
./tests/bench/test_bench_mpas_spmd_gates.py-60-        assert flag in src
./tests/bench/test_bench_mpas_spmd_gates.py-61-
./tests/bench/test_bench_mpas_spmd_gates.py-62-
./tests/bench/test_bench_mpas_spmd_gates.py-63-def test_lloyd_flag_reaches_the_mesh_builder(monkeypatch):
./tests/bench/test_bench_mpas_spmd_gates.py:64:    """``--lloyd 0`` must select the synthetic scaling mesh, not lloyd=50.
./tests/bench/test_bench_mpas_spmd_gates.py-65-
./tests/bench/test_bench_mpas_spmd_gates.py-66-    Non-vacuous by construction: the sentinel records the kwarg
./tests/bench/test_bench_mpas_spmd_gates.py-67-    ``create_voronoi_mesh`` actually receives, so dropping the plumbing
--
./tests/bench/test_bench_mpas_spmd_gates.py-161-@pytest.mark.timeout(600)
./tests/bench/test_bench_mpas_spmd_gates.py-162-def test_single_device_reference_leg_with_reorder_for(tmp_path):
./tests/bench/test_bench_mpas_spmd_gates.py-163-    """The ladder's nd=1 reference: mesh partitioned/ghost-padded for the
./tests/bench/test_bench_mpas_spmd_gates.py:164:    ladder target (--reorder-for 4 pads 642 -> 644 cells) but run on ONE
./tests/bench/test_bench_mpas_spmd_gates.py-165-    device — the codex-r1 stale-model/dev-config hazard regression guard."""
./tests/bench/test_bench_mpas_spmd_gates.py-166-    out = tmp_path / "ref.jsonl"
./tests/bench/test_bench_mpas_spmd_gates.py-167-    env = dict(os.environ)
--
./tests/bench/test_bench_mpas_spmd_gates.py-170-    proc = subprocess.run(
./tests/bench/test_bench_mpas_spmd_gates.py-171-        [
./tests/bench/test_bench_mpas_spmd_gates.py-172-            sys.executable, str(_BENCH),
./tests/bench/test_bench_mpas_spmd_gates.py:173:            "--n-devices", "1", "--reorder-for", "4",
./tests/bench/test_bench_mpas_spmd_gates.py-174-            "--subdivision", "3", "--nlev", "4",
./tests/bench/test_bench_mpas_spmd_gates.py-175-            "--steps", "2", "--warmup", "0",
./tests/bench/test_bench_mpas_spmd_gates.py-176-            "--partition-method", "sfc",
--
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1854-| 150 | 4.654 | 1.64× |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1855-| 160 | 7.645 | 1.64× |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1856-| 170 | 12.55 | 1.64× |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md:1857:| 180 | 20.59 | 1.64× |
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1858-
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1859-**Pure exponential, no saturation.**  Doubling time locks at ~14
./docs/ocean/experiments/density_jacobian_pgf_mpas.md-1860-days post day-90 and stays there.  Combined damping pushes NaN
--
./docs/ocean/experiments/realistic_geometry_topology_fixes.md-43-| Taiwan Strait | 24° | 119.5° | 80 | 60 | South China Sea |
./docs/ocean/experiments/realistic_geometry_topology_fixes.md-44-| Windward Passage | 20° | -73.5° | 60 | 1500 | Caribbean |
./docs/ocean/experiments/realistic_geometry_topology_fixes.md-45-| Florida Strait | 25.5° | -79.5° | 100 | 800 | Gulf Stream source |
./docs/ocean/experiments/realistic_geometry_topology_fixes.md:46:| Luzon Strait | 20.5° | 121.5° | 100 | 2000 | South China Sea |
./docs/ocean/experiments/realistic_geometry_topology_fixes.md-47-
./docs/ocean/experiments/realistic_geometry_topology_fixes.md-48-**Mechanism**: for each strait, all grid cells within a great-circle
./docs/ocean/experiments/realistic_geometry_topology_fixes.md-49-corridor of half-width `min_width_km / 2` (scaled by
--
./docs/performance/scaling/crm_les_scaling.md-102-|----------------------|-------------------------------:|-------------------:|
./docs/performance/scaling/crm_les_scaling.md-103-| 64³ (262 k)          | 27.7 Mc/s                      | 8.5 Mc/s           |
./docs/performance/scaling/crm_les_scaling.md-104-| 96³ (590 k)          | 36.0 Mc/s                      | 9.1 Mc/s           |
./docs/performance/scaling/crm_les_scaling.md:105:| 128³ (1.05 M)        | **41.0 Mc/s**                  | —                  |
./docs/performance/scaling/crm_les_scaling.md-106-
./docs/performance/scaling/crm_les_scaling.md-107-- **fp32 scales healthily** (rising 27.7→41 Mc/s with size) — production mode.
./docs/performance/scaling/crm_les_scaling.md-108-- **fp64 works but ~4× slower** (consumer 5090 fp64 ≈ 1/64 + the FFT in fp64) —
--
./docs/performance/scaling/crm_les_scaling.md-489-| precision | np1 | np2 | np4 | speedup @4 |
./docs/performance/scaling/crm_les_scaling.md-490-|-----------|----:|----:|----:|-----------|
./docs/performance/scaling/crm_les_scaling.md-491-| fp64 | 88.9 | 77.9 | 74.6 ms | 1.19× (~30% eff) |
./docs/performance/scaling/crm_les_scaling.md:492:| fp32 | 60.6 | 51.9 | 41.0 ms | 1.48× (~37% eff) |
./docs/performance/scaling/crm_les_scaling.md-493-
./docs/performance/scaling/crm_les_scaling.md-494-**Weak** (per-rank 32×64×32):
./docs/performance/scaling/crm_les_scaling.md-495-
--
./docs/performance/scaling/scaling_indicators.csv-28-2026-06-14,58475e41,tiled_3d_geo,atm_cube,spmd,capability,6,ops3d_np24,8485746,compute_geopotential (Simmons-Burridge vertical integration) np24-tiled — vertical-local per-column exact-partition; 6 shared 3D-PE ops np24-tiled
./docs/performance/scaling/scaling_indicators.csv-29-2026-06-14,2c53abea,ocean_spmd_pcg,ocean_latlon,spmd,strong,0.3087,eff_2gpu_ocean_pcg_standard,8486096,FIRST ocean-GPU SPMD number — latlon 360x720 M60 f64 barotropic PCG on 2x RTX8000 PCIe (1GPU 5.06ms 2GPU 8.20ms) kernel anti-scales (psum-latency-bound — audit wall MEASURED)
./docs/performance/scaling/scaling_indicators.csv-30-2026-06-14,2c53abea,ocean_spmd_pcg,ocean_latlon,spmd,strong,0.4280,eff_2gpu_ocean_pcg_single_reduce,8486096,single-reduce (1 psum/iter lever #1) cuts 2-GPU penalty to 7.59 vs 8.20ms — opt-in for the comm-bound multi-GPU regime (1-GPU slower: extra matvec)
./docs/performance/scaling/scaling_indicators.csv:31:2026-06-14,246df03f,ocean_fullstep_2gpu,ocean_latlon,mpi,strong,0.659,eff_2gpu_ocean_fullstep,8486172,FULL ocean step 2xRTX8000 mpi4jax HOST-STAGED 180x360 nlev30 implicit_cn f64 (1GPU 27.0ms 2GPU 20.5ms 1.32x) — AMORTIZATION CONFIRMED barotropic-only anti-scales 0.31 to full-step 0.66 (cuda-aware MPI next lever toward 0.8-0.93)
./docs/performance/scaling/scaling_indicators.csv-32-2026-06-14,bf9c21f4,ocean_fullstep_def,ocean_latlon,mpi,strong,0.624,eff_2gpu_ocean_fullstep_180n30,8486182,SOLVER-MATCHED (np1 force-pcg) full ocean step 2xRTX8000 180x360 nlev30 (np1 25.5ms np2 20.4ms 1.25x) — small config comm-bound; GPU-distinctness PROVEN CVD0/CVD1 distinct PIDs
./docs/performance/scaling/scaling_indicators.csv-33-2026-06-14,bf9c21f4,ocean_fullstep_def,ocean_latlon,mpi,strong,0.920,eff_2gpu_ocean_fullstep_360n60,8486182,SOLVER-MATCHED full ocean step 2xRTX8000 360x720 nlev60 (np1 272ms np2 148ms 1.84x) NEAR-IDEAL at production scale — amortization eff rises 0.62 to 0.92 with problem size; host-staged MPI = lower bound (cuda-aware next)
./docs/performance/scaling/scaling_indicators.csv-34-2026-06-14,32a1d490,atm_latlon_2gpu,atm_latlon,mpi,strong,0.821,eff_2gpu_atm_latlon,8486204,FIRST atm lat-lon 2-GPU (route-A overlay venv mpi4jax host-staged) 180x360 nlev26 held_suarez f64 (np1 18.9ms np2 11.5ms 1.64x) — explicit FV PE dycore halo-bound scales cleanly (no ocean barotropic bottleneck); eff rises 0.72 res90 to 0.82 res180
--
./docs/performance/scaling/amip_mpi_scaling.md-423-| latlon  | LL96  |   368 640 |  1.97   |  187.0   |
./docs/performance/scaling/amip_mpi_scaling.md-424-| latlon  | LL128 |   655 360 |  2.56   |  256.3   |
./docs/performance/scaling/amip_mpi_scaling.md-425-| latlon  | LL192 | 1 474 560 |  4.67   | **315.6**|
./docs/performance/scaling/amip_mpi_scaling.md:426:| MPAS    | I4    |    51 240 |  2.50   |   20.5   |
./docs/performance/scaling/amip_mpi_scaling.md-427-| MPAS    | I5    |   204 840 |  3.43   |   59.7   |
./docs/performance/scaling/amip_mpi_scaling.md-428-| MPAS    | I6    |   819 240 |  7.97   | **102.8**|
./docs/performance/scaling/amip_mpi_scaling.md-429-
--
./tests/land/unit/test_canopy_vcmax25_table.py-23-
./tests/land/unit/test_canopy_vcmax25_table.py-24-def test_corrected_c3_values_match_differbess():
./tests/land/unit/test_canopy_vcmax25_table.py-25-    # The old transcription errors are gone.
./tests/land/unit/test_canopy_vcmax25_table.py:26:    assert PFT_VCMAX25_C3["DBF"] == [41.0, 57.7, 57.7]   # was [66, 62, 96]
./tests/land/unit/test_canopy_vcmax25_table.py-27-    assert PFT_VCMAX25_C3["GRA"] == [78.2, 78.2, 78.2]   # was [78, 78, 142]
./tests/land/unit/test_canopy_vcmax25_table.py-28-    assert PFT_VCMAX25_C3["DNF"] == [39.1, 39.1, 39.1]   # was [57, 57, 57]
./tests/land/unit/test_canopy_vcmax25_table.py-29-    assert PFT_VCMAX25_C3["EBF"][0] == 55.0              # tropical, was 41
--
./tests/land/unit/test_canopy.py-461-        state0 = init_multilayer_land_state(ncol, config, T_init=288.0)
./tests/land/unit/test_canopy.py-462-        forcing = _make_forcing(ncol)
./tests/land/unit/test_canopy.py-463-        lat = jnp.full(ncol, 38.47)
./tests/land/unit/test_canopy.py:464:        doy, dt = 120.5, 1800.0
./tests/land/unit/test_canopy.py-465-
./tests/land/unit/test_canopy.py-466-        # 1) Warm-start with a concrete forward (cold) step → builds mlcanopy.
./tests/land/unit/test_canopy.py-467-        state1, _, _ = step_multilayer_land(
--
./tests/land/unit/test_canopy.py-513-
./tests/land/unit/test_canopy.py-514-        new_state, response, _ = step_multilayer_land(
./tests/land/unit/test_canopy.py-515-            state, forcing, config, U_min=1.0, dt=1800.0,
./tests/land/unit/test_canopy.py:516:            lat=jnp.full(NCOL, 38.47), doy=120.5)
./tests/land/unit/test_canopy.py-517-
./tests/land/unit/test_canopy.py-518-        # Check TileResponse is finite
./tests/land/unit/test_canopy.py-519-        for name in response._fields:
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1108-  shape-dependent). Recorded as observed; not chased further at subdiv-8
./docs/performance/scaling/levante_campaign_2026-07-24.md-1109-  since the mesh is below the floor at all these counts anyway.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1110-* Payoff ladder submitted (job 26549775): subdiv-9 at 32/64/128 GPUs =
./docs/performance/scaling/levante_campaign_2026-07-24.md:1111:  81.9k/41.0k/20.5k cells/GPU — the first MPAS many-GPU ladder whose
./docs/performance/scaling/levante_campaign_2026-07-24.md-1112-  lower rungs sit ABOVE the ~30k floor.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1113-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1114-OPERATIONAL NOTE: a Lustre incident mid-implementation left the module
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1462-   with the system toolchain (GLIBCXX mismatch with gcc-11-built OpenMPI
./docs/performance/scaling/levante_campaign_2026-07-24.md-1463-   module).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1464-9. Diagnosis tool halo/overlap phases timed UN-JITTED eager pads
./docs/performance/scaling/levante_campaign_2026-07-24.md:1465:   (20.5e6 us per "exchange", bandwidth 0.0 GB/s; job 26447827) - FIXED
./docs/performance/scaling/levante_campaign_2026-07-24.md-1466-   this campaign (jit + dtype-correct bytes + refuse a bandwidth at
./docs/performance/scaling/levante_campaign_2026-07-24.md-1467-   world_size==1; overlap fractions >100% now refused), contract test
./docs/performance/scaling/levante_campaign_2026-07-24.md-1468-   tests/bench/test_halo_profiler_contract.py, verified 27-162 us at
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1638-  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
./docs/performance/scaling/levante_campaign_2026-07-24.md-1639-  timing, but a "production config" claim would need a gated arm).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1640-
./docs/performance/scaling/levante_campaign_2026-07-24.md:1641:### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
./docs/performance/scaling/levante_campaign_2026-07-24.md-1642-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
./docs/performance/scaling/levante_campaign_2026-07-24.md-1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1650-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1651-| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1652-|---|---|---|---|
./docs/performance/scaling/levante_campaign_2026-07-24.md:1653:| 32 | 81.9k | 12.47 | 5.47 |
./docs/performance/scaling/levante_campaign_2026-07-24.md:1654:| 64 | 41.0k | 9.60 | 7.10 |
./docs/performance/scaling/levante_campaign_2026-07-24.md:1655:| 128 | 20.5k | 11.48 | 5.94 |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1656-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1657-* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
./docs/performance/scaling/levante_campaign_2026-07-24.md-1658-  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
./docs/performance/scaling/levante_campaign_2026-07-24.md-1659-  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1660-* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
./docs/performance/scaling/levante_campaign_2026-07-24.md:1661:  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
./docs/performance/scaling/levante_campaign_2026-07-24.md-1662-  the other lanes (single unreplicated point on a lane with known
./docs/performance/scaling/levante_campaign_2026-07-24.md-1663-  count-specific codegen variation — not by itself proof).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1664-* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1688-   attribution (fabric contention vs placement/topology vs drift) is a
./docs/performance/scaling/levante_campaign_2026-07-24.md-1689-   follow-up, not a conclusion of this job.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
./docs/performance/scaling/levante_campaign_2026-07-24.md:1691:   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
./docs/performance/scaling/levante_campaign_2026-07-24.md:1692:   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
./docs/performance/scaling/levante_campaign_2026-07-24.md-1693-   recomputed only from these.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1694-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1695-### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1724-| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1725-| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1726-| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
./docs/performance/scaling/levante_campaign_2026-07-24.md:1727:| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
./docs/performance/scaling/levante_campaign_2026-07-24.md:1728:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1729-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1730-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1731-
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1755-result JSON — only `decomposition: 2d`; a follow-up could add them to
./docs/performance/scaling/levante_campaign_2026-07-24.md-1756-the bench metadata.)
./docs/performance/scaling/levante_campaign_2026-07-24.md-1757-
./docs/performance/scaling/levante_campaign_2026-07-24.md:1758:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
./docs/performance/scaling/levante_campaign_2026-07-24.md-1759-
./docs/performance/scaling/levante_campaign_2026-07-24.md:1760:s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
./docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
./docs/performance/scaling/levante_campaign_2026-07-24.md-1762-7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
./docs/performance/scaling/levante_campaign_2026-07-24.md-1763-untouched): **6.58 / 6.43 / 7.29 ms**.
./docs/performance/scaling/levante_campaign_2026-07-24.md-1764-
./docs/performance/scaling/levante_campaign_2026-07-24.md:1765:Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):
./docs/performance/scaling/levante_campaign_2026-07-24.md-1766-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1767-| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1768-|---|---|---|---|---|
./docs/performance/scaling/levante_campaign_2026-07-24.md:1769:| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
./docs/performance/scaling/levante_campaign_2026-07-24.md:1770:| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
./docs/performance/scaling/levante_campaign_2026-07-24.md:1771:| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
./docs/performance/scaling/levante_campaign_2026-07-24.md-1772-
./docs/performance/scaling/levante_campaign_2026-07-24.md-1773-* The falsifiability block's CONFIRM branch fires: ratios stay well
./docs/performance/scaling/levante_campaign_2026-07-24.md-1774-  above 1, so the round-20 retraction's CONFOUND did not manufacture
./docs/performance/scaling/levante_campaign_2026-07-24.md-1775-  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
./docs/performance/scaling/levante_campaign_2026-07-24.md-1776-  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
./docs/performance/scaling/levante_campaign_2026-07-24.md:1777:  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1778-* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
./docs/performance/scaling/levante_campaign_2026-07-24.md-1779-  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
./docs/performance/scaling/levante_campaign_2026-07-24.md-1780-  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
--
./docs/performance/scaling/levante_campaign_2026-07-24.md-1783-  decay with parts; the metis receipt argues against pure
./docs/performance/scaling/levante_campaign_2026-07-24.md-1784-  partition-cut explanations, on the CPU lane at least).
./docs/performance/scaling/levante_campaign_2026-07-24.md-1785-* The non-monotone tile dependence of the ratio (largest at the
./docs/performance/scaling/levante_campaign_2026-07-24.md:1786:  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
--
./docs/performance/scaling/CRM_LES_SUMMARY.md-112-|------------|----------------------:|-------------------:|
./docs/performance/scaling/CRM_LES_SUMMARY.md-113-| 64³ (262 k)  | 27.7 Mc/s | 8.5 Mc/s |
./docs/performance/scaling/CRM_LES_SUMMARY.md-114-| 96³ (590 k)  | 36.0 Mc/s | 9.1 Mc/s |
./docs/performance/scaling/CRM_LES_SUMMARY.md:115:| 128³ (1.05 M)| **41.0 Mc/s** | — |
./docs/performance/scaling/CRM_LES_SUMMARY.md-116-
./docs/performance/scaling/CRM_LES_SUMMARY.md-117-- **fp32 scales healthily** (rising 27.7→41 Mc/s) — the production GPU mode.
./docs/performance/scaling/CRM_LES_SUMMARY.md-118-- fp64 works but ~4× slower (consumer fp64 + the FFT pressure solve in fp64).
--
./docs/performance/scaling/CRM_LES_SUMMARY.md-124-  dynamic SGS** (test filters + box3 ∂y halo). A full `step()` matches single-rank
./docs/performance/scaling/CRM_LES_SUMMARY.md-125-  to **1e-9** (state) / **1e-10** (`u_*`), `lasd_cs2` to **1e-12**, all AD-safe
./docs/performance/scaling/CRM_LES_SUMMARY.md-126-  (np 1/2/4). Scaling, dynamic LASD, both precisions:
./docs/performance/scaling/CRM_LES_SUMMARY.md:127:  - strong (global 64²×32): fp64 88.9→77.9→74.6, fp32 60.6→51.9→41.0 ms/step
./docs/performance/scaling/CRM_LES_SUMMARY.md-128-    (np 1/2/4; fp32 1.48× @4) — modest, bandwidth + all-to-all bound.
./docs/performance/scaling/CRM_LES_SUMMARY.md-129-  - weak (per-rank 32×64×32): fp64 40.3→81.2→136.3 ms — **poor BY ALGORITHM**:
./docs/performance/scaling/CRM_LES_SUMMARY.md-130-    the pressure-Poisson FFT globally couples the domain, so weak-scaling grows
--
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py-16-MITGCM_ETA_MAX = 0.0722208208401975
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py-17-MITGCM_UVEL_MAX = 0.12803771977288
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py-18-MITGCM_VVEL_MAX = 0.0090404137970896
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py:19:MITGCM_THETA_MAX = 20.576472287889
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py-20-
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py-21-
./tests/ocean/fidelity/test_mitgcm_front_relax_recipe.py-22-def test_geometry_is_fplane_channel():
--
./docs/performance/scaling/spectral_gpu_feasibility.md-32-| T | grid | n_sh | legacy round trip | **GEMM round trip** | GEMM share of model FLOPs | AI |
./docs/performance/scaling/spectral_gpu_feasibility.md-33-|---|------|------|------------------:|--------------------:|--------------------------:|---:|
./docs/performance/scaling/spectral_gpu_feasibility.md-34-| T42 | 64×128 | 946 | 5.34 ms | 3.04 ms | 46% | 10.0 F/B |
./docs/performance/scaling/spectral_gpu_feasibility.md:35:| T85 | 130×260 | 3,741 | 33.5 ms | 20.5 ms | 59% | 12.1 F/B |
./docs/performance/scaling/spectral_gpu_feasibility.md-36-| T170 | 256×512 | 14,706 | 145.6 ms | **43.2 ms** | 72% | 13.4 F/B |
./docs/performance/scaling/spectral_gpu_feasibility.md-37-
./docs/performance/scaling/spectral_gpu_feasibility.md-38-(FLOP model: the Legendre matrices are REAL against complex fields — a
--
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md-59-  Codex audit 2026-06-15 named this as the ONE untried measurable step-kernel
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md-60-  lever. Measured:
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md-61-  - job 8488104 (A40, I6 np1): default-cmdbuf (FUSION,CUSTOM_CALL,COLLECTIVES)
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md:62:    vs +CUBLAS,CUDNN = **noise** — f32 11.07→11.03 ms/step, f64 20.50→20.42.
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md-63-    Adding CUBLAS/CUDNN command buffers does nothing.
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md-64-  - job 8490224 (RTX8000, I6 np1): the missing **OFF arm** (command buffers
./docs/performance/scaling/scaling_levers_audit_2026-06-15.md-65-    fully DISABLED, `--xla_gpu_enable_command_buffer=`) vs default vs aggressive.
--
./docs/science/specs/FV3_3D.md-4992-    => 4 passed in 81.72 s
./docs/science/specs/FV3_3D.md-4993-
./docs/science/specs/FV3_3D.md-4994-    Full NH suite (baseline + iter-168/169/170/171/172):
./docs/science/specs/FV3_3D.md:4995:    => 59 passed in 420.55 s
./docs/science/specs/FV3_3D.md-4996-
./docs/science/specs/FV3_3D.md-4997-### Status
./docs/science/specs/FV3_3D.md-4998-
--
./docs/land_high_elevation_snow_ice.md-259-
./docs/land_high_elevation_snow_ice.md-260-**Seasonal multi-year spin-up (annual-mean Stage-A → N seasonal years) only half-works:**
./docs/land_high_elevation_snow_ice.md-261-halves N-lat (JJA +0.124 → +0.066, DJF +0.061 → +0.027) but WORSENS Tibet (−0.166 → −0.226
./docs/land_high_elevation_snow_ice.md:262:even as Tibet SWE climbs 6.8 → 20.5 kg/m²) — the extra snow accumulates on the cold high
./docs/land_high_elevation_snow_ice.md-263-peaks while the broad plateau floor stays bare, because the `_DAYS` subsample lacks the
./docs/land_high_elevation_snow_ice.md-264-snowfall events to build the floor pack.
./docs/land_high_elevation_snow_ice.md-265-
--
./tests/unit/test_matrix_jet_and_known_failures.py-39-
./tests/unit/test_matrix_jet_and_known_failures.py-40-def test_jet_floor_passes_healthy_full_run():
./tests/unit/test_matrix_jet_and_known_failures.py-41-    """Healthy jet (spectral/latlon/ico reach 27-66 m/s) keeps PASS."""
./tests/unit/test_matrix_jet_and_known_failures.py:42:    ok, notes = mod._apply_jet_strength_floor(True, "n", 41.0, "held_suarez", 200.0)
./tests/unit/test_matrix_jet_and_known_failures.py-43-    assert ok is True and notes == "n"
./tests/unit/test_matrix_jet_and_known_failures.py-44-
./tests/unit/test_matrix_jet_and_known_failures.py-45-
--
./tests/unit/test_matrix_jet_and_known_failures.py-58-
./tests/unit/test_matrix_jet_and_known_failures.py-59-def test_jet_floor_idempotent_on_failed():
./tests/unit/test_matrix_jet_and_known_failures.py-60-    """A run already FAILed upstream stays FAILed."""
./tests/unit/test_matrix_jet_and_known_failures.py:61:    ok, _ = mod._apply_jet_strength_floor(False, "n", 41.0, "held_suarez", 200.0)
./tests/unit/test_matrix_jet_and_known_failures.py-62-    assert ok is False
./tests/unit/test_matrix_jet_and_known_failures.py-63-
./tests/unit/test_matrix_jet_and_known_failures.py-64-
--
./tests/ocean/unit/test_veros_acc_recipe.py-281-    # offset — the bug this guards against.
./tests/ocean/unit/test_veros_acc_recipe.py-282-    np.testing.assert_allclose(lon, np.arange(NX) * 2.0 - 1.0, atol=1e-3)
./tests/ocean/unit/test_veros_acc_recipe.py-283-    # interior lat rows (1..NY; rows 0/-1 are N/S walls) match Veros yt.
./tests/ocean/unit/test_veros_acc_recipe.py:284:    np.testing.assert_allclose(lat[1:-1], np.arange(NY) * 2.0 - 41.0, atol=1e-3)
./tests/ocean/unit/test_veros_acc_recipe.py-285-
./tests/ocean/unit/test_veros_acc_recipe.py-286-
./tests/ocean/unit/test_veros_acc_recipe.py-287-def test_acc_land_mask_matches_veros_ocean_footprint():
--
./tests/unit/test_build_woce_ic_latlon.py-55-    field[LAT < 0] = np.nan                          # mask south half as land
./tests/unit/test_build_woce_ic_latlon.py-56-    ocean[LAT < 0] = False
./tests/unit/test_build_woce_ic_latlon.py-57-    out = regrid_curv_to_latlon(field, LAT, LON, ocean,
./tests/unit/test_build_woce_ic_latlon.py:58:                                np.array([10.5]), np.array([20.5]), k=4)
./tests/unit/test_build_woce_ic_latlon.py-59-    assert np.isfinite(out).all()
./tests/unit/test_build_woce_ic_latlon.py-60-    assert abs(float(out[0, 0]) - 10.5) < 1.5        # ~ nearest northern value
./tests/unit/test_build_woce_ic_latlon.py-61-
--
./tests/ocean/unit/test_rgb_chl_penetration.py-41-def _nemo_class_index_1based(chl: float) -> int:
./tests/ocean/unit/test_rgb_chl_penetration.py-42-    """NEMO itab = NINT(41 + 20*log10(Chl)), Chl clamped to [0.03, 10] (1-based)."""
./tests/ocean/unit/test_rgb_chl_penetration.py-43-    c = min(max(chl, 0.03), 10.0)
./tests/ocean/unit/test_rgb_chl_penetration.py:44:    return int(math.floor(41.0 + 20.0 * math.log10(c) + 0.5))
./tests/ocean/unit/test_rgb_chl_penetration.py-45-
./tests/ocean/unit/test_rgb_chl_penetration.py-46-
./tests/ocean/unit/test_rgb_chl_penetration.py-47-def _nemo_rgb_recursion(sw, chl_col, dz, n_wet, rho0=_RHO0, csw=_CSW):

exec
/usr/bin/bash -lc "rg --files -g '"'!uv.lock'"' | rg '26628076|26600095|s8_lloyd0|s9.*ladder|levante.*(json|log)|scaling.*(json|log)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 361ms:
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch

exec
/usr/bin/bash -lc "sed -n '1630,1700p' docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '1750,1800p' docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '1,160p' scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch && rg -n -C 2 \"26600095|s9.*payoff|subdivision 9|--reorder-for 128|lloyd 0\" scripts/cluster/scaling_levante" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 295ms:
  a byte-identical partition). NOTE the second `--distribution` field is
  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
  identically; the swing is socket-level. Mechanism (per-socket
  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
  2.13x receipt; never instrumented with bandwidth counters.
* Caveats: timing-only receipt — no parity/conservation gate ran in
  these arms, and the CPU nodes emit `UCX WARN transports
  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
  timing, but a "production config" claim would need a gated arm).

### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)

f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
reads `unknown` — same allocation, same submitted script, so the same
binary is PLAUSIBLE but that row stays non-reproduction-grade on its
own (codex r20/r21):

| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
|---|---|---|---|
| 32 | 81.9k | 12.47 | 5.47 |
| 64 | 41.0k | 9.60 | 7.10 |
| 128 | 20.5k | 11.48 | 5.94 |

* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
  the other lanes (single unreplicated point on a lane with known
  count-specific codegen variation — not by itself proof).
* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
  term".** Confounds: (a) every existing s8 receipt is the generator's
  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
  family, not the same protocol; (b) two comparator points came from the
  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
  no weak-scaling direction is claimed until it lands.

### Next receipts submitted 2026-08-02

1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
   after — BRACKETED, not fully counterbalanced; a penalty's attribution
   to fabric vs placement/drift needs the per-step nodelist table +
   follow-up). steps=5000 so the stepping window
   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
   wall-clock brackets logged as overlap evidence. CONFIRM bar:
   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
   >10 % = a CO-EXECUTION penalty, quantified per replica — its
   attribution (fabric contention vs placement/topology vs drift) is a
   follow-up, not a conclusion of this job.
2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
   recomputed only from these.

### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)

LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
@64 row (job 26502539, f32 6.73 ms):

| arm | ms/step | GC/s (col-levels) |
CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
incorrect results", parallel/reductions.py runtime check) and UCX
VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
result JSON — only `decomposition: 2d`; a follow-up could add them to
the bench metadata.)

### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL

s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
warmup 3 — protocol-identical to the s9 ladder (26600095), git
7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
untouched): **6.58 / 6.43 / 7.29 ms**.

Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):

| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
|---|---|---|---|---|
| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |

* The falsifiability block's CONFIRM branch fires: ratios stay well
  above 1, so the round-20 retraction's CONFOUND did not manufacture
  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
  fraction growth, collective latency vs count, sfc partition-quality
  decay with parts; the metis receipt argues against pure
  partition-cut explanations, on the CPU lane at least).
* The non-monotone tile dependence of the ratio (largest at the
  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
#!/bin/bash -l
#SBATCH --job-name=mpas_s8_l0
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_s8_l0.%j.log
# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
#
# Falsifiability, written BEFORE submit:
#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
#             matched cells/GPU stay well above 1 (prior draft saw
#             1.80/1.35/1.41 on the CONFOUNDED pairs)
#   REFUTE  : ratios collapse toward ~1.0 -> the draft's "term" was the
#             Lloyd-mesh confound, and MPAS-GPU weak scaling is near
#             ideal at matched tile.
# Mesh: prewarmed into LEGOESM_MESH_CACHE_DIR (subdiv-8 is below the
# big-mesh refuse threshold, so a cache miss falls back to in-process
# builds — slower, still correct).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s8_l0_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
for NP in 8 16 32; do
  NODES=$(( NP / 4 ))
  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NP" \
      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
      --partition-method sfc --reorder-for 128 \
      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
done
echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
for NP in 8 16 32; do
  F="$OUTDIR/np${NP}.jsonl"
  "$PY" -c "
import json,sys
try:
    d=json.loads(open('$F').readline())
    print(f'np$NP: {d[\"steady_median_ms\"]:8.2f} ms')
except Exception as e:
    print('np$NP: MISSING/UNPARSEABLE ->', e); sys.exit(1)" \
    || { echo "np$NP receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
scripts/cluster/scaling_levante/prewarm_s10.sbatch-20-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/prewarm_s10.sbatch-21-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
scripts/cluster/scaling_levante/prewarm_s10.sbatch:22:"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-12-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:18:# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:49:      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-34-#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:36:# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-69-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:71:      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:72:      --partition-method sfc --reorder-for 128 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?

exec
/usr/bin/bash -lc "git status --short && git log --all --oneline --decorate -12 && git log --all --oneline -- scripts/cluster/scaling_levante | head -40 && rg -n -i -g '"'!uv.lock'"' \"padded n_cells|n_cells.*2,621,568|26600095|s9 np32|mpas_s9.*ladder|s9.*np32\" .claude scripts docs/performance/scaling 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 17433ms:
 M docs/performance/scaling/levante_campaign_2026-07-24.md
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md
?? scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
?? scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
?? scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
?? scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
?? scripts/cluster/scaling_levante/prewarm_s10.sbatch
8916b1172 (origin/main, origin/HEAD) fix(cmor): clivi reports only the ice radiation actually sees (#1443)
91c499254 (origin/fix/clivi-radiative-ice) fix(cmor): clivi reports only the ice radiation actually sees
cef9d4ab4 feat(cmor): publish clt on the MPAS lane via a shared overlap reduction (#1437)
3840be87d (origin/feat/mpas-clt-cmor) feat(cmor): publish clt on the MPAS lane via a shared overlap reduction
8b0ab8c85 docs(config): scope the latlon24 convection evidence to its own resolution (#1436)
cc4a52e16 (origin/mpas-stability-campaign, mpas-stability-campaign) docs(config): scope the latlon24 convection evidence to its own resolution
fca8543a5 Merge pull request #1435 from climate-federation/fix/1388-fp32-conservation-gates
297fe4ee8 (origin/fix/1388-fp32-conservation-gates) fix(tests): #1388 codex round-2 — correct my own attribution, keep fp32 coverage
11fa0a2b4 fix(tests): #1388 the ocean barotropic reds were three different non-defects
3a5b10be8 chore(deps): drop dataclasses-json — a hard dependency nothing used (#1434)
5fca45323 (origin/chore/drop-dataclasses-json) chore(deps): drop dataclasses-json — a hard dependency nothing used
bd04cb2bd Merge pull request #1433 from climate-federation/fix/1387-ocean-sweep
1620b9653 fix(bench): #1361 round 2 — Levante launchers gated, ocean sharding matched to the fallback
8ab402ea8 fix(cluster): CFL-scaled dt on the tiled cube lane (C768 closed loop went non-finite)
91e36645b fix: codex round-3 findings — launcher world size, geometry fingerprint guard, gated ladders
9051c5126 fix(cluster,ocean): OUTDIR job-id suffix (same-second collision) + replicated-geometry broadcast under multicontroller
6f8949ac5 fix(cluster): tiled-cube Levante lane — same SLURM double-pin as the multinode lanes
78791cc93 fix(cluster): drop CVD pre-pin on Levante route-B lanes (jax SLURM double-pin) + solver-matched ladder flags
ee7976330 fix(bench,cluster): solver-matched ocean scaling ladders + Levante lane hardening
b07f844b6 perf(scaling): full collective census + wire the diagnosis tool to the clusters
821d5d40c feat(scaling): first-class HLO collective census + Derecho/Levante diagnosis lanes
e37357fc2 fix(scaling): serialize NCCL comm-init on the closed-loop tiled-cube lane (#921) (#959)
8fc4c94ec feat(parallel): cube np>6 sub-face PRODUCTION assembly — blocked persistent tiled step (#896)
240b42fc3 feat(bench): SYPD in SPMD lanes + JSONL aggregation + lane-T comm-tuning A/B (post-Derecho census) (#894)
06b884543 feat(bench): sub-face tiled cube step scaling lane + 24-GPU Derecho/Levante jobs (audit item 5) (#890)
1585e2305 fix(cluster): cube-tiled wrappers fail-fast (set -e) so a red parity gate blocks Stage B (codex)
1a4a403ab fix(cluster): cube-tiled wrappers follow the _env.sh/$PY convention + kt=2 allocation refusal (codex)
2c66509e7 feat(bench): sub-face tiled cube step scaling lane + Derecho/Levante 24-GPU jobs (audit item 5)
7e5287ceb Merge origin/main into omip-faithful-nemo-comparison (pull latest main)
97fcde113 perf(scaling): icosahedral/MPAS multicontroller SPMD lane — cell-partition reorder across processes (#757)
e986860aa perf(scaling): icosahedral/MPAS multicontroller SPMD lane — cell-partition reorder across processes
2d1cf0783 perf(scaling): Derecho/Levante hardening on top of the scaling roadmap — gates, federation robustness, ocean/multinode jobs, aws-ofi-nccl build (#751)
0766fca8a perf(scaling): Derecho/Levante hardening on top of the scaling roadmap — gates, federation robustness, ocean/multinode jobs, aws-ofi-nccl build
c2b3e4c7d feat(bench): Levante multi-node route-A GPU-MPI scaling scripts (roadmap item 2)
1f5f8be13 feat(bench): Levante multi-node route-A GPU-MPI scaling scripts (roadmap item 2)
docs/performance/scaling/levante_campaign_2026-07-24.md:1641:### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
docs/performance/scaling/levante_campaign_2026-07-24.md:1644:scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:18:# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:36:# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/plot/plot_scaling_paper_figure.py:38:                "s8 np32-128 26549646/26538474, s9 26600095, "

exec
/usr/bin/bash -lc "sed -n '1,240p' .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md && sed -n '1,260p' .physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 361ms:
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
session id: 019fc238-56a1-75b1-b93f-8220e8fecfaf
--------
user
Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
exec
/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 2031ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
./uv.lock-2170-sdist = { url = "https://files.pythonhosted.org/packages/0e/4a/c27b42ed9b1c7d13d9ba8b6905dece787d6259152f2309338aed29b2447b/ml_dtypes-0.5.4.tar.gz", hash = "sha256:8ab06a50fb9bf9666dd0fe5dfb4676fa2b0ac0f31ecff72a6c3af8e22c063453", size = 692314, upload-time = "2025-11-17T22:32:31.031Z" }
./uv.lock-2171-wheels = [
./uv.lock-2172-    { url = "https://files.pythonhosted.org/packages/c6/5e/712092cfe7e5eb667b8ad9ca7c54442f21ed7ca8979745f1000e24cf8737/ml_dtypes-0.5.4-cp311-cp311-macosx_10_9_universal2.whl", hash = "sha256:6c7ecb74c4bd71db68a6bea1edf8da8c34f3d9fe218f038814fd1d310ac76c90", size = 679734, upload-time = "2025-11-17T22:31:39.223Z" },
./uv.lock:2173:    { url = "https://files.pythonhosted.org/packages/4f/cf/912146dfd4b5c0eea956836c01dcd2fce6c9c844b2691f5152aca196ce4f/ml_dtypes-0.5.4-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl", hash = "sha256:bc11d7e8c44a65115d05e2ab9989d1e045125d7be8e05a071a48bc76eb6d6040", size = 5056165, upload-time = "2025-11-17T22:31:41.071Z" },
./uv.lock-2174-    { url = "https://files.pythonhosted.org/packages/a9/80/19189ea605017473660e43762dc853d2797984b3c7bf30ce656099add30c/ml_dtypes-0.5.4-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", hash = "sha256:19b9a53598f21e453ea2fbda8aa783c20faff8e1eeb0d7ab899309a0053f1483", size = 5034975, upload-time = "2025-11-17T22:31:42.758Z" },
./uv.lock-2175-    { url = "https://files.pythonhosted.org/packages/b4/24/70bd59276883fdd91600ca20040b41efd4902a923283c4d6edcb1de128d2/ml_dtypes-0.5.4-cp311-cp311-win_amd64.whl", hash = "sha256:7c23c54a00ae43edf48d44066a7ec31e05fdc2eee0be2b8b50dd1903a1db94bb", size = 210742, upload-time = "2025-11-17T22:31:44.068Z" },
./uv.lock-2176-    { url = "https://files.pythonhosted.org/packages/a0/c9/64230ef14e40aa3f1cb254ef623bf812735e6bec7772848d19131111ac0d/ml_dtypes-0.5.4-cp311-cp311-win_arm64.whl", hash = "sha256:557a31a390b7e9439056644cb80ed0735a6e3e3bb09d67fd5687e4b04238d1de", size = 160709, upload-time = "2025-11-17T22:31:46.557Z" },
--
./uv.lock-2660-    { url = "https://files.pythonhosted.org/packages/c4/d3/b7da1d5d7dbdc5ef52ed7debd2b484313b832982266905315dad5a0bf0b1/pandas-3.0.2-cp311-cp311-macosx_11_0_arm64.whl", hash = "sha256:dbbd4aa20ca51e63b53bbde6a0fa4254b1aaabb74d2f542df7a7959feb1d760c", size = 9926987, upload-time = "2026-03-31T06:46:11.724Z" },
./uv.lock-2661-    { url = "https://files.pythonhosted.org/packages/52/77/9b1c2d6070b5dbe239a7bc889e21bfa58720793fb902d1e070695d87c6d0/pandas-3.0.2-cp311-cp311-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl", hash = "sha256:339dda302bd8369dedeae979cb750e484d549b563c3f54f3922cb8ff4978c5eb", size = 10757067, upload-time = "2026-03-31T06:46:14.903Z" },
./uv.lock-2662-    { url = "https://files.pythonhosted.org/packages/20/17/ec40d981705654853726e7ac9aea9ddbb4a5d9cf54d8472222f4f3de06c2/pandas-3.0.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl", hash = "sha256:61c2fd96d72b983a9891b2598f286befd4ad262161a609c92dc1652544b46b76", size = 11258787, upload-time = "2026-03-31T06:46:17.683Z" },
./uv.lock:2663:    { url = "https://files.pythonhosted.org/packages/90/e3/3f1126d43d3702ca8773871a81c9f15122a1f412342cc56284ffda5b1f70/pandas-3.0.2-cp311-cp311-musllinux_1_2_aarch64.whl", hash = "sha256:c934008c733b8bbea273ea308b73b3156f0181e5b72960790b09c18a2794fe1e", size = 11771616, upload-time = "2026-03-31T06:46:20.532Z" },
./uv.lock-2664-    { url = "https://files.pythonhosted.org/packages/2e/cf/0f4e268e1f5062e44a6bda9f925806721cd4c95c2b808a4c82ebe914f96b/pandas-3.0.2-cp311-cp311-musllinux_1_2_x86_64.whl", hash = "sha256:60a80bb4feacbef5e1447a3f82c33209c8b7e07f28d805cfd1fb951e5cb443aa", size = 12337623, upload-time = "2026-03-31T06:46:23.754Z" },
./uv.lock-2665-    { url = "https://files.pythonhosted.org/packages/44/a0/97a6339859d4acb2536efb24feb6708e82f7d33b2ed7e036f2983fcced82/pandas-3.0.2-cp311-cp311-win_amd64.whl", hash = "sha256:ed72cb3f45190874eb579c64fa92d9df74e98fd63e2be7f62bce5ace0ade61df", size = 9897372, upload-time = "2026-03-31T06:46:26.703Z" },
./uv.lock-2666-    { url = "https://files.pythonhosted.org/packages/8f/eb/781516b808a99ddf288143cec46b342b3016c3414d137da1fdc3290d8860/pandas-3.0.2-cp311-cp311-win_arm64.whl", hash = "sha256:f12b1a9e332c01e09510586f8ca9b108fd631fd656af82e452d7315ef6df5f9f", size = 9154922, upload-time = "2026-03-31T06:46:30.284Z" },
--
./uv.lock-2779-    { url = "https://files.pythonhosted.org/packages/5e/26/d325f9f56c7e039034897e7380e9cc202b1e368bfd04d4cbe6a441f02885/pillow-12.2.0-cp314-cp314-musllinux_1_2_aarch64.whl", hash = "sha256:9aba9a17b623ef750a4d11b742cbafffeb48a869821252b30ee21b5e91392c50", size = 6507628, upload-time = "2026-04-01T14:45:12.378Z" },
./uv.lock-2780-    { url = "https://files.pythonhosted.org/packages/5f/f7/769d5632ffb0988f1c5e7660b3e731e30f7f8ec4318e94d0a5d674eb65a4/pillow-12.2.0-cp314-cp314-musllinux_1_2_x86_64.whl", hash = "sha256:deede7c263feb25dba4e82ea23058a235dcc2fe1f6021025dc71f2b618e26104", size = 7209321, upload-time = "2026-04-01T14:45:15.122Z" },
./uv.lock-2781-    { url = "https://files.pythonhosted.org/packages/6a/7a/c253e3c645cd47f1aceea6a8bacdba9991bf45bb7dfe927f7c893e89c93c/pillow-12.2.0-cp314-cp314-win32.whl", hash = "sha256:632ff19b2778e43162304d50da0181ce24ac5bb8180122cbe1bf4673428328c7", size = 6479723, upload-time = "2026-04-01T14:45:17.797Z" },
./uv.lock:2782:    { url = "https://files.pythonhosted.org/packages/cd/8b/601e6566b957ca50e28725cb6c355c59c2c8609751efbecd980db44e0349/pillow-12.2.0-cp314-cp314-win_amd64.whl", hash = "sha256:4e6c62e9d237e9b65fac06857d511e90d8461a32adcc1b9065ea0c0fa3a28150", size = 7217400, upload-time = "2026-04-01T14:45:20.529Z" },
./uv.lock-2783-    { url = "https://files.pythonhosted.org/packages/d6/94/220e46c73065c3e2951bb91c11a1fb636c8c9ad427ac3ce7d7f3359b9b2f/pillow-12.2.0-cp314-cp314-win_arm64.whl", hash = "sha256:b1c1fbd8a5a1af3412a0810d060a78b5136ec0836c8a4ef9aa11807f2a22f4e1", size = 2554835, upload-time = "2026-04-01T14:45:23.162Z" },
./uv.lock-2784-    { url = "https://files.pythonhosted.org/packages/b6/ab/1b426a3974cb0e7da5c29ccff4807871d48110933a57207b5a676cccc155/pillow-12.2.0-cp314-cp314t-macosx_10_15_x86_64.whl", hash = "sha256:57850958fe9c751670e49b2cecf6294acc99e562531f4bd317fa5ddee2068463", size = 5314225, upload-time = "2026-04-01T14:45:25.637Z" },
./uv.lock-2785-    { url = "https://files.pythonhosted.org/packages/19/1e/dce46f371be2438eecfee2a1960ee2a243bbe5e961890146d2dee1ff0f12/pillow-12.2.0-cp314-cp314t-macosx_11_0_arm64.whl", hash = "sha256:d5d38f1411c0ed9f97bcb49b7bd59b6b7c314e0e27420e34d99d844b9ce3b6f3", size = 4698541, upload-time = "2026-04-01T14:45:28.355Z" },
--
./uv.lock-2864-    { url = "https://files.pythonhosted.org/packages/9e/f8/91c27b22ccda1dbc7967f921c42825564fa5336a01ecd72eb78a9f4f53c2/propcache-0.4.1-cp311-cp311-musllinux_1_2_armv7l.whl", hash = "sha256:67fad6162281e80e882fb3ec355398cf72864a54069d060321f6cd0ade95fe85", size = 202064, upload-time = "2025-10-08T19:46:36.993Z" },
./uv.lock-2865-    { url = "https://files.pythonhosted.org/packages/f2/26/7f00bd6bd1adba5aafe5f4a66390f243acab58eab24ff1a08bebb2ef9d40/propcache-0.4.1-cp311-cp311-musllinux_1_2_ppc64le.whl", hash = "sha256:f10207adf04d08bec185bae14d9606a1444715bc99180f9331c9c02093e1959e", size = 212429, upload-time = "2025-10-08T19:46:38.398Z" },
./uv.lock-2866-    { url = "https://files.pythonhosted.org/packages/84/89/fd108ba7815c1117ddca79c228f3f8a15fc82a73bca8b142eb5de13b2785/propcache-0.4.1-cp311-cp311-musllinux_1_2_s390x.whl", hash = "sha256:e9b0d8d0845bbc4cfcdcbcdbf5086886bc8157aa963c31c777ceff7846c77757", size = 216727, upload-time = "2025-10-08T19:46:39.732Z" },
./uv.lock:2867:    { url = "https://files.pythonhosted.org/packages/79/37/3ec3f7e3173e73f1d600495d8b545b53802cbf35506e5732dd8578db3724/propcache-0.4.1-cp311-cp311-musllinux_1_2_x86_64.whl", hash = "sha256:981333cb2f4c1896a12f4ab92a9cc8f09ea664e9b7dbdc4eff74627af3a11c0f", size = 205097, upload-time = "2025-10-08T19:46:41.025Z" },
./uv.lock-2868-    { url = "https://files.pythonhosted.org/packages/61/b0/b2631c19793f869d35f47d5a3a56fb19e9160d3c119f15ac7344fc3ccae7/propcache-0.4.1-cp311-cp311-win32.whl", hash = "sha256:f1d2f90aeec838a52f1c1a32fe9a619fefd5e411721a9117fbf82aea638fe8a1", size = 38084, upload-time = "2025-10-08T19:46:42.693Z" },
./uv.lock-2869-    { url = "https://files.pythonhosted.org/packages/f4/78/6cce448e2098e9f3bfc91bb877f06aa24b6ccace872e39c53b2f707c4648/propcache-0.4.1-cp311-cp311-win_amd64.whl", hash = "sha256:364426a62660f3f699949ac8c621aad6977be7126c5807ce48c0aeb8e7333ea6", size = 41637, upload-time = "2025-10-08T19:46:43.778Z" },
./uv.lock-2870-    { url = "https://files.pythonhosted.org/packages/9c/e9/754f180cccd7f51a39913782c74717c581b9cc8177ad0e949f4d51812383/propcache-0.4.1-cp311-cp311-win_arm64.whl", hash = "sha256:e53f3a38d3510c11953f3e6a33f205c6d1b001129f972805ca9b42fc308bc239", size = 38064, upload-time = "2025-10-08T19:46:44.872Z" },
--
./uv.lock-2906-    { url = "https://files.pythonhosted.org/packages/36/1d/fc272a63c8d3bbad6878c336c7a7dea15e8f2d23a544bda43205dfa83ada/propcache-0.4.1-cp313-cp313t-manylinux2014_s390x.manylinux_2_17_s390x.manylinux_2_28_s390x.whl", hash = "sha256:af223b406d6d000830c6f65f1e6431783fc3f713ba3e6cc8c024d5ee96170a4b", size = 280420, upload-time = "2025-10-08T19:47:36.338Z" },
./uv.lock-2907-    { url = "https://files.pythonhosted.org/packages/07/0c/01f2219d39f7e53d52e5173bcb09c976609ba30209912a0680adfb8c593a/propcache-0.4.1-cp313-cp313t-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl", hash = "sha256:a78372c932c90ee474559c5ddfffd718238e8673c340dc21fe45c5b8b54559a0", size = 263254, upload-time = "2025-10-08T19:47:37.692Z" },
./uv.lock-2908-    { url = "https://files.pythonhosted.org/packages/2d/18/cd28081658ce597898f0c4d174d4d0f3c5b6d4dc27ffafeef835c95eb359/propcache-0.4.1-cp313-cp313t-musllinux_1_2_aarch64.whl", hash = "sha256:564d9f0d4d9509e1a870c920a89b2fec951b44bf5ba7d537a9e7c1ccec2c18af", size = 261205, upload-time = "2025-10-08T19:47:39.659Z" },
./uv.lock:2909:    { url = "https://files.pythonhosted.org/packages/7a/71/1f9e22eb8b8316701c2a19fa1f388c8a3185082607da8e406a803c9b954e/propcache-0.4.1-cp313-cp313t-musllinux_1_2_armv7l.whl", hash = "sha256:17612831fda0138059cc5546f4d12a2aacfb9e47068c06af35c400ba58ba7393", size = 247873, upload-time = "2025-10-08T19:47:41.084Z" },
./uv.lock-2910-    { url = "https://files.pythonhosted.org/packages/4a/65/3d4b61f36af2b4eddba9def857959f1016a51066b4f1ce348e0cf7881f58/propcache-0.4.1-cp313-cp313t-musllinux_1_2_ppc64le.whl", hash = "sha256:41a89040cb10bd345b3c1a873b2bf36413d48da1def52f268a055f7398514874", size = 262739, upload-time = "2025-10-08T19:47:42.51Z" },
./uv.lock-2911-    { url = "https://files.pythonhosted.org/packages/2a/42/26746ab087faa77c1c68079b228810436ccd9a5ce9ac85e2b7307195fd06/propcache-0.4.1-cp313-cp313t-musllinux_1_2_s390x.whl", hash = "sha256:e35b88984e7fa64aacecea39236cee32dd9bd8c55f57ba8a75cf2399553f9bd7", size = 263514, upload-time = "2025-10-08T19:47:43.927Z" },
./uv.lock-2912-    { url = "https://files.pythonhosted.org/packages/94/13/630690fe201f5502d2403dd3cfd451ed8858fe3c738ee88d095ad2ff407b/propcache-0.4.1-cp313-cp313t-musllinux_1_2_x86_64.whl", hash = "sha256:6f8b465489f927b0df505cbe26ffbeed4d6d8a2bbc61ce90eb074ff129ef0ab1", size = 257781, upload-time = "2025-10-08T19:47:45.448Z" },
--
./uv.lock-3085-    { url = "https://files.pythonhosted.org/packages/f8/85/c2b1706e51942de19076eff082f8495e57d5151364e78b5bef4af4a1d94a/pyproj-3.7.2-cp313-cp313-manylinux_2_28_x86_64.whl", hash = "sha256:5141a538ffdbe4bfd157421828bb2e07123a90a7a2d6f30fa1462abcfb5ce681", size = 9514269, upload-time = "2025-08-14T12:04:34.599Z" },
./uv.lock-3086-    { url = "https://files.pythonhosted.org/packages/34/38/07a9b89ae7467872f9a476883a5bad9e4f4d1219d31060f0f2b282276cbe/pyproj-3.7.2-cp313-cp313-musllinux_1_2_aarch64.whl", hash = "sha256:f000841e98ea99acbb7b8ca168d67773b0191de95187228a16110245c5d954d5", size = 10808437, upload-time = "2025-08-14T12:04:36.485Z" },
./uv.lock-3087-    { url = "https://files.pythonhosted.org/packages/12/56/fda1daeabbd39dec5b07f67233d09f31facb762587b498e6fc4572be9837/pyproj-3.7.2-cp313-cp313-musllinux_1_2_x86_64.whl", hash = "sha256:8115faf2597f281a42ab608ceac346b4eb1383d3b45ab474fd37341c4bf82a67", size = 10745540, upload-time = "2025-08-14T12:04:38.568Z" },
./uv.lock:3088:    { url = "https://files.pythonhosted.org/packages/0d/90/c793182cbba65a39a11db2ac6b479fe76c59e6509ae75e5744c344a0da9d/pyproj-3.7.2-cp313-cp313-win32.whl", hash = "sha256:f18c0579dd6be00b970cb1a6719197fceecc407515bab37da0066f0184aafdf3", size = 5896506, upload-time = "2025-08-14T12:04:41.059Z" },
./uv.lock-3089-    { url = "https://files.pythonhosted.org/packages/be/0f/747974129cf0d800906f81cd25efd098c96509026e454d4b66868779ab04/pyproj-3.7.2-cp313-cp313-win_amd64.whl", hash = "sha256:bb41c29d5f60854b1075853fe80c58950b398d4ebb404eb532536ac8d2834ed7", size = 6310195, upload-time = "2025-08-14T12:04:42.974Z" },
./uv.lock-3090-    { url = "https://files.pythonhosted.org/packages/82/64/fc7598a53172c4931ec6edf5228280663063150625d3f6423b4c20f9daff/pyproj-3.7.2-cp313-cp313-win_arm64.whl", hash = "sha256:2b617d573be4118c11cd96b8891a0b7f65778fa7733ed8ecdb297a447d439100", size = 6230748, upload-time = "2025-08-14T12:04:44.491Z" },
./uv.lock-3091-    { url = "https://files.pythonhosted.org/packages/aa/f0/611dd5cddb0d277f94b7af12981f56e1441bf8d22695065d4f0df5218498/pyproj-3.7.2-cp313-cp313t-macosx_13_0_x86_64.whl", hash = "sha256:d27b48f0e81beeaa2b4d60c516c3a1cfbb0c7ff6ef71256d8e9c07792f735279", size = 6241729, upload-time = "2025-08-14T12:04:46.274Z" },
--
./scripts/bench/bench_mpas_spmd_scaling.py-103-    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
./scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
./scripts/bench/bench_mpas_spmd_scaling.py-105-    device count of THIS run's mesh/model (the two differ for the
./scripts/bench/bench_mpas_spmd_scaling.py:106:    single-device reference leg of a ladder, via ``--reorder-for``).
./scripts/bench/bench_mpas_spmd_scaling.py-107-    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
./scripts/bench/bench_mpas_spmd_scaling.py-108-    wave) so the sharded step's packed tracer halo exchange + RK tracer
./scripts/bench/bench_mpas_spmd_scaling.py-109-    advection sit on the timed/gated path.
--
./scripts/bench/bench_mpas_spmd_scaling.py-125-        raise SystemExit(
./scripts/bench/bench_mpas_spmd_scaling.py-126-            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
./scripts/bench/bench_mpas_spmd_scaling.py-127-            f"divisible by --n-devices {run_nd}; use a ladder where every "
./scripts/bench/bench_mpas_spmd_scaling.py:128:            f"count divides --reorder-for ({reorder_target}).")
./scripts/bench/bench_mpas_spmd_scaling.py-129-    sigma = create_sigma_coordinate(nlev)
./scripts/bench/bench_mpas_spmd_scaling.py-130-    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
./scripts/bench/bench_mpas_spmd_scaling.py-131-    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
--
./scripts/bench/bench_mpas_spmd_scaling.py-186-                        "mesh (scaling receipts only, never physics — "
./scripts/bench/bench_mpas_spmd_scaling.py-187-                        "must match the prewarmed cache key at subdiv>=9).")
./scripts/bench/bench_mpas_spmd_scaling.py-188-    p.add_argument("--n-devices", type=int, required=True)
./scripts/bench/bench_mpas_spmd_scaling.py:189:    p.add_argument("--reorder-for", type=int, default=None,
./scripts/bench/bench_mpas_spmd_scaling.py-190-                   help="partition/reorder the mesh for THIS device count "
./scripts/bench/bench_mpas_spmd_scaling.py-191-                        "(default: --n-devices). Pin it to the ladder's "
./scripts/bench/bench_mpas_spmd_scaling.py-192-                        "max so single-device reference runs time the "
--
./scripts/bench/bench_mpas_spmd_scaling.py-308-    reorder_for = args.reorder_for if args.reorder_for is not None else nd
./scripts/bench/bench_mpas_spmd_scaling.py-309-    if reorder_for < nd:
./scripts/bench/bench_mpas_spmd_scaling.py-310-        raise SystemExit(
./scripts/bench/bench_mpas_spmd_scaling.py:311:            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
./scripts/bench/bench_mpas_spmd_scaling.py-312-            f"the ghost padding only guarantees divisibility for the "
./scripts/bench/bench_mpas_spmd_scaling.py-313-            f"partition target.")
./scripts/bench/bench_mpas_spmd_scaling.py-314-    mesh, model, s0, dev_config = build_model_and_state(
--
./scripts/run/run_omip_core2.py-1237-                                     # basin past its 47N/54E geographic edge
./scripts/run/run_omip_core2.py-1238-                                     # (ignition at 48.5N/51E, job 8486227).)
./scripts/run/run_omip_core2.py-1239-        (43.0, 48.0, 57.0, 62.0),    # Aral Sea (endorheic)
./scripts/run/run_omip_core2.py:1240:        (41.0, 49.0, 268.0, 285.0),  # Great Lakes (inland; no-op if WOA-land)
./scripts/run/run_omip_core2.py-1241-    ]
./scripts/run/run_omip_core2.py-1242-    n_before = int(out.sum())
./scripts/run/run_omip_core2.py-1243-    for lat0, lat1, lon0, lon1 in boxes:
--
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-11-#SBATCH --output=mpas_s8_l0.%j.log
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-12-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:18:# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-21-#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
--
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-48-      --multicontroller --n-devices "$NP" \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:49:      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:53:echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-54-for NP in 8 16 32; do
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-55-  F="$OUTDIR/np${NP}.jsonl"
./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-56-  "$PY" -c "
--
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-61-RUN_MPAS="${RUN_MPAS:-1}"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-62-# Lat-band lanes C/D task count (8 = 2 nodes; 16 = sbatch --nodes=4).
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-63-NP_LL="${NP_LL:-8}"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:64:# Lane E device count; >6 pads the mesh via --reorder-for (even split).
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-65-NP_MPAS="${NP_MPAS:-6}"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-66-# Slurm job STEPS do not inherit the job's GPU allocation on all Slurm
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-67-# versions/configs — without an explicit step gres some tasks see zero
--
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-198-if [ "$RUN_MPAS" = "1" ]; then
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-199-    echo "=== LANE E: icosahedral MPAS multicontroller np=$NP_MPAS (L$ICO_LEVEL) ==="
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-200-    # >6 devices: 10*4^L+2 = 2*odd has no even split beyond 6 — pad the mesh
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:201:    # to the target count via --reorder-for (identical padded mesh per ladder).
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-202-    E_TPN=3; [ $((NP_MPAS % 4)) -eq 0 ] && E_TPN=4
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-203-    E_REORDER=""
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:204:    [ "$NP_MPAS" -gt 6 ] && E_REORDER="--reorder-for $NP_MPAS"
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-205-    # shellcheck disable=SC2086
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-206-    srun --ntasks="$NP_MPAS" --ntasks-per-node="$E_TPN" $STEP_GPU_OPTS --kill-on-bad-exit=1 bash -c "$PIN" _ \
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-207-        "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
--
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-34-#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:36:# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
--
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-68-    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-69-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-70-      --multicontroller --n-devices 32 \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:71:      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:72:      --partition-method sfc --reorder-for 128 \
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-73-      --out "$OUTDIR/$1.jsonl"
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-74-  s=$?
./scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-75-  echo "[$1] exit=$s epoch=$(date +%s.%N)"
--
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-9-#SBATCH --output=prewarm_s10.%j.log
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-10-# Prewarm subdiv-10 lloyd=0 (10.5M cells, ~7 GB npz) into the shared
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-11-# mesh cache — unlocks MPAS at 128-224 GPUs ABOVE the ~30k tile floor
./scripts/cluster/scaling_levante/prewarm_s10.sbatch:12:# (np128 = 81.9k, np224 = 46.8k cells/GPU).  s9 (2.62M) built in 67 min;
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-13-# s10 estimated ~4-5 h.
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-14-set -uo pipefail
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-15-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
--
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-19-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-20-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
./scripts/cluster/scaling_levante/prewarm_s10.sbatch-21-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
./scripts/cluster/scaling_levante/prewarm_s10.sbatch:22:"$PY" scripts/data/prewarm_voronoi_mesh.py --level 10 --lloyd 0
--
./data/les_cases/LBA/snd-28-11330.0000  230.1000  345.6043    0.1231   -4.2600    2.6600
./data/les_cases/LBA/snd-29-11835.9990  213.2000  347.0636    0.0781   -7.5200    4.7900
./data/les_cases/LBA/snd-30-12342.0000  197.0000  347.6107    0.0500   -8.8800    3.4000
./data/les_cases/LBA/snd:31:12841.0000  182.3000  348.0988    0.0326   -9.0000    3.1400
./data/les_cases/LBA/snd-32-13348.0000  167.9000  348.9892    0.0175   -7.7700    3.9300
./data/les_cases/LBA/snd:33:13841.0000  154.9000  350.1689    0.0092   -5.3700    7.5700
./data/les_cases/LBA/snd-34-14313.0000  143.0000  352.8745    0.0055   -3.8800    2.5800
./data/les_cases/LBA/snd-35-14826.0000  131.1000  355.2420    0.0030   -1.1500    2.5000
./data/les_cases/LBA/snd-36-15328.0000  119.7000  358.9141    0.0017   -2.3600    6.4400
./data/les_cases/LBA/snd-37-15888.9990  108.9000  362.8484    0.0011   -9.2000    6.8400
./data/les_cases/LBA/snd-38-16361.0000  100.1000  371.5976    0.0011   -8.0100    0.1900
./data/les_cases/LBA/snd:39:16831.0020   92.1000  381.9160    0.0012   -5.6800   -2.2000
./data/les_cases/LBA/snd-40-17312.0000   84.6000  388.5070    0.0011   -8.8300   -3.6000
./data/les_cases/LBA/snd-41-17804.0000   77.5000  398.8011    0.0011  -14.5100    0.5600
./data/les_cases/LBA/snd-42-18267.0000   71.4000  414.3179    0.0016  -15.5500    6.6800
--
./data/les_cases/LBA/snd-76-11330.0000  230.1000  345.6043    0.1231   -4.2600    2.6600
./data/les_cases/LBA/snd-77-11835.9990  213.2000  347.0636    0.0781   -7.5200    4.7900
./data/les_cases/LBA/snd-78-12342.0000  197.0000  347.6107    0.0500   -8.8800    3.4000
./data/les_cases/LBA/snd:79:12841.0000  182.3000  348.0988    0.0326   -9.0000    3.1400
./data/les_cases/LBA/snd-80-13348.0000  167.9000  348.9892    0.0175   -7.7700    3.9300
./data/les_cases/LBA/snd:81:13841.0000  154.9000  350.1689    0.0092   -5.3700    7.5700
./data/les_cases/LBA/snd-82-14313.0000  143.0000  352.8745    0.0055   -3.8800    2.5800
./data/les_cases/LBA/snd-83-14826.0000  131.1000  355.2420    0.0030   -1.1500    2.5000
./data/les_cases/LBA/snd-84-15328.0000  119.7000  358.9141    0.0017   -2.3600    6.4400
./data/les_cases/LBA/snd-85-15888.9990  108.9000  362.8484    0.0011   -9.2000    6.8400
./data/les_cases/LBA/snd-86-16361.0000  100.1000  371.5976    0.0011   -8.0100    0.1900
./data/les_cases/LBA/snd:87:16831.0020   92.1000  381.9160    0.0012   -5.6800   -2.2000
./data/les_cases/LBA/snd-88-17312.0000   84.6000  388.5070    0.0011   -8.8300   -3.6000
./data/les_cases/LBA/snd-89-17804.0000   77.5000  398.8011    0.0011  -14.5100    0.5600
./data/les_cases/LBA/snd-90-18267.0000   71.4000  414.3179    0.0016  -15.5500    6.6800
--
./data/les_cases/BOMEX/grd_74-21-   1503.486              21   136.6805    
./data/les_cases/BOMEX/grd_74-22-   1653.834              22   150.3486    
./data/les_cases/BOMEX/grd_74-23-   1819.218              23   165.3835    
./data/les_cases/BOMEX/grd_74:24:   2001.140              24   181.9218    
./data/les_cases/BOMEX/grd_74-25-   2201.311              25   200.1709    
./data/les_cases/BOMEX/grd_74-26-   2431.507              26   230.1966    
./data/les_cases/BOMEX/grd_74-27-   2696.233              27   264.7261    
--
./packages/ocean/legoesm/ocean/bathymetry.py-50-    ("Hormuz",                    26.5,  56.5,  40.0,  100.0),
./packages/ocean/legoesm/ocean/bathymetry.py-51-    ("Malacca",                    2.5, 101.5,  40.0,   50.0),
./packages/ocean/legoesm/ocean/bathymetry.py-52-    ("Indonesian Throughflow",    -3.0, 120.0, 200.0, 1500.0),
./packages/ocean/legoesm/ocean/bathymetry.py:53:    ("Mozambique Channel",       -17.0,  41.0, 200.0, 2500.0),
./packages/ocean/legoesm/ocean/bathymetry.py-54-    ("Denmark Strait",            66.0, -27.0, 150.0,  600.0),
./packages/ocean/legoesm/ocean/bathymetry.py-55-    ("Faroe Bank Channel",        61.5,  -8.5, 100.0,  800.0),
./packages/ocean/legoesm/ocean/bathymetry.py-56-    ("Bering Strait",             65.8,-169.0,  60.0,   40.0),
--
./packages/ocean/legoesm/ocean/bathymetry.py-59-    ("Taiwan Strait",             24.0, 119.5,  80.0,   60.0),
./packages/ocean/legoesm/ocean/bathymetry.py-60-    ("Windward Passage",          20.0, -73.5,  60.0, 1500.0),
./packages/ocean/legoesm/ocean/bathymetry.py-61-    ("Florida Strait",            25.5, -79.5, 100.0,  800.0),
./packages/ocean/legoesm/ocean/bathymetry.py:62:    ("Luzon Strait",              20.5, 121.5, 100.0, 2000.0),
./packages/ocean/legoesm/ocean/bathymetry.py-63-]
./packages/ocean/legoesm/ocean/bathymetry.py-64-
./packages/ocean/legoesm/ocean/bathymetry.py-65-
--
./scripts/validate/compare_mie_vs_microhh.py-89-  print(f"     {'band':>4} {'r_eff':>6} {'g_mc':>7} {'g_ref':>7} "
./scripts/validate/compare_mie_vs_microhh.py-90-        f"{'<ang>_mc':>9} {'<ang>_ref':>10} {'L1':>6}")
./scripts/validate/compare_mie_vs_microhh.py-91-  for b in (1, 7, 13):
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
session id: 019fc236-6b9b-76b1-b8af-1eeede817a22
--------
user
Round-4 final check. Your round-3 had 2 items: (1) doc ensemble wording contradicted v3 script (counterbalanced claim, stale job id 26628021, refute-bar attribution); (2) CPU r512 ladder needed unsupported-stack qualification. Both now patched in docs/performance/scaling/levante_campaign_2026-07-24.md: 'BRACKETED, not fully counterbalanced' + follow-up attribution language at the ensemble bullet AND its REFUTE line, job table row now 26628196 (v3, supersessions noted), CPU receipt labelled TIMING-ONLY/unvalidated with the mpi4jax 0.9.0 out-of-tested-range + UCX VM_UNMAP quotes. Read the '### Next receipts' + CPU-receipt sections and confirm consistency with scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch. Anything still inconsistent? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll cross-check the two documentation sections against the v3 ensemble submission script, focusing only on the prior discrepancies and any remaining contradictory claims.
exec
/usr/bin/bash -lc 'pwd && rg -n -C 10 "Next receipts|CPU|ensemble|26628196|26628021|TIMING-ONLY|mpi4jax|VM_UNMAP|counterbalanced|BRACKETED|REFUTE" docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 335ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-24-#   solo_post : solo again AFTER phase B (brackets ordering/thermal
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-25-#               drift; contrast uses mean of the two solos)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-26-#
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-27-# Falsifiability, written BEFORE submit:
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-28-#   numbers : 2 solo + 4 replica steady_median_ms
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-29-#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-30-#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-31-#             reproduces 5.47) = ~3.3x the observed 128-GPU
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-32-#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-33-#             margin below it is the measured contention.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:34:#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-35-#             per replica.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-36-# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-37-# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-38-# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-39-# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-40-# table sacct prints at the end.  Absolute ms/step is therefore only
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-41-# compared WITHIN this job (solo vs replicas), never against the
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-42-# steps-12 ladder rows.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-43-# INTERPRETATION LIMIT (codex r21 item 4): this design SCREENS for a
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-44-# co-execution penalty vs the solo brackets; if a penalty appears, its
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-45-# attribution (shared IB fabric vs node/topology placement vs drift)
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-46-# needs the per-step nodelist table + follow-up, and the solo/replica
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:47:# ordering is bracketed (pre+post), not fully counterbalanced.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-48-set -uo pipefail
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-49-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-50-export JAX_PLATFORMS=cuda,cpu
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-51-export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-52-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-53-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-54-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-55-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-56-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s9_ens_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-57-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
--
docs/performance/scaling/levante_campaign_2026-07-24.md-5-fixing what broke, and moving the worst axis (ocean strong scaling) to a
docs/performance/scaling/levante_campaign_2026-07-24.md-6-measured 2× improvement. All receipts on the post-merge tree `d3ec1ccce`+
docs/performance/scaling/levante_campaign_2026-07-24.md-7-(campaign branch `worktree-scaling-campaign`); job IDs cited throughout are
docs/performance/scaling/levante_campaign_2026-07-24.md-8-Levante SLURM jobs from 2026-07-24. Codex adversarial review: 4 rounds
docs/performance/scaling/levante_campaign_2026-07-24.md-9-(transcripts under `.physics-validator/scaling_campaign/`); every
docs/performance/scaling/levante_campaign_2026-07-24.md-10-measurement claim below carries the round-3 corrections.
docs/performance/scaling/levante_campaign_2026-07-24.md-11-
docs/performance/scaling/levante_campaign_2026-07-24.md-12-Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
docs/performance/scaling/levante_campaign_2026-07-24.md-13-`compute` (2× AMD Milan 7763). All GPU multinode = route-B
docs/performance/scaling/levante_campaign_2026-07-24.md-14-(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
docs/performance/scaling/levante_campaign_2026-07-24.md:15:route-A CUDA-aware mpi4jax not exercised on Levante).
docs/performance/scaling/levante_campaign_2026-07-24.md-16-
docs/performance/scaling/levante_campaign_2026-07-24.md-17-## Headline results (strong scaling, f32 unless noted)
docs/performance/scaling/levante_campaign_2026-07-24.md-18-
docs/performance/scaling/levante_campaign_2026-07-24.md-19-| Axis | Ladder | Result | Job(s) |
docs/performance/scaling/levante_campaign_2026-07-24.md-20-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-21-| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
docs/performance/scaling/levante_campaign_2026-07-24.md-22-| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
docs/performance/scaling/levante_campaign_2026-07-24.md-23-| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
docs/performance/scaling/levante_campaign_2026-07-24.md-24-| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
docs/performance/scaling/levante_campaign_2026-07-24.md-25-| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
--
docs/performance/scaling/levante_campaign_2026-07-24.md-95-26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
docs/performance/scaling/levante_campaign_2026-07-24.md-96-Taking np2 as the base (it has the BEST per-device throughput, 430
docs/performance/scaling/levante_campaign_2026-07-24.md-97-Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
docs/performance/scaling/levante_campaign_2026-07-24.md-98-floor).
docs/performance/scaling/levante_campaign_2026-07-24.md-99-
docs/performance/scaling/levante_campaign_2026-07-24.md-100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
docs/performance/scaling/levante_campaign_2026-07-24.md-101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
docs/performance/scaling/levante_campaign_2026-07-24.md-102-than np2 while np8 is 2.5x faster than np4. Evidence gathered:
docs/performance/scaling/levante_campaign_2026-07-24.md-103-- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
docs/performance/scaling/levante_campaign_2026-07-24.md-104-  17.14/17.04, 6.87/6.97).
docs/performance/scaling/levante_campaign_2026-07-24.md:105:- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
docs/performance/scaling/levante_campaign_2026-07-24.md-106-  two nodes (17.12 ms), so node crossing is irrelevant.
docs/performance/scaling/levante_campaign_2026-07-24.md:107:- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
docs/performance/scaling/levante_campaign_2026-07-24.md-108-  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
docs/performance/scaling/levante_campaign_2026-07-24.md-109-  falling, and under 3 % of owned cells at every count.
docs/performance/scaling/levante_campaign_2026-07-24.md:110:- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
docs/performance/scaling/levante_campaign_2026-07-24.md-111-  device count is a compile-time property so the HLO matches what the GPUs
docs/performance/scaling/levante_campaign_2026-07-24.md-112-  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
docs/performance/scaling/levante_campaign_2026-07-24.md-113-  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
docs/performance/scaling/levante_campaign_2026-07-24.md-114-  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
docs/performance/scaling/levante_campaign_2026-07-24.md-115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
docs/performance/scaling/levante_campaign_2026-07-24.md-116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
docs/performance/scaling/levante_campaign_2026-07-24.md:117:- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
docs/performance/scaling/levante_campaign_2026-07-24.md-118-  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
docs/performance/scaling/levante_campaign_2026-07-24.md-119-  (geometric). Every method shows the same 2.5-3.1x jump.
docs/performance/scaling/levante_campaign_2026-07-24.md:120:- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
docs/performance/scaling/levante_campaign_2026-07-24.md-121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
docs/performance/scaling/levante_campaign_2026-07-24.md-122-  command-buffers-off — every arm within 1 %, none recovers np4.
docs/performance/scaling/levante_campaign_2026-07-24.md-123-  (The multi-output-fusion arm errored on an unsupported flag name and is
docs/performance/scaling/levante_campaign_2026-07-24.md-124-  not counted.)
docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
docs/performance/scaling/levante_campaign_2026-07-24.md-126-collective count, partition method, codegen env knobs). The cheap levers known to this campaign are exhausted; the remaining suspect — per-device kernel efficiency
docs/performance/scaling/levante_campaign_2026-07-24.md-127-for this shape — needs a GPU op-level profile (nsys / XLA op profile of
docs/performance/scaling/levante_campaign_2026-07-24.md-128-np4 vs np8), which is a separate instrumented project, not another timing
docs/performance/scaling/levante_campaign_2026-07-24.md-129-run. Per-GPU throughput across the ladder is non-monotone in tile size
docs/performance/scaling/levante_campaign_2026-07-24.md-130-(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
--
docs/performance/scaling/levante_campaign_2026-07-24.md-206-
docs/performance/scaling/levante_campaign_2026-07-24.md-207-Both ladders are monotone; the bigger tile is uniformly better at every
docs/performance/scaling/levante_campaign_2026-07-24.md-208-device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).
docs/performance/scaling/levante_campaign_2026-07-24.md-209-
docs/performance/scaling/levante_campaign_2026-07-24.md-210-So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
docs/performance/scaling/levante_campaign_2026-07-24.md-211-multinode improvement measured at LL576 was partly a floor effect, and at
docs/performance/scaling/levante_campaign_2026-07-24.md-212-a production tile the identical code scales substantially better (0.37 ->
docs/performance/scaling/levante_campaign_2026-07-24.md-213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
docs/performance/scaling/levante_campaign_2026-07-24.md-214-Config is byte-identical between the two rows; only the grid changes.
docs/performance/scaling/levante_campaign_2026-07-24.md-215-
docs/performance/scaling/levante_campaign_2026-07-24.md:216:REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
docs/performance/scaling/levante_campaign_2026-07-24.md-217-np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
docs/performance/scaling/levante_campaign_2026-07-24.md-218-compile-time cliff at that device count. It is not: the third attempt ran
docs/performance/scaling/levante_campaign_2026-07-24.md-219-the identical configuration in **99 seconds** with a 21.6 s compile (job
docs/performance/scaling/levante_campaign_2026-07-24.md-220-26457693). The earlier hangs were transient/environmental, not
docs/performance/scaling/levante_campaign_2026-07-24.md-221-reproducible, and no compile-time defect is claimed.
docs/performance/scaling/levante_campaign_2026-07-24.md-222-
docs/performance/scaling/levante_campaign_2026-07-24.md-223-## Weak scaling at production per-device size (job 26453523)
docs/performance/scaling/levante_campaign_2026-07-24.md-224-
docs/performance/scaling/levante_campaign_2026-07-24.md-225-The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
docs/performance/scaling/levante_campaign_2026-07-24.md-226-latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
--
docs/performance/scaling/levante_campaign_2026-07-24.md-256-- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
docs/performance/scaling/levante_campaign_2026-07-24.md-257-  to `explicit_substep`; our explicit/wide arm reproduces that class
docs/performance/scaling/levante_campaign_2026-07-24.md-258-  (0.88 @2, LL384) — the production implicit config was never measured
docs/performance/scaling/levante_campaign_2026-07-24.md-259-  there. So the gap is explained by the solver the old bench selected;
docs/performance/scaling/levante_campaign_2026-07-24.md-260-  labelling it 'protocol, not regression' is an inference from that
docs/performance/scaling/levante_campaign_2026-07-24.md-261-  config difference, not an independent bisect.
docs/performance/scaling/levante_campaign_2026-07-24.md-262-- No merge regression: nd=1 stock-CG LL192 8.90 ms pre-merge (job 26445836)
docs/performance/scaling/levante_campaign_2026-07-24.md-263-  vs 8.94 ms post-merge (smoke on tree d3ec1ccce) - one sample each, so
docs/performance/scaling/levante_campaign_2026-07-24.md-264-  this bounds a large regression only.
docs/performance/scaling/levante_campaign_2026-07-24.md-265-
docs/performance/scaling/levante_campaign_2026-07-24.md:266:## CPU-MPI (compute nodes)
docs/performance/scaling/levante_campaign_2026-07-24.md-267-
docs/performance/scaling/levante_campaign_2026-07-24.md-268-Single-node ladders (job 26445986, f64): atm latlon strong eff
docs/performance/scaling/levante_campaign_2026-07-24.md-269-0.87–0.92@np2 → 0.11–0.23@np32–64; ocean implicit 0.90@2 → 0.24@32; weak
docs/performance/scaling/levante_campaign_2026-07-24.md-270-collapses ≤0.14@32. CAVEATS: np=1 ocean leg was stock-CG (solver-mismatched
docs/performance/scaling/levante_campaign_2026-07-24.md-271-— fixed via `--force-pcg` in the job scripts; np≥2 slopes valid), and
docs/performance/scaling/levante_campaign_2026-07-24.md-272-single-node ladders conflate Milan DRAM contention with comm. The first
docs/performance/scaling/levante_campaign_2026-07-24.md-273-4-node pair collided into one OUTDIR (same-second stamp) and was discarded.
docs/performance/scaling/levante_campaign_2026-07-24.md-274-
docs/performance/scaling/levante_campaign_2026-07-24.md-275-4-node SPREAD ladder (job 26452578, f64, ranks round-robin, solver-matched):
docs/performance/scaling/levante_campaign_2026-07-24.md-276-atm latlon np2 eff ~1.00; spreading ranks over 4 nodes nearly doubles
docs/performance/scaling/levante_campaign_2026-07-24.md-277-efficiency at high rank counts (r128 np32: 0.38 spread vs 0.20 packed),
docs/performance/scaling/levante_campaign_2026-07-24.md-278-which is CONSISTENT with per-node memory-bandwidth contention in the
docs/performance/scaling/levante_campaign_2026-07-24.md-279-packed ladder (not isolated by a bandwidth counter), decaying to 0.06–0.16
docs/performance/scaling/levante_campaign_2026-07-24.md-280-at np64–128 — the 1-D band perimeter ceiling as designed (r256/np128 = 2
docs/performance/scaling/levante_campaign_2026-07-24.md-281-rows/rank). Ocean strong spread: np2 eff 1.32 (superlinear - typical of a base leg whose working set does
docs/performance/scaling/levante_campaign_2026-07-24.md-282-not fit cache; not instrumented here), 0.88@8,
docs/performance/scaling/levante_campaign_2026-07-24.md-283-0.35@32, wall at np64 (79 ms > np32's 74 ms). The job died in a high-rank
docs/performance/scaling/levante_campaign_2026-07-24.md-284-ocean case (one rank exit-3 → kill-on-bad-exit) before the weak tail —
docs/performance/scaling/levante_campaign_2026-07-24.md-285-np128 ocean + weak ladders and the rank-failure attribution remain open.
docs/performance/scaling/levante_campaign_2026-07-24.md-286-
docs/performance/scaling/levante_campaign_2026-07-24.md:287:## Atm ICOSAHEDRAL CPU-MPI (job 26452579, 1 node, f64 moist, L26)
docs/performance/scaling/levante_campaign_2026-07-24.md-288-
docs/performance/scaling/levante_campaign_2026-07-24.md-289-The last measurement gap, and the healthiest strong-scaling curve in the
docs/performance/scaling/levante_campaign_2026-07-24.md-290-campaign. Efficiency t1/(n*tn) by subdivision:
docs/performance/scaling/levante_campaign_2026-07-24.md-291-
docs/performance/scaling/levante_campaign_2026-07-24.md-292-| subdiv | np2 | np4 | np8 | np16 | np32 | np64 |
docs/performance/scaling/levante_campaign_2026-07-24.md-293-|---|---|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-294-| 4 | 0.90 | 0.76 | 1.00 | 0.53 | 0.26 | 0.15 |
docs/performance/scaling/levante_campaign_2026-07-24.md-295-| 5 | 1.02 | 0.88 | 1.15 | 0.56 | 0.58 | 0.30 |
docs/performance/scaling/levante_campaign_2026-07-24.md-296-| 6 | 1.02 | 0.88 | 1.07 | 0.51 | 0.54 | 0.57 |
docs/performance/scaling/levante_campaign_2026-07-24.md-297-| 7 | 1.03 | 0.92 | 1.02 | 0.51 | 0.51 | 0.52 |
docs/performance/scaling/levante_campaign_2026-07-24.md-298-
docs/performance/scaling/levante_campaign_2026-07-24.md-299-**REVISION 2026-07-27 — the subdiv-7 row was PLACEMENT-LIMITED, not
docs/performance/scaling/levante_campaign_2026-07-24.md-300-comm-limited.** Re-running it with `--distribution=block:cyclic` (the
docs/performance/scaling/levante_campaign_2026-07-24.md-301-Milan fix, discovered after this sweep) gives f64 np64 **efficiency 0.71,
docs/performance/scaling/levante_campaign_2026-07-24.md-302-up from 0.52**, and the high-rank columns move most: placement alone is
docs/performance/scaling/levante_campaign_2026-07-24.md-303-worth 2.00x at np16, 1.75x at np32, 1.38x at np64 (job 26495437 vs
docs/performance/scaling/levante_campaign_2026-07-24.md-304-26452579). The f32 ladder at matched placement (job 26495083) reaches
docs/performance/scaling/levante_campaign_2026-07-24.md-305-**0.88**. So the "np16 dip" visible across every row of this table is
docs/performance/scaling/levante_campaign_2026-07-24.md:306:substantially the same NUMA effect found later in the packed CPU atm
docs/performance/scaling/levante_campaign_2026-07-24.md-307-ladder — one fix, two symptoms. Precision itself is worth a near-constant
docs/performance/scaling/levante_campaign_2026-07-24.md-308-~1.4x here; the naive cross-job comparison would have read 2.81x at np16
docs/performance/scaling/levante_campaign_2026-07-24.md-309-and attributed placement to precision.
docs/performance/scaling/levante_campaign_2026-07-24.md-310-| 8 | 1.04 | 0.93 | 1.18 | 0.61 | 0.49 | — |
docs/performance/scaling/levante_campaign_2026-07-24.md-311-
docs/performance/scaling/levante_campaign_2026-07-24.md-312-THE TILE-SIZE PATTERN, THIRD LANE (cube and ico are both atmosphere:
docs/performance/scaling/levante_campaign_2026-07-24.md-313-two components, three decomposition lanes): the np64 column collapses
docs/performance/scaling/levante_campaign_2026-07-24.md-314-on coarse grids (0.15 at subdiv4, 0.30 at subdiv5) and holds on fine ones
docs/performance/scaling/levante_campaign_2026-07-24.md-315-(0.52-0.57 at subdiv6-7). Same pattern as the cube (0.23 -> 1.04) and the ocean (0.37 -> 0.63), now
docs/performance/scaling/levante_campaign_2026-07-24.md:316:on a third lane and a different transport (CPU-MPI, not NCCL). It is the
docs/performance/scaling/levante_campaign_2026-07-24.md-317-campaign's most reproducible ASSOCIATION — but changing C-resolution, LL
docs/performance/scaling/levante_campaign_2026-07-24.md-318-size or ico subdivision also changes the global problem, so tile size is
docs/performance/scaling/levante_campaign_2026-07-24.md-319-not causally isolated.
docs/performance/scaling/levante_campaign_2026-07-24.md-320-
docs/performance/scaling/levante_campaign_2026-07-24.md-321-Shape: ~1.0 through np8, one step down, then FLAT 0.5 from np16 to np64 —
docs/performance/scaling/levante_campaign_2026-07-24.md-322-4x more ranks with no further loss. CONSISTENT with a fixed per-rank cost
docs/performance/scaling/levante_campaign_2026-07-24.md-323-rather than growing communication, but flat efficiency alone does not
docs/performance/scaling/levante_campaign_2026-07-24.md-324-identify which; that needs phase-level timing. (The >1 points at np8 are the same
docs/performance/scaling/levante_campaign_2026-07-24.md-325-base-leg-working-set effect noted for the cube; read as "at ideal".)
docs/performance/scaling/levante_campaign_2026-07-24.md-326-
--
docs/performance/scaling/levante_campaign_2026-07-24.md-345-| nd | measured | calibrated bound | measured/bound | at % of floor |
docs/performance/scaling/levante_campaign_2026-07-24.md-346-|---|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-347-| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-348-| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |
docs/performance/scaling/levante_campaign_2026-07-24.md-349-
docs/performance/scaling/levante_campaign_2026-07-24.md-350-Bound = per-device compute (32.58 / 14.63 ms) + modelled comm (2.21) +
docs/performance/scaling/levante_campaign_2026-07-24.md-351-modelled reduction (2.19). The unmodelled gap is **~5 ms/step and roughly
docs/performance/scaling/levante_campaign_2026-07-24.md-352-FLAT** with device count (7.9 ms at nd2, 6.7 at nd4), which is why the
docs/performance/scaling/levante_campaign_2026-07-24.md-353-ratio worsens as compute shrinks.
docs/performance/scaling/levante_campaign_2026-07-24.md-354-
docs/performance/scaling/levante_campaign_2026-07-24.md:355:WHAT THE GAP IS NOT — the omitted-traffic explanation is REFUTED
docs/performance/scaling/levante_campaign_2026-07-24.md:356:(`scripts/tmp/probe_ocean_halo_bytes.py`, HLO byte census on CPU virtual
docs/performance/scaling/levante_campaign_2026-07-24.md-357-devices). The bench's `comm_scope_note` correctly warns that its census is
docs/performance/scaling/levante_campaign_2026-07-24.md-358-"barotropic implicit-CN PCG scope only … baroclinic 3-D pads NOT counted",
docs/performance/scaling/levante_campaign_2026-07-24.md-359-and the true volume IS much larger: **16.22 MB/step across 110
docs/performance/scaling/levante_campaign_2026-07-24.md-360-collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md-361-completing the census moves the bound by only **0.22 ms**, because the
docs/performance/scaling/levante_campaign_2026-07-24.md-362-comm term is LATENCY-dominated: at 122 messages x 17.82 us the latency part
docs/performance/scaling/levante_campaign_2026-07-24.md-363-is 2.174 ms while even 16 MB at 64.22 GB/s is just 0.253 ms.
docs/performance/scaling/levante_campaign_2026-07-24.md-364-
docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
docs/performance/scaling/levante_campaign_2026-07-24.md-366-(nd2) and 4.3 ms (nd4).
docs/performance/scaling/levante_campaign_2026-07-24.md-367-
docs/performance/scaling/levante_campaign_2026-07-24.md:368:SECOND CANDIDATE ALSO REFUTED (`scripts/tmp/probe_sharded_overhead.py`,
docs/performance/scaling/levante_campaign_2026-07-24.md-369-job 26458553): the sharded formulation does NOT do measurably more work.
docs/performance/scaling/levante_campaign_2026-07-24.md-370-Timing the SHARDED step on a 1-device mesh (all the padding, band-edge and
docs/performance/scaling/levante_campaign_2026-07-24.md-371-v-row-reconstruction machinery present, ppermutes self-to-self so no real
docs/performance/scaling/levante_campaign_2026-07-24.md-372-traffic) against the UNSHARDED step at the identical tile:
docs/performance/scaling/levante_campaign_2026-07-24.md-373-
docs/performance/scaling/levante_campaign_2026-07-24.md-374-| tile | unsharded | sharded on 1 device | overhead |
docs/performance/scaling/levante_campaign_2026-07-24.md-375-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-376-| 288x1152x20 | 33.13 ms | 32.79 ms | **-0.34 ms (-1.0 %)** |
docs/performance/scaling/levante_campaign_2026-07-24.md-377-| 144x1152x20 | 15.88 ms | 15.92 ms | **+0.04 ms (+0.2 %)** |
docs/performance/scaling/levante_campaign_2026-07-24.md-378-
docs/performance/scaling/levante_campaign_2026-07-24.md-379-Zero within noise at both tiles, so the bound's compute term is the RIGHT
docs/performance/scaling/levante_campaign_2026-07-24.md-380-reference and extra sharded work is not the gap.
docs/performance/scaling/levante_campaign_2026-07-24.md-381-
docs/performance/scaling/levante_campaign_2026-07-24.md-382-WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
docs/performance/scaling/levante_campaign_2026-07-24.md-383-divided by the message count is **83 us/message at nd2 and 73 us at nd4**,
docs/performance/scaling/levante_campaign_2026-07-24.md-384-versus **17.8 us** for the same collective measured in isolation — an in-
docs/performance/scaling/levante_campaign_2026-07-24.md-385-context cost 4-5x the best case. That is consistent with EXPOSED,
docs/performance/scaling/levante_campaign_2026-07-24.md-386-un-overlapped communication rather than raw wire time.
docs/performance/scaling/levante_campaign_2026-07-24.md-387-
docs/performance/scaling/levante_campaign_2026-07-24.md:388:THIRD CANDIDATE REFUTED, AND IT IDENTIFIES THE MECHANISM (job 26458930).
docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
docs/performance/scaling/levante_campaign_2026-07-24.md-390-behind, DISABLING XLA's latency-hiding scheduler would hurt. It does not:
docs/performance/scaling/levante_campaign_2026-07-24.md-391-
docs/performance/scaling/levante_campaign_2026-07-24.md-392-| arm | nd2 | nd4 | vs default |
docs/performance/scaling/levante_campaign_2026-07-24.md-393-|---|---|---|---|
docs/performance/scaling/levante_campaign_2026-07-24.md-394-| default (LHS on) | 42.69 | 23.53 ms | — |
docs/performance/scaling/levante_campaign_2026-07-24.md-395-| `latency_hiding_scheduler=false` | 42.44 | 23.49 | **+0.6 % / +0.2 %** |
docs/performance/scaling/levante_campaign_2026-07-24.md-396-| `enable_pipelined_p2p=true` | 42.76 | 23.54 | -0.2 % / -0.1 % |
diff --git a/docs/performance/scaling/levante_campaign_2026-07-24.md b/docs/performance/scaling/levante_campaign_2026-07-24.md
index 24988a632..1cb204d87 100644
--- a/docs/performance/scaling/levante_campaign_2026-07-24.md
+++ b/docs/performance/scaling/levante_campaign_2026-07-24.md
@@ -1583,3 +1583,204 @@ DONE during the campaign (were open at the start): C768 same-path ladder
 production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
 the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
 which were found broken and fixed with a contract test.
+
+## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)
+
+Both jobs the dropped session left behind COMPLETED; neither had been
+analysed. First read-out below, CORRECTED per codex round-20
+(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
+VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
+in the first draft's weak-scaling claim).
+
+### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
+
+Matrix at a NOMINAL MEAN target of 5,120 cells/rank (the JSONL's
+`cells_per_rank_achieved` is global floor division —
+`int(mesh.nCells) // n_ranks`, bench_ocean_mpas_scaling.py:751 — NOT a
+balance statement; method pinned per arm, never `auto`, which flipped
+meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
+nlev 20, 32 ranks/node. Actual per-rank OWNED ranges
+(`metadata.partition_metrics.cells_per_rank_min/max`): geometric
+5,120–5,121 at BOTH scales; metis 5,100–5,145 @32 and 5,093–5,144 @128
+(±0.5 %). WET load is looser still under metis — per-rank wet
+cell-levels min/max: geometric@32 100,740–102,420; metis@32
+96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
+mean) — METIS balances owned cells (approximately), not wet cells, on
+this bathymetry.
+
+| arm | config | ms/step |
+|---|---|---|
+| A | s7 np32 geometric, block:cyclic | 189.82 |
+| B | s8 np128 geometric, block:cyclic | 308.96 |
+| C | s7 np32 metis, block:cyclic | 191.99 |
+| D | s8 np128 metis, block:cyclic | 333.39 |
+| E | s8 np128 metis, block:block | 537.91 |
+
+* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
+  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
+  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
+  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
+  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
+  -> step-time inference FAILS on this lane; part of metis's loss is
+  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
+  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
+  out partition/mapping improvements generally (e.g. wet-cell-weighted
+  METIS was NOT tested).
+* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
+  a byte-identical partition). NOTE the second `--distribution` field is
+  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
+  identically; the swing is socket-level. Mechanism (per-socket
+  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
+  2.13x receipt; never instrumented with bandwidth counters.
+* Caveats: timing-only receipt — no parity/conservation gate ran in
+  these arms, and the CPU nodes emit `UCX WARN transports
+  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
+  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
+  timing, but a "production config" claim would need a gated arm).
+
+### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
+
+f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
+scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
+L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
+np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
+reads `unknown` — same allocation, same submitted script, so the same
+binary is PLAUSIBLE but that row stays non-reproduction-grade on its
+own (codex r20/r21):
+
+| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
+|---|---|---|---|
+| 32 | 81.9k | 12.47 | 5.47 |
+| 64 | 41.0k | 9.60 | 7.10 |
+| 128 | 20.5k | 11.48 | 5.94 |
+
+* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
+  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
+  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
+* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
+  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
+  the other lanes (single unreplicated point on a lane with known
+  count-specific codegen variation — not by itself proof).
+* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
+  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
+  term".** Confounds: (a) every existing s8 receipt is the generator's
+  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
+  family, not the same protocol; (b) two comparator points came from the
+  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
+  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
+  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
+  no weak-scaling direction is claimed until it lands.
+
+### Next receipts submitted 2026-08-02
+
+1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
+   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
+   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
+   after — BRACKETED, not fully counterbalanced; a penalty's attribution
+   to fabric vs placement/drift needs the per-step nodelist table +
+   follow-up). steps=5000 so the stepping window
+   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
+   wall-clock brackets logged as overlap evidence. CONFIRM bar:
+   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
+   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
+   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
+   >10 % = a CO-EXECUTION penalty, quantified per replica — its
+   attribution (fabric contention vs placement/topology vs drift) is a
+   follow-up, not a conclusion of this job.
+2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
+   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
+   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
+   recomputed only from these.
+
+### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
+
+LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
+@64 row (job 26502539, f32 6.73 ms):
+
+| arm | ms/step | GC/s (col-levels) |
+|---|---|---|
+| f32 @128 (65,536 cols/GPU) | 5.5767 | **39.11** |
+| f64 @128 | 9.6015 | 22.72 |
+
+f32 strong 64->128: 1.207x for 2x devices (eff 0.60) with the tile at
+**65.5k cols/GPU — comfortably ABOVE the ~30k floor** (codex round-21
+caught the first draft halving this), so the loss is NOT
+floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
+band thinning to 16 rows/rank raising halo/compute ratio, and the
+16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
+highest measured throughput of ANY lane in the campaign. The companion
+oc128 (26534067) FAILED pre-#1370-fix with the 109.5 GB resident-args
+signature; retry submitted post-fix (below).
+
+## Hundreds-of-devices push (user directive 2026-08-02)
+
+"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
+GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
+partition effectively unbounded for our rank counts. Submitted set:
+
+| job | what | devices | why |
+|---|---|---|---|
+| 26628196 | s9 ensemble contention (v3; 26628021/26627810 superseded pre-start) | 128 GPU (4x32) | lever #1 receipt |
+| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
+| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
+| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
+| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
+| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
+
+s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
+
+### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
+
+r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
+wall-pole 2-D pencil lane (labelled; NOT the pole fold):
+
+| ranks | cols/rank | ms/step | speedup vs np64 | eff |
+|---|---|---|---|---|
+| 64 | 8,192 | 297.57 | 1.00 | 1.00 |
+| 128 | 4,096 | 161.03 | 1.848 | 0.92 |
+| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
+| 512 | 1,024 | 44.78 | 6.645 | **0.83** |
+
+Distribution verified against the masquerade trap: result rows carry
+`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
+count on this mpi4jax lane, not the world size). 128->256 is
+SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
+transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
+64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
+CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
+an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
+incorrect results", parallel/reductions.py runtime check) and UCX
+VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
+unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
+result JSON — only `decomposition: 2d`; a follow-up could add them to
+the bench metadata.)
+
+### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
+
+s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
+warmup 3 — protocol-identical to the s9 ladder (26600095), git
+7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
+untouched): **6.58 / 6.43 / 7.29 ms**.
+
+Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):
+
+| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
+|---|---|---|---|---|
+| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
+| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
+| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
+
+* The falsifiability block's CONFIRM branch fires: ratios stay well
+  above 1, so the round-20 retraction's CONFOUND did not manufacture
+  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
+  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
+  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
+* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
+  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
+  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
+  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
+  fraction growth, collective latency vs count, sfc partition-quality
+  decay with parts; the metis receipt argues against pure
+  partition-cut explanations, on the CPU lane at least).
+* The non-monotone tile dependence of the ratio (largest at the
+  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
diff --git a/scripts/plot/plot_scaling_paper_figure.py b/scripts/plot/plot_scaling_paper_figure.py
index 93056ac19..990305ee5 100644
--- a/scripts/plot/plot_scaling_paper_figure.py
+++ b/scripts/plot/plot_scaling_paper_figure.py
@@ -31,10 +31,14 @@ from matplotlib.lines import Line2D
 # --- Measured data -------------------------------------------------------
 # (devices, ms/step). Job ids are the provenance for each series.
 SOURCES = {
-    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64)",
+    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64), "
+                  "LL2048@64 26502539, LL2048@128 26534060",
     "atm_cube": "26452894/26453782",
-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734",
-    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic",
+    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
+                "s8 np32-128 26549646/26538474, s9 26600095, "
+                "s8-lloyd0 26628076",
+    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
+                   "lat-lon 2-D r512 26628073",
     "oc_latlon": "26460444-501/26460365/26493592",
     "oc_tripole": "26493837/26493648",
     "oc_mpas": "26494036 (f64), 26494908 (f32)",
@@ -45,9 +49,10 @@ PANELS = [
         key="atm_latlon", title="lat–lon", sub="720×1440 L26 · A100 NCCL",
         series=[("float32", [(4, 7.72), (8, 5.40), (16, 3.54)]),
                 ("float64", [(4, 16.11), (8, 11.34), (16, 5.62)]),
+                ("float32 (LL2048)", [(64, 6.73), (128, 5.58)]),
                 ],
-        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 @64", 64, 6.73)],
-        note="+ 64-GPU high-res points",
+        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 f64 @128", 128, 9.60)],
+        note="LL2048@128 = 39.1 GC/s",
     ),
     dict(
         key="atm_cube", title="cubed-sphere", sub="C384/C768 L60 · A100 NCCL",
@@ -56,21 +61,27 @@ PANELS = [
         note="f64 pending",
     ),
     dict(
-        key="atm_mpas", title="MPAS icosahedral", sub="L8 28 km L26 · A100 NCCL",
-        series=[("float32", [(2, 19.90), (4, 14.12), (8, 6.92), (16, 7.10)]),
-                ("float64", [(2, 38.34), (4, 20.09), (8, 18.98)])],
-        note="incl. fusion fix",
+        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
+        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
+                                        (16, 7.10), (32, 8.13), (64, 5.27),
+                                        (128, 6.47)]),
+                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
+                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
+                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
+        note="weak eff 0.53–0.67 at\nmatched tile (lloyd-0 pairs)",
     ),
     dict(
-        key="atm_ico_cpu", title="icosahedral", sub="subdiv-7 L26 · Milan CPU–MPI",
+        key="atm_ico_cpu", title="ico + lat-lon 2-D", sub="subdiv-7 / r512 L26 · Milan CPU–MPI",
         series=[("float32", [(1, 7399.72), (2, 3250.27), (4, 1673.07),
                              (8, 827.06), (16, 444.58), (32, 230.35),
                              (64, 131.58)]),
                 ("float64", [(1, 9987.81), (2, 4727.46), (4, 2349.06),
                              (8, 1161.98), (16, 624.06), (32, 350.89),
                              (64, 220.57), (128, 92.4), (256, 56.2),
-                             (512, 66.7)])],
-        note="to 512 ranks",
+                             (512, 66.7)]),
+                ("f64 lat-lon 2-D (r512)", [(64, 297.57), (128, 161.03),
+                                            (256, 72.06), (512, 44.78)])],
+        note="ico to 1024; lat-lon 2-D\neff 0.83 @512",
     ),
     dict(
         key="oc_latlon", title="lat–lon", sub="576×1152 L20 · A100 NCCL",
@@ -102,11 +113,19 @@ COLORS = {"float32": "#0072B2", "float64": "#D55E00",
           "float64 (packed)": "#E69F00",
           "mixed (f64 store)": "#009E73",
           "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
+          "float32 (LL2048)": "#009E73",
+          "f64 lat-lon 2-D (r512)": "#CC79A7",
+          "f32 (s8 lloyd-0)": "#009E73",
+          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
           "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
 MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
            "f32 · LL1536/2048 @64": "*",
            "float64 (packed)": "s",
            "float32 (C768)": "o", "float32 (C384)": "^",
+           "float32 (LL2048)": "^",
+           "f64 lat-lon 2-D (r512)": "D",
+           "f32 (s8 lloyd-0)": "v",
+           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
            "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
 
 

exec
/usr/bin/bash -lc 'rg -n "VERDICT|''^##|''^###|Arithmetic|arithmetic|residual|dirty|mesh padding|protocol" .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md | tail -80 && wc -l .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md && sed -n '"'700,820p' .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 347ms:
14:Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
16:I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
18:/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
113:./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
1484:./docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
1485:./docs/performance/scaling/levante_campaign_2026-07-24.md-1762-7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
1625:### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
1652:  family, not the same protocol; (b) two comparator points came from the
1658:### Next receipts submitted 2026-08-02
1679:### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
1681:LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
1693:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
1696:warmup 3 — protocol-identical to the s9 ladder (26600095), git
1697:7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
1736:# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
1765:  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
1794:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
1873:docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
1895:Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
1897:I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
1899:/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
1994:./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
2135:Round-4 final check. Your round-3 had 2 items: (1) doc ensemble wording contradicted v3 script (counterbalanced claim, stale job id 26628021, refute-bar attribution); (2) CPU r512 ladder needed unsupported-stack qualification. Both now patched in docs/performance/scaling/levante_campaign_2026-07-24.md: 'BRACKETED, not fully counterbalanced' + follow-up attribution language at the ensemble bullet AND its REFUTE line, job table row now 26628196 (v3, supersessions noted), CPU receipt labelled TIMING-ONLY/unvalidated with the mpi4jax 0.9.0 out-of-tested-range + UCX VM_UNMAP quotes. Read the '### Next receipts' + CPU-receipt sections and confirm consistency with scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch. Anything still inconsistent? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
2229:docs/performance/scaling/levante_campaign_2026-07-24.md-125-VERDICT: five hypotheses refuted by measurement (placement, halo volume,
2262:docs/performance/scaling/levante_campaign_2026-07-24.md-260-  labelling it 'protocol, not regression' is an inference from that
2350:docs/performance/scaling/levante_campaign_2026-07-24.md-365-So with the byte census completed the unexplained residual is still 5.5 ms
2367:docs/performance/scaling/levante_campaign_2026-07-24.md-382-WHERE THAT LEAVES IT (quantified, one candidate standing): the residual
2374:docs/performance/scaling/levante_campaign_2026-07-24.md-389-If the residual were communication the scheduler is currently hiding work
2396:+VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
2472:+  family, not the same protocol; (b) two comparator points came from the
2501:+LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
2565:+warmup 3 — protocol-identical to the s9 ladder (26600095), git
2566:+7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
2683:/usr/bin/bash -lc 'rg -n "VERDICT|''^##|''^###|Arithmetic|arithmetic|residual|dirty|mesh padding|protocol" .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md | tail -80 && wc -l .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md && sed -n '"'700,820p' .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
2683 .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md
./data/les_cases/ARM9707/rad-3544-    208.590       0.430149E-05
./data/les_cases/ARM9707/rad:3545:    181.964       0.741450E-05
./data/les_cases/ARM9707/rad-3546-    157.217       0.107873E-04
./data/les_cases/ARM9707/rad-3547-    134.209       0.129942E-04
./data/les_cases/ARM9707/rad-3548-    112.668       0.150130E-04
--
./data/les_cases/ARM9707/rad-3853-    607.706      -0.591957E-05
./data/les_cases/ARM9707/rad-3854-    564.887      -0.702651E-05
./data/les_cases/ARM9707/rad-3855-    522.926      -0.723073E-05
./data/les_cases/ARM9707/rad:3856:    481.992      -0.505495E-05
./data/les_cases/ARM9707/rad-3857-    442.220      -0.239989E-05
./data/les_cases/ARM9707/rad-3858-    403.728      -0.171788E-05
./data/les_cases/ARM9707/rad-3859-    366.625      -0.334984E-05
--
./data/les_cases/ARM9707/rad-5590-    267.596      -0.147840E-04
./data/les_cases/ARM9707/rad-5591-    237.155      -0.113387E-04
./data/les_cases/ARM9707/rad-5592-    208.608      -0.714148E-05
./data/les_cases/ARM9707/rad:5593:    181.982      -0.128787E-05
./data/les_cases/ARM9707/rad-5594-    157.235       0.290112E-05
./data/les_cases/ARM9707/rad-5595-    134.227       0.587965E-05
./data/les_cases/ARM9707/rad-5596-    112.686       0.692960E-05
--
./data/les_cases/ARM9707/rad-6710-    267.563       0.123273E-05
./data/les_cases/ARM9707/rad-6711-    237.122       0.478705E-05
./data/les_cases/ARM9707/rad-6712-    208.576       0.731151E-05
./data/les_cases/ARM9707/rad:6713:    181.949       0.122005E-04
./data/les_cases/ARM9707/rad-6714-    157.202       0.150230E-04
./data/les_cases/ARM9707/rad-6715-    134.194       0.135030E-04
./data/les_cases/ARM9707/rad-6716-    112.653       0.160325E-04
--
./data/les_cases/ARM9707/rad-6727-    865.032       0.137901E-04
./data/les_cases/ARM9707/rad-6728-    825.788       0.136008E-04
./data/les_cases/ARM9707/rad-6729-    784.144       0.133342E-04
./data/les_cases/ARM9707/rad:6730:    741.094       0.130099E-04
./data/les_cases/ARM9707/rad-6731-    697.394       0.114475E-04
./data/les_cases/ARM9707/rad-6732-    653.608       0.114078E-04
./data/les_cases/ARM9707/rad-6733-    610.152       0.988671E-05
--
./data/les_cases/ARM9707/rad-6950-    898.368      -0.119571E-04
./data/les_cases/ARM9707/rad-6951-    862.804      -0.119106E-04
./data/les_cases/ARM9707/rad-6952-    823.560      -0.129452E-04
./data/les_cases/ARM9707/rad:6953:    781.916      -0.164470E-04
./data/les_cases/ARM9707/rad-6954-    738.866      -0.219518E-04
./data/les_cases/ARM9707/rad-6955-    695.165      -0.204682E-04
./data/les_cases/ARM9707/rad-6956-    651.380      -0.234704E-04
--
./data/les_cases/ARM9707/rad-7014-    898.361      -0.191285E-04
./data/les_cases/ARM9707/rad-7015-    862.796      -0.187612E-04
./data/les_cases/ARM9707/rad-7016-    823.552      -0.190103E-04
./data/les_cases/ARM9707/rad:7017:    781.908      -0.219219E-04
./data/les_cases/ARM9707/rad-7018-    738.858      -0.277873E-04
./data/les_cases/ARM9707/rad-7019-    695.158      -0.254392E-04
./data/les_cases/ARM9707/rad-7020-    651.372      -0.294572E-04
--
./data/les_cases/ARM9707/rad-7046-    898.373      -0.311032E-04
./data/les_cases/ARM9707/rad-7047-    862.808      -0.272020E-04
./data/les_cases/ARM9707/rad-7048-    823.564      -0.270769E-04
./data/les_cases/ARM9707/rad:7049:    781.920      -0.258723E-04
./data/les_cases/ARM9707/rad-7050-    738.870      -0.277522E-04
./data/les_cases/ARM9707/rad-7051-    695.170      -0.272416E-04
./data/les_cases/ARM9707/rad-7052-    651.384      -0.284310E-04
--
./data/les_cases/ARM9707/rad-7078-    898.373      -0.321893E-04
./data/les_cases/ARM9707/rad-7079-    862.808      -0.284772E-04
./data/les_cases/ARM9707/rad-7080-    823.564      -0.283293E-04
./data/les_cases/ARM9707/rad:7081:    781.920      -0.271958E-04
./data/les_cases/ARM9707/rad-7082-    738.870      -0.289815E-04
./data/les_cases/ARM9707/rad-7083-    695.170      -0.285068E-04
./data/les_cases/ARM9707/rad-7084-    651.384      -0.296240E-04
--
./data/les_cases/ARM9707/rad-7110-    898.357      -0.333246E-04
./data/les_cases/ARM9707/rad-7111-    862.792      -0.297437E-04
./data/les_cases/ARM9707/rad-7112-    823.548      -0.296220E-04
./data/les_cases/ARM9707/rad:7113:    781.904      -0.285236E-04
./data/les_cases/ARM9707/rad-7114-    738.854      -0.302473E-04
./data/les_cases/ARM9707/rad-7115-    695.154      -0.297754E-04
./data/les_cases/ARM9707/rad-7116-    651.368      -0.308731E-04
--
./data/les_cases/ARM9707/rad-7245-    607.696      -0.252965E-04
./data/les_cases/ARM9707/rad-7246-    564.877      -0.237714E-04
./data/les_cases/ARM9707/rad-7247-    522.917      -0.220184E-04
./data/les_cases/ARM9707/rad:7248:    481.983      -0.208050E-04
./data/les_cases/ARM9707/rad-7249-    442.211      -0.200445E-04
./data/les_cases/ARM9707/rad-7250-    403.718      -0.174390E-04
./data/les_cases/ARM9707/rad-7251-    366.616      -0.167932E-04
--
./data/les_cases/ARM9707/rad-7277-    607.642      -0.225973E-04
./data/les_cases/ARM9707/rad-7278-    564.823      -0.209596E-04
./data/les_cases/ARM9707/rad-7279-    522.862      -0.190562E-04
./data/les_cases/ARM9707/rad:7280:    481.928      -0.177441E-04
./data/les_cases/ARM9707/rad-7281-    442.156      -0.169307E-04
./data/les_cases/ARM9707/rad-7282-    403.663      -0.141003E-04
./data/les_cases/ARM9707/rad-7283-    366.561      -0.134054E-04
--
./data/les_cases/ARM9707/rad-7309-    607.616      -0.208463E-04
./data/les_cases/ARM9707/rad-7310-    564.797      -0.191017E-04
./data/les_cases/ARM9707/rad-7311-    522.836      -0.171574E-04
./data/les_cases/ARM9707/rad:7312:    481.902      -0.158608E-04
./data/les_cases/ARM9707/rad-7313-    442.130      -0.150035E-04
./data/les_cases/ARM9707/rad-7314-    403.638      -0.122643E-04
./data/les_cases/ARM9707/rad-7315-    366.535      -0.115118E-04
--
./data/les_cases/ARM9707/rad-7341-    607.627      -0.140850E-04
./data/les_cases/ARM9707/rad-7342-    564.807      -0.140112E-04
./data/les_cases/ARM9707/rad-7343-    522.846      -0.144801E-04
./data/les_cases/ARM9707/rad:7344:    481.913      -0.131021E-04
./data/les_cases/ARM9707/rad-7345-    442.141      -0.117486E-04
./data/les_cases/ARM9707/rad-7346-    403.648      -0.101241E-04
./data/les_cases/ARM9707/rad-7347-    366.546      -0.842294E-05
--
./data/les_cases/ARM9707/rad-7373-    607.633      -0.819892E-05
./data/les_cases/ARM9707/rad-7374-    564.814      -0.856586E-05
./data/les_cases/ARM9707/rad-7375-    522.853      -0.939671E-05
./data/les_cases/ARM9707/rad:7376:    481.919      -0.872497E-05
./data/les_cases/ARM9707/rad-7377-    442.147      -0.779431E-05
./data/les_cases/ARM9707/rad-7378-    403.655      -0.657955E-05
./data/les_cases/ARM9707/rad-7379-    366.553      -0.520208E-05
--
./data/les_cases/ARM9707/rad-7845-    925.647      -0.420709E-04
./data/les_cases/ARM9707/rad-7846-    895.386      -0.360121E-04
./data/les_cases/ARM9707/rad-7847-    859.821      -0.328525E-04

exec
/usr/bin/bash -lc "rg --files .physics-validator | head -200 && rg -n -g '"'!*.md'"' -i \"26600095|26628076|7151d12a1|steady_median_ms|2,621,568|2621568\" .physics-validator scripts docs/performance/scaling" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 422ms:
.physics-validator/ocean/inventory.md
.physics-validator/ocean/diff_probes.py
.physics-validator/ocean/packet-1.md
.physics-validator/ocean/review-1.md
.physics-validator/ocean/static.md
.physics-validator/mpas_qv_smoothing/review-2.md
.physics-validator/mpas_qv_smoothing/packet-5.md
.physics-validator/mpas_qv_smoothing/review-3.md
.physics-validator/mpas_qv_smoothing/rebuttals.md
.physics-validator/mpas_qv_smoothing/review-4.md
.physics-validator/mpas_qv_smoothing/packet-4.md
.physics-validator/mpas_qv_smoothing/packet-1.md
.physics-validator/mpas_qv_smoothing/review-5.md
.physics-validator/mpas_qv_smoothing/review-1.md
.physics-validator/mpas_qv_smoothing/packet-3.md
.physics-validator/mpas_qv_smoothing/packet-2.md
.physics-validator/mpas_cmor_flux_feed/review-2.md
.physics-validator/mpas_cmor_flux_feed/rebuttals.md
.physics-validator/mpas_cmor_flux_feed/packet-1.md
.physics-validator/mpas_cmor_flux_feed/review-1.md
.physics-validator/mpas_cmor_flux_feed/packet-2.md
.physics-validator/ocean-revalidation/diff_probe_kpp_proxy.py
.physics-validator/hard_saturation_adjustment/packet-3.md
.physics-validator/ocean-revalidation/diff_probe_kpp_integration_bsalt.py
.physics-validator/ocean-revalidation/diff_probe_visbeck_grad.py
.physics-validator/ocean-revalidation/diff_probe_kpp_vt_grad.py
.physics-validator/ocean-revalidation/review-1.md
.physics-validator/ocean-revalidation/packet-2.md
.physics-validator/hard_saturation_adjustment/verdict-2.txt
.physics-validator/ocean-revalidation/REPORT.md
.physics-validator/hard_saturation_adjustment/packet-2.md
.physics-validator/hard_saturation_adjustment/verdict-7.txt
.physics-validator/hard_saturation_adjustment/verdict-4b.txt
.physics-validator/ocean-revalidation/diff_probe_kpp_nonlocal_conservation.py
.physics-validator/hard_saturation_adjustment/packet-7.md
.physics-validator/hard_saturation_adjustment/packet-6.md
.physics-validator/hard_saturation_adjustment/verdict-3.txt
.physics-validator/ocean-revalidation/diff_probe_kpp_av_floor.py
.physics-validator/ocean-revalidation/packet-1.md
.physics-validator/ocean-revalidation/diff_probe_kpp_bsalt.py
.physics-validator/hard_saturation_adjustment/packet-5.md
.physics-validator/hard_saturation_adjustment/verdict-5.txt
.physics-validator/ocean-revalidation/diff_probe_kpp_proxy_sharp.py
.physics-validator/hard_saturation_adjustment/verdict-6.txt
.physics-validator/hard_saturation_adjustment/packet-4.md
.physics-validator/ocean-revalidation/codex-adversarial-review-followup.md
.physics-validator/ocean-revalidation/review-2.md
.physics-validator/hard_saturation_adjustment/packet-1.md
.physics-validator/hard_saturation_adjustment/review-1.md
.physics-validator/fiction_field_deletion/review-1.md
.physics-validator/hard_sat_ice_curve/packet-1.md
.physics-validator/hard_sat_ice_curve/warm_rain.diff
.physics-validator/hard_sat_ice_curve/probe_ice_curve.py
.physics-validator/hard_sat_ice_curve/config.diff
.physics-validator/hard_sat_ice_curve/review-1.md
.physics-validator/hard_sat_ice_curve/packet-3.md
.physics-validator/hard_sat_ice_curve/packet-2.md
.physics-validator/hard_sat_ice_curve/rebuttals.md
.physics-validator/hard_sat_ice_curve/review-4.md
.physics-validator/hard_sat_ice_curve/packet-4.md
.physics-validator/hard_sat_ice_curve/model_driver.diff
.physics-validator/hard_sat_ice_curve/review-3.md
.physics-validator/hard_sat_ice_curve/review-2.md
.physics-validator/volcanic_lw_fix/review-2.md
.physics-validator/volcanic_lw_fix/review-3.md
.physics-validator/volcanic_lw_fix/fix.diff
.physics-validator/volcanic_lw_fix/review-4.md
.physics-validator/volcanic_lw_fix/packet-4.md
.physics-validator/volcanic_lw_fix/packet-1.md
.physics-validator/volcanic_lw_fix/review-1.md
.physics-validator/volcanic_lw_fix/packet-3.md
.physics-validator/volcanic_lw_fix/packet-2.md
.physics-validator/scaling_campaign/review10_batch.md
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md
.physics-validator/scaling_campaign/review8_final.md
.physics-validator/scaling_campaign/review12_gated_barrier.md
.physics-validator/scaling_campaign/review2.md
.physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md
.physics-validator/scaling_campaign/review6_final.md
.physics-validator/scaling_campaign/review16_cube_halo.md
.physics-validator/scaling_campaign/review14_scaleout.md
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
.physics-validator/scaling_campaign/review11_np4_strategy.md
.physics-validator/scaling_campaign/iter5_ocean.patch
.physics-validator/scaling_campaign/review19_meshcap.md
.physics-validator/scaling_campaign/review5.md
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
.physics-validator/scaling_campaign/evidence_for_strategy.md
.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md
.physics-validator/scaling_campaign/review18_memfix.md
.physics-validator/scaling_campaign/review13_precision.md
.physics-validator/scaling_campaign/iter1_diff.patch
.physics-validator/scaling_campaign/review7_mechanism.md
.physics-validator/scaling_campaign/iter3_diff.patch
.physics-validator/scaling_campaign/iter4_code.patch
.physics-validator/scaling_campaign/review9_gate_reframe.md
.physics-validator/scaling_campaign/review1.md
.physics-validator/scaling_campaign/review17_globalmem.md
.physics-validator/scaling_campaign/review3.md
.physics-validator/scaling_campaign/iter2_diff.patch
.physics-validator/scaling_campaign/review15_plateaus.md
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
.physics-validator/atmosphere-physics/microphysics/inventory.md
.physics-validator/mpas_land_boundary/review-2.md
.physics-validator/atmosphere-physics/microphysics/rebuttals.md
.physics-validator/mpas_land_boundary/packet-5.md
.physics-validator/atmosphere-physics/microphysics/packet-1.md
.physics-validator/mpas_land_boundary/review-3.md
.physics-validator/atmosphere-physics/microphysics/review-1.md
.physics-validator/mpas_land_boundary/review-4.md
.physics-validator/atmosphere-physics/microphysics/static.md
.physics-validator/mpas_land_boundary/packet-4.md
.physics-validator/mpas_land_boundary/packet-1.md
.physics-validator/mpas_land_boundary/review-5.md
.physics-validator/mpas_land_boundary/review-1.md
.physics-validator/mpas_land_boundary/packet-3.md
.physics-validator/mpas_land_boundary/packet-2.md
.physics-validator/tropopause_refined_sigma/ask-A.txt
.physics-validator/tropopause_refined_sigma/ask-C.txt
.physics-validator/tropopause_refined_sigma/ask-D.txt
.physics-validator/tropopause_refined_sigma/ask-F.txt
.physics-validator/tropopause_refined_sigma/packet-1-full.md
.physics-validator/tropopause_refined_sigma/ask-B.txt
.physics-validator/tropopause_refined_sigma/packet-1.md
.physics-validator/tropopause_refined_sigma/ask-E.txt
.physics-validator/tropopause_refined_sigma/review-1.md
.physics-validator/tropopause_refined_sigma/packet-1-tight.md
.physics-validator/tropopause_refined_sigma/packet-1-lean.md
.physics-validator/atmosphere-physics/convection/packet-2-1.md
.physics-validator/atmosphere-physics/convection/packet-1.md
.physics-validator/atmosphere-physics/convection/packet-2-3.md
.physics-validator/atmosphere-physics/convection/review-1.md
.physics-validator/atmosphere-physics/convection/review-2-3.md
.physics-validator/atmosphere-physics/convection/static.md
.physics-validator/atmosphere-physics/convection/review-2-2.md
.physics-validator/atmosphere-physics/convection/review-2-1.md
.physics-validator/atmosphere-physics/convection/packet-2-2.md
.physics-validator/atmosphere-physics/convection/inventory.md
.physics-validator/atmosphere-physics/convection/rebuttals.md
.physics-validator/atmosphere-physics/convection/static-2.md
.physics-validator/gwd_config_for/review-1.md
.physics-validator/gwd_config_for/packet-3.md
.physics-validator/gwd_config_for/packet-2.md
.physics-validator/gwd_config_for/review-3.md
.physics-validator/gwd_config_for/packet-1.md
.physics-validator/gwd_config_for/review-2.md
.physics-validator/bechtold_mpas_runaway/review-1.md
.physics-validator/bechtold_mpas_runaway/packet-3.md
.physics-validator/bechtold_mpas_runaway/packet-2.md
.physics-validator/bechtold_mpas_runaway/review-3.md
.physics-validator/bechtold_mpas_runaway/packet-1.md
.physics-validator/bechtold_mpas_runaway/review-2.md
.physics-validator/morrison_wiring/review-4.md
.physics-validator/morrison_wiring/review-5.md
.physics-validator/morrison_wiring/review-1.md
.physics-validator/morrison_wiring/review-3.md
.physics-validator/morrison_wiring/review-2.md
.physics-validator/atmosphere-physics/radiation/static.md
.physics-validator/atmosphere-physics/radiation/inventory.md
.physics-validator/mpas_ice_skin/packet-4.md
.physics-validator/mpas_ice_skin/packet-1.md
.physics-validator/mpas_ice_skin/review-5.md
.physics-validator/mpas_ice_skin/review-1.md
.physics-validator/mpas_ice_skin/packet-3.md
.physics-validator/mpas_ice_skin/packet-2.md
.physics-validator/mpas_ice_skin/review-3.md
.physics-validator/mpas_ice_skin/rebuttals.md
.physics-validator/mpas_ice_skin/review-4.md
.physics-validator/mpas_ice_skin/packet-5.md
.physics-validator/mpas_ice_skin/review-2.md
.physics-validator/mpas_cmip_feed/packet-5-full.md
.physics-validator/mpas_cmip_feed/packet-2-full.md
.physics-validator/mpas_cmip_feed/review-4.md
.physics-validator/mpas_cmip_feed/packet-4.md
.physics-validator/mpas_cmip_feed/packet-7-full.md
.physics-validator/mpas_cmip_feed/packet-1.md
.physics-validator/mpas_cmip_feed/packet-6-full.md
.physics-validator/mpas_cmip_feed/review-5.md
.physics-validator/mpas_cmip_feed/changes.diff
.physics-validator/mpas_cmip_feed/review-1.md
.physics-validator/mpas_cmip_feed/packet-3.md
.physics-validator/mpas_cmip_feed/packet-3-full.md
.physics-validator/mpas_cmip_feed/packet-2.md
.physics-validator/mpas_cmip_feed/my_changes_v5.md
.physics-validator/mpas_cmip_feed/my_changes.md
.physics-validator/mpas_cmip_feed/packet-1-full.md
.physics-validator/mpas_cmip_feed/my_changes_v2.md
.physics-validator/mpas_cmip_feed/my_changes_v4.md
.physics-validator/mpas_cmip_feed/review-7.md
.physics-validator/mpas_cmip_feed/packet-4-full.md
.physics-validator/mpas_cmip_feed/packet-5.md
.physics-validator/mpas_cmip_feed/review-3.md
.physics-validator/mpas_cmip_feed/review-6.md
.physics-validator/mpas_cmip_feed/my_changes_v3.md
.physics-validator/mpas_cmip_feed/review-2.md
.physics-validator/conservative_clamp/review-3.md
.physics-validator/conservative_clamp/review-1.md
.physics-validator/conservative_clamp/review-2.md
.physics-validator/conservative_clamp/diff.patch
.physics-validator/atmosphere-physics/REPORT.md
scripts/bench/bench_mpas_spmd_scaling.py:515:        steady_median_ms=round(med, 2),
scripts/bench/bench_atm_latlon_spmd_scaling.py:380:        steady_median_ms=round(med, 4),
scripts/plot/plot_scaling_paper_figure.py:38:                "s8 np32-128 26549646/26538474, s9 26600095, "
scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076",
scripts/bench/bench_ocean_mpas_scaling.py:27:the aggregator-facing ``steady_median_ms`` carries (the same deliberate
scripts/bench/bench_ocean_mpas_scaling.py:726:    # aggregator-facing ``steady_median_ms`` carries the FUSED per-step
scripts/bench/bench_ocean_mpas_scaling.py:742:        steady_median_ms=(round(fused_step_ms, 4)
scripts/bench/bench_cube_tiled_step_scaling.py:340:        steady_median_ms=round(med, 2),
scripts/bench/metadata.py:585:    ``steady_median_ms`` + ``cells`` — no top-level ``sypd`` — so
scripts/bench/bench_ocean_latlon_spmd_scaling.py:671:    # measurement contract).  ``steady_median_ms`` keeps its aggregator-
scripts/bench/bench_ocean_latlon_spmd_scaling.py:701:        steady_median_ms=(round(med, 4) if med is not None else None),
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:14:# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:18:# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:21:#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:60:    print(f'np$NP: {d[\"steady_median_ms\"]:8.2f} ms')
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:232:# scaling curves); compare arms via steady_median_ms / sypd per JSONL row.
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:28:#   numbers : 2 solo + 4 replica steady_median_ms
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:36:# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:101:    ms=d['steady_median_ms']
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:54:    ms=d['steady_median_ms']
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:297:# the per-record steady_median_ms / sypd printed in each JSONL row.

exec
/usr/bin/bash -lc "sed -n '80,145p' scripts/bench/bench_mpas_spmd_scaling.py && sed -n '290,335p' scripts/bench/bench_mpas_spmd_scaling.py && rg -n -C 4 \"reorder_target|padded|padding|pad.*cell|target.*devices|partition_method\" scripts/bench/bench_mpas_spmd_scaling.py src leg oes m 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 2 in 329ms:
# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
# NOT a bug margin; a real halo/partition regression shows up orders of
# magnitude above these.  Values extend the 1-step envelope of
# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
                "p_s": (1.0e-5, 1.0)},
    "float32": {"u": (1.0e-3, 1.0e-3), "T": (1.0e-4, 1.0e-3),
                "p_s": (1.0e-3, 50.0)},
}
MPAS_PARITY_MAX_STEPS = 8

# Conservation gate default: with fix_mass=True the step restores the global
# dry mass to the pre-step value each step, so the drift over a smoke window
# is the allreduce rounding floor, not scheme drift.
MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}


def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
                          moist=False, lloyd_iterations=50):
    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.

    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
    device count of THIS run's mesh/model (the two differ for the
    single-device reference leg of a ladder, via ``--reorder-for``).
    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
    wave) so the sharded step's packed tracer halo exchange + RK tracer
    advection sit on the timed/gated path.
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import create_voronoi_device_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = create_voronoi_mesh(subdivision_level=subdivision,
                               lloyd_iterations=lloyd_iterations)
    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
        # Padding only guarantees divisibility for reorder_target.
        raise SystemExit(
            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
            f"divisible by --n-devices {run_nd}; use a ladder where every "
            f"count divides --reorder-for ({reorder_target}).")
    sigma = create_sigma_coordinate(nlev)
    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
    # energy-conserving PV flux, SSP-RK3, global mass fixer.
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=run_nd,
    )
    if dev_config.n_devices > 1:
        from legoesm.parallel.mesh import replicate_pytree
        mesh_model = replicate_pytree(mesh, dev_config)
    else:
        mesh_model = mesh

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one shard per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
            f"device count ({avail} across {jax.process_count()} processes).")

    dt = args.dt
    if dt is None:
        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)

    reorder_for = args.reorder_for if args.reorder_for is not None else nd
    if reorder_for < nd:
        raise SystemExit(
            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
            f"the ghost padding only guarantees divisibility for the "
            f"partition target.")
    mesh, model, s0, dev_config = build_model_and_state(
        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)

    if args.multicontroller:
        # Every process computed the reorder independently — assert the
        # partitions agree before any collective uses the halo schedule.
        # The checksum covers the entity ORDER (coordinates) and the
        # connectivity the ppermute schedule + TRiSK stencils read; a
        # rank-divergent partition (e.g. one rank resolving
        # --partition-method auto to METIS, another to RCB) cannot slip
        # through on cell positions alone.
        from jax.experimental import multihost_utils
        crc = 0
        for arr, dtype in (
            (mesh.latCell, np.float64), (mesh.latEdge, np.float64),
            (mesh.cellsOnEdge, np.int64), (mesh.edgesOnCell, np.int64),
            (mesh.cellsOnCell, np.int64), (mesh.areaCell, np.float64),
        ):
            crc = zlib.crc32(np.ascontiguousarray(
                np.asarray(arr, dtype=dtype)).tobytes(), crc)
        crc = zlib.crc32(
scripts/bench/bench_mpas_spmd_scaling.py-4-
scripts/bench/bench_mpas_spmd_scaling.py-5-The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
scripts/bench/bench_mpas_spmd_scaling.py-6-flag-for-flag where the grids allow): the global mesh is REORDERED with
scripts/bench/bench_mpas_spmd_scaling.py-7-``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
scripts/bench/bench_mpas_spmd_scaling.py:8:padded to an even device split) so each device's contiguous ``P("device")``
scripts/bench/bench_mpas_spmd_scaling.py-9-shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
scripts/bench/bench_mpas_spmd_scaling.py-10-only the partition-boundary halo per stage via ``jax.lax.ppermute``.
scripts/bench/bench_mpas_spmd_scaling.py-11-
scripts/bench/bench_mpas_spmd_scaling.py-12-  strong: fixed subdivision level, vary n_devices -> speedup = t(1)/t(n).
--
scripts/bench/bench_mpas_spmd_scaling.py-95-# is the allreduce rounding floor, not scheme drift.
scripts/bench/bench_mpas_spmd_scaling.py-96-MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
scripts/bench/bench_mpas_spmd_scaling.py-97-
scripts/bench/bench_mpas_spmd_scaling.py-98-
scripts/bench/bench_mpas_spmd_scaling.py:99:def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py-100-                          moist=False, lloyd_iterations=50):
scripts/bench/bench_mpas_spmd_scaling.py:101:    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
scripts/bench/bench_mpas_spmd_scaling.py-102-
scripts/bench/bench_mpas_spmd_scaling.py:103:    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
scripts/bench/bench_mpas_spmd_scaling.py-105-    device count of THIS run's mesh/model (the two differ for the
scripts/bench/bench_mpas_spmd_scaling.py-106-    single-device reference leg of a ladder, via ``--reorder-for``).
scripts/bench/bench_mpas_spmd_scaling.py-107-    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
--
scripts/bench/bench_mpas_spmd_scaling.py-118-    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
scripts/bench/bench_mpas_spmd_scaling.py-119-
scripts/bench/bench_mpas_spmd_scaling.py-120-    mesh = create_voronoi_mesh(subdivision_level=subdivision,
scripts/bench/bench_mpas_spmd_scaling.py-121-                               lloyd_iterations=lloyd_iterations)
scripts/bench/bench_mpas_spmd_scaling.py:122:    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
scripts/bench/bench_mpas_spmd_scaling.py-123-    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
scripts/bench/bench_mpas_spmd_scaling.py:124:        # Padding only guarantees divisibility for reorder_target.
scripts/bench/bench_mpas_spmd_scaling.py-125-        raise SystemExit(
scripts/bench/bench_mpas_spmd_scaling.py:126:            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
scripts/bench/bench_mpas_spmd_scaling.py-127-            f"divisible by --n-devices {run_nd}; use a ladder where every "
scripts/bench/bench_mpas_spmd_scaling.py:128:            f"count divides --reorder-for ({reorder_target}).")
scripts/bench/bench_mpas_spmd_scaling.py-129-    sigma = create_sigma_coordinate(nlev)
scripts/bench/bench_mpas_spmd_scaling.py-130-    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
scripts/bench/bench_mpas_spmd_scaling.py-131-    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
scripts/bench/bench_mpas_spmd_scaling.py-132-    # energy-conserving PV flux, SSP-RK3, global mass fixer.
--
scripts/bench/bench_mpas_spmd_scaling.py-177-def main() -> int:
scripts/bench/bench_mpas_spmd_scaling.py-178-    p = argparse.ArgumentParser()
scripts/bench/bench_mpas_spmd_scaling.py-179-    p.add_argument("--subdivision", type=int, default=5,
scripts/bench/bench_mpas_spmd_scaling.py-180-                   help="icosahedral subdivision level L "
scripts/bench/bench_mpas_spmd_scaling.py:181:                        "(nCells = 10*4^L + 2 before ghost padding)")
scripts/bench/bench_mpas_spmd_scaling.py-182-    p.add_argument("--nlev", type=int, default=8)
scripts/bench/bench_mpas_spmd_scaling.py-183-    p.add_argument("--lloyd", type=int, default=50,
scripts/bench/bench_mpas_spmd_scaling.py-184-                   help="Lloyd relaxation iterations for the mesh. 50 = "
scripts/bench/bench_mpas_spmd_scaling.py-185-                        "production SCVT; 0 = labelled synthetic scaling "
--
scripts/bench/bench_mpas_spmd_scaling.py-308-    reorder_for = args.reorder_for if args.reorder_for is not None else nd
scripts/bench/bench_mpas_spmd_scaling.py-309-    if reorder_for < nd:
scripts/bench/bench_mpas_spmd_scaling.py-310-        raise SystemExit(
scripts/bench/bench_mpas_spmd_scaling.py-311-            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
scripts/bench/bench_mpas_spmd_scaling.py:312:            f"the ghost padding only guarantees divisibility for the "
scripts/bench/bench_mpas_spmd_scaling.py-313-            f"partition target.")
scripts/bench/bench_mpas_spmd_scaling.py-314-    mesh, model, s0, dev_config = build_model_and_state(
scripts/bench/bench_mpas_spmd_scaling.py:315:        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
scripts/bench/bench_mpas_spmd_scaling.py-316-        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
scripts/bench/bench_mpas_spmd_scaling.py-317-
scripts/bench/bench_mpas_spmd_scaling.py-318-    if args.multicontroller:
scripts/bench/bench_mpas_spmd_scaling.py-319-        # Every process computed the reorder independently — assert the
--
scripts/bench/bench_mpas_spmd_scaling.py-496-    rec = dict(
scripts/bench/bench_mpas_spmd_scaling.py-497-        component="mpas_atm",
scripts/bench/bench_mpas_spmd_scaling.py-498-        subdivision=args.subdivision, n_devices=nd,
scripts/bench/bench_mpas_spmd_scaling.py-499-        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py:500:        partition_method=args.partition_method, physics=args.physics,
scripts/bench/bench_mpas_spmd_scaling.py-501-        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
scripts/bench/bench_mpas_spmd_scaling.py-502-        # a row without this field could pass as a production-SCVT receipt.
scripts/bench/bench_mpas_spmd_scaling.py-503-        lloyd_iterations=args.lloyd,
scripts/bench/bench_mpas_spmd_scaling.py-504-        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
--
scripts/bench/bench_mpas_spmd_scaling.py-554-        cells_per_rank=int(mesh.nCells) * args.nlev
scripts/bench/bench_mpas_spmd_scaling.py-555-        // max(jax.process_count(), 1),
scripts/bench/bench_mpas_spmd_scaling.py-556-        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
scripts/bench/bench_mpas_spmd_scaling.py-557-        extra={
scripts/bench/bench_mpas_spmd_scaling.py:558:            "partition_method": args.partition_method,
scripts/bench/bench_mpas_spmd_scaling.py-559-            "physics": args.physics,
scripts/bench/bench_mpas_spmd_scaling.py-560-            "steps": args.steps,
scripts/bench/bench_mpas_spmd_scaling.py-561-            "multicontroller": bool(args.multicontroller),
scripts/bench/bench_mpas_spmd_scaling.py-562-            "cells_per_device": int(mesh.nCells) // nd * args.nlev,

exec
/usr/bin/bash -lc "rg -n -C 6 \"def reorder_voronoi_for_sharding|def pad|nCells.*target|target.*nCells|n_cells.*target|ghost\" legoesm/parallel/voronoi_partition.py legoesm/grids/voronoi.py && sed -n '480,575p' scripts/bench/bench_mpas_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 2 in 284ms:
rg: legoesm/parallel/voronoi_partition.py: No such file or directory (os error 2)
rg: legoesm/grids/voronoi.py: No such file or directory (os error 2)

exec
/usr/bin/bash -lc "rg --files | rg 'voronoi_partition|grids/voronoi' && rg -n -C 6 \"def reorder_voronoi_for_sharding|def pad|target.*nCells|nCells.*target|ghost\" src/legoesm || true && sed -n '480,575p' scripts/bench/bench_mpas_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 319ms:
scripts/bench/bench_voronoi_partition_methods.py
tests/bench/test_bench_voronoi_partition_methods.py
tests/unit/test_voronoi_partition_method.py
tests/parallel/test_voronoi_partition_metrics.py
packages/core/legoesm/grids/voronoi.py
packages/core/legoesm/parallel/voronoi_partition.py
                got = np.asarray(got)
                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
                ok &= field_ok
                if rank0:
                    mx = (float(np.max(np.abs(got - want)))
                          if want.size else 0.0)
                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
            if not ok:
                if rank0:
                    print("ERROR: SPMD parity gate MISMATCH vs the "
                          "single-device reference.", flush=True)
                return 5

    steady = per_step_ms[args.warmup:]
    med = float(np.median(steady))
    rec = dict(
        component="mpas_atm",
        subdivision=args.subdivision, n_devices=nd,
        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
        partition_method=args.partition_method, physics=args.physics,
        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
        # a row without this field could pass as a production-SCVT receipt.
        lloyd_iterations=args.lloyd,
        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
        # saying "auto" would not reveal whether ppermute or allgather
        # was actually measured (codex M3c-2 MINOR).
        halo_strategy_requested=args.halo_strategy,
        halo_strategy_effective=getattr(
            step, "_halo_strategy_effective", "serial"),
        steps=args.steps, dt=dt,
        platform=jax.default_backend(),
        n_processes=jax.process_count(),
        multicontroller=bool(args.multicontroller),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=round(med, 2),
        steady_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=int(mesh.nCells) * args.nlev,
        # ppermute round count/step (static compile property; #1113) — the
        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
        # belongs on every row like the cube benches.
        hlo_collective_permutes=hlo_cp,
        # full per-family census (permute + all-reduce + all-gather + ...) on
        # the SAME compile: exposes any reduction the ico step introduces.
        hlo_collectives=hlo_census,
    )
    # Flat aggregator-compatible identity + metric fields (see the latlon
    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
    # icosahedral convention so both lanes land on the same plot curves.
    rec.update(
        grid_type="icosahedral",
        resolution=args.subdivision,
        n_levels=args.nlev,
        mode="strong",  # this bench fixes the mesh and sweeps devices
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        physics_level=args.physics,
        backend=jax.default_backend(),
        **tidy_throughput_fields(
            dt_seconds=dt, time_per_step_ms=med,
            total_cells=int(mesh.nCells) * args.nlev),
    )
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="icosahedral",
        component="atmosphere",
        resolution=f"L{args.subdivision}",
        n_levels=args.nlev,
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
                else 0),
        decomposition="cell_partition" if nd > 1 else "none",
        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
        # share lives in extra.cells_per_device — a single-process 4-device
        # SPMD run has 1 rank owning ALL cells (codex finding 3).
        cells_per_rank=int(mesh.nCells) * args.nlev
        // max(jax.process_count(), 1),
        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
        extra={
            "partition_method": args.partition_method,
            "physics": args.physics,
            "steps": args.steps,
            "multicontroller": bool(args.multicontroller),
            "cells_per_device": int(mesh.nCells) // nd * args.nlev,
        },
    ))
    # Multi-controller: every process times the same program; process 0 owns
    # the JSONL + stdout (others would duplicate/corrupt the append).
    if jax.process_index() == 0:
        _outdir = os.path.dirname(args.out)
        if _outdir:  # a bare basename --out needs no mkdir
            os.makedirs(_outdir, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        print(f"[mpas nd={nd} L{args.subdivision} nCells={mesh.nCells} "
              f"nlev={args.nlev}] compile={rec['compile_ms']}ms "

exec
/usr/bin/bash -lc "rg -n -C 8 \"def reorder_voronoi_for_sharding|def _.*pad|target.*nCells|nCells.*target|pad.*mesh|padded\" packages/core/legoesm/parallel/voronoi_partition.py && sed -n '1,360p' packages/core/legoesm/parallel/voronoi_partition.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 289ms:
898-    }[entity]
899-    return global_field[idx]
900-
901-
902-# ============================================================================
903-# Mesh padding for even sharding
904-# ============================================================================
905-
906:def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
907-    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.
908-
909-    Adds ghost cells/edges that are inert in physics:
910-    - Ghost cells: ``areaCell=1`` (avoids 0/0 NaN in divergence), all
911-      connectivity = -1 (masked by operators), signs/weights = 0.
912-    - Ghost edges: ``dvEdge=0`` (zero flux contribution), ``dcEdge=1``
913-      (avoids 0/0 in gradient), ``cellsOnEdge=[0,0]`` (valid references
914-      for unmasked operators like ``gradient_edge`` and ``cell_to_edge_avg``).
915-
916-    Returns *mesh* unchanged when no padding is required.
917-    """
918:    pad_cells = (-mesh.nCells) % n_devices
919:    pad_edges = (-mesh.nEdges) % n_devices
920-
921-    if pad_cells == 0 and pad_edges == 0:
922-        return mesh
923-
924-    # --- helpers ---
925-    def pad_1d(arr, n_pad, fill=0.0):
926-        if n_pad == 0:
927-            return arr
--
940-        # --- dimensions ---
941-        nCells=mesh.nCells + pad_cells,
942-        nEdges=mesh.nEdges + pad_edges,
943-        nVertices=mesh.nVertices,
944-        maxEdges=mesh.maxEdges,
945-        vertexDegree=mesh.vertexDegree,
946-        radius=mesh.radius,
947-        # --- cell coordinates (ghost at origin) ---
948:        latCell=pad_1d(mesh.latCell, pad_cells, 0.0),
949:        lonCell=pad_1d(mesh.lonCell, pad_cells, 0.0),
950:        xCell=pad_1d(mesh.xCell, pad_cells, 0.0),
951:        yCell=pad_1d(mesh.yCell, pad_cells, 0.0),
952:        zCell=pad_1d(mesh.zCell, pad_cells, 0.0),
953-        # --- edge coordinates (ghost at origin) ---
954:        latEdge=pad_1d(mesh.latEdge, pad_edges, 0.0),
955:        lonEdge=pad_1d(mesh.lonEdge, pad_edges, 0.0),
956:        xEdge=pad_1d(mesh.xEdge, pad_edges, 0.0),
957:        yEdge=pad_1d(mesh.yEdge, pad_edges, 0.0),
958:        zEdge=pad_1d(mesh.zEdge, pad_edges, 0.0),
959-        # --- vertex coordinates (unchanged) ---
960-        latVertex=mesh.latVertex,
961-        lonVertex=mesh.lonVertex,
962-        xVertex=mesh.xVertex,
963-        yVertex=mesh.yVertex,
964-        zVertex=mesh.zVertex,
965-        # --- connectivity ---
966-        # cellsOnEdge: operators (gradient_edge, cell_to_edge_avg) index
967-        # directly without masking, so ghost edges need valid cell refs.
968:        cellsOnEdge=pad_2d_col(mesh.cellsOnEdge, pad_edges, fill=0),
969:        edgesOnCell=pad_2d_col(mesh.edgesOnCell, pad_cells, fill=-1),
970:        verticesOnCell=pad_2d_col(mesh.verticesOnCell, pad_cells, fill=-1),
971:        verticesOnEdge=pad_2d_col(mesh.verticesOnEdge, pad_edges, fill=0),
972-        edgesOnVertex=mesh.edgesOnVertex,  # vertex-indexed, unchanged
973-        cellsOnVertex=mesh.cellsOnVertex,  # vertex-indexed, unchanged
974:        cellsOnCell=pad_2d_col(mesh.cellsOnCell, pad_cells, fill=-1),
975:        edgesOnEdge=pad_2d_col(mesh.edgesOnEdge, pad_edges, fill=-1),
976:        nEdgesOnCell=pad_1d(mesh.nEdgesOnCell, pad_cells, fill=0),
977:        nEdgesOnEdge=pad_1d(mesh.nEdgesOnEdge, pad_edges, fill=0),
978-        # --- geometry ---
979-        # areaCell=1 for ghosts avoids 0/0 NaN in divergence (numerator is
980-        # exactly zero because all connectivity = -1 and signs = 0).
981:        areaCell=pad_1d(mesh.areaCell, pad_cells, fill=1.0),
982-        areaTriangle=mesh.areaTriangle,  # vertex-indexed, unchanged
983:        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
984:        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
985:        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
986-        # --- weights and signs ---
987:        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
988-        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
989:        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
990-        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
991:        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
992-        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
993:        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
994-    )
995-
996-
997-# ============================================================================
998-# Mesh reordering for JAX SPMD sharding
999-# ============================================================================
1000-
1001:def reorder_voronoi_for_sharding(
1002-    mesh: VoronoiMesh,
1003-    n_devices: int,
1004-    *,
1005-    method: str = "auto",
1006-) -> VoronoiMesh:
1007-    """Reorder a Voronoi mesh so that JAX NamedSharding gives spatial locality.
1008-
1009-    Partitions cells via RCB (or METIS), then reorders cells, edges, and
"""Domain decomposition for Voronoi (MPAS-style) meshes.

Partitions an unstructured Voronoi mesh across MPI ranks or JAX devices
and constructs local sub-meshes with halo (ghost) entities for parallel
stencil computation.

Two partitioning methods:

1. **Geometric (RCB)**: Recursive Coordinate Bisection on cell-center
   Cartesian coordinates.  No external dependencies.
2. **METIS** (optional): k-way graph partitioning via ``pymetis``.

After partitioning, each rank holds owned + halo entities.  The halo
exchange (:mod:`legoesm.parallel.halo_exchange_voronoi`) updates halo
values from their owning ranks between timesteps.

Usage
-----
::

    partition = partition_voronoi_mesh(mesh, n_ranks=4, rank=0)
    local_mesh = build_local_mesh(mesh, partition)

    # In the time loop, exchange halo data:
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    halo = VoronoiHaloExchange(partition, backend="mpi")
    h_local = halo.exchange_cell_field(h_local)
"""

from __future__ import annotations

from typing import NamedTuple

import importlib.util
import logging

import numpy as np
import jax.numpy as jnp

from legoesm.grids.voronoi import VoronoiMesh

logger = logging.getLogger("legoesm.parallel.voronoi_partition")

# One-time log guard so a per-rank/per-call "auto" resolution does not spam.
_AUTO_METHOD_LOGGED = False

# Hilbert space-filling-curve resolution: a 2^order x 2^order (lat, lon) grid.
# order=10 -> 1024^2 ~ 1.05e6 buckets, finer than any production Voronoi mesh
# (level-9 SCVT ~2.6e6 cells is the practical ceiling; ties break by stable
# sort), so distinct cells almost never collide. Module constant, not config:
# it is a numerics resolution knob, not a tunable.
_DEFAULT_HILBERT_ORDER = 10


def _metis_available() -> bool:
    """True if the optional ``pymetis`` graph-partitioning package is importable."""
    return importlib.util.find_spec("pymetis") is not None


def resolve_partition_method(method: str) -> str:
    """Resolve a partition method, expanding ``"auto"`` by available capability.

    ``"auto"`` (the default) selects ``"metis"`` when ``pymetis`` is importable —
    graph partitioning minimizes the edge cut, giving better load balance and
    smaller halos on irregular/variable-resolution meshes (the MPAS lesson:
    geometric RCB leaves lopsided cell counts and fat halos at scale) — and
    otherwise falls back to ``"geometric"`` (RCB, no dependency).

    ``"geometric"``, ``"metis"``, and any unknown value pass through UNCHANGED so
    the caller's own dispatch guard still raises on an unknown method. Returns the
    concrete method name.
    """
    global _AUTO_METHOD_LOGGED
    if method != "auto":
        return method
    chosen = "metis" if _metis_available() else "geometric"
    if not _AUTO_METHOD_LOGGED:
        _AUTO_METHOD_LOGGED = True
        if chosen == "metis":
            logger.info(
                "Voronoi partition method='auto' -> 'metis' (pymetis available; "
                "graph partitioning for load balance + smaller halos)."
            )
        else:
            logger.info(
                "Voronoi partition method='auto' -> 'geometric' RCB (pymetis not "
                "installed; `pip install pymetis` for better load balance at scale)."
            )
    return chosen


# ============================================================================
# Data structures
# ============================================================================

class HaloCommSchedule(NamedTuple):
    """Communication schedule for halo exchange of one entity type.

    For neighbor rank ``neighbor_ranks[i]``:

    - Send ``send_counts[i]`` values starting at cumulative offset in
      ``send_idx``.
    - Recv ``recv_counts[i]`` values starting at cumulative offset in
      ``recv_idx``.
    """
    neighbor_ranks: tuple[int, ...]
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    send_idx: jnp.ndarray   # (total_send,) local indices to pack
    recv_idx: jnp.ndarray   # (total_recv,) local indices to fill


class BatchedHaloSchedule(NamedTuple):
    """Union-neighbor comm schedule joining the cell + edge index spaces.

    Built once at layout-build time by :func:`build_batched_halo_schedule`
    from a partition's ``cell_comm`` and ``edge_comm``.  Lets the MPAS
    state exchange send ONE message per neighbor per dtype group (u edges
    + T/p_s cells + tracer cells packed into a single flat buffer) instead
    of one message per neighbor per entity exchange.

    ``neighbor_ranks`` is the sorted union of the cell and edge neighbor
    lists.  A rank present in only one of the two entity schedules gets
    zero counts for the other entity (zero-length pack segments).  The
    union relation is symmetric across ranks whenever the underlying
    entity schedules are (rank A lists B iff B lists A) — see
    ``tests/distributed/test_voronoi_batched_halo.py`` for the mechanical
    cross-rank check.

    For union neighbor ``i``:

    - cell send rows: ``cell_send_idx[sum(cell_send_counts[:i]) : ... +
      cell_send_counts[i]]`` (local OWNED cell indices to pack);
    - cell recv rows: same slicing of ``cell_recv_idx`` (local HALO cell
      indices to fill);
    - edge send/recv rows: identical layout in ``edge_send_idx`` /
      ``edge_recv_idx``.

    All counts are Python ints (layout constants) so every pack/unpack
    slice has a static shape under JIT.  Concatenating the per-neighbor
    recv rows in union order yields exactly ``cell_recv_idx`` /
    ``edge_recv_idx``, so the unpack can do a single functional scatter
    per field.
    """
    neighbor_ranks: tuple[int, ...]
    cell_send_counts: tuple[int, ...]
    cell_recv_counts: tuple[int, ...]
    edge_send_counts: tuple[int, ...]
    edge_recv_counts: tuple[int, ...]
    cell_send_idx: jnp.ndarray   # (total_cell_send,) local indices to pack
    cell_recv_idx: jnp.ndarray   # (total_cell_recv,) local indices to fill
    edge_send_idx: jnp.ndarray   # (total_edge_send,)
    edge_recv_idx: jnp.ndarray   # (total_edge_recv,)

    def messages_per_exchange(self, n_dtype_groups: int = 1) -> int:
        """Messages one batched state exchange posts per rank.

        Pure schedule math for the homogeneous case where every dtype
        group touches both index spaces (the expected production case:
        all prognostic fields share one dtype, so ``n_dtype_groups=1``).
        For heterogeneous groups (e.g. a cell-only dtype group facing an
        edge-only neighbor) the exact count is
        :func:`legoesm.parallel.halo_exchange_voronoi.count_batched_messages`,
        which never exceeds this bound.
        """
        return len(self.neighbor_ranks) * n_dtype_groups


class VoronoiPartition(NamedTuple):
    """Domain decomposition descriptor for one rank of a Voronoi mesh.

    Entities are ordered: owned first (sorted by global index), then
    halo (sorted by global index).
    """
    rank: int
    n_ranks: int

    # Global counts
    nCells_global: int
    nEdges_global: int
    nVertices_global: int

    # Owned counts
    n_owned_cells: int
    n_owned_edges: int
    n_owned_vertices: int

    # Local counts (owned + halo)
    n_local_cells: int
    n_local_edges: int
    n_local_vertices: int

    # Global indices of local entities (owned first, then halo)
    local_cells: np.ndarray       # (n_local_cells,)
    local_edges: np.ndarray       # (n_local_edges,)
    local_vertices: np.ndarray    # (n_local_vertices,)

    # Global-to-local mapping (-1 for non-local entities)
    cell_g2l: np.ndarray          # (nCells_global,)
    edge_g2l: np.ndarray          # (nEdges_global,)
    vertex_g2l: np.ndarray        # (nVertices_global,)

    # Communication schedules
    cell_comm: HaloCommSchedule
    edge_comm: HaloCommSchedule
    vertex_comm: HaloCommSchedule


# ============================================================================
# Partitioners
# ============================================================================

def partition_cells_geometric(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
    """Partition cells via Recursive Coordinate Bisection (RCB).

    Uses cell-center Cartesian coordinates on the unit sphere.

    Parameters
    ----------
    mesh : VoronoiMesh
    n_ranks : int

    Returns
    -------
    cell_owner : np.ndarray, shape (nCells,), dtype int32
        ``cell_owner[c]`` is the rank that owns cell ``c``.
    """
    coords = np.stack([
        np.asarray(mesh.xCell) / mesh.radius,
        np.asarray(mesh.yCell) / mesh.radius,
        np.asarray(mesh.zCell) / mesh.radius,
    ], axis=1)
    return _rcb(coords, n_ranks)


def _rcb(coords: np.ndarray, n_ranks: int) -> np.ndarray:
    """Recursive Coordinate Bisection on a point cloud."""
    n = len(coords)
    if n_ranks <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)

    axis = int(np.argmax(np.ptp(coords, axis=0)))
    order = np.argsort(coords[:, axis])

    n_left_ranks = n_ranks // 2
    n_right_ranks = n_ranks - n_left_ranks
    split = max(1, min(n - 1, n * n_left_ranks // n_ranks))

    left, right = order[:split], order[split:]
    result = np.empty(n, dtype=np.int32)
    result[left] = _rcb(coords[left], n_left_ranks)
    result[right] = _rcb(coords[right], n_right_ranks) + n_left_ranks
    return result


def partition_cells_metis(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
    """Partition cells via METIS k-way graph partitioning.

    Requires the ``pymetis`` package.

    Parameters
    ----------
    mesh : VoronoiMesh
    n_ranks : int

    Returns
    -------
    cell_owner : np.ndarray, shape (nCells,), dtype int32
    """
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError(
            "METIS partitioning requires pymetis.  "
            "Install with: pip install pymetis"
        ) from exc

    coc = np.asarray(mesh.cellsOnCell)
    nec = np.asarray(mesh.nEdgesOnCell)
    adjacency = []
    for c in range(mesh.nCells):
        nbrs = [int(coc[k, c]) for k in range(int(nec[c])) if coc[k, c] >= 0]
        adjacency.append(np.array(nbrs, dtype=np.int32))

    _, membership = pymetis.part_graph(n_ranks, adjacency=adjacency)
    return np.array(membership, dtype=np.int32)


def _hilbert_xy2d(order: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Hilbert-curve distance ``d`` for integer grid coords ``(x, y)``.

    Vectorized form of the canonical Wikipedia ``xy2d`` integer algorithm on a
    ``2^order x 2^order`` grid (rotation uses the full side length ``n``, not the
    current level ``s``).  Returns a bijection ``[0, n)^2 -> [0, n^2)`` whose
    1-D ordering preserves 2-D locality: cells adjacent on the curve are spatially
    close, which keeps each contiguous partition compact (small halo surface).
    """
    n = 1 << order
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    d = np.zeros(x.shape, dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rot(n, x, y, rx, ry): reflect when ry==0 (and x,y when rx==1), then swap.
        ry0 = ry == 0
        flip = ry0 & (rx == 1)
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        tx = np.where(ry0, y, x)
        ty = np.where(ry0, x, y)
        x, y = tx, ty
        s >>= 1
    return d


def hilbert_cell_keys(mesh: VoronoiMesh, order: int = _DEFAULT_HILBERT_ORDER) -> np.ndarray:
    """Per-cell Hilbert space-filling-curve key from cell (lat, lon).

    Maps each cell center to a ``2^order x 2^order`` (lon, lat) grid and returns
    its Hilbert distance.  Sorting cells by this key yields a 1-D ordering with
    strong 2-D spatial locality — used to build compact, contiguous partitions
    and locality-friendly local indexings.

    Parameters
    ----------
    mesh : VoronoiMesh
    order : int
        SFC grid resolution (side = ``2^order``).

    Returns
    -------
    np.ndarray, shape (nCells,), dtype int64
    """
    two_pi = 2.0 * np.pi
    lon = np.mod(np.asarray(mesh.lonCell, dtype=np.float64), two_pi)
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    n = 1 << order
    u = lon / two_pi                       # [0, 1)
    v = (lat + 0.5 * np.pi) / np.pi        # [0, 1]
    gx = np.clip((u * n).astype(np.int64), 0, n - 1)
    gy = np.clip((v * n).astype(np.int64), 0, n - 1)
    return _hilbert_xy2d(order, gx, gy)


def partition_cells_sfc(
    mesh: VoronoiMesh, n_ranks: int, order: int = _DEFAULT_HILBERT_ORDER,
) -> np.ndarray:
    """Partition cells into contiguous Hilbert space-filling-curve chunks.

    Orders cells along a Hilbert curve, then assigns ``n_ranks`` balanced
    contiguous runs.  Dependency-free (unlike METIS) and gives compact,
    spatially-local partitions (smaller halos than RCB on irregular meshes).

    Returns
    -------
    cell_owner : np.ndarray, shape (nCells,), dtype int32
    """

exec
/usr/bin/bash -lc "sed -n '900,1165p' packages/core/legoesm/parallel/voronoi_partition.py && rg -n \"s8.*lloyd|lloyd.*s8|26628076|nCells.*655|655,?4|655488|655360|655.*padded\" docs scripts .physics-validator/scaling_campaign -g '"'!codex_recovery_review_r5_2026-08-02.md'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 446ms:


# ============================================================================
# Mesh padding for even sharding
# ============================================================================

def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.

    Adds ghost cells/edges that are inert in physics:
    - Ghost cells: ``areaCell=1`` (avoids 0/0 NaN in divergence), all
      connectivity = -1 (masked by operators), signs/weights = 0.
    - Ghost edges: ``dvEdge=0`` (zero flux contribution), ``dcEdge=1``
      (avoids 0/0 in gradient), ``cellsOnEdge=[0,0]`` (valid references
      for unmasked operators like ``gradient_edge`` and ``cell_to_edge_avg``).

    Returns *mesh* unchanged when no padding is required.
    """
    pad_cells = (-mesh.nCells) % n_devices
    pad_edges = (-mesh.nEdges) % n_devices

    if pad_cells == 0 and pad_edges == 0:
        return mesh

    # --- helpers ---
    def pad_1d(arr, n_pad, fill=0.0):
        if n_pad == 0:
            return arr
        return jnp.concatenate([arr, jnp.full((n_pad,), fill, dtype=arr.dtype)])

    def pad_2d_col(arr, n_pad, fill=0):
        """Pad along axis 1 (entity axis for (K, nEntities) layout)."""
        if n_pad == 0:
            return arr
        K = arr.shape[0]
        return jnp.concatenate(
            [arr, jnp.full((K, n_pad), fill, dtype=arr.dtype)], axis=1,
        )

    return VoronoiMesh(
        # --- dimensions ---
        nCells=mesh.nCells + pad_cells,
        nEdges=mesh.nEdges + pad_edges,
        nVertices=mesh.nVertices,
        maxEdges=mesh.maxEdges,
        vertexDegree=mesh.vertexDegree,
        radius=mesh.radius,
        # --- cell coordinates (ghost at origin) ---
        latCell=pad_1d(mesh.latCell, pad_cells, 0.0),
        lonCell=pad_1d(mesh.lonCell, pad_cells, 0.0),
        xCell=pad_1d(mesh.xCell, pad_cells, 0.0),
        yCell=pad_1d(mesh.yCell, pad_cells, 0.0),
        zCell=pad_1d(mesh.zCell, pad_cells, 0.0),
        # --- edge coordinates (ghost at origin) ---
        latEdge=pad_1d(mesh.latEdge, pad_edges, 0.0),
        lonEdge=pad_1d(mesh.lonEdge, pad_edges, 0.0),
        xEdge=pad_1d(mesh.xEdge, pad_edges, 0.0),
        yEdge=pad_1d(mesh.yEdge, pad_edges, 0.0),
        zEdge=pad_1d(mesh.zEdge, pad_edges, 0.0),
        # --- vertex coordinates (unchanged) ---
        latVertex=mesh.latVertex,
        lonVertex=mesh.lonVertex,
        xVertex=mesh.xVertex,
        yVertex=mesh.yVertex,
        zVertex=mesh.zVertex,
        # --- connectivity ---
        # cellsOnEdge: operators (gradient_edge, cell_to_edge_avg) index
        # directly without masking, so ghost edges need valid cell refs.
        cellsOnEdge=pad_2d_col(mesh.cellsOnEdge, pad_edges, fill=0),
        edgesOnCell=pad_2d_col(mesh.edgesOnCell, pad_cells, fill=-1),
        verticesOnCell=pad_2d_col(mesh.verticesOnCell, pad_cells, fill=-1),
        verticesOnEdge=pad_2d_col(mesh.verticesOnEdge, pad_edges, fill=0),
        edgesOnVertex=mesh.edgesOnVertex,  # vertex-indexed, unchanged
        cellsOnVertex=mesh.cellsOnVertex,  # vertex-indexed, unchanged
        cellsOnCell=pad_2d_col(mesh.cellsOnCell, pad_cells, fill=-1),
        edgesOnEdge=pad_2d_col(mesh.edgesOnEdge, pad_edges, fill=-1),
        nEdgesOnCell=pad_1d(mesh.nEdgesOnCell, pad_cells, fill=0),
        nEdgesOnEdge=pad_1d(mesh.nEdgesOnEdge, pad_edges, fill=0),
        # --- geometry ---
        # areaCell=1 for ghosts avoids 0/0 NaN in divergence (numerator is
        # exactly zero because all connectivity = -1 and signs = 0).
        areaCell=pad_1d(mesh.areaCell, pad_cells, fill=1.0),
        areaTriangle=mesh.areaTriangle,  # vertex-indexed, unchanged
        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
        # --- weights and signs ---
        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
    )


# ============================================================================
# Mesh reordering for JAX SPMD sharding
# ============================================================================

def reorder_voronoi_for_sharding(
    mesh: VoronoiMesh,
    n_devices: int,
    *,
    method: str = "auto",
) -> VoronoiMesh:
    """Reorder a Voronoi mesh so that JAX NamedSharding gives spatial locality.

    Partitions cells via RCB (or METIS), then reorders cells, edges, and
    vertices so that entities owned by device 0 come first, then device 1,
    etc.  When JAX splits the reordered arrays into ``n_devices`` contiguous
    chunks along axis 0, each chunk corresponds to a spatially contiguous
    domain — minimizing cross-device communication in TRiSK stencils.

    Parameters
    ----------
    mesh : VoronoiMesh
        Original global mesh.
    n_devices : int
        Number of devices (partitions).
    method : str
        ``"auto"`` (default: METIS if ``pymetis`` available, else RCB),
        ``"geometric"`` (RCB), ``"metis"``, or ``"sfc"`` (Hilbert
        space-filling-curve contiguous chunks).

    Returns
    -------
    VoronoiMesh
        Mesh with reordered entities and remapped connectivity.
    """
    # Validate at entry (CLAUDE.md: fail early) BEFORE the single-device shortcut,
    # so an unknown method raises even when no partitioning happens.
    method = resolve_partition_method(method)
    if method not in ("geometric", "metis", "sfc"):
        raise ValueError(f"Unknown partitioning method: {method!r}")
    if n_devices <= 1:
        return mesh

    # --- Partition cells ---
    if method == "geometric":
        cell_owner = partition_cells_geometric(mesh, n_devices)
    elif method == "metis":
        cell_owner = partition_cells_metis(mesh, n_devices)
    else:  # "sfc" (validated above)
        cell_owner = partition_cells_sfc(mesh, n_devices)

    # --- Cell permutation: group by owner (primary), then order WITHIN each
    # owner by the Hilbert space-filling curve (secondary) so each contiguous
    # NamedSharding shard is spatially compact -> better cache/GPU locality and
    # smaller cross-shard stencil reach.  Ownership is unchanged; this sets only
    # the intra-shard order (the prior stable sort left it as arbitrary mesh
    # order).
    hkeys = hilbert_cell_keys(mesh)
    cell_perm = np.lexsort((hkeys, cell_owner))
    cell_inv = np.empty_like(cell_perm)
    cell_inv[cell_perm] = np.arange(len(cell_perm))

    # --- Edge owner: owner of the cell with the smaller global index ---
    cellsOnEdge_np = np.asarray(mesh.cellsOnEdge)  # (2, nEdges)
    c0 = cellsOnEdge_np[0]
    c1 = cellsOnEdge_np[1]
    edge_owner = cell_owner[np.minimum(c0, c1)]
    edge_perm = np.argsort(edge_owner, kind="stable")
    edge_inv = np.empty_like(edge_perm)
    edge_inv[edge_perm] = np.arange(len(edge_perm))

    # --- Vertex owner: owner of the cell with the smallest global index ---
    cellsOnVertex_np = np.asarray(mesh.cellsOnVertex)  # (vertexDegree, nVertices)
    cov_safe = np.where(cellsOnVertex_np >= 0, cellsOnVertex_np, mesh.nCells)
    min_cell_v = np.min(cov_safe, axis=0)
    vertex_owner = np.where(
        min_cell_v < mesh.nCells,
        cell_owner[np.minimum(min_cell_v, mesh.nCells - 1)],
        0,
    ).astype(np.int32)
    vert_perm = np.argsort(vertex_owner, kind="stable")
    vert_inv = np.empty_like(vert_perm)
    vert_inv[vert_perm] = np.arange(len(vert_perm))

    # --- Helper: remap connectivity values through an inverse permutation ---
    def remap_conn(conn, inv_perm):
        """Remap integer connectivity array: old_global → new_global."""
        arr = np.asarray(conn)
        valid = arr >= 0
        safe = np.where(valid, arr, 0)
        remapped = np.where(valid, inv_perm[safe], -1)
        return jnp.array(remapped, dtype=conn.dtype)

    # --- Helper: reorder along entity axis (last axis for (K, nEntities)) ---
    def reorder_col(arr, perm):
        """Reorder columns: arr[:, perm] for 2D, arr[perm] for 1D."""
        a = np.asarray(arr)
        if a.ndim == 1:
            return jnp.array(a[perm], dtype=arr.dtype)
        return jnp.array(a[:, perm], dtype=arr.dtype)

    def reorder_1d(arr, perm):
        a = np.asarray(arr)
        return jnp.array(a[perm], dtype=arr.dtype)

    reordered = VoronoiMesh(
        nCells=mesh.nCells,
        nEdges=mesh.nEdges,
        nVertices=mesh.nVertices,
        maxEdges=mesh.maxEdges,
        vertexDegree=mesh.vertexDegree,
        radius=mesh.radius,
        # Cell coordinates (reorder by cell_perm)
        latCell=reorder_1d(mesh.latCell, cell_perm),
        lonCell=reorder_1d(mesh.lonCell, cell_perm),
        xCell=reorder_1d(mesh.xCell, cell_perm),
        yCell=reorder_1d(mesh.yCell, cell_perm),
        zCell=reorder_1d(mesh.zCell, cell_perm),
        # Edge coordinates (reorder by edge_perm)
        latEdge=reorder_1d(mesh.latEdge, edge_perm),
        lonEdge=reorder_1d(mesh.lonEdge, edge_perm),
        xEdge=reorder_1d(mesh.xEdge, edge_perm),
        yEdge=reorder_1d(mesh.yEdge, edge_perm),
        zEdge=reorder_1d(mesh.zEdge, edge_perm),
        # Vertex coordinates (reorder by vert_perm)
        latVertex=reorder_1d(mesh.latVertex, vert_perm),
        lonVertex=reorder_1d(mesh.lonVertex, vert_perm),
        xVertex=reorder_1d(mesh.xVertex, vert_perm),
        yVertex=reorder_1d(mesh.yVertex, vert_perm),
        zVertex=reorder_1d(mesh.zVertex, vert_perm),
        # Connectivity: reorder columns AND remap values
        cellsOnEdge=remap_conn(reorder_col(mesh.cellsOnEdge, edge_perm), cell_inv),
        edgesOnCell=remap_conn(reorder_col(mesh.edgesOnCell, cell_perm), edge_inv),
        verticesOnCell=remap_conn(reorder_col(mesh.verticesOnCell, cell_perm), vert_inv),
        verticesOnEdge=remap_conn(reorder_col(mesh.verticesOnEdge, edge_perm), vert_inv),
        edgesOnVertex=remap_conn(reorder_col(mesh.edgesOnVertex, vert_perm), edge_inv),
        cellsOnVertex=remap_conn(reorder_col(mesh.cellsOnVertex, vert_perm), cell_inv),
        cellsOnCell=remap_conn(reorder_col(mesh.cellsOnCell, cell_perm), cell_inv),
        edgesOnEdge=remap_conn(reorder_col(mesh.edgesOnEdge, edge_perm), edge_inv),
        nEdgesOnCell=reorder_1d(mesh.nEdgesOnCell, cell_perm),
        nEdgesOnEdge=reorder_1d(mesh.nEdgesOnEdge, edge_perm),
        # Geometry (reorder by entity)
        areaCell=reorder_1d(mesh.areaCell, cell_perm),
        areaTriangle=reorder_1d(mesh.areaTriangle, vert_perm),
        dcEdge=reorder_1d(mesh.dcEdge, edge_perm),
        dvEdge=reorder_1d(mesh.dvEdge, edge_perm),
        angleEdge=reorder_1d(mesh.angleEdge, edge_perm),
        # Weights and signs (reorder columns by entity)
        weightsOnEdge=reorder_col(mesh.weightsOnEdge, edge_perm),
        kiteAreasOnVertex=reorder_col(mesh.kiteAreasOnVertex, vert_perm),
        fEdge=reorder_1d(mesh.fEdge, edge_perm),
        fVertex=reorder_1d(mesh.fVertex, vert_perm),
        edgeSignOnCell=reorder_col(mesh.edgeSignOnCell, cell_perm),
        edgeSignOnVertex=reorder_col(mesh.edgeSignOnVertex, vert_perm),
        meshDensity=reorder_1d(mesh.meshDensity, cell_perm),
    )

    # --- Pad so that nCells and nEdges are divisible by n_devices ---
    return _pad_voronoi_for_sharding(reordered, n_devices)
.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:639:docs/performance/scaling/levante_campaign_2026-07-24.md-1671-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:658:docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:687:docs/performance/scaling/levante_campaign_2026-07-24.md-1728-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:849:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:860:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:17:3. scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch — s8 lloyd0 np8/16/32 matched to s9 protocol (steps 12/warmup 3, sfc, reorder-for 128) with falsifiability block.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:24:/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_2026-08-02.md' -g 'levante_campaign_2026-07-24.md' -g 'mpas_s9_ensemble.sbatch' -g 'mpas_s8_lloyd0_ladder.sbatch' -g 'atm_latlon_hundreds.sbatch' -g 'atm_latlon2d_cpu_hundreds.sbatch' -g 'prewarm_s10.sbatch' -g 'plot_scaling_paper_figure.py' -g 'README*' -g 'pyproject.toml' -g 'pytest.ini' -g 'tox.ini' -g 'setup.cfg'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:43:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:87:2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:967:12947:5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:999:12990:5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1639:1662-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1654:1677:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1688:1711:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1693:/usr/bin/bash -lc "sed -n '1587,1735p' docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && printf '\\n--- s8 ladder ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch && printf '\\n--- latlon hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch && printf '\\n--- latlon2d hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && printf '\\n--- prewarm ---\\n' && nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch && printf '\\n--- plot ---\\n' && nl -ba scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1770:  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1785:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1819:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1946:    21	#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1969:    44	  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2186:    68	        note="s8 production mesh;\ns9 lloyd-0 synthetic",
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2367:/usr/bin/bash -lc "nl -ba scripts/cluster/scaling_levante/_env.sh && printf '\\n--- git diff stats/status ---\\n' && git status --short && git diff --check HEAD && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch scripts/cluster/scaling_levante/prewarm_s10.sbatch scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2463:?? scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2550:+  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2565:+2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2599:+| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2647:+        note="s8 production mesh;\ns9 lloyd-0 synthetic",
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2698:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2699:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-14-# subdiv-9 ladder (job 26600095) is the lloyd=0 synthetic family — so no
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2700:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2701:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:16:# np8/16/32 on the SAME lloyd=0 family, same sfc + --reorder-for 128,
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2702:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-17-# same steps/warmup as the s9 ladder.  Weak pairs at matched cells/GPU
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2703:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-18-# (81.9k / 41.0k / 20.5k) are computed ONLY from these rows vs 26600095.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2704:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-19-#
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2705:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-20-# Falsifiability, written BEFORE submit:
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2706:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:21:#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2707:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-22-#   CONFIRM (a matched-tile scale-out term exists): s9/s8 ratios at
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2708:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-23-#             matched cells/GPU stay well above 1 (prior draft saw
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2709:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-24-#             1.80/1.35/1.41 on the CONFOUNDED pairs)
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2711:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-45-  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2712:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-46-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2713:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-47-    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2714:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:48:      --multicontroller --n-devices "$NP" \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2715:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-49-      --subdivision 8 --nlev 26 --steps 12 --warmup 3 --lloyd 0 \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2716:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:50:      --partition-method sfc --reorder-for 128 \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2717:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-51-      --out "$OUTDIR/np${NP}.jsonl" || { echo "np$NP FAILED"; rc=1; }
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2718:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-52-done
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2719:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-53-echo "=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ==="
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2721:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-57-import json,sys
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2722:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-58-try:
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2723:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-59-    d=json.loads(open('$F').readline())
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2724:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:60:    print(f'np$NP: {d[\"steady_median_ms\"]:8.2f} ms')
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2725:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-61-except Exception as e:
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2726:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-62-    print('np$NP: MISSING/UNPARSEABLE ->', e); sys.exit(1)" \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2727:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-63-    || { echo "np$NP receipt invalid"; rc=1; }
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6271:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:7:#SBATCH --gpus-per-node=4
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6272:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:45:  JAX_ENABLE_X64=0 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=4 \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6273:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:46:      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6309:18-2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6463:6465:/usr/bin/bash -lc "python -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' && printf '%s\\n' '--- stored MPAS receipts ---' && rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200 && printf '%s\\n' '--- campaign figure handling of cell count / plot source ---' && rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6468:6470:/usr/bin/bash -lc "/usr/bin/python3 -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' ; printf '%s\\n' '--- stored MPAS receipts ---'; rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200; printf '%s\\n' '--- campaign figure handling of cell count / plot source ---'; rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6479:6512-./mpasoc_2x2b.26508336.log:2128:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12748.4ms fused=313.165ms/step (probe_latency=308.818ms) gate_loop_latency=312.67ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6488:6521-./mpasoc_2x2b.26508336.log:4492:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13919.2ms fused=617.6562ms/step (probe_latency=585.002ms) gate_loop_latency=589.96ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6499:6554-./mpasoc_rpn.26505286.log-7160-[1785176811.585797] [l40609:286091:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6502:6557-./mpasoc_rpn.26505286.log:7163:[mpas-ocean np=32 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=15717.1ms fused=861.2525ms/step (probe_latency=861.989ms) gate_loop_latency=866.56ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6511:6566-./mpasoc_rpn.26505286.log:8069:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13606.9ms fused=494.8875ms/step (probe_latency=460.002ms) gate_loop_latency=467.45ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6520:6575-./mpasoc_rpn.26505286.log:9854:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12817.6ms fused=309.0494ms/step (probe_latency=309.777ms) gate_loop_latency=312.49ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6529:6584-./mpasoc_rpn.26505286.log:13465:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12760.0ms fused=254.4062ms/step (probe_latency=233.377ms) gate_loop_latency=237.42ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6539:6685-./mpasoc_512.26508063.log:14184:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14009.9ms fused=194.8ms/step (probe_latency=175.611ms) gate_loop_latency=179.54ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6548:6694-./cpu_f32.26534068.log:19850:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13967.3ms fused=491.1337ms/step (probe_latency=457.995ms) gate_loop_latency=460.91ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6557:6703-./cpu_f32.26534068.log:23769:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12882.5ms fused=258.8344ms/step (probe_latency=233.395ms) gate_loop_latency=238.02ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6566:6712-./cpu_f32.26534068.log:31939:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14299.2ms fused=195.6313ms/step (probe_latency=174.221ms) gate_loop_latency=179.19ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6592:6788:./mpasoc_metis.26600094.log:2150:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6609:6805:./mpasoc_metis.26600094.log:4388:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6618:6814:./mpasoc_metis.26600094.log:6256:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6640:6857-./mpasoc_out.26504842.log:7183:[mpas-ocean np=32 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=15274.6ms fused=563.7044ms/step (probe_latency=591.262ms) gate_loop_latency=586.85ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6649:6866-./mpasoc_out.26504842.log:8034:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13659.2ms fused=404.4325ms/step (probe_latency=382.988ms) gate_loop_latency=383.33ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6658:6875-./mpasoc_out.26504842.log:9793:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12670.2ms fused=309.9656ms/step (probe_latency=308.874ms) gate_loop_latency=312.57ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6667:6884-./mpasoc_out.26504842.log:13260:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12914.2ms fused=298.2331ms/step (probe_latency=269.868ms) gate_loop_latency=277.99ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6677:7019-./mpasoc_2x2.26508258.log:3991:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13799.9ms fused=610.4875ms/step (probe_latency=580.0ms) gate_loop_latency=584.34ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6951:10381:/usr/bin/bash -lc "printf '%s\\n' '--- s9 32 full exact record ---'; rg -n '"'^'"\\{\"component\": \"mpas_atm\"' mpas_s9.26600095.log | head -n 1; printf '%s\\n' '--- s8 weak-pair receipt candidates ---'; rg -n -l 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | sort; rg -n -C 0 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | head -n 100; printf '%s\\n' '--- lloyd state in old benchmark commit ---'; git show 10533f1f9:scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,220p'; printf '%s\\n' '--- phase job log start and launch flags ---'; sed -n '1,180p' mpas_s9.26600095.log; printf '%s\\n' '--- METIS job command echo / distribution provenance ---'; rg -n -C 3 -i 'distribution|block:|srun|"'^===|command|geometric|metis'"' mpasoc_metis.26600094.log | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6986:10883-2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7022:11075:4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7040:11219-6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7067:11254-The receipts expose two substantive confounds beyond wording: METIS does not hold every rank at 5,120 cells (it spans 5,093–5,144 at 128 ranks), and the s8 comparison rows predate the `--lloyd` control and therefore use the default relaxed mesh while s9 explicitly uses `lloyd=0`. Those invalidate the strong causal versions of the lever claims.
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7090:11277-2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7110:11307:4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7121:11342-6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7131:12350-6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:7539:/usr/bin/bash -lc "rg -n -C 5 'SLURM_STEP_NODELIST|coordinator|SLURM_JOB_ID|port|initialize' packages/core/legoesm/parallel/early_init.py && printf '\\n--- srun docs local available ---\\n' && (srun --help 2>&1 | rg -n -C 2 -- '--exact|exclusive|gpus-per-node|gpu-bind' || true) && printf '\\n--- shell static syntax ---\\n' && bash -n scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch scripts/cluster/scaling_levante/prewarm_s10.sbatch && echo bash-syntax-ok && printf '\\n--- py compile ---\\n' && /usr/bin/python3 -m py_compile scripts/plot/plot_scaling_paper_figure.py && echo pycompile-ok" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:119:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:313:?? scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:332:3. scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch — s8 lloyd0 np8/16/32 matched to s9 protocol (steps 12/warmup 3, sfc, reorder-for 128) with falsifiability block.
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:339:/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_2026-08-02.md' -g 'levante_campaign_2026-07-24.md' -g 'mpas_s9_ensemble.sbatch' -g 'mpas_s8_lloyd0_ladder.sbatch' -g 'atm_latlon_hundreds.sbatch' -g 'atm_latlon2d_cpu_hundreds.sbatch' -g 'prewarm_s10.sbatch' -g 'plot_scaling_paper_figure.py' -g 'README*' -g 'pyproject.toml' -g 'pytest.ini' -g 'tox.ini' -g 'setup.cfg'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:358:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:402:2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:694:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:912:1693-/usr/bin/bash -lc "sed -n '1587,1735p' docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && printf '\\n--- s8 ladder ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch && printf '\\n--- latlon hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch && printf '\\n--- latlon2d hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && printf '\\n--- prewarm ---\\n' && nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch && printf '\\n--- plot ---\\n' && nl -ba scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:935:1770-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:948:1785:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:998:6309-18-2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1544:1724-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1660:  1671	  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1675:  1686	2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1713:  1724	| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:2299:    69	        note="s8 production mesh;\ns9 lloyd-0 synthetic",
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3106:1686-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3659:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-6-#SBATCH --nodes=8
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3660:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-7-#SBATCH --gpus-per-node=4
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3661:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:8:#SBATCH --exclusive
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3662:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-9-#SBATCH --mem=0
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3663:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-10-#SBATCH --time=01:30:00
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4330:1724-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4493:rg --files -uu | rg '(26600094|26600095|26534060|26502539|26628072|26628076)' || true
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4504:mpas_s8_l0.26628076.log
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4608:2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4620:4388:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4627:6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:18:2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6465:/usr/bin/bash -lc "python -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' && printf '%s\\n' '--- stored MPAS receipts ---' && rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200 && printf '%s\\n' '--- campaign figure handling of cell count / plot source ---' && rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6470:/usr/bin/bash -lc "/usr/bin/python3 -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' ; printf '%s\\n' '--- stored MPAS receipts ---'; rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200; printf '%s\\n' '--- campaign figure handling of cell count / plot source ---'; rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6485:./mpas_bar.26480310.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=4574.5ms steady_median=22.03ms/step (per-step: [4574.5, 22.4, 22.2, 22.2, 22.2, 22.2, 22.2, 22.0, 22.0, 22.0, 22.0, 22.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6494:./mpas_bar.26480310.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=3717.8ms steady_median=14.09ms/step (per-step: [3717.8, 15.1, 14.3, 14.2, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6503:./mpas_bar.26480310.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=4522.9ms steady_median=7.04ms/step (per-step: [4522.9, 8.1, 7.3, 7.2, 7.1, 7.0, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6512:./mpasoc_2x2b.26508336.log:2128:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12748.4ms fused=313.165ms/step (probe_latency=308.818ms) gate_loop_latency=312.67ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6521:./mpasoc_2x2b.26508336.log:4492:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13919.2ms fused=617.6562ms/step (probe_latency=585.002ms) gate_loop_latency=589.96ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6530:./mpas_nsys.26479922.log:9:[mpas nd=2 L8 nCells=655376 nlev=26] compile=8379.7ms steady_median=21.01ms/step (per-step: [8379.7, 21.3, 21.1, 21.0, 21.3, 21.0, 21.6, 21.0, 21.0, 20.9, 20.8, 20.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6539:./mpas_nsys.26479922.log:85:[mpas nd=4 L8 nCells=655376 nlev=26] compile=6209.0ms steady_median=17.78ms/step (per-step: [6209.0, 18.1, 18.0, 17.9, 18.2, 17.8, 17.7, 18.5, 17.8, 17.7, 17.6, 18.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6548:./mpas_nsys.26479922.log:175:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6621.3ms steady_median=7.17ms/step (per-step: [6621.3, 8.9, 8.4, 7.2, 7.2, 7.1, 7.1, 7.1, 11.7, 9.0, 10.9, 9.7])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6557:./mpasoc_rpn.26505286.log:7163:[mpas-ocean np=32 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=15717.1ms fused=861.2525ms/step (probe_latency=861.989ms) gate_loop_latency=866.56ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6566:./mpasoc_rpn.26505286.log:8069:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13606.9ms fused=494.8875ms/step (probe_latency=460.002ms) gate_loop_latency=467.45ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6575:./mpasoc_rpn.26505286.log:9854:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12817.6ms fused=309.0494ms/step (probe_latency=309.777ms) gate_loop_latency=312.49ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6584:./mpasoc_rpn.26505286.log:13465:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12760.0ms fused=254.4062ms/step (probe_latency=233.377ms) gate_loop_latency=237.42ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6593:./mpas32.26549646.log-68-[mpas nd=32 L8 nCells=655392 nlev=26] compile=6060.7ms steady_median=8.13ms/step (per-step: [6060.7, 11.4, 8.6, 8.3, 8.3, 8.2, 8.1, 8.1, 8.0, 8.0, 8.8, 8.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6601:./mpas32.26549646.log-83-[mpas nd=32 L8 nCells=655392 nlev=26] compile=7409.8ms steady_median=14.26ms/step (per-step: [7409.8, 18.2, 15.9, 15.2, 14.4, 14.3, 14.3, 14.3, 14.2, 14.3, 14.2, 14.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6608:./mpas_f64r.26493734.log:4:[mpas nd=4 L8 nCells=655376 nlev=26] compile=6089.0ms steady_median=20.09ms/step (per-step: [6089.0, 19.7, 19.6, 19.7, 20.1, 20.2, 20.1, 20.1, 19.8, 20.1, 20.1, 19.7])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6617:./mpas_f64r.26493734.log:19:[mpas nd=8 L8 nCells=655376 nlev=26] compile=4603.3ms steady_median=18.98ms/step (per-step: [4603.3, 19.4, 19.5, 19.1, 19.2, 19.0, 19.1, 18.9, 18.9, 19.0, 18.9, 18.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6626:./legoesm_gpu_multinode.26449147.log:6535:[mpas nd=16 L8 nCells=655376 nlev=26] compile=6357.0ms steady_median=7.08ms/step (per-step: [6357.0, 9.3, 7.9, 7.5, 7.4, 7.9, 7.2, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.2, 7.1, 7.1, 7.0, 7.0, 6.9, 6.9, 7.0, 6.9, 6.9, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6635:./mpas_fuse.26480162.log:12:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5257.7ms steady_median=17.27ms/step (per-step: [5257.7, 17.6, 17.5, 17.4, 17.3, 17.2, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6644:./mpas_fuse.26480162.log:89:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5272.1ms steady_median=17.30ms/step (per-step: [5272.1, 17.6, 17.5, 17.3, 17.3, 17.4, 17.3, 17.3, 17.2, 17.3, 17.1, 17.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6652:./mpas128.26538474.log:131:{"component": "mpas_atm", "subdivision": 8, "n_devices": 64, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 5605.2, "steady_median_ms": 5.27, "steady_min_ms": 5.15, "per_step_ms": [5605.2, 8.5, 5.9, 6.2, 5.5, 5.4, 5.3, 5.2, 5.3, 5.2, 5.1, 5.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 5.267729051411152, "total_cells": 17042688, "sypd": 15.59220734634407, "mcells_per_s": 3235.3007973017325, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:06:38.803951+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 266292, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 266292}}}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6653:./mpas128.26538474.log-132-[mpas nd=64 L8 nCells=655488 nlev=26] compile=5605.2ms steady_median=5.27ms/step (per-step: [5605.2, 8.5, 5.9, 6.2, 5.5, 5.4, 5.3, 5.2, 5.3, 5.2, 5.1, 5.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6660:./mpas128.26538474.log:420:{"component": "mpas_atm", "subdivision": 8, "n_devices": 128, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7701.1, "steady_median_ms": 6.47, "steady_min_ms": 6.21, "per_step_ms": [7701.1, 11.9, 8.6, 6.5, 6.6, 6.3, 6.9, 7.1, 6.3, 6.2, 6.8, 6.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 6.468050996772945, "total_cells": 17042688, "sypd": 12.698651209605844, "mcells_per_s": 2634.903158386194, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:14:39.861856+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 133146, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 133146}}}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6661:./mpas128.26538474.log-421-[mpas nd=128 L8 nCells=655488 nlev=26] compile=7701.1ms steady_median=6.47ms/step (per-step: [7701.1, 11.9, 8.6, 6.5, 6.6, 6.3, 6.9, 7.1, 6.3, 6.2, 6.8, 6.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6668:./mpas128.26538474.log:750:{"component": "mpas_atm", "subdivision": 8, "n_devices": 64, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 5597.0, "steady_median_ms": 8.96, "steady_min_ms": 8.84, "per_step_ms": [5597.0, 12.2, 9.9, 9.6, 9.0, 9.0, 9.0, 9.0, 9.0, 8.9, 8.9, 8.8], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 8.959555067121983, "total_cells": 17042688, "sypd": 9.167366347841075, "mcells_per_s": 1902.180172153851, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:18:32.347778+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 266292, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "9745e4da6", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 266292}}}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6669:./mpas128.26538474.log-751-[mpas nd=64 L8 nCells=655488 nlev=26] compile=5597.0ms steady_median=8.96ms/step (per-step: [5597.0, 12.2, 9.9, 9.6, 9.0, 9.0, 9.0, 9.0, 9.0, 8.9, 8.9, 8.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6676:./mpas128.26538474.log:945:{"component": "mpas_atm", "subdivision": 8, "n_devices": 128, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 6265.2, "steady_median_ms": 11.41, "steady_min_ms": 11.28, "per_step_ms": [6265.2, 18.0, 13.8, 12.2, 11.6, 11.4, 11.8, 11.4, 12.2, 11.3, 11.4, 11.3], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.414309963583946, "total_cells": 17042688, "sypd": 7.195837845301824, "mcells_per_s": 1493.0984049296674, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:26:25.491528+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 133146, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "9745e4da6", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 133146}}}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6677:./mpas128.26538474.log-946-[mpas nd=128 L8 nCells=655488 nlev=26] compile=6265.2ms steady_median=11.41ms/step (per-step: [6265.2, 18.0, 13.8, 12.2, 11.6, 11.4, 11.8, 11.4, 12.2, 11.3, 11.4, 11.3])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6685:./mpasoc_512.26508063.log:14184:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14009.9ms fused=194.8ms/step (probe_latency=175.611ms) gate_loop_latency=179.54ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6694:./cpu_f32.26534068.log:19850:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13967.3ms fused=491.1337ms/step (probe_latency=457.995ms) gate_loop_latency=460.91ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6703:./cpu_f32.26534068.log:23769:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12882.5ms fused=258.8344ms/step (probe_latency=233.395ms) gate_loop_latency=238.02ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6712:./cpu_f32.26534068.log:31939:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14299.2ms fused=195.6313ms/step (probe_latency=174.221ms) gate_loop_latency=179.19ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6766:./mpas_hlo.26480096.log:11:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5597.9ms steady_median=17.51ms/step (per-step: [5597.9, 17.9, 17.6, 17.5, 17.5, 17.5])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6788:./mpasoc_metis.26600094.log:2150:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6805:./mpasoc_metis.26600094.log:4388:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6814:./mpasoc_metis.26600094.log:6256:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6830:./mpas_f64.26493638.log:4:[mpas nd=2 L8 nCells=655376 nlev=26] compile=7096.0ms steady_median=38.34ms/step (per-step: [7096.0, 38.6, 38.5, 37.0, 38.4, 38.3, 38.4, 38.3, 38.4, 38.5, 38.3, 38.3])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6839:./mpas_f64.26493638.log:14:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5999.2ms steady_median=20.10ms/step (per-step: [5999.2, 19.8, 19.6, 19.8, 20.2, 20.1, 20.1, 20.2, 20.1, 20.1, 20.1, 20.1])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6848:./mpas_f64.26493638.log:29:[mpas nd=8 L8 nCells=655376 nlev=26] compile=4624.3ms steady_median=21.42ms/step (per-step: [4624.3, 23.2, 21.8, 21.8, 21.8, 21.7, 21.4, 21.5, 21.4, 21.2, 20.6, 20.7])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6857:./mpasoc_out.26504842.log:7183:[mpas-ocean np=32 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=15274.6ms fused=563.7044ms/step (probe_latency=591.262ms) gate_loop_latency=586.85ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6866:./mpasoc_out.26504842.log:8034:[mpas-ocean np=64 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13659.2ms fused=404.4325ms/step (probe_latency=382.988ms) gate_loop_latency=383.33ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6875:./mpasoc_out.26504842.log:9793:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12670.2ms fused=309.9656ms/step (probe_latency=308.874ms) gate_loop_latency=312.57ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6884:./mpasoc_out.26504842.log:13260:[mpas-ocean np=256 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12914.2ms fused=298.2331ms/step (probe_latency=269.868ms) gate_loop_latency=277.99ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6893:./mpas_bar.26486123.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=7009.7ms steady_median=20.65ms/step (per-step: [7009.7, 21.2, 21.0, 20.9, 20.9, 20.8, 20.6, 20.6, 20.6, 20.6, 20.6, 20.6])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6902:./mpas_bar.26486123.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4979.1ms steady_median=17.78ms/step (per-step: [4979.1, 18.0, 17.9, 18.1, 17.8, 17.9, 17.8, 17.8, 17.7, 17.5, 17.8, 17.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6911:./mpas_bar.26486123.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6179.0ms steady_median=7.02ms/step (per-step: [6179.0, 8.4, 8.0, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6920:./mpas_bar.26486288.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=6962.6ms steady_median=19.86ms/step (per-step: [6962.6, 21.1, 21.0, 20.9, 20.8, 19.8, 19.8, 19.8, 19.9, 19.9, 19.9, 19.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6929:./mpas_bar.26486288.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=3696.7ms steady_median=14.12ms/step (per-step: [3696.7, 14.4, 14.3, 14.2, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6938:./mpas_bar.26486288.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6232.7ms steady_median=6.99ms/step (per-step: [6232.7, 7.5, 7.3, 7.0, 7.2, 7.0, 7.0, 7.0, 6.9, 7.0, 6.9, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6947:./mpas_recheck.26454618.log:8:[mpas nd=2 L8 nCells=655376 nlev=26] compile=7021.0ms steady_median=19.92ms/step (per-step: [7021.0, 21.1, 21.0, 20.9, 21.0, 20.9, 20.9, 20.8, 20.6, 20.6, 19.8, 19.9, 19.9, 20.0, 19.9, 19.9, 19.9, 19.8, 19.9, 19.9, 20.0, 19.9, 20.0, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6956:./mpas_recheck.26454618.log:26:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4983.0ms steady_median=17.14ms/step (per-step: [4983.0, 18.8, 17.9, 17.7, 17.8, 17.8, 17.7, 17.9, 17.8, 17.7, 17.7, 17.7, 17.9, 18.1, 17.5, 17.3, 17.1, 17.0, 17.0, 17.0, 17.0, 17.1, 17.0, 17.1, 17.0, 17.0, 17.0, 17.0, 17.2, 17.0, 17.0, 17.0, 17.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6965:./mpas_recheck.26454618.log:49:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4973.9ms steady_median=17.18ms/step (per-step: [4973.9, 18.0, 17.8, 17.9, 17.7, 17.8, 17.7, 17.8, 17.9, 17.7, 17.8, 17.8, 17.7, 17.5, 17.7, 17.6, 17.2, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.2, 17.0, 17.0, 17.0, 17.2, 16.9, 16.9, 17.0, 17.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6974:./mpas_recheck.26454618.log:80:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6067.6ms steady_median=6.87ms/step (per-step: [6067.6, 8.4, 7.2, 7.0, 7.0, 6.9, 6.9, 6.9, 6.9, 6.8, 6.8, 6.9, 6.9, 6.8, 6.9, 6.9, 6.8, 6.9, 6.8, 6.9, 6.9, 6.9, 6.9, 6.8, 6.9, 6.9, 6.8, 6.8, 6.7, 6.8, 6.8, 6.7, 6.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6983:./mpas_recheck.26454618.log:111:[mpas nd=2 L8 nCells=655376 nlev=26] compile=6961.9ms steady_median=19.87ms/step (per-step: [6961.9, 21.2, 21.0, 20.9, 20.9, 21.0, 20.9, 20.9, 21.0, 20.9, 20.9, 20.2, 19.9, 19.9, 19.9, 19.8, 19.9, 19.9, 19.9, 19.8, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9, 19.8, 19.8, 19.9])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6992:./mpas_recheck.26454618.log:128:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4976.5ms steady_median=17.04ms/step (per-step: [4976.5, 18.0, 17.8, 17.8, 18.1, 17.8, 17.7, 17.7, 17.7, 17.7, 17.5, 17.6, 17.3, 17.6, 17.5, 17.2, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7001:./mpas_recheck.26454618.log:151:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4976.3ms steady_median=17.06ms/step (per-step: [4976.3, 19.0, 17.9, 17.8, 18.1, 17.8, 17.8, 17.8, 17.8, 17.8, 17.8, 17.7, 17.5, 17.3, 17.5, 17.6, 17.1, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7010:./mpas_recheck.26454618.log:182:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6244.8ms steady_median=6.97ms/step (per-step: [6244.8, 9.2, 7.6, 7.1, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.9, 6.9, 7.0, 7.1, 6.9, 6.9, 7.0, 7.0, 6.9, 6.9, 6.9, 6.9, 6.9, 6.9, 6.9])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7019:./mpasoc_2x2.26508258.log:3991:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13799.9ms fused=610.4875ms/step (probe_latency=580.0ms) gate_loop_latency=584.34ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7028:./mpas_codegen.26455948.log:12:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5243.6ms steady_median=17.12ms/step (per-step: [5243.6, 17.6, 17.4, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.1, 17.3, 17.3, 17.2, 17.2, 17.1, 16.8, 16.6, 16.6, 16.6, 16.6, 16.5, 16.6, 16.5, 16.6, 16.5, 16.6, 16.5, 16.6, 16.6])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7037:./mpas_codegen.26455948.log:35:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5255.0ms steady_median=17.06ms/step (per-step: [5255.0, 17.5, 17.4, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 17.2, 17.3, 17.2, 17.3, 17.0, 17.2, 17.1, 17.2, 17.1, 16.6, 16.6, 16.6, 16.6, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5, 16.9, 16.5, 16.5])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7046:./mpas_codegen.26455948.log:68:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5265.5ms steady_median=17.10ms/step (per-step: [5265.5, 17.6, 17.4, 17.4, 17.4, 17.3, 17.3, 17.4, 17.2, 17.1, 17.1, 17.1, 17.2, 17.1, 17.1, 17.1, 17.1, 17.1, 17.2, 16.8, 16.7, 16.6, 16.8, 16.9, 16.8, 16.8, 16.7, 16.8, 16.9, 16.8, 16.7, 16.8, 16.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7055:./mpas_codegen.26455948.log:91:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5244.5ms steady_median=17.01ms/step (per-step: [5244.5, 17.8, 17.6, 17.5, 17.5, 17.4, 17.4, 17.4, 17.4, 17.4, 17.4, 17.2, 17.4, 17.3, 17.3, 17.3, 17.2, 17.2, 16.8, 16.7, 16.6, 16.8, 16.6, 16.7, 16.6, 16.6, 16.6, 16.7, 16.7, 16.6, 16.7, 16.7, 16.6])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7064:./atm_ladders.26454476.log:10:[mpas nd=2 L8 nCells=655376 nlev=26] compile=7123.9ms steady_median=19.83ms/step (per-step: [7123.9, 21.2, 21.0, 20.9, 20.9, 20.9, 20.9, 20.6, 20.7, 19.8, 19.8, 19.8, 19.8, 19.9, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.8, 19.9, 19.9, 19.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7073:./atm_ladders.26454476.log:27:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5067.3ms steady_median=17.26ms/step (per-step: [5067.3, 18.5, 18.1, 18.0, 17.9, 17.9, 18.0, 18.0, 18.0, 17.8, 17.4, 17.3, 17.2, 17.3, 17.3, 17.3, 17.2, 17.3, 17.3, 17.2, 17.3, 17.3, 17.2, 17.3, 17.3, 17.2, 17.3, 17.2, 17.2, 17.3, 17.2, 17.2, 17.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7082:./atm_ladders.26454476.log:55:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6395.2ms steady_median=6.96ms/step (per-step: [6395.2, 7.9, 7.9, 7.2, 7.1, 7.2, 7.1, 7.1, 7.1, 7.1, 7.1, 7.0, 7.1, 7.1, 7.0, 7.0, 7.0, 6.9, 7.0, 7.0, 6.9, 6.9, 7.0, 6.9, 6.9, 6.9, 6.8, 6.8, 6.8, 6.9, 6.9, 6.9, 6.9])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7091:./atm_ladders.26454476.log:114:[mpas nd=16 L8 nCells=655376 nlev=26] compile=6542.5ms steady_median=7.10ms/step (per-step: [6542.5, 10.4, 7.9, 7.4, 7.3, 7.3, 7.3, 7.2, 7.2, 7.6, 7.2, 7.2, 7.2, 7.2, 7.1, 7.1, 7.0, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0, 7.1, 7.0, 7.2, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7100:./mpas_bar.26480261.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=6578.8ms steady_median=21.40ms/step (per-step: [6578.8, 22.0, 21.5, 21.4, 21.4, 21.4, 21.4, 21.4, 21.2, 21.1, 21.2, 21.1])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7109:./mpas_bar.26480261.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4134.1ms steady_median=16.58ms/step (per-step: [4134.1, 16.9, 16.6, 16.6, 16.6, 16.6, 16.6, 16.5, 16.5, 16.5, 16.6, 16.6])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7118:./mpas_bar.26480261.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=5041.0ms steady_median=7.01ms/step (per-step: [5041.0, 8.2, 8.1, 7.1, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 6.9])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7127:./mpas_part_ab.26455829.log:12:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5061.7ms steady_median=17.22ms/step (per-step: [5061.7, 18.1, 17.9, 18.0, 17.8, 18.0, 17.8, 18.0, 17.8, 18.2, 17.8, 17.8, 17.7, 17.9, 17.6, 17.6, 17.3, 17.0, 17.0, 17.0, 17.2, 17.0, 17.0, 17.2, 17.1, 17.1, 17.0, 17.1, 17.2, 17.0, 17.2, 17.0, 17.2])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7136:./mpas_part_ab.26455829.log:43:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6058.9ms steady_median=7.00ms/step (per-step: [6058.9, 8.9, 7.5, 7.1, 7.1, 7.1, 7.0, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 6.8, 6.9, 6.9, 6.8, 6.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7145:./mpas_part_ab.26455829.log:72:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5311.8ms steady_median=17.20ms/step (per-step: [5311.8, 17.9, 17.7, 17.5, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.6, 17.5, 17.5, 17.2, 17.3, 16.8, 16.8, 16.8, 16.9, 17.1, 16.8, 16.9, 17.2, 16.8, 17.2, 16.8, 16.9, 16.8, 16.8, 16.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7154:./mpas_part_ab.26455829.log:103:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6400.5ms steady_median=6.97ms/step (per-step: [6400.5, 7.9, 7.7, 7.1, 7.1, 7.1, 7.0, 7.0, 7.0, 7.1, 7.0, 7.0, 7.0, 6.9, 7.0, 7.0, 6.9, 7.0, 6.9, 7.0, 6.9, 6.9, 7.0, 6.9, 7.0, 7.0, 6.9, 7.0, 6.9, 7.0, 7.0, 6.9, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7163:./mpas_part_ab.26455829.log:138:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5476.0ms steady_median=19.35ms/step (per-step: [5476.0, 20.4, 20.2, 20.2, 20.1, 20.2, 20.1, 20.3, 20.1, 20.2, 20.1, 19.9, 20.1, 19.3, 19.3, 19.3, 19.3, 19.3, 19.3, 19.4, 19.3, 19.3, 19.3, 19.4, 19.3, 19.4, 19.7, 19.3, 19.3, 19.4, 19.3, 19.4, 19.3])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7172:./mpas_part_ab.26455829.log:169:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6192.6ms steady_median=6.26ms/step (per-step: [6192.6, 7.1, 7.8, 6.5, 6.4, 6.3, 6.3, 6.3, 6.3, 6.2, 6.3, 6.3, 6.3, 6.3, 6.3, 6.2, 6.3, 6.2, 6.3, 6.2, 6.3, 6.2, 6.3, 6.2, 6.3, 6.2, 6.3, 6.2, 6.3, 6.2, 6.2, 6.2, 6.3])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7181:./mpas_bar.26486163.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=6976.0ms steady_median=20.90ms/step (per-step: [6976.0, 21.2, 21.0, 20.9, 20.9, 20.9, 20.9, 20.9, 20.8, 20.5, 20.1, 19.8])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7190:./mpas_bar.26486163.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=3688.3ms steady_median=14.10ms/step (per-step: [3688.3, 14.6, 14.4, 14.2, 14.2, 14.1, 14.2, 14.1, 14.1, 14.0, 14.0, 14.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7199:./mpas_bar.26486163.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6144.4ms steady_median=7.02ms/step (per-step: [6144.4, 7.5, 7.6, 7.2, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 6.9, 7.0])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7208:./legoesm_gpu_multinode.26453240.log:2422:[mpas nd=6 L8 nCells=655362 nlev=26] compile=5732.1ms steady_median=8.66ms/step (per-step: [5732.1, 9.1, 8.9, 8.8, 8.7, 8.7, 8.7, 8.7, 8.6, 8.7, 8.6, 8.7, 8.7, 8.6, 9.1, 8.6, 9.2, 9.0, 8.6, 9.1, 8.6, 9.0, 8.5, 8.9, 8.5, 8.9, 9.0, 9.0, 8.4, 8.4, 8.5, 8.4, 8.4])
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:9998:131:{"component": "mpas_atm", "subdivision": 8, "n_devices": 64, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 5605.2, "steady_median_ms": 5.27, "steady_min_ms": 5.15, "per_step_ms": [5605.2, 8.5, 5.9, 6.2, 5.5, 5.4, 5.3, 5.2, 5.3, 5.2, 5.1, 5.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 5.267729051411152, "total_cells": 17042688, "sypd": 15.59220734634407, "mcells_per_s": 3235.3007973017325, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:06:38.803951+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 266292, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 266292}}}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:9999:420:{"component": "mpas_atm", "subdivision": 8, "n_devices": 128, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7701.1, "steady_median_ms": 6.47, "steady_min_ms": 6.21, "per_step_ms": [7701.1, 11.9, 8.6, 6.5, 6.6, 6.3, 6.9, 7.1, 6.3, 6.2, 6.8, 6.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 6.468050996772945, "total_cells": 17042688, "sypd": 12.698651209605844, "mcells_per_s": 2634.903158386194, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:14:39.861856+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 133146, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 133146}}}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:10381:/usr/bin/bash -lc "printf '%s\\n' '--- s9 32 full exact record ---'; rg -n '"'^'"\\{\"component\": \"mpas_atm\"' mpas_s9.26600095.log | head -n 1; printf '%s\\n' '--- s8 weak-pair receipt candidates ---'; rg -n -l 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | sort; rg -n -C 0 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | head -n 100; printf '%s\\n' '--- lloyd state in old benchmark commit ---'; git show 10533f1f9:scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,220p'; printf '%s\\n' '--- phase job log start and launch flags ---'; sed -n '1,180p' mpas_s9.26600095.log; printf '%s\\n' '--- METIS job command echo / distribution provenance ---'; rg -n -C 3 -i 'distribution|block:|srun|"'^===|command|geometric|metis'"' mpasoc_metis.26600094.log | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:10883:2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:11075:4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:11219:6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:11254:The receipts expose two substantive confounds beyond wording: METIS does not hold every rank at 5,120 cells (it spans 5,093–5,144 at 128 ranks), and the s8 comparison rows predate the `--lloyd` control and therefore use the default relaxed mesh while s9 explicitly uses `lloyd=0`. Those invalidate the strong causal versions of the lever claims.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:11277:2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:11307:4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:11342:6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12350:6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12947:5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12990:5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.
docs/ocean/fidelity/dino_pgf_alignment.md:27:| 8 | EOS coefficients | `a0=0.165, b0=7.6554e-1, lambda1=0.06, lambda2=0, mu1=1.4970e-4, mu2=0, nu=0` | `NemoSEOSConfig`: all seven identical (verified by instantiation) | **MATCH** |
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:21:#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:44:  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
docs/ocean/fidelity/dino_l1_exactness_audit.md:36:| EOS | **S-EOS**: `a0=0.165, b0=0.76554, λ1=0.06, λ2=0, μ1=1.497e-4, μ2=0, ν=0` |
docs/ocean/fidelity/dino_wiring_diagram.md:102:  legoESM's `nemo_seos` defaults ARE DINO's deployed coeffs (a0=0.165, b0=0.76554,
docs/performance/scaling/levante_campaign_2026-07-24.md:1671:  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
docs/performance/scaling/levante_campaign_2026-07-24.md:1690:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
docs/performance/scaling/levante_campaign_2026-07-24.md:1728:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
docs/performance/scaling/levante_campaign_2026-07-24.md:1758:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
docs/performance/scaling/levante_campaign_2026-07-24.md:1760:s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md:173:`np_seos` with the Kamm 2025 DINO coefficients (a0=0.165, b0=0.76554, λ1=0.06,
scripts/validate/ec_site_regression_golden.json:235:      "beta_eff": 0.36655469119670014
scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076",
scripts/plot/plot_scaling_paper_figure.py:69:                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
scripts/plot/plot_scaling_paper_figure.py:118:          "f32 (s8 lloyd-0)": "#009E73",
scripts/plot/plot_scaling_paper_figure.py:127:           "f32 (s8 lloyd-0)": "v",

exec
/usr/bin/bash -lc "rg -n -C 2 \"26628076|mpas_s8_l0|6\\.58|6\\.43|7\\.29|7151d12a1-dirty|n_cells.*655\" .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md | tail -300" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 325ms:
1104-./data/les_cases/TOGA/snd-1242-  -999.000000   250.000000   345.674805     0.253516    -6.658720     4.035020
1105---
1106:./data/les_cases/TOGA/snd-1275-  -999.000000   400.000000   335.113281     2.041161     3.537080     6.581560
1107-./data/les_cases/TOGA/snd-1276-  -999.000000   375.000000   337.317352     1.740726     2.006520     5.928990
1108-./data/les_cases/TOGA/snd-1277-  -999.000000   350.000000   339.278625     1.361720     1.217470     5.595150
--
1180-./scripts/plot/plot_scaling_paper_figure.py-37-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
1181-./scripts/plot/plot_scaling_paper_figure.py:38:                "s8 np32-128 26549646/26538474, s9 26600095, "
1182:./scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076",
1183-./scripts/plot/plot_scaling_paper_figure.py-40-    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
1184-./scripts/plot/plot_scaling_paper_figure.py-41-                   "lat-lon 2-D r512 26628073",
--
1188-./scripts/plot/plot_scaling_paper_figure.py-67-                                        (128, 6.47)]),
1189-./scripts/plot/plot_scaling_paper_figure.py-68-                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
1190:./scripts/plot/plot_scaling_paper_figure.py:69:                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
1191-./scripts/plot/plot_scaling_paper_figure.py-70-                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
1192-./scripts/plot/plot_scaling_paper_figure.py:71:        note="weak eff 0.53–0.67 at\nmatched tile (lloyd-0 pairs)",
--
1471-./docs/performance/scaling/levante_campaign_2026-07-24.md-1726-| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
1472-./docs/performance/scaling/levante_campaign_2026-07-24.md:1727:| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
1473:./docs/performance/scaling/levante_campaign_2026-07-24.md:1728:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
1474-./docs/performance/scaling/levante_campaign_2026-07-24.md-1729-
1475-./docs/performance/scaling/levante_campaign_2026-07-24.md-1730-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
--
1479-./docs/performance/scaling/levante_campaign_2026-07-24.md-1756-the bench metadata.)
1480-./docs/performance/scaling/levante_campaign_2026-07-24.md-1757-
1481:./docs/performance/scaling/levante_campaign_2026-07-24.md:1758:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
1482-./docs/performance/scaling/levante_campaign_2026-07-24.md-1759-
1483-./docs/performance/scaling/levante_campaign_2026-07-24.md:1760:s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
1484-./docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
1485:./docs/performance/scaling/levante_campaign_2026-07-24.md-1762-7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
1486:./docs/performance/scaling/levante_campaign_2026-07-24.md-1763-untouched): **6.58 / 6.43 / 7.29 ms**.
1487-./docs/performance/scaling/levante_campaign_2026-07-24.md-1764-
1488-./docs/performance/scaling/levante_campaign_2026-07-24.md:1765:Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):
--
1490-./docs/performance/scaling/levante_campaign_2026-07-24.md-1767-| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
1491-./docs/performance/scaling/levante_campaign_2026-07-24.md-1768-|---|---|---|---|---|
1492:./docs/performance/scaling/levante_campaign_2026-07-24.md:1769:| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
1493:./docs/performance/scaling/levante_campaign_2026-07-24.md:1770:| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
1494:./docs/performance/scaling/levante_campaign_2026-07-24.md:1771:| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
1495-./docs/performance/scaling/levante_campaign_2026-07-24.md-1772-
1496-./docs/performance/scaling/levante_campaign_2026-07-24.md-1773-* The falsifiability block's CONFIRM branch fires: ratios stay well
--
1498-./docs/performance/scaling/levante_campaign_2026-07-24.md-1775-  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
1499-./docs/performance/scaling/levante_campaign_2026-07-24.md-1776-  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
1500:./docs/performance/scaling/levante_campaign_2026-07-24.md:1777:  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
1501-./docs/performance/scaling/levante_campaign_2026-07-24.md-1778-* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
1502-./docs/performance/scaling/levante_campaign_2026-07-24.md-1779-  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
--
1605-
1606-exec
1607:/usr/bin/bash -lc "rg --files -g '"'!uv.lock'"' | rg '26628076|26600095|s8_lloyd0|s9.*ladder|levante.*(json|log)|scaling.*(json|log)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
1608- succeeded in 361ms:
1609-scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
--
1691-the bench metadata.)
1692-
1693:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
1694-
1695-s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
1696-warmup 3 — protocol-identical to the s9 ladder (26600095), git
1697:7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
1698:untouched): **6.58 / 6.43 / 7.29 ms**.
1699-
1700-Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):
--
1702-| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
1703-|---|---|---|---|---|
1704:| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
1705:| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
1706:| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
1707-
1708-* The falsifiability block's CONFIRM branch fires: ratios stay well
--
1710-  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
1711-  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
1712:  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
1713-* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
1714-  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
--
1721-  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
1722-#!/bin/bash -l
1723:#SBATCH --job-name=mpas_s8_l0
1724-#SBATCH --account=bb1596_gpu
1725-#SBATCH --partition=gpu
--
1730-#SBATCH --mem=0
1731-#SBATCH --time=01:30:00
1732:#SBATCH --output=mpas_s8_l0.%j.log
1733-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
1734-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
--
1758-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
1759-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
1760:OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s8_l0_j${SLURM_JOB_ID}}"
1761-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
1762-rc=0
--
1893---------
1894-user
1895:Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
1896-codex
1897-I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
1898-exec
1899:/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
1900- succeeded in 2031ms:
1901-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
--
1988-./scripts/run/run_omip_core2.py-1243-    for lat0, lat1, lon0, lon1 in boxes:
1989---
1990:./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-11-#SBATCH --output=mpas_s8_l0.%j.log
1991-./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-12-# DE-CONFOUND RERUN (codex round-20 item 5): every prior subdiv-8 GPU
1992-./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
--
2530-+| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
2531-+| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
2532:+| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
2533-+
2534-+s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
--
2560-+the bench metadata.)
2561-+
2562:+### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
2563-+
2564-+s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
2565-+warmup 3 — protocol-identical to the s9 ladder (26600095), git
2566:+7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
2567:+untouched): **6.58 / 6.43 / 7.29 ms**.
2568-+
2569-+Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):
--
2571-+| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
2572-+|---|---|---|---|---|
2573:+| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
2574:+| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
2575:+| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
2576-+
2577-+* The falsifiability block's CONFIRM branch fires: ratios stay well
--
2579-+  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
2580-+  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
2581:+  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
2582-+* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
2583-+  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
--
2605-+    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
2606-+                "s8 np32-128 26549646/26538474, s9 26600095, "
2607:+                "s8-lloyd0 26628076",
2608-+    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
2609-+                   "lat-lon 2-D r512 26628073",
--
2637-+                                        (128, 6.47)]),
2638-+                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
2639:+                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
2640-+                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
2641-+        note="weak eff 0.53–0.67 at\nmatched tile (lloyd-0 pairs)",
--
2683-/usr/bin/bash -lc 'rg -n "VERDICT|''^##|''^###|Arithmetic|arithmetic|residual|dirty|mesh padding|protocol" .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md | tail -80 && wc -l .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md && sed -n '"'700,820p' .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
2684- succeeded in 347ms:
2685:14:Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
2686-16:I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
2687:18:/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
2688-113:./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
2689-1484:./docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
2690:1485:./docs/performance/scaling/levante_campaign_2026-07-24.md-1762-7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
2691-1625:### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
2692-1652:  family, not the same protocol; (b) two comparator points came from the
--
2694-1679:### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
2695-1681:LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
2696:1693:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
2697-1696:warmup 3 — protocol-identical to the s9 ladder (26600095), git
2698:1697:7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
2699-1736:# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
2700-1765:  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
2701-1794:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
2702-1873:docs/performance/scaling/levante_campaign_2026-07-24.md:1761:warmup 3 — protocol-identical to the s9 ladder (26600095), git
2703:1895:Round-5 spot review, one new claim set. Job 26628076 (s8 lloyd=0 np8/16/32, protocol-identical to s9 job 26600095: f32, sfc, --reorder-for 128, steps 12/warmup 3, lloyd 0): 6.58/6.43/7.29 ms, sha 7151d12a1-dirty. New doc section (end of docs/performance/scaling/levante_campaign_2026-07-24.md) claims: clean weak pairs at matched cells/GPU 81.9k: 6.58->12.47 ratio 1.895 eff 0.53; 41.0k: 6.43->9.60 ratio 1.493 eff 0.67; 20.5k: 7.29->11.48 ratio 1.575 eff 0.64; conclusion 'matched-tile scale-out term REAL, confound only biased size (prior s8-production np8 6.92 vs lloyd0 6.58)'; mechanism kept UNATTRIBUTED; non-monotone tile dependence 'recorded, not theorised'. Figure adds series f32 (s8 lloyd-0) [(8,6.58),(16,6.43),(32,7.29)] and note 'weak eff 0.53-0.67 at matched tile (lloyd-0 pairs)'. Check arithmetic (ratios, effs), protocol-match validity (any residual confound between 26628076 and 26600095 rows: mesh padding, reorder, dirty-sha risk), and wording overclaims. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
2704-1897:I’ll audit the documented claim against the campaign records, recompute every derived value, and separate confirmed protocol matching from residual provenance risks.
2705:1899:/usr/bin/bash -lc "pwd && rg -n -i -C 3 \"26628076|26600095|81\\.9|41\\.0|20\\.5|lloyd-0|lloyd 0|reorder-for|mesh padding\" ." in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
2706-1994:./scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-15-# s8-vs-s9 weak-scaling pair was protocol-clean.  This ladder reruns s8
2707-2135:Round-4 final check. Your round-3 had 2 items: (1) doc ensemble wording contradicted v3 script (counterbalanced claim, stale job id 26628021, refute-bar attribution); (2) CPU r512 ladder needed unsupported-stack qualification. Both now patched in docs/performance/scaling/levante_campaign_2026-07-24.md: 'BRACKETED, not fully counterbalanced' + follow-up attribution language at the ensemble bullet AND its REFUTE line, job table row now 26628196 (v3, supersessions noted), CPU receipt labelled TIMING-ONLY/unvalidated with the mpi4jax 0.9.0 out-of-tested-range + UCX VM_UNMAP quotes. Read the '### Next receipts' + CPU-receipt sections and confirm consistency with scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch. Anything still inconsistent? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
--
2715-2501:+LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
2716-2565:+warmup 3 — protocol-identical to the s9 ladder (26600095), git
2717:2566:+7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
2718-2683:/usr/bin/bash -lc 'rg -n "VERDICT|''^##|''^###|Arithmetic|arithmetic|residual|dirty|mesh padding|protocol" .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md | tail -80 && wc -l .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md && sed -n '"'700,820p' .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
2719-2683 .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md
--
2841-
2842-exec
2843:/usr/bin/bash -lc "rg --files .physics-validator | head -200 && rg -n -g '"'!*.md'"' -i \"26600095|26628076|7151d12a1|steady_median_ms|2,621,568|2621568\" .physics-validator scripts docs/performance/scaling" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
2844- succeeded in 422ms:
2845-.physics-validator/ocean/inventory.md
--
3046-scripts/bench/bench_atm_latlon_spmd_scaling.py:380:        steady_median_ms=round(med, 4),
3047-scripts/plot/plot_scaling_paper_figure.py:38:                "s8 np32-128 26549646/26538474, s9 26600095, "
3048:scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076",
3049-scripts/bench/bench_ocean_mpas_scaling.py:27:the aggregator-facing ``steady_median_ms`` carries (the same deliberate
3050-scripts/bench/bench_ocean_mpas_scaling.py:726:    # aggregator-facing ``steady_median_ms`` carries the FUSED per-step
--
3841-
3842-exec
3843:/usr/bin/bash -lc "sed -n '900,1165p' packages/core/legoesm/parallel/voronoi_partition.py && rg -n \"s8.*lloyd|lloyd.*s8|26628076|nCells.*655|655,?4|655488|655360|655.*padded\" docs scripts .physics-validator/scaling_campaign -g '"'!codex_recovery_review_r5_2026-08-02.md'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
3844- succeeded in 446ms:
3845-
--
4100-.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:639:docs/performance/scaling/levante_campaign_2026-07-24.md-1671-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
4101-.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:658:docs/performance/scaling/levante_campaign_2026-07-24.md-1690-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4102:.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:687:docs/performance/scaling/levante_campaign_2026-07-24.md-1728-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4103-.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:849:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4104:.physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md:860:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4105-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:17:3. scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch — s8 lloyd0 np8/16/32 matched to s9 protocol (steps 12/warmup 3, sfc, reorder-for 128) with falsifiability block.
4106-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:24:/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_2026-08-02.md' -g 'levante_campaign_2026-07-24.md' -g 'mpas_s9_ensemble.sbatch' -g 'mpas_s8_lloyd0_ladder.sbatch' -g 'atm_latlon_hundreds.sbatch' -g 'atm_latlon2d_cpu_hundreds.sbatch' -g 'prewarm_s10.sbatch' -g 'plot_scaling_paper_figure.py' -g 'README*' -g 'pyproject.toml' -g 'pytest.ini' -g 'tox.ini' -g 'setup.cfg'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
--
4111-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1639:1662-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
4112-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1654:1677:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4113:.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1688:1711:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4114-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1693:/usr/bin/bash -lc "sed -n '1587,1735p' docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && printf '\\n--- s8 ladder ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch && printf '\\n--- latlon hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch && printf '\\n--- latlon2d hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && printf '\\n--- prewarm ---\\n' && nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch && printf '\\n--- plot ---\\n' && nl -ba scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
4115-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1770:  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
4116-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1785:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4117:.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1819:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4118-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1946:    21	#   numbers : s8-lloyd0 np8/16/32 steady_median_ms
4119-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1969:    44	  echo "=== s8 lloyd0 np=$NP f32 (matched to s9 ladder protocol) ==="
--
4123-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2550:+  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
4124-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2565:+2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4125:.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2599:+| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4126-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2647:+        note="s8 production mesh;\ns9 lloyd-0 synthetic",
4127-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:2698:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-13-# receipt is the generator's default PRODUCTION-Lloyd mesh, while the
--
4157-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6273:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:46:      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
4158-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6309:18-2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
4159:.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6463:6465:/usr/bin/bash -lc "python -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' && printf '%s\\n' '--- stored MPAS receipts ---' && rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200 && printf '%s\\n' '--- campaign figure handling of cell count / plot source ---' && rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
4160:.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6468:6470:/usr/bin/bash -lc "/usr/bin/python3 -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' ; printf '%s\\n' '--- stored MPAS receipts ---'; rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200; printf '%s\\n' '--- campaign figure handling of cell count / plot source ---'; rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
4161-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6479:6512-./mpasoc_2x2b.26508336.log:2128:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12748.4ms fused=313.165ms/step (probe_latency=308.818ms) gate_loop_latency=312.67ms/step
4162-.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:6488:6521-./mpasoc_2x2b.26508336.log:4492:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13919.2ms fused=617.6562ms/step (probe_latency=585.002ms) gate_loop_latency=589.96ms/step
--
4199-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:948:1785:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4200-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:998:6309-18-2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
4201:.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1544:1724-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4202-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1660:  1671	  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
4203-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1675:  1686	2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4204:.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:1713:  1724	| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4205-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:2299:    69	        note="s8 production mesh;\ns9 lloyd-0 synthetic",
4206-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3106:1686-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
--
4210-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3662:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-9-#SBATCH --mem=0
4211-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:3663:scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-10-#SBATCH --time=01:30:00
4212:.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4330:1724-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4213:.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4493:rg --files -uu | rg '(26600094|26600095|26534060|26502539|26628072|26628076)' || true
4214:.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4504:mpas_s8_l0.26628076.log
4215-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4608:2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
4216-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4620:4388:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
4217-.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md:4627:6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
4218-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:18:2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
4219:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6465:/usr/bin/bash -lc "python -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' && printf '%s\\n' '--- stored MPAS receipts ---' && rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200 && printf '%s\\n' '--- campaign figure handling of cell count / plot source ---' && rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
4220:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6470:/usr/bin/bash -lc "/usr/bin/python3 -c 'from decimal import Decimal as D; A,B,C,DD,E=map(D,[\"189.82\",\"308.96\",\"191.99\",\"333.39\",\"537.91\"]); print(\"B/A\",B/A,\"D/C\",DD/C,\"D/B\",DD/B,\"E/D\",E/DD,\"E/B\",E/B); s8=[D(\"6.92\"),D(\"7.10\"),D(\"8.13\")]; s9=[D(\"12.47\"),D(\"9.60\"),D(\"11.48\")]; print(\"weak efficiency\",*[x/y for x,y in zip(s8,s9)]); print(\"weak time ratio\",*[y/x for x,y in zip(s8,s9)]); print(\"32->64 speedup\",s9[0]/s9[1],\"eff\",s9[0]/s9[1]/2,\"64->128 speedup\",s9[1]/s9[2],\"eff\",s9[1]/s9[2]/2); cells=D(2621442); lev=D(26); print(\"s9 GC/s\",*[cells*lev/(t/D(1000))/D(10)**9 for t in s9]); print(\"s8 strict cells GC/s\",D(655362)*lev/(D(\"5.27\")/D(1000))/D(10)**9); print(\"s8 padded cells GC/s\",D(655376)*lev/(D(\"5.27\")/D(1000))/D(10)**9); solo=D(\"5.47\"); single128=D(\"5.94\"); print(\"all healthy 4xsolo\",4*solo,\"threshold\",4*solo/D(\"1.10\"),\"threshold vs s9-128\",4*solo/D(\"1.10\")/single128,\"all vs 128\",4*solo/single128)' ; printf '%s\\n' '--- stored MPAS receipts ---'; rg -n -C 3 '\"n_cells\"\\s*:\\s*655|\"nCells\"\\s*:\\s*655|655376|655362|\"steady_median_ms\"\\s*:\\s*5\\.27|\"steady_median_ms\"\\s*:\\s*9\\.60|26600095|26600094' . --glob '*.jsonl' --glob '*.log' --glob '*.out' --glob '*.md' --glob '*.csv' --glob '*.txt' 2>/dev/null | head -n 1200; printf '%s\\n' '--- campaign figure handling of cell count / plot source ---'; rg -n -C 4 '655_?3(62|76)|GC/s|mcells_per_s|gcell|throughput' docs/performance/scaling scripts/plot/plot_scaling_paper_figure.py | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
4221-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6485:./mpas_bar.26480310.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=4574.5ms steady_median=22.03ms/step (per-step: [4574.5, 22.4, 22.2, 22.2, 22.2, 22.2, 22.2, 22.0, 22.0, 22.0, 22.0, 22.0])
4222-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6494:./mpas_bar.26480310.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=3717.8ms steady_median=14.09ms/step (per-step: [3717.8, 15.1, 14.3, 14.2, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1])
--
4238-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6635:./mpas_fuse.26480162.log:12:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5257.7ms steady_median=17.27ms/step (per-step: [5257.7, 17.6, 17.5, 17.4, 17.3, 17.2, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3])
4239-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6644:./mpas_fuse.26480162.log:89:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5272.1ms steady_median=17.30ms/step (per-step: [5272.1, 17.6, 17.5, 17.3, 17.3, 17.4, 17.3, 17.3, 17.2, 17.3, 17.1, 17.2])
4240:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6652:./mpas128.26538474.log:131:{"component": "mpas_atm", "subdivision": 8, "n_devices": 64, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 5605.2, "steady_median_ms": 5.27, "steady_min_ms": 5.15, "per_step_ms": [5605.2, 8.5, 5.9, 6.2, 5.5, 5.4, 5.3, 5.2, 5.3, 5.2, 5.1, 5.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 5.267729051411152, "total_cells": 17042688, "sypd": 15.59220734634407, "mcells_per_s": 3235.3007973017325, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:06:38.803951+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 266292, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 266292}}}
4241-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6653:./mpas128.26538474.log-132-[mpas nd=64 L8 nCells=655488 nlev=26] compile=5605.2ms steady_median=5.27ms/step (per-step: [5605.2, 8.5, 5.9, 6.2, 5.5, 5.4, 5.3, 5.2, 5.3, 5.2, 5.1, 5.2])
4242:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6660:./mpas128.26538474.log:420:{"component": "mpas_atm", "subdivision": 8, "n_devices": 128, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7701.1, "steady_median_ms": 6.47, "steady_min_ms": 6.21, "per_step_ms": [7701.1, 11.9, 8.6, 6.5, 6.6, 6.3, 6.9, 7.1, 6.3, 6.2, 6.8, 6.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 6.468050996772945, "total_cells": 17042688, "sypd": 12.698651209605844, "mcells_per_s": 2634.903158386194, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:14:39.861856+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 133146, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 133146}}}
4243-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6661:./mpas128.26538474.log-421-[mpas nd=128 L8 nCells=655488 nlev=26] compile=7701.1ms steady_median=6.47ms/step (per-step: [7701.1, 11.9, 8.6, 6.5, 6.6, 6.3, 6.9, 7.1, 6.3, 6.2, 6.8, 6.2])
4244:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6668:./mpas128.26538474.log:750:{"component": "mpas_atm", "subdivision": 8, "n_devices": 64, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 5597.0, "steady_median_ms": 8.96, "steady_min_ms": 8.84, "per_step_ms": [5597.0, 12.2, 9.9, 9.6, 9.0, 9.0, 9.0, 9.0, 9.0, 8.9, 8.9, 8.8], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 8.959555067121983, "total_cells": 17042688, "sypd": 9.167366347841075, "mcells_per_s": 1902.180172153851, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:18:32.347778+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 266292, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "9745e4da6", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 266292}}}
4245-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6669:./mpas128.26538474.log-751-[mpas nd=64 L8 nCells=655488 nlev=26] compile=5597.0ms steady_median=8.96ms/step (per-step: [5597.0, 12.2, 9.9, 9.6, 9.0, 9.0, 9.0, 9.0, 9.0, 8.9, 8.9, 8.8])
4246:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6676:./mpas128.26538474.log:945:{"component": "mpas_atm", "subdivision": 8, "n_devices": 128, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 6265.2, "steady_median_ms": 11.41, "steady_min_ms": 11.28, "per_step_ms": [6265.2, 18.0, 13.8, 12.2, 11.6, 11.4, 11.8, 11.4, 12.2, 11.3, 11.4, 11.3], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.414309963583946, "total_cells": 17042688, "sypd": 7.195837845301824, "mcells_per_s": 1493.0984049296674, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:26:25.491528+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 133146, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "9745e4da6", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 133146}}}
4247-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6677:./mpas128.26538474.log-946-[mpas nd=128 L8 nCells=655488 nlev=26] compile=6265.2ms steady_median=11.41ms/step (per-step: [6265.2, 18.0, 13.8, 12.2, 11.6, 11.4, 11.8, 11.4, 12.2, 11.3, 11.4, 11.3])
4248-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6685:./mpasoc_512.26508063.log:14184:[mpas-ocean np=512 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14009.9ms fused=194.8ms/step (probe_latency=175.611ms) gate_loop_latency=179.54ms/step
--
4285-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7091:./atm_ladders.26454476.log:114:[mpas nd=16 L8 nCells=655376 nlev=26] compile=6542.5ms steady_median=7.10ms/step (per-step: [6542.5, 10.4, 7.9, 7.4, 7.3, 7.3, 7.3, 7.2, 7.2, 7.6, 7.2, 7.2, 7.2, 7.2, 7.1, 7.1, 7.0, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0, 7.1, 7.0, 7.2, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])
4286-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7100:./mpas_bar.26480261.log:35:[mpas nd=2 L8 nCells=655376 nlev=26] compile=6578.8ms steady_median=21.40ms/step (per-step: [6578.8, 22.0, 21.5, 21.4, 21.4, 21.4, 21.4, 21.4, 21.2, 21.1, 21.2, 21.1])
4287:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7109:./mpas_bar.26480261.log:52:[mpas nd=4 L8 nCells=655376 nlev=26] compile=4134.1ms steady_median=16.58ms/step (per-step: [4134.1, 16.9, 16.6, 16.6, 16.6, 16.6, 16.6, 16.5, 16.5, 16.5, 16.6, 16.6])
4288-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7118:./mpas_bar.26480261.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=5041.0ms steady_median=7.01ms/step (per-step: [5041.0, 8.2, 8.1, 7.1, 7.1, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 6.9])
4289-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7127:./mpas_part_ab.26455829.log:12:[mpas nd=4 L8 nCells=655376 nlev=26] compile=5061.7ms steady_median=17.22ms/step (per-step: [5061.7, 18.1, 17.9, 18.0, 17.8, 18.0, 17.8, 18.0, 17.8, 18.2, 17.8, 17.8, 17.7, 17.9, 17.6, 17.6, 17.3, 17.0, 17.0, 17.0, 17.2, 17.0, 17.0, 17.2, 17.1, 17.1, 17.0, 17.1, 17.2, 17.0, 17.2, 17.0, 17.2])
--
4297-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7199:./mpas_bar.26486163.log:83:[mpas nd=8 L8 nCells=655376 nlev=26] compile=6144.4ms steady_median=7.02ms/step (per-step: [6144.4, 7.5, 7.6, 7.2, 7.1, 7.0, 7.0, 7.0, 7.0, 7.0, 6.9, 7.0])
4298-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7208:./legoesm_gpu_multinode.26453240.log:2422:[mpas nd=6 L8 nCells=655362 nlev=26] compile=5732.1ms steady_median=8.66ms/step (per-step: [5732.1, 9.1, 8.9, 8.8, 8.7, 8.7, 8.7, 8.7, 8.6, 8.7, 8.6, 8.7, 8.7, 8.6, 9.1, 8.6, 9.2, 9.0, 8.6, 9.1, 8.6, 9.0, 8.5, 8.9, 8.5, 8.9, 9.0, 9.0, 8.4, 8.4, 8.5, 8.4, 8.4])
4299:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:9998:131:{"component": "mpas_atm", "subdivision": 8, "n_devices": 64, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 5605.2, "steady_median_ms": 5.27, "steady_min_ms": 5.15, "per_step_ms": [5605.2, 8.5, 5.9, 6.2, 5.5, 5.4, 5.3, 5.2, 5.3, 5.2, 5.1, 5.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 5.267729051411152, "total_cells": 17042688, "sypd": 15.59220734634407, "mcells_per_s": 3235.3007973017325, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:06:38.803951+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 266292, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 266292}}}
4300:.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:9999:420:{"component": "mpas_atm", "subdivision": 8, "n_devices": 128, "n_cells": 655488, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7701.1, "steady_median_ms": 6.47, "steady_min_ms": 6.21, "per_step_ms": [7701.1, 11.9, 8.6, 6.5, 6.6, 6.3, 6.9, 7.1, 6.3, 6.2, 6.8, 6.2], "cells": 17042688, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 6.468050996772945, "total_cells": 17042688, "sypd": 12.698651209605844, "mcells_per_s": 2634.903158386194, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:14:39.861856+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 133146, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26538474", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 133146}}}
4301-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:10381:/usr/bin/bash -lc "printf '%s\\n' '--- s9 32 full exact record ---'; rg -n '"'^'"\\{\"component\": \"mpas_atm\"' mpas_s9.26600095.log | head -n 1; printf '%s\\n' '--- s8 weak-pair receipt candidates ---'; rg -n -l 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | sort; rg -n -C 0 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | head -n 100; printf '%s\\n' '--- lloyd state in old benchmark commit ---'; git show 10533f1f9:scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,220p'; printf '%s\\n' '--- phase job log start and launch flags ---'; sed -n '1,180p' mpas_s9.26600095.log; printf '%s\\n' '--- METIS job command echo / distribution provenance ---'; rg -n -C 3 -i 'distribution|block:|srun|"'^===|command|geometric|metis'"' mpasoc_metis.26600094.log | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
4302-.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:10883:2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
--
4317-docs/performance/scaling/levante_campaign_2026-07-24.md:1671:  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
4318-docs/performance/scaling/levante_campaign_2026-07-24.md:1690:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
4319:docs/performance/scaling/levante_campaign_2026-07-24.md:1728:| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
4320:docs/performance/scaling/levante_campaign_2026-07-24.md:1758:### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
4321-docs/performance/scaling/levante_campaign_2026-07-24.md:1760:s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
4322-docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md:173:`np_seos` with the Kamm 2025 DINO coefficients (a0=0.165, b0=0.76554, λ1=0.06,
4323-scripts/validate/ec_site_regression_golden.json:235:      "beta_eff": 0.36655469119670014
4324:scripts/plot/plot_scaling_paper_figure.py:39:                "s8-lloyd0 26628076",
4325:scripts/plot/plot_scaling_paper_figure.py:69:                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
4326-scripts/plot/plot_scaling_paper_figure.py:118:          "f32 (s8 lloyd-0)": "#009E73",
4327-scripts/plot/plot_scaling_paper_figure.py:127:           "f32 (s8 lloyd-0)": "v",
4328-
4329-exec
4330:/usr/bin/bash -lc "rg -n -C 2 \"26628076|mpas_s8_l0|6\\.58|6\\.43|7\\.29|7151d12a1-dirty|n_cells.*655\" .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md | tail -300" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign

exec
/usr/bin/bash -lc "sed -n '10375,10560p' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md && sed -n '11235,11370p' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md && sed -n '12900,13030p' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 208ms:
2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}

exec
/usr/bin/bash -lc "printf '%s\\n' '--- s9 32 full exact record ---'; rg -n '"'^'"\\{\"component\": \"mpas_atm\"' mpas_s9.26600095.log | head -n 1; printf '%s\\n' '--- s8 weak-pair receipt candidates ---'; rg -n -l 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | sort; rg -n -C 0 'steady_median_ms\": (6\\.92|7\\.10|8\\.13)' --glob 'mpas*.log' . | head -n 100; printf '%s\\n' '--- lloyd state in old benchmark commit ---'; git show 10533f1f9:scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,220p'; printf '%s\\n' '--- phase job log start and launch flags ---'; sed -n '1,180p' mpas_s9.26600095.log; printf '%s\\n' '--- METIS job command echo / distribution provenance ---'; rg -n -C 3 -i 'distribution|block:|srun|"'^===|command|geometric|metis'"' mpasoc_metis.26600094.log | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 328ms:
--- s9 32 full exact record ---
67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
--- s8 weak-pair receipt candidates ---
./mpas32.26549646.log
./mpas32.26549646.log:67:{"component": "mpas_atm", "subdivision": 8, "n_devices": 32, "n_cells": 655392, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 6060.7, "steady_median_ms": 8.13, "steady_min_ms": 8.01, "per_step_ms": [6060.7, 11.4, 8.6, 8.3, 8.3, 8.2, 8.1, 8.1, 8.0, 8.0, 8.8, 8.0], "cells": 17040192, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 8.125972002744675, "total_cells": 17040192, "sypd": 10.107778317008782, "mcells_per_s": 2097.0035331458694, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:29:33.554535+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50112.lvt.dkrz.de", "slurm_job_id": "26549646", "git_sha": "10533f1f9", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
--- lloyd state in old benchmark commit ---
}
MPAS_PARITY_MAX_STEPS = 8

# Conservation gate default: with fix_mass=True the step restores the global
# dry mass to the pre-step value each step, so the drift over a smoke window
# is the allreduce rounding floor, not scheme drift.
MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}


def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
                          moist=False):
    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.

    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
    device count of THIS run's mesh/model (the two differ for the
    single-device reference leg of a ladder, via ``--reorder-for``).
    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
    wave) so the sharded step's packed tracer halo exchange + RK tracer
    advection sit on the timed/gated path.
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.mesh import create_voronoi_device_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
        # Padding only guarantees divisibility for reorder_target.
        raise SystemExit(
            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
            f"divisible by --n-devices {run_nd}; use a ladder where every "
            f"count divides --reorder-for ({reorder_target}).")
    sigma = create_sigma_coordinate(nlev)
    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
    # energy-conserving PV flux, SSP-RK3, global mass fixer.
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=run_nd,
    )
    if dev_config.n_devices > 1:
        from legoesm.parallel.mesh import replicate_pytree
        mesh_model = replicate_pytree(mesh, dev_config)
    else:
        mesh_model = mesh
    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
    # #1100 MPAS twin: the timed path never global-builds the state.
    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
    # jax.make_array_from_callback (only THIS process's shard rows are
    # ever materialised; value-identical (few-ULP contract, measured
    # exact on the pinned CPU stack) to global-build + shard_pytree —
    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
    # state is built lazily in main() only for the parity/conservation
    # gates (small smoke scales).  The mesh itself is still global per
    # process — its SFC-partition-local construction is the open
    # remainder of #1100.
    from tests.test_cases.baroclinic_wave import (
        build_sharded_baroclinic_wave_state_mpas,
    )
    state_sharded = build_sharded_baroclinic_wave_state_mpas(
        mesh, sigma, dev_config, perturbed=True, moist=moist)
    return mesh, model, state_sharded, dev_config


def _block(state):
    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
                           if leaf is not None])


def _global_dry_mass(state, mesh):
    """sum(p_s * areaCell) on host arrays — the quantity fix_mass pins."""
    ps = np.asarray(state.p_s.data)
    area = np.asarray(mesh.areaCell)
    return float(np.sum(ps * area))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subdivision", type=int, default=5,
                   help="icosahedral subdivision level L "
                        "(nCells = 10*4^L + 2 before ghost padding)")
    p.add_argument("--nlev", type=int, default=8)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--reorder-for", type=int, default=None,
                   help="partition/reorder the mesh for THIS device count "
                        "(default: --n-devices). Pin it to the ladder's "
                        "max so single-device reference runs time the "
                        "identical reordered mesh.")
    p.add_argument("--partition-method",
                   choices=["auto", "geometric", "metis", "sfc"],
                   default="auto")
    p.add_argument("--physics", choices=["none", "held_suarez", "kessler"],
                   default="none",
                   help="Operator-split physics on the timed path. "
                        "'kessler' also attaches the q_v/q_c/q_r moist-"
                        "baroclinic-wave tracers (packed tracer halo "
                        "exchange + RK tracer advection on the gated "
                        "path) and extends the parity gate to the "
                        "tracer fields.")
    p.add_argument("--halo-strategy",
                   choices=["auto", "ppermute", "allgather"],
                   default="auto",
                   help="Halo strategy for make_voronoi_sharded_step. "
                        "'auto' picks allgather below the per-device "
                        "cell threshold — force 'ppermute' to exercise "
                        "the neighbor-round schedule on small gate "
                        "meshes (the multicontroller selfspawn tests "
                        "do).  Recorded in the JSONL row.")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dt", type=float, default=None,
                   help="timestep [s]; default auto: 600 * 4**(4-L) "
                        "(CFL: dx halves per level), min 30 s.")
    p.add_argument("--out", type=str,
                   default="results/a1/mpas_spmd_scaling.jsonl")
    p.add_argument(
        "--parity-gate", action="store_true",
        help="Correctness gate: compare the gathered sharded trajectory "
             "against the single-device model.step trajectory on the SAME "
             "reordered mesh (smoke windows only; the re-association floor "
             "grows with steps).")
--- phase job log start and launch flags ---
outdir=/scratch/b/b381103/legoesm_scaling/mpas_s9_j26600095
=== mpas ico subdiv-9 np=32 f32 (padded-128, lloyd=0 scaling mesh) ===
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
--
6305-*                       We hope you enjoyed the DKRZ supercomputer LEVANTE ... *
6306-*
6307-* JobID            : 26600094
6308:* JobName          : mpasoc_metis                                      
6309-* Account          : bb1596
6310-* User             : b381103 (200166), bd1083 (1468)                   
6311-* Partition        : compute
--
6315-* Start time       : 2026-07-31T19:17:48
6316-* End time         : 2026-07-31T19:23:16
6317-* Elapsed time     : 00:05:28 (Timelimit=03:00:00)                     
6318:* Command          : /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/
6319:*                    worktrees/scaling-campaign/scripts/tmp/mpasoc_metis_placement.sbatch
6320-* WorkDir          : /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/
6321-*                    worktrees/scaling-campaign
6322-*

codex
The receipts expose two substantive confounds beyond wording: METIS does not hold every rank at 5,120 cells (it spans 5,093–5,144 at 128 ranks), and the s8 comparison rows predate the `--lloyd` control and therefore use the default relaxed mesh while s9 explicitly uses `lloyd=0`. Those invalidate the strong causal versions of the lever claims.
exec
/usr/bin/bash -lc "printf '%s\\n' '--- literal weak-pair rows, with provenance ---'; rg -n -F -e '\"steady_median_ms\": 6.92' -e '\"steady_median_ms\": 7.1' -e '\"steady_median_ms\": 8.13' mpas*.log; printf '%s\\n' '--- METIS quality statements / exact values ---'; rg -n -i -C 4 'partq|edge_cut|halo mean|halo_mean|1\\.89|2\\.01|592|629|cells_per_rank_(min|max)' mpasoc_metis.26600094.log | head -n 1400; printf '%s\\n' '--- test/warning and result tail ---'; rg -n -C 2 'RuntimeWarning: Detected versions outside|outside legoESM|stage_halo_correct|parity_gate|check_conservation|DONE rc|RESULTS' mpasoc_metis.26600094.log | tail -n 1000; printf '%s\\n' '--- exact current PLOT source table rendering / parser static check ---'; /usr/bin/python3 -c \"import ast; ast.parse(open('scripts/plot/plot_scaling_paper_figure.py').read()); print('AST OK')\"; tail -n 85 scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 418ms:
--- literal weak-pair rows, with provenance ---
mpas32.26549646.log:67:{"component": "mpas_atm", "subdivision": 8, "n_devices": 32, "n_cells": 655392, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 6060.7, "steady_median_ms": 8.13, "steady_min_ms": 8.01, "per_step_ms": [6060.7, 11.4, 8.6, 8.3, 8.3, 8.2, 8.1, 8.1, 8.0, 8.0, 8.8, 8.0], "cells": 17040192, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 8.125972002744675, "total_cells": 17040192, "sypd": 10.107778317008782, "mcells_per_s": 2097.0035331458694, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-30T00:29:33.554535+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50112.lvt.dkrz.de", "slurm_job_id": "26549646", "git_sha": "10533f1f9", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
mpas_nsys.26479922.log:174:{"component": "mpas_atm", "subdivision": 8, "n_devices": 8, "n_cells": 655376, "n_edges": 1966080, "nlev": 26, "partition_method": "sfc", "physics": "none", "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 8, "multicontroller": true, "compile_ms": 6621.3, "steady_median_ms": 7.17, "steady_min_ms": 7.06, "per_step_ms": [6621.3, 8.9, 8.4, 7.2, 7.2, 7.1, 7.1, 7.1, 11.7, 9.0, 10.9, 9.7], "cells": 17039776, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 8, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 7.166521972976625, "total_cells": 17039776, "sypd": 11.461002132370208, "mcells_per_s": 2377.6911679407726, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-26T10:40:28.596045+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L8", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 8, "n_gpus": 8, "device_count": 8, "process_count": 8, "devices_per_rank": 1, "cells_per_rank": 2129972, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50115.lvt.dkrz.de", "slurm_job_id": "26479922", "git_sha": "5fcfcb691", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2129972}}}
--- METIS quality statements / exact values ---
446-[1785518297.254555] [l20119:1439087:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
447-[1785518297.281831] [l20119:1439072:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
448-[1785518297.249801] [l20119:1439095:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
449-[1785518297.254104] [l20119:1439079:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
451-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12255.1ms fused=189.8206ms/step (probe_latency=187.99ms) gate_loop_latency=187.13ms/step
452-[1785518297.289277] [l20119:1439070:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
453---- s8 np=128 partition=geometric dist=block:cyclic ---
454-[l20131.lvt.dkrz.de:270332] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
2145-[1785518353.342904] [l20119:1440313:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2146-[1785518353.389333] [l20119:1440311:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2147-[1785518353.342035] [l20119:1440310:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2148-[1785518353.342454] [l20119:1440297:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
2151-[1785518353.372656] [l20119:1440282:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2152-[1785518357.149452] [l20130:50192:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2153-[1785518357.157028] [l20130:50198:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
2172-[1785518357.156306] [l20130:50176:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2173-[1785518357.152334] [l20130:50195:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2174-[1785518357.148830] [l20130:50189:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2175-[1785518357.154931] [l20130:50174:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2176:[1785518357.145929] [l20130:50194:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2177-[1785518357.128706] [l20130:50187:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2178-[1785518357.152538] [l20130:50199:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2179-[1785518357.154331] [l20130:50201:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2180-[1785518357.140150] [l20130:50178:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
2683-[1785518411.094262] [l20119:1441493:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2684-[1785518411.087402] [l20119:1441479:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2685-[1785518411.080442] [l20119:1441486:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2686-[1785518411.064649] [l20119:1441490:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
2688-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12579.2ms fused=191.9856ms/step (probe_latency=188.993ms) gate_loop_latency=190.29ms/step
2689-[1785518411.064180] [l20119:1441477:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2690-[1785518411.073813] [l20119:1441504:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2691-[1785518411.088174] [l20119:1441503:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
4383-[1785518454.796638] [l20119:1442701:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4384-[1785518454.816638] [l20119:1442698:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4385-[1785518454.766060] [l20119:1442707:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4386-[1785518454.774753] [l20119:1442679:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
4388-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
4389-[1785518454.797213] [l20119:1442677:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4390-[1785518454.772658] [l20119:1442708:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4391-[1785518454.799331] [l20130:51417:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
4573-[l20147.lvt.dkrz.de:3470839] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4574-[l20130.lvt.dkrz.de:52587] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4575-[l20147.lvt.dkrz.de:3470832] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4576-[l20131.lvt.dkrz.de:272755] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4577:[l20130.lvt.dkrz.de:52592] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4578-[l20130.lvt.dkrz.de:52590] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4579-[l20147.lvt.dkrz.de:3470831] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4580-[l20119.lvt.dkrz.de:1443909] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4581-[l20131.lvt.dkrz.de:272754] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
6229-[1785518522.187344] [l20131:272751:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6230-[1785518522.118272] [l20131:272755:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6231-[1785518522.091145] [l20131:272748:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6232-[1785518522.232866] [l20131:272757:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6233:[1785518522.013632] [l20130:52596:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6234-[1785518522.163639] [l20130:52599:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6235-[1785518521.757802] [l20119:1443890:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6236-[1785518521.841632] [l20119:1443896:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6237-[1785518521.789676] [l20119:1443897:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6238:[1785518522.019943] [l20130:52600:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6239-[1785518522.163771] [l20130:52606:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6240-[1785518522.160026] [l20130:52595:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6241-[1785518521.803223] [l20119:1443882:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6242-[1785518521.823188] [l20130:52609:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
6251-[1785518522.085039] [l20130:52608:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6252-[1785518521.847542] [l20119:1443892:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6253-[1785518521.996105] [l20130:52583:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6254-[1785518521.884325] [l20119:1443901:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
6257-[1785518521.623596] [l20119:1443880:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6258-[1785518522.112727] [l20130:52580:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6259-[1785518521.530617] [l20119:1443887:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6260:[1785518522.112729] [l20130:52592:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6261-[1785518521.920271] [l20130:52582:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6262:[1785518522.016436] [l20130:52589:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6263-[1785518522.193227] [l20130:52593:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6264-[1785518521.776949] [l20119:1443886:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6265-[1785518521.542688] [l20119:1443888:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6266-[1785518521.924018] [l20130:52594:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
6269-[1785518521.797135] [l20119:1443904:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6270-[1785518521.749343] [l20119:1443881:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6271-[1785518522.044599] [l20130:52585:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6272-[1785518522.126415] [l20130:52587:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6273:[1785518521.629059] [l20119:1443883:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6274-[1785518521.829217] [l20119:1443911:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6275-[1785518522.006923] [l20130:52591:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6276-[1785518521.744555] [l20119:1443895:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6277-[1785518521.842315] [l20119:1443900:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--- test/warning and result tail ---
5178-  mpi4jax, MPI = require_mpi_stack()
5179:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:549: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5180-  mpi4jax, MPI = require_mpi_stack()
5181:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:549: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5182-  mpi4jax, MPI = require_mpi_stack()
5183:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:549: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5184-  mpi4jax, MPI = require_mpi_stack()
--- venv and installed JAX candidates ---
ls: cannot access '.venv': No such file or directory
drwxrws---+ 5 200166 1468 4096 Jul 22 21:24 /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/__pycache__/distributed.cpython-314.pyc
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/clusters/cluster.py
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/__pycache__/distributed.cpython-314.pyc
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/distributed.py
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/distributed.py
--- metadata source path ---
/usr/bin/bash: .venv/bin/python: No such file or directory
0.10.0
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax
--- all local site-packages JAX source hits ---
/work/bd1083/b309178/diffESM/legoesm_ggn/.venv/lib/python3.12/site-packages/jax
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/lib/python3.13/site-packages/jax
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax
/work/bd1083/b309178/mambaforge/pkgs/jax-0.9.0-pyhd8ed1ab_0/site-packages/jax
/work/bd1083/b309178/mambaforge/envs/diffesm/lib/python3.12/site-packages/jax
/work/bd1083/b309178/mambaforge/envs/jax-gcm-icon/lib/python3.11/site-packages/jax

codex
One correction to my earlier scheduler concern: each step requests all four GPUs on each of eight nodes with `--exact`, so Slurm should allocate disjoint node sets. The remaining flaw is evidentiary and causal: no logged node sets or synchronized timed interval, and the single solo is not paired to each replica’s topology.
codex
Arithmetic checks:

| Check | Result |
|---|---:|
| geometric B/A | 1.62765 → **1.628** |
| METIS D/C | 1.73650 → **1.736** |
| METIS D vs B | 1.07907 → **+7.91%** |
| placement E/D | 1.61346 → **1.61×** |
| s9 32→64 speedup / efficiency | 1.29896 / **0.6495** |
| s9 64→128 speedup | **0.8362** (anti-scaling) |
| weak efficiencies | **0.5549, 0.7396, 0.7082** |
| weak time costs | **1.802×, 1.352×, 1.412×** |
| s9 GC/s | **5.466, 7.099, 5.936** → 5.47/7.10/5.94 |
| s8 np64 GC/s | 3.233 using natural cells; 3.235 using executed padded cells |
| s9/s8 peak ratio | **2.19×** → 2.2× |

1. The METIS matrix does not hold every arm at exactly 5,120 cells/rank. The receipts show geometric 5,120–5,121, but METIS is 5,100–5,145 at 32 ranks and 5,093–5,144 at 128. The wording at [the Phase-3 table](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/docs/performance/scaling/levante_campaign_2026-07-24.md:1594) is false; call it a nominal 5,120-cell target and report min/max.

2. “Partition-quality lever DEAD/refuted” overclaims. This is evidence that this METIS configuration loses to geometric at L8/128 despite lower cut/halo metrics; it does not eliminate partition quality or mapping generally. The 32→128 terms are weak-scale, 1→4-node scale-out terms, not isolated rank-count terms.

3. The placement mechanism is misdescribed. Both `block:cyclic` and `block:block` use `block` for rank-to-node placement; the second field changes CPU allocation across sockets. E/D does support “`block:cyclic` is 1.61× faster in this receipt,” but “per-socket bandwidth beats communication locality” is **PLAUSIBLE**, not confirmed.

4. The METIS receipt ran without parity or conservation gates and logs an unsupported MPI/JAX pairing warning. That does not erase the timing observation, but blocks a production-quality “lever closed” claim until the stack is supported and at least one gated arm per partition passes.

5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.

6. “Consistent ~1.4×” is arithmetically false: the three costs are 1.80×, 1.35×, and 1.41×. The first is a large outlier. The proposed GPU rank-count term is therefore **PLAUSIBLE at best**, even after the Lloyd confound is fixed.

7. The reported mesh counts are natural, not executed padded counts. The s9 receipt executed 2,621,568 cells, not 2,621,442; its np32 provenance SHA is also `unknown`. Throughput rounding remains correct, but report executed `n_cells` and resolve that provenance hole. “New MPAS-atmosphere peak” should be scoped to the synthetic, `physics=none` benchmark.

8. “Floor confirmed” should be **PLAUSIBLE/consistent with the floor**, not CONFIRMED: one below-floor point anti-scales, with no replication and known shape/codegen variation.

9. The ensemble script masks failures: each background `run_arm` changes `rc` only in its subshell, and `srun ... || { rc=1; }` returns success. Missing/corrupt JSON is silently skipped, so the job can print `DONE rc=0`. [The script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/mpas_s9_ensemble.sbatch:44) must preserve each background PID/status and require all five valid receipts. It is also ignored by Git, so it is not reproducibly reviewable.

10. GPU binding is otherwise sound: `--gpu-bind=none` leaves all GPUs visible and JAX’s Slurm auto-detection selects by local rank. Full-GPU `--exact` requests should force disjoint eight-node steps. But log each `SLURM_STEP_NODELIST` and coordinator host; that is the evidence the shared JAX job-ID port is safe.

11. The ensemble does not synchronize measurement windows, pair each replica topology with a solo control, or counterbalance solo-first ordering. It can measure different node/fabric groups or non-overlapping windows rather than contention. Twelve steps give nine dependent steady samples per launch, not enough independent evidence for a 10% confirmation bar.

12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.

VERDICT: FIX-FIRST
tokens used
232,430
Arithmetic checks:

| Check | Result |
|---|---:|
| geometric B/A | 1.62765 → **1.628** |
| METIS D/C | 1.73650 → **1.736** |
| METIS D vs B | 1.07907 → **+7.91%** |
| placement E/D | 1.61346 → **1.61×** |
| s9 32→64 speedup / efficiency | 1.29896 / **0.6495** |
| s9 64→128 speedup | **0.8362** (anti-scaling) |
| weak efficiencies | **0.5549, 0.7396, 0.7082** |
| weak time costs | **1.802×, 1.352×, 1.412×** |
| s9 GC/s | **5.466, 7.099, 5.936** → 5.47/7.10/5.94 |
| s8 np64 GC/s | 3.233 using natural cells; 3.235 using executed padded cells |
| s9/s8 peak ratio | **2.19×** → 2.2× |

1. The METIS matrix does not hold every arm at exactly 5,120 cells/rank. The receipts show geometric 5,120–5,121, but METIS is 5,100–5,145 at 32 ranks and 5,093–5,144 at 128. The wording at [the Phase-3 table](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/docs/performance/scaling/levante_campaign_2026-07-24.md:1594) is false; call it a nominal 5,120-cell target and report min/max.

2. “Partition-quality lever DEAD/refuted” overclaims. This is evidence that this METIS configuration loses to geometric at L8/128 despite lower cut/halo metrics; it does not eliminate partition quality or mapping generally. The 32→128 terms are weak-scale, 1→4-node scale-out terms, not isolated rank-count terms.

3. The placement mechanism is misdescribed. Both `block:cyclic` and `block:block` use `block` for rank-to-node placement; the second field changes CPU allocation across sockets. E/D does support “`block:cyclic` is 1.61× faster in this receipt,” but “per-socket bandwidth beats communication locality” is **PLAUSIBLE**, not confirmed.

4. The METIS receipt ran without parity or conservation gates and logs an unsupported MPI/JAX pairing warning. That does not erase the timing observation, but blocks a production-quality “lever closed” claim until the stack is supported and at least one gated arm per partition passes.

5. The s8→s9 weak-pair claim is confounded. s9 explicitly used `lloyd=0`; the s8 source predates the `--lloyd` option and therefore used the generator’s default production relaxation. They are not the “same protocol.” The first two comparator points are also from the earlier np2–16 ladder, not the claimed np32–128 extension rows. Remove the causal GPU rank-count claim and the plot annotation until s8 is rerun with `lloyd=0` and matched reorder policy.

6. “Consistent ~1.4×” is arithmetically false: the three costs are 1.80×, 1.35×, and 1.41×. The first is a large outlier. The proposed GPU rank-count term is therefore **PLAUSIBLE at best**, even after the Lloyd confound is fixed.

7. The reported mesh counts are natural, not executed padded counts. The s9 receipt executed 2,621,568 cells, not 2,621,442; its np32 provenance SHA is also `unknown`. Throughput rounding remains correct, but report executed `n_cells` and resolve that provenance hole. “New MPAS-atmosphere peak” should be scoped to the synthetic, `physics=none` benchmark.

8. “Floor confirmed” should be **PLAUSIBLE/consistent with the floor**, not CONFIRMED: one below-floor point anti-scales, with no replication and known shape/codegen variation.

9. The ensemble script masks failures: each background `run_arm` changes `rc` only in its subshell, and `srun ... || { rc=1; }` returns success. Missing/corrupt JSON is silently skipped, so the job can print `DONE rc=0`. [The script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/mpas_s9_ensemble.sbatch:44) must preserve each background PID/status and require all five valid receipts. It is also ignored by Git, so it is not reproducibly reviewable.

10. GPU binding is otherwise sound: `--gpu-bind=none` leaves all GPUs visible and JAX’s Slurm auto-detection selects by local rank. Full-GPU `--exact` requests should force disjoint eight-node steps. But log each `SLURM_STEP_NODELIST` and coordinator host; that is the evidence the shared JAX job-ID port is safe.

11. The ensemble does not synchronize measurement windows, pair each replica topology with a solo control, or counterbalance solo-first ordering. It can measure different node/fabric groups or non-overlapping windows rather than contention. Twelve steps give nine dependent steady samples per launch, not enough independent evidence for a 10% confirmation bar.

12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.

VERDICT: FIX-FIRST

exec
/usr/bin/bash -lc "git show -s --format='%H%n%P%n%ci%n%s' 7151d12a1 && git diff --name-status 7151d12a1 -- && git diff --check 7151d12a1 -- && git status --short && git diff --numstat 7151d12a1 -- scripts/bench/bench_mpas_spmd_scaling.py packages/core/legoesm/parallel/voronoi_partition.py scripts/cluster/scaling_levante/_env.sh && git log --all --oneline --decorate --ancestry-path 7151d12a1..HEAD | head -40" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1620ms:
7151d12a184cf7324495b264e8bfa087d6ba4737
00624f397690232a2df6937b121129c301fa538e
2026-07-31 19:12:56 +0200
fix(bench): --lloyd on the MPAS SPMD bench so subdiv>=9 finds its cache
M	docs/performance/scaling/levante_campaign_2026-07-24.md
M	scripts/plot/plot_scaling_paper_figure.py
 M docs/performance/scaling/levante_campaign_2026-07-24.md
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r4_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r5_2026-08-02.md
?? scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
?? scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
?? scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
?? scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
?? scripts/cluster/scaling_levante/prewarm_s10.sbatch

exec
/usr/bin/bash -lc "rg -n -C 1 '\"subdivision\": 9, \"n_devices\": (32|64|128)' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md | head -60 && rg -n -C 2 'mpas_s9\\.26600095\\.log.*(np|n_devices)|\"lloyd_iterations\": 0.*\"git_sha\"' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md | head -120 && git show --stat --oneline 7151d12a1 && git show --format=fuller --no-ext-diff -- scripts/bench/bench_mpas_spmd_scaling.py 7151d12a1 | sed -n '1,220p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 472ms:
6732-./mpas_s9.26600095.log-66-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6733:./mpas_s9.26600095.log:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
6734-./mpas_s9.26600095.log-68-[mpas nd=32 L9 nCells=2621568 nlev=26] compile=7179.2ms steady_median=12.47ms/step (per-step: [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2])
--
6740-./mpas_s9.26600095.log-293-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6741:./mpas_s9.26600095.log:294:{"component": "mpas_atm", "subdivision": 9, "n_devices": 64, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 6474.6, "steady_median_ms": 9.6, "steady_min_ms": 9.55, "per_step_ms": [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 9.601722995284945, "total_cells": 68160768, "sypd": 8.55424840461413, "mcells_per_s": 7098.805915716509, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:31:30.728837+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 1065012}}}
6742-./mpas_s9.26600095.log-295-[mpas nd=64 L9 nCells=2621568 nlev=26] compile=6474.6ms steady_median=9.60ms/step (per-step: [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6])
--
6748-./mpas_s9.26600095.log-928-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6749:./mpas_s9.26600095.log:929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
6750-./mpas_s9.26600095.log-930-[mpas nd=128 L9 nCells=2621568 nlev=26] compile=7532.3ms steady_median=11.48ms/step (per-step: [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4])
--
10305-FILE=mpas_s9.26549775.log
10306:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10307-FILE=mpas_s9.26600095.log
--
10369---- current s9 exact receipt (all JSON rows) ---
10370:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10371:294:{"component": "mpas_atm", "subdivision": 9, "n_devices": 64, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 6474.6, "steady_median_ms": 9.6, "steady_min_ms": 9.55, "per_step_ms": [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 9.601722995284945, "total_cells": 68160768, "sypd": 8.55424840461413, "mcells_per_s": 7098.805915716509, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:31:30.728837+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 1065012}}}
10372:929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
10373---- current metis exact receipt (all JSON rows) ---
--
10383---- s9 32 full exact record ---
10384:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10385---- s8 weak-pair receipt candidates ---
--
10586-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
10587:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10588-[mpas nd=32 L9 nCells=2621568 nlev=26] compile=7179.2ms steady_median=12.47ms/step (per-step: [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2])
--
12805-    66	  return lax_numpy.astype(self, dtype, copy=copy, device=device)
12806:    67	{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
12807-    68	[mpas nd=32 L9 nCells=2621568 nlev=26] compile=7179.2ms steady_median=12.47ms/step (per-step: [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2])
6724---
6725-./mpas_s9.26600095.log:1:outdir=/scratch/b/b381103/legoesm_scaling/mpas_s9_j26600095
6726:./mpas_s9.26600095.log-2-=== mpas ico subdiv-9 np=32 f32 (padded-128, lloyd=0 scaling mesh) ===
6727-./mpas_s9.26600095.log-3-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
6728-./mpas_s9.26600095.log-4-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
--
6731-./mpas_s9.26600095.log-65-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
6732-./mpas_s9.26600095.log-66-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6733:./mpas_s9.26600095.log:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
6734-./mpas_s9.26600095.log-68-[mpas nd=32 L9 nCells=2621568 nlev=26] compile=7179.2ms steady_median=12.47ms/step (per-step: [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2])
6735-./mpas_s9.26600095.log-69-W0801 05:24:48.442624  225571 pjrt_client.cc:1604] WatchTasksAsync failed for task 25: CANCELLED: CANCELLED
--
6739-./mpas_s9.26600095.log-292-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
6740-./mpas_s9.26600095.log-293-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6741:./mpas_s9.26600095.log:294:{"component": "mpas_atm", "subdivision": 9, "n_devices": 64, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 6474.6, "steady_median_ms": 9.6, "steady_min_ms": 9.55, "per_step_ms": [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 9.601722995284945, "total_cells": 68160768, "sypd": 8.55424840461413, "mcells_per_s": 7098.805915716509, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:31:30.728837+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 1065012}}}
6742-./mpas_s9.26600095.log-295-[mpas nd=64 L9 nCells=2621568 nlev=26] compile=6474.6ms steady_median=9.60ms/step (per-step: [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6])
6743-./mpas_s9.26600095.log-296-W0801 05:31:38.185920  251239 pjrt_client.cc:1604] WatchTasksAsync failed for task 19: CANCELLED: CANCELLED
--
6747-./mpas_s9.26600095.log-927-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
6748-./mpas_s9.26600095.log-928-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
6749:./mpas_s9.26600095.log:929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
6750-./mpas_s9.26600095.log-930-[mpas nd=128 L9 nCells=2621568 nlev=26] compile=7532.3ms steady_median=11.48ms/step (per-step: [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4])
6751-./mpas_s9.26600095.log-931-W0801 05:45:51.458955  432147 pjrt_client.cc:1604] WatchTasksAsync failed for task 75: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
--
10304-FILE=mpas_recheck.26454618.log
10305-FILE=mpas_s9.26549775.log
10306:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10307-FILE=mpas_s9.26600095.log
10308-FILE=mpasoc_2x2.26508258.log
--
10368-497-        # was actually measured (codex M3c-2 MINOR).
10369---- current s9 exact receipt (all JSON rows) ---
10370:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10371:294:{"component": "mpas_atm", "subdivision": 9, "n_devices": 64, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 64, "multicontroller": true, "compile_ms": 6474.6, "steady_median_ms": 9.6, "steady_min_ms": 9.55, "per_step_ms": [6474.6, 13.0, 10.1, 10.0, 9.6, 9.8, 9.6, 9.6, 9.6, 9.6, 9.5, 9.6], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 9.601722995284945, "total_cells": 68160768, "sypd": 8.55424840461413, "mcells_per_s": 7098.805915716509, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:31:30.728837+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 1065012}}}
10372:929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
10373---- current metis exact receipt (all JSON rows) ---
10374-450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
--
10382- succeeded in 328ms:
10383---- s9 32 full exact record ---
10384:67:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10385---- s8 weak-pair receipt candidates ---
10386-./mpas32.26549646.log
--
10585-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
10586-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
10587:{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
10588-[mpas nd=32 L9 nCells=2621568 nlev=26] compile=7179.2ms steady_median=12.47ms/step (per-step: [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2])
10589-W0801 05:24:48.442624  225571 pjrt_client.cc:1604] WatchTasksAsync failed for task 25: CANCELLED: CANCELLED
--
12804-    65	/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
12805-    66	  return lax_numpy.astype(self, dtype, copy=copy, device=device)
12806:    67	{"component": "mpas_atm", "subdivision": 9, "n_devices": 32, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 32, "multicontroller": true, "compile_ms": 7179.2, "steady_median_ms": 12.47, "steady_min_ms": 12.22, "per_step_ms": [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 12.468885004636832, "total_cells": 68160768, "sypd": 6.587238841598036, "mcells_per_s": 5466.468571540511, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:24:40.404674+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "unknown", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 2130024}}}
12807-    68	[mpas nd=32 L9 nCells=2621568 nlev=26] compile=7179.2ms steady_median=12.47ms/step (per-step: [7179.2, 12.9, 12.6, 12.5, 12.5, 12.5, 12.5, 12.4, 12.5, 12.5, 12.2, 12.2])
12808-    69	W0801 05:24:48.442624  225571 pjrt_client.cc:1604] WatchTasksAsync failed for task 25: CANCELLED: CANCELLED
7151d12a1 fix(bench): --lloyd on the MPAS SPMD bench so subdiv>=9 finds its cache
 .../codex_scaleout_improvements_2026-07-31.md      | 4439 ++++++++++++++++++++
 scripts/bench/bench_mpas_spmd_scaling.py           |   15 +-
 tests/bench/test_bench_mpas_spmd_gates.py          |   27 +-
 3 files changed, 4477 insertions(+), 4 deletions(-)
commit 7151d12a184cf7324495b264e8bfa087d6ba4737
Author:     Pierre Gentine <b381103@levante6.lvt.dkrz.de>
AuthorDate: Fri Jul 31 19:12:56 2026 +0200
Commit:     Pierre Gentine <b381103@levante6.lvt.dkrz.de>
CommitDate: Fri Jul 31 19:12:56 2026 +0200

    fix(bench): --lloyd on the MPAS SPMD bench so subdiv>=9 finds its cache
    
    The subdiv-9 payoff ladder (job 26549775) died at every rank: the bench
    hardcoded create_voronoi_mesh's lloyd_iterations=50 default, so it asked
    for the production-SCVT cache key while the prewarmed file (job 26549180)
    is the labelled synthetic scaling mesh, lloyd=0. The big-mesh policy then
    correctly REFUSED to build 2.6M cells under a 128-rank launch.
    
    --lloyd plumbs through to the builder and is recorded in the JSONL row, so
    a synthetic-mesh scaling receipt can never masquerade as a production-SCVT
    one. Test asserts the kwarg reaches create_voronoi_mesh via a sentinel
    (fails if the plumbing is dropped).
    
    Also files the codex consult on remaining scale-out levers (ranked, with
    per-item refutation tests) under .physics-validator/scaling_campaign/.
    
    Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01QcWJegdGyaRHM3EQ55FrK3

diff --git a/scripts/bench/bench_mpas_spmd_scaling.py b/scripts/bench/bench_mpas_spmd_scaling.py
index c0228353a..7ca7f9424 100644
--- a/scripts/bench/bench_mpas_spmd_scaling.py
+++ b/scripts/bench/bench_mpas_spmd_scaling.py
@@ -97,7 +97,7 @@ MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
 
 
 def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
-                          moist=False):
+                          moist=False, lloyd_iterations=50):
     """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
 
     ``reorder_target`` sets the PARTITION (and ghost padding) so every run
@@ -117,7 +117,8 @@ def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
     from legoesm.parallel.mesh import create_voronoi_device_mesh
     from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
 
-    mesh = create_voronoi_mesh(subdivision_level=subdivision)
+    mesh = create_voronoi_mesh(subdivision_level=subdivision,
+                               lloyd_iterations=lloyd_iterations)
     mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
     if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
         # Padding only guarantees divisibility for reorder_target.
@@ -179,6 +180,11 @@ def main() -> int:
                    help="icosahedral subdivision level L "
                         "(nCells = 10*4^L + 2 before ghost padding)")
     p.add_argument("--nlev", type=int, default=8)
+    p.add_argument("--lloyd", type=int, default=50,
+                   help="Lloyd relaxation iterations for the mesh. 50 = "
+                        "production SCVT; 0 = labelled synthetic scaling "
+                        "mesh (scaling receipts only, never physics — "
+                        "must match the prewarmed cache key at subdiv>=9).")
     p.add_argument("--n-devices", type=int, required=True)
     p.add_argument("--reorder-for", type=int, default=None,
                    help="partition/reorder the mesh for THIS device count "
@@ -307,7 +313,7 @@ def main() -> int:
             f"partition target.")
     mesh, model, s0, dev_config = build_model_and_state(
         args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
-        moist=(args.physics == "kessler"))
+        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
 
     if args.multicontroller:
         # Every process computed the reorder independently — assert the
@@ -492,6 +498,9 @@ def main() -> int:
         subdivision=args.subdivision, n_devices=nd,
         n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
         partition_method=args.partition_method, physics=args.physics,
+        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
+        # a row without this field could pass as a production-SCVT receipt.
+        lloyd_iterations=args.lloyd,
         # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
         # saying "auto" would not reveal whether ppermute or allgather
         # was actually measured (codex M3c-2 MINOR).

codex
The numerical table is correct. I found two material qualification gaps: the “matched” tiles differ slightly after padding, and the 81.9k comparator is the s9 row whose recorded SHA is `unknown`; the other two s9 rows are pinned to `7151d12a1`.
exec
/usr/bin/bash -lc "find /work/bd1083/b309178 -type f \\( -name '*26628076*' -o -name '*s8_l0*' \\) -print 2>/dev/null | head -80 && awk 'BEGIN { printf \"12.47/6.58 = %.9f; efficiency %.9f\\n\",12.47/6.58,6.58/12.47; printf \"9.60/6.43 = %.9f; efficiency %.9f\\n\",9.60/6.43,6.43/9.60; printf \"11.48/7.29 = %.9f; efficiency %.9f\\n\",11.48/7.29,7.29/11.48; printf \"padding: L8=%d, L9=%d\\n\",655488-655362,2621568-2621442; printf \"L8/L9 tiles: %d/%d, %d/%d, %d/%d; total scale %.9f\\n\",655488/8,2621568/32,655488/16,2621568/64,655488/32,2621568/128,2621568/655488; printf \"relative tile mismatch %.7f%%\\n\",100*((655488/8)/(2621568/32)-1); printf \"6.58 vs 6.92: %.7f%%\\n\",100*(6.58/6.92-1)}' && rg -n -C 3 \"halo-strategy\" scripts/bench/bench_mpas_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
exec
/usr/bin/bash -lc "awk 'BEGIN { printf \"12.47/6.58 = %.9f; efficiency %.9f\\n\",12.47/6.58,6.58/12.47; printf \"9.60/6.43 = %.9f; efficiency %.9f\\n\",9.60/6.43,6.43/9.60; printf \"11.48/7.29 = %.9f; efficiency %.9f\\n\",11.48/7.29,7.29/11.48; printf \"padding: L8=%d, L9=%d\\n\",655488-655362,2621568-2621442; printf \"L8/L9 tiles: %d/%d, %d/%d, %d/%d; total scale %.9f\\n\",655488/8,2621568/32,655488/16,2621568/64,655488/32,2621568/128,2621568/655488; printf \"relative tile mismatch %.7f%%\\n\",100*((655488/8)/(2621568/32)-1); printf \"6.58 vs 6.92: %.7f%%\\n\",100*(6.58/6.92-1)}' && rg -n -C 3 \"halo-strategy\" scripts/bench/bench_mpas_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 347ms:
12.47/6.58 = 1.895136778; efficiency 0.527666399
9.60/6.43 = 1.493001555; efficiency 0.669791667
11.48/7.29 = 1.574759945; efficiency 0.635017422
padding: L8=126, L9=126
L8/L9 tiles: 81936/81924, 40968/40962, 20484/20481; total scale 3.999414177
relative tile mismatch 0.0146477%
6.58 vs 6.92: -4.9132948%
202-                        "exchange + RK tracer advection on the gated "
203-                        "path) and extends the parity gate to the "
204-                        "tracer fields.")
205:    p.add_argument("--halo-strategy",
206-                   choices=["auto", "ppermute", "allgather"],
207-                   default="auto",
208-                   help="Halo strategy for make_voronoi_sharded_step. "

exec
/usr/bin/bash -lc "git diff --check && nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '1754,1790p' && nl -ba scripts/plot/plot_scaling_paper_figure.py | sed -n '60,76p' && git diff --name-only 7151d12a1 --" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 457ms:
  1754	unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
  1755	result JSON — only `decomposition: 2d`; a follow-up could add them to
  1756	the bench metadata.)
  1757	
  1758	### s8 lloyd=0 de-confound ladder landed (job 26628076): the matched-tile scale-out term is REAL
  1759	
  1760	s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
  1761	warmup 3 — protocol-identical to the s9 ladder (26600095), git
  1762	7151d12a1-dirty (dirty = this session's doc/plot edits; bench path
  1763	untouched): **6.58 / 6.43 / 7.29 ms**.
  1764	
  1765	Clean weak pairs (4x cells with 4x GPUs, SAME lloyd-0 family):
  1766	
  1767	| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
  1768	|---|---|---|---|---|
  1769	| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
  1770	| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
  1771	| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
  1772	
  1773	* The falsifiability block's CONFIRM branch fires: ratios stay well
  1774	  above 1, so the round-20 retraction's CONFOUND did not manufacture
  1775	  the effect — it only biased its size (confounded draft 1.80/1.35/1.41
  1776	  vs clean 1.90/1.49/1.57; the production-mesh s8 np8 was 6.92 vs
  1777	  lloyd-0 6.58, -5 %, so the mesh family does shift absolutes).
  1778	* Restated: at MATCHED per-GPU tile, quadrupling devices+problem costs
  1779	  1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
  1780	  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
  1781	  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
  1782	  fraction growth, collective latency vs count, sfc partition-quality
  1783	  decay with parts; the metis receipt argues against pure
  1784	  partition-cut explanations, on the CPU lane at least).
  1785	* The non-monotone tile dependence of the ratio (largest at the
  1786	  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
    60	                ("float32 (C384)", [(6, 15.44), (24, 8.81)])],
    61	        note="f64 pending",
    62	    ),
    63	    dict(
    64	        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
    65	        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
    66	                                        (16, 7.10), (32, 8.13), (64, 5.27),
    67	                                        (128, 6.47)]),
    68	                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
    69	                ("f32 (s8 lloyd-0)", [(8, 6.58), (16, 6.43), (32, 7.29)]),
    70	                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
    71	        note="weak eff 0.53–0.67 at\nmatched tile (lloyd-0 pairs)",
    72	    ),
    73	    dict(
    74	        key="atm_ico_cpu", title="ico + lat-lon 2-D", sub="subdiv-7 / r512 L26 · Milan CPU–MPI",
    75	        series=[("float32", [(1, 7399.72), (2, 3250.27), (4, 1673.07),
    76	                             (8, 827.06), (16, 444.58), (32, 230.35),
docs/performance/scaling/levante_campaign_2026-07-24.md
scripts/plot/plot_scaling_paper_figure.py

codex
Arithmetic is correct:

| Pair | Ratio | Efficiency |
|---|---:|---:|
| 6.58 → 12.47 | 1.89514 → 1.895 | 0.52767 → 0.53 |
| 6.43 → 9.60 | 1.49300 → 1.493 | 0.66979 → 0.67 |
| 7.29 → 11.48 | 1.57476 → 1.575 | 0.63502 → 0.64 |

Issues to fix in [the new section](docs/performance/scaling/levante_campaign_2026-07-24.md:1758):

- “Protocol-identical,” “Clean,” “matched,” “REAL,” and “only biased its size” overclaim. The s9 np32 / 81.9k row records `git_sha: unknown`; only np64 and np128 are pinned to `7151d12a1`. The s8 `-dirty` identifier also lacks an archived dirty-file manifest. Call the 81.9k pair configuration-matched but not reproduction-grade.
- Padding makes tiles near-, not exactly, matched. Both meshes add 126 cells, yielding L8/L9 tiles of 81,936/81,924, 40,968/40,962, and 20,484/20,481 cells/GPU: a 0.01465% difference. The global size ratio is 3.999414×. This is negligible against the observed 49–90% time cost, but wording should say “near-matched after padding” and “approximately 4×.”
- The 6.92 → 6.58 change is correctly −4.91% (~−5%), and the rerun removes the known Lloyd-family mismatch. It does not prove that Lloyd “only” biased the old effect: these are unreplicated measurements from separate allocations, with one comparator lacking provenance.
- Keeping mechanism unattributed and the non-monotonicity untheorised is appropriate.
- The figure data are consistent, but soften its note to “observed weak efficiency … at near-matched tile”; label the production subdiv-8 series as Lloyd=50 for clarity.

`git diff --check` passes, and the current diff from `7151d12a1` touches only the doc and plot, which supports—but cannot retrospectively prove—the dirty-tree assertion.

VERDICT: FIX-FIRST
2026-08-02T11:29:37.166688Z ERROR codex_core::session: failed to record rollout items: thread 019fc238-56a1-75b1-b93f-8220e8fecfaf not found
2026-08-02T11:29:37.166726Z ERROR codex_core::session: failed to record rollout items: thread 019fc238-56a1-75b1-b93f-8220e8fecfaf not found
tokens used
153,514
Arithmetic is correct:

| Pair | Ratio | Efficiency |
|---|---:|---:|
| 6.58 → 12.47 | 1.89514 → 1.895 | 0.52767 → 0.53 |
| 6.43 → 9.60 | 1.49300 → 1.493 | 0.66979 → 0.67 |
| 7.29 → 11.48 | 1.57476 → 1.575 | 0.63502 → 0.64 |

Issues to fix in [the new section](docs/performance/scaling/levante_campaign_2026-07-24.md:1758):

- “Protocol-identical,” “Clean,” “matched,” “REAL,” and “only biased its size” overclaim. The s9 np32 / 81.9k row records `git_sha: unknown`; only np64 and np128 are pinned to `7151d12a1`. The s8 `-dirty` identifier also lacks an archived dirty-file manifest. Call the 81.9k pair configuration-matched but not reproduction-grade.
- Padding makes tiles near-, not exactly, matched. Both meshes add 126 cells, yielding L8/L9 tiles of 81,936/81,924, 40,968/40,962, and 20,484/20,481 cells/GPU: a 0.01465% difference. The global size ratio is 3.999414×. This is negligible against the observed 49–90% time cost, but wording should say “near-matched after padding” and “approximately 4×.”
- The 6.92 → 6.58 change is correctly −4.91% (~−5%), and the rerun removes the known Lloyd-family mismatch. It does not prove that Lloyd “only” biased the old effect: these are unreplicated measurements from separate allocations, with one comparator lacking provenance.
- Keeping mechanism unattributed and the non-monotonicity untheorised is appropriate.
- The figure data are consistent, but soften its note to “observed weak efficiency … at near-matched tile”; label the production subdiv-8 series as Lloyd=50 for clarity.

`git diff --check` passes, and the current diff from `7151d12a1` touches only the doc and plot, which supports—but cannot retrospectively prove—the dirty-tree assertion.

VERDICT: FIX-FIRST
