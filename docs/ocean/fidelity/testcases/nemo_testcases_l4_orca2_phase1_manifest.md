# NEMO testcase Lane 4 ORCA2 Phase-1 artifact manifest

This is the review index for the immutable artifacts below
`/data/abyssal/dbalwada/nemo-testcases-l4`. Large data are not committed.
Every digest is SHA-256 unless explicitly identified as MD5. The authoritative
per-file inventories remain beside the data and are themselves hash-pinned
here.

Independent-review status: the corrected replacement binary and executed run
below are the accepted Phase-1 oracle. The prior accepted artifacts remain
immutable and hash-pinned, but are superseded; no record from them is mixed
into this oracle.

## Source, input archive, compiler, and executables

| artifact | bytes | digest |
|---|---:|---|
| NEMO source git commit | - | `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796` |
| `inputs/ORCA2_ICE_v5.0.0.tar.gz` | 1,365,673,011 | `5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4` |
| archive published MD5 | - | `872eea51f22c5dabaffd268c0220eba6` |
| `arch/arch-conda-scalarmath.fcm` | - | `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561` |
| `build/nemo_ORCA2_OMIP_L4_uninstrumented.exe` | - | `c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343` |
| `build/build_ORCA2_OMIP_L4_uninstrumented.log` | - | `b4622a7fb64cabd0fd7f8133b4a61d08c7d67947ba9d50aa87d622bb875360c8` |
| `build/nm_D_uninstrumented_ZGV.txt` (empty) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `build/nemo_ORCA2_OMIP_L4_instrumented_rank0_schema.exe` | 54,904,016 | `07cec34c5683e6a3e37fa432ac5796ab43c747ab6f468fd131a8be796996dbb8` |
| `build/build_ORCA2_OMIP_L4_instrumented_rank0_schema.log` | - | `42614a4b1c2b62129acdfa93569f2d172c51bbd09a0c8616422ca9aae854d028` |
| `build/instrumented_rank0_schema_MY_SRC.sha256` | - | `84861aaab9aaa7a2ea35cb5a749cfb0350595c485d10e9b288db6f5e9a0e62c6` |
| `build/instrumented_rank0_schema_ZGV.txt` (empty) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `build/nemo_ORCA2_OMIP_L4_instrumented_reviewfix.exe` | 54,899,920 | `ca2355aa777adc47825c4b777c5fa4e89cc60dc2c509a0e2f379e439acefd025` |
| `build/build_ORCA2_OMIP_L4_instrumented_reviewfix.log` | 105,973 | `96a38f0415970a5d3987d67374d1583ad30a19725d1b199465a60b46857526a7` |
| `build/instrumented_reviewfix_MY_SRC.sha256` (14 files) | 2,100 | `78e1465fe7d9cea4d3adf24fb369b6a458c7bb7a912cc75f85d7ca6177e59152` |
| `build/instrumented_reviewfix_ZGV.txt` (empty) | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Archive provenance: fetched by the user's shell on 2026-09-05 from
`https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/ORCA2_ICE_v5.0.0.tar.gz`;
the agent sandbox had no network. The archive was retained unchanged.

## Deck and active inputs

The final run stores ordinary copies of the 19-file deck and absolute symlinks
to the 40 immutable input payloads. Its complete deck manifest is
`e1974d9db5974f54d451fe1112b8f22a404516adcc7cbcf0b035fbe1cccba70b`;
its active-input manifest is
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
The four verbatim inherited/resolved source namelists and NEMO's two resolved
echoes are retained in the final run directory:

| file | digest |
|---|---|
| `namelist_cfg` | `62f4746cf3846254c18af73bcde53e42f7d4cb9ebcdaf5f63cc9ed5316a9f8f4` |
| `namelist_ice_cfg` | `9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8` |
| `namelist_ref` | `b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd` |
| `namelist_ice_ref` | `055ab5e6839c12dbae61bf1c3bcdda46a8f95cf275b7f294d68d7f8e73179307` |
| `output.namelist.dyn` | `31ceebedb15a0e427d9fc530029cc98e18db6749545ddc354ca749d2192e4148` |
| `output.namelist.ice` | `2ccc2d012159bcc1d182dc48bf34ff2abdfb09aecb45ff227cd30ebadd9a8603` |

Every file read by an active resolved descriptor follows. This is a verbatim
copy of `input_files.sha256`; no other archive member was opened:

```text
cbab44dbbff4b87999c54063f6fbd9314201c7dc91ecbe175ae9dde966f4d907  EMPave_old.dat
0421cdebf62244168847ae7fcb7cae22fa459786a37a354ec286addcada6aa10  K1rowdrg.nc
5e267a27c17ba1e3e1ce30d79c94dc82c6a155a0acc38a1b2dd4aa91c7bdf465  M2rowdrg.nc
7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839  ORCA_R2_zps_domcfg.nc
16213186719d81af672dee7ad77558a7294282b433942ddf791a94d01a86d5e5  ahmcoef.nc
77e77c89ec8aa9963f079e13c8cf9c5378ff697ca21bb819207e98e14cfbaeb0  assim_background_increments.nc
c126f3bc7346fa5a22933303a8ace184c104bf3f5743ebf684fd5e759dbb988c  calving.nc
9178f44245f9465c584168eeb47b28e83c358a9ef78ed614cd623c4456ad7890  chlorophyll.nc
ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a  data_1m_potential_temperature_nomask.nc
ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c  data_1m_salinity_nomask.nc
d40b5034e133662a0b8395ecf65bee9be52976f3934b2cb58b5bc0732307f90f  eddy_viscosity_2D.nc
fc142ce094a0f68d255ff79b04f8fe5d6634b33b0bc1c07b61870cede39499cb  eddy_viscosity_3D.nc
e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb  geothermal_heating.nc
b4f23c827cbc190e508d037c61b75c4c9d02a7352e78d382a964fe6d8d5eedae  gh_flux_Lucazeau-orca2.nc
c17c9b03e9b25ce51fa983cd5e29e2283390acaed7308955fe32ac94210e6727  int_wave_mix.nc
709b516ae53f394f0fb8c621357160a4b11251659187c96413fcc643f4e5a163  mask_itf.nc
ed17b5069ddb13cd2a596079a0102e7792899e5dbe4ce1ba5a20e29802b5337a  ncar_precip.15JUNE2009_fill.nc
050ee263198d49ce285070bc4d3e75953091c61c8f6a0d992484e79974fc37a7  ncar_rad.15JUNE2009_fill.nc
f135428e7e43287e3c17faf4c2869c10cc92e82417a955d4f39f18a7b930b44b  profiles_01.nc
4c46f4527e8e5c1f3cc12942a87d0938c10431fcdb56cff7d25a9b5db2e69806  q_10.15JUNE2009_fill.nc
f4b6aec4695f984eab3a3ad9f2efdcf80d19180cf5f55aa4a93e3f45639b8634  resto.nc
b92ff46e4c406096a1e1fe3815dfdcce12b234e84541a3b6dffb1037e777d4eb  runoff_core_monthly.nc
7f80998af82a33819a7e3b2e54b30b64e847e39377658dcebe7e21cb8b2d0c30  sali_ref_clim_monthly.nc
9a819f5db674feb12a0dab0cf99a04925d27de2da3e06df8a4ad7554f564d60a  sdw_ecwaves_orca2.nc
4125a5efe8e30d732a1fad3798a535a367075e28f13b55a64cab2fc562747292  seaice_01.nc
8d9d05d04bd616eedd2be0fffdfc6af2ce8e5b62e3c55047bca7b279f42536cb  sic_01.nc
dc9402d1c22fd806361fb228f3801b63f7e2734ec35f5a5f63a891aa63d54e49  slaReferenceLevel.nc
72bbd7480d2a208e627a213a58ca0a74581646ba280bda6f3e5ae3f98027e721  sla_01.nc
19dea3069caa8f513218c7cbc11b3dce83d7cb9363e44b787e892917625c81fb  slp.15JUNE2009_fill.nc
ea4609d6deca4ccf95b6307a7b7eed1bff3b767146eb69c2a82762824bf1ed49  sss_data.nc
209e9559312531af188fc35339866ea6e9a0604282eae48fc2763024c0478663  sst_01.nc
79af1b1a9f3c2209acc1cfbee807127adb9e54a40555e980648265958984b3d5  sst_data.nc
3d3d174a869f758b2f2fa0fc9a713be0e25a5cf61384147327ed9005348aafb2  subbasins.nc
bee5df15a69d757f3911660910efc54c936555ec8bce742215ec047a16a4338b  t_10.15JUNE2009_fill.nc
219bf3a6891dfe25871ba9d739e8ad465e12934b62f6524422a985b3858ae96b  u_10.15JUNE2009_fill.nc
b3913b212b6a194fa8f79214ea56819728cbef94a06c6a82e94d65337d6e7b3a  v_10.15JUNE2009_fill.nc
1f20fde5cd70367767a1db4a4ed3aeccee62fa484040ec066b357539f1f59465  vel_01.nc
2cfefd58b0231f355cf999f943a48586f63e4e9264f0cbcc7197334f12b3e7e8  weights_core2_orca2_bicub.nc
e5fe31582615262d84b06d01aeaa1893f4d942dc978aebc3e13f56cda484d766  weights_core2_orca2_bilin.nc
f4279899be77119fda4f9e20b3d5e3a70e4e478fe5ead0dc661656a1bba296c2  zdfiwm_forcing_orca2.nc
```

## Run and gate artifacts

| artifact | digest |
|---|---|
| accepted replacement ten-step `run.sh` | `756d6ac26e851ecba33b0ef8a6061238e4b8ca289d9e2011b4b0e8747c3d7415` |
| final ten-step `run.user.stdout.log` | `e527622f8470458702f16b74c81c4d7c36f2cfdc6e95ac948a9e289dfe9634e5` |
| final ten-step `run.user.time.log` | `54e003c1c35a06be5efc832816353a825ff346230e45408f6e9459188cfd07c4` |
| final ten-step `run.launcher.log` | `06d9fdaa12f428f078ea8d292daa5ba429a6c6679b136879c77f073ca15174cb` |
| final ten-step `ocean.output` | `06c41ebf47b42e248879c32a2c47bc6acd9b3c06766378ebd0ea7267e730a534` |
| final ten-step `time.step` | `83e4e460507d78e2bd843e9a6961b3b204c2b5229499b5e2f10f11b3a7d5bb78` |
| accepted replacement 190-file census | `329de5e238356664eb31371042c49ebcf2e7d2d2039f35fc157710318d628f7b` |
| accepted replacement 90-record census | `70c3779bc11e4b3df41ebf756abf25362dab82d9622c87829012618412cf273a` |
| accepted replacement gate JSON with controls | `dc42802b3f925c4197b75003431e3ba1762bfa18a39a038f8a864f10cc553577` |
| committed gate script | `3d0b61762a4dbe0acbdaa3b52791696346a73afc84a78bb7388a167999aa00ce` |
| historical gate on superseded prior run, all nine plants | `22866e967457e0e05eb112f0179a58578c50e483b48cd1f8d9624421c46d276f` |
| committed superseded prior-run launcher | `59e794abc299cc34897eddd905e492edf8fe41dde456977283abf0161fc767ea` |
| committed uninstrumented identity-control launcher | `2753bbc0233cc4926062053a638f7aa347c4461ab5c072edbd3f1cfb3a9eff67` |
| historical prepared review-correction 63-file census | `20c7d059bab0b50e11b60ba6c237ef8ab136e6b961f98b375f260769257b30a9` |
| uninstrumented ten-step 97-file census | `924867d6e066d450715a0997ecfd1264005b02abe8f4d05228cf29870fff03a7` |
| uninstrumented 30-day 97-file census | `efe02aa764feb07ba045ee25c0bd304240b7780e39c01376dec741b4520413f9` |
| retained first, incomplete 63-record run census | `b633486295afca2c8f6930258cdcb4a9dc86d0cf0bc0eb0bcda372814505a4b6` |

All historical instrumented runs are preserved. The first three
(`instrumented_10step_np2`, `instrumented_full_10step_np2`, and
`instrumented_rank0_10step_np2`) are retracted because their record sets are
incomplete, concurrently written, or carry the malformed reassociation
header. `instrumented_rank0_schema_10step_np2` passed its contemporary gate
but is superseded by the independent-review replacement. All four are
deliberately excluded from the accepted oracle inventories, and their exact
prepared decks remain hash-pinned by the corresponding preparation censuses.

## Committed evidence documents

| artifact | digest |
|---|---|
| `nemo_testcases_l4_orca2_phase1_receipt.md` | `7619dfa1ce041053671da3517cbd60dc3d2a2ee4671255cfa3765e5245ff1668` |
| `nemo_testcases_l4_orca2_phase1_preregister.md` | `2a51809a6277310a2441faa7be0d1588f17294ef151d979438ccc0645e1fac6b` |
| `nemo_testcases_l4_orca2_phase1_run_recipes.md` | `544e4da05de4469cd851139606cf91d2357377cff9362d4d4567b20bafe93ae3` |
| `nemo_testcases_l4_orca2_phase1_instrument_rebuild_preregister.md` | `0548b565ec7804ee45ca2c96f65344fdc94e9783a8bba71c2cfb5ca48271e5ba` |
| `nemo_testcases_l4_orca2_phase1_mpi_writer_correction_preregister.md` | `802bdaff23cc6bbccc6f69c4fbcdb445886af0b8838a8c81fea92dec71bfe0c9` |
| `nemo_testcases_l4_orca2_phase1_payload_count_correction_preregister.md` | `493c9833e385341a0e21fd89c07b04b2d43272620deb30119f1430aa8b5569b8` |
| `nemo_testcases_l4_orca2_phase1_review_rerun_preregister.md` | `2a47e2a33f89f2ad4f4db6fe5d283eb5455efd5755601b37e203d53c6f0cf1cc` |

## Terminal restart hashes

| run | shard | digest |
|---|---|---|
| ten-step final/control | ocean 0000 | `9ba6054040f11ea38a6637c4e6e7e26261359540fbf2d5be2cf63b78857dd508` |
| ten-step final/control | ocean 0001 | `d06e099765da5e6eba9f2f6ecb8aa66b20fb034fa506456daeb8da0cb8fea1ba` |
| ten-step final/control | iceberg 0000 | `1dc69243e465964a99217c5729df5f21ee23a583507e7cbdb274502700eb3926` |
| ten-step final/control | iceberg 0001 | `5b183bb7e79003a2d36325247246d9265b8030e21ebc4ffd3507ab6a9450e5b1` |
| ten-step final/control | SI3 0000 | `5266345352fecb609820d3e62f495019d2bd05d4e0bdb40d0a327bd6c9dafd20` |
| ten-step final/control | SI3 0001 | `2634bd96f623fe476c3c389ea232b03583793686dde8781120427320676b68e2` |
| 30-day | ocean 0000 | `02765873a4a75e817eae8847cca1b24c25c27fe2f9b4731e61867eeb857b989e` |
| 30-day | ocean 0001 | `3e858ab85e34f1d770d738f39ccaa2cb8fbe5f9aa8e612f2fd6e6ecb80f8b57c` |
| 30-day | iceberg 0000 | `60d92b35538fbc4526554a6151961c878bb6127ac352ee35e2105a5f7b1dfd4a` |
| 30-day | iceberg 0001 | `5b183bb7e79003a2d36325247246d9265b8030e21ebc4ffd3507ab6a9450e5b1` |
| 30-day | SI3 0000 | `2e1e961ab4ecb53050e4f2897399d1da32905e07300292d444acb7d7474e70d2` |
| 30-day | SI3 0001 | `b746e0581c28224889e0b1b9bf3420c45f57ef24ad6a9e32d4bab853c45891e0` |
