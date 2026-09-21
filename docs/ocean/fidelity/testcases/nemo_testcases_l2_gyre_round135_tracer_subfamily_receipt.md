# NEMO testcase L2 GYRE phase 3 — round 135 tracer-subfamily receipt

Date: 2026-09-21

Incoming tip: `4569f934d4e08deb8415218655619814c33533ab`

Status: **HELD — splitting the successful daily T/S reset does not produce an
additive owner. Temperature-only is the better singleton at day 240, reducing
the T3D RMS gap by `5.286573087889906e-3 K`, but it worsens every other scored
field there and worsens temperature at day 360; salinity-only increases the
day-240 T gap to `1.4291755266806412 K`. Two-, four-, and eight-day
temperature resets all worsen day 240. No source statement, physics, carried
state, or configuration lands.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round135/`

## Outcome first

The clean same-tip control is bit-identical to the immutable Round-130 arm for
all five saved fields on all 360 days. The admitted oracle record was then used
for six independent production-JIT year integrations: daily temperature-only,
daily salinity-only, and temperature-only every 2, 4, and 8 days, alongside
the control.

At day 240 the free T3D RMS gap is `1.6446740233175394e-2 K`:

| intervention | reset-vs-NEMO T RMS (K) | free minus reset (K) | verdict |
|---|---:|---:|---|
| combined daily T/S, Round 134 parent | `5.209685836156627e-5` | `+1.6394643374813826e-2` | 99.683239% removed |
| daily temperature only | `1.1160167145285487e-2` | `+5.286573087889906e-3` | singleton winner, only 32.2457% of parent removal |
| daily salinity only | `1.4291755266806412` | `-1.4127287864474658` | catastrophic worsening |
| temperature every 2 days | `2.033656627544235e-2` | `-3.8898260422669563e-3` | worsening |
| temperature every 4 days | `1.7212630812249602e-1` | `-1.5567956788932064e-1` | worsening |
| temperature every 8 days | `4.4102754481466183e-1` | `-4.2458080458148645e-1` | worsening |

This is a refutation of additive single-tracer ownership. Resetting temperature
alone also makes the day-240 salinity, velocity, and SSH scores much worse;
resetting salinity alone is worse still. At day 30 the temperature-only T gap
is already `4.734540389865452e-2 K`, versus the free gap
`6.890431487825909e-5 K`. The combined T/S intervention therefore owes its
effect to keeping the pair together; neither singleton result can be promoted
to the first wrong tracer-producing statement. This conclusion is numerical:
it does not guess which density, advection, diffusion, or mixing mechanism
creates the interaction.

The cadence prediction is also refuted. Day-240 temperature removal is
strictly non-increasing as the interval grows, as predicted, but only the
daily arm is positive. The apparent daily benefit is not robust to even a
two-day interval and does not survive to day 360, where the daily arm worsens
the T gap from `1.1223571247843664e-2 K` to
`2.734737495902395e-2 K`.

## Compiled-source contract and reset timing

The source citations name the compiled branch that produced the oracle
record. NEMO finishes stage 3 and swaps the accepted state into `Nbb` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:220-229`. After the
completed-step diagnostics it calls the restart writer with that `Nbb` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:249-260`. The writer
stores temperature `tn` and salinity `sn` as separate components of the
accepted `Kbb` tracer state at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:176-184`.

The harness snapshots the model before each reset. A day-1 replacement is
therefore first consumed by steps 7–12 and first visible in the day-2
pre-reset snapshot. All singleton arms are bit-identical to the free run at
day 1 in `T`, `S`, `u`, `v`, and `ssh`. Daily arms apply 359 resets at days
1–359. The 2-, 4-, and 8-day arms use only exact recorded boundaries and apply
179, 89, and 44 resets, respectively. No oracle value is interpolated,
shifted, or reused at a different boundary.

## Frozen prediction ledger

* **P0 CONFIRMED.** The record audit found exactly 360 boundaries at steps
  `6,12,...,2160`, all 18 required variables, and 12/12 monthly overlaps
  bit-identical. The fresh control has zero unequal cells for every saved field
  across all 360 days. Only after that result was recorded was the certified
  harness digest moved to the Round-135 diagnostic extension.
* **P1 PARTLY REFUTED.** Temperature is the better singleton at day 240, so
  the frozen winner is **CONFIRMED**. Its predicted removal of at least
  `1.55e-2 K` is **REFUTED**: measured removal is only
  `5.286573087889906e-3 K`. The literal bound that salinity removes less than
  `1.0e-3 K` is satisfied, but not in the benign sense anticipated: it removes
  `-1.4127287864474658 K`, a major worsening. All 80 family/day/field rows are
  registered.
* **P2 PARTLY REFUTED.** The non-increasing day-240 removal ordering is
  **CONFIRMED**, but positivity at 2, 4, and 8 days is **REFUTED**. The first
  western-third, upper-100-m reduction is day 2 as predicted. All 160
  cadence/day/field rows are registered.
* **P3 CONFIRMED.** The family isolation, source digest, exact boundary lists,
  cadence, and score registries are fail-closed and plant-controlled. No
  production card, physical statement, default, carried state, or NEMO source
  changed. The status remains HELD.

## Complete registered score table

Each cell is `reset-vs-NEMO RMS (free-minus-reset RMS)`. Positive parentheses
mean improvement; negative values mean worsening. Units are K for T, g/kg for
S, m/s for u/v, and m for SSH. The displayed tables contain every unique
arm/day/field row. The cadence registry's 1-day arm is exactly the displayed
daily-temperature table and is not duplicated a second time. The
machine-readable artifact `tracer_subfamily_cadence.json` (SHA-256
`7e37ed066adb3653791a1d773ca975a39e72b5ffa873de3bf1337d9f490a0d7c`)
also registers free-vs-NEMO and reset-vs-free RMS for every row. It reports
`subfamily_registered_row_count=80`, `cadence_registered_row_count=160`, both
exact-registry booleans true, and `all_moved_rows_registered=true`.

### Daily temperature only / cadence 1 day

| day | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 30 | `0.04734540389865452 (-0.04727649958377626)` | `0.015481841973726089 (-0.015470060751245246)` | `0.005052151887998517 (-0.0050465717899726)` | `0.0030004456089589046 (-0.002995825530867757)` | `0.002430505501622569 (-0.0024237027198605026)` |
| 60 | `0.2783860930795474 (-0.27819279324034335)` | `0.9119736098121667 (-0.9119534864925082)` | `0.05534203374550158 (-0.0553215288783996)` | `0.05548699690881997 (-0.05541057466457935)` | `0.04030369470687228 (-0.04028689212817864)` |
| 90 | `0.05610861653629811 (-0.054244114812979326)` | `0.9628009853432217 (-0.9625661997393778)` | `0.01683675425671015 (-0.016678353426615285)` | `0.01516489845338531 (-0.014993893873417626)` | `0.020571792003605392 (-0.020536384714396054)` |
| 120 | `0.020477547744061624 (-0.019427422867482114)` | `0.9621027768231013 (-0.9618600299563417)` | `0.011661467693148181 (-0.011529986564084788)` | `0.009935272320361097 (-0.009671846420806006)` | `0.01886264837289756 (-0.018816821644024293)` |
| 180 | `0.01037639724123029 (-0.006795846947940658)` | `0.9573767458827928 (-0.9564766385723261)` | `0.010378124561803658 (-0.010250797500068472)` | `0.008442452779080081 (-0.008272534618895854)` | `0.022668060375761287 (-0.022589488119768207)` |
| 240 | `0.011160167145285487 (0.005286573087889906)` | `0.953342767993089 (-0.9521110481341499)` | `0.011069286135951067 (-0.010761991564181096)` | `0.008872652904114393 (-0.008436557841921087)` | `0.02746801050858822 (-0.02727453733609848)` |
| 300 | `0.013275477143297432 (0.00032192529324964747)` | `0.9489863439466562 (-0.9477762532992557)` | `0.012541826235294434 (-0.01222629326739711)` | `0.01005651909010128 (-0.009760596965886829)` | `0.03366735898473374 (-0.033488612223249566)` |
| 360 | `0.02734737495902395 (-0.016123803711180287)` | `0.9331049609896781 (-0.93198776668114)` | `0.015329095552862224 (-0.014894061319811611)` | `0.012238153821113863 (-0.01181223073401229)` | `0.041686795609700227 (-0.041493695329911665)` |

### Daily salinity only

| day | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 30 | `0.00007392567497717767 (-0.000005021360098918584)` | `0.000012460536523633654 (-6.793140427900449e-7)` | `0.000008108508428697556 (-0.000002528410402780814)` | `0.000005747698056680795 (-0.000001127619965533383)` | `0.000009189174561125694 (-0.0000023863927990593867)` |
| 60 | `0.004383600181464499 (-0.0041903003422604276)` | `0.0012040718087700368 (-0.001183948489111499)` | `0.0005596517576380386 (-0.0005391468905360559)` | `0.0002663306624857084 (-0.000189908418245087)` | `0.000040608181243720175 (-0.00002380560255008003)` |
| 90 | `0.0888654857916342 (-0.08700098406831541)` | `0.020392058891383517 (-0.020157273287539742)` | `0.0012832533064988246 (-0.0011248524764039614)` | `0.0011891634288952561 (-0.001018158848927573)` | `0.0013183955598429675 (-0.0012829882706336308)` |
| 120 | `0.3962672363567878 (-0.39521711148020827)` | `0.08741124497855378 (-0.08716849811179417)` | `0.004360629646217776 (-0.004229148517154382)` | `0.00399999628672667 (-0.003736570387171579)` | `0.004346089569972375 (-0.004300262841099109)` |
| 180 | `0.8623148395121409 (-0.8587342892188513)` | `0.12723763296516988 (-0.12633752565470324)` | `0.007509503816545171 (-0.007382176754809985)` | `0.006180078113211244 (-0.006010159953027017)` | `0.013998234863979825 (-0.013919662607986745)` |
| 240 | `1.4291755266806412 (-1.4127287864474658)` | `0.1779285778241674 (-0.17669685796522835)` | `0.01355644436458897 (-0.013249149792818998)` | `0.011693737534873672 (-0.011257642472680367)` | `0.0347744216938993 (-0.034580948521409555)` |
| 300 | `1.7568990548274999 (-1.7433016523909528)` | `0.22871006061656843 (-0.22749996996916796)` | `0.022685982891714477 (-0.022370449923817152)` | `0.017978333795978668 (-0.017682411671764216)` | `0.05607886380927202 (-0.05590011704778784)` |
| 360 | `2.0441732116852167 (-2.0329496404373733)` | `0.2946952176026373 (-0.2935780232940992)` | `0.02588366850220355 (-0.025448634269152937)` | `0.0228706256496267 (-0.02244470256252513)` | `0.08713726749375708 (-0.08694416721396851)` |

### Temperature every 2 days

| day | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 30 | `0.00011932143903323384 (-0.000050417124154974755)` | `0.000040232001289713653 (-0.000028450778808870042)` | `0.000009938185690874029 (-0.000004358087664957287)` | `0.000013028173846186444 (-0.000008408095755039031)` | `0.000007682385561055554 (-8.796037989892468e-7)` |
| 60 | `0.2872656892615269 (-0.2870723894223228)` | `0.10849938761153621 (-0.10847926429187768)` | `0.011076987475662431 (-0.011056482608560448)` | `0.009053522145447766 (-0.008977099901207144)` | `0.007728051975513135 (-0.007711249396819495)` |
| 90 | `0.42321277308407573 (-0.4213482713607569)` | `0.7776228237512633 (-0.7773880381474195)` | `0.046280831522137994 (-0.04612243069204313)` | `0.03972479806624538 (-0.0395537934862777)` | `0.035341588045417906 (-0.03530618075620857)` |
| 120 | `0.17476382259134837 (-0.17371369771476886)` | `0.9034690762610653 (-0.9032263293943057)` | `0.02109045387777813 (-0.020958972748714735)` | `0.020737802033810725 (-0.020474376134255633)` | `0.022908694154122642 (-0.022862867425249376)` |
| 180 | `0.038134652606073 (-0.03455410231278337)` | `0.931590273397437 (-0.9306901660869704)` | `0.010222777006023535 (-0.010095449944288348)` | `0.009600946253439482 (-0.009431028093255255)` | `0.022173006849860715 (-0.022094434593867635)` |
| 240 | `0.02033656627544235 (-0.0038898260422669563)` | `0.9295309225471741 (-0.928299202688235)` | `0.010346037901957445 (-0.010038743330187474)` | `0.008477468540889262 (-0.008041373478695957)` | `0.026682570962197828 (-0.026489097789708087)` |
| 300 | `0.021356308707171163 (-0.007758906270624083)` | `0.9254997362035201 (-0.9242896455561197)` | `0.011832044032376554 (-0.011516511064479231)` | `0.009015820963315135 (-0.008719898839100683)` | `0.0327232859695937 (-0.03254453920810953)` |
| 360 | `0.03599630859442802 (-0.024772737346584355)` | `0.9100936547996038 (-0.9089764604910657)` | `0.015025944532975762 (-0.01459091029992515)` | `0.010886568443996342 (-0.01046064535689477)` | `0.04088619182635236 (-0.040693091546563795)` |

### Temperature every 4 days

| day | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 30 | `0.00002102840917345305 (0.00004787590570480604)` | `0.000010483973868040084 (0.0000012972486128035257)` | `0.00000433387474989351 (0.0000012462232760232318)` | `0.0000040678792474491005 (5.521988436983115e-7)` | `0.000004179582734993105 (0.000002623199027073203)` |
| 60 | `0.00038409155636560146 (-0.00019079171716152946)` | `0.00012748834007148148 (-0.00010736502041294355)` | `0.000033166360817044246 (-0.000012661493715061573)` | `0.00002867857773424327 (0.00004774366650637812)` | `0.000025041003789727886 (-0.000008238425096087743)` |
| 90 | `0.08190336111168575 (-0.08003885938836697)` | `0.02906793144425079 (-0.028833145840407016)` | `0.0027519778552259223 (-0.002593577025131059)` | `0.0021098629071199548 (-0.0019388583271522715)` | `0.0020865708119431913 (-0.002051163522733855)` |
| 120 | `0.44424672015692906 (-0.44319659528034955)` | `0.23055025330117357 (-0.23030750643441394)` | `0.02128324110909052 (-0.021151759980027125)` | `0.015543282612105358 (-0.015279856712550267)` | `0.024818706225730847 (-0.02477287949685758)` |
| 180 | `0.39813379859932907 (-0.39455324830603944)` | `0.747471718843848 (-0.7465716115333814)` | `0.03762460437763064 (-0.03749727731589546)` | `0.029806679328562465 (-0.029636761168378236)` | `0.043337606510919995 (-0.043259034254926915)` |
| 240 | `0.17212630812249602 (-0.15567956788932064)` | `0.8465816263132259 (-0.8453499064542868)` | `0.02613086794442408 (-0.025823573372654108)` | `0.019962428555168547 (-0.01952633349297524)` | `0.0451300241948876 (-0.04493655102239785)` |
| 300 | `0.08967103506661002 (-0.07607363263006293)` | `0.8618563855276985 (-0.8606462948802981)` | `0.021733132828621556 (-0.02141759986072423)` | `0.01651992328123317 (-0.01622400115701872)` | `0.051141800420781335 (-0.05096305365929716)` |
| 360 | `0.07429437699336107 (-0.0630708057455174)` | `0.8546993761384043 (-0.8535821818298662)` | `0.021152434581088543 (-0.02071740034803793)` | `0.01627017822972303 (-0.015844255142621458)` | `0.05700025667155184 (-0.05680715639176328)` |

### Temperature every 8 days

| day | T | S | u | v | ssh |
|---:|---:|---:|---:|---:|---:|
| 30 | `0.00003273796266175568 (0.00003616635221650341)` | `0.000007894267780776463 (0.000003886954700067147)` | `8.138608892855983e-7 (0.000004766237136631144)` | `9.566242246560061e-7 (0.000003663453866491406)` | `8.942124444464895e-7 (0.000005908569317619818)` |
| 60 | `0.00007157487120388416 (0.00012172496800018785)` | `0.000030548257263651727 (-0.000010424937605113799)` | `0.000033458214869766365 (-0.000012953347767783693)` | `0.00005655775694716509 (0.000019864487293456296)` | `0.000011265179316413899 (0.0000055373993772262445)` |
| 90 | `0.006233169774154219 (-0.004368668050835434)` | `0.001833796368512543 (-0.001599010764668769)` | `0.0002728265021628902 (-0.00011442567206802686)` | `0.000128793693960746 (0.000042210886006937214)` | `0.00005613878202747437 (-0.000020731492818137673)` |
| 120 | `0.020608486616424632 (-0.019558361739845122)` | `0.0069249278173704925 (-0.00668218095061088)` | `0.0003976473160750065 (-0.00026616618701161314)` | `0.0004867829361595921 (-0.0002233570366045012)` | `0.00026685085082764196 (-0.00022102412195437564)` |
| 180 | `0.21180132534658191 (-0.20822077505329228)` | `0.1012422612159232 (-0.10034215390545656)` | `0.008822332060066194 (-0.008695004998331007)` | `0.008044606525249652 (-0.007874688365065425)` | `0.007370292179468791 (-0.007291719923475711)` |
| 240 | `0.44102754481466183 (-0.42458080458148645)` | `0.26486937021326673 (-0.2636376503543277)` | `0.018955760826052038 (-0.018648466254282067)` | `0.014946759159764502 (-0.014510664097571197)` | `0.03517009279186543 (-0.03497661961937568)` |
| 300 | `0.5397754766420735 (-0.5261780742055264)` | `0.49860251691821394 (-0.4973924262708135)` | `0.048189034859322064 (-0.04787350189142474)` | `0.06071667677796052 (-0.06042075465374607)` | `0.0664626754162446 (-0.06628392865476042)` |
| 360 | `0.3732868948504476 (-0.3620633236026039)` | `0.6214930758901185 (-0.6203758815815804)` | `0.0379086371660922 (-0.03747360293304158)` | `0.032462349177865994 (-0.032036426090764424)` | `0.06435990715142054 (-0.06416680687163197)` |

## Birth localization and first-statement boundary

For the measured singleton winner, the first western-upper-100-m T-gap
reduction occurs at day 2. In that intersection the free gap is
`2.1276338868048135e-6 K`, the reset gap is
`2.1103739081116893e-6 K`, and the reduction is
`1.7259978693124212e-8 K`. Across the whole wet domain, the day-2 reset
response has 21,120 unequal cells and `5.849752515667283e-7 K` RMS. The
upper 100 m carries 64.8603% of summed `dT^2`; the western third carries
49.9401%. The peak is `-9.437160972680658e-6 K` at `(j=2,i=30,k=4)`.

No first non-bit statement is named. The measured birth interval is the six
production steps after the day-1 boundary and through the day-2 snapshot
(steps 7–12). The daily oracle supplies the two endpoint states, not
compiled-order process operands inside that interval. The existing stage
records cover the first two steps, while the long vertical decomposition
record covers steps 1081–1440; neither covers steps 7–12. Moreover, the
singleton split itself is a destructive, non-additive intervention. Naming an
operator from it would violate the preregistered ownership caveat.

## Controls and provenance

The record audit is `daily_record_audit.json`, SHA-256
`e2f8d6b4e20c12537e796440b374bef39c0c46e214905aeb1c00bb9882e0904e`.
Every arm ran from clean measurement commit
`b4cf0795769951ba1c49c37a5a734615069bdb82` under fp64/libm production JIT.

| arm | resets | manifest SHA-256 | wall time |
|---|---:|---|---:|
| control | 0 | `1100409c9aa8e39fd04a47d730232e2132ed7c60d8ea833a5bd4393427078955` | `1692.28 s` |
| temperature daily | 359 | `36b3f31b0f05fdc662645c0bb5358c6f005cafe86efb7c87221a1e98e76cdeda` | `1763.63 s` |
| salinity daily | 359 | `21107a54b2aef9516e20d2cc75da40f6f5dd4986eb656789233666cd4df77da7` | `1756.92 s` |
| temperature 2-day | 179 | `a7885ec159531e2c9e600f3eaa3d98feb99b4a109168a9d421dc3c7bd569a4c4` | `1716.52 s` |
| temperature 4-day | 89 | `ef87d15d7553052f2bb2b300dfd0704e96e39a475d782fb7fd93d4cd6799cfc8` | `1760.20 s` |
| temperature 8-day | 44 | `795b6470703df743ba339bebd1ca24ac3907ec73a490504eced8dc005429aa6b` | `1749.90 s` |

The temperature-only boundary unit test proves only `T` is replaced; the
salinity-only test proves only `S` is replaced. Omitted diagnostics and an
explicit `None` remain bit-identical through the production step. The
following plants each exited nonzero and printed `STATUS PLANT-FIRED`:

| planted violation | decisive refusal |
|---|---|
| missing oracle boundary | `missing-boundary` |
| missing required oracle variable | `required-variable` |
| one-ULP monthly overlap | `monthly-overlap-ulp` |
| one-ULP reset source | `daily-source-ulp` |
| family registry mutation | `daily-family-registry` |
| invalid cadence registry | `daily-cadence-registry` |
| 79/80 subfamily score rows | `daily-subfamily-registry` |
| 159/160 cadence score rows | `daily-cadence-score-registry` |

## Trajectory and cross-card disposition

This round adds only private, default-off diagnostic modes to the existing
year harness and scorer. The normal GYRE trajectory is certified by the
360-day, five-field bit-identical control. Consequently no ladder row moves:
kt2 T/S remain `1.4210854715202004e-14` /
`2.1316282072803006e-14`, kt2 U/V remain
`2.7377110452773967e-12` / `3.2849219221489645e-12`, and kt3 T/S remain
`8.659373840202989e-7` / `7.027291104577671e-8`. First-over-bar remains kt2
U/V, and no immutable before arm changes.

| campaign surface | disposition |
|---|---|
| GYRE | normal 360-day trajectory bit-identical; intervention rows diagnostic only |
| DINO | default-off diagnostic seam; no card or normal state path changed |
| LOCK_EXCHANGE / OVERFLOW / tanks | no private reset source supplied; statement does not execute |
| ORCA2 | UNMEASURED-WITH-SPEC; card, selector, and one-category SI3 scope untouched |
| production physics/configuration/carried state | none landed |

## Independent adversarial review

The required separate command was invoked from clean draft-receipt commit
`436ffd83545fd3345c4222125929121cb76cb858` with `codex exec --sandbox
read-only -C` and a prompt that tried to refute record reuse, the control,
one-variable isolation, cadence counts, every registered row, non-additive
interpretation, birth localization, plants, the no-production-change table,
and cross-card scope. It returned exit 1 before reading the diff. Its complete
Codex output, verbatim, was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus **independent review unavailable in-sandbox**. The retained log is
`phase3/round135/codex_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
It issued no verdict and therefore no `DO NOT SHIP` verdict. This receipt does
not mislabel the infrastructure failure as an approving review.

## Verification

All commands used the required CPU/fp64 environment. The clean focused suite
covering the Round-135 instrument, both shared year tools, the oracle-record
gate, and the spread-floor digest ratchet reported exactly:

```text
80 passed in 59.99s
```

The clean citation gate found three citations, zero unmapped citations, zero
failures, an empty whole-map audit, and `status: PASS`. Its JSON is
`phase3/round135/citation_gate.json`, SHA-256
`dc9e51b6fd0a79e7267aa78f5b3160dac4f0e0dda24311dffbd82fe4ce68b4f2`.
Shifting the accepted-state citation's first endpoint by two lines exited 1,
reported `SYMBOL-NOT-AT-LINE`, and produced
`citation_gate_shifted_plant.json`, SHA-256
`f15c825d4bee55470326663a23e95d7284e2d22eb757b9b61d77d2638bb1716a`.
The focused log SHA-256 is
`5094b4b6cc76bb5c82631baf54f785ebe22dc9ed66baf6a4cd8c65367412351a`.

The clean post-review citation/ratchet suite reported exactly:

```text
30 passed in 17.38s
```

It covers the complete citation-gate regression file, all Round-135 tests,
and the certified spread-floor harness pin. After the broad attempt described
next exhausted compiler workers, the same affected suite was run once more in
a fresh process and reported:

```text
30 passed in 17.37s
```

The logs have SHA-256 digests
`5c74d281ec45991f4c1eb61298e3e2854dea8aa540c0e69c6748e32586316a0d`
and
`59dd3e2f444e2627cf478b2b526d94920dbeaeeb92fb4e766c0b2a11ef0cc22d`,
respectively.

The requested combined `tests/ocean/fidelity tests/ocean/unit -n 12` run was
attempted from the clean post-review commit. It collected 8,353 tests and
reached 93%, but nine JAX processes aborted while compiling and xdist reported
eight workers as “Not properly terminated.” The controller then ceased making
progress and was interrupted. There is no pytest summary line, so the run is
**INCOMPLETE** and is not represented as either a pass or a stable failure
set. In particular, the partial failure IDs cannot honestly be diffed against
the inherited-red list. The retained log is `phase3/round135/full_ocean_tests.log`,
SHA-256
`9007eaf0da1d8759d8b96b33aa0bbe5e0c02e182df97db464e5741e7f572fb04`.
The two completed affected suites above are green before and after this
compiler exhaustion.

## OPEN — next round

1. Treat temperature and salinity as a coupled reset family; do not infer an
   independent T or S source owner from these singleton arms. The first useful
   discriminator must preserve the paired T/S state.
2. Preregister a six-step, day-1-to-day-2 compiled-order tracer-source
   decomposition driven from the admitted day-1 NEMO state. First inventory
   whether any committed oracle record contains the internal operands for
   steps 7–12. If not, specify the minimum passive NEMO record before writing
   an acquisition script; do not substitute endpoints for missing operands.
3. Score each paired-T/S source-family substitution in the western-third,
   upper-100-m intersection first, then the whole wet domain. Name a first
   non-bit statement only when one compiled-order boundary closes under
   production JIT and its plant fires.
4. Keep the complete signed singleton and cadence tables as controls. In
   particular, do not smooth away the 2/4/8-day worsenings or call the
   day-240-only temperature improvement a year improvement.
5. No new card, selector, stabilizer, or carried state is authorized. The
   pending five-category ORCA2/SI3 decision remains outside this GYRE
   diagnostic round.

No current NEMO acquisition and no new user configuration decision are
requested. Production remains on the incoming physical trajectory.
