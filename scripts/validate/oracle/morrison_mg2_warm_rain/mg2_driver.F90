! Standalone driver for CAM cam_cesm2_1_rel_60 micro_mg2_0.F90 micro_mg_tend
! (compiled UNMODIFIED with micro_mg_utils.F90).  One column per call, so the
! specified droplet number (nccons, ncnst) can follow each column.
! CAM6 + CLUBB settings: micro_mg_precip_frac_method='in_cloud'
! (namelist_defaults_cam.xml:1248), allow_sed_supersat=.false. under CLUBB
! (micro_mg_cam.F90:668-672), do_sb_physics=.false., berg_eff=1, dcs=500e-6
! (mg2, :1244), do_cldice=.true., use_hetfrz_classnuc=.true. (frzimm/cnt/dep=0),
! microp_uniform from argv(3).  Constants = legoESM's (constants.py).
! usage: mg2_driver in.txt out.txt [microp_uniform T|F]
program mg2_driver
  use micro_mg_utils, only: r8
  use micro_mg2_0, only: micro_mg_init, micro_mg_tend
  use wv_sat_methods, only: wv_sat_methods_init
  implicit none
  real(r8), parameter :: g = 9.80616_r8, rair = 287.05_r8, rh2o = 461.51_r8
  real(r8), parameter :: cpair = 1004.64_r8, tmelt = 273.15_r8
  real(r8), parameter :: latvap = 2.501e6_r8, latice = 3.337e5_r8
  integer :: ncol, nlev, ic, k, m
  real(r8) :: dt, ncnst
  character(len=512) :: fin, fout, aflag
  logical :: uni
  character(128) :: err
  character(len=16) :: pfm = 'in_cloud'
  real(r8), allocatable :: inp(:,:)
  real(r8), allocatable, dimension(:,:) :: t,q,qc,qi,nc,ni,qr,qs,nr,ns,relvar,accre,p,pdel, &
       cldn,liqcldf,icecldf,qsatfac,naai,npccn,rate1,tlat,qvlat,qctend,qitend,nctend,nitend, &
       qrtend,qstend,nrtend,nstend,effc,effc_fn,effi,sadice,sadsnow,nevapr,evapsnow,am_evp_st, &
       prain,prodsnow,cmeout,deffi,pgamrad,lamcrad,qsout,dsout,lflx,iflx,rflx,sflx,qrout, &
       reff_rain,reff_snow,qcsevap,qisevap,qvres,cmeitot,vtrmc,vtrmi,umr,ums,qcsedten,qisedten, &
       qrsedten,qssedten,pratot,prctot,mnuccctot,mnuccttot,msacwitot,psacwstot,bergstot,bergtot, &
       melttot,homotot,qcrestot,prcitot,praitot,qirestot,mnuccrtot,pracstot,meltsdttot,frzrdttot, &
       mnuccdtot,nrout,nsout,refl,arefl,areflz,frefl,csrfl,acsrfl,fcsrfl,rercld,ncai,ncal, &
       qrout2,qsout2,nrout2,nsout2,drout2,dsout2,freqs,freqr,nfice,qcrat,prer_evap, &
       tnd_qsnow,tnd_nsnow,re_ice,frzimm,frzcnt,frzdep
  real(r8), allocatable :: rndst(:,:,:), nacon(:,:,:)
  real(r8) :: prect(1), preci(1)
  real(r8), allocatable :: allout(:,:,:), allprect(:)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, aflag)
  uni = (trim(aflag) == 'T')
  call wv_sat_methods_init(r8, tmelt, 273.16_r8, 373.16_r8, 20._r8, rair/rh2o, err)
  open(10, file=trim(fin), status='old')
  read(10, *) ncol, nlev, dt
  allocate(inp(nlev, 18), allout(ncol, nlev, 15), allprect(ncol))
  call alloc2(nlev)
  allocate(rndst(1,nlev,4), nacon(1,nlev,4)); rndst = 1.e-6_r8; nacon = 0._r8
  do ic = 1, ncol
     read(10, *) ncnst
     do k = 1, nlev
        read(10, *) (inp(k, m), m = 1, 18)
     end do
     t(1,:)=inp(:,1); q(1,:)=inp(:,2); qc(1,:)=inp(:,3); qi(1,:)=inp(:,4)
     nc(1,:)=inp(:,5); ni(1,:)=inp(:,6); qr(1,:)=inp(:,7); qs(1,:)=inp(:,8)
     nr(1,:)=inp(:,9); ns(1,:)=inp(:,10); relvar(1,:)=inp(:,11); accre(1,:)=inp(:,12)
     p(1,:)=inp(:,13); pdel(1,:)=inp(:,14); cldn(1,:)=inp(:,15); liqcldf(1,:)=inp(:,16)
     icecldf(1,:)=inp(:,17); qsatfac(1,:)=inp(:,18)
     naai = 0._r8; npccn = 0._r8
     tnd_qsnow = 0._r8; tnd_nsnow = 0._r8; re_ice = 0._r8
     frzimm = 0._r8; frzcnt = 0._r8; frzdep = 0._r8
     call micro_mg_init(kind=r8, gravit=g, rair=rair, rh2o=rh2o, cpair=cpair, &
          tmelt_in=tmelt, latvap=latvap, latice=latice, rhmini_in=0.8_r8, &
          micro_mg_dcs=500.e-6_r8, microp_uniform_in=uni, do_cldice_in=.true., &
          use_hetfrz_classnuc_in=.true., micro_mg_precip_frac_method_in=pfm, &
          micro_mg_berg_eff_factor_in=1._r8, allow_sed_supersat_in=.false., &
          do_sb_physics_in=.false., nccons_in=.true., nicons_in=.false., &
          ncnst_in=ncnst, ninst_in=0._r8, errstring=err)
     if (len_trim(err) > 0) then
        print *, 'INIT ERROR: ', trim(err); stop 2
     end if
     call micro_mg_tend(mgncol=1, nlev=nlev, deltatin=dt, t=t, q=q, qcn=qc, qin=qi, &
          ncn=nc, nin=ni, qrn=qr, qsn=qs, nrn=nr, nsn=ns, relvar=relvar, accre_enhan=accre, &
          p=p, pdel=pdel, cldn=cldn, liqcldf=liqcldf, icecldf=icecldf, qsatfac=qsatfac, &
          qcsinksum_rate1ord=rate1, naai=naai, npccn=npccn, rndst=rndst, nacon=nacon, &
          tlat=tlat, qvlat=qvlat, qctend=qctend, qitend=qitend, nctend=nctend, nitend=nitend, &
          qrtend=qrtend, qstend=qstend, nrtend=nrtend, nstend=nstend, effc=effc, &
          effc_fn=effc_fn, effi=effi, sadice=sadice, sadsnow=sadsnow, prect=prect, preci=preci, &
          nevapr=nevapr, evapsnow=evapsnow, am_evp_st=am_evp_st, prain=prain, &
          prodsnow=prodsnow, cmeout=cmeout, deffi=deffi, pgamrad=pgamrad, lamcrad=lamcrad, &
          qsout=qsout, dsout=dsout, lflx=lflx, iflx=iflx, rflx=rflx, sflx=sflx, qrout=qrout, &
          reff_rain=reff_rain, reff_snow=reff_snow, qcsevap=qcsevap, qisevap=qisevap, &
          qvres=qvres, cmeitot=cmeitot, vtrmc=vtrmc, vtrmi=vtrmi, umr=umr, ums=ums, &
          qcsedten=qcsedten, qisedten=qisedten, qrsedten=qrsedten, qssedten=qssedten, &
          pratot=pratot, prctot=prctot, mnuccctot=mnuccctot, mnuccttot=mnuccttot, &
          msacwitot=msacwitot, psacwstot=psacwstot, bergstot=bergstot, bergtot=bergtot, &
          melttot=melttot, homotot=homotot, qcrestot=qcrestot, prcitot=prcitot, &
          praitot=praitot, qirestot=qirestot, mnuccrtot=mnuccrtot, pracstot=pracstot, &
          meltsdttot=meltsdttot, frzrdttot=frzrdttot, mnuccdtot=mnuccdtot, nrout=nrout, &
          nsout=nsout, refl=refl, arefl=arefl, areflz=areflz, frefl=frefl, csrfl=csrfl, &
          acsrfl=acsrfl, fcsrfl=fcsrfl, rercld=rercld, ncai=ncai, ncal=ncal, qrout2=qrout2, &
          qsout2=qsout2, nrout2=nrout2, nsout2=nsout2, drout2=drout2, dsout2=dsout2, &
          freqs=freqs, freqr=freqr, nfice=nfice, qcrat=qcrat, errstring=err, &
          tnd_qsnow=tnd_qsnow, tnd_nsnow=tnd_nsnow, re_ice=re_ice, prer_evap=prer_evap, &
          frzimm=frzimm, frzcnt=frzcnt, frzdep=frzdep)
     if (len_trim(err) > 0) then
        print *, 'TEND ERROR: ', trim(err); stop 3
     end if
     allprect(ic) = prect(1)
     do k = 1, nlev
        allout(ic,k,:) = [tlat(1,k), qvlat(1,k), qctend(1,k), qrtend(1,k), nctend(1,k), &
             nrtend(1,k), nevapr(1,k), pratot(1,k), prctot(1,k), qrsedten(1,k), &
             qcsedten(1,k), umr(1,k), qcrat(1,k), rflx(1,k+1), lflx(1,k+1)]
     end do
  end do
  close(10)
  open(11, file=trim(fout), status='replace')
  do ic = 1, ncol
     write(11, '(i4,1x,es25.16e3)') ic, allprect(ic)
  end do
  do ic = 1, ncol
     do k = 1, nlev
        write(11, '(2i4,15(1x,es25.16e3))') ic, k, allout(ic,k,:)
     end do
  end do
  close(11)
  print *, 'MG2_DRIVER_OK ncol=', ncol, ' nlev=', nlev, ' uniform=', uni
contains
  subroutine alloc2(n)
    integer, intent(in) :: n
    allocate(t(1,n),q(1,n),qc(1,n),qi(1,n),nc(1,n),ni(1,n),qr(1,n),qs(1,n),nr(1,n),ns(1,n), &
         relvar(1,n),accre(1,n),p(1,n),pdel(1,n),cldn(1,n),liqcldf(1,n),icecldf(1,n), &
         qsatfac(1,n),naai(1,n),npccn(1,n),rate1(1,n),tlat(1,n),qvlat(1,n),qctend(1,n), &
         qitend(1,n),nctend(1,n),nitend(1,n),qrtend(1,n),qstend(1,n),nrtend(1,n),nstend(1,n), &
         effc(1,n),effc_fn(1,n),effi(1,n),sadice(1,n),sadsnow(1,n),nevapr(1,n),evapsnow(1,n), &
         am_evp_st(1,n),prain(1,n),prodsnow(1,n),cmeout(1,n),deffi(1,n),pgamrad(1,n), &
         lamcrad(1,n),qsout(1,n),dsout(1,n),lflx(1,n+1),iflx(1,n+1),rflx(1,n+1),sflx(1,n+1), &
         qrout(1,n),reff_rain(1,n),reff_snow(1,n),qcsevap(1,n),qisevap(1,n),qvres(1,n), &
         cmeitot(1,n),vtrmc(1,n),vtrmi(1,n),umr(1,n),ums(1,n),qcsedten(1,n),qisedten(1,n), &
         qrsedten(1,n),qssedten(1,n),pratot(1,n),prctot(1,n),mnuccctot(1,n),mnuccttot(1,n), &
         msacwitot(1,n),psacwstot(1,n),bergstot(1,n),bergtot(1,n),melttot(1,n),homotot(1,n), &
         qcrestot(1,n),prcitot(1,n),praitot(1,n),qirestot(1,n),mnuccrtot(1,n),pracstot(1,n), &
         meltsdttot(1,n),frzrdttot(1,n),mnuccdtot(1,n),nrout(1,n),nsout(1,n),refl(1,n), &
         arefl(1,n),areflz(1,n),frefl(1,n),csrfl(1,n),acsrfl(1,n),fcsrfl(1,n),rercld(1,n), &
         ncai(1,n),ncal(1,n),qrout2(1,n),qsout2(1,n),nrout2(1,n),nsout2(1,n),drout2(1,n), &
         dsout2(1,n),freqs(1,n),freqr(1,n),nfice(1,n),qcrat(1,n),prer_evap(1,n), &
         tnd_qsnow(1,n),tnd_nsnow(1,n),re_ice(1,n),frzimm(1,n),frzcnt(1,n),frzdep(1,n))
  end subroutine alloc2
end program mg2_driver
