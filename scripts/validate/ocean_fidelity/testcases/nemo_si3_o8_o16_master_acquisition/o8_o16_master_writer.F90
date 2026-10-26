MODULE o8_o16_master_writer
   USE, INTRINSIC :: iso_fortran_env, ONLY: int32, real64
   USE par_oce, ONLY: wp, jpi, jpj, nn_hls, ntsi, ntei, ntsj, ntej
   USE par_ice, ONLY: jpl, nlay_i, nlay_s
   USE ice
   USE ice1D
   USE sbc_oce, ONLY: qsr, qns, emp, sfx, fr_i, fwfice, qns_tot, qsr_tot, frq_m
   USE sbc_ice, ONLY: qsr_oce, qns_oce, qemp_oce, emp_oce, emp_ice, qsr_ice, qns_ice, qevap_ice, &
      &               tn_ice, alb_ice, snwice_mass, snwice_mass_b, snwice_fmass
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o8o16_set_context, o8o16_frazil, o8o16_category, o8o16_picard
   PUBLIC :: o8o16_global, o8o16_update

   INTEGER :: current_kt=-1, current_kmm=-1, current_nit000=-1, current_rank=-1, current_jl=-1
   INTEGER :: current_jpni=-1, current_jpnj=-1
   REAL(wp) :: current_rdt=0._wp, current_r1dt=0._wp
   REAL(wp) :: current_beta=0._wp, current_dmin=0._wp, current_hinew=0._wp
   REAL(wp) :: current_picard=0._wp

CONTAINS

   SUBROUTINE o8o16_set_context(kt,kmm,nit000,rank,jpni,jpnj,rdt,r1dt)
      INTEGER, INTENT(in) :: kt,kmm,nit000,rank,jpni,jpnj
      REAL(wp), INTENT(in) :: rdt,r1dt
      current_kt=kt ; current_kmm=kmm ; current_nit000=nit000 ; current_rank=rank
      current_jpni=jpni ; current_jpnj=jpnj ; current_rdt=rdt ; current_r1dt=r1dt
      current_picard=0._wp
   END SUBROUTINE o8o16_set_context

   SUBROUTINE open_record(kind,jl,npti_local,nfields,u)
      INTEGER, INTENT(in) :: kind,jl,npti_local,nfields
      INTEGER, INTENT(out) :: u
      CHARACTER(LEN=8), PARAMETER :: magic='O816M001'
      CHARACTER(LEN=192) :: path,label,suffix
      INTEGER(int32), DIMENSION(20) :: meta
      REAL(real64), DIMENSION(6) :: scalars
      SELECT CASE(kind)
      CASE(1); label='frazil_entry'
      CASE(2); label='frazil_exit'
      CASE(3); label='category_entry'
      CASE(4); label='zdf_exit'
      CASE(5); label='dh_exit'
      CASE(6); label='temp1_exit_sal_entry'
      CASE(7); label='sal_exit'
      CASE(8); label='temp2_exit_da_entry'
      CASE(9); label='da_exit'
      CASE(10); label='pond_entry_category_done'
      CASE(11); label='pond_exit_itd_entry'
      CASE(12); label='itd_exit_do_entry'
      CASE(13); label='do_exit_cor_entry'
      CASE(14); label='cor_exit'
      CASE(15); label='update_flx_entry'
      CASE(16); label='update_flx_exit_ocean_boundary'
      CASE(17); label='picard_final'
      CASE DEFAULT; ERROR STOP 'O8-O16 unknown record kind'
      END SELECT
      IF(jl > 0) THEN
         WRITE(suffix,'("_cat",I2.2)') jl
      ELSE
         suffix=''
      ENDIF
      WRITE(path,'("o8_o16_master_records/o8o16_kind",I2.2,"_",A,A,"_step",I4.4,"_rank",I4.4,".bin")') &
         kind,TRIM(label),TRIM(suffix),current_kt,current_rank
      OPEN(NEWUNIT=u,FILE=TRIM(path),ACCESS='stream',FORM='unformatted',STATUS='replace',ACTION='write')
      meta=[kind,current_kt,jl,current_rank,jpi,jpj,jpl,nlay_i,nlay_s,ntsi,ntei,ntsj,ntej, &
            npti_local,nfields,1,current_nit000,current_kmm,1,1]
      scalars=[current_rdt,current_r1dt,current_beta,current_dmin,current_hinew,current_picard]
      WRITE(u) magic ; WRITE(u) INT(1,int32) ; WRITE(u) meta ; WRITE(u) scalars
   END SUBROUTINE open_record

   SUBROUTINE put_i2(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      INTEGER::ni,nj
      ni=ntei-ntsi+1 ; nj=ntej-ntsj+1
      IF(SIZE(a,1)==ni .AND. SIZE(a,2)==nj) THEN
         WRITE(u) a
      ELSEIF(SIZE(a,1)==jpi .AND. SIZE(a,2)==jpj) THEN
         WRITE(u) a(ntsi:ntei,ntsj:ntej)
      ELSE
         ERROR STOP 'O8-O16 I2 extent mismatch'
      ENDIF
   END SUBROUTINE put_i2

   SUBROUTINE put_f2(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi .OR. SIZE(a,2)/=jpj) ERROR STOP 'O8-O16 F2 extent mismatch'
      WRITE(u) a
   END SUBROUTINE put_f2

   SUBROUTINE put_i3(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:),INTENT(in)::a
      IF(SIZE(a,3)/=jpl) ERROR STOP 'O8-O16 I3 category mismatch'
      IF(SIZE(a,1)==jpi .AND. SIZE(a,2)==jpj) THEN
         WRITE(u) a(ntsi:ntei,ntsj:ntej,1:jpl)
      ELSEIF(SIZE(a,1)==ntei-ntsi+1 .AND. SIZE(a,2)==ntej-ntsj+1) THEN
         WRITE(u) a
      ELSE
         ERROR STOP 'O8-O16 I3 extent mismatch'
      ENDIF
   END SUBROUTINE put_i3

   SUBROUTINE put_ii4(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi .OR. SIZE(a,2)/=jpj .OR. SIZE(a,3)/=nlay_i .OR. SIZE(a,4)/=jpl) &
         ERROR STOP 'O8-O16 II4 extent mismatch'
      WRITE(u) a(ntsi:ntei,ntsj:ntej,1:nlay_i,1:jpl)
   END SUBROUTINE put_ii4

   SUBROUTINE put_is4(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:,:,:),INTENT(in)::a
      IF(SIZE(a,1)/=jpi .OR. SIZE(a,2)/=jpj .OR. SIZE(a,3)/=nlay_s .OR. SIZE(a,4)/=jpl) &
         ERROR STOP 'O8-O16 IS4 extent mismatch'
      WRITE(u) a(ntsi:ntei,ntsj:ntej,1:nlay_s,1:jpl)
   END SUBROUTINE put_is4

   SUBROUTINE put_g1(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:),INTENT(in)::a
      IF(SIZE(a)<npti) ERROR STOP 'O8-O16 G1 extent mismatch'
      WRITE(u) a(1:npti)
   END SUBROUTINE put_g1

   SUBROUTINE put_gi(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)<npti .OR. SIZE(a,2)/=nlay_i) ERROR STOP 'O8-O16 GI extent mismatch'
      WRITE(u) a(1:npti,1:nlay_i)
   END SUBROUTINE put_gi

   SUBROUTINE put_gs(u,a)
      INTEGER,INTENT(in)::u
      REAL(wp),DIMENSION(:,:),INTENT(in)::a
      IF(SIZE(a,1)<npti .OR. SIZE(a,2)/=nlay_s) ERROR STOP 'O8-O16 GS extent mismatch'
      WRITE(u) a(1:npti,1:nlay_s)
   END SUBROUTINE put_gs

   SUBROUTINE put_idx(u)
      INTEGER,INTENT(in)::u
      WRITE(u) INT(nptidx(1:npti),int32)
   END SUBROUTINE put_idx

   SUBROUTINE o8o16_frazil(kind,rn_hinew)
      INTEGER,INTENT(in)::kind
      REAL(wp),INTENT(in)::rn_hinew
      INTEGER::u
      current_hinew=rn_hinew
      IF(kind==1) THEN
         CALL open_record(kind,0,0,3,u)
         CALL put_i2(u,qlead); CALL put_i2(u,ht_i_new); CALL put_i2(u,fraz_frac)
      ELSEIF(kind==2) THEN
         CALL open_record(kind,0,0,2,u)
         CALL put_i2(u,ht_i_new); CALL put_i2(u,fraz_frac)
      ELSE
         ERROR STOP 'O8 frazil kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE o8o16_frazil

   SUBROUTINE put_cat_base(u)
      INTEGER,INTENT(in)::u
      CALL put_g1(u,a_i_1d); CALL put_g1(u,at_i_1d); CALL put_g1(u,h_i_1d); CALL put_g1(u,h_s_1d)
      CALL put_g1(u,t_su_1d); CALL put_g1(u,s_i_1d); CALL put_g1(u,t_bo_1d); CALL put_g1(u,sss_1d)
      CALL put_g1(u,sst_1d); CALL put_g1(u,qns_ice_1d); CALL put_g1(u,qsr_ice_1d)
      CALL put_g1(u,dqns_ice_1d); CALL put_g1(u,qtr_ice_top_1d); CALL put_g1(u,qsb_ice_bot_1d)
      CALL put_g1(u,fhld_1d); CALL put_g1(u,frq_m_1d); CALL put_g1(u,sprecip_1d)
      CALL put_g1(u,qprec_ice_1d); CALL put_g1(u,evap_ice_1d); CALL put_g1(u,hfx_err_dif_1d)
      CALL put_g1(u,hfx_dif_1d); CALL put_g1(u,qml_ice_1d); CALL put_g1(u,sfx_bri_1d)
      CALL put_g1(u,sfx_lam_1d); CALL put_g1(u,wfx_lam_1d)
      CALL put_gi(u,t_i_1d); CALL put_gi(u,e_i_1d)
      CALL put_gi(u,sz_i_1d); CALL put_gs(u,t_s_1d); CALL put_gs(u,e_s_1d)
   END SUBROUTINE put_cat_base

   SUBROUTINE put_dh(u)
      INTEGER,INTENT(in)::u
      CALL put_g1(u,dh_s_tot); CALL put_g1(u,dh_i_sum); CALL put_g1(u,dh_i_itm)
      CALL put_g1(u,dh_i_bom); CALL put_g1(u,dh_i_bog); CALL put_g1(u,dh_i_sub)
      CALL put_g1(u,dh_s_sum); CALL put_g1(u,dh_s_itm); CALL put_g1(u,dh_snowice)
   END SUBROUTINE put_dh

   SUBROUTINE put_dh_fluxes(u)
      INTEGER,INTENT(in)::u
      CALL put_g1(u,hfx_res_1d); CALL put_g1(u,wfx_snw_sum_1d); CALL put_g1(u,hfx_spr_1d)
      CALL put_g1(u,wfx_spr_1d); CALL put_g1(u,hfx_sub_1d); CALL put_g1(u,wfx_snw_sub_1d)
      CALL put_g1(u,hfx_snw_1d); CALL put_g1(u,wfx_res_1d); CALL put_g1(u,sfx_res_1d)
      CALL put_g1(u,hfx_thd_1d); CALL put_g1(u,hfx_sum_1d); CALL put_g1(u,wfx_sum_1d)
      CALL put_g1(u,sfx_sum_1d); CALL put_g1(u,wfx_ice_sub_1d); CALL put_g1(u,sfx_sub_1d)
      CALL put_g1(u,wfx_err_sub_1d); CALL put_g1(u,hfx_bog_1d); CALL put_g1(u,wfx_bog_1d)
      CALL put_g1(u,sfx_bog_1d); CALL put_g1(u,hfx_bom_1d); CALL put_g1(u,wfx_bom_1d)
      CALL put_g1(u,sfx_bom_1d); CALL put_g1(u,sfx_sni_1d); CALL put_g1(u,wfx_sni_1d)
      CALL put_g1(u,wfx_snw_sni_1d)
   END SUBROUTINE put_dh_fluxes

   SUBROUTINE o8o16_category(kind,jl,rn_beta,rn_dmin)
      INTEGER,INTENT(in)::kind,jl
      REAL(wp),INTENT(in),OPTIONAL::rn_beta,rn_dmin
      INTEGER::u, record_jl
      record_jl=jl
      IF(jl > 0) current_jl=jl
      IF(jl == 0) record_jl=current_jl
      IF(record_jl < 1 .OR. record_jl > jpl) ERROR STOP 'O9-O11 category unavailable'
      IF(PRESENT(rn_beta)) current_beta=rn_beta
      IF(PRESENT(rn_dmin)) current_dmin=rn_dmin
      SELECT CASE(kind)
      CASE(3,5,6,8,9)
         CALL open_record(kind,record_jl,npti,65,u); CALL put_idx(u); CALL put_cat_base(u); CALL put_dh(u); CALL put_dh_fluxes(u)
      CASE(4)
         CALL open_record(kind,record_jl,npti,15,u); CALL put_idx(u)
         CALL put_g1(u,t_su_1d); CALL put_g1(u,qns_ice_1d); CALL put_g1(u,qcn_ice_top_1d)
         CALL put_g1(u,qcn_ice_bot_1d); CALL put_g1(u,qtr_ice_bot_1d); CALL put_g1(u,hfx_err_dif_1d)
         CALL put_g1(u,hfx_dif_1d); CALL put_g1(u,cnd_ice_1d); CALL put_g1(u,t1_ice_1d)
         CALL put_g1(u,t_si_1d); CALL put_gi(u,t_i_1d); CALL put_gi(u,e_i_1d)
         CALL put_gs(u,t_s_1d); CALL put_gs(u,e_s_1d)
      CASE(7)
         CALL open_record(kind,record_jl,npti,10,u); CALL put_idx(u)
         CALL put_g1(u,a_i_1d); CALL put_g1(u,h_i_1d); CALL put_g1(u,s_i_1d)
         CALL put_g1(u,sss_1d); CALL put_g1(u,sfx_bri_1d); CALL put_g1(u,sfx_res_1d)
         CALL put_gi(u,t_i_1d); CALL put_gi(u,e_i_1d); CALL put_gi(u,sz_i_1d)
      CASE DEFAULT
         ERROR STOP 'O9-O11 category kind'
      END SELECT
      CLOSE(u)
   END SUBROUTINE o8o16_category

   SUBROUTINE o8o16_picard(iconv,lconv)
      INTEGER,INTENT(in)::iconv
      LOGICAL,DIMENSION(:),INTENT(in)::lconv
      INTEGER::u
      INTEGER(int32),ALLOCATABLE::conv(:)
      IF(current_kt/=1 .AND. current_kt/=2 .AND. current_kt/=100) RETURN
      current_picard=REAL(iconv,wp)
      ALLOCATE(conv(npti)); conv=MERGE(1_int32,0_int32,lconv(1:npti))
      IF(current_jl < 1 .OR. current_jl > jpl) ERROR STOP 'O9 Picard category unavailable'
      CALL open_record(17,current_jl,npti,4,u); CALL put_idx(u); WRITE(u) conv
      CALL put_gi(u,t_i_1d); CALL put_gs(u,t_s_1d); CLOSE(u); DEALLOCATE(conv)
   END SUBROUTINE o8o16_picard

   SUBROUTINE put_state(u)
      INTEGER,INTENT(in)::u
      CALL put_i3(u,a_i); CALL put_i3(u,v_i); CALL put_i3(u,v_s); CALL put_i3(u,h_i)
      CALL put_i3(u,h_s); CALL put_i3(u,s_i); CALL put_i3(u,sv_i); CALL put_i3(u,o_i)
      CALL put_i3(u,oa_i); CALL put_i3(u,t_su); CALL put_i3(u,a_ip); CALL put_i3(u,v_ip)
      CALL put_i3(u,h_ip); CALL put_i3(u,v_il); CALL put_i3(u,h_il); CALL put_i3(u,a_ip_frac)
      CALL put_i3(u,a_ip_eff); CALL put_ii4(u,t_i); CALL put_ii4(u,e_i); CALL put_ii4(u,sz_i)
      CALL put_ii4(u,szv_i); CALL put_is4(u,t_s); CALL put_is4(u,e_s)
   END SUBROUTINE put_state

   SUBROUTINE put_totals(u)
      INTEGER,INTENT(in)::u
      CALL put_i2(u,at_i); CALL put_i2(u,ato_i); CALL put_i2(u,vt_i); CALL put_i2(u,vt_s)
      CALL put_i2(u,st_i); CALL put_i2(u,et_i); CALL put_i2(u,et_s); CALL put_i2(u,tm_i)
      CALL put_i2(u,tm_s); CALL put_i2(u,sm_i); CALL put_i2(u,tm_su); CALL put_i2(u,hm_i)
      CALL put_i2(u,hm_s); CALL put_i2(u,om_i); CALL put_i2(u,at_ip); CALL put_i2(u,at_ip_eff)
      CALL put_i2(u,hm_ip); CALL put_i2(u,vt_ip); CALL put_i2(u,hm_il); CALL put_i2(u,vt_il)
      CALL put_i2(u,vm_ibr)
   END SUBROUTINE put_totals

   SUBROUTINE put_pond(u)
      INTEGER,INTENT(in)::u
      CALL put_i3(u,dh_i_sum_2d); CALL put_i3(u,dh_s_sum_2d)
   END SUBROUTINE put_pond

   SUBROUTINE put_flux(u)
      INTEGER,INTENT(in)::u
      CALL put_i2(u,wfx_snw); CALL put_i2(u,wfx_snw_sni); CALL put_i2(u,wfx_snw_sum)
      CALL put_i2(u,wfx_pnd); CALL put_i2(u,wfx_spr); CALL put_i2(u,wfx_sub)
      CALL put_i2(u,wfx_snw_sub); CALL put_i2(u,wfx_ice_sub); CALL put_i2(u,wfx_ice)
      CALL put_i2(u,wfx_sni); CALL put_i2(u,wfx_opw); CALL put_i2(u,wfx_bog)
      CALL put_i2(u,wfx_dyn); CALL put_i2(u,wfx_bom); CALL put_i2(u,wfx_sum)
      CALL put_i2(u,wfx_lam); CALL put_i2(u,wfx_res); CALL put_i2(u,wfx_err_sub)
      CALL put_i2(u,sfx_bog); CALL put_i2(u,sfx_bom); CALL put_i2(u,sfx_lam)
      CALL put_i2(u,sfx_sum); CALL put_i2(u,sfx_sni); CALL put_i2(u,sfx_opw)
      CALL put_i2(u,sfx_bri); CALL put_i2(u,sfx_dyn); CALL put_i2(u,sfx_res)
      CALL put_i2(u,sfx_sub); CALL put_i2(u,hfx_bog); CALL put_i2(u,hfx_bom)
      CALL put_i2(u,hfx_sum); CALL put_i2(u,hfx_opw); CALL put_i2(u,hfx_dif)
      CALL put_i2(u,hfx_snw); CALL put_i2(u,hfx_err_dif); CALL put_i2(u,hfx_sub)
      CALL put_i2(u,hfx_spr); CALL put_i2(u,hfx_thd); CALL put_i2(u,hfx_dyn)
      CALL put_i2(u,hfx_res); CALL put_i2(u,qlead); CALL put_i2(u,qsb_ice_bot)
      CALL put_i2(u,fhld); CALL put_i2(u,qt_atm_oi); CALL put_i2(u,qt_oce_ai)
   END SUBROUTINE put_flux

   SUBROUTINE o8o16_global(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u
      IF(kind<10 .OR. kind>14) ERROR STOP 'O12-O15 global kind'
      CALL open_record(kind,0,0,91,u); CALL put_state(u); CALL put_totals(u); CALL put_pond(u); CALL put_flux(u)
      CLOSE(u)
   END SUBROUTINE o8o16_global

   SUBROUTINE o8o16_update(kind)
      INTEGER,INTENT(in)::kind
      INTEGER::u
      IF(kind==15) THEN
         CALL open_record(kind,0,0,104,u); CALL put_state(u); CALL put_totals(u); CALL put_flux(u)
         CALL put_i2(u,qsr_oce); CALL put_i2(u,qns_oce); CALL put_i2(u,qemp_oce)
         CALL put_i2(u,emp_oce); CALL put_i2(u,emp_ice); CALL put_i2(u,qns_tot)
         CALL put_i2(u,qsr_tot); CALL put_i2(u,frq_m); CALL put_i3(u,a_i_b)
         CALL put_i3(u,qsr_ice); CALL put_i3(u,qns_ice); CALL put_i3(u,qtr_ice_bot)
         CALL put_i3(u,qevap_ice); CALL put_f2(u,snwice_mass); CALL put_f2(u,snwice_mass_b)
      ELSEIF(kind==16) THEN
         CALL open_record(kind,0,0,16,u)
         CALL put_i2(u,qsr); CALL put_i2(u,qns); CALL put_i2(u,emp); CALL put_i2(u,sfx)
         CALL put_i2(u,fr_i); CALL put_i2(u,fwfice); CALL put_i2(u,qt_atm_oi); CALL put_i2(u,qt_oce_ai)
         CALL put_i2(u,wfx_ice); CALL put_i2(u,wfx_snw); CALL put_i2(u,wfx_pnd)
         CALL put_i3(u,tn_ice); CALL put_i3(u,alb_ice)
         CALL put_f2(u,snwice_mass); CALL put_f2(u,snwice_mass_b); CALL put_f2(u,snwice_fmass)
      ELSE
         ERROR STOP 'O16 update kind'
      ENDIF
      CLOSE(u)
   END SUBROUTINE o8o16_update

END MODULE o8_o16_master_writer
