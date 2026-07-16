load "~/Models/NCL/open_wks.ncl"
load "~/Models/NCL/plotXYmarker.ncl"

begin

wks = open_wks("NCGM")
;wks = open_wks("X11")
;wks = open_wks("PS")

nens=0
nfiles_ens=20       ; number of files in the ensemble
nfiles_dat=1

nfiles=nens*nfiles_ens+nfiles_dat

filename=new((/100/),string)
filename_root=new((/100/),string)
file_in=new((/100/),file)

;====================================================================

filename_root(1)="."

filename(1) = "TOGA_32x32x128_4000m.evol"
;====================================================================
plot_ens = "False"

filename_obs="toga_coare.nc"
day0=353.75

time_scale = 6.

opt=0

opt@case_names = (/"1","Observations"/)
;opt@case_names = (/"128x128x64_2km","64x24_4km(2D-SP)","64x24_4km_precipfall(2D-SP)","Observations"/)

;opt@case_names = (/"128x128x64_2km","64x24_4km(2D-SP)","32x24_4km(2D-SP)","16x24_4km(2D-SP)","8x24_4km(2D-SP)","8x8x24_4km (3D-SP)","Observations"/)
opt@item_count=dimsizes(opt@case_names)

opt@line_colors = (/2,1,1,6,7,8,1,31,31,1,1,1/)
;opt@line_colors = (/2,1,1,4,1,31,31,31,1,1,1/)

opt@dash_patterns = (/0,0,0,0,0,0,0,0,0,0,0,0/)
opt@line_thicknesses = (/3.0,3.0,3.0,3.0,3.0,2.0,2.5,0.5,1.0,2.0,0.5/)
opt@marker_colors = opt@line_colors
opt@marker_types = (/1,1,1,7,1,9,8,12,15,1/)
opt@marker_sizes = (/0.008,0.008,0.008,0.008,0.008,0.008,0.004,0.004,0.004,0.001, 0.004/)
opt@markers="False"



opt@reverse = "False"
opt@legend="False"
opt@title=" "
opt@leg_fontsize=0.008
if(plot_ens.eq."True") then
  un=1
else
  un=0
end if
;---------------------------------------------------------------
;plot_flag=(/1,14,21,23,24,25/)
;plot_flag=(/1,2,14,15,16,17,18,19,21,23,24,25/)
plot_flag=(/1,5,6,13,14,15,21,23,24,25/)
;plot_flag=(/1,2,9,10,14,15,16,17,18,19,21,22,23,24,25/)
nplot_flag=dimsizes(plot_flag)
;-----------------------------------------------------------------

nplotx=2
nploty=5
xleft=0.06   
ytop=0.97
dxx=0.485
dyy=0.15
xl=new((/nplotx*nploty+1/),float)
yt=xl
dx=xl
dy=xl
do jj=0,nploty-1
 do ii=0,nplotx-1
   k=ii+nplotx*jj+1
   xl(k)=xleft+dxx*ii
   dx(k)=dxx-xleft
   yt(k)=ytop-dyy*jj
   dy(k)=dyy-1.8*(1.-ytop)
 end do
end do

;-------------------------------------------------------------------


file_obs = addfile(filename_obs, "r")
day_obs = (file_obs->tsec-file_obs->tsec(0))/86400.+day0
prec_obs = file_obs->Prec(:,0,0)*86400.*1000.
lhf_obs =  file_obs->lhflx(:,0,0)
shf_obs =  file_obs->shflx(:,0,0)
cld_obs = day_obs
cld_obs =  0.   ;file_obs->cld(:,0,0)
lw_obs = file_obs->TOA_LWup(:,0,0)
swdn_obs = file_obs->SFC_SWdn(:,0,0)
swup_obs = file_obs->TOA_SWup(:,0,0)
tmin_obs = dim_min(file_obs->T(:,:,0,0))
ptmin_obs = tmin_obs
do k=0,dimsizes(day_obs)-1
ptmin_obs(k) = file_obs->lev(minind(file_obs->T(k,:,0,0)))
end do
print(file_obs->lev)

;-------------------------------------------------------------------
nx = new((/nfiles+1/),integer)
day = new((/nfiles+1,10000/),float)
i = new((/nfiles+1/),integer)
day=0.
f=new((/nfiles+1,10000/),float)
f=0.
fn = new((/nfiles+1/),string)



do j=1,nens
 do l=1,nfiles_ens
  fn(l+(j-1)*nfiles_ens)=filename_root(j)+l+".evol"
  print(filename_root(j)+l+".evol")
 end do
end do

do l=1,nfiles_dat
 fn(nfiles_ens*nens+l)=filename(l)
end do




do n=1,nfiles

i(n)=0 
caseid = chartostring(fbinrecread(fn(n),i(n),-1,"character")) 
print("caseid: "+caseid)
i(n)=i(n)+1 
nx(n) = fbinrecread(fn(n),i(n),1,"integer")  
print(nx(n)+" ")
i(n)=i(n)+1
nfield = fbinrecread(fn(n),i(n),1,"integer") 
i(n)=i(n)+1 
day(n,:nx(n)-1) = fbinrecread(fn(n),i(n),(/nx(n)/),"float")  
day(n,:nx(n)-1) = day(n,:nx(n)-1) - 0.5*( day(n,2)-day(n,1))

;day(n,:nx(n)-1)=day(n,:nx(n)-1)-day(n,0)

end do
print(day(1,:nx(1)-1))
lwp_obs = new((/nx(1)/),float)
;-------------------------------------------------------------------

iplot=0
iplot_flag=0
print("nfiled="+nfield)
opt@y_axis = "True"


do m=1,nfield


  print(m+" ")
 if(m.eq.plot_flag(iplot_flag)) then

  iplot=iplot+1

  if(iplot.ge.nplotx*nploty-1) then
    opt@xaxis = "Julian Day"
    opt@x_lab="True"
  else
    opt@xaxis = ""
    opt@x_lab="False"
  end if

 end if

  do n=1,nfiles

   i(n)=i(n)+1
   opt@title = chartostring(fbinrecread(fn(n),i(n),-1,"character")) 
   i(n)=i(n)+1
   opt@yaxis=chartostring(fbinrecread(fn(n),i(n),-1,"character")) 
   i(n)=i(n)+1
   f(n,:nx(n)-1) = fbinrecread(fn(n),i(n),(/nx(n)/),"float")
   if(n.eq.1) then
     print(m+" "+ opt@title)
   end if

  end do

  if(m.eq.20) then
     lwp_obs = f(1,:nx(1)-1) 
  end if


  opt@xl = xl(iplot)
  opt@dx = dx(iplot)
  opt@yt = yt(iplot)
  opt@dy = dy(iplot)

  opt@xmin=min(day(1:nfiles,0))
  opt@xmax=max(day(1:nfiles,max(nx)-1))
  opt@ymin=1.0*min(f(1:nfiles,:min(nx)-1))
  opt@ymax=1.0*max(f(1:nfiles,:min(nx)-1))
  if(m.eq.1) then
    opt@ymin=0.
    opt@ymax=1.
  end if
  if(m.eq.5) then
    opt@ymin=180.
    opt@ymax=210.
  end if
  if(m.eq.6) then
    opt@ymin=80.
    opt@ymax=140.
  end if
  if(m.eq.13) then
    opt@ymin=0.
    opt@ymax=300.
  end if
  if(m.eq.14) then
    opt@ymin=100.
    opt@ymax=400.
  end if
  if(m.eq.15) then
    opt@ymin=100.
    opt@ymax=400.
  end if
  if(m.eq.21) then
    opt@ymin=20.
    opt@ymax=80.
  end if
  if(m.eq.23) then
    opt@ymin=0.
    opt@ymax=300.
  end if
  if(m.eq.24) then
    opt@ymin=0.
    opt@ymax=40.
  end if
  if(m.eq.25) then
    opt@ymin=0.
    opt@ymax=60.
  end if
  

  if(m.eq.plot_flag(iplot_flag)) then  

  opt@line_color = opt@item_count-1
  opt@dash_pattern =  opt@item_count-1
  opt@line_thickness =  opt@item_count-1
  opt@marker_no =  opt@item_count-1
  if(m.eq.1) then
       nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         cld_obs(:)=wgt_runave( cld_obs(:),wgt,1)
         delete(wgt)
       end if
     plotXY(wks, opt, cld_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.5) then
       nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         tmin_obs(:)=wgt_runave( tmin_obs(:),wgt,1)
         tmin_obs(:)=wgt_runave( tmin_obs(:),wgt,1)
         delete(wgt)
       end if
     plotXY(wks, opt, tmin_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.6) then
       nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         ptmin_obs(:)=wgt_runave( ptmin_obs(:),wgt,1)
         ptmin_obs(:)=wgt_runave( ptmin_obs(:),wgt,1)
         delete(wgt)
       end if
     plotXY(wks, opt, ptmin_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.14) then
       nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         lw_obs(:)=wgt_runave( lw_obs(:),wgt,1)
         delete(wgt)
       end if
     plotXY(wks, opt, lw_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.13) then
       nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         swup_obs(:)=wgt_runave( swup_obs(:),wgt,1)
         swup_obs(:)=wgt_runave( swup_obs(:),wgt,1)
         delete(wgt)
       end if
     plotXY(wks, opt, swup_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.15) then
       nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         swdn_obs(:)=wgt_runave( swdn_obs(:),wgt,1)
         swdn_obs(:)=wgt_runave( swdn_obs(:),wgt,1)
         delete(wgt)
       end if
     plotXY(wks, opt, swdn_obs,day_obs)
     opt@legend="True"
  end if

  if(m.eq.25) then
      nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
      if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         prec_obs(:)=wgt_runave( prec_obs(:),wgt,1)
         delete(wgt)
      end if
     plotXY(wks, opt, prec_obs,day_obs)
     opt@legend="True"
  end if
 if(m.eq.23) then
      nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
      if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         lhf_obs(:)=wgt_runave( lhf_obs(:),wgt,1)
         delete(wgt)
      end if
     plotXY(wks, opt, lhf_obs,day_obs)
     opt@legend="false"
  end if
  if(m.eq.24) then
      nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
      if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         shf_obs(:)=wgt_runave( shf_obs(:),wgt,1)
         delete(wgt)
      end if
     plotXY(wks, opt, shf_obs,day_obs)
     opt@legend="false"
  end if

  if(m.eq.21) then
      nnn=floattointeger(time_scale/((day_obs(2)-day_obs(1))*24.))+1
      if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         lwp_obs(:)=wgt_runave( lwp_obs(:),wgt,1)
         delete(wgt)
      end if
     plotXY(wks, opt, lwp_obs, day(1,:nx(1)-1)) 
     opt@legend="false"
  end if
  do j=0,nens-1
  do l=1,nfiles_ens

      opt@line_color = 0
      opt@dash_pattern = 0
      opt@line_thickness = 0
      opt@marker_no = 0
      n=j*nfiles_ens+l
      if(plot_ens.eq."True") then
       nnn=floattointeger(time_scale/((day(n,2)-day(n,1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         f(n,:nx(l)-1)=wgt_runave( f(n,:nx(l)-1),wgt,1)
         delete(wgt)
       end if
       plotXY(wks, opt, f(n,:nx(n)-1),day(n,:nx(n)-1))
      end if

  end do
  end do

  do j=0,nens-1
    opt@line_color = un+j 
    opt@dash_pattern = un+j
    opt@line_thickness = un+j
    opt@marker_no = un+j
    n=j*nfiles_ens+1
    f!0="ens"
    f!1="x"
    f1 = dim_avg(f(x|:nx(n)-1,ens|n:(j+1)*nfiles_ens))
    nnn=floattointeger(time_scale/((day(n,2)-day(n,1))*24.))+1
    if(nnn.gt.1) then
     wgt=new((/nnn/),float)
     wgt=1./nnn
     f1(:nx(l)-1)=wgt_runave( f1(:nx(l)-1),wgt,1)
     delete(wgt)
    end if
    plotXY(wks, opt, f1(:nx(n)-1:nnn),day(n,:nx(n)-1:nnn))
  end do

  do l=nens*nfiles_ens+1,nfiles

      opt@line_color = l - nfiles_ens*nens +un+(nens-1)
      opt@line_thickness = l - nfiles_ens*nens +un+(nens-1)
      opt@dash_pattern = l - nfiles_ens*nens +un+(nens-1)
      opt@marker_no = l - nfiles_ens*nens +un+(nens-1)

  if(m.eq.14) then
;   f(l,:nx(l)-1) = f(l,:nx(l)-1) - f(l,0)+lw_obs(0)
  end if
  if(m.eq.21) then
   f(l,:nx(l)-1) = f(l,:nx(l)-1) - f(l,0)+lwp_obs(0)
  end if
  nnn=floattointeger(time_scale/((day(l,2)-day(l,1))*24.))+1
  print(" "+nnn)
  if(nnn.gt.1) then
    wgt=new((/nnn/),float)
    wgt=1./nnn
    f(l,:nx(l)-1)=wgt_runave( f(l,:nx(l)-1),wgt,1)
    delete(wgt)
  end if

      plotXY(wks, opt, f(l,:nx(l)-1:nnn),day(l,:nx(l)-1:nnn))

  end do

  end if


  if(m.eq.plot_flag(iplot_flag)) then

  opt@legend="False"
  axisXY(wks, opt)
  if(iplot.eq.4) then
    item_max=7
    opt@leg_xl =  opt@xl+0.42*opt@dx
    opt@leg_yt =  opt@yt-0.05*opt@dy
    opt@leg_dx =  0.15*opt@dx
    opt@leg_dy =  0.12*opt@dy*opt@item_count
    if(opt@item_count.le.item_max) then
      legendXY(wks,opt)
    else
      item_count=opt@item_count
      opt@item_count = item_max
      opt@leg_dy =  0.10*opt@dy*opt@item_count
      legendXY(wks,opt)
      case_names = opt@case_names
      line_colors = opt@line_colors
      dash_patterns = opt@dash_patterns
      line_thicknesses = opt@line_thicknesses
      marker_colors = opt@marker_colors
      marker_types = opt@marker_types 
      marker_sizes = opt@marker_sizes 
      opt@case_names(0:item_count-item_max-1) = case_names(item_max:item_count-1)
      opt@line_colors(0:item_count-item_max-1) = line_colors(item_max:item_count-1)
      opt@dash_patterns(0:item_count-item_max-1) = dash_patterns(item_max:item_count-1)
      opt@line_thicknesses(0:item_count-item_max-1)=line_thicknesses(item_max:item_count-1)
      opt@marker_colors(0:item_count-item_max-1)=marker_colors(item_max:item_count-1)
      opt@marker_types(0:item_count-item_max-1)=marker_types(item_max:item_count-1)
      opt@marker_sizes(0:item_count-item_max-1)=marker_sizes(item_max:item_count-1)
      opt@item_count=item_count-item_max
      opt@leg_xl =  opt@leg_xl+1.1*opt@leg_dx
      opt@leg_dy =  0.10*opt@dy*opt@item_count
      legendXY(wks,opt)
      opt@item_count=item_count
      opt@case_names = case_names
      opt@line_colors = line_colors
      opt@dash_patterns = dash_patterns
      opt@line_thicknesses = line_thicknesses
      opt@marker_colors = marker_colors
      opt@marker_types = marker_types 
      opt@marker_sizes = marker_sizes


    end if
  end if

  if(iplot.eq.nplotx*nploty) then
    iplot=0
    frame(wks)
  end if


  iplot_flag=iplot_flag+1
  if(iplot_flag.ge.nplot_flag) then
   break
  end if

 end if

end do

frame(wks)

end
