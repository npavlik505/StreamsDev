subroutine manage_solver
!
 use mod_streams
 implicit none
!
 logical :: updatestat, savefield, saverst, saveprobe, savespanaverage
!
!call write_wallpressure
!
 updatestat = .false.
 savefield  = .false.
 saverst    = .false.
 saveprobe = .false.
 savespanaverage = .false.
!
 if (mod(icyc,istat)==0) updatestat = .true.
 if (telaps>tsol(istore)) savefield = .true.
 !if (mod(icyc, 5000) == 0) savefield = .true.

 if (telaps>tsol_restart(istore_restart)) saverst = .true.
 if (mod(icyc, 100) == 0) savespanaverage= .true.

 ! check if we should write span average information
 if (save_span_average_steps > 0) then
    if (mod(icyc, save_span_average_steps) == 0) savespanaverage = .true.
 end if

!
 if (updatestat.or.savefield.or.saverst) then
  if (xrecyc>0._mykind) call recyc
  call updateghost()
  call prims()
  call copy_gpu_to_cpu()
 endif
!
!Statistics
 if (updatestat) then
  if (iflow==-1) then
  elseif (iflow==0) then
   call stats1d()
  else
   call stats2d()
  endif
 endif

 if (savespanaverage) then
    !call write_span_averaged
 endif
!
!Writing fields
 if (savefield) then
  if (enable_plot3d>0) call writefield()
  if (enable_vtk>0) call writefieldvtk()
  if (iflow>0) call writestatzbl()
  istore = istore+1
 endif
!
 if (saverst) then
  if (io_type==1) call writerst_serial()
  if (io_type==2) call writerst()
  if (iflow==-1) then
  elseif (iflow==0) then
   call writestat1d()
  else
   if (io_type==1) then
    call writestat2d_serial()
    call writedf_serial()
   else
    call writestat2d()
    call writedf()
   endif
  endif
  istore_restart = istore_restart+1
 endif

!
 if (updatestat.or.savefield.or.saverst) then
  call reset_cpu_gpu()
 endif
!
end subroutine manage_solver
