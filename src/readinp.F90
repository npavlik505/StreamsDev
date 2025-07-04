subroutine readinp
!
! Reading the input file
!
 use mod_streams
 implicit none
!
 integer :: l, i_skip
 integer :: ibc_inflow
 real(mykind) :: turb_inflow
!
!
! iflow = 0 ==> Channel flow
!
!       rm:             Mach number based on bulk velocity and wall temperature
!       retauinflow:    Estimated friction Reynolds number (target)
!       trat:           Bulk temperature/wall temperature (if <=0., bulk temperature free to evolve)
!       pgradf:         Assigned pressure gradient (if 0. pressure gradient computed to maintain bulk velocity constant)
!
! iflow = 1,2 ==> Boundary layer,SBLI
!
!       rm:             Free-stream Mach number
!       retauinflow:    Estimated friction Reynolds number (target) at inflow trat:           Wall-to-recovery-temperature ratio deflec:         Deflection angle after shock (only for SBLI)
!
 open (unit=12,file='input.dat',form='formatted')
 do i_skip=1,33
  read (12,*)
 enddo
 read (12,*)
 read (12,*) tresduc, ximp, deflec, pgradf
 read (12,*)
 read (12,*)
 read (12,*) idiski, ncyc, cfl, nstep, nprint, io_type
 read (12,*)
 read (12,*)
 read (12,*) rm, retauinflow, trat, visc_type, tref_dimensional, turb_inflow
 read (12,*)
 read (12,*)
 read (12,*) istat, nstat
 allocate( xstat(nstat))
 allocate(ixstat(nstat))
 allocate(igxstat(nstat))
 read (12,*)
 read (12,*)
 read (12,*) (xstat(l),l=1,nstat)
 read (12,*)
 read (12,*)
 read (12,*) dtsave, dtsave_restart, enable_plot3d, enable_vtk
 read (12,*)
 read (12,*)
 read (12,*) rand_start
 read (12,*)
 read (12,*)
 read (12,*) save_probe_steps, save_span_average_steps
 read (12,*)
 read (12,*)
 read (12,*) force_sbli_blowing_bc, slot_start_x_global, slot_end_x_global

 ! write(*,*) "slot bounds", slot_start_x_global, slot_end_x_global

 close(12)
!
 call check_input(2)
!
 xrecyc = -1._mykind
 if (iflow>0) then
  if (turb_inflow<0._mykind) then
   ibc_inflow = 10
   xrecyc = -turb_inflow
   if (xrecyc>rlx) call fail_input("Recycling station outside computational domain")
  else
   ibc_inflow = 9
   dftscaling = turb_inflow
  endif
 endif
!
 ibc   = 0
 ibcnr = 0
 select case (iflow)
 case (-1)
  ibc(1) = 1 ! Free stream
  ibc(2) = 4 ! Extrapolation + non reflecting treatment
  ibc(3) = 2 ! Extrapolation
  ibc(4) = 2
  ibc(5) = 2
  ibc(6) = 2
 case (0) ! Channel
  ibc(3) = 5 ! Staggered wall
  ibc(4) = 5
 case (1) ! BL
  ibc(1) = ibc_inflow
  ibc(2) = 4 ! Extrapolation + non reflecting treatment
  !ibc(3) = 8 
  ibc(4) = 4
  ibcnr(1) = 1

  if (force_sbli_blowing_bc == 1) then
      ! use blowing boundary condition
      ibc(3) = blowing_sbli_boundary_condition
  else
      ! use default solver BC
      ! Wall + reflecting treatment
      ibc(3) = 8
  endif

 case (2) ! SBLI
  ibc(1) = ibc_inflow
  ibc(2) = 4

  if (force_sbli_blowing_bc == 1) then
      ! use blowing boundary condition
      ibc(3) = blowing_sbli_boundary_condition
  else
      ! use default solver BC
      ibc(3) = 8
  endif
    
  ibc(4) = 7
  ibcnr(1) = 1
 end select
!
 ndim = 3 ! default number of dimensions
 if (nzmax==1) ndim = 2
!
 do l=0,nsolmax
  tsol(l)         = l*dtsave
  tsol_restart(l) = l*dtsave_restart
 enddo
!
end subroutine readinp
