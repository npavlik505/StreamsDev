subroutine setup
!
! Setup for the computation
!
 use mod_streams
 implicit none
 print *, ">>> [setup] entering setup()"
! find out the dimensions of the blowing slot
!
!===================================================
 if (masterproc) write(error_unit,*) 'Allocation of variables'
 call allocate_vars()
 print *, '>>> [debug] after allocate_vars: nx_slot =', nx_slot
!===================================================
 if (masterproc) write(error_unit,*) 'Reading input'
 call readinp()
 print *, '>>> [debug] after readinp: nx_slot =', nx_slot

! separate allocations after the input files have been read so that 
! the correct boundary conditions may be established
 call local_slot_locations()
 print *, '>>> [debug] after local_slot_locations: nx_slot =', nx_slot
 call allocate_blowing_bcs()
 print *, '>>> [debug] after allocate_blowing_bcs: nx_slot =', nx_slot

!===================================================
 if (idiski==0) then
  if (masterproc) write(*,*) 'Generating mesh'
  call generategrid()
  print *, '>>> [debug] after generategrid: nx_slot =', nx_slot
 else
  if (masterproc) write(*,*) 'Reading mesh'
  call readgrid()
  print *, '>>> [debug] after readgrid: nx_slot =', nx_slot
 endif
 if (masterproc) write(*,*) 'Computing metrics'
 call computemetrics()
 print *, '>>> [debug] after computemetrics: nx_slot =', nx_slot
 if (enable_plot3d>0) call writegridplot3d()
 print *, '>>> [debug] after writegridplot3d (if called): nx_slot =', nx_slot
!===================================================
 call constants()
 print *, '>>> [debug] after constants: nx_slot =', nx_slot
!===================================================
 if (xrecyc>0._mykind) call recyc_prepare()
 print *, '>>> [debug] after recyc_prepare (if called): nx_slot =', nx_slot
!===================================================
 if (masterproc) write(*,*) 'Initialize field'
 call init()
 print *, '>>> [debug] after init: nx_slot =', nx_slot
 call checkdt()
 print *, '>>> [debug] after checkdt: nx_slot =', nx_slot
 call generate_full_fdm_stencil()
 print *, '>>> [debug] after generate_full_fdm_stencil: nx_slot =', nx_slot
!===================================================
!
end subroutine setup
