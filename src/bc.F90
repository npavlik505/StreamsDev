subroutine bc(inr)
!
! Application of boundary conditions
!
 use mod_streams
 implicit none
!
 integer :: inr,ilat
!
! Table of values for parameter (ibc(ilat))
! ilat = 1,6
!
!     ____________           ibc = 1  -> freestream
!    /|         /|           ibc = 2  -> extrapolation 
!   / |  4     / |           ibc = 3  -> 
!  /  |    5  /  |  j        ibc = 4  -> nonreflecting
! /__________/   |  |        ibc = 5  -> wall (staggered)
! | 1 |______|_2 |  |____ i  ibc = 6  -> wall
! |  / 6     |  /  /         ibc = 7  -> oblique shock imposed
! | /    3   | /  /          ibc = 8  -> wall (PL type bc)
! |/         |/  k           ibc = 9  -> digital filtering for turbulent inflow
! /----------/               ibc = 10 -> Recycling-rescaling turbulent inflow
!                            ibc = blowing_sbli_boundary_condition -> use blowing boundary condition
!
! inr = 0 -> steady-type    boundary conditions
! inr = 1 -> non-reflecting boundary conditions
!

! FOR SHOCK BL CASE:
! surface 1 = 9 digitial filtering
! surface 2 = 4 nonreflecting
! surface 3 = 8 wall (PL type)
! surface 4 = 7 oblique shock imposed
! 5/6 are = 0 ?

  if (inr==0) then 
!
!  Steady-type BCs
!
!  'Physical' boundary conditions
!
   do ilat=1,2*ndim ! loop on all sides of the boundary (3D -> 6, 2D -> 4)
    !if (masterproc) then
    !  write(*,*) "ibc(ilat) is", ibc(ilat)
    !endif

    if (ibc(ilat)==1) call bcfree(ilat)
    ! types 2 and 4 share the same boundary condition subroutine even though they 
    ! have different names?
    if (ibc(ilat)==2) call bcextr(ilat)
    if (ibc(ilat)==4) call bcextr(ilat)
    if (ibc(ilat)==5) call bcwall_staggered(ilat)
    if (ibc(ilat)==6) call bcwall(ilat)
    if (ibc(ilat)==7) call bcshk(ilat)
    if (ibc(ilat)==8) call bcwall(ilat)
    if (ibc(ilat)==9) then
     if (.not.dfupdated) then
      call bcdf(ilat)
      dfupdated = .true.
     endif
    endif
    if (ibc(ilat)==10) call bcrecyc(ilat)
    !if (ibc(ilat)==10) call bc_constant_input(ilat)
    if (ibc(ilat)== blowing_sbli_boundary_condition) then 
        call bcblow(ilat)
    endif
   enddo
!
  else
!
!  Unsteady-type BCs (update boundary fluxes)
!
   do ilat=1,2*ndim ! loop on all sides of the boundary (3D -> 6, 2D -> 4)
    select case (ibc(ilat))
    case(4,7,9,10)
     call bcrelax(ilat)
    case(8)
     call bcwall_pl(ilat)
    end select
   enddo
  endif
!  
end subroutine bc
