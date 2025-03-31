subroutine wrap_startmpi() bind(C, name="wrap_startmpi")
    use iso_c_binding
    !f2py intent(c) wrap_startmpi
    !f2py intent(hide)
    call startmpi()
end subroutine wrap_startmpi

subroutine wrap_setup() bind(C, name="wrap_setup")
    use iso_c_binding
    !f2py intent(c) wrap_setup
    !f2py intent(hide)
    use mod_streams, only: tauw_x
    call setup()
end subroutine wrap_setup

subroutine wrap_init_solver() bind(C, name="wrap_init_solver")
    use iso_c_binding
    !f2py intent(c) wrap_init_solver
    !f2py intent(hide)
    call init_solver()
end subroutine wrap_init_solver

subroutine wrap_step_solver() bind(C, name="wrap_step_solver")
    use iso_c_binding
    !f2py intent(c) wrap_step_solver
    !f2py intent(hide)
    call step_solver()
end subroutine wrap_step_solver

subroutine wrap_finalize_solver() bind(C, name="wrap_finalize_solver")
    use iso_c_binding
    !f2py intent(c) wrap_finalize_solver
    !f2py intent(hide)
    call finalize_solver()
end subroutine wrap_finalize_solver

subroutine wrap_finalize() bind(C, name="wrap_finalize")
    use iso_c_binding
    !f2py intent(c) wrap_finalize
    !f2py intent(hide)
    use mod_streams
    call finalize()
    call mpi_finalize(iermpi)
end subroutine wrap_finalize

subroutine wrap_copy_gpu_to_cpu() bind(C, name="wrap_copy_gpu_to_cpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_gpu_to_cpu
    !f2py intent(hide)
    call updateghost()
    call prims()
    call copy_gpu_to_cpu()
end subroutine wrap_copy_gpu_to_cpu

subroutine wrap_copy_cpu_to_gpu() bind(C, name="wrap_copy_cpu_to_gpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_cpu_to_gpu
    !f2py intent(hide)
    call copy_cpu_to_gpu()
end subroutine wrap_copy_cpu_to_gpu

subroutine wrap_copy_blowing_bc_to_gpu() bind(C, name="wrap_copy_blowing_bc_to_gpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_blowing_bc_to_gpu
    !f2py intent(hide)
    call copy_blowing_bc_to_gpu()
end subroutine wrap_copy_blowing_bc_to_gpu

subroutine wrap_copy_blowing_bc_to_cpu() bind(C, name="wrap_copy_blowing_bc_to_cpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_blowing_bc_to_cpu
    !f2py intent(hide)
    call copy_blowing_bc_to_cpu()
end subroutine wrap_copy_blowing_bc_to_cpu

subroutine wrap_tauw_calculate() bind(C, name="wrap_tauw_calculate")
    use iso_c_binding
    !f2py intent(c) wrap_tauw_calculate
    !f2py intent(hide)
    use mod_streams, only: tauw_x, w_av, mykind, y, nx, ny, ncoords, masterproc
    implicit none
    integer :: i, j
    real(mykind), dimension(nx, ny) :: ufav, vfav, wfav
    real(mykind) :: dudyw, dy, rmuw, tauw

    call stats2d()
    if (ncoords(3) == 0) then
        do j = 1, ny
            do i = 1, nx
                ufav(i, j) = w_av(13, i, j)/w_av(1, i, j)
                vfav(i, j) = w_av(14, i, j)/w_av(1, i, j)
                wfav(i, j) = w_av(15, i, j)/w_av(1, i, j)
            end do
        end do
        do i = 1, nx
            dudyw = (-22._mykind*ufav(i, 1) + 36._mykind*ufav(i, 2) - 18._mykind*ufav(i, 3) + 4._mykind*ufav(i, 4))/12._mykind
            dy = (-22._mykind*y(1) + 36._mykind*y(2) - 18._mykind*y(3) + 4._mykind*y(4))/12._mykind
            dudyw = dudyw/dy
            rmuw = w_av(20, i, 1)
            tauw = rmuw*dudyw
            tauw_x(i) = tauw
        end do
    end if
end subroutine wrap_tauw_calculate

subroutine wrap_dissipation_calculation() bind(C, name="wrap_dissipation_calculation")
    use iso_c_binding
    !f2py intent(c) wrap_dissipation_calculation
    !f2py intent(hide)
    implicit none
    call dissipation_calculation()
end subroutine wrap_dissipation_calculation

subroutine wrap_energy_calculation() bind(C, name="wrap_energy_calculation")
    use iso_c_binding
    !f2py intent(c) wrap_energy_calculation
    !f2py intent(hide)
    implicit none
    call energy_calculation()
end subroutine wrap_energy_calculation

subroutine wrap_deallocate_all() bind(C, name="wrap_deallocate_all")
    use iso_c_binding
    !f2py intent(c) wrap_deallocate_all
    !f2py intent(hide)
    implicit none
    call deallocate_all()
end subroutine wrap_deallocate_all

subroutine wrap_get_x(x_out) bind(C, name="wrap_get_x")
    use mod_streams
    implicit none
    real*8, intent(out) :: x_out(:)
    x_out = x
end subroutine wrap_get_x

subroutine wrap_get_y(y_out) bind(C, name="wrap_get_y")
    use mod_streams
    implicit none
    real*8, intent(out) :: y_out(:)
    y_out = y
end subroutine wrap_get_y

subroutine wrap_get_z(z_out) bind(C, name="wrap_get_z")
    use mod_streams
    implicit none
    real*8, intent(out) :: z_out(:)
    z_out = z
end subroutine wrap_get_z

subroutine wrap_get_w(w_out) bind(C, name="wrap_get_w")
    use mod_streams
    implicit none
    real*8, intent(out) :: w_out(:,:,:,:)
    w_out = w
end subroutine wrap_get_w

subroutine wrap_get_tauw_x(tauw_out) bind(C, name="wrap_get_tauw_x")
    use mod_streams
    implicit none
    real*8, intent(out) :: tauw_out(:)
    tauw_out = tauw_x
end subroutine wrap_get_tauw_x

subroutine wrap_get_x_start_slot(val) bind(C, name="wrap_get_x_start_slot")
  use iso_c_binding
  use mod_streams, only: x_start_slot
  integer(c_int), intent(out) :: val
  val = x_start_slot
end subroutine

subroutine wrap_get_nx_slot(val) bind(C, name="wrap_get_nx_slot")
  use iso_c_binding
  use mod_streams, only: nx_slot
  integer(c_int), intent(out) :: val
  val = nx_slot
end subroutine

subroutine wrap_get_nz_slot(val) bind(C, name="wrap_get_nz_slot")
  use iso_c_binding
  use mod_streams, only: nz_slot
  integer(c_int), intent(out) :: val
  val = nz_slot
end subroutine

subroutine wrap_get_blowing_bc_slot_velocity(arr) bind(C, name="wrap_get_blowing_bc_slot_velocity")
  use iso_c_binding
  use mod_streams, only: blowing_bc_slot_velocity, mykind
  real(c_double), intent(out) :: arr(:,:)
  arr = blowing_bc_slot_velocity
end subroutine

