subroutine wrap_get_x(x_out, n) bind(C, name="wrap_get_x")
    use iso_c_binding
    use mod_streams, only: x
    !f2py intent(c) wrap_get_x
    !f2py intent(out) x_out
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: x_out(n)
    integer :: i

    do i = 1, n
        x_out(i) = x(i)
    end do
end subroutine wrap_get_x

subroutine wrap_get_y(y_out, n) bind(C, name="wrap_get_y")
    use iso_c_binding
    use mod_streams, only: y
    !f2py intent(c) wrap_get_y
    !f2py intent(out) y_out
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: y_out(n)
    integer :: i

    do i = 1, n
        y_out(i) = y(i)
    end do
end subroutine wrap_get_y

subroutine wrap_get_z(z_out, n) bind(C, name="wrap_get_z")
    use iso_c_binding
    use mod_streams, only: z
    !f2py intent(c) wrap_get_z
    !f2py intent(out) z_out
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: z_out(n)
    integer :: i

    do i = 1, n
        z_out(i) = z(i)
    end do
end subroutine wrap_get_z

subroutine wrap_get_tauw_x(tauw_out, n) bind(C, name="wrap_get_tauw_x")
    use iso_c_binding
    use mod_streams, only: tauw_x
    !f2py intent(c) wrap_get_tauw_x
    !f2py intent(out) tauw_out
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: tauw_out(n)
    integer :: i

    do i = 1, n
        tauw_out(i) = tauw_x(i)
    end do
end subroutine wrap_get_tauw_x

subroutine wrap_get_w(w_out, n) bind(C, name="wrap_get_w")
    use iso_c_binding
    use mod_streams, only: w
    !f2py intent(c) wrap_get_w
    !f2py intent(out) w_out
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: w_out(n)
    integer :: i

    do i = 1, n
        w_out(i) = w(i)
    end do
end subroutine wrap_get_w

subroutine wrap_get_x_start_slot(val) bind(C, name="wrap_get_x_start_slot")
    use iso_c_binding
    use mod_streams, only: x_start_slot
    !f2py intent(c) wrap_get_x_start_slot
    !f2py intent(out) val
    implicit none
    integer, intent(out) :: val
    val = x_start_slot
end subroutine wrap_get_x_start_slot

subroutine wrap_get_nx_slot(val) bind(C, name="wrap_get_nx_slot")
    use iso_c_binding
    use mod_streams, only: nx_slot
    !f2py intent(c) wrap_get_nx_slot
    !f2py intent(out) val
    implicit none
    integer, intent(out) :: val
    print *, '>>> wrap_get_nx_slot called, nx_slot =', nx_slot
    val = nx_slot
end subroutine wrap_get_nx_slot

subroutine wrap_get_nz_slot(val) bind(C, name="wrap_get_nz_slot")
    use iso_c_binding
    use mod_streams, only: nz_slot
    !f2py intent(c) wrap_get_nz_slot
    !f2py intent(out) val
    implicit none
    integer, intent(out) :: val
    val = nz_slot
end subroutine wrap_get_nz_slot

subroutine wrap_get_blowing_bc_slot_velocity(arr, n) bind(C, name="wrap_get_blowing_bc_slot_velocity")
    use iso_c_binding
    use mod_streams, only: blowing_bc_slot_velocity
    !f2py intent(c) wrap_get_blowing_bc_slot_velocity
    !f2py intent(out) arr
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: arr(n)
    integer :: i

    do i = 1, n
        arr(i) = blowing_bc_slot_velocity(i)
    end do
end subroutine wrap_get_blowing_bc_slot_velocity

subroutine wrap_set_nx_slot(val) bind(C, name="wrap_set_nx_slot")
    use iso_c_binding
    use mod_streams, only: nx_slot
    !f2py intent(c) wrap_set_nx_slot
    !f2py intent(in) val
    implicit none
    integer, intent(in) :: val
    nx_slot = val
    print *, '>>> wrap_set_nx_slot: setting nx_slot =', val
end subroutine wrap_set_nx_slot

