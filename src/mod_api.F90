subroutine wrap_get_x(x_out, n)
    use mod_streams, only: x
    implicit none
    integer, intent(in) :: n
    real(8), intent(out) :: x_out(n)
    integer :: i

    do i = 1, n
        x_out(i) = x(i)
    end do
end subroutine wrap_get_x

subroutine wrap_get_y(y_out, n)
    use mod_streams, only: y
    implicit none
    integer, intent(in) :: n
    real(8), intent(out) :: y_out(n)
    integer :: i

    do i = 1, n
        y_out(i) = y(i)
    end do
end subroutine wrap_get_y

subroutine wrap_get_z(z_out, n)
    use mod_streams, only: z
    implicit none
    integer, intent(in) :: n
    real(8), intent(out) :: z_out(n)
    integer :: i

    do i = 1, n
        z_out(i) = z(i)
    end do
end subroutine wrap_get_z

subroutine wrap_get_tauw_x(tauw_out, n)
    use mod_streams, only: tauw_x
    implicit none
    integer, intent(in) :: n
    real(8), intent(out) :: tauw_out(n)
    integer :: i

    do i = 1, n
        tauw_out(i) = tauw_x(i)
    end do
end subroutine wrap_get_tauw_x

subroutine wrap_get_w(w_out, n)
    use mod_streams, only: w
    implicit none
    integer, intent(in) :: n
    real(8), intent(out) :: w_out(n)
    integer :: i

    do i = 1, n
        w_out(i) = w(i)
    end do
end subroutine wrap_get_w

subroutine wrap_get_x_start_slot(val)
    use mod_streams, only: x_start_slot
    implicit none
    integer, intent(out) :: val
    val = x_start_slot
end subroutine wrap_get_x_start_slot

subroutine wrap_get_nx_slot(val)
    use mod_streams, only: nx_slot
    implicit none
    integer, intent(out) :: val
    val = nx_slot
end subroutine wrap_get_nx_slot

subroutine wrap_get_nz_slot(val)
    use mod_streams, only: nz_slot
    implicit none
    integer, intent(out) :: val
    val = nz_slot
end subroutine wrap_get_nz_slot

subroutine wrap_get_blowing_bc_slot_velocity(arr, n)
    use mod_streams, only: blowing_bc_slot_velocity
    implicit none
    integer, intent(in) :: n
    real(8), intent(out) :: arr(n)
    integer :: i

    do i = 1, n
        arr(i) = blowing_bc_slot_velocity(i)
    end do
end subroutine wrap_get_blowing_bc_slot_velocity

