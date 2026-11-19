! C3 instrument: list-directed readback of every whitespace token of a text file
! as real(8), one per line in es26.17e3 (18 significant digits = exact double).
! The oracle drivers read their inputs list-directed; this reads the same tokens
! the same way.  usage: readback.x <in.txt> <out.txt>
program readback
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  character(len=4000000) :: line
  character(len=512) :: fin, fout
  real(r8), allocatable :: v(:)
  integer :: ios, n, i
  logical :: intok
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old')
  open(20, file=trim(fout), status='replace')
  do
     read(10, '(a)', iostat=ios) line
     if (ios /= 0) exit
     n = 0; intok = .false.
     do i = 1, len_trim(line)
        if (line(i:i) /= ' ' .and. .not. intok) then
           n = n + 1; intok = .true.
        else if (line(i:i) == ' ') then
           intok = .false.
        end if
     end do
     if (n == 0) cycle
     allocate(v(n))
     read(line, *) v
     do i = 1, n
        write(20, '(es26.17e3)') v(i)
     end do
     deallocate(v)
  end do
  close(10); close(20)
end program readback
