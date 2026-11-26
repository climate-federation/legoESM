! Drop-in replacement for the CLOUDSC dwarf's hdf5_file_mod: same type name,
! same methods and the same hyperslab semantics (1-based start, count per
! Fortran dimension), reading the per-dataset text files written by
! export_h5.py from the directory <name> passed to open_file (the dwarf passes
! NAME//'.h5'; the '.h5' suffix is stripped).  Harness glue only.
module hdf5_file_mod
  use parkind1, only: jpim, jprd
  implicit none
  private
  public :: hdf5_file

  type hdf5_file
    character(len=512) :: dir = ''
    logical :: is_open = .false.
  contains
    procedure :: init => tx_init
    procedure :: create_file => tx_create
    procedure :: open_file => tx_open
    procedure :: assert_file_open => tx_assert
    procedure :: close_file => tx_close
    procedure, private :: tx_l0, tx_i0, tx_r0, tx_l1, tx_i1, tx_r1, tx_r2, tx_r3
    generic :: load => tx_l0, tx_i0, tx_r0, tx_l1, tx_i1, tx_r1, tx_r2, tx_r3
  end type hdf5_file

contains

  subroutine tx_init(self)
    class(hdf5_file) :: self
  end subroutine tx_init

  subroutine tx_create(self, filename)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: filename
    call abor1('textio: create_file not supported')
  end subroutine tx_create

  subroutine tx_open(self, filename)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: filename
    integer :: n
    n = len_trim(filename)
    if (n > 3) then
      if (filename(n-2:n) == '.h5') n = n - 3
    end if
    self%dir = filename(1:n)
    self%is_open = .true.
  end subroutine tx_open

  subroutine tx_assert(self)
    class(hdf5_file) :: self
    if (.not. self%is_open) call abor1('textio: file not open')
  end subroutine tx_assert

  subroutine tx_close(self)
    class(hdf5_file) :: self
    self%is_open = .false.
  end subroutine tx_close

  ! read the whole dataset as real(jprd), flat in Fortran order, with its extents
  subroutine read_all(self, name, ext, flat)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    integer, intent(out) :: ext(3)
    real(kind=jprd), allocatable, intent(out) :: flat(:)
    integer :: u, rank, i, n, ios
    character(len=600) :: path
    call self%assert_file_open()
    path = trim(self%dir)//'/'//trim(name)//'.txt'
    open(newunit=u, file=path, status='old', action='read', iostat=ios)
    if (ios /= 0) call abor1('textio: missing dataset '//trim(name))
    ext = 1
    read(u, *) rank
    backspace(u)
    read(u, *) rank, (ext(i), i = 1, rank)
    n = product(ext)
    allocate(flat(n))
    read(u, *, iostat=ios) flat
    if (ios /= 0) call abor1('textio: short read '//trim(name))
    close(u)
  end subroutine read_all

  subroutine tx_l0(self, name, val)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    logical, intent(inout) :: val
    integer :: e(3); real(kind=jprd), allocatable :: f(:)
    call read_all(self, name, e, f); val = f(1) > 0
  end subroutine tx_l0

  subroutine tx_i0(self, name, val)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    integer(kind=jpim), target, intent(inout) :: val
    integer :: e(3); real(kind=jprd), allocatable :: f(:)
    call read_all(self, name, e, f); val = nint(f(1), jpim)
  end subroutine tx_i0

  subroutine tx_r0(self, name, val)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    real(kind=jprd), intent(inout) :: val
    integer :: e(3); real(kind=jprd), allocatable :: f(:)
    call read_all(self, name, e, f); val = f(1)
  end subroutine tx_r0

  subroutine tx_l1(self, name, buffer, start, count)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    logical, intent(inout) :: buffer(:)
    integer(kind=jpim), intent(in), optional :: start, count
    integer :: e(3), s, c; real(kind=jprd), allocatable :: f(:)
    call read_all(self, name, e, f)
    s = 1; c = size(buffer)
    if (present(start)) s = start
    if (present(count)) c = count
    buffer(1:c) = f(s:s+c-1) > 0
  end subroutine tx_l1

  subroutine tx_i1(self, name, buffer, start, count)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    integer(kind=jpim), target, intent(inout) :: buffer(:)
    integer(kind=jpim), intent(in), optional :: start, count
    integer :: e(3), s, c; real(kind=jprd), allocatable :: f(:)
    call read_all(self, name, e, f)
    s = 1; c = size(buffer)
    if (present(start)) s = start
    if (present(count)) c = count
    buffer(1:c) = nint(f(s:s+c-1), jpim)
  end subroutine tx_i1

  subroutine tx_r1(self, name, buffer, start, count)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    real(kind=jprd), target, intent(inout) :: buffer(:)
    integer(kind=jpim), intent(in), optional :: start, count
    integer :: e(3), s, c; real(kind=jprd), allocatable :: f(:)
    call read_all(self, name, e, f)
    s = 1; c = size(buffer)
    if (present(start)) s = start
    if (present(count)) c = count
    buffer(1:c) = f(s:s+c-1)
  end subroutine tx_r1

  subroutine tx_r2(self, name, buffer, start, count)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    real(kind=jprd), target, intent(inout) :: buffer(:,:)
    integer(kind=jpim), intent(in), optional :: start(2), count(2)
    integer :: e(3), s(2), c(2); real(kind=jprd), allocatable :: f(:), a(:,:)
    call read_all(self, name, e, f)
    a = reshape(f, e(1:2))
    s = 1; c = shape(buffer)
    if (present(start)) s = start
    if (present(count)) c = count
    buffer(1:c(1), 1:c(2)) = a(s(1):s(1)+c(1)-1, s(2):s(2)+c(2)-1)
  end subroutine tx_r2

  subroutine tx_r3(self, name, buffer, start, count)
    class(hdf5_file) :: self
    character(len=*), intent(in) :: name
    real(kind=jprd), target, intent(inout) :: buffer(:,:,:)
    integer(kind=jpim), intent(in), optional :: start(3), count(3)
    integer :: e(3), s(3), c(3); real(kind=jprd), allocatable :: f(:), a(:,:,:)
    call read_all(self, name, e, f)
    a = reshape(f, e)
    s = 1; c = shape(buffer)
    if (present(start)) s = start
    if (present(count)) c = count
    buffer(1:c(1), 1:c(2), 1:c(3)) = a(s(1):s(1)+c(1)-1, s(2):s(2)+c(2)-1, s(3):s(3)+c(3)-1)
  end subroutine tx_r3

end module hdf5_file_mod
