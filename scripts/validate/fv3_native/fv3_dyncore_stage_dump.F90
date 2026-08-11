! Stage-dump support for the STAGED dyn_core / fv_dynamics copies.
!
! Companion of gen_dyncore_stage_copy.py: the generated staged copies
! insert single-line hooks of the form
!
!   if (stage_dumps_active() .and. it == 1) call stage_dump3('S03_csw_uc', uc)
!
! at pinned line anchors of the ORACLE STAGE-STATE instrument.  The module
! is pure I/O -- no numerics -- and is INERT until the driver calls
! stage_dump_open (so the REAL certification arm, which runs before it,
! is untouched).
!
! Output format matches the extchain instrument: a raw f64 stream .dat
! plus a text manifest (.mf) of `name shape... byte-offset` records and
! free-form NOTE/CERT lines.  Fortran stream write preserves memory
! (column-major) order, so a reader uses numpy order='F'.
module stage_dump_mod
  implicit none
  private
  public :: stage_dump_open, stage_dump_close, stage_dumps_active, &
            stage_dump2, stage_dump3, stage_dump4, stage_mf_note
  integer :: dump_unit = -1, mf_unit = -1
  integer(kind=8) :: dump_off = 0
  logical :: active = .false.
contains
  logical function stage_dumps_active()
    stage_dumps_active = active
  end function stage_dumps_active

  subroutine stage_dump_open(datname, mfname)
    character(len=*), intent(in) :: datname, mfname
    open (newunit=dump_unit, file=trim(datname), access='stream', &
          form='unformatted', status='replace', action='write')
    open (newunit=mf_unit, file=trim(mfname), status='replace', &
          action='write')
    dump_off = 0
    active = .true.
  end subroutine stage_dump_open

  subroutine stage_dump_close()
    if (dump_unit >= 0) close (dump_unit)
    if (mf_unit >= 0) close (mf_unit)
    dump_unit = -1
    mf_unit = -1
    active = .false.
  end subroutine stage_dump_close

  subroutine stage_dump2(name, arr)
    character(len=*), intent(in) :: name
    real, intent(in) :: arr(:, :)
    if (.not. active) return
    write (dump_unit) arr
    write (mf_unit, '(A,2(1X,I6),1X,I6,1X,I6,1X,I15)') trim(name), &
      shape(arr), 1, 1, dump_off
    dump_off = dump_off + int(size(arr), 8)*8
  end subroutine stage_dump2

  subroutine stage_dump3(name, arr)
    character(len=*), intent(in) :: name
    real, intent(in) :: arr(:, :, :)
    if (.not. active) return
    write (dump_unit) arr
    write (mf_unit, '(A,3(1X,I6),1X,I6,1X,I15)') trim(name), shape(arr), &
      1, dump_off
    dump_off = dump_off + int(size(arr), 8)*8
  end subroutine stage_dump3

  subroutine stage_dump4(name, arr)
    character(len=*), intent(in) :: name
    real, intent(in) :: arr(:, :, :, :)
    if (.not. active) return
    write (dump_unit) arr
    write (mf_unit, '(A,4(1X,I6),1X,I15)') trim(name), shape(arr), dump_off
    dump_off = dump_off + int(size(arr), 8)*8
  end subroutine stage_dump4

  subroutine stage_mf_note(text)
    character(len=*), intent(in) :: text
    if (mf_unit < 0) return
    write (mf_unit, '(A)') trim(text)
  end subroutine stage_mf_note
end module stage_dump_mod
