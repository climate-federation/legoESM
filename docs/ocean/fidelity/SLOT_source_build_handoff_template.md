# SLOT NEMO source-build handoff template

This copy preamble is mandatory for future held NEMO SLOT build blocks. Never
use `cp -a <NEMO-root>/.`: oracle roots can contain tens of GiB of run archives.

Before the block is frozen, the lane must name:

- a lean `/tmp/nemo-*` core/config donor;
- the complete ordered cumulative source/patch stack needed to reproduce the
  OFF stream set, with a committed SHA manifest for any materialized source
  snapshot;
- every new patch, in application order and with committed SHA-256;
- the expected shared/new stream counts used by the later bracket.

Canonical preamble:

```bash
set -euo pipefail
cd /tmp/codex-zdf-sweep
repo=/tmp/codex-zdf-sweep
base=/tmp/nemo-__LEAN_DONOR__
oracle_root=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
copy_guard=$repo/scripts/validate/ocean_fidelity/dino_1226/safe_copy_nemo_source.sh
test "$(sha256sum "$copy_guard" | awk '{print $1}')" = __COPY_GUARD_SHA256__

# Freeze the donor itself, not only files that a later overlay replaces.
test "$(git -C "$base" rev-parse HEAD)" = __LEAN_DONOR_COMMIT__
test -z "$(git -C "$base" status --porcelain --untracked-files=no)"
test "$(sha256sum "$base/cfgs/DINO/cpp_DINO.fcm" | awk '{print $1}')" = __CPP_DINO_SHA256__
test "$(sha256sum "$base/arch/arch-conda.fcm" | awk '{print $1}')" = __ARCH_SHA256__

# Three pre-copy red controls. Each destination must remain empty.
reject_path=$(mktemp -d /tmp/nemo-__LANE__-reject-path.XXXXXX)
if NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
    "$copy_guard" "$oracle_root" "$reject_path"; then
  echo 'path red control unexpectedly copied the oracle root' >&2; exit 2
fi
test -z "$(find "$reject_path" -mindepth 1 -print -quit)"
rmdir "$reject_path"

reject_size=$(mktemp -d /tmp/nemo-__LANE__-reject-size.XXXXXX)
if NEMO_SOURCE_BASE_MAX_MIB=1 NEMO_SOURCE_COPY_MAX_MIB=512 NEMO_SOURCE_MIN_FREE_MIB=4096 \
    "$copy_guard" "$base" "$reject_size"; then
  echo 'size red control unexpectedly copied an over-cap donor' >&2; exit 2
fi
test -z "$(find "$reject_size" -mindepth 1 -print -quit)"
rmdir "$reject_size"

reject_free=$(mktemp -d /tmp/nemo-__LANE__-reject-free.XXXXXX)
filesystem_total_mib=$(df -Pm -- "$reject_free" | awk 'NR==2 {print $2}')
if NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 \
    NEMO_SOURCE_MIN_FREE_MIB=$((filesystem_total_mib + 1)) \
    "$copy_guard" "$base" "$reject_free"; then
  echo 'free-space red control unexpectedly copied below its floor' >&2; exit 2
fi
test -z "$(find "$reject_free" -mindepth 1 -print -quit)"
rmdir "$reject_free"

nemo_src=$(mktemp -d /tmp/nemo-__LANE__-src.XXXXXX)
NEMO_SOURCE_BASE_MAX_MIB=3072 NEMO_SOURCE_COPY_MAX_MIB=512 \
  NEMO_SOURCE_MIN_FREE_MIB=4096 "$copy_guard" "$base" "$nemo_src"

# Reconstruct and verify the FULL cumulative OFF source stack here. A pristine
# tree plus only the newest patch is invalid. Bind a committed manifest before
# applying the ordered new patches with `patch --fuzz=0`.
```

The guard refuses non-`/tmp/nemo-*` donors, oversized donors, insufficient
free space, nonempty/non-`/tmp` destinations, copied `RUN_*`/`BLD`/`WORK`
directories, and post-copy trees above its cap. A held block may tighten its
environment overrides but must not raise them without an explicit preregistered
reason. Its red controls must prove an oracle-root path and an over-cap donor
both fail before the destination gains a file. They must also force a
free-space floor above the currently available space and prove that rejection
is likewise pre-copy. Supply exact limits on the success path and every red
control so ambient variables cannot relax the safety contract.
