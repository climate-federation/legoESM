#!/usr/bin/env bash
# Copy a lean NEMO source donor without copying run/build archives.
# Mandatory for held SLOT build blocks: never use `cp -a <nemo-root>/.`.
set -euo pipefail

if test "$#" -ne 2; then
  echo "usage: $0 LEAN_NEMO_BASE EMPTY_DESTINATION" >&2
  exit 2
fi

source_root=$(realpath "$1")
destination=$(realpath "$2")
max_base_mib=${NEMO_SOURCE_BASE_MAX_MIB:-3072}
max_copy_mib=${NEMO_SOURCE_COPY_MAX_MIB:-512}
min_free_mib=${NEMO_SOURCE_MIN_FREE_MIB:-4096}

case "$source_root" in
  /tmp/nemo-*) ;;
  *) echo "refusing non-lean/non-/tmp NEMO base: $source_root" >&2; exit 2;;
esac
case "$destination" in
  /tmp/*) ;;
  *) echo "destination must be an explicit /tmp path: $destination" >&2; exit 2;;
esac
test -d "$source_root/cfgs/DINO/MY_SRC"
test -d "$destination"
if test -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)"; then
  echo "destination is not empty: $destination" >&2
  exit 2
fi

base_mib=$(du -sm --one-file-system -- "$source_root" | awk '{print $1}')
free_mib=$(df -Pm -- "$destination" | awk 'NR==2 {print $4}')
if test "$base_mib" -gt "$max_base_mib"; then
  echo "NEMO base is ${base_mib} MiB; cap is ${max_base_mib} MiB" >&2
  exit 2
fi
if test "$free_mib" -lt "$min_free_mib"; then
  echo "only ${free_mib} MiB free; require ${min_free_mib} MiB before copy" >&2
  exit 2
fi

tar -C "$source_root" \
  --one-file-system \
  --exclude='./.git' \
  --exclude='./cfgs/*/RUN_*' \
  --exclude='./cfgs/*/BLD' \
  --exclude='./cfgs/*/WORK' \
  --exclude='./cfgs/BLD' \
  --exclude='./cfgs/WORK' \
  -cf - . | tar -C "$destination" -xf -

if find "$destination/cfgs" -type d \( -name 'RUN_*' -o -name BLD -o -name WORK \) \
    -print -quit | grep -q .; then
  echo "excluded run/build directory appeared in destination" >&2
  exit 2
fi
copy_mib=$(du -sm --one-file-system -- "$destination" | awk '{print $1}')
if test "$copy_mib" -gt "$max_copy_mib"; then
  echo "guarded copy is ${copy_mib} MiB; cap is ${max_copy_mib} MiB" >&2
  exit 2
fi

printf 'SAFE_NEMO_SOURCE_COPY source=%s base_mib=%s copied_mib=%s free_before_mib=%s destination=%s\n' \
  "$source_root" "$base_mib" "$copy_mib" "$free_mib" "$destination"
