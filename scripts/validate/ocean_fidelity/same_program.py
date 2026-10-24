"""Shared byte/ELF comparison used by NEMO same-program gates."""

from __future__ import annotations

import struct

BUILD_CONFIG_FILES = ("arch_nemo.fcm", "cpp.fcm", "bld.cfg")


def _elf_view(blob: bytes, root: bytes, ref_root: bytes) -> tuple:
    """Return a path-normalised, otherwise byte-exact ELF64 object view."""
    if not (blob[:6] == b"\x7fELF\x02\x01" and blob[16] == 1):
        raise ValueError("not an ELF64-LE relocatable")
    (shoff,) = struct.unpack_from("<Q", blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    heads = [
        struct.unpack_from("<IIQQQQIIQQ", blob, shoff + k * shentsize)
        for k in range(shnum)
    ]
    body = [blob[h[4] : h[4] + h[5]] if h[1] != 8 else b"" for h in heads]
    strings = {
        k: body[k].replace(root, ref_root)
        for k, h in enumerate(heads)
        if h[1] == 1 and h[2] & 0x30 == 0x30
    }

    def name(table: int, at: int) -> bytes:
        return body[table][at : body[table].index(b"\0", at)]

    view: list[object] = [blob[:0x28] + blob[0x30:0x40]]
    for k, (nm, kind, flags, addr, _off, size, link, info, align, entsize) in enumerate(heads):
        content: object = body[k]
        if k in strings:
            content = [item for item in strings[k].split(b"\0") if item]
        elif kind == 2:
            content = []
            for at in range(0, size, 24):
                symbol = list(struct.unpack_from("<IBBHQQ", body[k], at))
                if symbol[3] in strings:
                    symbol[4] = name(symbol[3], symbol[4]).replace(root, ref_root)
                content.append(tuple(symbol))
        view.append(
            (
                name(shstrndx, nm), kind, flags, addr, link, info, align,
                entsize, size if kind == 8 else None, content,
            )
        )
    return tuple(view)


def objects_equal(ref: bytes, new: bytes, ref_root: bytes, new_root: bytes) -> bool:
    """Compare objects exactly, allowing only their embedded build-root spelling."""
    if ref == new:
        return True
    try:
        return _elf_view(ref, ref_root, ref_root) == _elf_view(new, new_root, ref_root)
    except (struct.error, ValueError, IndexError):
        return False
