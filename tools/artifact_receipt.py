#!/usr/bin/env python3
"""Decide what an extension artifact is, and whether two of them carry the same code.

A `.so` receipt is not an md5.  Two identical-source builds differ -- mxcc names
its intermediate `maca_topk-<6 hex>.cpp`, so the embedded name string moves --
and a comment-only source edit moves the embedded *line numbers*.  Neither "the
md5 changed" nor "the md5 is the same" answers the question that matters: **did
the code change**.  This does, from the artifact itself, using the method
`CLAUDE.md` spells out by hand:

    # what is this artifact
    tools/artifact_receipt.py deep_select/deep_select_maca_xcore1600.so

    # did the code move between two builds (the device image is the claim)
    tools/artifact_receipt.py OLD.so NEW.so

    # the tool's own controls (run after changing it)
    tools/artifact_receipt.py --self-test

The identity block is the one a commit message quotes: size, md5, `.text`,
`.mc_fatbin` (size, md5, per-bundle md5), the defined-symbol digest, and the
counts this tree keeps as receipts -- `topk_kernel_radix` and `__device_stub__`
over `.symtab` (`nm -S --defined-only`), `__tvm_ffi_*` over `.dynsym`
(`nm -D`).  The two spellings exist because the first two count device-side
symbols while the third is the exported ABI, and they are separate tables.

The symbol population matches binutils': `STT_FILE` and `STT_SECTION` entries
are skipped, exactly as `nm` skips them.  That is load-bearing rather than
tidiness -- mxcc names each translation unit's intermediate `maca_topk-<6
hex>.cpp` at random, so the artifact carries a file symbol whose name changes
between two builds of identical source.  Keeping it would make every symbol
diff report one added and one removed name; the device image is the claim, and
this is why the bundle comparison is the one that answers it.

The comparison is the device-identity proof: `.mc_fatbin` is a clang offload
container, so it is split into its bundles and compared bundle by bundle --
a byte-identical device-ELF bundle proves the device image did not move even
when host launchers in `.text` did.  Then the defined symbols by (name, size),
then the raw byte difference over the whole file, so an embedded line number is
distinguishable from an instruction.

The **verdict** (`--verdict`) is about that device image: exit 0 when every
device-ELF bundle is byte-identical, 1 otherwise.  Bitcode and host-bundle
differences are printed as facts but are not part of the verdict -- that is why
a `phi` reshuffle the bitcode bundle shows can land with exit 0, and a changed
kernel cannot.

No binutils: the ELF section table, the symbol table and the offload bundles are
parsed here, so a receipt reads the same on any host and `--self-test` builds
its own artifacts rather than needing a build to exist.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import struct
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

FATBIN_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
SHN_UNDEF = 0
SHT_SYMTAB = 2
SHT_DYNSYM = 11
SHT_PROGBITS = 1      # an ordinary section; what .text/.strtab are
# STT_SECTION and STT_FILE.  binutils does not print either, and the difference
# is what makes the receipt's symbol count the same number `nm` reports: mxcc
# leaves one STT_FILE per source file holding its intermediate name, which is
# random per build -- the entry a symbol diff must not mistake for code.
SKIP_SYMBOL_TYPES = (3, 4)

# The counts a commit message quotes, by substring of the defined symbol name.
# The first two are `.symtab` facts (`nm -S --defined-only`); the exported one
# is a `.dynsym` fact (`nm -D`), which is where a shared object's ABI lives.
COUNTS = ("topk_kernel_radix", "__device_stub__")
EXPORTED_COUNT = "__tvm_ffi_"

# How many names of a symbol difference to print before saying "and N more".
DEFAULT_LIMIT = 12


class ReceiptError(Exception):
    """A file that cannot be read as this tool's idea of a built extension."""


# ── ELF ────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Section:
    name: str
    type: int
    offset: int
    size: int
    link: int
    entsize: int


def _take(fmt: str, data: bytes, off: int, where: str):
    size = struct.calcsize(fmt)
    if off < 0 or off + size > len(data):
        raise ReceiptError(
            f"{where}: reads {size} B at {off:#x}, past the end of a "
            f"{len(data)} B file")
    return struct.unpack_from(fmt, data, off)


def _slice(data: bytes, off: int, size: int, what: str) -> bytes:
    if off < 0 or size < 0 or off + size > len(data):
        raise ReceiptError(
            f"{what}: [{off:#x}, {off + size:#x}) is outside a "
            f"{len(data)} B file")
    return data[off:off + size]


def _cstr(blob: bytes, off: int, what: str) -> str:
    if off < 0 or off >= len(blob):
        raise ReceiptError(
            f"{what}: string offset {off:#x} is outside "
            f"a {len(blob)} B string table")
    end = blob.find(b"\0", off)
    if end < 0:
        raise ReceiptError(f"{what}: unterminated string at {off:#x}")
    return blob[off:end].decode("utf-8", "replace")


def elf_sections(data: bytes, where: str) -> List[Section]:
    """The section table, names resolved.  ELF64 little-endian only."""
    if data[:4] != b"\x7fELF":
        raise ReceiptError(f"{where}: not an ELF file (magic {data[:4]!r})")
    if len(data) < 64:
        raise ReceiptError(f"{where}: {len(data)} B is too short for an ELF64 "
                           f"header")
    if data[4] != 2:
        raise ReceiptError(f"{where}: not ELF64 (class {data[4]})")
    if data[5] != 1:
        raise ReceiptError(f"{where}: not little-endian (data encoding {data[5]})")
    shoff = _take("<Q", data, 0x28, where)[0]
    shentsize, shnum, shstrndx = _take("<HHH", data, 0x3A, where)
    if shoff == 0 or shnum == 0:
        raise ReceiptError(f"{where}: carries no section table")
    if shentsize < 64:
        raise ReceiptError(f"{where}: section header size {shentsize} < 64")
    raw = []
    for i in range(shnum):
        off = shoff + i * shentsize
        name, typ = _take("<II", data, off, where)
        offset, size = _take("<QQ", data, off + 0x18, where)
        link = _take("<I", data, off + 0x28, where)[0]
        entsize = _take("<Q", data, off + 0x38, where)[0]
        raw.append([name, typ, offset, size, link, entsize])
    if shstrndx >= shnum:
        raise ReceiptError(f"{where}: section-name table index {shstrndx} "
                           f"is not a section")
    asec = raw[shstrndx]
    names = _slice(data, asec[2], asec[3], f"{where}: section-name table")
    return [Section(_cstr(names, r[0], f"{where}: section name"),
                    r[1], r[2], r[3], r[4], r[5]) for r in raw]


def elf_symbols(data: bytes, sections: List[Section], where: str,
                table_type: int = SHT_SYMTAB) -> List[Tuple[str, int]]:
    """`(name, size)` of every defined symbol in that table, sorted.

    Empty when the table is absent.  Imports skipped: `SKIP_SYMBOL_TYPES` --
    the same two binutils leaves out, which is why the count here is the count
    `nm` prints rather than the raw table length.
    """
    table = next((s for s in sections if s.type == table_type), None)
    if table is None:
        return []
    if table.link >= len(sections):
        raise ReceiptError(f"{where}: symbol table links to section "
                           f"{table.link}, which does not exist")
    strs = sections[table.link]
    strtab = _slice(data, strs.offset, strs.size, f"{where}: symbol strings")
    blob = _slice(data, table.offset, table.size, f"{where}: symbol table")
    entsize = table.entsize or 24
    if entsize < 24:
        raise ReceiptError(f"{where}: symbol entry size {entsize} < 24")
    symbols = []
    for off in range(0, len(blob) - entsize + 1, entsize):
        name = struct.unpack_from("<I", blob, off)[0]
        info = blob[off + 4]
        shndx = struct.unpack_from("<H", blob, off + 6)[0]
        size = struct.unpack_from("<Q", blob, off + 16)[0]
        if shndx == SHN_UNDEF or (info & 0xF) in SKIP_SYMBOL_TYPES:
            continue
        symbols.append((_cstr(strtab, name, f"{where}: symbol name"), size))
    return sorted(symbols)


# ── the clang offload container ────────────────────────────────────────────
def offload_bundles(blob: bytes,
                    where: str) -> Tuple[int, List[Tuple[str, bytes]]]:
    """`(format version, [(triple, payload)])` from a `.mc_fatbin` section."""
    if blob[:len(FATBIN_MAGIC)] != FATBIN_MAGIC:
        raise ReceiptError(f"{where}: `.mc_fatbin` does not open with "
                           f"{FATBIN_MAGIC.decode()}")
    if len(blob) < 32:
        raise ReceiptError(f"{where}: `.mc_fatbin` is {len(blob)} B, too "
                           f"short for a container header")
    version = struct.unpack_from("<Q", blob, 24)[0]
    pos, bundles = 32, []
    while True:
        if pos + 24 > len(blob):
            raise ReceiptError(f"{where}: bundle descriptor at {pos:#x} is "
                               f"truncated")
        off, size, triple_size = struct.unpack_from("<QQQ", blob, pos)
        pos += 24
        if off == 0 and size == 0 and triple_size == 0:
            break
        if off + size > len(blob) or pos + triple_size > len(blob):
            raise ReceiptError(f"{where}: bundle at {pos:#x} escapes the "
                               f"{len(blob)} B section")
        triple = blob[pos:pos + triple_size].decode("utf-8", "replace")
        pos += triple_size
        bundles.append((triple, blob[off:off + size]))
    if not bundles:
        raise ReceiptError(f"{where}: `.mc_fatbin` carries no bundles")
    return version, bundles


def bundle_kind(triple: str) -> str:
    if triple.endswith("-bc"):
        return "bitcode"
    if triple.startswith("host-"):
        return "host"
    return "device"


# ── the artifact ───────────────────────────────────────────────────────────
@dataclasses.dataclass
class Artifact:
    path: str
    size: int
    md5: str
    text: bytes
    fatbin: bytes
    version: int
    bundles: List[Tuple[str, bytes]]
    symbols: List[Tuple[str, int]]       # `.symtab` -- what nm -S prints
    exported: List[Tuple[str, int]]      # `.dynsym` -- what nm -D prints

    @property
    def symbol_digest(self) -> str:
        """Digest of the sorted (name, size) list -- the one-line identity."""
        text = "\n".join(f"{n} {s}" for n, s in self.symbols)
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]

    def counts(self) -> Dict[str, int]:
        return {c: sum(1 for n, _ in self.symbols if c in n) for c in COUNTS}

    def exported_count(self) -> int:
        return sum(1 for n, _ in self.exported if EXPORTED_COUNT in n)


def load(path: str) -> Artifact:
    """Read one artifact, or raise `ReceiptError` saying which part is wrong."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ReceiptError(f"{path}: {e.strerror}") from None
    sections = elf_sections(data, path)
    section = {s.name: s for s in sections}
    for want in (".text", ".mc_fatbin"):
        if want not in section:
            raise ReceiptError(f"{path}: no {want} section -- not a built "
                               f"extension?")
    text = _slice(data, section[".text"].offset, section[".text"].size,
                  f"{path}: .text")
    fatbin = _slice(data, section[".mc_fatbin"].offset,
                    section[".mc_fatbin"].size, f"{path}: .mc_fatbin")
    version, bundles = offload_bundles(fatbin, path)
    return Artifact(path=path, size=len(data),
                    md5=hashlib.md5(data).hexdigest(), text=text,
                    fatbin=fatbin, version=version, bundles=bundles,
                    symbols=elf_symbols(data, sections, path),
                    exported=elf_symbols(data, sections, path, SHT_DYNSYM))


@dataclasses.dataclass
class BundleRow:
    triple: str
    kind: str
    a: Optional[bytes]
    b: Optional[bytes]

    @property
    def same(self) -> bool:
        return self.a is not None and self.b is not None and self.a == self.b


@dataclasses.dataclass
class Comparison:
    a: Artifact
    b: Artifact
    bundles: List[BundleRow]
    added: List[str]
    removed: List[str]
    changed: List[Tuple[str, int, int]]      # name, size in a, size in b
    diff_bytes: int
    first_diff: Optional[int]
    last_diff: Optional[int]
    bundle_sets_differ: bool

    @property
    def device_identical(self) -> bool:
        """Every device-ELF bundle exists on both sides and is byte-identical.

        A container with no device bundle at all is **not** a pass: there
        would be nothing to compare, and "no device code" is a different claim
        from "the device code did not move".
        """
        devices = [row for row in self.bundles if row.kind == "device"]
        return bool(devices) and all(row.same for row in devices)


def byte_diff(a: bytes, b: bytes) -> Tuple[int, Optional[int], Optional[int]]:
    """`(count, first offset, last offset)` of differing bytes in the common prefix."""
    n = min(len(a), len(b))
    count, first, last = 0, None, None
    step = 1 << 16
    for base in range(0, n, step):
        ca, cb = a[base:base + step], b[base:base + step]
        if ca == cb:
            continue
        for i, (x, y) in enumerate(zip(ca, cb)):
            if x != y:
                count += 1
                if first is None:
                    first = base + i
                last = base + i
    return count, first, last


def compare(a: Artifact, b: Artifact) -> Comparison:
    bmap = {t: blob for t, blob in b.bundles}
    rows = [BundleRow(t, bundle_kind(t), blob, bmap.get(t))
            for t, blob in a.bundles]
    seen = {t for t, _ in a.bundles}
    rows += [BundleRow(t, bundle_kind(t), None, blob)
             for t, blob in b.bundles if t not in seen]
    sa, sb = dict(a.symbols), dict(b.symbols)
    diff_bytes, first, last = byte_diff(
        _whole(a), _whole(b))
    return Comparison(
        a=a, b=b, bundles=rows,
        added=sorted(sb.keys() - sa.keys()),
        removed=sorted(sa.keys() - sb.keys()),
        changed=sorted((n, sa[n], sb[n]) for n in sa.keys() & sb.keys()
                       if sa[n] != sb[n]),
        diff_bytes=diff_bytes, first_diff=first, last_diff=last,
        bundle_sets_differ=[t for t, _ in a.bundles] != [t for t, _ in b.bundles])


def _whole(artifact: Artifact) -> bytes:
    """The artifact's bytes, re-read -- only `compare` needs them whole."""
    with open(artifact.path, "rb") as f:
        return f.read()


# ── rendering ──────────────────────────────────────────────────────────────
def _bytes(n: int) -> str:
    return f"{n:,} B"


def render_identity(a: Artifact) -> List[str]:
    out = [f"artifact  {a.path}",
           f"  size       {_bytes(a.size)}",
           f"  md5        {a.md5}",
           f"  .text      {_bytes(len(a.text))}",
           f"  .mc_fatbin {_bytes(len(a.fatbin))}  "
           f"md5 {hashlib.md5(a.fatbin).hexdigest()}  "
           f"(format {a.version}, {len(a.bundles)} bundles)"]
    width = max(len(t) for t, _ in a.bundles)
    for i, (triple, blob) in enumerate(a.bundles):
        out.append(f"  bundle {i}   {triple.ljust(width)}  "
                   f"{_bytes(len(blob)):>12}  "
                   f"md5 {hashlib.md5(blob).hexdigest()}")
    out.append(f"  symbols    {len(a.symbols)} defined  "
               f"digest {a.symbol_digest}")
    counts = "  ".join(f"{c} {n}" for c, n in a.counts().items())
    out.append(f"  counts     {counts}  (nm -S)")
    if a.exported:
        out.append(f"  exported   {len(a.exported)} defined  "
                   f"{EXPORTED_COUNT} {a.exported_count()}  (nm -D)")
    else:
        out.append("  exported   (no .dynsym)")
    return out


def render_comparison(c: Comparison, limit: int) -> List[str]:
    out = [f"compare   A {c.a.path}  {_bytes(c.a.size)}  md5 {c.a.md5}",
           f"          B {c.b.path}  {_bytes(c.b.size)}  md5 {c.b.md5}",
           f"  .text     {_bytes(len(c.a.text))} -> "
           f"{_bytes(len(c.b.text))}"]
    if c.bundle_sets_differ:
        out.append("  .mc_fatbin bundle sets DIFFER -- the container does not "
                   "name the same bundles")
    width = max(len(r.triple) for r in c.bundles)
    out.append(f"  .mc_fatbin bundle by bundle (format "
               f"{c.a.version} -> {c.b.version}):")
    for row in c.bundles:
        if row.a is None:
            verdict = f"only in B ({_bytes(len(row.b))})"
        elif row.b is None:
            verdict = f"only in A ({_bytes(len(row.a))})"
        elif row.same:
            verdict = "identical"
        else:
            n, _, _ = byte_diff(row.a, row.b)
            verdict = (f"differs: {n} B of {_bytes(len(row.a))} "
                       f"(md5 {hashlib.md5(row.a).hexdigest()} -> "
                       f"{hashlib.md5(row.b).hexdigest()})")
        out.append(f"    {row.triple.ljust(width)}  {row.kind:<8}  {verdict}")
    device = [r for r in c.bundles if r.kind == "device"]
    same = sum(1 for r in device if r.same)
    out.append(f"  device images   {'identical' if c.device_identical else 'DIFFER'}"
               f": {same} of {len(device)} device bundle(s) byte-identical")

    def _names(names: List[str], what: str) -> None:
        if not names:
            out.append(f"  symbols    {what} 0")
            return
        shown = names[:limit]
        more = "" if len(names) <= limit else f"  (and {len(names) - limit} more)"
        out.append(f"  symbols    {what} {len(names)}"
                   + "".join(f"\n    {n}" for n in shown) + more)

    out.append(f"  symbols    {len(c.a.symbols)} -> {len(c.b.symbols)} defined")
    _names(c.added, "added")
    _names(c.removed, "removed")
    if c.changed:
        shown = c.changed[:limit]
        more = ("" if len(c.changed) <= limit
                else f"  (and {len(c.changed) - limit} more)")
        out.append(f"  symbols    changed {len(c.changed)}"
                   + "".join(f"\n    {n}  {old:,} -> {new:,} B"
                             for n, old, new in shown) + more)
    else:
        out.append("  symbols    changed 0")
    span = ""
    if c.first_diff is not None:
        span = f" (first at {c.first_diff:#x}, last at {c.last_diff:#x})"
    out.append(f"  bytes      {c.diff_bytes:,} differing of {_bytes(c.a.size)}"
               + (f" -> {_bytes(c.b.size)}" if c.a.size != c.b.size else "")
               + span)
    return out


# ── self-test: synthetic artifacts, no build needed ────────────────────────
def build_fatbin(bundles: List[Tuple[str, bytes]], version: int = 1) -> bytes:
    triples = [t.encode() for t, _ in bundles]
    payloads = [b for _, b in bundles]
    pos = 32 + sum(24 + len(t) for t in triples) + 24     # + the terminator
    offsets = []
    for payload in payloads:
        offsets.append(pos)
        pos += len(payload)
    out = bytearray(FATBIN_MAGIC + struct.pack("<Q", version))
    for triple, payload, off in zip(triples, payloads, offsets):
        out += struct.pack("<QQQ", off, len(payload), len(triple)) + triple
    out += b"\0" * 24
    for payload in payloads:
        out += payload
    return bytes(out)


def build_elf(sections: List[Tuple[str, bytes]],
              symbols: List[Tuple] = (),
              exported: List[Tuple] = ()) -> bytes:
    """A minimal ELF64 this tool (and `readelf`, when present) can read.

    `symbols` fills `.symtab`, `exported` fills `.dynsym`.  An entry is
    `(name, size)` for a defined global, or `(name, size, st_info)` to reach a
    class the reader treats specially (a file or section symbol).
    """
    names = [n for n, _ in sections] + [".symtab", ".strtab", ".dynsym",
                                        ".shstrtab"]
    shstr = b"\0"
    name_off = {}
    for name in names:
        name_off[name] = len(shstr)
        shstr += name.encode() + b"\0"
    strtab = b"\0"

    def table(entries: List[Tuple]) -> bytes:
        nonlocal strtab
        blob = bytearray(b"\0" * 24)
        for entry in entries:
            name, size = entry[0], entry[1]
            info = entry[2] if len(entry) > 2 else 0x10     # GLOBAL NOTYPE
            blob += struct.pack("<IBBHQQ", len(strtab), info, 0, 1, 0, size)
            strtab += name.encode() + b"\0"
        return bytes(blob)

    symtab, dynsym = table(symbols), table(exported)
    body = bytearray(b"\0" * 64)

    def place(blob: bytes) -> int:
        nonlocal body
        while len(body) % 8:
            body.append(0)
        off = len(body)
        body += blob
        return off

    placed = [(name, place(blob), len(blob)) for name, blob in sections]
    sym_off, str_off = place(symtab), place(strtab)
    dyn_off, shstr_off = place(dynsym), place(shstr)
    strtab_index = len(placed) + 2
    shdrs = [b"\0" * 64]
    for name, off, size in placed:
        shdrs.append(struct.pack("<IIQQQQIIQQ", name_off[name], SHT_PROGBITS, 0, 0,
                                 off, size, 0, 0, 8, 0))
    for name, typ, off, blob in ((".symtab", SHT_SYMTAB, sym_off, symtab),
                                 (".strtab", SHT_PROGBITS, str_off, strtab),
                                 (".dynsym", SHT_DYNSYM, dyn_off, dynsym),
                                 (".shstrtab", SHT_PROGBITS, shstr_off, shstr)):
        shdrs.append(struct.pack("<IIQQQQIIQQ", name_off[name], typ, 0, 0, off,
                                 len(blob), strtab_index, 0, 8,
                                 24 if typ != SHT_PROGBITS else 0))
    shoff = place(b"".join(shdrs))
    body[0:16] = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    struct.pack_into("<HHIQQQIHHHHHH", body, 16, 2, 0, 1, 0, 0, shoff, 0, 64,
                     56, 0, 64, len(shdrs), len(placed) + 4)
    return bytes(body)


def _write(path: str, data: bytes) -> str:
    with open(path, "wb") as f:
        f.write(data)
    return path


def _base_artifact(path: str, device_payload: bytes = b"device-image",
                   text: bytes = b"host-code", bc_payload: bytes = b"bitcode",
                   symbols: Optional[List[Tuple]] = None,
                   file_symbol: str = "maca_topk-aaaaaa.cpp") -> str:
    fatbin = build_fatbin([("host-x86_64-unknown-linux-gnu", b""),
                           ("maca-mxc-metax-macahca--xcore1600-bc", bc_payload),
                           ("maca-mxc-metax-macahca--xcore1600", device_payload)])
    if symbols is None:
        symbols = [("topk_kernel_radix<0>", 100),
                   ("__device_stub__topk_kernel_radix<0>", 8),
                   ("helper", 16)]
    # STT_FILE, LOCAL: mxcc names each TU `maca_topk-<6 hex>.cpp` at random, so
    # this entry is the one a symbol diff must be blind to.
    symbols = list(symbols) + [(file_symbol, 0, 0x04)]
    return _write(path, build_elf(
        [(".text", text), (".mc_fatbin", fatbin)], symbols,
        exported=[("__tvm_ffi_topk", 4),
                  ("__tvm_ffi_get_alignment_requirement", 4)]))


def self_test() -> int:
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="artifact_receipt_selftest_")
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    try:
        base = _base_artifact(os.path.join(tmp, "base.so"))
        a = load(base)

        # Break the circularity: the synthetic artifact is the tool's own
        # builder output, so at least one control must read it with something
        # else.  binutils is not a dependency of the tool, only of this check.
        readelf = shutil.which("readelf")
        if readelf:
            check("synthetic artifact is a valid ELF (readelf)",
                  subprocess.run([readelf, "-S", base],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL).returncode == 0)
        else:
            print("SKIP  readelf not installed -- builder cross-check not run")

        check("identity size", a.size == os.path.getsize(base))
        check("identity counts",
              a.counts() == {"topk_kernel_radix": 2, "__device_stub__": 1},
              str(a.counts()))
        check("identity exported", a.exported_count() == 2,
              str(a.exported_count()))
        check("file symbol excluded", len(a.symbols) == 3, str(a.symbols))
        check("identity bundles", [bundle_kind(t) for t, _ in a.bundles]
              == ["host", "bitcode", "device"])

        same = compare(a, a)
        check("same artifact: device identical", same.device_identical)
        check("same artifact: no symbol diff",
              not same.added and not same.removed and not same.changed)
        check("same artifact: no bytes differ", same.diff_bytes == 0,
              str(same.diff_bytes))

        # One byte inside the device-ELF bundle: the expensive mistake.
        dev = _base_artifact(os.path.join(tmp, "dev.so"),
                             device_payload=b"device-imagX")
        c = compare(a, load(dev))
        check("device byte flip: verdict false", not c.device_identical)
        check("device byte flip: only the device bundle moved",
              [r.same for r in c.bundles] == [True, True, False],
              str([(r.triple, r.same) for r in c.bundles]))
        check("device byte flip: symbols unchanged",
              not c.changed and not c.added and not c.removed)

        # One byte in the bitcode bundle: a fact, not a device-image claim.
        bc = _base_artifact(os.path.join(tmp, "bc.so"), bc_payload=b"bitcodX")
        c = compare(a, load(bc))
        check("bitcode flip: verdict true", c.device_identical)
        check("bitcode flip: bitcode bundle differs",
              [r.same for r in c.bundles] == [True, False, True])

        # One byte in .text: the host moved, the device did not.
        tx = _base_artifact(os.path.join(tmp, "text.so"), text=b"host-codX")
        c = compare(a, load(tx))
        check("text flip: verdict true", c.device_identical)
        check("text flip: exactly one byte", c.diff_bytes == 1,
              str(c.diff_bytes))

        # A symbol size change: the device image is untouched, the table is not.
        sy = _base_artifact(os.path.join(tmp, "sym.so"),
                            symbols=[("topk_kernel_radix<0>", 100),
                                     ("__device_stub__topk_kernel_radix<0>", 8),
                                     ("__tvm_ffi_module_version", 4),
                                     ("helper", 24)])
        c = compare(a, load(sy))
        check("symbol resize: verdict true", c.device_identical)
        check("symbol resize: one changed",
              c.changed == [("helper", 16, 24)], str(c.changed))

        # mxcc renames its intermediate per build; that is an STT_FILE entry,
        # and it must not read as a symbol difference (it would, if the reader
        # kept file symbols -- the failure mode this control exists for).
        fn = _base_artifact(os.path.join(tmp, "file.so"),
                            file_symbol="maca_topk-bbbbbb.cpp")
        c = compare(a, load(fn))
        check("mxcc file rename: no symbol diff",
              not c.added and not c.removed and not c.changed,
              str((c.added, c.removed, c.changed)))
        check("mxcc file rename: a byte still differs", c.diff_bytes > 0,
              str(c.diff_bytes))

        # A bundle set that is not the same container.
        two = _write(os.path.join(tmp, "two.so"), build_elf(
            [(".text", b"t"),
             (".mc_fatbin", build_fatbin(
                 [("maca-mxc-metax-macahca--xcore1600", b"d")]))],
            [("x", 1)]))
        c = compare(a, load(two))
        check("bundle set: flagged", c.bundle_sets_differ)
        check("bundle set: verdict false", not c.device_identical)

        # "No device bundle" is a different claim from "the device did not
        # move"; two identical host-only containers must not pass.
        host_only = _write(os.path.join(tmp, "hostonly.so"), build_elf(
            [(".text", b"t"), (".mc_fatbin", build_fatbin(
                [("host-x86_64-unknown-linux-gnu", b"")]))], []))
        c = compare(load(host_only), load(host_only))
        check("no device bundle: not a pass", not c.device_identical)
        short = _write(os.path.join(tmp, "shortfat.so"), build_elf(
            [(".text", b"t"), (".mc_fatbin", FATBIN_MAGIC)], []))
        try:
            load(short)
            check("short container: refused", False)
        except ReceiptError:
            check("short container: refused", True)

        # Malformed inputs are errors with a message, not tracebacks.
        for name, data in (("not elf", b"hello"),
                           ("magic only", b"\x7fELF"),
                           ("short elf", b"\x7fELF" + b"\0" * 10)):
            path = _write(os.path.join(tmp, name.replace(" ", "_") + ".so"), data)
            try:
                load(path)
                check(f"{name}: refused", False)
            except ReceiptError:
                check(f"{name}: refused", True)
        no_fatbin = _write(os.path.join(tmp, "nofat.so"),
                           build_elf([(".text", b"t")], []))
        try:
            load(no_fatbin)
            check("no .mc_fatbin: refused", False)
        except ReceiptError:
            check("no .mc_fatbin: refused", True)
        bad_magic = _write(os.path.join(tmp, "badmagic.so"), build_elf(
            [(".text", b"t"), (".mc_fatbin", b"x" * 40)], []))
        try:
            load(bad_magic)
            check("bad bundle magic: refused", False)
        except ReceiptError:
            check("bad bundle magic: refused", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n{len(results) - len(failed)}/{len(results)} controls passed")
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="the artifact to describe")
    ap.add_argument("other", nargs="?",
                    help="a second artifact: compare instead of describe")
    ap.add_argument("--verdict", action="store_true",
                    help="exit 0 iff every device-ELF bundle is byte-identical")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"symbol names to print per difference "
                         f"(default {DEFAULT_LIMIT})")
    ap.add_argument("--self-test", action="store_true",
                    help="run the tool's own controls")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.path:
        ap.error("give an artifact path, two to compare, or --self-test")
    try:
        if args.other:
            comparison = compare(load(args.path), load(args.other))
            print("\n".join(render_comparison(comparison, args.limit)))
            return 0 if (comparison.device_identical or not args.verdict) else 1
        print("\n".join(render_identity(load(args.path))))
        return 0
    except ReceiptError as e:
        print(f"artifact_receipt: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
