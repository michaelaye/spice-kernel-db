"""Metakernel (.tm) parser and utilities.

SPICE metakernels use a simple text format with \\begindata / \\begintext
blocks containing variable assignments. This module parses the three
key variables: PATH_VALUES, PATH_SYMBOLS, and KERNELS_TO_LOAD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedMetakernel:
    """Parsed representation of a SPICE metakernel (.tm) file."""

    source_path: Path
    header: str
    path_values: list[str] = field(default_factory=list)
    path_symbols: list[str] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)

    @property
    def symbol_map(self) -> dict[str, str]:
        """Map from $SYMBOL to its PATH_VALUE."""
        return dict(zip(self.path_symbols, self.path_values))

    def _ordered_symbols(self) -> list[tuple[str, str]]:
        """Return (symbol, value) pairs sorted longest-symbol-first.

        H3: naive ``str.replace`` order makes ``$KERNELS_DATA`` get
        clobbered when ``$KERNELS`` is processed first. Sorting by
        descending symbol length ensures the longest match wins.
        """
        return sorted(
            zip(self.path_symbols, self.path_values),
            key=lambda kv: -len(kv[0]),
        )

    def resolve(self, raw_entry: str) -> str:
        """Replace $SYMBOLs in a kernel entry with their PATH_VALUES."""
        result = raw_entry
        for sym, val in self._ordered_symbols():
            result = result.replace(f"${sym}", val)
        return result

    def kernel_filenames(self) -> list[str]:
        """Return just the basenames of all KERNELS_TO_LOAD entries."""
        return [Path(k).name for k in self.kernels]

    def kernel_relpaths(self) -> list[str]:
        """Return the relative paths (with $SYMBOL stripped) for each kernel.

        E.g. '$KERNELS/lsk/naif0012.tls' -> 'lsk/naif0012.tls'
        H3: process longest symbols first to avoid prefix collision.
        """
        result = []
        ordered = [s for s, _ in self._ordered_symbols()]
        for raw in self.kernels:
            rel = raw
            for sym in ordered:
                rel = rel.replace(f"${sym}/", "").replace(f"${sym}", "")
            result.append(rel)
        return result


def _split_data_regions(text: str) -> tuple[str, str]:
    """Split a metakernel into (header, combined_data) using a real
    line-aware state machine (H2).

    Previously a naive ``re.split(r'\\\\begindata', text)`` was used, which
    happily fired on any literal ``\\begindata`` substring — including
    examples inside ``\\begintext`` comment blocks. SPICE's spec is
    explicit that these markers must appear at the start of a line.
    """
    header_parts: list[str] = []
    data_parts: list[str] = []
    in_data = False
    saw_data = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("\\begindata"):
            in_data = True
            saw_data = True
            continue
        if stripped.startswith("\\begintext"):
            in_data = False
            continue
        if not saw_data:
            header_parts.append(line)
        elif in_data:
            data_parts.append(line)
    return "".join(header_parts), "".join(data_parts)


def _parse_quoted_string(text: str, i: int) -> tuple[str, int]:
    """Parse a SPICE single-quoted string starting at ``text[i] == "'"``.

    Honors the ``''`` escape (a doubled single quote inside a string
    means a literal ``'``). Returns ``(value, index_after_closing_quote)``.
    """
    assert text[i] == "'"
    buf: list[str] = []
    i += 1
    while i < len(text):
        c = text[i]
        if c == "'":
            if i + 1 < len(text) and text[i + 1] == "'":
                buf.append("'")
                i += 2
                continue
            return "".join(buf), i + 1
        buf.append(c)
        i += 1
    # Unterminated string — return what we have
    return "".join(buf), i


def _parse_paren_string_list(text: str, start: int) -> tuple[list[str], int]:
    """Parse ``( 'a' 'b' ... )`` starting at ``text[start] == "("``.

    Skips arbitrary whitespace and SPICE's ``+`` continuation markers,
    and properly handles strings containing ``)`` or escaped quotes.
    Returns ``(values, index_after_closing_paren)``.
    """
    assert text[start] == "("
    i = start + 1
    out: list[str] = []
    while i < len(text):
        c = text[i]
        if c.isspace() or c == "+":
            i += 1
            continue
        if c == ")":
            return out, i + 1
        if c == "'":
            val, i = _parse_quoted_string(text, i)
            out.append(val)
            continue
        # Any other character (e.g. a comma in some flavours) — skip
        i += 1
    return out, len(text)


_VAR_RE = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*")


def _extract_assignments(data_text: str) -> dict[str, list[str]]:
    """Parse all ``VAR = ...`` assignments in *data_text*.

    Handles both list (``VAR = ( 'a' 'b' )``) and scalar string
    (``VAR = 'a'``) forms. Returns a dict mapping variable name to a
    list of string values (scalars become a one-element list).
    """
    out: dict[str, list[str]] = {}
    for m in _VAR_RE.finditer(data_text):
        var = m.group(1)
        i = m.end()
        # Skip leading whitespace
        while i < len(data_text) and data_text[i] in " \t":
            i += 1
        if i >= len(data_text):
            continue
        if data_text[i] == "(":
            vals, _ = _parse_paren_string_list(data_text, i)
            out[var] = vals
        elif data_text[i] == "'":
            val, _ = _parse_quoted_string(data_text, i)
            out[var] = [val]
        # else: numeric or unknown — not what KERNELS_TO_LOAD/PATH_*
        # ever look like; ignore.
    return out


def _parse_text(text: str, source_path: Path) -> ParsedMetakernel:
    """Parse metakernel content from text.

    Handles multiple ``\\begindata`` / ``\\begintext`` blocks as per
    SPICE spec, and parses string lists with full awareness of the
    ``''`` quote-escape and embedded ``)`` characters (H2).
    """
    header, data_text = _split_data_regions(text)
    assignments = _extract_assignments(data_text)
    return ParsedMetakernel(
        source_path=source_path,
        header=header,
        path_values=assignments.get("PATH_VALUES", []),
        path_symbols=assignments.get("PATH_SYMBOLS", []),
        kernels=assignments.get("KERNELS_TO_LOAD", []),
    )


def parse_metakernel(path: str | Path) -> ParsedMetakernel:
    """Parse a SPICE metakernel (.tm) file.

    Handles multiple \\begindata / \\begintext blocks as per SPICE spec.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    return _parse_text(text, path.resolve())


def parse_metakernel_text(text: str, source: str) -> ParsedMetakernel:
    """Parse a SPICE metakernel from text content.

    Use this for metakernels fetched from remote URLs where no local
    file exists.

    Args:
        text: The metakernel text content.
        source: Source identifier (e.g. URL) stored as source_path.
    """
    return _parse_text(text, Path(source))


def write_metakernel(
    parsed: ParsedMetakernel,
    output: str | Path,
    *,
    path_values: list[str] | None = None,
    path_symbols: list[str] | None = None,
    kernels: list[str] | None = None,
) -> Path:
    """Write a metakernel file, optionally overriding specific fields.

    The header (comments) from the original are preserved. Only the
    \\begindata block is rewritten.

    Args:
        parsed: The original parsed metakernel (used for header/defaults).
        output: Destination path.
        path_values: Override PATH_VALUES (default: keep original).
        path_symbols: Override PATH_SYMBOLS (default: keep original).
        kernels: Override KERNELS_TO_LOAD entries (default: keep original).

    Returns:
        The output path.
    """
    pv = path_values or parsed.path_values
    ps = path_symbols or parsed.path_symbols
    kl = kernels or parsed.kernels

    lines = [parsed.header.rstrip(), ""]
    lines.append("\\begindata")
    lines.append("")

    if ps and pv:
        pv_str = "\n".join(f"    '{v}'" for v in pv)
        lines.append(f"  PATH_VALUES  = (\n{pv_str}\n  )")
        lines.append("")
        ps_str = "\n".join(f"    '{s}'" for s in ps)
        lines.append(f"  PATH_SYMBOLS = (\n{ps_str}\n  )")
        lines.append("")

    lines.append("  KERNELS_TO_LOAD = (")
    lines.append("")
    for entry in kl:
        lines.append(f"    '{entry}'")
    lines.append("")
    lines.append("  )")
    lines.append("")
    lines.append("\\begintext")
    lines.append("")

    out = Path(output)
    _atomic_write_text(out, "\n".join(lines))
    return out


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via tmp+rename (C7).

    Without this, a concurrent reader (e.g. SPICE ``furnsh`` mid-load)
    can observe the file truncated to zero bytes while ``write_text``
    runs, and silently load an empty metakernel.
    """
    import os
    import uuid
    tmp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(content)
        # fsync the tmp file's contents before rename, so a power loss
        # between rename and the next sync doesn't yield an empty file.
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
