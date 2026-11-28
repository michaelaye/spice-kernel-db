"""Metakernel (.tm) parser and utilities.

SPICE metakernels use a simple text format with \\begindata / \\begintext
blocks containing variable assignments. This module parses the three
key variables: PATH_VALUES, PATH_SYMBOLS, and KERNELS_TO_LOAD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from yarl import URL


@dataclass
class ParsedMetakernel:
    """Parsed representation of a SPICE metakernel (``.tm``) file.

    A dataclass returned by ``parse_metakernel`` and ``parse_metakernel_text``.

    Attributes
    ----------
    source : Path | yarl.URL
        Where the text came from. ``parse_metakernel`` gives the absolute path of the file.
        ``parse_metakernel_text`` gives a ``yarl.URL`` when *source* has a scheme (``https://...``),
        unchanged, and a ``Path`` otherwise.
    header : str
        Everything before the first ``\\begindata``.
    path_values : list[str]
        Contents of ``PATH_VALUES``.
    path_symbols : list[str]
        Contents of ``PATH_SYMBOLS``.
    kernels : list[str]
        Contents of ``KERNELS_TO_LOAD``.

    Notes
    -----
    The properties and methods are:

    ``symbol_map``
        Mapping from symbol name to path value (``dict[str, str]``).
    ``resolve(entry)``
        Replace the ``$SYMBOL`` references in an entry with their values (``str``).
    ``kernel_filenames()``
        Basenames of all kernels (``list[str]``).
    ``kernel_relpaths()``
        Relative paths with ``$SYMBOL`` stripped (``list[str]``).
    """

    source: Path | URL
    header: str
    path_values: list[str] = field(default_factory=list)
    path_symbols: list[str] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)

    @property
    def symbol_map(self) -> dict[str, str]:
        """Mapping from symbol name to its ``PATH_VALUES`` entry.

        The ``PATH_SYMBOLS`` and ``PATH_VALUES`` lists are paired in order. The keys are the symbol
        names as written in ``PATH_SYMBOLS``, without the ``$``.

        Returns
        -------
        dict[str, str]
            Symbol name to path value.
        """
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
        """Replace the ``$SYMBOL`` references in a kernel entry with their ``PATH_VALUES``.

        Parameters
        ----------
        raw_entry : str
            A ``KERNELS_TO_LOAD`` entry, for example ``"$KERNELS/lsk/naif0012.tls"``.

        Returns
        -------
        str
            The entry with every symbol replaced by its value.
        """
        result = raw_entry
        for sym, val in self._ordered_symbols():
            result = result.replace(f"${sym}", val)
        return result

    def kernel_filenames(self) -> list[str]:
        """Return the basenames of all ``KERNELS_TO_LOAD`` entries.

        Returns
        -------
        list[str]
            One basename per entry, in order.
        """
        return [Path(k).name for k in self.kernels]

    def kernel_relpaths(self) -> list[str]:
        """Return the relative path of each kernel, with the ``$SYMBOL`` stripped.

        For example ``'$KERNELS/lsk/naif0012.tls'`` becomes ``'lsk/naif0012.tls'``. H3: the longest
        symbols are processed first, to avoid a prefix collision.

        Returns
        -------
        list[str]
            One relative path per entry, in order.
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


def _parse_text(text: str, source: Path | URL) -> ParsedMetakernel:
    """Parse metakernel content from text.

    Handles multiple ``\\begindata`` / ``\\begintext`` blocks as per
    SPICE spec, and parses string lists with full awareness of the
    ``''`` quote-escape and embedded ``)`` characters (H2).
    """
    header, data_text = _split_data_regions(text)
    assignments = _extract_assignments(data_text)
    return ParsedMetakernel(
        source=source,
        header=header,
        path_values=assignments.get("PATH_VALUES", []),
        path_symbols=assignments.get("PATH_SYMBOLS", []),
        kernels=assignments.get("KERNELS_TO_LOAD", []),
    )


def parse_metakernel(path: str | Path) -> ParsedMetakernel:
    """Parse a SPICE metakernel (``.tm``) file from a local path, without needing a database.

    Handles multiple ``\\begindata`` / ``\\begintext`` blocks as the SPICE specification describes.

    Parameters
    ----------
    path : str | Path
        The ``.tm`` file.

    Returns
    -------
    ParsedMetakernel
        The parsed ``PATH_VALUES``, ``PATH_SYMBOLS`` and ``KERNELS_TO_LOAD``, with the header.
        ``source`` is the absolute path of the file.

    Raises
    ------
    FileNotFoundError
        *path* does not exist.

    Examples
    --------
    >>> from spice_kernel_db import parse_metakernel
    >>> mk = parse_metakernel("demo.tm")  # a file with the two kernels of the next example
    >>> mk.symbol_map
    {'KERNELS': '/data/kernels'}
    >>> mk.kernel_filenames()
    ['naif0012.tls', 'jup365.bsp']
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    return _parse_text(text, path.resolve())


def parse_metakernel_text(text: str, source: str) -> ParsedMetakernel:
    """Parse a SPICE metakernel from text content.

    Use this for a metakernel fetched from a remote URL, where no local file exists.

    Parameters
    ----------
    text : str
        The metakernel text content.
    source : str
        Where the text came from, usually the URL it was fetched from. It is stored as the
        ``source`` attribute of the result: a ``yarl.URL`` when it has a scheme, a ``Path``
        otherwise.

    Returns
    -------
    ParsedMetakernel
        The parsed ``PATH_VALUES``, ``PATH_SYMBOLS`` and ``KERNELS_TO_LOAD``, with the header.

    Examples
    --------
    >>> from spice_kernel_db import parse_metakernel_text
    >>> lines = [
    ...     "KPL/MK",
    ...     "\\\\begindata",
    ...     "  PATH_VALUES  = ( '/data/kernels' )",
    ...     "  PATH_SYMBOLS = ( 'KERNELS' )",
    ...     "  KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls' '$KERNELS/spk/jup365.bsp' )",
    ...     "\\\\begintext",
    ... ]
    >>> mk = parse_metakernel_text("\\n".join(lines), "https://example.org/mk/demo.tm")
    >>> mk.symbol_map
    {'KERNELS': '/data/kernels'}
    >>> mk.kernel_relpaths()
    ['lsk/naif0012.tls', 'spk/jup365.bsp']
    >>> mk.resolve(mk.kernels[0])
    '/data/kernels/lsk/naif0012.tls'
    >>> str(mk.source)
    'https://example.org/mk/demo.tm'
    """
    url = URL(source)
    return _parse_text(text, url if url.scheme else Path(source))


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
    except Exception:  # broad on purpose: cleanup only, re-raised below
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
