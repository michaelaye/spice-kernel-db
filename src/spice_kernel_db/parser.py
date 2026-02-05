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

    def resolve(self, raw_entry: str) -> str:
        """Replace $SYMBOLs in a kernel entry with their PATH_VALUES."""
        result = raw_entry
        for sym, val in zip(self.path_symbols, self.path_values):
            result = result.replace(f"${sym}", val)
        return result

    def kernel_filenames(self) -> list[str]:
        """Return just the basenames of all KERNELS_TO_LOAD entries."""
        return [Path(k).name for k in self.kernels]

    def kernel_relpaths(self) -> list[str]:
        """Return the relative paths (with $SYMBOL stripped) for each kernel.

        E.g. '$KERNELS/lsk/naif0012.tls' -> 'lsk/naif0012.tls'
        """
        result = []
        for raw in self.kernels:
            rel = raw
            for sym in self.path_symbols:
                rel = rel.replace(f"${sym}/", "").replace(f"${sym}", "")
            result.append(rel)
        return result


def _parse_text(text: str, source_path: Path) -> ParsedMetakernel:
    """Parse metakernel content from text.

    Handles multiple \\begindata / \\begintext blocks as per SPICE spec.
    """
    # Split on \begindata blocks — collect all data sections
    data_sections: list[str] = []
    parts = re.split(r"\\begindata", text)
    header = parts[0]
    for part in parts[1:]:
        end = part.find("\\begintext")
        if end != -1:
            data_sections.append(part[:end])
        else:
            data_sections.append(part)

    combined = "\n".join(data_sections)

    def _extract_list(varname: str) -> list[str]:
        """Extract a SPICE-style list assignment: VAR = ( 'a' 'b' ... )"""
        pattern = rf"{varname}\s*=\s*\((.*?)\)"
        m = re.search(pattern, combined, re.DOTALL)
        if not m:
            return []
        block = m.group(1)
        return re.findall(r"'([^']*)'", block)

    return ParsedMetakernel(
        source_path=source_path,
        header=header,
        path_values=_extract_list("PATH_VALUES"),
        path_symbols=_extract_list("PATH_SYMBOLS"),
        kernels=_extract_list("KERNELS_TO_LOAD"),
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
    out.write_text("\n".join(lines))
    return out
