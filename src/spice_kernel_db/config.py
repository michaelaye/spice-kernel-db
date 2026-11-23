"""Configuration for spice-kernel-db: ~/.config/spice-kernel-db/config.toml.

This module neither prints nor prompts (#49): a corrupt or missing file is a
``ConfigError``, and the first-run prompt and the config display live in
``cli/configure.py``.

Since #48 the storage section holds a *list* of kernel roots. Every root is
scanned, resolved against and deduplicated across, and a download lands in
the root that already holds its metakernel or its mission
(``KernelDB.root_for``); no root is special. A config file written before
that holds one ``kernel_dir`` and loads as a one-root list.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DB_PATH = "~/.local/share/spice-kernel-db/kernels.duckdb"
DEFAULT_KERNEL_DIR = "~/.local/share/spice-kernel-db/kernels"

# The pre-0.19 default. 0.19.x and 0.20.x still opened a database found here,
# with a FutureWarning naming 0.21.0 as the cut-off. 0.21.0 is this release, so
# the automatic lookup is gone -- the name stays only so the error below can
# point at the file and say what to do with it.
LEGACY_DB_PATH = "~/.spice_kernels.duckdb"

CONFIG_DIR = Path("~/.config/spice-kernel-db").expanduser()
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    """User configuration for spice-kernel-db.

    ``kernel_dirs`` is the list of kernel roots and is never empty. No root
    is special: where a download lands is derived from what is already on
    disk (``KernelDB.root_for``).
    """

    db_path: str = DEFAULT_DB_PATH
    kernel_dirs: list[str] = field(default_factory=lambda: [DEFAULT_KERNEL_DIR])

    def __post_init__(self) -> None:
        self.kernel_dirs = list(self.kernel_dirs) or [DEFAULT_KERNEL_DIR]


class ConfigError(RuntimeError):
    """The config file is corrupt, or there is none and nothing says what to write.

    Not a ``ValueError``: ``remove_root`` raises those for a bad argument, and an
    ``except ValueError`` around it must not swallow a broken config file.
    """


class LegacyDatabaseFound(RuntimeError):
    """A database sits at the pre-0.19 default and nothing says to use it."""


def default_db_path() -> str:
    """The database path to use when no config file exists: ``DEFAULT_DB_PATH``.

    Until 0.21.0 a database found only at ``LEGACY_DB_PATH`` was opened instead,
    with a ``FutureWarning``. That lookup is gone. Rather than silently create
    an empty database beside a populated legacy one -- which would look like
    data loss -- this raises and names the two commands that fix it.
    """
    if (not Path(DEFAULT_DB_PATH).expanduser().exists()
            and Path(LEGACY_DB_PATH).expanduser().is_file()):
        raise LegacyDatabaseFound(
            f"A spice-kernel-db database exists at the pre-0.19 default "
            f"{LEGACY_DB_PATH}, which is no longer opened automatically "
            f"(the FutureWarning in 0.19.x and 0.20.x said 0.21.0 would stop).\n"
            f"  Either move it:   mv {LEGACY_DB_PATH} {DEFAULT_DB_PATH}\n"
            f"  or record it:     spice-kernel-db config --setup"
        )
    return DEFAULT_DB_PATH


def load_config() -> Config | None:
    """Load config from the TOML file; None when there is no file yet.

    A file that does not parse raises ``ConfigError`` rather than reading as
    "not configured": the next step after None is first-run setup, which
    would overwrite it.
    """
    if not CONFIG_FILE.is_file():
        return None
    try:
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            f"Config file is corrupt: {CONFIG_FILE}\n"
            f"  {e}\n"
            f"  Fix the file by hand or re-run: spice-kernel-db config --setup"
        ) from e
    storage = data.get("storage", {})
    # A file written before #48 holds one kernel_dir.
    roots = storage.get("kernel_dirs") or [storage.get("kernel_dir", DEFAULT_KERNEL_DIR)]
    return Config(
        db_path=data.get("database", {}).get("path", DEFAULT_DB_PATH),
        kernel_dirs=roots,
    )


def save_config(config: Config) -> None:
    """Write config to TOML file (the roots as ``kernel_dirs``, #48)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    roots = ", ".join(f'"{r}"' for r in config.kernel_dirs)
    text = (
        "# spice-kernel-db configuration\n"
        "\n"
        "[database]\n"
        f'path = "{config.db_path}"\n'
        "\n"
        "[storage]\n"
        f"kernel_dirs = [{roots}]\n"
    )
    CONFIG_FILE.write_text(text)


def _same_root(a: str, b: str) -> bool:
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()


def _loaded(config: Config | None) -> Config:
    """*config* as given, else the file, else the defaults (no prompt)."""
    return config if config is not None else (load_config() or Config())


def kernel_roots(config: Config | None = None) -> list[Path]:
    """The configured kernel roots as expanded, resolved paths (#48).

    Without a config file the one default root. Pass *config* to avoid a
    second read of the file.
    """
    return [Path(r).expanduser().resolve() for r in _loaded(config).kernel_dirs]


def add_root(directory: str | Path, config: Config | None = None) -> Config:
    """Append *directory* to the kernel roots and save (#48).

    The directory is created when missing. A root already present is not
    added twice; the order of the others is kept. Runs no
    prompt, so planetarypy can call it; without a config file one is
    written with the default database path.
    """
    config = _loaded(config)
    root = str(Path(directory).expanduser())
    if not any(_same_root(root, r) for r in config.kernel_dirs):
        Path(root).mkdir(parents=True, exist_ok=True)
        config.kernel_dirs.append(root)
    save_config(config)
    return config


def remove_root(directory: str | Path, config: Config | None = None) -> Config:
    """Remove *directory* from the kernel roots and save (#48).

    Raises ``ValueError`` when it is not a root or is the last one; the
    files on disk and the database rows are untouched (``clean`` drops
    rows whose file is gone).
    """
    config = _loaded(config)
    root = str(Path(directory).expanduser())
    kept = [r for r in config.kernel_dirs if not _same_root(root, r)]
    if len(kept) == len(config.kernel_dirs):
        raise ValueError(f"{directory} is not a configured kernel root")
    if not kept:
        raise ValueError(
            f"{directory} is the only kernel root; add another before removing it"
        )
    config.kernel_dirs = kept
    save_config(config)
    return config


def write_first_config(db_path: str, kernel_dir: str) -> Config:
    """Create both directories and write the first config file."""
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(kernel_dir).expanduser().mkdir(parents=True, exist_ok=True)
    config = Config(db_path=db_path, kernel_dirs=[kernel_dir])
    save_config(config)
    return config


def ensure_config(seed_root: str | Path | None = None) -> Config:
    """Load the config, or write a first one around *seed_root*.

    With *seed_root* and no config file yet, write a config whose first
    kernel root is *seed_root* (#48: planetarypy seeds skd's root from its
    own storage root on first run). An existing config is never changed by
    *seed_root*. With no file and no seed there is nothing to write, so
    ``ConfigError`` names the command that asks (#49); the CLI runs that
    prompt itself.
    """
    config = load_config()
    if config is None:
        if seed_root is None:
            raise ConfigError(
                "No configuration yet. Run: spice-kernel-db config --setup"
            )
        config = write_first_config(default_db_path(), str(Path(seed_root).expanduser()))
    return config
