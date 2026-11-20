"""Configuration management and first-run setup for spice-kernel-db.

Stores user preferences in ~/.config/spice-kernel-db/config.toml.
On first use, prompts interactively for database and kernel storage paths.

Since #48 the storage section holds a *list* of kernel roots. The first is
where ``get`` downloads; every root is scanned, resolved against and
deduplicated across. A config file written before that holds one
``kernel_dir`` and loads as a one-root list.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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


@dataclass(init=False)
class Config:
    """User configuration for spice-kernel-db.

    ``kernel_dirs`` is the ordered list of kernel roots; ``kernel_dir`` is
    the first of them (the download target) and stays settable, so code
    written for one root keeps working.
    """

    db_path: str
    kernel_dirs: list[str]

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        kernel_dir: str | None = None,
        kernel_dirs: list[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.kernel_dirs = list(kernel_dirs) if kernel_dirs else [kernel_dir or DEFAULT_KERNEL_DIR]

    @property
    def kernel_dir(self) -> str:
        """The first root: where ``get`` downloads."""
        return self.kernel_dirs[0]

    @kernel_dir.setter
    def kernel_dir(self, value: str) -> None:
        self.kernel_dirs[0] = value


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
    """Load config from TOML file. Returns None if not configured yet."""
    if not CONFIG_FILE.is_file():
        return None
    try:
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        import sys
        print(
            f"Error: config file is corrupt: {CONFIG_FILE}\n"
            f"  {e}\n"
            f"  Fix the file manually or re-run: spice-kernel-db config --setup",
            file=sys.stderr,
        )
        return None
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


def add_kernel_root(directory: str | Path, config: Config | None = None) -> Config:
    """Append *directory* to the kernel roots and save (#48).

    The directory is created when missing. A root already present is not
    added twice; the first root (the download target) never moves. Runs no
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


def remove_kernel_root(directory: str | Path, config: Config | None = None) -> Config:
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


def setup_interactive() -> Config:
    """Run first-time interactive setup.

    Prompts for database location and kernel storage directory,
    creates directories, saves config.
    """
    print("spice-kernel-db: First-time setup\n")

    db_path = input(f"  Database location [{DEFAULT_DB_PATH}]: ").strip()
    if not db_path:
        db_path = DEFAULT_DB_PATH

    kernel_dir = input(f"  Kernel storage directory [{DEFAULT_KERNEL_DIR}]: ").strip()
    if not kernel_dir:
        kernel_dir = DEFAULT_KERNEL_DIR

    config = _write_first_config(db_path, kernel_dir)
    print(f"\n  Config saved to {CONFIG_FILE}")
    return config


def _write_first_config(db_path: str, kernel_dir: str) -> Config:
    """Create both directories and write the first config file."""
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(kernel_dir).expanduser().mkdir(parents=True, exist_ok=True)
    config = Config(db_path=db_path, kernel_dir=kernel_dir)
    save_config(config)
    return config


def show_config(config: Config) -> None:
    """Print current configuration."""
    print("\nspice-kernel-db configuration\n")
    print(f"  Config file:   {CONFIG_FILE}")
    print(f"  Database:      {config.db_path}")
    for i, root in enumerate(config.kernel_dirs):
        label = "Kernel roots:  " if i == 0 else "               "
        note = "  (download target)" if i == 0 and len(config.kernel_dirs) > 1 else ""
        print(f"  {label}{root}{note}")
    print()


def ensure_config(seed_root: str | Path | None = None) -> Config:
    """Load config or run interactive setup if first time.

    With *seed_root* and no config file yet, write a config whose first
    kernel root is *seed_root* instead of prompting (#48: planetarypy seeds
    skd's root from its own storage root on first run). An existing config
    is never changed by *seed_root*.
    """
    config = load_config()
    if config is None:
        if seed_root is not None:
            config = _write_first_config(default_db_path(), str(Path(seed_root).expanduser()))
        else:
            config = setup_interactive()
    return config
