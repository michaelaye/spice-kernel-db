"""Configuration management and first-run setup for spice-kernel-db.

Stores user preferences in ~/.config/spice-kernel-db/config.toml.
On first use, prompts interactively for database and kernel storage paths.
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


@dataclass
class Config:
    """User configuration for spice-kernel-db."""

    db_path: str = DEFAULT_DB_PATH
    kernel_dir: str = DEFAULT_KERNEL_DIR


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
    return Config(
        db_path=data.get("database", {}).get("path", DEFAULT_DB_PATH),
        kernel_dir=data.get("storage", {}).get("kernel_dir", DEFAULT_KERNEL_DIR),
    )


def save_config(config: Config) -> None:
    """Write config to TOML file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    text = (
        "# spice-kernel-db configuration\n"
        "\n"
        "[database]\n"
        f'path = "{config.db_path}"\n'
        "\n"
        "[storage]\n"
        f'kernel_dir = "{config.kernel_dir}"\n'
    )
    CONFIG_FILE.write_text(text)


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

    # Create directories
    Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(kernel_dir).expanduser().mkdir(parents=True, exist_ok=True)

    config = Config(db_path=db_path, kernel_dir=kernel_dir)
    save_config(config)
    print(f"\n  Config saved to {CONFIG_FILE}")
    return config


def show_config(config: Config) -> None:
    """Print current configuration."""
    print(f"\nspice-kernel-db configuration\n")
    print(f"  Config file:   {CONFIG_FILE}")
    print(f"  Database:      {config.db_path}")
    print(f"  Kernel dir:    {config.kernel_dir}")
    print()


def ensure_config() -> Config:
    """Load config or run interactive setup if first time."""
    config = load_config()
    if config is None:
        config = setup_interactive()
    return config
