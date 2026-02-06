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

CONFIG_DIR = Path("~/.config/spice-kernel-db").expanduser()
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    """User configuration for spice-kernel-db."""

    db_path: str = DEFAULT_DB_PATH
    kernel_dir: str = DEFAULT_KERNEL_DIR


def load_config() -> Config | None:
    """Load config from TOML file. Returns None if not configured yet."""
    if not CONFIG_FILE.is_file():
        return None
    with open(CONFIG_FILE, "rb") as f:
        data = tomllib.load(f)
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
