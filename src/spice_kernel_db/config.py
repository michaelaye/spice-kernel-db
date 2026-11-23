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

    Held in ``~/.config/spice-kernel-db/config.toml``. ``kernel_dirs`` is the list of kernel roots
    and is never empty: an empty list becomes the default store. No root is special. Where a
    download lands is derived from what is already on disk (``KernelDB.root_for``).

    The ``[storage]`` section of the file holds ``kernel_dirs``, and ``save_config`` writes only
    that key there. A file written before the list existed has only ``kernel_dir``. It loads as a
    one-root list and is rewritten as ``kernel_dirs`` on the next save.

    Parameters
    ----------
    db_path : str, default "~/.local/share/spice-kernel-db/kernels.duckdb"
        Path to the DuckDB database.
    kernel_dirs : list[str], default ["~/.local/share/spice-kernel-db/kernels"]
        The kernel roots. Never empty: an empty list becomes the default store.

    Examples
    --------
    >>> from spice_kernel_db import Config
    >>> config = Config(kernel_dirs=["~/spice/kernels", "~/planetarypy_data/spice_kernels"])
    >>> config.db_path
    '~/.local/share/spice-kernel-db/kernels.duckdb'
    >>> Config(kernel_dirs=[]).kernel_dirs
    ['~/.local/share/spice-kernel-db/kernels']
    """

    db_path: str = DEFAULT_DB_PATH
    kernel_dirs: list[str] = field(default_factory=lambda: [DEFAULT_KERNEL_DIR])

    def __post_init__(self) -> None:
        self.kernel_dirs = list(self.kernel_dirs) or [DEFAULT_KERNEL_DIR]


class ConfigError(RuntimeError):
    """The config file is corrupt, or there is none and nothing says what to write.

    Raised by ``load_config`` and ``ensure_config`` when the config file does not parse, or when
    there is no file and no ``seed_root`` to write one from. A file that does not parse never reads
    as "not configured".

    This is a subclass of ``RuntimeError``, not of ``ValueError``. ``remove_root`` raises
    ``ValueError`` for a bad argument, and an ``except ValueError`` around it must not swallow a
    broken config file.

    The CLI prints the message and exits with code 78, the conventional code for a configuration
    error (EX_CONFIG). It is outside the 0 to 7 that ``check`` uses as verdict bits.
    ``spice-kernel-db config --setup`` writes a fresh file.
    """


class LegacyDatabaseFound(RuntimeError):
    """A database sits at the pre-0.19 default and nothing says to use it.

    Raised by ``KernelDB()`` when no config file exists and a database sits at the pre-0.19 default
    ``~/.spice_kernels.duckdb`` while the new default does not exist. ``ensure_config`` raises it
    too when it needs the default database path to write a first config. The message names the move
    command. Subclass of ``RuntimeError``.

    Before 0.21.0 such a database was opened with a ``FutureWarning``. Since 0.21.0 it is not,
    rather than quietly starting an empty database beside it.
    """


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

    Without a config file this is the one default root.

    Parameters
    ----------
    config : Config | None, default None
        A configuration to read the roots from. Passing it avoids a second read of the file.

    Returns
    -------
    list[Path]
        The kernel roots.

    See Also
    --------
    add_root, remove_root : Edit the list.
    """
    return [Path(r).expanduser().resolve() for r in _loaded(config).kernel_dirs]


def add_root(directory: str | Path, config: Config | None = None) -> Config:
    """Append *directory* to the kernel roots and save (#48).

    The directory is created when missing. A root that is already present is not added twice, and
    the order of the others is kept. The function runs no prompt, so planetarypy can call it.
    Without a config file one is written with the default database path.

    The CLI equivalent is ``skd config add-root`` (see "Several kernel roots" in the CLI
    documentation).

    Parameters
    ----------
    directory : str | Path
        The directory to add.
    config : Config | None, default None
        A configuration to edit instead of loading the file.

    Returns
    -------
    Config
        The updated configuration.

    Examples
    --------
    >>> from spice_kernel_db import add_root, kernel_roots
    >>> add_root("~/planetarypy_data/spice_kernels")
    >>> kernel_roots()
    [PosixPath('/Users/me/.local/share/spice-kernel-db/kernels'),
     PosixPath('/Users/me/planetarypy_data/spice_kernels')]
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

    Files on disk and database rows are untouched. ``clean`` drops the rows whose file is gone. The
    CLI equivalent is ``skd config remove-root``.

    Parameters
    ----------
    directory : str | Path
        The root to remove.
    config : Config | None, default None
        A configuration to edit instead of loading the file.

    Returns
    -------
    Config
        The updated configuration.

    Raises
    ------
    ValueError
        *directory* is not a configured root, or it is the last one.
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

    With *seed_root* and no config file yet, the function writes a configuration with the default
    database path and *seed_root* as the first kernel root, and creates both directories (#48:
    planetarypy seeds skd's root from its own storage root on first run). An existing configuration
    is never changed by *seed_root*.

    With no file and no *seed_root* there is nothing to write, so ``ConfigError`` names the command
    that asks, ``spice-kernel-db config --setup`` (#49). The CLI runs that prompt itself. Nothing in
    this module prints or prompts: the first-run questions and the ``skd config`` display belong to
    the CLI.

    Parameters
    ----------
    seed_root : str | Path | None, default None
        The first kernel root to write when there is no config file.

    Returns
    -------
    Config
        The configuration read from the file, or the one just written.

    Raises
    ------
    ConfigError
        The file does not parse, or there is no file and no *seed_root*.
    LegacyDatabaseFound
        There is no file, and a database sits at the pre-0.19 default location.
    """
    config = load_config()
    if config is None:
        if seed_root is None:
            raise ConfigError(
                "No configuration yet. Run: spice-kernel-db config --setup"
            )
        config = write_first_config(default_db_path(), str(Path(seed_root).expanduser()))
    return config
