"""The package version, written here and nowhere else in the Python code.

hatchling reads it when building (``[tool.hatch.version]`` in pyproject.toml),
and the package imports it, so an editable install reports the current
version without being reinstalled.
"""

__version__ = "0.30.1"
