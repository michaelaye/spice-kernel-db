"""The offline-guard exception, in an importable module.

conftest.py is loaded by pytest but is not importable as `conftest` from a
test module, so the exception type lives here and both import it.
"""


class NetworkUseInOfflineTest(RuntimeError):
    """A test not marked ``network`` tried to open a connection."""
