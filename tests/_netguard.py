"""The offline-guard exception, in an importable module.

conftest.py is loaded by pytest but is not importable as `conftest` from a
test module, so the exception type lives here and both import it.
"""


class NetworkUseInOfflineTest(BaseException):
    """A test not marked ``network`` tried to open a connection.

    A ``BaseException``, not an ``Exception``, on purpose: the leak this
    guard exists to catch was invisible because a broad ``except Exception``
    in the size lookup swallowed it (#33). No handler in the package can
    hide this one.
    """
