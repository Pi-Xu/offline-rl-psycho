import os
import sys


def pytest_sessionstart(session):  # pragma: no cover - test harness setup
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

