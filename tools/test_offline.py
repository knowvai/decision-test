"""Run the selected regression suite with network connections disabled."""
import os
from pathlib import Path
import sys


for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)
os.environ.update(LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network disabled during offline regression tests")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    sys.addaudithook(deny_network)
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:] or ["-q"]))
