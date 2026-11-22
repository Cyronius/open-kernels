"""Compiler discovery for native probes with GNU-style flags."""
import os
from pathlib import Path
import shutil


def find_cxx():
    """Return a compiler executable, respecting an explicit CXX override."""
    configured = os.environ.get("CXX")
    if configured:
        return shutil.which(configured)
    return shutil.which("c++") or shutil.which("clang++")


def executable_path(path: Path) -> Path:
    return path.with_suffix(".exe") if os.name == "nt" else path
