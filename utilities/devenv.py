"""Discover and configure the host tools used by the kernel developer scripts."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
VENV = Path(os.environ.get("OFLM_VENV_DIR") or (
    r"C:\dev\mlir-aie\ironenv" if WINDOWS else ROOT / "ironvenv"))
XRT_ROOT = Path(os.environ.get("XRT_ROOT") or os.environ.get("XILINX_XRT") or (
    r"C:\Xilinx\XRT" if WINDOWS else "/opt/xilinx/xrt"))
XRT_BIN = XRT_ROOT / "bin"
XRT_PY = XRT_ROOT / "python"
REQS = ROOT / "ironvenv-requirements.txt"
PY = f"{sys.version_info.major}.{sys.version_info.minor}"
PYTHON = PY if sys.version_info.major == 3 and 11 <= sys.version_info.minor <= 14 else "3.11"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")


def wheel_dir(name: str, variable: str) -> Path:
    configured = os.environ.get(variable)
    if configured:
        return Path(configured)
    if WINDOWS:
        return VENV / "Lib" / "site-packages" / name
    found = sorted(VENV.glob(f"lib/python*/site-packages/{name}"))
    return found[-1] if found else VENV / "lib" / f"python{PYTHON}" / "site-packages" / name


def llvm_aie_bin() -> Path:
    return wheel_dir("llvm-aie", "PEANO_INSTALL_DIR") / "bin"


def mlir_aie_bin() -> Path:
    return wheel_dir("mlir_aie", "MLIR_AIE_INSTALL_DIR") / "bin"


def system_pyxrt_dir() -> Path | None:
    spec = importlib.util.find_spec("pyxrt") if PYTHON == PY else None
    if spec is None or not spec.origin:
        return None
    out = VENV / "system-pyxrt"
    out.mkdir(exist_ok=True)
    for src in Path(spec.origin).parent.glob("pyxrt*"):
        dest = out / src.name
        if WINDOWS:
            if not dest.exists():
                if src.is_dir():
                    shutil.copytree(src, dest)
                else:
                    shutil.copy2(src, dest)
        elif not dest.exists():
            dest.symlink_to(src)
    return out


def ensure_venv() -> Path:
    py = venv_python()
    if py.is_file():
        if os.environ.get("OFLM_SKIP_VENV_SETUP") == "1":
            return py
        if subprocess.run([py, "-c", "import aie"], capture_output=True).returncode == 0:
            return py
    print(f"-- setting up IRON in {VENV}", flush=True)
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", "--python", PYTHON, VENV], check=True)
        subprocess.run(["uv", "pip", "install", "--python", py, "-r", REQS], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
        subprocess.run([py, "-m", "pip", "install", "-r", REQS], check=True)
    return py


def directory(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def prepend(env: dict, key: str, paths) -> None:
    parts = [str(path) for path in paths if directory(path)]
    parts.extend(env.get(key, "").split(os.pathsep))
    env[key] = os.pathsep.join(dict.fromkeys(part for part in parts if part))


def toolchain_env(*, pyxrt: bool = True) -> dict:
    env = dict(os.environ)
    mlir, peano = mlir_aie_bin(), llvm_aie_bin()
    env.update(MLIR_AIE_INSTALL_DIR=str(mlir.parent), PEANO_INSTALL_DIR=str(peano.parent),
               XRT_ROOT=str(XRT_ROOT), VIRTUAL_ENV=str(VENV))
    prepend(env, "PATH", (mlir, mlir.parent / "lib", XRT_BIN, venv_python().parent))
    if WINDOWS:
        prepend(env, "PATH", (XRT_ROOT, XRT_ROOT / "lib",
                              Path(env.get("SystemRoot", r"C:\Windows")) / "System32" / "AMD"))
        env.setdefault("NPU2", "1")
    else:
        prepend(env, "PATH", (peano,))
        env.setdefault("XILINX_XRT", str(XRT_ROOT))
        prepend(env, "LD_LIBRARY_PATH", (XRT_ROOT / "lib", mlir.parent / "lib"))
    bind = XRT_PY if directory(XRT_PY) else (system_pyxrt_dir() if pyxrt else None)
    if bind is not None:
        prepend(env, "PYTHONPATH", (bind,))
    return env


def toolchain() -> tuple[Path, dict]:
    return ensure_venv(), toolchain_env()


def harness_binary(work: Path, env: dict | None = None) -> Path:
    env = toolchain_env() if env is None else env
    build = work.resolve() / "harness"
    name = "run_kernel.exe" if WINDOWS else "run_kernel"
    paths = (build / "Release" / name, build / name)
    for exe in paths:
        if exe.is_file():
            return exe
    xrt = Path(env.get("XRT_ROOT") or XRT_ROOT)
    include = env.get("XRT_INCLUDE_DIR") or str(xrt / "include")
    lib = env.get("XRT_LIB_DIR") or str(xrt / "lib")
    subprocess.run(["cmake", "-S", ROOT / "open_kernels" / "harness", "-B", build,
                    "-DCMAKE_BUILD_TYPE=Release", f"-DXRT_INCLUDE_DIR={include}",
                    f"-DXRT_LIB_DIR={lib}"], env=env, check=True)
    subprocess.run(["cmake", "--build", build, "--config", "Release"], env=env, check=True)
    for exe in paths:
        if exe.is_file():
            return exe
    raise RuntimeError(f"the harness build produced no {name} under {build}")


def npu_present() -> bool:
    if not WINDOWS:
        return any(Path("/dev/accel").glob("accel*"))
    command = ("Get-PnpDevice -PresentOnly | "
               "Where-Object { $_.FriendlyName -match 'AMD.*(IPU|NPU|XDNA)' } | "
               "Select-Object -First 1 -ExpandProperty FriendlyName")
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def check() -> int:
    env = toolchain_env(pyxrt=False)
    xrt = directory(XRT_ROOT) or shutil.which("xrt-smi", path=env["PATH"])
    mlir, peano = mlir_aie_bin(), llvm_aie_bin()
    npu = npu_present()
    script = Path(r"C:\dev\mlir-aie\iron_env.ps1") if WINDOWS else ROOT / "utilities" / "env.sh"
    print(f"repo:      {ROOT}")
    print(f"python:    {venv_python() if venv_python().is_file() else sys.executable}")
    print(f"XRT:       {f'ok ({XRT_ROOT})' if xrt else 'missing'}")
    for label, path in (("mlir-aie", mlir), ("Peano", peano)):
        print(f"{label + ':':11}{f'ok ({path})' if path.is_dir() else 'missing (ironvenv)'}")
    location = "Windows Plug and Play" if WINDOWS else "/dev/accel*"
    print(f"NPU dev:   {f'present ({location})' if npu else 'not detected'}")
    print(f"{script.name + ':':11}{script if script.is_file() else 'not found'}")
    return 0 if xrt and mlir.is_dir() and peano.is_dir() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    parser.parse_args()
    raise SystemExit(check())
