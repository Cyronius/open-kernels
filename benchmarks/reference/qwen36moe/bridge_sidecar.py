# Write phlegm-reference-bench.build.json the way phlegm's benchmarks/reference/build.ps1 does, for a bridge built by bridge/build.cmd (2026-11-23).
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ref, phlegm, binary = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
xrt_inc, xrt_lib = Path(r"C:\dev\XRT\src\runtime_src\core\include"), Path(r"C:\dev\xrtNPUfromDLL\xrt_coreutil.lib")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


files = {"bridge/bridge.cpp": sha(phlegm / "benchmarks/reference/bridge.cpp")}
eng = ref / "src/open_qwen36"
for p in eng.rglob("*"):
    if p.is_file() and p.suffix in (".cpp", ".hpp"):
        files["engine/" + p.relative_to(eng).as_posix()] = sha(p)
for base, prefix in ((ref / "src/include/nlohmann", "json"), (xrt_inc, "xrt")):
    for p in base.rglob("*"):
        if p.is_file() and p.suffix in (".h", ".hpp"):
            files[f"{prefix}/" + p.relative_to(base).as_posix()] = sha(p)
files["harness/stream_patch.hpp"] = sha(ref / "open_kernels/harness/stream_patch.hpp")
files["link/xrt_coreutil.lib"] = sha(xrt_lib)
h = hashlib.sha256()
for k in sorted(files):
    h.update(k.encode() + b"\0" + files[k].encode() + b"\n")
git = ["git", "-C", str(ref)]
commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], text=True).strip())
build = {"protocol": 2, "engine": "openflowlm-cpp-reference", "source_commit": commit, "source_dirty": dirty,
         "source_tree_sha256": h.hexdigest(), "adapter_sha256": files["bridge/bridge.cpp"], "binary_sha256": sha(binary),
         "compiler_flags": "MSVC /O2 /MD /std:c++17 /openmp /arch:AVX2 /DELAYLOAD:VCOMP140.DLL", "source_files": files}
binary.with_suffix(".build.json").write_text(json.dumps(build, indent=2))
print("wrote", binary.with_suffix(".build.json"), "dirty" if dirty else "clean")
