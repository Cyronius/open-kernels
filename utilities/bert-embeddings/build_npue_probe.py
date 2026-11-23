import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZipFile


REPO = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO / "open_kernels" / "upstream-patches"
NPUE = Path("src/open_npue")
SOURCES = (
    "npue.cpp",
    "npue_encoder.cpp",
    "npu_device.cpp",
    "json_min.cpp",
    "tokenizer.cpp",
    "tokenizer_xlmr.cpp",
    "tokenizer_gemma.cpp",
    "tokenizer_bbpe.cpp",
    "gemma_encode.cpp",
    "gemma_kernels.cpp",
)


def main():
    parser = argparse.ArgumentParser(
        description="Build the BERT NPUE probe from an x64 MSVC developer shell."
    )
    parser.add_argument("--oflm", type=Path, required=True, help="OpenFlowLM-Next checkout")
    parser.add_argument("--xrt-include", type=Path, required=True, help="XRT core include directory")
    parser.add_argument("--xrt-lib", type=Path, required=True, help="XRT coreutil import library")
    parser.add_argument("--boost-include", type=Path, required=True, help="Directory containing boost/any.hpp")
    parser.add_argument("--output", type=Path, required=True, help="Output .exe path")
    parser.add_argument("--worker", action="store_true", help="Build the resident embedding worker")
    args = parser.parse_args()

    compiler = shutil.which("cl.exe")
    if not compiler:
        parser.error("cl.exe is missing from PATH. Open an x64 MSVC developer shell.")
    xrt_include = args.xrt_include.resolve(strict=True)
    xrt_lib = args.xrt_lib.resolve(strict=True)
    boost_include = args.boost_include.resolve(strict=True)
    output = args.output.resolve()
    if output.suffix.lower() != ".exe":
        parser.error("--output must end in .exe")
    entry = json.loads((PATCH_DIR / "sources.json").read_text(encoding="utf-8"))[
        "bert_embedding_probe"
    ]
    archive = subprocess.check_output([
        "git", "-C", str(args.oflm.resolve(strict=True)), "archive", "--format=zip",
        entry["source_head"], NPUE.as_posix(),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bert-npue-") as directory:
        stage = Path(directory)
        with ZipFile(BytesIO(archive)) as files:
            files.extractall(stage)
        patch = (PATCH_DIR / entry["patch"]).read_text(encoding="utf-8")
        subprocess.run(["git", "apply", "-"], input=patch.encode("utf-8"), cwd=stage, check=True)
        sources = [str(NPUE / name) for name in SOURCES]
        program = "npue_worker.cpp" if args.worker else "profile_npue.cpp"
        subprocess.run([
            compiler, "/nologo", "/EHsc", "/O2", "/arch:AVX2", "/std:c++20", "/utf-8",
            f"/I{NPUE}", f"/I{xrt_include}", f"/I{boost_include}",
            f"/I{REPO / 'npu_offload' / 'gemm_rtp'}",
            f"/I{REPO / 'src' / 'include'}",
            str(Path(__file__).with_name(program)), *sources, str(xrt_lib), "bcrypt.lib",
            f"/Fe{output}",
        ], cwd=stage, check=True)


if __name__ == "__main__":
    main()
