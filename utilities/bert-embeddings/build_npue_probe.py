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
PACKER_SOURCES = (
    "npue_pack.cpp",
    "json_min.cpp",
    "gemma_tokenizer_gen.cpp",
    "xlmr_tokenizer_gen.cpp",
    "bbpe_tokenizer_gen.cpp",
)


def main():
    parser = argparse.ArgumentParser(
        description="Build BERT NPUE tools from an x64 MSVC developer shell."
    )
    parser.add_argument("--oflm", type=Path, required=True, help="OpenFlowLM-Next checkout")
    parser.add_argument("--xrt-include", type=Path, help="XRT core include directory")
    parser.add_argument("--xrt-lib", type=Path, help="XRT coreutil import library")
    parser.add_argument("--boost-include", type=Path, help="Directory containing boost/any.hpp")
    parser.add_argument("--output", type=Path, required=True, help="Output .exe path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", action="store_true", help="Build the resident embedding worker")
    mode.add_argument("--packer", action="store_true", help="Build the CPU-only checkpoint packer")
    args = parser.parse_args()

    compiler = shutil.which("cl.exe")
    if not compiler:
        parser.error("cl.exe is missing from PATH. Open an x64 MSVC developer shell.")
    includes = []
    libs = []
    if not args.packer:
        if not (args.xrt_include and args.xrt_lib and args.boost_include):
            parser.error("the probe and worker require --xrt-include, --xrt-lib and --boost-include")
        includes = [
            f"/I{args.xrt_include.resolve(strict=True)}",
            f"/I{args.boost_include.resolve(strict=True)}",
            f"/I{REPO / 'npu_offload' / 'gemm_rtp'}",
            f"/I{REPO / 'src' / 'include'}",
        ]
        libs = [str(args.xrt_lib.resolve(strict=True)), "bcrypt.lib"]
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
        if args.packer:
            program, source_names = "pack_npue.cpp", PACKER_SOURCES
        else:
            patch = (PATCH_DIR / entry["patch"]).read_text(encoding="utf-8")
            subprocess.run(["git", "apply", "-"], input=patch.encode("utf-8"), cwd=stage, check=True)
            program = "npue_worker.cpp" if args.worker else "profile_npue.cpp"
            source_names = SOURCES
        sources = [str(NPUE / name) for name in source_names]
        subprocess.run([
            compiler, "/nologo", "/EHsc", "/O2", "/arch:AVX2", "/std:c++20", "/utf-8",
            f"/I{NPUE}", *includes, str(Path(__file__).with_name(program)), *sources, *libs,
            f"/Fe{output}",
        ], cwd=stage, check=True)


if __name__ == "__main__":
    main()
