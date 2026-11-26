"""Python port of phlegm/benchmarks/reference/build.ps1 (same inputs, compile command and .build.json fields)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(adapter, engine, root, xrt_include, xrt_lib):
    files = {'bridge/bridge.cpp': sha(adapter)}
    for f in sorted(engine.rglob('*')):
        if f.is_file() and f.suffix in ('.cpp', '.hpp'):
            files['engine/' + f.relative_to(engine).as_posix()] = sha(f)
    for base, prefix in ((root / 'src/include/nlohmann', 'json'), (xrt_include, 'xrt')):
        for f in sorted(base.rglob('*')):
            if f.is_file() and f.suffix in ('.h', '.hpp'):
                files[f'{prefix}/' + f.relative_to(base).as_posix()] = sha(f)
    files['harness/stream_patch.hpp'] = sha(root / 'open_kernels/harness/stream_patch.hpp')
    files['link/xrt_coreutil.lib'] = sha(xrt_lib)
    return files


def manifest_hash(files):
    text = ''.join(f'{name}\0{files[name]}\n' for name in sorted(files))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-root', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--xrt-include', required=True, type=Path)
    ap.add_argument('--xrt-lib', required=True, type=Path)
    ap.add_argument('--adapter-root', type=Path, default=Path('C:/code/phlegm/benchmarks/reference'))
    ap.add_argument('--vcvars', type=Path, default=Path(os.environ['ProgramFiles(x86)'])
                    / 'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
    a = ap.parse_args()
    root, out = a.reference_root.resolve(), a.out_dir.resolve()
    xrt_include, xrt_lib, vcvars = a.xrt_include.resolve(), a.xrt_lib.resolve(), a.vcvars.resolve()
    adapter = (a.adapter_root / 'bridge.cpp').resolve()
    engine = root / 'src/open_qwen36'
    for p in (vcvars, xrt_lib, xrt_include / 'xrt/xrt_device.h', engine / 'core.cpp', engine / 'gguf_file.cpp'):
        if not p.is_file():
            raise SystemExit(f'Missing build input: {p}')
    for p in (root, out, xrt_include, xrt_lib, vcvars):
        if any(c in str(p) for c in '"&|<>^%!\r\n'):
            raise SystemExit(f'Unsupported shell character in path: {p}')
    out.mkdir(parents=True, exist_ok=True)
    binary = out / 'phlegm-reference-bench.exe'
    metadata = binary.with_suffix('.build.json')
    metadata.unlink(missing_ok=True)
    before = source_files(adapter, engine, root, xrt_include, xrt_lib)
    source_hash = manifest_hash(before)
    git = ['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root)]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=no'], text=True)
    names = ['core.cpp', 'block_host.cpp', 'pools.cpp', 'manifest.cpp', 'q4nx_file.cpp', 'gguf_file.cpp']
    if (engine / 'inkling2_npu.hpp').exists():
        names += ['inkling2_host.cpp', 'inkling2_npu.cpp']
    if (engine / 'inkling2_dspark.hpp').exists():
        names.append('inkling2_dspark.cpp')
    if (engine / 'vision/inkling2_vision_npu.hpp').exists():
        names += ['vision/inkling2_vision.cpp', 'vision/inkling2_vision_npu.cpp', 'vision/vit.cpp']
    sources = ' '.join(f'"{engine / n}"' for n in names)
    batch = f'''@echo off
set "PATH=%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer;%PATH%"
call "{vcvars}" >nul
if errorlevel 1 exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /DDISABLE_ABI_CHECK=1 /bigobj /openmp /arch:AVX2 /I "{xrt_include}" /I "{root}/src" /I "{root}/src/include" /I "{root}/open_kernels/harness" {sources} "{adapter}" "{xrt_lib}" /Fe:"{binary}" /Fo:.\\ /link /DELAYLOAD:VCOMP140.DLL delayimp.lib
exit /b %errorlevel%
'''
    (out / 'build.cmd').write_bytes(batch.encode('ascii'))
    subprocess.run(['cmd.exe', '/d', '/c', str(out / 'build.cmd')], cwd=out, check=True)
    if manifest_hash(source_files(adapter, engine, root, xrt_include, xrt_lib)) != source_hash:
        raise SystemExit('Build inputs changed during compilation; rebuild required')
    build = {
        'protocol': 2 if (engine / 'inkling2_npu.hpp').exists() else 1,
        'engine': 'openflowlm-cpp-reference',
        'source_commit': commit,
        'source_dirty': bool(status.strip()),
        'source_tree_sha256': source_hash,
        'adapter_sha256': before['bridge/bridge.cpp'],
        'binary_sha256': sha(binary),
        'compiler_flags': 'MSVC /O2 /MD /std:c++17 /openmp /arch:AVX2 /DELAYLOAD:VCOMP140.DLL',
        'source_files': before,
    }
    metadata.write_text(json.dumps(build, indent=2), encoding='utf-8')
    print(f'Built {binary}\nProvenance {metadata}')


if __name__ == '__main__':
    main()
