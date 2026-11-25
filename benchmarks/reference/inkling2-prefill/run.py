"""Profile Inkling 2 Mini through the Phlegm protocol-2 reference bridge."""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import time

from windows_power import collect_events, keep_awake

BLOCK = 256
VOCAB = 200058
LOGITS_BYTES = VOCAB * 4
MAX_CONTEXT = 1048576
WINDOWS = os.name == 'nt'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def save(path, row):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def bridge_header(line):
    response = json.loads(line)
    if not response.get('ok'):
        raise RuntimeError(response.get('error', 'Phlegm bridge rejected the request'))
    return response


class Bridge:
    def __init__(self, process):
        self.process = process

    def reply(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Phlegm bridge closed its response pipe ({self.process.poll()})')
        header = bridge_header(line)
        size = header.get('logits_f32_le_bytes', 0)
        if size not in (0, LOGITS_BYTES):
            raise ValueError(f'unexpected Phlegm logits payload size {size}')
        payload = self.process.stdout.read(size)
        if len(payload) != size:
            raise ValueError('truncated Phlegm logits')
        return header, payload

    def send(self, command, **fields):
        self.process.stdin.write((json.dumps(dict(command=command, **fields)) + '\n').encode())
        self.process.stdin.flush()
        return self.reply()


@contextmanager
def device_lock(path):
    owner = str(os.getpid())
    try:
        handle = path.open('x')
    except FileExistsError as error:
        raise RuntimeError(f'Phlegm device lock already exists at {path}; inspect its owner before retrying') from error
    try:
        with handle:
            handle.write(owner)
        yield
    finally:
        if path.is_file() and path.read_text(encoding='ascii').strip() == owner:
            path.unlink()


def read_profile(log, *, complete_lines_only=False):
    rows = []
    marker = 'inkling_prefill_profile '
    raw = log.read_bytes()
    if complete_lines_only:
        raw = raw[:raw.rfind(b'\n') + 1]
    for line in raw.decode('utf-8').splitlines():
        if not line.startswith(marker):
            continue
        fields = dict(field.split('=', 1) for field in line[len(marker):].split())
        row = {name: float(value) if name.endswith('_ms') else int(value) for name, value in fields.items()}
        if (not {'position', 'rows', 'block_ms', 'group_total_ms', 'wait_ms'} <= row.keys()
                or not all(math.isfinite(value) and value >= 0 for value in row.values())
                or (rows and row.keys() != rows[0].keys())):
            raise ValueError('incomplete or invalid prefill stage counters')
        rows.append(row)
    return rows


def block_sequence(length, block):
    return [(start, min(block, length - start)) for start in range(0, length, block)]


def events_during(path, window):
    events = json.loads(path.read_bytes())
    start, end = (datetime.fromisoformat(window[key]) for key in ('start', 'end'))
    return {
        key: sum(start <= datetime.fromisoformat(event['time']) <= end for event in events[key])
        for key in ('power_events', 'pci_context_errors')
    }


def profile(bridge, args, tokens, env, output):
    with (output / 'bridge.stderr.log').open('wb') as errors, subprocess.Popen(
        [bridge], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
    ) as process:
        try:
            client = Bridge(process)
            hello, _ = client.reply()
            if hello.get('protocol') != 2:
                raise RuntimeError('Phlegm bridge protocol 2 is required')
            capacity = max(len(tokens), len(args.warmup))
            ready, _ = client.send('init', model_dir=str(args.model_dir), kernel_dir=str(args.bundle),
                                   max_ctx=min(math.ceil(capacity / args.block) * args.block, MAX_CONTEXT),
                                   prefill_route='block', prefill_mode='fast', dispatch_timeout_ms=60000)
            if ready.get('prefill_route') != 'layer-major' or ready.get('prefill_chunk') != args.block:
                raise RuntimeError(f'Phlegm did not select {args.block}-token layer-major prefill')
            client.send('reset', dspark=False)
            client.send('prefill', prompt=args.warmup)
            client.send('reset', dspark=False)
            prefill_start = datetime.now().astimezone()
            started = time.perf_counter()
            for pos, rows in block_sequence(len(tokens), args.block):
                end = pos + rows
                if end == len(tokens):
                    _, logits = client.send('prefill_logits', prompt=tokens[pos:end])
                else:
                    reply, _ = client.send('prefill_chunk', prompt=tokens[pos:end])
                    if reply.get('position') != end:
                        raise RuntimeError('unexpected committed prefill position')
                emit(dict(committed=end, elapsed_seconds=time.perf_counter() - started))
            elapsed = time.perf_counter() - started
            prefill_end = datetime.now().astimezone()
            if len(logits) != LOGITS_BYTES or not all(math.isfinite(v) for (v,) in struct.iter_unpack('<f', logits)):
                raise ValueError('the final prefill logits are incomplete or non-finite')
            (output / 'logits.f32').write_bytes(logits)
            save(output / 'prefill-timing.json', dict(start=prefill_start.isoformat(), end=prefill_end.isoformat(),
                                                      wall_seconds=elapsed))
            client.send('quit')
            if process.wait(timeout=15) != 0:
                raise RuntimeError('Phlegm exited unsuccessfully')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--prefix', type=int)
    parser.add_argument('--block', type=int, default=BLOCK, help='expected layer-major block size reported by the bridge')
    parser.add_argument('--warmup-tokens', type=int, help='prefix tokens to replay before timing')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--keep-display', action='store_true', help='request the display stay on during a Windows measurement')
    parser.add_argument('--prepare-only', action='store_true', help='check inputs and protocol without opening the NPU')
    args = parser.parse_args()
    bridge, output = args.bridge.resolve(), args.output.resolve()
    args.model_dir, args.bundle = args.model_dir.resolve(), args.bundle.resolve()
    ids = list(map(int, args.ids.read_text(encoding='utf-8-sig').split()))
    prefix = args.prefix if args.prefix is not None else len(ids)
    if not 1 < args.block <= MAX_CONTEXT or args.threads < 1:
        parser.error(f'choose a block size from 2 to {MAX_CONTEXT} and a positive thread count')
    warmup = min(args.block, prefix) if args.warmup_tokens is None else args.warmup_tokens
    available = min(len(ids), MAX_CONTEXT)
    if not (0 < prefix <= available and 0 < warmup <= available):
        parser.error('choose positive prompt and warm-up lengths within the available IDs and model context')
    if not all(0 <= token < VOCAB for token in ids[:max(prefix, warmup)]):
        parser.error('the prompt or warm-up has an ID outside the Inkling 2 Mini vocabulary')
    tokens, args.warmup = ids[:prefix], ids[:warmup]
    model = args.model_dir / 'model.gguf'
    manifest = args.bundle / 'manifest.json'
    if not model.is_file() or not manifest.is_file():
        parser.error('expected model.gguf in --model-dir and manifest.json in --bundle')
    meta = json.loads(bridge.with_suffix('.build.json').read_text(encoding='utf-8-sig'))
    if (meta.get('binary_sha256') != sha(bridge) or meta.get('protocol') != 2
            or 'engine/inkling2_prefill_profile.hpp' not in meta.get('source_files', {})):
        raise ValueError('the bridge provenance does not match a profiled protocol-2 build')

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_WAIT_POLICY='PASSIVE', INKLING2_OLD_QUERY_GROUPS='0')
    if WINDOWS:
        env['PATH'] = r'C:\Windows\System32\AMD' + os.pathsep + env['PATH']
    hello = subprocess.run([bridge], input=b'{"command":"quit"}\n', env=env,
                           capture_output=True, timeout=15, check=True)
    if bridge_header(hello.stdout.splitlines()[0]).get('protocol') != 2:
        raise RuntimeError('Phlegm bridge did not advertise protocol 2')

    output.mkdir(parents=True)
    stat = model.stat()
    inputs = dict(bridge=meta['binary_sha256'], engine=meta['source_tree_sha256'],
                  manifest=sha(manifest), ids=sha(args.ids), prefix=prefix,
                  model_path=str(model), model_size=stat.st_size, model_mtime_ns=stat.st_mtime_ns,
                  block=args.block, warmup_tokens=warmup, threads=args.threads, platform=platform.platform())
    save(output / 'input.json', inputs)
    emit(inputs)
    if args.prepare_only:
        return

    lock = Path(env.get('PHLEGM_NPU_LOCK', Path(tempfile.gettempdir()) / 'phlegm-npu.lock'))
    with device_lock(lock), (keep_awake(keep_display=args.keep_display) if WINDOWS else nullcontext()) as power:
        start = datetime.now().astimezone()
        elapsed = profile(bridge, args, tokens, env, output)
        end = datetime.now().astimezone()
    trace = output / 'bridge.stderr.log'
    all_rows = read_profile(trace)
    warmup_blocks = block_sequence(warmup, args.block)
    prompt_blocks = block_sequence(prefix, args.block)
    if [(row['position'], row['rows']) for row in all_rows] != warmup_blocks + prompt_blocks:
        raise ValueError('the prefill trace differs from the requested block sequence')
    reports = all_rows[len(warmup_blocks):]
    totals = {key: sum(row[key] for row in reports) for key in reports[0] if key.endswith(('_ms', '_calls'))}
    timed = json.loads((output / 'prefill-timing.json').read_bytes())
    result = dict(inputs, prefill_wall_seconds=elapsed, tokens_per_second=prefix / elapsed,
                  prefill_interval={key: timed[key] for key in ('start', 'end')},
                  events=None, power_status=power, stage_totals=totals, blocks=reports,
                  trace=sha(trace), logits=sha(output / 'logits.f32'))
    if WINDOWS:
        result['event_interval'] = dict(start=start.isoformat(), end=end.isoformat())
    save(output / 'result.json', result)
    if WINDOWS:
        event_log = output / 'windows-events.json'
        result['events'] = collect_events(start, end, event_log)
        result['prefill_events'] = events_during(event_log, result['prefill_interval'])
        save(output / 'result.json', result)


if __name__ == '__main__':
    main()
