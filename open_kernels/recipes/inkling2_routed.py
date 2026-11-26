"""Write the Inkling 2 bundle for routed decode, block prefill and dense attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil


STREAMS = ("iq4-up", "iq4-down", "q8-up", "q8-down")
FORMATS = ("iq4", "q8")
DENSE = ("dense-fused", "dense-query", "dense-kv", "dense-output")
VISION_PROJECTIONS = ((1024, 1024), (1024, 3072), (1024, 4096), (4096, 1024), (4096, 2304), (4096, 4096))
VISION_ATTENTION_ROWS = (256, 512, 1024, 2304, 4096, 8192, 16384, 32768, 42496)
WIDE_OLD_ATTENTION_KEYS = (4096, 16384)
WIDE_OLD_ATTENTION_FORMAT = "bfp16-qk-pv-v1"
NEAR_ATTENTION_FORMAT = "inkling2-near-v1"
NEAR_ATTENTION_CHUNKS = 80
LOOP_CONTROLLER_FORMAT = "inkling2-controller-v1"
LOOP_CONTROLLER_STREAMS = ("qkv", "ffn")
# Fused singleton experts: core r holds gate/up band pairs 3r..3r+2 (pools.cpp paired_band).
FUSED_EXPERTS = ("iq4-moe", "q8-moe")
GATE_UP_ORDER = "paired-v1"


def image_parts(root, pattern):
    return {p.name: p.read_bytes() for p in (root / "final.prj").glob(pattern)}


def check_image(roots):
    baseline = next(iter(roots.values()))
    core_pattern = "elfs_main_core_*/*.elf"
    cdo_pattern = "cdo_main/main_aie_cdo_*.bin"
    cores = image_parts(baseline, core_pattern)
    init = image_parts(baseline, cdo_pattern)
    if not cores or not init or not (baseline / "final.xclbin").is_file():
        raise ValueError(f"missing core image or initialization binaries in {baseline}")
    for name, root in roots.items():
        if image_parts(root, core_pattern) != cores or image_parts(root, cdo_pattern) != init:
            raise ValueError(f"{name} has a different core image or hardware initialization")
        if not (root / "insts.bin").is_file():
            raise ValueError(f"missing {name} instruction stream")
    return len(cores)


def check_formats(roots):
    counts = [check_image({f"{quant}-{leg}": roots[f"{quant}-{leg}"] for leg in ("up", "down")})
              for quant in FORMATS]
    if counts[0] != counts[1] or counts[0] not in (8, 16, 32):
        raise ValueError("both expert formats need the same supported core count")
    return counts[0] // 8


def check_wide_old_attention(roots):
    if check_image(roots) != 32:
        raise ValueError("wide older-key attention requires 32 worker cores")
    workers = {f"{col}_{row}" for col in range(8) for row in range(4)}
    for keys, root in roots.items():
        ir = (root / "final.prj/input_with_addresses.mlir").read_text(encoding="utf-8")
        writes = re.findall(r"aiex\.npu\.rtp_write\(@key_chunks_(\d+_\d+), (\d+), %c(\d+)_i32\)", ir)
        expected = {(worker, slot, str(value)) for worker in workers
                    for slot, value in (("0", keys // 32), ("4", 1), ("5", 1))}
        if len(writes) != len(expected) or set(writes) != expected:
            raise ValueError(f"{keys}-key stream must use split BFP16 scores and values")


def check_near_attention(root):
    if check_image({"near": root}) != 32:
        raise ValueError("near attention requires 32 worker cores")
    ir = (root / "final.prj/input_with_addresses.mlir").read_text(encoding="utf-8")
    writes = set(re.findall(r"aiex\.npu\.rtp_write\(@chunks_(\d+_\d+), 0, %c(\d+)_i32\)", ir))
    expected = {(f"{col}_{row}", str(NEAR_ATTENTION_CHUNKS)) for col in range(8) for row in range(4)}
    if writes != expected:
        raise ValueError(f"near attention stream must walk {NEAR_ATTENTION_CHUNKS} key chunks on every worker")


def split_scale_prefill(roots):
    # A split-scale Q8 weight tile is 96 rows of two FP16 scales and 32 codes.
    tiles = set()
    for root in roots:
        ir = (root / "final.prj/input_with_addresses.mlir").read_text(encoding="utf-8")
        tiles.add(frozenset(re.findall(r"memref<(3264|3456)xui8>", ir)))
    if tiles not in ({frozenset({"3264"})}, {frozenset({"3456"})}):
        raise ValueError("grouped Q8 prefill streams must use one weight tile layout")
    return tiles == {frozenset({"3456"})}


def aligned_attention_image(root):
    # The aligned entry point is retained only by the 24 attention workers.
    cores = image_parts(root, "elfs_main_core_*/*.elf")
    marked = sum(b"inkling_attn_chunk_aligned\0" in elf for elf in cores.values())
    if marked not in (0, 24):
        raise ValueError(f"inconsistent aligned attention entry points in {root}")
    return marked == 24


def decode_attention_image(root):
    # The keys-as-lanes decode mode ships inside the attention worker source.
    source = root / "final.prj/inkling_attn_init.cc"
    return source.is_file() and "decode_rows(" in source.read_text(encoding="utf-8")


def copy_formats(roots, output, direct, prefix):
    for quant in FORMATS:
        name = f"{prefix}-{quant}"
        image = name + ".xclbin"
        shutil.copyfile(roots[f"{quant}-up"] / "final.xclbin", output / image)
        direct["contexts"][name] = image
    for name, root in roots.items():
        key = f"{prefix}-{name}"
        stream = key + ".bin"
        shutil.copyfile(root / "insts.bin", output / stream)
        direct["kernels"][key] = {"insts": stream}


def attention_tiers(parent, prefix):
    tiers = {}
    for root in parent.glob(prefix + "*"):
        n = int(root.name.removeprefix(prefix))
        if not 1 <= n <= 128 or root.name != f"{prefix}{n}":
            raise ValueError("expected canonical relative attention tier names for 1–128 chunks")
        tiers[n] = root
    if 1 not in tiers:
        raise ValueError("relative attention needs a one-chunk tier")
    return dict(sorted(tiers.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("builds", type=Path, help="parent of the routed-*, compact-* or shared-* builds")
    parser.add_argument("output", type=Path, help="new directory for the image, streams and manifest")
    parser.add_argument("--head", type=Path, help="build of the Q8 logits kernel for K=2304 and N=200064")
    parser.add_argument("--head-pair", type=Path, help="build of the two-row Q8 head with band-major output")
    prefill = parser.add_mutually_exclusive_group()
    prefill.add_argument("--blocks", type=Path, help="parent of block-{iq4,q8}-{up,down} builds")
    prefill.add_argument("--grouped-blocks", type=Path, help="parent of prefill-{iq4,q8}-{up,down} builds (16 token rows)")
    parser.add_argument("--compact", action="store_true", help="share the block weights with compact decode")
    parser.add_argument("--attention", type=Path, help="parent of the relative attention tier builds (attention-<chunks>)")
    parser.add_argument("--shared-attention", action="store_true",
                        help="use shared-{iq4,q8} expert and attention tiers under BUILDS")
    parser.add_argument("--dense", type=Path, help="parent of dense-{fused,query,kv,output} Q8 builds")
    parser.add_argument("--vision-gemm", type=Path, help="parent of Inkling vision k<K>-n<N> BF16 GEMM builds")
    parser.add_argument("--vision-attention", type=Path, help="parent of Inkling bidirectional rows-<L> builds")
    parser.add_argument("--vision-split-keys", action="store_true",
                        help="the vision attention streams were built with AF_KSPLIT=1")
    parser.add_argument("--vision-split-values", action="store_true",
                        help="the vision attention streams were built with AF_VSPLIT=1")
    parser.add_argument("--wide-old-attention", type=Path,
                        help="parent of the optional wide-old-attention-{4096,16384} builds")
    parser.add_argument("--near-attention", type=Path, help="build of the 256-row near-range attention stream")
    parser.add_argument("--reclaim-packed-keys", action="store_true", help="retain exact copies of retired raw keys")
    parser.add_argument("--singleton-decode", type=Path, help="parent of the shared singleton builds")
    parser.add_argument("--loop-controller", type=Path,
                        help="parent of the controller-{qkv,ffn} builds: loop controllers of prefill blocks on the NPU")
    parser.add_argument("--split-prefill-q8", action="store_true",
                        help="the prefill-q8 streams read the split-scale experts that singleton decode packs")
    parser.add_argument("--fused-experts", action="store_true",
                        help="add the one-launch gate/up, SiTU and down singleton streams; every gate/up pool reader "
                             "then uses the paired band order")
    args = parser.parse_args()
    if args.compact and not (args.blocks or args.grouped_blocks):
        parser.error("--compact requires --blocks or --grouped-blocks")
    if args.attention and not args.compact:
        parser.error("--attention requires --compact")
    if args.shared_attention and (not args.compact or args.attention):
        parser.error("--shared-attention requires --compact and selects its own attention tiers")
    if args.grouped_blocks and not args.shared_attention:
        parser.error("--grouped-blocks requires --shared-attention")
    if args.dense and not args.grouped_blocks:
        parser.error("--dense requires --grouped-blocks")
    if bool(args.vision_gemm) != bool(args.vision_attention) or (args.vision_gemm and not args.dense):
        parser.error("Inkling vision requires --vision-gemm, --vision-attention and --dense")
    if args.vision_split_keys and not args.vision_attention:
        parser.error("--vision-split-keys requires --vision-attention")
    if args.vision_split_values and not args.vision_split_keys:
        parser.error("--vision-split-values requires --vision-split-keys")
    if args.wide_old_attention and not args.vision_split_values:
        parser.error("--wide-old-attention requires a v11 bundle")
    if args.reclaim_packed_keys and not args.wide_old_attention:
        parser.error("--reclaim-packed-keys requires --wide-old-attention")
    if args.near_attention and not args.vision_split_values:
        parser.error("--near-attention requires a v11 bundle")
    if args.head_pair and not (args.head and args.vision_split_values):
        parser.error("--head-pair requires a v11 bundle with --head")
    if args.singleton_decode and not args.vision_split_values:
        parser.error("--singleton-decode requires a v11 bundle")
    if args.split_prefill_q8 and not (args.singleton_decode and args.grouped_blocks):
        parser.error("--split-prefill-q8 requires --singleton-decode and --grouped-blocks")
    if args.fused_experts and not (args.singleton_decode and args.grouped_blocks):
        parser.error("--fused-experts requires --singleton-decode and --grouped-blocks")
    prefix = "shared" if args.shared_attention else "compact" if args.compact else "routed"
    roots = {name: args.builds / f"{prefix}-{name}" for name in STREAMS}
    baseline = roots["iq4-up"]
    blocks = {name: args.blocks / f"block-{name}" for name in STREAMS} if args.blocks else {}
    grouped = {name: args.grouped_blocks / f"prefill-{name}" for name in STREAMS} if args.grouped_blocks else {}
    dense = {name: args.dense / name for name in DENSE} if args.dense else {}
    vision_gemm = ({f"vision-gemm-{k}-{n}": args.vision_gemm / f"k{k}-n{n}" for k, n in VISION_PROJECTIONS}
                   if args.vision_gemm else {})
    vision_attention = ({f"vision-attention-{rows}": args.vision_attention / f"rows-{rows}"
                         for rows in VISION_ATTENTION_ROWS} if args.vision_attention else {})
    attention = {}
    shared = {}
    singleton = {}
    aligned_attention = False
    wide_old = ({keys: args.wide_old_attention / f"wide-old-attention-{keys}" for keys in WIDE_OLD_ATTENTION_KEYS}
                if args.wide_old_attention else {})
    try:
        if args.compact:
            decode_core_rows = check_formats(roots)
        else:
            cores = check_image(roots)
            if cores not in (1, 2, 3, 4, 6, 9, 12):
                raise ValueError("unsupported routed core count")
        block_rows = 16 * check_formats(blocks) if blocks else 0
        if grouped and check_formats(grouped) != 4:
            raise ValueError("grouped prefill requires 32 worker cores")
        if dense and check_image(dense) != 32:
            raise ValueError("Q8 dense projections require 32 worker cores")
        if vision_gemm and (check_image(vision_gemm) != 32 or check_image(vision_attention) != 32):
            raise ValueError("Inkling vision projections and attention require 32 worker cores")
        if args.attention:
            attention = attention_tiers(args.attention, "attention-")
            if check_image(attention) != 24:
                raise ValueError("relative attention requires 24 worker cores")
        if args.shared_attention:
            for quant in FORMATS:
                tiers = attention_tiers(args.builds, f"shared-{quant}-attention-")
                image_roots = {"up": roots[f"{quant}-up"], "down": roots[f"{quant}-down"], **tiers}
                if check_image(image_roots) != 32:
                    raise ValueError("shared attention requires 32 worker cores")
                shared[quant] = tiers
            if shared["iq4"].keys() != shared["q8"].keys():
                raise ValueError("IQ4 and Q8 must provide the same attention tiers")
            aligned = [aligned_attention_image(roots[f"{quant}-up"]) for quant in FORMATS]
            if aligned[0] != aligned[1]:
                raise ValueError("IQ4 and Q8 must provide the same attention partition")
            aligned_attention = aligned[0]
        if args.singleton_decode:
            if not aligned_attention:
                raise ValueError("singleton decode requires aligned relative attention")
            tiers = attention_tiers(args.singleton_decode, "singleton-attention-")
            if tiers.keys() != shared["iq4"].keys():
                raise ValueError("singleton decode must provide the same attention tiers")
            names = (*STREAMS, *(FUSED_EXPERTS if args.fused_experts else ()), "fused", "output")
            singleton = {name: args.singleton_decode / f"singleton-{name}" for name in names}
            singleton.update({f"attention-{n}": root for n, root in tiers.items()})
            if check_image(singleton) != 32 or not aligned_attention_image(singleton["iq4-up"]):
                raise ValueError("singleton decode requires one aligned 32-worker image")
        q8_prefill = [grouped[name] for name in ("q8-up", "q8-down")] if grouped else []
        described = q8_prefill and all((root / "final.prj/input_with_addresses.mlir").is_file() for root in q8_prefill)
        if args.split_prefill_q8 != bool(described and split_scale_prefill(q8_prefill)):
            raise ValueError("--split-prefill-q8 must match the weight tile layout of the prefill-q8 builds")
        if wide_old:
            if not aligned_attention or max(shared["iq4"]) != 128:
                raise ValueError("wide older-key attention requires aligned v11 relative attention with 128 chunks")
            check_wide_old_attention(wide_old)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    for build in (args.head, args.head_pair):
        if build and not all((build / name).is_file() for name in ("final.xclbin", "insts.bin")):
            parser.error(f"expected final.xclbin and insts.bin in {build}")
    args.output.mkdir()
    direct = {"architecture": "inkling2_loop", "contexts": {}, "kernels": {}}
    manifest = {"family": "inkling2", "gguf": direct}
    if args.compact:
        copy_formats(roots, args.output, direct, prefix)
        manifest["decode_core_rows"] = decode_core_rows
    else:
        shutil.copyfile(baseline / "final.xclbin", args.output / "final.xclbin")
        direct["contexts"]["experts"] = "final.xclbin"
        for name, root in roots.items():
            shutil.copyfile(root / "insts.bin", args.output / f"{name}.bin")
            direct["kernels"][name] = {"insts": f"{name}.bin"}
        manifest["cores"] = cores
    version = 1
    if args.head:
        shutil.copyfile(args.head / "final.xclbin", args.output / "head.xclbin")
        shutil.copyfile(args.head / "insts.bin", args.output / "head.bin")
        version = 2
        direct["contexts"]["head"] = "head.xclbin"
        direct["kernels"]["head"] = {"insts": "head.bin"}
    if args.head_pair:
        shutil.copyfile(args.head_pair / "final.xclbin", args.output / "head-pair.xclbin")
        shutil.copyfile(args.head_pair / "insts.bin", args.output / "head-pair.bin")
        direct["contexts"]["head-pair"] = "head-pair.xclbin"
        direct["kernels"]["head-pair"] = {"insts": "head-pair.bin"}
    if blocks:
        version = 4 if args.compact else 3
        manifest["block_rows"] = block_rows
        copy_formats(blocks, args.output, direct, "block")
    if attention:
        version = 5
        image = "relative-attention.xclbin"
        shutil.copyfile(attention[1] / "final.xclbin", args.output / image)
        direct["contexts"]["relative-attention"] = image
        manifest["attention_chunks"] = list(attention)
        for n, root in attention.items():
            name = f"attention-{n}"
            shutil.copyfile(root / "insts.bin", args.output / (name + ".bin"))
            direct["kernels"][name] = {"insts": name + ".bin"}
    if shared:
        version = 6
        manifest["attention_chunks"] = list(shared["iq4"])
        if aligned_attention:
            manifest["attention_chunk_alignment"] = 32
            if all(decode_attention_image(root) for tiers in shared.values() for root in tiers.values()):
                manifest["attention_decode"] = "keys-as-lanes-v1"
        for quant, tiers in shared.items():
            for n, root in tiers.items():
                name = f"shared-{quant}-attention-{n}"
                shutil.copyfile(root / "insts.bin", args.output / (name + ".bin"))
                direct["kernels"][name] = {"insts": name + ".bin"}
    if grouped:
        version = 7
        manifest["grouped_block_rows"] = 16
        copy_formats(grouped, args.output, direct, "prefill")
    if dense:
        version = 8
        manifest["dense_q8_format"] = "split-scale"
        shutil.copyfile(dense["dense-fused"] / "final.xclbin", args.output / "dense-q8.xclbin")
        direct["contexts"]["dense-q8"] = "dense-q8.xclbin"
        for name, root in dense.items():
            stream = name + ".bin"
            shutil.copyfile(root / "insts.bin", args.output / stream)
            direct["kernels"][name] = {"insts": stream}
    if vision_gemm:
        version = 9 + args.vision_split_keys + args.vision_split_values
        manifest["vision"] = {"projection_rows": 256, "attention_tiers": list(VISION_ATTENTION_ROWS)}
        if args.vision_split_keys:
            manifest["vision"]["key_planes"] = 2
        if args.vision_split_values:
            manifest["vision"]["value_planes"] = 2
        for name, streams in (("vision-gemm", vision_gemm), ("vision-attention", vision_attention)):
            image = name + ".xclbin"
            shutil.copyfile(next(iter(streams.values())) / "final.xclbin", args.output / image)
            direct["contexts"][name] = image
            for kernel, root in streams.items():
                stream = kernel + ".bin"
                shutil.copyfile(root / "insts.bin", args.output / stream)
                direct["kernels"][kernel] = {"insts": stream}
    if wide_old:
        image = "wide-old-attention.xclbin"
        shutil.copyfile(wide_old[4096] / "final.xclbin", args.output / image)
        direct["contexts"]["wide-old-attention"] = image
        for keys, root in wide_old.items():
            name = f"wide-old-attention-{keys}"
            stream = name + ".bin"
            shutil.copyfile(root / "insts.bin", args.output / stream)
            direct["kernels"][name] = {"insts": stream}
        manifest["wide_old_attention"] = WIDE_OLD_ATTENTION_FORMAT
        if args.reclaim_packed_keys:
            manifest["wide_old_key_storage"] = "lossless-packed"
    if args.near_attention:
        check_near_attention(args.near_attention)
        shutil.copyfile(args.near_attention / "final.xclbin", args.output / "near-attention.xclbin")
        shutil.copyfile(args.near_attention / "insts.bin", args.output / "near-attention.bin")
        direct["contexts"]["near-attention"] = "near-attention.xclbin"
        direct["kernels"]["near-attention"] = {"insts": "near-attention.bin"}
        manifest["near_attention"] = {"format": NEAR_ATTENTION_FORMAT, "chunks": NEAR_ATTENTION_CHUNKS}
    if singleton:
        image = "singleton-decode.xclbin"
        shutil.copyfile(singleton["iq4-up"] / "final.xclbin", args.output / image)
        streams = {name: f"singleton-{name}.bin" for name in singleton}
        for name, root in singleton.items():
            shutil.copyfile(root / "insts.bin", args.output / streams[name])
        manifest["singleton_decode"] = {
            "format": "inkling2-shared-split-q8", "image": image,
            "experts": {name: streams[name] for name in (*STREAMS, *(FUSED_EXPERTS if args.fused_experts else ()))},
            "attention": {str(n): streams[f"attention-{n}"] for n in shared["iq4"]},
            "dense": {name: streams[name] for name in ("fused", "output")},
        }
        if args.fused_experts:
            manifest["singleton_decode"]["gate_up_order"] = GATE_UP_ORDER
        if args.split_prefill_q8:
            manifest["shared_prefill_q8"] = True
    if args.loop_controller:
        roots = {name: args.loop_controller / f"controller-{name}" for name in LOOP_CONTROLLER_STREAMS}
        if check_image(roots) != 32:
            raise ValueError("the loop controller requires 32 worker cores")
        shutil.copyfile(roots["qkv"] / "final.xclbin", args.output / "controller.xclbin")
        for name, root in roots.items():
            shutil.copyfile(root / "insts.bin", args.output / f"controller-{name}.bin")
        eps = {int(e) for root in roots.values() for e in re.findall(
            r"aiex\.npu\.rtp_write\(@params_\d+_\d+, 1, %c(-?\d+)_i32\)",
            (root / "final.prj/input_with_addresses.mlir").read_text(encoding="utf-8"))}
        if len(eps) != 1:
            raise ValueError("loop controller streams must share one epsilon")
        manifest["loop_controller"] = {
            "format": LOOP_CONTROLLER_FORMAT, "image": "controller.xclbin", "rows": 256,
            "streams": {name: f"controller-{name}.bin" for name in LOOP_CONTROLLER_STREAMS},
            "eps_bits": eps.pop(),
        }
    manifest["abi"] = f"inkling2-routed-v{version}"
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
