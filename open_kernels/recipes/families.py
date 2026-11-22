"""The family recipes, by ModelSpec.family. Each module exposes the same
surface: recipe(spec, max_ctx), layout(spec, max_ctx), pack_plan(spec),
programs(spec), builds(spec), hf_config_check(spec), manifest_layout(spec,
max_ctx), KERNEL_SOURCES, GEN_KERNELS."""
from __future__ import annotations

from types import ModuleType

from .spec import ModelSpec


def family_module(name: str) -> ModuleType:
    if name == "qwen36moe":
        from . import qwen36moe
        return qwen36moe
    if name == "qwen35":
        from . import qwen35
        return qwen35
    if name == "lfm2":
        from . import lfm2
        return lfm2
    if name == "dflash2":
        from . import dflash2
        return dflash2
    if name == "gemma4":
        from . import gemma4
        return gemma4
    if name == "gptoss":
        from . import gptoss
        return gptoss
    if name == "flint":
        from . import flint
        return flint
    if name == "hyv3":
        from . import hyv3
        return hyv3
    if name == "lfm2moe":
        from . import lfm2moe
        return lfm2moe
    if name == "lfm3":
        from . import lfm3
        return lfm3
    if name == "nemotronh":
        from . import nemotronh
        return nemotronh
    if name == "xing":
        from . import xing
        return xing
    if name in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2", "k2"):
        from . import dense
        return dense
    if name in NOT_IMPLEMENTED:
        raise NotImplementedError(f"the open kernels have no recipe for {name!r} yet: "
                                  f"{NOT_IMPLEMENTED[name]}")
    raise ValueError(f"no recipe for family {name!r} (have {', '.join(FAMILIES)})")


def for_spec(spec: ModelSpec) -> ModuleType:
    return family_module(spec.family)


FAMILIES = ("qwen36moe", "qwen35", "qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2",
            "lfm2", "k2", "dflash2", "gemma4", "gptoss", "flint", "hyv3", "lfm2moe", "lfm3", "nemotronh", "xing")

# Families whose ModelSpec derives but whose kernels do not exist. Naming the gap here is
# the point: routing such a model to the nearest recipe would emit kernels that drop a whole
# stage of the layer and then report parity against a replica making the same mistake.
NOT_IMPLEMENTED: dict[str, str] = {}


def family_builds(F: ModuleType, spec: ModelSpec, max_ctx: int = 4096) -> dict:
    """F.builds(spec), with the context capacity for a family whose builds depend on it (dense's attn_fa)."""
    import inspect
    if max_ctx != 4096 and "max_ctx" in inspect.signature(F.builds).parameters:
        return F.builds(spec, max_ctx=max_ctx)
    return F.builds(spec)


FAMILY_ALIASES = [
    ("qwen3.5-omni", "qwen3.5-omni"),
    ("qwen3.6", "qwen3.6-moe"),
    ("qwen3.5-moe", "qwen3.6-moe"),
    ("qwen3.8", "qwen3.5"),      # Qwen3.8-Distilled-*: the Qwen3.5 engine, not Qwen3
    ("qwen3.5", "qwen3.5"),
    ("qwen3", "qwen3"),
    ("qwen2.5vl", "qwen2.5vl"),
    ("qwen2.5", "qwen2"),
    ("qwen2vl", "qwen2vl"),
    ("qwen2", "qwen2"),
    ("gemma4", "gemma4e"),
    ("gemma-4", "gemma4e"),
    ("gemma3", "gemma3"),
    ("nanbeige", "llama3"),
    ("llama3", "llama3"),
    ("llama", "llama3"),
    # Stock quant repos keep the upstream org/model prefix the fine-tune
    # slugs drop: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF,
    # google/gemma-3-4b-it-GGUF, microsoft/Phi-4-mini-instruct-GGUF.
    ("meta-llama", "llama3"),
    ("gemma-3", "gemma3"),
    ("phi-4", "phi4"),
    # Granite is its own family now (the dense recipe, head_dim 64 at hidden
    # 2560). It aliased onto llama3 because nothing served it; leaving that
    # would route a Granite directory to the Llama 3 AutoModel, whose sampler
    # defaults and chat-template probe are the wrong ones -- and the closed
    # The (since-removed) closed llama_npu refused hidden_size 2560 outright.
    ("granite", "granite"),
    ("hunyuan", "hunyuan"),
    ("crow", "qwen3.5"),
    ("huihui", "qwen3.5"),
    ("qwythos", "qwen3.5"),
    ("qwopus", "qwen3.5"),
    ("darwin", "qwen3.6-moe"),
    ("deepseek-r1-0528", "deepseek-r1-0528"),
    ("deepseek-r1", "deepseek-r1"),
    ("deepseek", "deepseek-r1"),
    ("nanbeige4", "nanbeige"),
    ("nanbeige", "nanbeige"),
    ("gpt-oss", "gpt-oss"),
    ("lfm2moe", "lfm2moe"),
    ("lfm2.5", "lfm2.5-tk"),
    ("lfm2", "lfm2"),
    ("phi4", "phi4"),
    ("whisper-v3", "whisper-v3"),
    ("whisper", "whisper-v3"),
    ("embed-gemma", "embed-gemma"),
    ("embedding-gemma", "embed-gemma"),
    ("embeddinggemma", "embed-gemma"),
    # NPU-embedding (NpuEmbeddings) families: HF safetensors checkpoints, no GGUF.
    ("bge", "bge"),
    ("nomic", "nomic"),
    ("all-minilm", "minilm"),
    ("minilm", "minilm"),
    ("gte", "gte"),
]

CONFIG_MODEL_FAMILIES = {
    "qwen3": "qwen3",
    "qwen2": "qwen2",
    "llama": "llama3",
    "nanbeige": "llama3",
    "nanbeige4": "llama3",
    "gemma3": "gemma3",
    "gemma3_text": "gemma3",
    "phi4": "phi4",
    # Phi-3 and Phi-4 both publish model_type `phi3`, and the engine family for the set we
    # ship (Phi4-mini-Instruct-NPU2) is `phi4`. Mapping it to `phi3` matched no official
    # entry, so an abliterated Phi-4-mini GGUF installed with "No official model matched; no
    # xclbins link" -- the geometry was right and nothing was linked.
    "phi3": "phi4",
    "granite": "granite",
    "hunyuan": "hunyuan",
    "hunyuan_v1_dense": "hunyuan",
    "k2_horizon": "k2",
    "lfm2": "lfm2",
    "lfm2_moe": "lfm2moe",
    "gpt_oss": "gpt-oss",
    # Qwen3.5 / 3.8's own model_type (the tag is still qwen3.5)
    "qwen3_5": "qwen3.5",
    "qwen3_5_text": "qwen3.5",
    "qwen3_next": "qwen3.5",
    "qwen3_vl": "qwen3vl",
    "qwen3_vl_text": "qwen3vl",
    "qwen2_vl": "qwen2vl",
    "qwen2_5_vl": "qwen2vl",
    "qwen2_5_vl_text": "qwen2vl",
    "whisper": "whisper-v3",
    # Gemma 4's open kernels (OPEN-FAMILY-GEMMA4): only the 12B (gemma4_unified /
    # gemma4_unified_text) is served -- the E2B / E4B (plain gemma4 / gemma4_text)
    # need per-layer embeddings and KV sharing and stay unmapped.
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4",
}


def family_from_config_dict(cfg):
    """details.family from a HuggingFace config.json dict, or None.

    model_type first (exact); then the architectures list, substring-matched
    against FAMILY_ALIASES so Qwen3ForCausalLM-style class names resolve.
    Neither fires for unknown shapes -- the caller must ask for --family."""
    if not isinstance(cfg, dict):
        return None
    model_type = cfg.get("model_type")
    if isinstance(model_type, str):
        lowered_type = model_type.strip().lower()
        # Vision-language checkpoints need their vision weights and their own
        # kernels; never fold one silently onto a text family. Unknown VL
        # shapes refuse here so the caller passes --family explicitly.
        if "vl" in lowered_type and lowered_type not in CONFIG_MODEL_FAMILIES:
            return None
        family = CONFIG_MODEL_FAMILIES.get(lowered_type)
        if family:
            return family
    archs = cfg.get("architectures") or []
    if isinstance(archs, str):
        archs = [archs]
    for arch in archs:
        if not isinstance(arch, str):
            continue
        lowered = arch.lower()
        for prefix, family in FAMILY_ALIASES:
            if prefix.lower() in lowered:
                return family
    return None
