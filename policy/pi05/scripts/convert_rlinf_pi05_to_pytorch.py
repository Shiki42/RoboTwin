#!/usr/bin/env python3
"""Convert an RLinf-layout PI0.5 export into the optimized OpenPI PyTorch layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from safetensors.torch import load_file
from safetensors.torch import save_file
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state_dict(
    exported: dict[str, torch.Tensor],
    template: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Invert RLinf's checked OpenPI-to-RLinf layout conversion."""
    source = {key.removeprefix("model."): value for key, value in exported.items() if key.startswith("model.")}
    result = dict(template)
    consumed: set[str] = set()

    def assign(target: str, source_key: str, transform=lambda value: value) -> None:
        if target not in result:
            raise KeyError(f"template is missing {target}")
        value = transform(source[source_key]).contiguous()
        if value.shape != result[target].shape:
            raise ValueError(f"shape mismatch for {target}: {value.shape} != {result[target].shape}")
        result[target] = value.to(dtype=result[target].dtype)
        consumed.add(source_key)

    vision = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    for suffix in ("weight", "bias"):
        assign(f"{vision}embeddings.patch_embedding.{suffix}", f"img.stem.{suffix}")
    assign(
        f"{vision}embeddings.position_embedding.weight",
        "img.pos_embedding",
        lambda value: value.squeeze(0),
    )
    for index in range(27):
        old = f"{vision}encoder.layers.{index}."
        new = f"img.encoder.layers.{index}."
        for old_name, new_name in (("layer_norm1", "norm1"), ("layer_norm2", "norm2")):
            for suffix in ("weight", "bias"):
                assign(f"{old}{old_name}.{suffix}", f"{new}{new_name}.{suffix}")
        weight_key = f"{new}attn.in_proj_weight"
        bias_key = f"{new}attn.in_proj_bias"
        weights = torch.chunk(source[weight_key], 3, dim=0)
        biases = torch.chunk(source[bias_key], 3, dim=0)
        for projection, weight, bias in zip(("q_proj", "k_proj", "v_proj"), weights, biases, strict=True):
            target = f"{old}self_attn.{projection}"
            if weight.shape != result[f"{target}.weight"].shape:
                raise ValueError(f"shape mismatch for {target}.weight")
            result[f"{target}.weight"] = weight.to(result[f"{target}.weight"].dtype).contiguous()
            result[f"{target}.bias"] = bias.to(result[f"{target}.bias"].dtype).contiguous()
        consumed.update((weight_key, bias_key))
        for suffix in ("weight", "bias"):
            assign(f"{old}self_attn.out_proj.{suffix}", f"{new}attn.out_proj.{suffix}")
        for name in ("fc1", "fc2"):
            for suffix in ("weight", "bias"):
                assign(f"{old}mlp.{name}.{suffix}", f"{new}mlp.{name}.{suffix}")
    for suffix in ("weight", "bias"):
        assign(f"{vision}post_layernorm.{suffix}", f"img.encoder.norm.{suffix}")
        assign(
            f"paligemma_with_expert.paligemma.model.multi_modal_projector.linear.{suffix}",
            f"img.head.{suffix}",
        )

    pali = "paligemma_with_expert.paligemma.model.language_model."
    expert = "paligemma_with_expert.gemma_expert.model."
    for index in range(18):
        new = f"llm.layers.{index}."
        for expert_index, old in ((0, f"{pali}layers.{index}."), (1, f"{expert}layers.{index}.")):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assign(
                    f"{old}self_attn.{projection}.weight",
                    f"{new}attn.{projection}.{expert_index}.weight",
                )
            assign(f"{old}mlp.gate_proj.weight", f"{new}mlps.{expert_index}.w_gating", lambda value: value[0].T)
            assign(f"{old}mlp.up_proj.weight", f"{new}mlps.{expert_index}.w_gating", lambda value: value[1].T)
            consumed.add(f"{new}mlps.{expert_index}.w_gating")
            assign(f"{old}mlp.down_proj.weight", f"{new}mlps.{expert_index}.w_linear", lambda value: value.T)
        assign(f"{pali}layers.{index}.input_layernorm.weight", f"{new}pre_attention_norms.0.scale")
        assign(f"{pali}layers.{index}.post_attention_layernorm.weight", f"{new}pre_ffw_norms.0.scale")
        for old_name, new_name in (
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ):
            for suffix in ("weight", "bias"):
                assign(
                    f"{expert}layers.{index}.{old_name}.dense.{suffix}",
                    f"{new}{new_name}.1.ada_modulation.{suffix}",
                )
    assign(f"{pali}norm.weight", "llm.final_norms.0.scale")
    for suffix in ("weight", "bias"):
        assign(f"{expert}norm.dense.{suffix}", f"llm.final_norms.1.ada_modulation.{suffix}")
    assign("paligemma_with_expert.paligemma.lm_head.weight", "llm.embedder.embedding.weight")

    for prefix in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        for suffix in ("weight", "bias"):
            assign(f"{prefix}.{suffix}", f"{prefix}.{suffix}")
    unconsumed = set(source) - consumed
    if unconsumed:
        raise ValueError(f"unconsumed action-model tensors: {sorted(unconsumed)[:12]}")
    return result, consumed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-export", type=Path, required=True)
    parser.add_argument("--pytorch-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--producer-commit", required=True)
    args = parser.parse_args()
    if len(args.producer_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.producer_commit
    ):
        raise ValueError("producer commit must be a full lowercase Git SHA")
    staging = args.output_dir.with_name(args.output_dir.name + ".part")
    if args.output_dir.exists() or staging.exists():
        raise FileExistsError(f"output or staging directory already exists: {args.output_dir}")
    staging.mkdir(parents=True)
    try:
        exported = torch.load(args.rlinf_export, map_location="cpu", weights_only=True, mmap=True)
        template = load_file(args.pytorch_template, device="cpu")
        converted, consumed = convert_state_dict(exported, template)
        part = staging / "model.safetensors.part"
        final = staging / "model.safetensors"
        save_file(converted, part)
        part.replace(final)
        manifest = {
            "schema": "parallelvla.rlinf_pi05_to_pytorch.v1",
            "producer_commit": args.producer_commit,
            "source_inventory_sha256": args.source_inventory_sha256,
            "source_export": str(args.rlinf_export),
            "source_export_sha256": sha256(args.rlinf_export),
            "pytorch_template": str(args.pytorch_template),
            "pytorch_template_sha256": sha256(args.pytorch_template),
            "converted_model_sha256": sha256(final),
            "converted_model_bytes": final.stat().st_size,
            "source_model_tensors": len(consumed),
            "output_tensors": len(converted),
            "retained_unused_template_tensors": ["paligemma_with_expert.gemma_expert.lm_head.weight"],
        }
        receipt = staging / "conversion_receipt.json.part"
        receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        receipt.replace(staging / "conversion_receipt.json")
        staging.replace(args.output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
