#!/usr/bin/env python3
"""Compare an RLinf PI0.5 export with its optimized OpenPI PyTorch conversion."""

from __future__ import annotations

import argparse
import gc
from importlib import import_module
from importlib import util
import json
import pathlib
import sys

import safetensors.torch
import torch

from openpi.models import model as openpi_model
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch


def _observation_inputs(batch_size: int, device: torch.device):
    images = {
        key: torch.full((batch_size, 3, 224, 224), -1.0, device=device)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool, device=device) for key in images}
    tokens = torch.zeros((batch_size, 200), dtype=torch.long, device=device)
    token_masks = torch.zeros((batch_size, 200), dtype=torch.bool, device=device)
    tokens[:, :10] = torch.arange(1, 11, device=device)
    token_masks[:, :10] = True
    mask = torch.zeros((batch_size, 50, 32), device=device)
    mask[..., :14] = 1.0
    observation = openpi_model.Observation(
        images=images,
        image_masks=image_masks,
        state=torch.zeros((batch_size, 32), device=device),
        action_mask=mask,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_masks,
    )
    generator = torch.Generator(device=device).manual_seed(87431)
    actions = torch.randn((batch_size, 50, 32), generator=generator, device=device)
    noise = torch.randn((batch_size, 50, 32), generator=generator, device=device)
    time = torch.full((batch_size,), 0.5, device=device)
    return observation, actions, noise, time


def _optimized_loss(
    converted: pathlib.Path,
    device: torch.device,
    observation,
    actions: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> float:
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        dtype="float32",
    )
    model = pi0_pytorch.PI0Pytorch(config)
    safetensors.torch.load_model(model, converted, strict=True, device="cpu")
    model.to(device).eval()
    torch.manual_seed(123)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        per_step = model(observation, actions, noise=noise, time=time)
    weights = observation.action_mask.sum(dim=-1)
    loss = (per_step * weights).sum() / weights.sum()
    value = float(loss)
    del model, per_step, loss
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _rlinf_loss(
    root: pathlib.Path,
    exported: pathlib.Path,
    device: torch.device,
    observation,
    actions: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> float:
    package_name = "_rlinf_pi0_model"
    package_root = root / "rlinf/models/embodiment/openpi_rlinf/pi0_model"
    spec = util.spec_from_file_location(
        package_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load RLinf PI0.5 package from {package_root}")
    package = util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    observation_cls = import_module(f"{package_name}.model").Observation
    pi0_cls = import_module(f"{package_name}.pi0").Pi0
    pi0_config_cls = import_module(f"{package_name}.pi0_config").Pi0Config

    config = pi0_config_cls(
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        pi05=True,
    )
    model = pi0_cls(config).to(device=device, dtype=torch.bfloat16)
    exported_state = torch.load(exported, map_location="cpu", weights_only=True, mmap=True)
    source = {key.removeprefix("model."): value for key, value in exported_state.items() if key.startswith("model.")}
    model.load_state_dict(source, strict=True)
    del exported_state, source
    local = observation_cls(
        images=observation.images,
        image_masks=observation.image_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )
    torch.manual_seed(123)
    with torch.inference_mode():
        per_dim = model.compute_loss_per_dim(
            local,
            actions,
            train=True,
            noise=noise,
            time=time,
        )
    mask = observation.action_mask.to(per_dim)
    loss = (per_dim * mask).sum() / mask.sum()
    value = float(loss)
    del model, per_dim, loss
    gc.collect()
    torch.cuda.empty_cache()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=pathlib.Path, required=True)
    parser.add_argument("--rlinf-export", type=pathlib.Path, required=True)
    parser.add_argument("--converted", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    observation, actions, noise, time = _observation_inputs(1, device)
    optimized = _optimized_loss(args.converted, device, observation, actions, noise, time)
    rlinf = _rlinf_loss(
        args.rlinf_root,
        args.rlinf_export,
        device,
        observation,
        actions,
        noise,
        time,
    )
    result = {
        "schema": "parallelvla.rlinf_optimized_forward_parity.v1",
        "optimized_action_loss": optimized,
        "rlinf_action_loss": rlinf,
        "absolute_difference": abs(optimized - rlinf),
        "ratio": optimized / rlinf,
        "batch": "constant_images_fixed_actions_noise_time_v1",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
