import logging
import math
from typing import Literal

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch import casm_pytorch
from openpi.models_pytorch import lora_pytorch
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time.float()[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(time.dtype)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.casm_mode = config.casm_mode
        if self.casm_mode not in {"none", "visual_phase_gate"}:
            raise ValueError(
                f"PyTorch PI0 does not implement CASM mode {self.casm_mode!r}; "
                "supported modes are 'none' and 'visual_phase_gate'"
            )

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.lora_replaced_modules = (
            *self._inject_gemma_lora(
                self.paligemma_with_expert.paligemma.language_model,
                paligemma_config,
                prefix="paligemma",
            ),
            *self._inject_gemma_lora(
                self.paligemma_with_expert.gemma_expert.model,
                action_expert_config,
                prefix="action_expert",
            ),
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        if self.casm_mode == "visual_phase_gate":
            self.phase_gate = casm_pytorch.VisualProprioceptionGate(
                paligemma_config.width,
                config.action_dim,
                config.coordination_gate_hidden_dim,
            )
            self.gate_loss_weight = config.gate_loss_weight
            self.gate_positive_weight = config.gate_positive_weight

        self.aux_subtask_classes = config.pytorch_aux_subtask_classes
        self.aux_subtask_state_dim = config.pytorch_aux_subtask_state_dim
        self.aux_subtask_loss_weight = config.pytorch_aux_subtask_loss_weight
        self.aux_subtask_class_weights = config.pytorch_aux_subtask_class_weights
        if self.aux_subtask_classes:
            self.subtask_head = casm_pytorch.VisualProprioceptionClassifier(
                paligemma_config.width,
                self.aux_subtask_state_dim,
                config.pytorch_aux_subtask_hidden_dim,
                self.aux_subtask_classes,
                stop_gradient=config.pytorch_aux_subtask_stop_gradient,
            )

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    @staticmethod
    def _inject_gemma_lora(module: nn.Module, gemma_config, *, prefix: str) -> tuple[str, ...]:
        adapters = gemma_config.lora_configs
        if not adapters:
            return ()
        if set(adapters) != {"attn", "ffn"}:
            raise ValueError(f"{prefix} requires matching attention and FFN LoRA configs")
        settings = {(adapter.rank, adapter.alpha) for adapter in adapters.values()}
        if len(settings) != 1:
            raise ValueError(f"{prefix} attention and FFN LoRA settings must match")
        rank, alpha = settings.pop()
        replaced = lora_pytorch.inject_lora(module, rank=rank, alpha=alpha)
        logging.info(
            "Injected %s LoRA modules=%d rank=%d alpha=%.1f",
            prefix,
            len(replaced),
            rank,
            alpha,
        )
        return tuple(f"{prefix}.{path}" for path in replaced)

    def gradient_checkpointing_enable(self, scope: Literal["full", "vision"] = "full"):
        """Enable an explicit checkpointing scope for memory optimization."""
        if scope not in {"full", "vision"}:
            raise ValueError(f"Unsupported gradient checkpointing scope: {scope}")
        self.gradient_checkpointing_enabled = True
        checkpoint_language = scope == "full"
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = checkpoint_language
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = checkpoint_language
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": False,
            }
        )

        logging.info("Enabled %s gradient checkpointing for PI0Pytorch model", scope)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_disable()
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def set_attention_implementation(self, implementation: str) -> None:
        if implementation not in {"eager", "sdpa"}:
            raise ValueError(f"Unsupported attention implementation: {implementation}")
        for module in self.modules():
            module_config = getattr(module, "config", None)
            if getattr(module_config, "_attn_implementation", None) is not None:
                module_config._attn_implementation = implementation  # noqa: SLF001

    def _embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Checkpoint only the high-activation vision encoder."""
        return self.paligemma_with_expert.embed_image(image)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        return _preprocessing.preprocess_observation_pytorch(observation, train=train)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        visual_summaries = []
        visual_presence = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._embed_image(img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            visual_summaries.append(img_emb.mean(dim=1))
            visual_presence.append(img_mask.to(dtype=img_emb.dtype))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        presence = torch.stack(visual_presence, dim=1)
        summaries = torch.stack(visual_summaries, dim=1)
        visual_summary = (summaries * presence[..., None]).sum(dim=1)
        visual_summary = visual_summary / presence.sum(dim=1, keepdim=True).clamp_min(1)

        return embs, pad_masks, att_masks, visual_summary

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            state_emb = self.state_proj(state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)  # swish == silu
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward_subtask(self, observation) -> dict[str, Tensor]:
        """Run only the detached visual/state classifier used by head-only probes."""
        if not self.aux_subtask_classes:
            raise ValueError("subtask-only forward requires an auxiliary subtask head")
        if observation.semantic_subtask_id is None:
            raise ValueError("auxiliary subtask prediction requires semantic_subtask_id")
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        _, _, _, visual_summary = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        return casm_pytorch.subtask_classification_loss(
            self.subtask_head(visual_summary, state[:, : self.aux_subtask_state_dim]),
            observation.semantic_subtask_id,
            self.aux_subtask_class_weights,
        ).metrics

    def forward(
        self,
        observation,
        actions=None,
        noise=None,
        time=None,
        *,
        return_aux=False,
        subtask_only=False,
    ):
        """Run a training forward pass and return per-action-step loss."""
        if subtask_only:
            if return_aux or noise is not None or time is not None:
                raise ValueError("subtask-only forward does not accept action-forward options")
            return self.forward_subtask(observation)
        if actions is None:
            raise ValueError("actions are required for an action forward pass")
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, visual_summary = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        squared_error = F.mse_loss(u_t, v_t, reduction="none")
        action_loss = casm_pytorch.reduce_action_loss(squared_error, observation.action_mask)
        total = action_loss
        metrics = {"action_loss": action_loss.mean()}
        if self.casm_mode == "visual_phase_gate":
            if observation.phase_id is None:
                raise ValueError("visual-phase-gate CASM requires phase_id during training")
            gate = casm_pytorch.visual_phase_gate_loss(
                action_loss,
                self.phase_gate(visual_summary, state),
                observation.phase_id,
                gate_loss_weight=self.gate_loss_weight,
                gate_positive_weight=self.gate_positive_weight,
            )
            total = gate.total
            metrics.update(gate.metrics)
        if self.aux_subtask_classes:
            if observation.semantic_subtask_id is None:
                raise ValueError("auxiliary subtask prediction requires semantic_subtask_id")
            subtask = casm_pytorch.subtask_classification_loss(
                self.subtask_head(visual_summary, state[:, : self.aux_subtask_state_dim]),
                observation.semantic_subtask_id,
                self.aux_subtask_class_weights,
            )
            metrics.update(subtask.metrics)
        return (total, metrics) if return_aux else total

    @torch.no_grad()
    def predict_async_probability(self, observation) -> Tensor:
        if self.casm_mode != "visual_phase_gate":
            raise ValueError("async probability is available only for visual-phase-gate CASM")
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation,
            train=False,
        )
        _, _, _, visual_summary = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        return torch.sigmoid(self.phase_gate(visual_summary, state))

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
