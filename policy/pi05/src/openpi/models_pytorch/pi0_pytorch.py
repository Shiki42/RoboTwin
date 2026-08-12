from collections.abc import Sequence
import logging
import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import tokenizer as _tokenizer
import openpi.models.gemma as _gemma
from openpi.models_pytorch import casm_pytorch
from openpi.models_pytorch import lora_pytorch
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.online_subtask import OnlineSubtaskLoss
from openpi.models_pytorch.online_subtask import next_token_cross_entropy
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


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
        self.online_subtask_prediction = config.online_subtask_prediction
        self.lambda_subtask = config.lambda_subtask
        self.subtask_tokenizer = (
            _tokenizer.PaligemmaTokenizer(config.subtask_max_token_len) if self.online_subtask_prediction else None
        )
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
        checkpoint_transformer = scope == "full"
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = checkpoint_transformer
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = checkpoint_transformer
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = checkpoint_transformer

        logging.info("Enabled %s gradient checkpointing for PI0Pytorch model", scope)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
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
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.paligemma_with_expert.embed_image, image, use_reentrant=False, preserve_rng_state=False
            )
        return self.paligemma_with_expert.embed_image(image)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Return the standard PI0 action-policy input tuple."""
        return _preprocessing.preprocess_observation_pytorch(
            observation,
            train=train,
        )

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
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        lang_att_masks=None,
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

        # Standard action prompts use full attention. Subtask targets may be causal.
        num_lang_embs = lang_emb.shape[1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        image_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(
            bsize, len(att_masks)
        )
        if lang_att_masks is None:
            language_att_masks = torch.zeros(bsize, num_lang_embs, dtype=torch.bool, device=pad_masks.device)
        else:
            language_att_masks = lang_att_masks.to(dtype=torch.bool, device=pad_masks.device)
        att_masks = torch.cat([image_att_masks, language_att_masks], dim=1)

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
        time_emb = time_emb.type(dtype=timestep.dtype)

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

    def _action_loss(
        self,
        observation,
        actions,
        noise=None,
        time=None,
        *,
        return_aux=False,
        preprocessed: tuple[tuple[Tensor, ...], tuple[Tensor, ...]] | None = None,
    ):
        """Run the unchanged standard flow-matching action path."""
        if preprocessed is None:
            images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        else:
            images, img_masks = preprocessed
            lang_tokens = observation.tokenized_prompt
            lang_masks = observation.tokenized_prompt_mask
            state = observation.state

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
        if self.casm_mode == "none":
            return action_loss
        if observation.phase_id is None:
            raise ValueError("visual-phase-gate CASM requires phase_id during training")
        gate_logits = self.phase_gate(visual_summary, state)
        loss = casm_pytorch.visual_phase_gate_loss(
            action_loss,
            gate_logits,
            observation.phase_id,
            gate_loss_weight=self.gate_loss_weight,
            gate_positive_weight=self.gate_positive_weight,
        )
        return (loss.total, loss.metrics) if return_aux else loss.total

    def _reduce_action_objective(
        self,
        action_loss: Tensor,
        action_mask: Tensor | None,
    ) -> Tensor:
        if action_mask is None:
            return action_loss.mean()
        weights = action_mask.to(
            device=action_loss.device,
            dtype=action_loss.dtype,
        ).sum(dim=-1)
        if weights.shape != action_loss.shape:
            raise ValueError(f"action loss/mask shape mismatch: {action_loss.shape} != {weights.shape}")
        total_weight = weights.sum()
        if torch.compiler.is_compiling():
            torch._assert_async(total_weight > 0, "action loss mask is empty")  # noqa: SLF001
        elif total_weight.item() <= 0:
            raise ValueError("action loss mask has no supervised dimensions")
        return (action_loss * weights).sum() / total_weight

    def _subtask_ce(
        self,
        observation,
        images: tuple[Tensor, ...],
        image_masks: tuple[Tensor, ...],
    ) -> Tensor:
        required = (
            observation.tokenized_subtask_prompt,
            observation.tokenized_subtask_prompt_mask,
            observation.subtask_ar_mask,
            observation.subtask_loss_mask,
        )
        if any(value is None for value in required):
            raise ValueError("online subtask training fields are missing")
        tokens = observation.tokenized_subtask_prompt
        token_masks = observation.tokenized_subtask_prompt_mask
        prefix_embs, pad_masks, att_masks, _ = self.embed_prefix(
            images,
            image_masks,
            tokens,
            token_masks,
            observation.subtask_ar_mask,
        )
        attention = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks)).to(
            dtype=prefix_embs.dtype
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        language_hidden = prefix_out[:, -tokens.shape[1] :]
        logits = self.paligemma_with_expert.language_logits(language_hidden[:, :-1])
        return next_token_cross_entropy(
            logits,
            tokens,
            observation.subtask_loss_mask,
        )

    def forward(
        self,
        observation,
        actions,
        noise=None,
        time=None,
        *,
        return_aux=False,
    ):
        if not self.online_subtask_prediction:
            return self._action_loss(
                observation,
                actions,
                noise=noise,
                time=time,
                return_aux=return_aux,
            )
        if return_aux:
            raise ValueError("online subtask prediction does not use CASM auxiliary loss")
        if self.casm_mode != "none":
            raise ValueError("online subtask prediction requires standard PI0.5 action loss")
        if observation.tokenized_action_prompt is None or observation.tokenized_action_prompt_mask is None:
            raise ValueError("teacher-forced action prompt is missing")
        action_observation = observation.replace(
            tokenized_prompt=observation.tokenized_action_prompt,
            tokenized_prompt_mask=observation.tokenized_action_prompt_mask,
        )
        images, image_masks, _, _, _ = self._preprocess_observation(
            action_observation,
            train=True,
        )
        action_elementwise = self._action_loss(
            action_observation,
            actions,
            noise=noise,
            time=time,
            preprocessed=(images, image_masks),
        )
        action_loss = self._reduce_action_objective(
            action_elementwise,
            observation.action_mask,
        )
        subtask_ce = self._subtask_ce(observation, images, image_masks)
        return OnlineSubtaskLoss(
            total=action_loss + self.lambda_subtask * subtask_ce,
            action=action_loss,
            subtask_ce=subtask_ce,
            action_elementwise=action_elementwise,
        )

    def _normalize_tasks(
        self,
        task: str | Sequence[str],
        batch_size: int,
    ) -> list[str]:
        if isinstance(task, str):
            return [task] * batch_size
        tasks = list(task)
        if len(tasks) != batch_size or not all(isinstance(item, str) for item in tasks):
            raise ValueError("task batch does not match the observation batch")
        return tasks

    def _tokenized_inference_observation(
        self,
        observation,
        tasks: list[str],
    ):
        if self.subtask_tokenizer is None:
            raise ValueError("online subtask prediction is disabled")
        states = observation.state.detach().to(torch.float32).cpu().numpy()
        tokenized = [
            self.subtask_tokenizer.tokenize_subtask(task, state) for task, state in zip(tasks, states, strict=True)
        ]
        tokens = torch.as_tensor(
            np.stack([item[0] for item in tokenized]),
            dtype=torch.long,
            device=observation.state.device,
        )
        masks = torch.as_tensor(
            np.stack([item[1] for item in tokenized]),
            dtype=torch.bool,
            device=observation.state.device,
        )
        return observation.replace(
            tokenized_prompt=tokens,
            tokenized_prompt_mask=masks,
        )

    @torch.no_grad()
    def predict_subtask(
        self,
        observation,
        task: str | Sequence[str],
        *,
        max_new_tokens: int = 32,
    ) -> str | list[str]:
        if not self.online_subtask_prediction:
            raise ValueError("online subtask prediction is disabled")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        batch_size = observation.state.shape[0]
        tasks = self._normalize_tasks(task, batch_size)
        prediction_observation = self._tokenized_inference_observation(
            observation,
            tasks,
        )
        images, image_masks, tokens, token_masks, _ = self._preprocess_observation(prediction_observation, train=False)
        prefix_embs, prefix_masks, prefix_att_masks, _ = self.embed_prefix(
            images,
            image_masks,
            tokens,
            token_masks,
        )
        prefix_attention = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_masks, prefix_att_masks)).to(
            dtype=prefix_embs.dtype
        )
        position_ids = torch.cumsum(prefix_masks, dim=1) - 1
        (prefix_out, _), cache = self.paligemma_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        batch_indices = torch.arange(batch_size, device=prefix_out.device)
        current_hidden = prefix_out[
            batch_indices,
            prefix_masks.sum(dim=1) - 1,
        ]
        generated = []
        generated_masks = []
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=prefix_out.device,
        )
        for step in range(max_new_tokens):
            logits = self.paligemma_with_expert.language_logits(current_hidden)
            next_token = torch.argmax(logits, dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(
                    next_token,
                    self.subtask_tokenizer.eos_token_id,
                ),
                next_token,
            )
            generated.append(next_token)
            finished |= next_token == self.subtask_tokenizer.eos_token_id
            token_emb = self.paligemma_with_expert.embed_language_tokens(next_token[:, None])
            token_emb *= math.sqrt(token_emb.shape[-1])
            generated_masks.append(
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=token_emb.device,
                )
            )
            full_mask = torch.cat(
                [prefix_masks, *generated_masks],
                dim=1,
            )
            attention = self._prepare_attention_masks_4d(full_mask[:, None, :]).to(dtype=token_emb.dtype)
            token_positions = prefix_masks.sum(dim=1, keepdim=True) + step
            (token_out, _), cache = self.paligemma_with_expert.forward(
                attention_mask=attention,
                position_ids=token_positions,
                past_key_values=cache,
                inputs_embeds=[token_emb, None],
                use_cache=True,
            )
            current_hidden = token_out[:, -1]

        generated_tokens = torch.stack(generated, dim=1).cpu().numpy()
        predictions = [self.subtask_tokenizer.decode_subtask(row) for row in generated_tokens]
        for prediction in predictions:
            self._validate_generated_subtask(prediction)
        if isinstance(task, str) and batch_size == 1:
            return predictions[0]
        return predictions

    def _validate_generated_subtask(self, subtask: str) -> None:
        if self.subtask_tokenizer is None:
            raise ValueError("online subtask prediction is disabled")
        self.subtask_tokenizer.validate_subtask_text(subtask)

    @torch.no_grad()
    def sample_actions_with_subtask(
        self,
        observation,
        task: str | Sequence[str],
        noise=None,
        num_steps=10,
    ) -> tuple[Tensor, str | list[str]]:
        if self.subtask_tokenizer is None:
            raise ValueError("online subtask prediction is disabled")
        device = observation.state.device
        tasks = self._normalize_tasks(task, observation.state.shape[0])
        generated = self.predict_subtask(observation, task)
        subtasks = [generated] if isinstance(generated, str) else generated
        states = observation.state.detach().to(torch.float32).cpu().numpy()
        tokenized = [
            self.subtask_tokenizer.tokenize_action_prompt(
                current_task,
                state,
                subtask,
                self.config.subtask_action_prompt_format,
            )
            for current_task, state, subtask in zip(
                tasks,
                states,
                subtasks,
                strict=True,
            )
        ]
        action_tokens = torch.as_tensor(
            np.stack([item[0] for item in tokenized]),
            dtype=torch.long,
            device=observation.state.device,
        )
        action_masks = torch.as_tensor(
            np.stack([item[1] for item in tokenized]),
            dtype=torch.bool,
            device=observation.state.device,
        )
        action_observation = observation.replace(
            tokenized_prompt=action_tokens,
            tokenized_prompt_mask=action_masks,
        )
        actions = self.sample_actions(
            device,
            action_observation,
            noise=noise,
            num_steps=num_steps,
        )
        if isinstance(task, str) and len(subtasks) == 1:
            return actions, subtasks[0]
        return actions, subtasks

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
