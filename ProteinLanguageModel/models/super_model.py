import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import FILTER_LOGITS_FN


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def get_nb_trainable_parameters(model):
    """
    Compute model parameter counts.

    Args:
        model (torch.nn.Module): The model to analyze.

    Returns:
        tuple[int, int]: (trainable_params, total_params)
            trainable_params: number of parameters with requires_grad=True.
            total_params: total number of parameters (including dtype-specific adjustments).
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging, description=""):
    """
    Log counts of trainable and total parameters as an informational message.

    Args:
        model (torch.nn.Module): The model to analyze.
        logging: Logger instance (with .info) for output.
        description (str, optional): Prefix description for the log message.

    Returns:
        None
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"{description} trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


class SuperModel(nn.Module):
    """
    SuperModel is a decoder-only transformer for autoregressive protein sequence modeling using the x-transformers library.

    It processes tokenized protein sequences through embeddings and transformer layers to predict next tokens.
    Supports training with teacher-forcing and inference with autoregressive generation.

    Args:
        configs (object): Model configuration with hyperparameters (e.g., dimension, depth, heads).
        tokenizer_vocab_size (int): Size of the input vocabulary.
        inference (bool, default=False): If True, wraps for autoregressive generation (full setup via prepare_models).
        pad_token_id (int, default=0): Padding token ID used for generation.

    Forward:
        Expects batch dict with 'input_ids' (tokens), 'mask' (attention mask), 'target_ids', 'original_sequence'.

        Returns dict:
            - Training: 'decoder_output' (logits), 'target_ids', 'mask', 'original_sequence'; 'loss'=None.
            - Inference: 'loss' (scalar), others as above; 'decoder_output'=None.

    Notes:
        Padding token=0 ignored in generation. Configurable for efficiency (e.g., Flash Attention, rotary embeddings).
    """
    def __init__(self, configs, tokenizer_vocab_size, inference=False, pad_token_id=0):
        super().__init__()
        self.configs = configs
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self._inference_mode = inference
        self.pad_token_id = pad_token_id
        model_cfg = self.configs.model
        self.num_memory_tokens = int(_cfg_get(model_cfg, "num_memory_tokens", 0) or 0)
        condition_cfg = _cfg_get(model_cfg, "condition_tokens", None)
        seq2struct_cfg = _cfg_get(condition_cfg, "sequence_to_structure", None)
        context_cfg = _cfg_get(seq2struct_cfg, "protein_encoder_context", None)
        self.sequence_to_structure_use_protein_encoder_context = bool(
            _cfg_get(context_cfg, "enable", _cfg_get(context_cfg, "enabled", False))
        )
        self.protein_encoder_context_model = None
        self.protein_encoder_context_project = nn.Identity()
        self.protein_encoder_context_trainable = False
        self.use_causal_no_pad_attention_mask_fast_path = (
            bool(_cfg_get(model_cfg, "enable_unmasked_attention_fast_path", False))
            and bool(_cfg_get(model_cfg, "attn_flash", False))
            and not self._inference_mode
            and not self.sequence_to_structure_use_protein_encoder_context
            and self.num_memory_tokens == 0
        )
        self.attention_fast_path_disabled_reasons = []
        if not bool(_cfg_get(model_cfg, "attn_flash", False)):
            self.attention_fast_path_disabled_reasons.append("model.attn_flash is false")
        if self._inference_mode:
            self.attention_fast_path_disabled_reasons.append("inference mode uses generation masks/cache")
        if self.sequence_to_structure_use_protein_encoder_context:
            self.attention_fast_path_disabled_reasons.append("E1 protein_encoder_context prepend is enabled")
        if self.num_memory_tokens != 0:
            self.attention_fast_path_disabled_reasons.append("model.num_memory_tokens is nonzero")

        decoder_layers = Decoder(
            dim=model_cfg.dimension,
            ff_mult=model_cfg.ff_mult,
            depth=model_cfg.depth,
            heads=model_cfg.heads,
            rotary_pos_emb=model_cfg.rotary_pos_emb,
            attn_flash=model_cfg.attn_flash,
            attn_kv_heads=model_cfg.attn_kv_heads,
            attn_qk_norm=model_cfg.qk_norm,
            pre_norm=model_cfg.pre_norm,
            residual_attn=model_cfg.residual_attn,
            layer_dropout=model_cfg.layer_dropout,
            attn_dropout=model_cfg.attn_dropout,
            ff_dropout=model_cfg.ff_dropout,
        )

        self.transformer = TransformerWrapper(
            num_tokens=self.tokenizer_vocab_size,
            max_seq_len=model_cfg.max_len,
            attn_layers=decoder_layers,
            emb_dropout=model_cfg.emb_dropout,
            tie_embedding=model_cfg.tie_embedding,
        )

        if self.sequence_to_structure_use_protein_encoder_context:
            model_type = str(_cfg_get(context_cfg, "model_type", "")).strip().lower()
            model_name = _cfg_get(context_cfg, "model_name")
            if model_type != "e1":
                raise ValueError(
                    "condition_tokens.sequence_to_structure.protein_encoder_context.model_type "
                    "must be 'e1' when enabled."
                )
            if not model_name:
                raise ValueError(
                    "condition_tokens.sequence_to_structure.protein_encoder_context.model_name "
                    "is required when enabled."
                )
            try:
                from E1.modeling import E1Model
            except ImportError as exc:
                raise ImportError(
                    "E1 package is required for protein_encoder_context (model_type='e1')."
                ) from exc

            # Patch upstream E1 for torch.compile compatibility.
            # This is a no-op if E1 isn't available (handled above) or already patched.
            from models.e1_compile_patch import patch_e1_for_torch_compile
            patch_e1_for_torch_compile()

            self.protein_encoder_context_model = E1Model.from_pretrained(model_name)
            for param in self.protein_encoder_context_model.parameters():
                param.requires_grad = False
            self.protein_encoder_context_model.eval()

            fine_tune_cfg = _cfg_get(context_cfg, "fine_tune", None)
            if bool(_cfg_get(fine_tune_cfg, "enable", False)):
                last_layers = max(0, int(_cfg_get(fine_tune_cfg, "last_layers_trainable", 1)))
                if last_layers > 0:
                    layers = getattr(self.protein_encoder_context_model, "layers", None)
                    if layers is None:
                        layers = getattr(getattr(self.protein_encoder_context_model, "model", None), "layers", None)
                    if layers is None:
                        raise RuntimeError("E1 model does not expose `layers` for fine-tuning.")
                    for block in layers[-last_layers:]:
                        for param in block.parameters():
                            param.requires_grad = True

            self.protein_encoder_context_trainable = any(
                p.requires_grad for p in self.protein_encoder_context_model.parameters()
            )
            hidden_size = int(getattr(self.protein_encoder_context_model.config, "hidden_size", 0))
            if hidden_size <= 0:
                raise RuntimeError("Failed to resolve E1 hidden_size from the config.")
            if hidden_size != model_cfg.dimension:
                self.protein_encoder_context_project = nn.Linear(hidden_size, model_cfg.dimension)

        if inference:
            self.transformer = AutoregressiveWrapper(
                self.transformer, 
                ignore_index=pad_token_id, 
                pad_value=pad_token_id
            )

    @staticmethod
    def _build_left_padded_prepend(
        residue_embeds: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        lengths = residue_mask.sum(dim=1).to(torch.long)
        if lengths.numel() == 0:
            return None, None, 0

        # Keep prepend width compile-stable: using a batch-dependent max length causes
        # graph churn and poor utilization under torch.compile.
        max_prepend = int(residue_embeds.shape[1])
        if max_prepend <= 0:
            return None, None, 0

        idx = (
            torch.arange(max_prepend, device=residue_embeds.device, dtype=torch.long).unsqueeze(0)
            - (max_prepend - lengths).unsqueeze(1)
        )
        prepend_mask = idx >= 0
        safe_idx = idx.clamp_min(0)
        prepend_embeds = residue_embeds.gather(
            dim=1,
            index=safe_idx.unsqueeze(-1).expand(-1, -1, residue_embeds.shape[-1]),
        )
        prepend_embeds = prepend_embeds.masked_fill(~prepend_mask.unsqueeze(-1), 0.0)
        return prepend_embeds, prepend_mask, max_prepend

    def _get_seq2struct_active_flags(self, batch) -> torch.Tensor | None:
        flags = batch.get("is_sequence_to_structure_conditioned")
        if flags is None:
            return None
        if isinstance(flags, torch.Tensor):
            return flags.to(device=batch["input_ids"].device, dtype=torch.bool).view(-1)
        return torch.as_tensor(flags, device=batch["input_ids"].device, dtype=torch.bool).view(-1)

    def _encode_protein_context(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        required = (
            "protein_encoder_context_input_ids",
            "protein_encoder_context_within_seq_position_ids",
            "protein_encoder_context_global_position_ids",
            "protein_encoder_context_sequence_ids",
            "protein_encoder_context_attention_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise RuntimeError(
                f"protein_encoder_context is enabled but missing batch keys: {missing}"
            )

        model = self.protein_encoder_context_model
        assert model is not None

        input_ids = batch["protein_encoder_context_input_ids"]
        within_seq_position_ids = batch["protein_encoder_context_within_seq_position_ids"]
        global_position_ids = batch["protein_encoder_context_global_position_ids"]
        sequence_ids = batch["protein_encoder_context_sequence_ids"]
        residue_attention_mask = batch["protein_encoder_context_attention_mask"]
        if residue_attention_mask.dtype != torch.bool:
            residue_attention_mask = residue_attention_mask.bool()

        if not self.protein_encoder_context_trainable:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    within_seq_position_ids=within_seq_position_ids,
                    global_position_ids=global_position_ids,
                    sequence_ids=sequence_ids,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                within_seq_position_ids=within_seq_position_ids,
                global_position_ids=global_position_ids,
                sequence_ids=sequence_ids,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

        # Exclude E1 boundary/special tokens: keep residue range only.
        residue_len = residue_attention_mask.shape[1]
        residue_embeds = outputs.last_hidden_state[:, 2:2 + residue_len, :]
        residue_embeds = residue_embeds * residue_attention_mask.unsqueeze(-1).to(residue_embeds.dtype)
        residue_embeds = self.protein_encoder_context_project(residue_embeds)
        return residue_embeds, residue_attention_mask

    def _split_protein_context_kwargs(
        self,
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> tuple[dict | None, dict]:
        forwarded = dict(kwargs)
        if not self.sequence_to_structure_use_protein_encoder_context:
            return None, forwarded

        required = (
            "protein_encoder_context_input_ids",
            "protein_encoder_context_within_seq_position_ids",
            "protein_encoder_context_global_position_ids",
            "protein_encoder_context_sequence_ids",
            "protein_encoder_context_attention_mask",
        )
        present = [key for key in required if key in forwarded]
        if not present:
            raise RuntimeError(
                "This checkpoint uses protein_encoder_context; generate() requires "
                "protein_encoder_context_* tensors."
            )

        missing = [key for key in required if key not in forwarded]
        if missing:
            raise RuntimeError(
                f"protein_encoder_context is enabled but missing generation kwargs: {missing}"
            )

        batch = {"input_ids": input_ids}
        for key in required:
            batch[key] = forwarded.pop(key)
        if "is_sequence_to_structure_conditioned" in forwarded:
            batch["is_sequence_to_structure_conditioned"] = forwarded.pop(
                "is_sequence_to_structure_conditioned"
            )

        return batch, forwarded

    def _prepare_protein_context_from_batch(self, batch: dict) -> dict:
        active_flags = self._get_seq2struct_active_flags(batch)
        if active_flags is None:
            active_flags = torch.ones(
                batch["input_ids"].shape[0],
                dtype=torch.bool,
                device=batch["input_ids"].device,
            )

        residue_embeds, residue_mask = self._encode_protein_context(batch)
        residue_mask = residue_mask & active_flags.unsqueeze(1)
        prepend_embeds, prepend_mask, _ = self._build_left_padded_prepend(
            residue_embeds,
            residue_mask,
        )
        if prepend_embeds is None or prepend_mask is None:
            return {}
        return {
            "prepend_embeds": prepend_embeds,
            "prepend_mask": prepend_mask,
        }

    @staticmethod
    def _drop_protein_context_source_kwargs(kwargs: dict) -> None:
        for key in (
            "protein_encoder_context_input_ids",
            "protein_encoder_context_within_seq_position_ids",
            "protein_encoder_context_global_position_ids",
            "protein_encoder_context_sequence_ids",
            "protein_encoder_context_attention_mask",
            "is_sequence_to_structure_conditioned",
        ):
            kwargs.pop(key, None)

    @staticmethod
    def _pop_precomputed_prepend_kwargs(
        input_ids: torch.Tensor,
        kwargs: dict,
    ) -> dict | None:
        has_prepend_embeds = "prepend_embeds" in kwargs
        has_prepend_mask = "prepend_mask" in kwargs
        if not has_prepend_embeds and not has_prepend_mask:
            return None
        if has_prepend_embeds != has_prepend_mask:
            raise RuntimeError(
                "Precomputed protein context requires both prepend_embeds and prepend_mask."
            )

        prepend_embeds = kwargs.pop("prepend_embeds")
        prepend_mask = kwargs.pop("prepend_mask")
        if not isinstance(prepend_embeds, torch.Tensor) or not isinstance(prepend_mask, torch.Tensor):
            raise RuntimeError("prepend_embeds and prepend_mask must be torch tensors.")
        if prepend_embeds.ndim != 3 or prepend_mask.ndim != 2:
            raise RuntimeError(
                "prepend_embeds must have shape (batch, prepend_len, dim) and "
                "prepend_mask must have shape (batch, prepend_len)."
            )
        if prepend_embeds.shape[:2] != prepend_mask.shape:
            raise RuntimeError("prepend_embeds and prepend_mask shapes do not match.")
        if prepend_embeds.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                "Precomputed protein context batch size does not match input_ids."
            )

        return {
            "prepend_embeds": prepend_embeds,
            "prepend_mask": prepend_mask,
        }

    def prepare_protein_context_kwargs(self, input_ids: torch.Tensor, **kwargs) -> dict:
        forwarded = dict(kwargs)
        precomputed = self._pop_precomputed_prepend_kwargs(input_ids, forwarded)
        if precomputed is not None:
            return precomputed

        batch, _ = self._split_protein_context_kwargs(input_ids, forwarded)
        if batch is None:
            return {}
        return self._prepare_protein_context_from_batch(batch)

    @staticmethod
    def _trim_left_prepend_pad_from_cache(cache, left_pad: int) -> None:
        if left_pad <= 0 or cache is None:
            return
        for inter in cache.attn_intermediates or []:
            cached_kv = getattr(inter, "cached_kv", None)
            if cached_kv is None or not isinstance(cached_kv, tuple) or len(cached_kv) != 2:
                continue
            k, v = cached_kv
            inter.cached_kv = (k[..., left_pad:, :], v[..., left_pad:, :])

    @staticmethod
    def _resolve_filter_logits_fn(filter_logits_fn):
        if callable(filter_logits_fn):
            return filter_logits_fn
        if isinstance(filter_logits_fn, str):
            if filter_logits_fn not in FILTER_LOGITS_FN:
                available = ", ".join(sorted(FILTER_LOGITS_FN))
                raise ValueError(f"Unknown filter_logits_fn='{filter_logits_fn}'. Available: {available}")
            return FILTER_LOGITS_FN[filter_logits_fn]
        return None

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        filter_logits_fn,
        filter_kwargs: dict,
    ) -> torch.Tensor:
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        if filter_logits_fn is None:
            raise ValueError("Stochastic generation requires a valid filter_logits_fn.")
        filtered_logits = filter_logits_fn(logits, **filter_kwargs)
        probs = torch.softmax(filtered_logits / temperature, dim=-1)
        return torch.multinomial(probs, 1)

    @torch.no_grad()
    def _generate_with_protein_prefix_cache(
        self,
        prompts: torch.Tensor,
        seq_len: int,
        eos_token: int | None,
        temperature: float,
        filter_logits_fn,
        filter_kwargs: dict,
        prompt_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if prompts.ndim != 2:
            raise ValueError(
                "protein_encoder_context generation currently expects prompts as a 2D tensor "
                "(batch, prompt_len)."
            )
        if seq_len <= 0:
            return prompts.new_empty((prompts.shape[0], 0))

        wrapper = self.transformer
        net = wrapper.net
        if not getattr(net, "can_cache_kv", False):
            raise RuntimeError("Transformer backend does not support KV caching.")

        filter_logits_fn = self._resolve_filter_logits_fn(filter_logits_fn)
        filter_kwargs = filter_kwargs or {}

        prompts = prompts.to(dtype=torch.long)
        batch_size = int(prompts.shape[0])
        if prompt_lens is not None:
            prompt_lens = prompt_lens.to(device=prompts.device, dtype=torch.long).view(-1)
            if prompt_lens.shape[0] != batch_size:
                raise ValueError("prompt_lens batch dimension does not match prompts.")

        generation_kwargs = dict(kwargs)
        prepend_kwargs = self._pop_precomputed_prepend_kwargs(prompts, generation_kwargs)
        if prepend_kwargs is None:
            context_batch, generation_kwargs = self._split_protein_context_kwargs(
                prompts,
                generation_kwargs,
            )
            if context_batch is None:
                raise RuntimeError("protein_encoder_context cache generation requires context tensors.")
            prepend_kwargs = self._prepare_protein_context_from_batch(context_batch)
        else:
            self._drop_protein_context_source_kwargs(generation_kwargs)
        if not prepend_kwargs:
            raise RuntimeError("Failed to prepare prepend context for cached generation.")

        prepend_mask = prepend_kwargs.get("prepend_mask")
        if not isinstance(prepend_mask, torch.Tensor):
            raise RuntimeError("Cached protein context generation requires prepend_mask.")

        prompt_lens_for_grouping = (
            prompt_lens
            if prompt_lens is not None
            else torch.full((batch_size,), prompts.shape[1], device=prompts.device, dtype=torch.long)
        )
        left_pads = (~prepend_mask).sum(dim=1).to(dtype=torch.long)

        groups: dict[tuple[int, int], list[int]] = {}
        for row_idx in range(batch_size):
            key = (
                int(prompt_lens_for_grouping[row_idx].item()),
                int(left_pads[row_idx].item()),
            )
            groups.setdefault(key, []).append(row_idx)

        outputs: list[torch.Tensor | None] = [None] * batch_size
        for (prompt_len, left_pad), row_indices in groups.items():
            index = torch.as_tensor(row_indices, device=prompts.device, dtype=torch.long)
            group_prompt = prompts.index_select(0, index)[:, :prompt_len]
            group_prepend_kwargs = {
                key: value.index_select(0, index)
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
                for key, value in prepend_kwargs.items()
            }
            group_generation_kwargs = {
                key: value.index_select(0, index)
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
                for key, value in generation_kwargs.items()
            }

            out = group_prompt
            logits, cache = net(
                out,
                return_intermediates=True,
                **group_prepend_kwargs,
                **group_generation_kwargs,
            )
            # Training uses a fixed-width, left-padded E1 prepend block. Grouping
            # by left_pad keeps this cache trim uniform across the batched decode.
            self._trim_left_prepend_pad_from_cache(cache, left_pad)
            next_token = self._sample_next_token(
                logits[:, -1],
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
            )
            group_size = int(group_prompt.shape[0])
            generated = group_prompt.new_full((group_size, seq_len), wrapper.pad_value)
            generated[:, 0:1] = next_token
            generated_steps = 1
            finished = (
                next_token.squeeze(-1).eq(eos_token)
                if eos_token is not None
                else torch.zeros(group_size, device=prompts.device, dtype=torch.bool)
            )

            for step_idx in range(1, seq_len):
                if eos_token is not None and finished.all():
                    break

                step_input = next_token.masked_fill(finished.unsqueeze(-1), wrapper.pad_value)
                logits, cache = net(
                    step_input,
                    return_intermediates=True,
                    cache=cache,
                    input_not_include_cache=True,
                    **group_generation_kwargs,
                )
                next_token = self._sample_next_token(
                    logits[:, -1],
                    temperature=temperature,
                    filter_logits_fn=filter_logits_fn,
                    filter_kwargs=filter_kwargs,
                )
                generated[:, step_idx : step_idx + 1] = next_token
                generated_steps = step_idx + 1

                if eos_token is not None:
                    finished = finished | next_token.squeeze(-1).eq(eos_token)

            generated = generated[:, :generated_steps]
            if eos_token is not None:
                is_eos_tokens = generated == eos_token
                shifted_is_eos = F.pad(is_eos_tokens, (1, -1))
                eos_mask = shifted_is_eos.float().cumsum(dim=-1) >= 1
                generated = generated.masked_fill(eos_mask, wrapper.pad_value)

            for local_idx, row_idx in enumerate(row_indices):
                outputs[row_idx] = generated[local_idx]

        if any(output is None for output in outputs):
            raise RuntimeError("Internal error: missing generated rows after batched decode.")
        return pad_sequence(outputs, batch_first=True, padding_value=wrapper.pad_value)

    def forward(self, batch):
        output_dict = {}

        input_mask = batch['mask']
        if input_mask.dtype != torch.bool:
            input_mask = input_mask.bool()

        prepend_embeds = None
        prepend_mask = None
        prepend_len = 0
        if self.sequence_to_structure_use_protein_encoder_context:
            active_flags = self._get_seq2struct_active_flags(batch)
            if active_flags is None:
                active_flags = torch.ones(
                    batch["input_ids"].shape[0],
                    dtype=torch.bool,
                    device=batch["input_ids"].device,
                )
            residue_embeds, residue_mask = self._encode_protein_context(batch)
            residue_mask = residue_mask & active_flags.unsqueeze(1)
            prepend_embeds, prepend_mask, prepend_len = self._build_left_padded_prepend(
                residue_embeds,
                residue_mask,
            )

        if prepend_embeds is None:
            if self.use_causal_no_pad_attention_mask_fast_path:
                decoder_output = self.transformer(batch["input_ids"])
            else:
                decoder_output = self.transformer(batch["input_ids"], mask=input_mask)
        else:
            decoder_output = self.transformer(
                batch["input_ids"],
                mask=input_mask,
                prepend_embeds=prepend_embeds,
                prepend_mask=prepend_mask,
            )
            if not self._inference_mode and prepend_len > 0:
                decoder_output = decoder_output[:, prepend_len:, :]

        if self._inference_mode:
            output_dict['loss'] = decoder_output  # scalar loss value
            output_dict["decoder_output"] = None
        else:
            output_dict["decoder_output"] = decoder_output  # (B, L, decoder_dim)
            output_dict['loss'] = None  # loss will be computed externally

        output_dict["original_sequence"] = batch["original_sequence"]  # Original sequences
        output_dict["target_ids"] = batch["target_ids"]
        output_dict["mask"] = input_mask  # (B, L) boolean mask

        return output_dict

    @staticmethod
    def _first_linear_weight_shape(module):
        weight = getattr(module, "weight", None)
        if weight is not None:
            return tuple(weight.shape)
        if isinstance(module, nn.Sequential):
            for child in module:
                shape = SuperModel._first_linear_weight_shape(child)
                if shape is not None:
                    return shape
        return None

    def _get_attention_flop_widths(self):
        """Read attention projection widths from the instantiated decoder."""
        model_cfg = self.configs.model
        d_model = int(model_cfg.dimension)
        num_heads = int(model_cfg.heads)
        kv_heads = int(getattr(model_cfg, "attn_kv_heads", num_heads) or num_heads)

        for module in self.modules():
            if module.__class__.__name__ != "Attention":
                continue

            q_shape = self._first_linear_weight_shape(getattr(module, "to_q", None))
            k_shape = self._first_linear_weight_shape(getattr(module, "to_k", None))
            v_shape = self._first_linear_weight_shape(getattr(module, "to_v", None))
            out_shape = self._first_linear_weight_shape(getattr(module, "to_out", None))

            if not all((q_shape, k_shape, v_shape, out_shape)):
                continue

            return {
                "q_in": int(q_shape[1]),
                "q_out": int(q_shape[0]),
                "k_in": int(k_shape[1]),
                "k_out": int(k_shape[0]),
                "v_in": int(v_shape[1]),
                "v_out": int(v_shape[0]),
                "out_in": int(out_shape[1]),
                "out_out": int(out_shape[0]),
                "head_dim": int(q_shape[0]) // num_heads,
            }

        # Fallback mirrors the current x-transformers default when no Attention
        # module is discoverable, but normal training should use the path above.
        head_dim = int(
            _cfg_get(model_cfg, "dim_head", None)
            or _cfg_get(model_cfg, "attn_dim_head", None)
            or 64
        )
        value_dim_head = int(_cfg_get(model_cfg, "value_dim_head", None) or head_dim)
        return {
            "q_in": d_model,
            "q_out": head_dim * num_heads,
            "k_in": d_model,
            "k_out": head_dim * kv_heads,
            "v_in": d_model,
            "v_out": value_dim_head * kv_heads,
            "out_in": value_dim_head * num_heads,
            "out_out": d_model,
            "head_dim": head_dim,
        }

    def get_attention_head_dim(self):
        return self._get_attention_flop_widths()["head_dim"]

    def estimate_forward_flops(self, seq_len, batch_size=1):
        """Estimate forward-pass FLOPs using the Chinchilla accounting scheme.

        Args:
            seq_len (int): Number of tokens processed per sequence (excluding padding).
            batch_size (int): Number of sequences processed in the batch.

        Returns:
            int: Estimated multiply-add operations for the forward pass of the batch. Note that input
            embedding lookups are treated as cheap gathers and therefore excluded from this tally so the
            result matches standard scaling-law FLOP accounting.
        """
        model_cfg = self.configs.model
        d_model = int(model_cfg.dimension)
        num_layers = int(model_cfg.depth)
        num_heads = int(model_cfg.heads)
        attn_widths = self._get_attention_flop_widths()
        ff_width = d_model * int(model_cfg.ff_mult)
        vocab_size = int(self.tokenizer_vocab_size)

        # Linear projections for Q, K, V, based on actual instantiated widths.
        q_proj = 2 * seq_len * attn_widths["q_in"] * attn_widths["q_out"]
        k_proj = 2 * seq_len * attn_widths["k_in"] * attn_widths["k_out"]
        v_proj = 2 * seq_len * attn_widths["v_in"] * attn_widths["v_out"]
        qkv_flops = q_proj + k_proj + v_proj

        # Decoder attention is causal, so only the lower triangle is valid work.
        causal_positions = seq_len * (seq_len + 1) // 2
        attn_scores = 2 * causal_positions * attn_widths["q_out"]
        softmax_ops = 3 * num_heads * causal_positions
        softmax_reduce = 2 * causal_positions * attn_widths["out_in"]

        # Output projection after attention heads are concatenated.
        attn_output = 2 * seq_len * attn_widths["out_in"] * attn_widths["out_out"]
        total_attention = qkv_flops + attn_scores + softmax_ops + softmax_reduce + attn_output

        # Feedforward block: projection up and down.
        dense_block = 2 * seq_len * (2 * d_model * ff_width)

        logits = 2 * seq_len * d_model * vocab_size

        forward_per_sample = num_layers * (total_attention + dense_block) + logits
        return int(forward_per_sample * batch_size)

    def estimate_total_training_flops(self, seq_len, batch_size, num_iterations):
        """Estimate total training FLOPs for a given number of iterations.

        Following the Chinchilla paper convention, the backward pass has twice the FLOPs
        of the forward pass, so total FLOPs per iteration = 3 × forward FLOPs.

        Args:
            seq_len (int): Number of tokens processed per sequence (excluding padding).
            batch_size (int): Number of sequences processed in each batch.
            num_iterations (int): Number of training iterations (gradient updates).

        Returns:
            int: Estimated total multiply-add operations for training, still excluding embedding lookups
            as described in :meth:`estimate_forward_flops`.
        """
        forward_flops = self.estimate_forward_flops(seq_len, batch_size)
        # Backward pass has 2× the FLOPs of forward pass (Chinchilla paper)
        flops_per_iteration = 3 * forward_flops
        total_flops = flops_per_iteration * num_iterations
        return int(total_flops)

    def generate(
        self,
        prompts: torch.Tensor,
        seq_len: int,
        eos_token: int = None,
        temperature: float = 1.0,
        filter_logits_fn: str = None,
        filter_kwargs: dict = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively.
        
        This method requires the model to be initialized with inference=True,
        which wraps the transformer with AutoregressiveWrapper.
        
        Args:
            prompts: Starting tokens (batch_size, prompt_len) - typically just BOS tokens
            seq_len: Number of tokens to generate
            eos_token: Token ID for end-of-sequence (stops generation when encountered)
            temperature: Sampling temperature (0 for greedy decoding, >0 for stochastic)
            filter_logits_fn: Filtering function name ('top_k', 'top_p', 'top_a', 'min_p')
            filter_kwargs: Arguments for the filtering function (e.g., {'k': 50} for top_k)
            **kwargs: Additional arguments passed to the underlying generate method
            
        Returns:
            Generated token sequences (batch_size, seq_len)
            
        Raises:
            RuntimeError: If model was not initialized with inference=True
        """
        if not self._inference_mode:
            raise RuntimeError(
                "generate() requires the model to be initialized with inference=True. "
                "The transformer must be wrapped with AutoregressiveWrapper for generation."
            )

        filter_kwargs = filter_kwargs or {}

        generation_kwargs = dict(kwargs)
        if self.sequence_to_structure_use_protein_encoder_context:
            prompt_lens = generation_kwargs.pop("prompt_lens", None)
            use_cache = bool(generation_kwargs.pop("cache_kv", True))
            prepend_kwargs = self._pop_precomputed_prepend_kwargs(prompts, generation_kwargs)
            if prepend_kwargs is not None:
                self._drop_protein_context_source_kwargs(generation_kwargs)
            if prepend_kwargs is None:
                context_batch, generation_kwargs = self._split_protein_context_kwargs(
                    prompts,
                    generation_kwargs,
                )
                if context_batch is not None:
                    prepend_kwargs = self._prepare_protein_context_from_batch(context_batch)

            if use_cache:
                if prepend_kwargs:
                    generation_kwargs.update(prepend_kwargs)
                return self._generate_with_protein_prefix_cache(
                    prompts=prompts,
                    seq_len=seq_len,
                    eos_token=eos_token,
                    temperature=temperature,
                    filter_logits_fn=filter_logits_fn,
                    filter_kwargs=filter_kwargs,
                    prompt_lens=prompt_lens,
                    **generation_kwargs,
                )

            if prompt_lens is not None:
                generation_kwargs["prompt_lens"] = prompt_lens
            if prepend_kwargs:
                generation_kwargs.update(prepend_kwargs)

        return self.transformer.generate(
            prompts=prompts,
            seq_len=seq_len,
            eos_token=eos_token,
            temperature=temperature,
            filter_logits_fn=filter_logits_fn,
            filter_kwargs=filter_kwargs,
            **generation_kwargs
        )


def prepare_models(configs, logging, inference=False, **kwargs):
    """
    Build and return the end-to-end model.

    Args:
        configs: Configuration object containing model and training settings.
        logging: Logger for informational output.
        inference (bool): If True, wrap with AutoregressiveWrapper and freeze model weights.
        **kwargs: Additional arguments:
            - tokenizer_vocab_size (int): Required. Size of the vocabulary.
            - pad_token_id (int): Optional. Padding token ID for generation (default: 0).

    Returns:
        SuperModel: Built final model ready for training or inference.
    """
    # Get pad_token_id from kwargs, default to 0
    pad_token_id = kwargs.get('pad_token_id', 0)

    # Prepare the model
    final_model = SuperModel(
        configs=configs,
        tokenizer_vocab_size=kwargs['tokenizer_vocab_size'],
        inference=inference,
        pad_token_id=pad_token_id
    )
    final_model._compiled_submodules = False
    if getattr(final_model, "use_causal_no_pad_attention_mask_fast_path", False):
        logging.info(
            "Attention fast path enabled: using causal decoder without the padding mask "
            "for eligible training forwards (E1 protein_encoder_context disabled, "
            "model.num_memory_tokens=0). This lets PyTorch SDPA select a Flash/FA2-style "
            "backend when the runtime supports it."
        )
    else:
        reasons = ", ".join(
            getattr(final_model, "attention_fast_path_disabled_reasons", [])
        ) or "unknown reason"
        logging.warning(
            "Attention fast path disabled (%s); using the default masked attention path. "
            "A FA2-compatible fast path for this setup has not been developed/tested yet.",
            reasons,
        )

    # Estimate FLOPs for the actual global batch seen by this launch.
    accelerator = kwargs.get("accelerator", None)
    number_of_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    global_batch_size = int(configs.train_settings.batch_size) * number_of_processes
    attention_head_dim = final_model.get_attention_head_dim()

    total_flops = final_model.estimate_total_training_flops(
        configs.model.max_len,
        global_batch_size,
        1,
    )
    mega_flops = total_flops / 1e6
    giga_flops = total_flops / 1e9
    tera_flops = total_flops / 1e12
    peta_flops = total_flops / 1e15
    logging.info(
        "Estimated non-embedding FLOPs per iteration: %s FLOPs | %.2f MFLOPs | "
        "%.2f GFLOPs | %.6f TFLOPs | %.6f PFLOPs "
        "(seq_len=%s, global_batch_size=%s, attention_head_dim=%s, distributed_processes=%s)",
        f"{total_flops:,}",
        mega_flops,
        giga_flops,
        tera_flops,
        peta_flops,
        configs.model.max_len,
        global_batch_size,
        attention_head_dim,
        number_of_processes,
    )

    if inference:
        # freeze all parameters
        for param in final_model.parameters():
            param.requires_grad = False
        logging.info('Frozen all parameters for inference')

    print_trainable_parameters(final_model, logging, 'SuperModel')

    return final_model
