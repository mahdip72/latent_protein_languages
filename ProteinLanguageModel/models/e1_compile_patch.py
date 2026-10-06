"""
Local compatibility patches for the external `E1` package.

We keep this in-repo (instead of editing site-packages) so the training stack is
reproducible and easy to version-control.
"""

from __future__ import annotations

from typing import Callable


def patch_e1_for_torch_compile() -> bool:
    """
    Patch E1 to make it compatible with `torch.compile`.

    Upstream E1's varlen flex attention builds a tensor via `torch.Tensor([t])`
    where `t` is a scalar tensor. Dynamo can't trace this with FakeTensors and
    compilation fails. We replace it with a pure tensor reshape/view.

    Upstream E1Model.forward also uses `Tensor.item()` only for range-check
    asserts. `Tensor.item()` forces a graph break under TorchDynamo; we replace
    it with tensor-based `torch._assert(...)` checks to keep the graph intact.

    Returns:
        True if the patch was applied (or already applied), False if E1 isn't available.
    """

    try:
        import E1.model.attention as e1_attention
        import E1.model.flash_attention_utils as fau
        import E1.modeling as modeling
        import torch
    except Exception:
        return False

    try:
        import E1.model.varlen_flex_attention as vfa
    except Exception:
        vfa = None

    vfa_patched = True if vfa is None else getattr(vfa, "_PLM_PATCHED_FOR_TORCH_COMPILE", False)
    modeling_patched = getattr(modeling, "_PLM_PATCHED_FOR_TORCH_COMPILE", False)
    attention_patched = getattr(e1_attention, "_PLM_PATCHED_FOR_TORCH_COMPILE", False)
    flash_utils_patched = getattr(fau, "_PLM_PATCHED_FOR_TORCH_COMPILE", False)
    if vfa_patched and modeling_patched and attention_patched and flash_utils_patched:
        return True

    if vfa is not None and not vfa_patched:
        orig_fn: Callable = vfa.block_min_max_seq_ids

        def block_min_max_seq_ids_patched(
            SLEN: "torch.Tensor", block_size: int = 128
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            device = SLEN.device
            total_tokens = torch.sum(SLEN)
            B = (total_tokens + block_size - 1) // block_size
            padding_tokens = B * block_size - total_tokens

            # NOTE: avoid `torch.Tensor([padding_tokens])` which breaks FakeTensor tracing.
            padding_tokens = padding_tokens.to(device=device, dtype=SLEN.dtype).view(1)
            SLEN = torch.cat([SLEN, padding_tokens], dim=0)

            # Cumulative ends (exclusive) for each sequence; cum[i] == end offset of seq i
            cum = torch.cumsum(SLEN.to(torch.long), dim=0)  # (N,)
            total_tokens = cum[-1]  # keep as a scalar tensor to avoid `.item()` graph breaks

            # Block start/end offsets [start, end) in token index space
            block_starts = torch.arange(0, B * block_size, block_size, device=device, dtype=torch.long)  # (B,)
            block_ends = torch.minimum(block_starts + block_size, total_tokens)  # (B,)

            # MIN_SEQ_ID[i] = first sequence whose end > block_start
            MIN_SEQ_ID = torch.searchsorted(cum, block_starts, right=True)

            # MAX_SEQ_ID[i] = sequence containing the last token in the block (block_end - 1)
            last_token_in_block = torch.clamp(block_ends - 1, min=0)  # valid only if block has at least 1 token
            MAX_SEQ_ID = torch.searchsorted(cum, last_token_in_block, right=True)

            return MIN_SEQ_ID, MAX_SEQ_ID

        # Preserve a reference for debugging.
        vfa._plm_original_block_min_max_seq_ids = orig_fn
        vfa.block_min_max_seq_ids = block_min_max_seq_ids_patched
        vfa._PLM_PATCHED_FOR_TORCH_COMPILE = True

    if not attention_patched:
        orig_rotary_forward: Callable = e1_attention.RotaryPositionalEmbedding.forward

        def rotary_forward_patched(  # type: ignore[no-untyped-def]
            self, q, k, position_ids, seq_len=None
        ):
            # Avoid `position_ids.max().item()` which graph-breaks under TorchDynamo.
            # The rotary cache is prebuilt up to `max_position_embeddings` at init,
            # so we only need to assert indices are within range.
            device, dtype = q.device, q.dtype
            if seq_len is None:
                max_pos = position_ids.max()
                torch._assert(max_pos < self.max_seq_len_cached, "Rotary position_ids exceed cached range.")
                seq_len = self.max_seq_len_cached

            if seq_len > self.max_seq_len_cached:
                self._set_sin_cos_cache(seq_len=seq_len, device=device)

            idxs = position_ids.to(device)
            cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]
            sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]

            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
            return q_embed, k_embed

        e1_attention._plm_original_rotary_forward = orig_rotary_forward
        e1_attention.RotaryPositionalEmbedding.forward = rotary_forward_patched  # type: ignore[assignment]
        e1_attention._PLM_PATCHED_FOR_TORCH_COMPILE = True

    if not flash_utils_patched:
        orig_get_unpad_data: Callable = fau._get_unpad_data

        def get_unpad_data_patched(sequence_ids: "torch.Tensor"):  # type: ignore[no-untyped-def]
            # Avoid `seqlens_in_batch.max().item()` graph breaks. For flash-attn kernels,
            # it's sufficient that max_seqlen_* is >= the true max; we use the padded
            # sequence length (static python int) as an upper bound.
            non_pad_indices = sequence_ids != -1
            non_pad_indices = torch.nonzero(non_pad_indices.flatten(), as_tuple=False).flatten()
            seqlen = int(sequence_ids.shape[1])

            sequence_ids_ = sequence_ids + torch.arange(len(sequence_ids), device=sequence_ids.device)[:, None] * 1e5
            sequence_ids_ = sequence_ids_.flatten()[non_pad_indices]
            _, seqlens_in_batch = torch.unique_consecutive(sequence_ids_, return_counts=True)
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
            return non_pad_indices, cu_seqlens, seqlen

        fau._plm_original_get_unpad_data = orig_get_unpad_data
        fau._get_unpad_data = get_unpad_data_patched  # type: ignore[assignment]
        fau._PLM_PATCHED_FOR_TORCH_COMPILE = True

    if not modeling_patched:
        orig_forward = modeling.E1Model.forward

        def forward_patched(  # type: ignore[no-untyped-def]
            self,
            input_ids,
            within_seq_position_ids,
            global_position_ids,
            sequence_ids,
            past_key_values=None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
        ):
            # NOTE: This is a copy of upstream E1Model.forward with only the `.item()`-based
            # range check replaced by tensor-only `torch._assert(...)` to prevent TorchDynamo
            # graph breaks.
            batch_size, seq_length = input_ids.shape

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                if use_cache:
                    modeling.logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

            if use_cache and past_key_values is None:
                past_key_values = modeling.DynamicCache()
            elif not use_cache:
                # To avoid weirdness with gradient checkpointing:
                # https://github.com/huggingface/transformers/issues/28499
                past_key_values = None

            global_position_ids = global_position_ids.view(-1, seq_length).long()
            within_seq_position_ids = within_seq_position_ids.view(-1, seq_length).long()
            sequence_ids = sequence_ids.view(-1, seq_length).long()

            max_position_id = torch.max(within_seq_position_ids)
            min_position_id = torch.min(within_seq_position_ids)
            modeling.torch._assert(
                (max_position_id < self.config.max_num_positions_within_seq) & (min_position_id >= -1),
                f"Position ids must be in the range [-1, {self.config.max_num_positions_within_seq}).",
            )

            inputs_embeds = self.embed_tokens(input_ids)
            # -1 is used to indicate padding tokens, so we need to clamp the sequence ids to 0
            inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids.clamp(min=0))

            # In case we need to do any manual typecasting
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
            hidden_states = inputs_embeds.to(target_dtype)

            # (batch_size, query_length, keyval_length)
            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

            # Create block mask for flex attention
            attention_args = None
            if past_key_values_length == 0:
                block_mask = modeling.create_block_causal_mask_optimized(sequence_ids)
                flex_attention_args = modeling.FlexAttentionArgs(block_mask=block_mask)
                attention_args = modeling.AttentionArgs(flex_attention_args=flex_attention_args)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)  # type: ignore[operator]

                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        within_seq_position_ids,
                        global_position_ids,
                        sequence_ids,
                        attention_args,
                        past_key_values,
                        output_attentions,
                        use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        within_seq_position_ids=within_seq_position_ids,
                        global_position_ids=global_position_ids,
                        sequence_ids=sequence_ids,
                        attention_args=attention_args,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                hidden_states, self_attn_weights, present_key_value = layer_outputs

                if use_cache:
                    # NOTE: it's necessary to re-assign past_key_values because FSDP2
                    # passes certain arguments by value, not by reference.
                    # See https://github.com/huggingface/transformers/issues/38190#issuecomment-2914016168
                    next_decoder_cache = past_key_values = present_key_value

                if output_attentions:
                    all_self_attns += (self_attn_weights,)  # type: ignore[operator]

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            next_cache = next_decoder_cache if use_cache else None

            return modeling.E1ModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        modeling.E1Model._plm_original_forward = orig_forward  # type: ignore[attr-defined]
        modeling.E1Model.forward = forward_patched  # type: ignore[assignment]
        modeling._PLM_PATCHED_FOR_TORCH_COMPILE = True

    return True
