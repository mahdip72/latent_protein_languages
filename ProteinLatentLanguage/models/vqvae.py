import math
from typing import Optional, Tuple
import torch
from torch import nn
from x_transformers import ContinuousTransformerWrapper, Encoder
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, LFQ


class VQVAE(nn.Module):
    """
    VQVAE implements a vector-quantized variational autoencoder for protein embeddings.

    The model projects encoder outputs to a discrete codebook space, applies vector quantization,
    and reconstructs embeddings via a transformer decoder. Each major block (encoder, quantizer,
    decoder) can be individually frozen through configuration flags when fine-grained control over
    trainable parameters is required.

    Args:
        configs: Configuration object with VQVAE and quantization parameters.
        logging: Logger for informational messages.
        protein_encoder_dim: Dimensionality of input embeddings from the ProteinEncoder.

    Forward inputs:
        x (Tensor): Input embeddings of shape (batch, seq_len, protein_encoder_dim).
        mask (Tensor): Boolean mask for sequence positions.

    Returns:
        reconstructed_embeddings: Embeddings before quantization (Tensor).
        indices: Codebook indices selected (Tensor).
        commitment_loss: Quantizer commitment loss (Tensor).
        decoder_output: Reconstructed embeddings after decoder (Tensor).
    """

    def __init__(self, configs, logging, protein_encoder_dim, decoder_only=False):
        super().__init__()
        self.configs = configs
        self.logging = logging
        self.encoder = None
        self.vector_quantizer = None
        self.decoder = None
        self.protein_encoder_dim = protein_encoder_dim
        self.max_length = int(self.configs.model.max_len)
        self.vq_frozen = False
        self.decoder_only = decoder_only

        # Optional causal attention toggles from config (default False if missing)
        self.encoder_causal = getattr(self.configs.model.vqvae.encoder, 'causal', False)
        self.decoder_causal = getattr(self.configs.model.vqvae.decoder, 'causal', False)

        logging.info(f'encoder_causal: {self.encoder_causal}')
        logging.info(f'decoder_causal: {self.decoder_causal}')

        vq_config = self.configs.model.vqvae.vector_quantization
        tik_tok_cfg = getattr(vq_config, 'tik_tok', {}) or {}
        self.tik_tok_enabled = bool(tik_tok_cfg.get('enabled', False))
        self.tik_tok_compression_factor = int(tik_tok_cfg.get('compression_factor', 1))
        self.residual_depth = int(tik_tok_cfg.get('residual_depth', 1))
        self.tik_tok_classifier_weight = float(tik_tok_cfg.get('classifier_weight', 0.0))
        self.use_residual_vq = self.tik_tok_enabled and self.residual_depth > 1

        if self.tik_tok_enabled:
            if self.tik_tok_compression_factor <= 0:
                raise ValueError("TikTok compression_factor must be a positive integer")
            if (self.tik_tok_compression_factor != 1) and (self.tik_tok_compression_factor % 2) != 0:
                raise ValueError("TikTok compression_factor must be an even integer")
            if self.encoder_causal:
                raise ValueError("TikTok latent tokens require a non-causal encoder.")
            if getattr(self.configs.model.vqvae.decoder, 'enabled', False) and self.decoder_causal:
                raise ValueError("TikTok latent tokens require a non-causal decoder.")
            self.latent_token_count = math.ceil(self.configs.model.max_len / self.tik_tok_compression_factor)
        else:
            self.latent_token_count = 0

        encoder_sequence_extension = self.latent_token_count if self.tik_tok_enabled else 0
        self.encoder_max_seq_len = self.max_length + encoder_sequence_extension

        if hasattr(self.configs.model.vqvae, 'decoder') and getattr(self.configs.model.vqvae.decoder, 'enabled', False):
            self.decoder_causal = getattr(self.configs.model.vqvae.decoder, 'causal', False)

        if not self.decoder_only:
            # Prepare the encoder.
            encoder_configs = self.configs.model.vqvae.encoder
            self.encoder = ContinuousTransformerWrapper(
                dim_in=encoder_configs.dimension,
                dim_out=encoder_configs.dimension,
                max_seq_len=self.configs.model.max_len,
                num_memory_tokens=encoder_configs.num_memory_tokens,
                attn_layers=Encoder(
                    dim=encoder_configs.dimension,
                    ff_mult=encoder_configs.ff_mult,
                    depth=encoder_configs.depth,
                    heads=encoder_configs.heads,
                    rotary_pos_emb=encoder_configs.rotary_pos_emb,
                    attn_flash=encoder_configs.attn_flash,
                    attn_kv_heads=encoder_configs.attn_kv_heads,
                    attn_qk_norm=encoder_configs.qk_norm,
                    pre_norm=encoder_configs.pre_norm,
                    residual_attn=encoder_configs.residual_attn,
                )
            )

            self.encoder_tail = nn.Linear(self.protein_encoder_dim, encoder_configs.dimension)

            # Get VQ dimension from config
            vq_dim = self.configs.model.vqvae.vector_quantization.dim
            self.encoder_head = nn.Linear(encoder_configs.dimension, vq_dim)

        self.decoder_mask_token = None
        self.tik_tok_latent_tokens = None
        self.tik_tok_padding_classifier = None

        if not self.decoder_only:
            if self.tik_tok_enabled and self.latent_token_count > 0:
                self.tik_tok_latent_tokens = nn.Parameter(
                    torch.randn(self.latent_token_count, encoder_configs.dimension)
                )
                self.decoder_mask_token = nn.Parameter(torch.randn(vq_dim))
                if configs.model.vqvae.vector_quantization.tik_tok.compression_factor > 1:
                    logging.info(f"TikTok compression factor: {self.tik_tok_compression_factor}")
                    logging.info(f"TikTok latent token count: {self.latent_token_count}")
                    self.tik_tok_padding_classifier = nn.Linear(vq_dim, self.tik_tok_compression_factor)

        # Create vector quantizer based on type
        if self.configs.model.vqvae.vector_quantization.enabled:
            self.vector_quantizer = self._create_vector_quantizer()
        else:
            self.vector_quantizer = None

        vq_cfg = self.configs.model.vqvae.vector_quantization
        self.vq_frozen = bool(getattr(vq_cfg, 'freeze_parameters', False))

        self.classifier_head_enabled = False
        if self.configs.model.vqvae.decoder.enabled:
            # Prepare the decoder.
            decoder_configs = self.configs.model.vqvae.decoder
            self.decoder = ContinuousTransformerWrapper(
                dim_in=decoder_configs.dimension,
                dim_out=decoder_configs.dimension,
                max_seq_len=self.encoder_max_seq_len,
                num_memory_tokens=decoder_configs.num_memory_tokens,
                attn_layers=Encoder(
                    dim=decoder_configs.dimension,
                    ff_mult=decoder_configs.ff_mult,
                    depth=decoder_configs.depth,
                    heads=decoder_configs.heads,
                    rotary_pos_emb=decoder_configs.rotary_pos_emb,
                    attn_flash=decoder_configs.attn_flash,
                    attn_kv_heads=decoder_configs.attn_kv_heads,
                    attn_qk_norm=decoder_configs.qk_norm,
                    pre_norm=decoder_configs.pre_norm,
                    residual_attn=decoder_configs.residual_attn,
                )
            )
            self.decoder_tail = nn.Linear(vq_dim, decoder_configs.dimension)
            classifier_cfg = getattr(decoder_configs, 'classifier_head', None)
            self.classifier_head_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False
            if self.classifier_head_enabled:
                num_classes = int(getattr(classifier_cfg, 'num_classes', 20))
                self.decoder_head = nn.Linear(decoder_configs.dimension, num_classes)
                self.decoder_output_dim = num_classes
            else:
                self.decoder_head = nn.Linear(decoder_configs.dimension, self.protein_encoder_dim)
                self.decoder_output_dim = self.protein_encoder_dim
        else:
            self.decoder_head = None
            self.decoder_output_dim = None

        # Next Token Prediction (NTP) head (optional)
        if getattr(configs.train_settings.losses, "next_token_prediction", False):
            self.ntp_enabled = configs.train_settings.losses.next_token_prediction.enabled
            self.ntp_depth = configs.train_settings.losses.next_token_prediction.blocks
        else:
            self.ntp_enabled = False

        self.codebook_size = int(getattr(self.configs.model.vqvae.vector_quantization, 'codebook_size', 0))

        if not self.decoder_only:
            if self.ntp_enabled:
                self.ntp_projector_head = nn.Linear(configs.model.vqvae.vector_quantization.dim, self.codebook_size)
                if self.ntp_depth > 0:
                    self.ntp_blocks = ContinuousTransformerWrapper(
                        dim_in=configs.model.vqvae.vector_quantization.dim,
                        dim_out=configs.model.vqvae.vector_quantization.dim,
                        max_seq_len=self.encoder_max_seq_len if not self.tik_tok_enabled else self.latent_token_count * self.residual_depth,
                        num_memory_tokens=configs.model.vqvae.encoder.num_memory_tokens,
                        attn_layers=Encoder(
                            dim=configs.model.vqvae.encoder.dimension,
                            ff_mult=encoder_configs.ff_mult,
                            depth=self.ntp_depth,
                            heads=encoder_configs.heads,
                            rotary_pos_emb=encoder_configs.rotary_pos_emb,
                            attn_flash=encoder_configs.attn_flash,
                            attn_kv_heads=encoder_configs.attn_kv_heads,
                            attn_qk_norm=encoder_configs.qk_norm,
                            pre_norm=encoder_configs.pre_norm,
                            residual_attn=encoder_configs.residual_attn,
                        )
                    )
            # Freeze components if requested
            if getattr(encoder_configs, 'freeze_parameters', False):
                self._freeze_module(self.encoder_tail)
                self._freeze_module(self.encoder)
                self._freeze_module(self.encoder_head)
                self._freeze_parameter(self.tik_tok_latent_tokens)

        if getattr(vq_cfg, 'freeze_parameters', False):
            self._freeze_module(self.vector_quantizer)
            if not self.decoder_only:
                self._freeze_module(self.tik_tok_padding_classifier)
            self._lock_vq_codebook()

        decoder_cfg = getattr(self.configs.model.vqvae, 'decoder', None)
        if decoder_cfg is not None and getattr(decoder_cfg, 'freeze_parameters', False):
            self._freeze_module(getattr(self, 'decoder_tail', None))
            self._freeze_module(getattr(self, 'decoder', None))
            self._freeze_module(getattr(self, 'decoder_head', None))
            self._freeze_parameter(self.decoder_mask_token)

    @staticmethod
    def _freeze_module(module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _freeze_parameter(param: Optional[torch.nn.Parameter]) -> None:
        if param is None:
            return
        param.requires_grad = False

    def _lock_vq_codebook(self) -> None:
        """Disable codebook gradient updates and freeze EMA refresh."""
        self.vq_frozen = True
        quantizer = self.vector_quantizer
        if quantizer is None:
            return
        quantizer.requires_grad_(False)
        for module in quantizer.modules():
            if hasattr(module, 'freeze_codebook'):
                module.freeze_codebook = True

    def create_causal_mask(self, seq_len, device):
        """
        Create a lower-triangular (causal) boolean attention mask of shape (seq_len, seq_len),
        where True indicates allowed attention (token i attends only to tokens j <= i).
        """
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()

    def _create_vector_quantizer(self):
        """
        Create the appropriate vector quantizer based on the VQ type.

        Returns:
            Vector quantizer module based on the specified type.
        """
        vq_config = self.configs.model.vqvae.vector_quantization
        vq_type = vq_config.type

        if vq_type == 'learnable':
            return self._create_learnable_vq(vq_config)
        elif vq_type == 'lfq':
            return self._create_lfq_vq(vq_config)
        else:
            raise ValueError(f"Unknown VQ type: {vq_type}. Supported types: ['learnable', 'lfq']")

    def _create_learnable_vq(self, vq_config):
        """Create a learnable vector quantizer (standard VQ-VAE)."""
        common_kwargs = dict(
            decay=vq_config.decay,
            commitment_weight=vq_config.commitment_weight,
            orthogonal_reg_weight=vq_config.orthogonal_reg_weight,
            orthogonal_reg_max_codes=vq_config.orthogonal_reg_max_codes,
            orthogonal_reg_active_codes_only=vq_config.orthogonal_reg_active_codes_only,
            rotation_trick=vq_config.rotation_trick,
            threshold_ema_dead_code=vq_config.threshold_ema_dead_code,
            kmeans_init=vq_config.kmeans_init,
            kmeans_iters=vq_config.kmeans_iters,
            stochastic_sample_codes=getattr(vq_config, 'stochastic_sample_codes', False),
            sample_codebook_temp=getattr(vq_config, 'sample_codebook_temp', 0.1),
        )

        if self.use_residual_vq:
            return ResidualVQ(
                dim=vq_config.dim,
                num_quantizers=self.residual_depth,
                codebook_size=vq_config.codebook_size,
                shared_codebook=True,
                **common_kwargs,
            )

        return VectorQuantize(
            dim=vq_config.dim,
            codebook_size=vq_config.codebook_size,
            **common_kwargs,
        )

    def _create_lfq_vq(self, vq_config):
        """Create a Lookup Free Quantizer (LFQ)."""
        return LFQ(
            codebook_size=vq_config.codebook_size,  # 14-bit
            num_codebooks=vq_config.num_codebooks,
            dim=vq_config.dim,  # addl. projections inserted automatically
            entropy_loss_weight=vq_config.entropy_loss_weight,
            commitment_loss_weight=vq_config.commitment_loss_weight,
            diversity_gamma=vq_config.diversity_gamma,
            spherical=vq_config.spherical,
            cosine_sim_project_in=vq_config.cosine_sim_project_in,
            frac_per_sample_entropy=vq_config.frac_per_sample_entropy  # default – keeps GPU util high on 32×DDP
        )

    def _apply_vector_quantization(
            self,
            quantizer_input: torch.Tensor,
            valid_mask: torch.Tensor,
            latent_mask_bool: Optional[torch.Tensor],
            active_tokens: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Pass encoder activations through the VQ layer, including TikTok/residual paths.

        Args:
            quantizer_input: Transformer encoder activations with optional TikTok latents
                appended at the tail when ``tik_tok`` is enabled.
            valid_mask: Standard key-padding mask for the original residues.
            latent_mask_bool: Boolean mask over TikTok latent slots; ``None`` when TikTok is
                disabled and the entire sequence should be quantized in one pass.
            active_tokens: Per-sample residue counts prior to compression, used to
                supervise the padding classifier.

        Returns:
            Tuple containing the decoder input (quantized embeddings), flattened codebook
            indices, VQ loss, updated latent mask, optional TikTok padding logits/targets,
            residual indices (when residual VQ is active), and latent-token counts.
        """

        tik_tok_padding_logits: Optional[torch.Tensor] = None
        tik_tok_padding_targets: Optional[torch.Tensor] = None
        unflatten_indices: Optional[torch.Tensor] = None
        latent_counts: Optional[torch.Tensor] = None

        valid_mask = valid_mask.to(torch.bool)

        if self.tik_tok_enabled and self.latent_token_count > 0:
            latent_tokens = quantizer_input[:, self.max_length:, :]
            if latent_mask_bool is None:
                raise RuntimeError("TikTok latent mask was not created.")
            latent_mask_bool = latent_mask_bool.to(torch.bool)
            decoder_input, indices, vq_loss = self.vector_quantizer(
                latent_tokens,
                mask=latent_mask_bool,
                freeze_codebook=self.vq_frozen,
            )
            unflatten_indices = indices
            if self.use_residual_vq:
                indices = self._flatten_residual_indices(indices)
                vq_loss = vq_loss.sum(dim=-1)

            if self.tik_tok_padding_classifier:
                tik_tok_padding_logits, tik_tok_padding_targets, latent_counts = self._compute_tik_tok_padding_output(
                    decoder_input,
                    latent_mask_bool,
                    active_tokens,
                )

        else:
            decoder_input, indices, vq_loss = self.vector_quantizer(
                quantizer_input,
                mask=valid_mask,
                freeze_codebook=self.vq_frozen,
            )

        return (
            decoder_input,
            indices,
            vq_loss,
            latent_mask_bool,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
            unflatten_indices,
            latent_counts,
        )

    def _prepare_ntp_inputs(
        self,
        decoder_input: torch.Tensor,
        latent_mask: Optional[torch.Tensor],
        unflatten_indices: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble inputs and masks for the NTP head across TikTok/residual modes.

        When TikTok compression is active, the NTP stream should ignore the
        original residues and operate only on latent tokens (flattened across
        residual quantizers if enabled). Otherwise, it consumes the full
        decoder input with the standard validity mask.

        Returns:
            Tuple of ``(ntp_input, ntp_mask)`` aligned with the logits the
            projector will produce.
        """

        if self.tik_tok_enabled and self.latent_token_count > 0 and latent_mask is not None:
            if self.use_residual_vq:
                if unflatten_indices is None:
                    raise RuntimeError("Residual VQ requires unflattened indices for NTP inputs.")
                ntp_valid_mask = self._flatten_residual_mask(latent_mask)
                ntp_input = self._build_residual_ntp_input(unflatten_indices)
            else:
                ntp_valid_mask = latent_mask
                ntp_input = decoder_input
        else:
            ntp_valid_mask = valid_mask
            ntp_input = decoder_input

        return ntp_input, ntp_valid_mask

    def _append_tik_tok_latents(self, x: torch.Tensor, valid_mask: torch.Tensor, encoder_mask_bool: torch.Tensor):
        """Append TikTok latent tokens to the encoder stream and update masks.

        This helper handles the boilerplate for TikTok-mode: it concatenates the
        learnable latent token bank to each sequence, derives how many of those
        latents should remain active based on the number of valid input tokens,
        and merges the resulting latent mask into the encoder's key padding mask.

        Args:
            x: Tensor of shape ``(B, L, D)`` containing the encoder activations
               directly after ``encoder_tail`` (before TikTok augmentation).
            valid_mask: Bool tensor ``(B, L)`` where True marks residues that are
               both unmasked and non-NaN; used to compute the latent keep count.
            encoder_mask_bool: Bool tensor ``(B, L)`` representing the current
               key-padding mask passed to the transformer encoder; typically this
               is identical to ``valid_mask`` prior to TikTok augmentation.

        Returns:
            tuple:
                - augmented activations ``(B, L + latent_count, D)``
                - latent activation mask ``(B, latent_count)`` with True for
                  latents that should remain active
                - updated key-padding mask ``(B, L + latent_count)`` ready to be
                  forwarded to the encoder blocks
                - active token counts ``(B,)`` prior to compression

        Raises:
            RuntimeError: If the TikTok latent parameter tensor has not been
                initialised (should not happen when TikTok is enabled).
        """
        batch_size = x.size(0)
        if self.tik_tok_latent_tokens is None:
            raise RuntimeError("TikTok latent tokens are not initialized.")

        latent_tokens = self.tik_tok_latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([x, latent_tokens], dim=1)

        active_tokens = valid_mask.to(torch.int64).sum(dim=1)
        latent_keep = (active_tokens + self.tik_tok_compression_factor - 1) // self.tik_tok_compression_factor
        latent_keep = latent_keep.clamp(min=0, max=self.latent_token_count)

        latent_positions = torch.arange(
            self.latent_token_count,
            device=x.device,
            dtype=latent_keep.dtype
        ).unsqueeze(0)
        latent_mask_bool = latent_positions < latent_keep.unsqueeze(1)
        encoder_mask_bool = torch.cat([encoder_mask_bool, latent_mask_bool], dim=1)

        return x, latent_mask_bool, encoder_mask_bool, active_tokens

    def _compute_tik_tok_padding_output(
            self,
            latent_tokens: torch.Tensor,
            latent_mask_bool: torch.Tensor,
            active_tokens: Optional[torch.Tensor],
    ):
        """Infer original residue length from TikTok latents.

        Each latent token represents ``compression_factor`` residues (minus any
        padding). The last active latent therefore carries information about the
        padding remainder. This helper selects that latent, runs it through the
        TikTok padding classifier, and returns the logits over remainder classes
        along with optional supervision targets and latent counts.

        Args:
            latent_tokens: Quantized TikTok latent embeddings of shape
                ``(B, latent_count, D)``.
            latent_mask_bool: Boolean mask ``(B, latent_count)`` indicating which
                latent slots are valid.
            active_tokens: Optional tensor ``(B,)`` giving the true residue
                counts before compression. When provided, supervision targets are
                computed as ``active_tokens % compression_factor``. When absent
                (decoder-only inference), targets are ``None``.

        Returns:
            ``(logits, targets, latent_counts)`` where ``logits`` has shape
            ``(B, compression_factor)``, ``targets`` is either a remainder tensor
            or ``None``, and ``latent_counts`` records the number of active
            latent tokens per sample.
        """
        mask = latent_mask_bool.to(latent_tokens.dtype)
        active_counts = mask.sum(dim=1).clamp(min=1).to(torch.long)
        last_indices = (active_counts - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, latent_tokens.size(-1))
        last_latent = latent_tokens.gather(1, last_indices).squeeze(1)

        logits = self.tik_tok_padding_classifier(last_latent)

        targets = None
        if active_tokens is not None:
            targets = active_tokens.to(torch.long) % self.tik_tok_compression_factor

        return logits, targets, active_counts

    def _flatten_residual_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Flatten residual codes and push padding to the tail.

        Args:
            indices: Tensor ``(B, L, D)`` produced by :class:`ResidualVQ`, where
                ``B`` is the batch size, ``L`` the latent token count, and ``D``
                the residual depth. ``-1`` marks padded tokens. TikTok ensures
                masking is uniform across depths, so a padded position is padded
                for every quantizer.

        Returns:
            Tensor ``(B, L * D)`` with valid entries reordered depth-by-depth at
            the head and all ``-1`` padding collected at the end. The same
            scatter pattern is reused for flattening masks and embeddings so all
            downstream tensors stay aligned.

        Example:
            ``[[[0, 10], [1, 11], [-1, -1]]]`` → ``[[0, 1, 10, 11, -1, -1]]``
        """

        if indices.dim() != 3:
            return indices

        batch, length, depth = indices.shape
        flat_len = length * depth

        per_level = indices.permute(0, 2, 1).contiguous().view(batch, flat_len)
        level_mask = (indices[..., 0] >= 0).unsqueeze(1).expand(batch, depth, length)
        mask_flat = level_mask.reshape(batch, flat_len)

        mask_long = mask_flat.long()
        valid_pos = mask_long.cumsum(dim=1) - 1
        invalid_pos = (~mask_flat).long().cumsum(dim=1) - 1
        valid_count = mask_long.sum(dim=1, keepdim=True)

        target_pos = torch.where(mask_flat, valid_pos, valid_count + invalid_pos)

        reordered = indices.new_full(per_level.shape, -1)
        reordered.scatter_(1, target_pos, per_level)
        return reordered

    def _flatten_residual_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Flatten residual masks to match :meth:`_flatten_residual_indices`.

        Args:
            mask: Bool tensor ``(B, L)`` with ``True`` marking valid tokens per
                sequence position (shared across depths).

        Returns:
            Bool tensor ``(B, L * D)`` whose ``True`` values occupy the same
            leading slots as the flattened indices, with ``False`` values filling
            the trailing padding region.
        """

        if mask.dim() != 2:
            return mask

        batch, length = mask.shape
        depth = self.residual_depth
        flat_len = length * depth
        valid_count = mask.sum(dim=1, keepdim=True) * depth
        positions = torch.arange(flat_len, device=mask.device).unsqueeze(0)
        return positions < valid_count

    def _build_residual_ntp_input(self, residual_indices: torch.Tensor) -> torch.Tensor:
        """Return flattened embeddings aligned with residual-flattened indices.

        Args:
            residual_indices: Tensor ``(B, L, D)`` of residual code indices.

        Returns:
            Tensor ``(B, L * D, dim)`` containing the per-depth embeddings in the
            exact order produced by :meth:`_flatten_residual_indices`. Valid
            vectors occupy the front, while padded slots are zero-filled at the
            tail. This ensures the NTP logits, labels, and mask reference the same
            sequence positions element-wise.
        """

        if residual_indices.dim() != 3:
            raise ValueError("Residual indices must have shape (B, L, D).")

        codes_per_level = self.vector_quantizer.get_codes_from_indices(residual_indices)
        codes_per_level = codes_per_level.permute(1, 0, 2, 3).contiguous()
        batch, depth, length, dim = codes_per_level.shape
        embeddings = codes_per_level.reshape(batch, depth * length, dim)

        mask = (residual_indices[..., 0] >= 0)
        expanded_mask = mask.unsqueeze(1).expand(batch, depth, length)
        mask_flat = expanded_mask.reshape(batch, depth * length)

        mask_long = mask_flat.long()
        valid_pos = mask_long.cumsum(dim=1) - 1
        invalid_pos = (~mask_flat).long().cumsum(dim=1) - 1
        valid_count = mask_long.sum(dim=1, keepdim=True)

        target_pos = torch.where(mask_flat, valid_pos, valid_count + invalid_pos)

        embeddings = embeddings * mask_flat.unsqueeze(-1)
        output = embeddings.new_zeros(embeddings.shape)
        scatter_indices = target_pos.unsqueeze(-1).expand_as(embeddings)
        output.scatter_(1, scatter_indices, embeddings)
        return output

    def _build_decoder_tik_tok_stream(
        self,
        latent_tokens: torch.Tensor,
        original_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the decoder input sequence when TikTok is enabled.

        Args:
            latent_tokens: Tensor ``(B, latent_count, D)`` output by the VQ layer.
            original_mask: Bool tensor ``(B, max_length)`` indicating valid
                residue positions prior to TikTok augmentation.

        Returns:
            tuple containing:
                - Concatenated decoder input ``(B, max_length + latent_count, D)``
                - Updated key padding mask of shape ``(B, max_length + latent_count)``

        Raises:
            RuntimeError: If the decoder mask token has not been created.
        """
        if self.decoder_mask_token is None:
            raise RuntimeError("TikTok decoder mask token is not initialized.")

        batch_size = latent_tokens.size(0)
        latent_token_length = latent_tokens.size(1)

        mask_token = self.decoder_mask_token.unsqueeze(0).unsqueeze(0)
        mask_tokens = mask_token.expand(batch_size, self.max_length, -1)
        mask_tokens = mask_tokens * original_mask.to(mask_tokens.dtype).unsqueeze(-1)

        active_tokens = original_mask.to(torch.int64).sum(dim=1)
        latent_keep = (active_tokens + self.tik_tok_compression_factor - 1) // self.tik_tok_compression_factor
        latent_keep = latent_keep.clamp(min=0, max=latent_token_length)

        latent_positions = torch.arange(
            latent_token_length,
            device=latent_tokens.device,
            dtype=latent_keep.dtype
        ).unsqueeze(0)
        latent_mask_bool = latent_positions < latent_keep.unsqueeze(1)
        latent_tokens = latent_tokens * latent_mask_bool.to(latent_tokens.dtype).unsqueeze(-1)

        decoder_mask_bool = torch.cat([original_mask, latent_mask_bool], dim=1)
        decoder_input = torch.cat([mask_tokens, latent_tokens], dim=1)

        return decoder_input, decoder_mask_bool

    def forward(self, x, **kwargs):
        pad_mask = kwargs['mask'].to(torch.bool)

        encoder_input = self.encoder_tail(x)
        encoder_mask = pad_mask
        vq_loss = torch.tensor([0.], device=x.device)
        reconstructed_embeddings: Optional[torch.Tensor] = None
        decoder_mask = pad_mask
        latent_mask_bool: Optional[torch.Tensor] = None
        active_tokens: Optional[torch.Tensor] = None
        sequence_lengths: Optional[torch.Tensor] = None
        ntp_logits: Optional[torch.Tensor] = None
        ntp_mask: Optional[torch.Tensor] = None

        if not self.decoder_only:
            if self.tik_tok_enabled and self.latent_token_count > 0:
                encoder_input, latent_mask_bool, encoder_mask, active_tokens = self._append_tik_tok_latents(
                    encoder_input,
                    valid_mask=pad_mask,
                    encoder_mask_bool=encoder_mask,
                )

            encoder_attn_mask = None
            if self.encoder_causal:
                seq_len = encoder_input.size(1)
                encoder_attn_mask = self.create_causal_mask(seq_len, device=encoder_input.device)
            encoder_embeddings = self.encoder(encoder_input, mask=encoder_mask, attn_mask=encoder_attn_mask)

            quantizer_input = self.encoder_head(encoder_embeddings)

            # Initialize indices with proper shape when VQ is disabled
            batch_size, seq_len = x.shape[:2]
            indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)

            tik_tok_padding_logits: Optional[torch.Tensor] = None
            tik_tok_padding_targets: Optional[torch.Tensor] = None

            if self.configs.model.vqvae.vector_quantization.enabled:
                (
                    decoder_input,
                    indices,
                    vq_loss,
                    latent_mask_bool,
                    tik_tok_padding_logits,
                    tik_tok_padding_targets,
                    unflatten_indices,
                    latent_counts,
                ) = self._apply_vector_quantization(quantizer_input, pad_mask, latent_mask_bool, active_tokens)

                if self.ntp_enabled:
                    ntp_input, ntp_mask = self._prepare_ntp_inputs(decoder_input, latent_mask_bool, unflatten_indices,
                                                                   pad_mask)
                    if self.ntp_depth > 0 and hasattr(self, 'ntp_blocks'):
                        ntp_attn_mask = self.create_causal_mask(ntp_input.size(1), device=ntp_input.device)
                        ntp_input = self.ntp_blocks(ntp_input, mask=ntp_mask, attn_mask=ntp_attn_mask)
                    ntp_logits = self.ntp_projector_head(ntp_input)

                if self.tik_tok_enabled and self.tik_tok_padding_classifier is not None:
                    predicted_remainder = tik_tok_padding_logits.argmax(dim=-1)
                    sequence_lengths = latent_counts.to(torch.long) * self.tik_tok_compression_factor + predicted_remainder
                    sequence_lengths = torch.min(sequence_lengths, torch.full_like(sequence_lengths, self.max_length))
                elif self.tik_tok_enabled and self.tik_tok_padding_classifier is None:
                    sequence_lengths = latent_mask_bool.sum(dim=-1)

            else:
                decoder_input = quantizer_input
                if self.tik_tok_enabled:
                    raise NotImplementedError("TikTok mode requires vector quantization to be enabled.")

            reconstructed_embeddings = decoder_input
            decoder_stream = decoder_input

            if self.tik_tok_enabled and self.latent_token_count > 0:
                decoder_stream, decoder_mask = self._build_decoder_tik_tok_stream(decoder_input, decoder_mask)

        else:
            indices = x
            (
                decoder_stream,
                latent_mask_bool,
                valid,
                ntp_valid_mask,
                sequence_lengths,
                tik_tok_padding_logits,
                tik_tok_padding_targets,
            ) = self._decode_from_indices(indices, encoder_mask, encoder_mask.device)


        if self.configs.model.vqvae.decoder.enabled:
            decoder_embeddings = self.decoder_tail(decoder_stream)
            decoder_attn_mask = None
            if self.decoder_causal:
                seq_len = decoder_embeddings.size(1)
                decoder_attn_mask = self.create_causal_mask(seq_len, device=decoder_embeddings.device)
            decoder_output = self.decoder(decoder_embeddings, mask=decoder_mask, attn_mask=decoder_attn_mask)
            if self.tik_tok_enabled and self.latent_token_count > 0:
                decoder_output = decoder_output[:, :self.max_length, :]
            decoder_output = self.decoder_head(decoder_output)
        else:
            decoder_output = decoder_stream
            if self.tik_tok_enabled and self.latent_token_count > 0:
                decoder_output = decoder_output[:, :self.max_length, :]

        return (
            reconstructed_embeddings,
            indices,
            vq_loss,
            decoder_output,
            ntp_logits,
            ntp_mask,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
            sequence_lengths,
        )

    def _decode_from_indices(
        self,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Rebuild decoder inputs in decoder-only mode and adjust masks for TikTok.

        Besides the decoded embeddings, this returns the latent mask derived
        from ``-1`` padding, the possibly-updated validity mask, the NTP mask,
        optional TikTok padding logits/targets, and inferred sequence lengths
        when TikTok compression is enabled.
        """

        latent_mask_bool = (indices != -1)
        decoder_input = self.vector_quantizer.get_output_from_indices(indices)

        tik_tok_padding_logits: Optional[torch.Tensor] = None
        tik_tok_padding_targets: Optional[torch.Tensor] = None
        sequence_lengths: Optional[torch.Tensor] = None
        ntp_valid_mask = valid_mask
        updated_valid = valid_mask

        if self.tik_tok_enabled:
            tik_tok_padding_logits, tik_tok_padding_targets, latent_counts = self._compute_tik_tok_padding_output(
                decoder_input,
                latent_mask_bool,
                None,
            )
            predicted_remainder = tik_tok_padding_logits.argmax(dim=-1)
            sequence_lengths = latent_counts.to(torch.long) * self.tik_tok_compression_factor + predicted_remainder

            token_positions = torch.arange(self.max_length, device=mask_device).unsqueeze(0)
            updated_valid = token_positions < sequence_lengths.unsqueeze(1)
            ntp_valid_mask = updated_valid

        return (
            decoder_input,
            latent_mask_bool,
            updated_valid,
            ntp_valid_mask,
            sequence_lengths,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
        )
