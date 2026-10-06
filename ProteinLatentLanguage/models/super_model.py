import re
import torch
from torch import nn
from transformers import EsmModel
from peft import prepare_model_for_kbit_training
from esm.models.esmc import ESMC
from huggingface_hub import hf_hub_download
from models.vqvae import VQVAE


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


class ProteinEncoder(nn.Module):
    """
    ProteinEncoder wraps a pretrained protein language model (ESM or ESMC) to extract per-residue embeddings.

    Args:
        logging: Logger for informational messages.
        configs: Configuration object with model and fine-tuning settings.
        model_name: Pretrained model identifier.
        model_type: Type of encoder to use ('esm_v2' or 'esmc').

    The encoder outputs a tensor of shape (batch, seq_len, embedding_dim).
    """

    def __init__(self, logging, configs, model_name='facebook/esm2_t33_650M_UR50D', model_type='esm_v2'):
        super().__init__()

        self.model_type = model_type

        if model_type == 'esm_v2':
            if configs.model.protein_encoder.quantization_4_bit:
                from transformers import BitsAndBytesConfig
                logging.info('load quantized 4-bit weights')
                # QLoRa fine-tuning:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = EsmModel.from_pretrained(model_name,
                                                      device_map="auto",
                                                      quantization_config=quantization_config)
                # self.model = prepare_model_for_kbit_training(self.model,
                #                                              use_gradient_checkpointing=True)
            else:
                self.model = EsmModel.from_pretrained(model_name)

            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.pooler.parameters():
                param.requires_grad = False

            for param in self.model.contact_head.parameters():
                param.requires_grad = False

            self.protein_encoder_dim = self.model.embeddings.word_embeddings.embedding_dim

            if configs.model.protein_encoder.protrek_checkpoint:
                # Check the esm2 model name is facebook/esm2_t33_650M_UR50D
                if model_name != 'facebook/esm2_t33_650M_UR50D':
                    raise ValueError(
                        f'ProTrek checkpoint is only supported for esm2 model: facebook/esm2_t33_650M_UR50D')

                self.model.embeddings.position_embeddings = None

                ckpt_path = hf_hub_download(
                    repo_id="westlake-repl/ProTrek_650M_UniRef50",  # HF repo
                    filename="ProTrek_650M_UniRef50.pt",  # blob in repo :contentReference[oaicite:1]{index=1}
                    repo_type="model"  # tell HF it's a model
                )

                raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = raw.get("model", raw)  # some snapshots wrap tensors in "model"

                # 3. strip the prefix **completely**
                clean_state = {
                    re.sub(r"^1\.model\.esm\.", "", k): v
                    for k, v in state.items()
                    if k.startswith("1.model.esm.")  # keep only the sequence branch
                }
                del state, raw

                missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
                logging.info(f"ignored extras: {unexpected},  still-missing: {missing}")
                logging.info(f'ProTrek checkpoint loaded for esm2 model.')

        elif model_type == 'esmc':
            self.model = ESMC.from_pretrained(model_name)

            if configs.model.protein_encoder.quantization_4_bit:
                RuntimeError('Quantization is not supported for ESMC model')

            if not configs.model.protein_encoder.quantization_4_bit and configs.model.protein_encoder.fine_tune.enable:
                # Freeze all layers
                for param in self.model.parameters():
                    param.requires_grad = False

                # Allow the parameters of the last transformer block to be updated during fine-tuning
                for block in self.model.transformer.blocks[
                    -configs.model.protein_encoder.fine_tune.last_layers_trainable:]:
                    for param in block.parameters():
                        param.requires_grad = True

            else:
                # Freeze all layers
                for param in self.model.parameters():
                    param.requires_grad = False

            self.protein_encoder_dim = self.model.embed.embedding_dim

            if configs.model.protein_encoder.protrek_checkpoint:
                # not supported for ESMC model
                raise RuntimeError('ProTrek checkpoint is not supported for ESMC model')
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    def forward(self, x):
        if not self.model_type == 'esmc':
            features = self.model(input_ids=x['input_ids'],
                                  attention_mask=x['attention_mask'])
            return features.last_hidden_state
        else:
            features = self.model(sequence_tokens=x['input_ids'])
            return features.embeddings


class SuperModel(nn.Module):
    """
    SuperModel combines a ProteinEncoder and a VQVAE into an end-to-end model.

    Args:
        protein_encoder (ProteinEncoder): Module to extract protein embeddings.
        vqvae_model (VQVAE): Vector-quantized variational autoencoder module.
        configs: Configuration object containing model settings.
        decoder_only (bool): If True, only use the decoder part of VQVAE only.

    Forward:
        batch (dict): Input batch with keys 'input_ids', 'attention_mask', and 'original_sequence'.
        **kwargs: Additional arguments such as 'mask'.

    Returns:
        dict: Outputs including 'decoder_output', 'reconstructed_embeddings', 'indices',
              'commitment_loss', 'encoder_embeddings', and 'original_sequence'.
    """

    def __init__(self, protein_encoder, vqvae_model, configs, decoder_only=False):  # Changed decoder to vqvae_model
        super().__init__()
        self.protein_encoder = protein_encoder
        self.decoder_only = decoder_only
        self.vqvae = vqvae_model  # Changed self.decoder to self.vqvae
        self.configs = configs

    def prepare_decoder(self):
        pass

    def forward(self, batch, **kwargs):
        """
        Forward pass through protein encoder and VQ-VAE model.

        Args:
            batch (dict): Input batch containing 'input_ids', 'attention_mask', and 'original_sequence'.

        **kwargs: Additional keyword arguments forwarded to the VQVAE. Common keys include:
            - `return_encoder_only` (bool, optional): If True, skip VQ-VAE processing and return only encoder outputs; otherwise the full VQ-VAE pipeline is executed.

        Returns:
            dict: Modular output dictionary with the following components:
                Always present:
                - 'encoder_embeddings': Protein encoder outputs of shape (B, L, encoder_dim)
                - 'original_sequence': Original protein sequences
                - 'mask': Boolean attention mask of shape (B, L)

                Present when return_encoder_only=False (default):
                - 'decoder_output': Decoder reconstructed embeddings of shape (B, L, encoder_dim)
                - 'reconstructed_embeddings': VQ-quantized embeddings of shape (B, L, vq_dim)
                - 'indices': VQ codebook indices of shape (B, L)
                - 'vq_loss': Vector quantization loss (scalar tensor)
        """
        output_dict = {}

        if not self.decoder_only:
            # Get protein encoder outputs
            protein_encoder_out = self.protein_encoder(batch)

            # Apply mask to encoder embeddings
            mask = batch['attention_mask'].unsqueeze(-1).float()
            protein_encoder_out = protein_encoder_out * mask

            # Always included components
            output_dict["encoder_embeddings"] = protein_encoder_out  # (B, L, encoder_dim)
            output_dict["original_sequence"] = batch["original_sequence"]  # Original sequences
            output_dict["mask"] = batch['attention_mask'].bool()  # (B, L) boolean mask

            # Check if we should skip VQ-VAE processing
            if kwargs.get('return_encoder_only', False):
                return output_dict

        else:
            # In decoder_only mode, we skip the protein encoder and create a dummy encoder output
            protein_encoder_out = None
            mask = batch['attention_mask'].unsqueeze(-1).float()

        # VQ-VAE processing
        (
            reconstructed_embeddings,
            indices,
            vq_loss,
            decoder_output,
            ntp_logits,
            ntp_mask,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
            sequence_lengths,
        ) = self.vqvae(protein_encoder_out, mask=batch['attention_mask'].bool(), **kwargs)

        # Apply mask to decoder output
        decoder_output = decoder_output * mask

        # VQ-VAE specific outputs
        output_dict["decoder_output"] = decoder_output  # (B, L, encoder_dim)
        output_dict["reconstructed_embeddings"] = reconstructed_embeddings  # (B, L, vq_dim)
        output_dict["indices"] = indices  # (B, L)
        output_dict["vq_loss"] = vq_loss  # scalar tensor
        output_dict["ntp_logits"] = ntp_logits  # (B, L, K) or None
        output_dict["ntp_mask"] = ntp_mask
        output_dict["tik_tok_padding_logits"] = tik_tok_padding_logits
        output_dict["tik_tok_padding_targets"] = tik_tok_padding_targets
        output_dict["sequence_lengths"] = sequence_lengths
        if 'classification_labels' in batch:
            output_dict["classification_targets"] = batch['classification_labels'].to(protein_encoder_out.device)

        return output_dict


def prepare_models(configs, logging, inference=False, **kwargs):
    """
    Build and return the end-to-end model pipeline: a ProteinEncoder, a VQVAE, and a combined SuperModel.

    Args:
        configs: Configuration object containing model and training settings.
        logging: Logger for informational output.
        inference (bool): If True, freeze model weights for inference.

    Returns:
        SuperModel: Combined model wrapping encoder and VQVAE ready for training or inference.
    """
    # Prepare the protein encoder.
    protein_encoder = ProteinEncoder(
        model_name=configs.model.protein_encoder.model_name,
        model_type=configs.model.protein_encoder.model_type,
        logging=logging,
        configs=configs
    )

    print_trainable_parameters(protein_encoder, logging, 'protein encoder')

    # Prepare VQVAE model
    vqvae_model = VQVAE(configs, logging, protein_encoder.protein_encoder_dim, decoder_only=kwargs.get("decoder_only", False))

    print_trainable_parameters(vqvae_model, logging, 'vqvae_model')

    # Build a supermodel.
    final_model = SuperModel(protein_encoder, vqvae_model, configs, decoder_only=kwargs.get("decoder_only", False))  # Pass vqvae_model

    print_trainable_parameters(final_model, logging, 'supermodel')

    if inference:
        # freeze all parameters
        for param in final_model.parameters():
            param.requires_grad = False
        logging.info(f'freeze all parameters for inference')

    return final_model
