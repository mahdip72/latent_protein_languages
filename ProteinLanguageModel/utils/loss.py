import torch.nn.functional as F


def calculate_loss(model_output, _configs=None):
    """
    Compute the training loss from model outputs.

    Returns:
        dict: {
            "scaled_loss": cross-entropy loss used for backward,
            "unscaled_loss": same scalar for logging.
        }
    """
    logits = model_output["decoder_output"]
    targets = model_output["target_ids"]

    # Pad token id is 0 and is excluded from loss/perplexity.
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0,
    )

    return {
        "scaled_loss": loss,
        "unscaled_loss": loss,
    }
