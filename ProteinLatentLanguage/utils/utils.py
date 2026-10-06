from box import Box
import yaml
import shutil
from pathlib import Path
import datetime
import os


def load_configs(config, inference=False, config_path=None):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.
            inference (bool): A boolean flag to indicate if the configuration is for inference.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Load and merge VQ config if enabled
    if not inference and tree_config.model.vqvae.vector_quantization.type in ['learnable', 'lfq']:

        vq_type = tree_config.model.vqvae.vector_quantization.type
        vq_config_path = os.path.join(os.path.dirname(config_path), f"{vq_type}.yaml") if config_path else f"configs/{vq_type}.yaml"

        if os.path.exists(vq_config_path):
            with open(vq_config_path) as f:
                vq_config = yaml.safe_load(f)
            # Merge VQ config into the main config, preserving enabled and type
            vq_config['enabled'] = tree_config.model.vqvae.vector_quantization.enabled
            vq_config['type'] = tree_config.model.vqvae.vector_quantization.type
            tree_config.model.vqvae.vector_quantization.update(vq_config)
        else:
            raise FileNotFoundError(f"VQ config file {vq_config_path} not found.")

    if not inference:
        # Convert the necessary values to floats.
        tree_config.optimizer.lr = float(tree_config.optimizer.lr)
        tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
        tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
        tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def validate_loss_configuration(configs):
    """
    Ensure decoder classifier usage does not conflict with reconstruction losses.

    Raises:
        ValueError: When classifier head / loss toggles violate mutual exclusion rules.
    """
    classifier_cfg = getattr(configs.model.vqvae.decoder, 'classifier_head', None)
    classifier_head_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False

    classification_cfg = getattr(configs.train_settings.losses, 'classification', None)
    classification_loss_enabled = bool(getattr(classification_cfg, 'enabled', False)) if classification_cfg is not None else False

    if classifier_head_enabled and not classification_loss_enabled:
        raise ValueError(
            "Decoder classifier head is enabled, but train_settings.losses.classification.enabled is False."
        )
    if classification_loss_enabled and not classifier_head_enabled:
        raise ValueError(
            "train_settings.losses.classification.enabled is True, but the decoder classifier head is disabled."
        )
    if classifier_head_enabled:
        reconstruction_losses = [
            ('train_settings.losses.mse.enabled', configs.train_settings.losses.mse.enabled),
            ('train_settings.losses.cosine_similarity.enabled', configs.train_settings.losses.cosine_similarity.enabled),
            ('train_settings.losses.cross_entropy.enabled', configs.train_settings.losses.cross_entropy.enabled),
        ]
        conflicting = [name for name, enabled in reconstruction_losses if enabled]
        if conflicting:
            raise ValueError(
                "When the decoder classifier head is enabled, the following reconstruction losses must be disabled: "
                + ", ".join(conflicting)
            )


def prepare_saving_dir(configs, config_file_path):
    """
    Prepares a dedicated directory structure for saving training results and artifacts.

    This function performs the following actions:
    1. Generates a unique run identifier (run_id) based on the current timestamp (YYYY-MM-DD__HH-MM-SS).
    2. Creates a main results directory using the `result_path` from the `configs` object and the generated `run_id`.
       The structure will be: `configs.result_path`/`run_id`/
    3. Inside the main results directory, it creates a subdirectory named 'checkpoints' for storing model checkpoints.
       The structure will be: `configs.result_path`/`run_id`/checkpoints/
    4. Copies the provided `config_file_path` (the configuration file used for the run) into the main results directory for record-keeping.
    5. If VQ is enabled, copies the corresponding VQ config file (e.g., learnable.yaml, lfq.yaml) to the results directory.
    6. Saves a merged config file containing all parameters including VQ settings for complete reproducibility.

    Args:
        configs: A python box object containing the configuration options.
        config_file_path: Path to the configuration file.

    Returns:
        tuple[str, str]: A tuple containing the path to the results directory and the path to the checkpoints directory.
    """

    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Copy VQ config if enabled
    if configs.model.vqvae.vector_quantization.type in ['learnable', 'lfq']:
        vq_type = configs.model.vqvae.vector_quantization.type
        src_vq_path = os.path.join(os.path.dirname(config_file_path), f"{vq_type}.yaml")
        if os.path.exists(src_vq_path):
            dest_vq_path = os.path.join(result_path, f"{vq_type}.yaml")
            shutil.copy(src_vq_path, dest_vq_path)
        else:
            raise FileNotFoundError(f"VQ config file {src_vq_path} not found.")

    # Return the path to the result directory.
    return result_path, checkpoint_path
