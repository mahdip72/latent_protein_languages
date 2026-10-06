import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging as log
from accelerate.logging import get_logger
from io import StringIO
from pathlib import Path


def test_gpu_cuda(logging, accelerator=None):
    logging.info(f'Testing gpu and cuda:')
    logging.info(f'\tcuda is available: {torch.cuda.is_available()}')
    logging.info(f'\tdevice count on one node: {torch.cuda.device_count()}')
    logging.info(f'\tcurrent device: {torch.cuda.current_device()}')
    logging.info(f'\tdevice: {torch.cuda.device(0)}')
    logging.info(f'\tdevice name: {torch.cuda.get_device_name()}')
    if accelerator is not None:
        logging.info(
            f'\tnumber of nodes: {accelerator.state.num_processes // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1}')
        logging.info(f'\ttotal number of gpus: {accelerator.state.num_processes}')


def get_logging(result_path, configs):
    # logger = log.getLogger(result_path)
    # logger.setLevel(log.INFO)

    logger = get_logger(__name__, log_level="INFO")
    logger.logger.propagate = False
    log_file_path = os.path.join(result_path, "logs.txt")

    # Create a file handler (logs will be saved to 'training.log')
    file_handler = log.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(log.INFO)

    # Define a log message format
    formatter = log.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Attach the file handler to the underlying logger
    logger.logger.addHandler(file_handler)

    import sys
    # Stream handler (prints logs to the console)
    stream_handler = log.StreamHandler(sys.stdout)
    stream_handler.setLevel(log.INFO)
    stream_formatter = log.Formatter("%(asctime)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.logger.addHandler(stream_handler)

    return logger


def log_mixed_precision(logger, configs, accelerator=None):
    """Log the effective mixed precision mode to the run log (single line).

    `accelerate.logging.get_logger` already handles main-process-only logging, so
    callers can invoke this unconditionally.
    """
    if logger is None or configs is None or accelerator is None:
        return

    requested = getattr(getattr(configs, "train_settings", None), "mixed_precision", None)
    active = getattr(getattr(accelerator, "state", None), "mixed_precision", None)

    fp8_backend = None
    if bool(getattr(accelerator, "fp8_enabled", False)):
        backend = getattr(accelerator, "fp8_backend", None)
        fp8_backend = getattr(backend, "value", backend)

    msg = f"Mixed precision: requested={requested} active={active}"
    if fp8_backend:
        msg += f" fp8_backend={fp8_backend}"

    logger.info(msg)


def get_dummy_logger():
    # Create a logger object
    logger = log.getLogger('dummy')
    logger.setLevel(log.INFO)

    # Create a string buffer to hold the logs
    log_buffer = StringIO()

    # Create a stream handler that writes to the string buffer
    handler = log.StreamHandler(log_buffer)
    formatter = log.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optionally disable propagation to prevent logging on the parent logger
    logger.propagate = False

    # Return both logger and buffer so you can inspect logs as needed
    return logger, log_buffer


def log_train_step_metrics(writer, global_step, lr, net=None, configs=None,
                           accelerator=None, compute_grad_norm=False, tokens_seen_total=None,
                           step_unscaled_loss=None, step_scaled_loss=None):
    """Log per-step training metrics to TensorBoard.
    
    This function logs the learning rate at each training step and optionally computes
    and logs the gradient norm based on the specified frequency in configs. It automatically
    handles all necessary checks (main process, tensorboard enabled) and returns early if
    conditions are not met.
    
    The gradient norm is computed as the L2 norm of all parameter gradients and is only
    logged at intervals specified by `configs.train_settings.gradient_norm_logging_freq`.
    When `tokens_seen_total` is provided, the running total number of tokens processed
    so far is logged at the same frequency, allowing monitoring of the effective data
    volume the model has consumed.
    
    Args:
        writer: TensorBoard SummaryWriter instance
        global_step (int): Current global training step
        lr (float): Current learning rate
        net (torch.nn.Module, optional): Model for gradient norm computation
        configs: Configuration object with tensorboard_log and gradient_norm_logging_freq
        accelerator: Accelerator instance for distributed training checks
        compute_grad_norm (bool): Whether to compute and log gradient norm. Defaults to False.
        tokens_seen_total (float, optional): Cumulative tokens processed up to this step.
        step_unscaled_loss (float, optional): Mean loss value at this step.
        step_scaled_loss (float, optional): Mean scaled loss value at this step.
    
    Returns:
        None. Logs are written directly to TensorBoard if conditions are met.
    """

    decay_cfg = getattr(getattr(configs, 'optimizer', None), 'decay', None)
    warmup_steps = int(getattr(decay_cfg, 'warmup', 0)) if decay_cfg is not None else 0
    if global_step < warmup_steps:
        return

    if accelerator.is_main_process and configs.tensorboard_log:
        if global_step % configs.train_settings.gradient_norm_logging_freq == 0:
            writer.add_scalar('lr', lr, global_step)

            if compute_grad_norm and net is not None:
                params_with_grad = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
                if params_with_grad:
                    per_param_norms = []
                    for p in params_with_grad:
                        grad = p.grad.detach()
                        # FSDP2 gradients may be DTensor; use the local shard for logging
                        # to avoid triggering rank-collective ops from rank0-only logging.
                        if hasattr(grad, "to_local"):
                            grad = grad.to_local()
                        per_param_norms.append(torch.norm(grad.float(), 2))

                    grad_norm = torch.norm(torch.stack(per_param_norms), 2)
                    writer.add_scalar('gradient norm/total_amp_scaled', grad_norm.item(), global_step)

            writer.add_scalar('step_metrics/loss', step_scaled_loss, global_step)
            writer.add_scalar('step_metrics/unscaled_loss', step_unscaled_loss, global_step)
            writer.add_scalar('step_metrics/loss_vs_tokens', step_unscaled_loss, tokens_seen_total)


def log_train_epoch_metrics(writer, epoch, avg_scaled_loss, avg_unscaled_loss, perplexity,
                            accelerator=None, configs=None,
                            tokens_seen_total=None):
    """Log end-of-epoch training metrics to TensorBoard.
    
    This function logs aggregated training metrics at the end of each epoch, including
    scaled loss, unscaled loss, and perplexity. It automatically handles all necessary
    checks (main process, tensorboard enabled) and returns early if conditions are not met.
    
    Metrics logged:
    - 'loss/total': Averaged scaled loss for the epoch
    - 'unscaled_loss/total': Averaged unscaled loss for the epoch
    - 'perplexity': Perplexity metric (exp of average negative log-likelihood)
    - 'tokens/seen_total_epoch_end' when token totals are provided
    
    Args:
        writer: TensorBoard SummaryWriter instance
        epoch (int): Current epoch number
        avg_scaled_loss (float): Average scaled loss over the epoch
        avg_unscaled_loss (float): Average unscaled loss over the epoch
        perplexity (float): Perplexity metric value
        accelerator: Accelerator instance for distributed training checks
        configs: Configuration object with tensorboard_log flag
        tokens_seen_total (float, optional): Cumulative tokens processed up to this epoch.
    
    Returns:
        None. Logs are written directly to TensorBoard if conditions are met.
    """
    if writer is None or accelerator is None or configs is None:
        return

    if not accelerator.is_main_process or not configs.tensorboard_log:
        return

    writer.add_scalar('epoch_metrics/loss', avg_scaled_loss, epoch)
    writer.add_scalar('epoch_metrics/unscaled_loss', avg_unscaled_loss, epoch)
    writer.add_scalar('epoch_metrics/perplexity', perplexity, epoch)
    writer.add_scalar('epoch_metrics/tokens', tokens_seen_total, epoch)
    writer.add_scalar('epoch_metrics/loss_vs_tokens', avg_scaled_loss, tokens_seen_total)
    writer.add_scalar('epoch_metrics/perplexity_vs_tokens', perplexity, tokens_seen_total)


def log_eval_epoch_metrics(writer, epoch, avg_scaled_loss, avg_unscaled_loss, perplexity, tokens_seen_total,
                           accelerator=None, configs=None):
    """Log end-of-epoch evaluation/validation metrics to TensorBoard.
    
    This function logs aggregated evaluation metrics at the end of each validation epoch,
    including scaled loss, unscaled loss, and perplexity. It automatically handles all
    necessary checks (main process, tensorboard enabled) and returns early if conditions
    are not met.
    
    Metrics logged:
    - 'loss/total': Averaged scaled loss for the evaluation
    - 'unscaled_loss/total': Averaged unscaled loss for the evaluation
    - 'perplexity': Perplexity metric (exp of average negative log-likelihood)
    
    Args:
        writer: TensorBoard SummaryWriter instance
        epoch (int): Current epoch number
        avg_scaled_loss (float): Average scaled loss over the evaluation
        avg_unscaled_loss (float): Average unscaled loss over the evaluation
        perplexity (float): Perplexity metric value
        tokens_seen_total (float): Cumulative tokens processed up to this evaluation.
        accelerator: Accelerator instance for distributed training checks
        configs: Configuration object with tensorboard_log flag
    
    Returns:
        None. Logs are written directly to TensorBoard if conditions are met.
    """
    if writer is None or accelerator is None or configs is None:
        return

    if not accelerator.is_main_process or not configs.tensorboard_log:
        return

    writer.add_scalar('epoch_metrics/loss', avg_scaled_loss, epoch)
    writer.add_scalar('epoch_metrics/unscaled_loss', avg_unscaled_loss, epoch)
    writer.add_scalar('epoch_metrics/perplexity', perplexity, epoch)
    writer.add_scalar('epoch_metrics/loss_vs_tokens', avg_unscaled_loss, tokens_seen_total)
    writer.add_scalar('epoch_metrics/perplexity_vs_tokens', perplexity, tokens_seen_total)


def log_eval_step_metrics(writer, global_step, avg_scaled_loss, avg_unscaled_loss, perplexity,
                          tokens_seen_total, accelerator=None,
                          configs=None):
    """Log evaluation metrics keyed by global training step.

    Designed for intermediate validation runs inside the training loop where multiple
    evaluations can occur within a single epoch.

    Args:
        writer: TensorBoard SummaryWriter instance.
        global_step (int): Current global training step.
        avg_scaled_loss (float): Averaged scaled loss over the evaluation run.
        avg_unscaled_loss (float): Averaged unscaled loss over the evaluation run.
        perplexity (float): Perplexity metric value.
        tokens_seen_total (float): Cumulative tokens processed up to this evaluation.
        accelerator: Accelerator instance for distributed training checks.
        configs: Configuration object with tensorboard_log flag.

    Returns:
        None. Metrics are written directly to TensorBoard when logging is enabled.
    """
    if writer is None or accelerator is None or configs is None:
        return

    if not accelerator.is_main_process or not configs.tensorboard_log:
        return

    decay_cfg = getattr(getattr(configs, 'optimizer', None), 'decay', None)
    warmup_steps = int(getattr(decay_cfg, 'warmup', 0)) if decay_cfg is not None else 0
    if global_step < warmup_steps:
        return

    writer.add_scalar('step_metrics/loss', avg_scaled_loss, global_step)
    writer.add_scalar('step_metrics/unscaled_loss', avg_unscaled_loss, global_step)
    writer.add_scalar('step_metrics/perplexity', perplexity, global_step)
    writer.add_scalar('step_metrics/loss_vs_tokens', avg_unscaled_loss, tokens_seen_total)
    writer.add_scalar('step_metrics/perplexity_vs_tokens', perplexity, tokens_seen_total)


def prepare_tensorboard(result_path):
    """Prepares TensorBoard SummaryWriter objects for training and validation logging.

    This function creates subdirectories for training and validation logs within the
    specified `result_path`. It then initializes and returns TensorBoard
    SummaryWriter objects for both training and validation phases.

    The directory structure created is:
    `result_path`/
        train/
            tensorboard/
        val/
            tensorboard/

    Args:
        result_path (str): The base directory path where the 'train' and 'val'
                           subdirectories for TensorBoard logs will be created.

    Returns:
        tuple[torch.utils.tensorboard.SummaryWriter, torch.utils.tensorboard.SummaryWriter]:
            A tuple containing two SummaryWriter objects:
            - The first element is the writer for training logs.
            - The second element is the writer for validation logs.
    """
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer
