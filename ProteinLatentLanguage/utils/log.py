import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging as log
from accelerate.logging import get_logger
from io import StringIO
from pathlib import Path
from torchmetrics.text import Perplexity
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


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


def prepare_metrics(accelerator, decoder_tokenizer_padding_index, num_classes):
    """Initializes and prepares a dictionary of evaluation metrics.

    This function sets up various metrics from the torchmetrics library, moves them
    to the specified accelerator device, and returns them in a dictionary.

    Args:
        accelerator: The Hugging Face Accelerate object.
        decoder_tokenizer_padding_index (int): The padding index used by the decoder tokenizer,
                                               to be ignored by the Perplexity metric.
        num_classes (int): The number of classes for multi-class classification metrics.

    Returns:
        dict: A dictionary where keys are metric names (str) and values are the
              corresponding torchmetrics objects moved to the accelerator's device.
              Metrics included:
              - 'ntp_perplexity_metric': Perplexity
              - 'accuracy_metric': Accuracy (multiclass)
              - 'f1_metric': F1Score (multiclass)
              - 'precision_metric': Precision (multiclass)
              - 'recall_metric': Recall (multiclass)
              - 'auroc_metric': AUROC (multiclass)
              - 'mse_metric': MeanSquaredError
              - 'mae_metric': MeanAbsoluteError
    """
    metrics_dict = {}

    # Text Generation Metric
    metrics_dict["ntp_perplexity_metric"] = Perplexity(ignore_index=decoder_tokenizer_padding_index)

    # Classification Metrics
    metrics_dict["accuracy_metric"] = Accuracy(task="multiclass", num_classes=num_classes)
    metrics_dict["f1_metric"] = F1Score(task="multiclass", num_classes=num_classes)
    metrics_dict["precision_metric"] = Precision(task="multiclass", num_classes=num_classes)
    metrics_dict["recall_metric"] = Recall(task="multiclass", num_classes=num_classes)
    metrics_dict["auroc_metric"] = AUROC(task="multiclass", num_classes=num_classes)

    # Regression Metrics
    metrics_dict["mse_metric"] = MeanSquaredError()
    metrics_dict["mae_metric"] = MeanAbsoluteError()

    # Move all metrics to the accelerator device
    for metric_name in metrics_dict:
        metrics_dict[metric_name] = metrics_dict[metric_name].to(accelerator.device)

    return metrics_dict


def compute_metrics(metrics_dict, predictions, targets, decoder_tokenizer_padding_index=None):
    # todo: need to check if the metrics are compatible with the predictions and targets

    return None
