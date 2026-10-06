import os
import argparse
import numpy as np
import torch
import transformers
import yaml
from time import time
from tqdm import tqdm
from utils.utils import load_configs, prepare_saving_dir, validate_loss_configuration
from utils.model import save_checkpoint, load_resume_checkpoint
from utils.optimizer import prepare_optimizer
from data.dataset import prepare_dataloaders
from models.super_model import prepare_models
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from utils.log import test_gpu_cuda, get_logging, prepare_tensorboard
from utils.model import compile_model as custom_compile_model
from utils.loss import calculate_loss, log_per_loss_grad_norms
from utils.training_helpers import (
    init_phase_metrics,
    update_phase_metrics,
    summarize_phase_metrics,
    init_loss_tracker,
    accumulate_train_losses,
    finalize_train_step,
    accumulate_eval_losses,
    update_unique_indices,
    compute_activation_ratio,
    summarize_running_losses,
    compute_epoch_loss_averages,
    apply_loss_toggles,
    format_progress_postfix,
    describe_train_progress,
    describe_eval_progress,
    log_tensorboard_phase,
)

torch.set_printoptions(sci_mode=False)

AMINO_ACID_CLASSES = list("ACDEFGHIKLMNPQRSTVWYX")


def train_loop(net, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    configs = kwargs.pop('configs')
    writer = kwargs.pop('writer')
    _ = kwargs.pop('logging')
    adaptive_loss_coeffs = kwargs.pop('adaptive_loss_coeffs', None)

    alpha = configs.model.vqvae.vector_quantization.alpha
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation
    classifier_cfg = getattr(configs.model.vqvae.decoder, 'classifier_head', None)
    classifier_head_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False
    losses_cfg = configs.train_settings.losses
    mse_enabled = bool(getattr(getattr(losses_cfg, "mse", None), 'enabled', False))
    cosine_enabled = bool(getattr(getattr(losses_cfg, 'cosine_similarity', None), 'enabled', False))
    ce_enabled = bool(getattr(getattr(losses_cfg, 'cross_entropy', None), 'enabled', False))
    ntp_cfg = getattr(losses_cfg, 'next_token_prediction', None)
    ntp_enabled = bool(getattr(ntp_cfg, 'enabled', False))

    metrics = init_phase_metrics(configs, accelerator, AMINO_ACID_CLASSES)
    loss_tracker = init_loss_tracker(accelerator.device, accum_iter)

    tik_tok_enabled = metrics.tik_tok_accuracy is not None
    loss_toggles = {
        'mse_loss': mse_enabled,
        'cosine_loss': cosine_enabled,
        'ce_loss': ce_enabled,
        'ntp_loss': ntp_enabled,
        'classification_loss': classifier_head_enabled,
        'tik_tok_padding_loss': tik_tok_enabled,
    }

    optimizer.zero_grad()
    global_step = kwargs.get('global_step', 0)

    progress_bar = tqdm(
        range(0, int(np.ceil(len(train_loader) / accum_iter))),
        leave=False,
        disable=not (configs.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description(f"Epoch {epoch}")

    net.train()
    for _, data in enumerate(train_loader):
        with accelerator.accumulate(net):
            model_output = net(data)
            loss_batch = calculate_loss(
                model_output,
                alpha,
                configs,
                adaptive_loss_coeffs=adaptive_loss_coeffs,
            )

            update_phase_metrics(metrics, model_output, detach=True)
            accumulate_train_losses(loss_tracker, loss_batch, accelerator.device)

            adaptive_loss_coeffs = log_per_loss_grad_norms(
                loss_batch,
                net,
                configs,
                writer,
                accelerator,
                global_step,
                adaptive_loss_coeffs=adaptive_loss_coeffs,
            )

            accelerator.backward(loss_batch['total_loss'])

            if accelerator.sync_gradients:
                finalize_train_step(loss_tracker, accelerator, model_output['indices'])

                if (
                        accelerator.is_main_process
                        and configs.tensorboard_log
                        and global_step % configs.train_settings.gradient_norm_logging_freq == 0
                ):
                    params_with_grad = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
                    if params_with_grad:
                        grad_norm = torch.norm(
                            torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]),
                            2,
                        )
                        writer.add_scalar('gradient norm/total_amp_scaled', grad_norm.item(), global_step)

                accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and configs.tensorboard_log:
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

                running_scaled, _ = summarize_running_losses(loss_tracker)
                running_scaled_masked = dict(running_scaled)
                for key, enabled in loss_toggles.items():
                    if not enabled and key in running_scaled_masked:
                        running_scaled_masked[key] = float('nan')

                tik_tok_display = float('nan')
                if metrics.tik_tok_accuracy is not None:
                    acc_tensor = metrics.tik_tok_accuracy.compute()
                    tik_tok_display = acc_tensor.item() if acc_tensor is not None else float('nan')

                progress_bar.set_description(
                    describe_train_progress(epoch, running_scaled_masked, tik_tok_display)
                )

        progress_bar.set_postfix(format_progress_postfix(optimizer, loss_batch, global_step))

    scaled_losses, unscaled_losses = compute_epoch_loss_averages(loss_tracker)
    scaled_losses_masked = dict(scaled_losses)
    unscaled_losses_masked = dict(unscaled_losses)
    apply_loss_toggles(scaled_losses_masked, unscaled_losses_masked, loss_toggles)

    metric_summary = summarize_phase_metrics(metrics)
    activation_ratio = compute_activation_ratio(loss_tracker['unique_indices'], codebook_size)
    activation_percent = round(activation_ratio * 100, 1)

    if accelerator.is_main_process and configs.tensorboard_log:
        log_tensorboard_phase(
            writer,
            split='train',
            epoch=epoch,
            scaled_losses=scaled_losses_masked,
            unscaled_losses=unscaled_losses_masked,
            metrics=metric_summary,
            activation_percent=activation_percent,
            include_scaled=True,
            include_classification=classifier_head_enabled,
            include_ntp=ntp_enabled,
            include_tik_tok=tik_tok_enabled,
            include_mse=mse_enabled,
            include_cosine=cosine_enabled,
            include_ce=ce_enabled,
        )

    result = {
        'loss': unscaled_losses_masked['unscaled_total_loss'],
        'rec_loss': unscaled_losses_masked['unscaled_rec_loss'],
        'vq_loss': unscaled_losses_masked['unscaled_vq_loss'],
        'mse_loss': unscaled_losses_masked['unscaled_mse_loss'],
        'cosine_loss': unscaled_losses_masked['unscaled_cosine_loss'],
        'ce_loss': unscaled_losses_masked['unscaled_ce_loss'],
        'ntp_loss': unscaled_losses_masked['unscaled_ntp_loss'],
        'tik_tok_padding_loss': unscaled_losses_masked['unscaled_tik_tok_padding_loss'],
        'classification_loss': unscaled_losses_masked['unscaled_classification_loss'],
        'nmse_median': metric_summary.get('nmse_median', float('nan')),
        'nmse_p95': metric_summary.get('nmse_p95', float('nan')),
        'nmse_mean': metric_summary.get('nmse_mean', float('nan')),
        'nmse_strict_coverage': metric_summary.get('nmse_strict_coverage', float('nan')),
        'nmse_token_count': metric_summary.get('nmse_token_count', 0),
        'classification_accuracy': metric_summary.get('classification_accuracy', float('nan')),
        'classification_accuracy_per_class': metric_summary.get('classification_accuracy_per_class', {}),
        'activation': activation_percent,
        'counter': loss_tracker['counter'],
        'global_step': global_step,
        'recon_ppl_ratio': metric_summary.get('recon_ppl_ratio', float('nan')),
        'recon_ppl_pct_delta': metric_summary.get('recon_ppl_pct_delta', float('nan')),
        'recon_kl': metric_summary.get('recon_kl', float('nan')),
    }

    if metrics.ntp_perplexity is not None:
        result['ntp_perplexity'] = metric_summary.get('ntp_perplexity', float('nan'))
    if metrics.tik_tok_accuracy is not None:
        result['tik_tok_padding_accuracy'] = metric_summary.get('tik_tok_padding_accuracy', float('nan'))

    return result


def evaluation_loop(net, dataloader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    configs = kwargs.pop('configs')
    writer = kwargs.pop('writer')
    _ = kwargs.pop('logging', None)
    name = kwargs.pop('name', 'validation')
    mode = kwargs.pop('mode', name)
    _ = kwargs.pop('global_step', None)
    _ = kwargs.pop('result_path', None)

    alpha = configs.model.vqvae.vector_quantization.alpha
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    classifier_cfg = getattr(configs.model.vqvae.decoder, 'classifier_head', None)
    classifier_head_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False
    losses_cfg = configs.train_settings.losses
    mse_enabled = bool(getattr(getattr(losses_cfg, 'mse', None), 'enabled', False))
    cosine_enabled = bool(getattr(getattr(losses_cfg, 'cosine_similarity', None), 'enabled', False))
    ce_enabled = bool(getattr(getattr(losses_cfg, 'cross_entropy', None), 'enabled', False))
    ntp_cfg = getattr(losses_cfg, 'next_token_prediction', None)
    ntp_enabled = bool(getattr(ntp_cfg, 'enabled', False))

    metrics = init_phase_metrics(configs, accelerator, AMINO_ACID_CLASSES)
    loss_tracker = init_loss_tracker(accelerator.device, accum_iter=1)

    tik_tok_enabled = metrics.tik_tok_accuracy is not None
    loss_toggles = {
        'mse_loss': mse_enabled,
        'cosine_loss': cosine_enabled,
        'ce_loss': ce_enabled,
        'ntp_loss': ntp_enabled,
        'classification_loss': classifier_head_enabled,
        'tik_tok_padding_loss': tik_tok_enabled,
    }

    progress_bar = tqdm(
        dataloader,
        leave=False,
        disable=not (configs.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description(f"{mode} {name}")

    net.eval()
    with torch.inference_mode():
        for data in progress_bar:
            model_output = net(data)
            loss_batch = calculate_loss(model_output, alpha, configs)

            update_phase_metrics(metrics, model_output, detach=False)
            accumulate_eval_losses(
                loss_tracker,
                loss_batch,
                accelerator,
                repeat=configs.valid_settings.batch_size,
                device=accelerator.device,
            )
            update_unique_indices(loss_tracker, model_output['indices'], accelerator)
            loss_tracker['counter'] += 1

            running_scaled, _ = summarize_running_losses(loss_tracker)
            running_scaled_masked = dict(running_scaled)
            for key, enabled in loss_toggles.items():
                if not enabled and key in running_scaled_masked:
                    running_scaled_masked[key] = float('nan')

            tik_tok_display = float('nan')
            if metrics.tik_tok_accuracy is not None:
                acc_tensor = metrics.tik_tok_accuracy.compute()
                tik_tok_display = acc_tensor.item() if acc_tensor is not None else float('nan')

            progress_bar.set_description(
                describe_eval_progress(mode, name, running_scaled_masked, tik_tok_display)
            )

    scaled_losses, unscaled_losses = compute_epoch_loss_averages(loss_tracker)
    scaled_losses_masked = dict(scaled_losses)
    unscaled_losses_masked = dict(unscaled_losses)
    apply_loss_toggles(scaled_losses_masked, unscaled_losses_masked, loss_toggles)

    metric_summary = summarize_phase_metrics(metrics)
    activation_ratio = compute_activation_ratio(loss_tracker['unique_indices'], codebook_size)
    activation_percent = round(activation_ratio * 100, 1)

    if accelerator.is_main_process and configs.tensorboard_log:
        log_tensorboard_phase(
            writer,
            split=mode,
            epoch=epoch,
            scaled_losses=scaled_losses_masked,
            unscaled_losses=unscaled_losses_masked,
            metrics=metric_summary,
            activation_percent=activation_percent,
            include_scaled=False,
            include_classification=classifier_head_enabled,
            include_ntp=ntp_enabled,
            include_tik_tok=tik_tok_enabled,
            include_mse=mse_enabled,
            include_cosine=cosine_enabled,
            include_ce=ce_enabled,
        )

    result = {
        'loss': unscaled_losses_masked['unscaled_total_loss'],
        'rec_loss': unscaled_losses_masked['unscaled_rec_loss'],
        'vq_loss': unscaled_losses_masked['unscaled_vq_loss'],
        'mse_loss': unscaled_losses_masked['unscaled_mse_loss'],
        'cosine_loss': unscaled_losses_masked['unscaled_cosine_loss'],
        'ce_loss': unscaled_losses_masked['unscaled_ce_loss'],
        'ntp_loss': unscaled_losses_masked['unscaled_ntp_loss'],
        'tik_tok_padding_loss': unscaled_losses_masked['unscaled_tik_tok_padding_loss'],
        'classification_loss': unscaled_losses_masked['unscaled_classification_loss'],
        'nmse_median': metric_summary.get('nmse_median', float('nan')),
        'nmse_p95': metric_summary.get('nmse_p95', float('nan')),
        'nmse_mean': metric_summary.get('nmse_mean', float('nan')),
        'nmse_strict_coverage': metric_summary.get('nmse_strict_coverage', float('nan')),
        'nmse_token_count': metric_summary.get('nmse_token_count', 0),
        'classification_accuracy': metric_summary.get('classification_accuracy', float('nan')),
        'classification_accuracy_per_class': metric_summary.get('classification_accuracy_per_class', {}),
        'activation': activation_percent,
        'recon_ppl_ratio': metric_summary.get('recon_ppl_ratio', float('nan')),
        'recon_ppl_pct_delta': metric_summary.get('recon_ppl_pct_delta', float('nan')),
        'recon_kl': metric_summary.get('recon_kl', float('nan')),
    }

    if metrics.ntp_perplexity is not None:
        result['ntp_perplexity'] = metric_summary.get('ntp_perplexity', float('nan'))
    if metrics.tik_tok_accuracy is not None:
        result['tik_tok_padding_accuracy'] = metric_summary.get('tik_tok_padding_accuracy', float('nan'))

    return result


def main(dict_config, config_file_path):
    configs = load_configs(dict_config, config_path=config_file_path)

    transformers.logging.set_verbosity_error()

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.cuda.manual_seed(configs.fix_seed)
        torch.cuda.manual_seed_all(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=configs.accelerate_timeout_minutes)),
                         DistributedDataParallelKwargs(find_unused_parameters=configs.find_unused_parameters)],
    )

    # initialize result and checkpoint paths
    result_path = checkpoint_path = None
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
        paths = [result_path, checkpoint_path]
    else:
        # Initialize with placeholders.
        paths = [None, None]

    if accelerator.num_processes > 1:
        import torch.distributed as dist
        # Broadcast the list of strings from the main process (src=0) to all others.
        dist.broadcast_object_list(paths, src=0)

        # Now every process has the shared values.
        result_path, checkpoint_path = paths

    logging = get_logging(result_path, configs)

    train_loader, valid_loader, test_loader = prepare_dataloaders(configs, logging)
    logging.info('preparing dataloaders are done')

    test_gpu_cuda(logging, accelerator=accelerator)

    validate_loss_configuration(configs)
    net = prepare_models(configs, logging)
    logging.info('preparing model is done')

    load_resume_checkpoint(net, configs, logging)

    # compile model to train faster and efficiently
    if configs.model.compile_model:
        net = custom_compile_model(net, mode='default')
        accelerator.wait_for_everyone()
        logging.info('custom compile model is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_loader), logging,
                                             accelerator=accelerator)
    logging.info('preparing optimizer is done')

    net, optimizer, train_loader, valid_loader, test_loader, scheduler = accelerator.prepare(
        net, optimizer, train_loader, valid_loader, test_loader, scheduler
    )

    # Create default adaptive coefficients on all processes
    adaptive_loss_coeffs = {
        'mse': 1.0,
        'cosine_similarity': 1.0,
        'cross_entropy': 1.0,
        'vq': 1.0,
        'ntp': 1.0,
        'tik_tok_padding': 1.0,
        'classification': 1.0,
    }

    if accelerator.is_main_process:
        # initialize tensorboards
        train_writer, valid_writer = prepare_tensorboard(result_path)
    else:
        train_writer, valid_writer = None, None

    train_steps = int(np.ceil(len(train_loader) / configs.train_settings.grad_accumulation))
    logging.info(f'number of train steps per epoch: {train_steps}')
    logging.info(f'number of valid steps per epoch: {int(len(valid_loader))}')

    global_step = 0
    start_epoch = 1

    classifier_cfg = getattr(configs.model.vqvae.decoder, 'classifier_head', None)
    classifier_head_enabled = bool(getattr(classifier_cfg, 'enabled', False)) if classifier_cfg is not None else False

    best_metric_key = 'classification_loss' if classifier_head_enabled else 'rec_loss'

    best_valid_metrics = {
        'loss': float('inf'),
        'rec_loss': float('inf'),
        'vq_loss': float('inf'),
        'mse_loss': float('inf'),
        'cosine_loss': float('inf'),
        'ce_loss': float('inf'),
        'classification_loss': float('inf'),
        'classification_accuracy': 0.0,
        'classification_accuracy_per_class': {},
        'activation': 0.0,
        'ntp_loss': float('inf'),
        'nmse_median': float('inf'),
        'ntp_perplexity': float('inf'),
        'recon_ppl_ratio': float('nan'),
        'recon_ppl_pct_delta': float('nan'),
        'tik_tok_padding_accuracy': 0.0,
    }

    logging.info(f'training start')
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        start_time = time()
        training_loop_reports = train_loop(
            net, train_loader, epoch, accelerator=accelerator, writer=train_writer,
            optimizer=optimizer, scheduler=scheduler, logging=logging,
            global_step=global_step, configs=configs, adaptive_loss_coeffs=adaptive_loss_coeffs
        )
        end_time = time()
        training_time = end_time - start_time
        metric_segments = []
        log_items = [
            ("loss", training_loop_reports.get("loss"), ".4f"),
            ("rec loss", training_loop_reports.get("rec_loss"), ".8f"),
            ("vq loss", training_loop_reports.get("vq_loss"), ".4f"),
            ("mse loss", training_loop_reports.get("mse_loss"), ".8f"),
            ("cosine loss", training_loop_reports.get("cosine_loss"), ".4f"),
            ("ce loss", training_loop_reports.get("ce_loss"), ".4f"),
            ("class loss", training_loop_reports.get("classification_loss"), ".4f"),
            ("nmse_median", training_loop_reports.get("nmse_median"), ".4f"),
            ("class acc", training_loop_reports.get("classification_accuracy"), ".4f"),
            ("activation", training_loop_reports.get("activation"), ".1f"),
            ("ntp loss", training_loop_reports.get("ntp_loss"), ".4f"),
            ("%ΔPPL", training_loop_reports.get("recon_ppl_pct_delta"), ".2f"),
            ("rec kl", training_loop_reports.get("recon_kl"), ".4f"),
            ("tik tok acc", training_loop_reports.get("tik_tok_padding_accuracy"), ".4f"),
            ("ntp_perplexity", training_loop_reports.get("ntp_perplexity"), ".2f"),
        ]
        for label, value, fmt in log_items:
            if value is not None and not np.isnan(value):
                metric_segments.append(f'{label} {value:{fmt}}')
        metric_suffix = ', ' + ', '.join(metric_segments) if metric_segments else ''
        logging.info(
            f'epoch {epoch} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}s, '
            f'global steps {training_loop_reports["global_step"]}{metric_suffix}')

        global_step = training_loop_reports["global_step"]
        accelerator.wait_for_everyone()

        if epoch % configs.checkpoints_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # Set the path to save the models checkpoint.
                model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
                save_checkpoint(epoch, model_path, net, optimizer, accelerator,
                                global_step=global_step, configs=configs)
                logging.info(f'\tcheckpoint models in {model_path}')
            accelerator.wait_for_everyone()

        if epoch % configs.valid_settings.do_every == 0:
            accelerator.wait_for_everyone()
            start_time = time()
            evaluation_reports = evaluation_loop(
                net, valid_loader, epoch,
                accelerator=accelerator,
                configs=configs,
                logging=logging, global_step=global_step,
                writer=valid_writer, result_path=result_path
            )
            end_time = time()
            evaluation_time = end_time - start_time
            accelerator.wait_for_everyone()
            eval_metric_segments = []
            eval_items = [
                ("loss", evaluation_reports.get("loss"), ".4f"),
                ("rec loss", evaluation_reports.get("rec_loss"), ".8f"),
                ("vq loss", evaluation_reports.get("vq_loss"), ".4f"),
                ("mse loss", evaluation_reports.get("mse_loss"), ".8f"),
                ("cosine loss", evaluation_reports.get("cosine_loss"), ".4f"),
                ("ce loss", evaluation_reports.get("ce_loss"), ".4f"),
                ("class loss", evaluation_reports.get("classification_loss"), ".4f"),
                ("nmse_median", evaluation_reports.get("nmse_median"), ".4f"),
                ("class acc", evaluation_reports.get("classification_accuracy"), ".4f"),
                ("activation", evaluation_reports.get("activation"), ".1f"),
                ("ntp loss", evaluation_reports.get("ntp_loss"), ".4f"),
                ("%ΔPPL", evaluation_reports.get("recon_ppl_pct_delta"), ".2f"),
                ("rec kl", training_loop_reports.get("recon_kl"), ".4f"),
                ("tik tok acc", evaluation_reports.get("tik_tok_padding_accuracy"), ".4f"),
                ("ntp_perplexity", evaluation_reports.get("ntp_perplexity"), ".2f"),
            ]
            for label, value, fmt in eval_items:
                if value is not None and not np.isnan(value):
                    eval_metric_segments.append(f'{label} {value:{fmt}}')
            eval_suffix = ', ' + ', '.join(eval_metric_segments) if eval_metric_segments else ''
            logging.info(
                f'validation - time {np.round(evaluation_time, 2)}s{eval_suffix}')

            # Check validation loss to save the best model
            if evaluation_reports[best_metric_key] < best_valid_metrics[best_metric_key]:
                best_valid_metrics.update({
                    'loss': evaluation_reports['loss'],
                    'rec_loss': evaluation_reports['rec_loss'],
                    'vq_loss': evaluation_reports['vq_loss'],
                    'mse_loss': evaluation_reports['mse_loss'],
                    'cosine_loss': evaluation_reports['cosine_loss'],
                    'ce_loss': evaluation_reports['ce_loss'],
                    'classification_loss': evaluation_reports['classification_loss'],
                    'classification_accuracy': evaluation_reports['classification_accuracy'],
                    'classification_accuracy_per_class': evaluation_reports['classification_accuracy_per_class'],
                    'activation': evaluation_reports['activation'],
                    'ntp_loss': evaluation_reports['ntp_loss'],
                    'ntp_perplexity': evaluation_reports.get('ntp_perplexity', float('nan')),
                    'recon_ppl_ratio': evaluation_reports['recon_ppl_ratio'],
                    'recon_ppl_pct_delta': evaluation_reports['recon_ppl_pct_delta'],
                    'nmse_median': evaluation_reports['nmse_median'],
                    'tik_tok_padding_accuracy': evaluation_reports.get('tik_tok_padding_accuracy', float('nan'))
                })

                # sync processes before saving best model
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # Set the path to save the model checkpoint.
                    model_path = os.path.join(checkpoint_path, f'best_valid.pth')
                    save_checkpoint(epoch, model_path, net, optimizer, accelerator,
                                    global_step=global_step, configs=configs)
                    logging.info(f'\tsaving the best models in {model_path}')
                    logging.info(f'\tbest validation loss: {best_valid_metrics[best_metric_key]:.4f}')
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        train_writer.close()
        valid_writer.close()

    # best_valid_metrics summary
    logging.info(f"best validation metrics: {best_valid_metrics}")

    for param in net.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()

    accelerator.free_memory()
    del net, optimizer, scheduler
    torch.cuda.empty_cache()

    logging.info('training is done')
    accelerator.end_training()
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Protein Latent model to learn the language of protein.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
