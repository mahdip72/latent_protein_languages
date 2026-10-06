import os
import argparse
import numpy as np
import torch
import transformers
import yaml
from contextlib import nullcontext
from time import time
from tqdm import tqdm
from utils.utils import (
    load_configs,
    prepare_saving_dir,
    save_tokenizer_vocab,
    set_datasets_curriculum_epoch,
    prepare_accelerator_handlers,
    configure_compile_cache_dirs,
    suppress_inductor_autotune_logging,
    get_fsdp_config,
)
from utils.model import (
    save_checkpoint,
    load_resume_checkpoint,
    save_step_checkpoint_if_needed,
    load_optimizer_state_from_resume,
    compile_model,
)
from utils.optimizer import prepare_optimizer
from data.dataset import prepare_dataloaders
from models.super_model import prepare_models
from accelerate import Accelerator

from utils.log import (
    test_gpu_cuda, get_logging, prepare_tensorboard,
    log_train_step_metrics, log_train_epoch_metrics, log_eval_epoch_metrics,
    log_eval_step_metrics,
    log_mixed_precision,
)
from utils.loss import calculate_loss
from torchmetrics.text import Perplexity

torch.set_printoptions(sci_mode=False)


def _to_python_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _is_e1_context_encoder_enabled(configs) -> bool:
    condition_cfg = getattr(getattr(configs, "model", None), "condition_tokens", None)
    seq2struct_cfg = getattr(condition_cfg, "sequence_to_structure", None)
    context_cfg = getattr(seq2struct_cfg, "protein_encoder_context", None)
    if not bool(getattr(seq2struct_cfg, "enabled", False)):
        return False
    if not bool(getattr(context_cfg, "enable", getattr(context_cfg, "enabled", False))):
        return False
    return str(getattr(context_cfg, "model_type", "")).strip().lower() == "e1"


def log_fp8_ddp_compile_heads_up(logging, configs):
    mixed_precision = str(getattr(configs.train_settings, "mixed_precision", "")).strip().lower()
    compile_enabled = bool(getattr(configs.model, "compile_model", False))
    compile_mode = str(getattr(configs.model, "compile_mode", "")).strip().lower()
    use_fsdp2 = bool(getattr(get_fsdp_config(configs), "enabled", False))
    if mixed_precision != "fp8" or use_fsdp2 or not compile_enabled or compile_mode != "max-autotune":
        return

    if _is_e1_context_encoder_enabled(configs):
        logging.warning(
            "DDP + fp8 + max-autotune compile is enabled with the E1 protein_encoder_context. "
            "The pure structure-LM path has been tested, but the E1 context-encoder path can "
            "exercise different compiled modules and should be validated separately.",
        )


def train_loop(net, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    configs = kwargs.pop('configs')
    writer = kwargs.pop('writer')
    eval_writer = kwargs.pop('eval_writer', None)
    valid_loader = kwargs.pop('valid_loader', None)
    logging = kwargs.pop('logging')
    cumulative_tokens_seen = int(kwargs.pop('tokens_seen_total', 0))
    checkpoint_path = kwargs.pop('checkpoint_path')

    accum_iter = configs.train_settings.grad_accumulation
    eval_step_interval = int(getattr(configs.valid_settings, 'do_every_steps', 0))

    epoch_loss_sum = 0.0
    epoch_scaled_loss_sum = 0.0
    epoch_step_count = 0
    tokens_in_step_accumulator = 0

    # initialize metrics
    # Use pad_token_id (0) as ignore_index to exclude padding from perplexity calculation
    perplexity_metric = Perplexity(ignore_index=0).to(accelerator.device)

    optimizer.zero_grad()
    global_step = kwargs.get('global_step', 0)

    progress_bar = tqdm(
        range(0, int(np.ceil(len(train_loader) / accum_iter))),
        leave=False,
        disable=not (configs.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description(f"Epoch {epoch}")

    net.train()
    mean_unscaled_loss_val = 0.0
    mean_scaled_loss_val = 0.0
    for _, data in enumerate(train_loader):
        save_step_checkpoint_if_needed(
            epoch,
            checkpoint_path,
            global_step,
            net,
            optimizer,
            scheduler,
            accelerator,
            logging,
            configs,
        )
        accumulation_ctx = accelerator.accumulate(net) if accum_iter > 1 else nullcontext()
        with accumulation_ctx:
            model_output = net(data)
            logits = model_output['decoder_output']
            targets_full = data['target_ids']

            perplexity_metric(logits, targets_full)

            loss_dict = calculate_loss(model_output, configs)

            gathered_unscaled_loss = accelerator.gather_for_metrics(loss_dict['unscaled_loss'].detach())
            mean_unscaled_loss_val = gathered_unscaled_loss.float().mean().item()
            gathered_scaled_loss = accelerator.gather_for_metrics(loss_dict['scaled_loss'].detach())
            mean_scaled_loss_val = gathered_scaled_loss.float().mean().item()

            batch_token_counts = data['unmasked_token_count'].to(accelerator.device)
            gathered_tokens = accelerator.gather_for_metrics(batch_token_counts.detach()).sum().item()
            tokens_in_step_accumulator += int(gathered_tokens)

            accelerator.backward(loss_dict['scaled_loss'])

            if accum_iter > 1:
                local_sync_flag = torch.tensor(
                    [1 if accelerator.sync_gradients else 0],
                    device=accelerator.device,
                    dtype=torch.int32,
                )
                sync_flags = accelerator.gather_for_metrics(local_sync_flag)
                sync_gradients_now = bool(sync_flags.max().item())
                sync_flags_consistent = bool(sync_flags.min().item() == sync_flags.max().item())
                if (not sync_flags_consistent) and accelerator.is_main_process:
                    logging.warning(
                        "Rank-local sync_gradients mismatch detected (%s); using synchronized step decision=%s.",
                        sync_flags.tolist(),
                        sync_gradients_now,
                    )
            else:
                sync_gradients_now = True

            if sync_gradients_now:
                current_lr = _to_python_float(optimizer.param_groups[0]['lr'])
                epoch_loss_sum += mean_unscaled_loss_val
                epoch_scaled_loss_sum += mean_scaled_loss_val
                epoch_step_count += 1

                tokens_for_step = tokens_in_step_accumulator
                cumulative_tokens_seen += tokens_for_step

                log_train_step_metrics(
                    writer, global_step, current_lr,
                    net=net, configs=configs, accelerator=accelerator, compute_grad_norm=True,
                    tokens_seen_total=cumulative_tokens_seen, step_unscaled_loss=mean_unscaled_loss_val,
                    step_scaled_loss=mean_scaled_loss_val
                )

                tokens_in_step_accumulator = 0

                grad_clip_norm = float(getattr(configs.optimizer, "grad_clip_norm", 0.0) or 0.0)
                if grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(net.parameters(), grad_clip_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                progress_bar.set_description(f"epoch {epoch} "
                                             + f"[loss: {mean_unscaled_loss_val:.3f}]")

                if eval_step_interval > 0 and global_step % eval_step_interval == 0:
                    accelerator.wait_for_everyone()
                    step_eval_reports = evaluation_loop(
                        net, valid_loader, epoch,
                        accelerator=accelerator,
                        configs=configs,
                        logging=logging,
                        global_step=global_step,
                        writer=eval_writer,
                        log_steps=True,
                        tokens_seen_total=cumulative_tokens_seen,
                        mode='validation',
                    )
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process and step_eval_reports is not None:
                        logging.info(
                            f'step validation - train step {global_step}, tokens seen {cumulative_tokens_seen:,}, '
                            f'loss {step_eval_reports["loss"]:.4f}, '
                            f'perplexity {step_eval_reports["perplexity"]:.2f}'
                        )

        if accelerator.is_main_process:
            current_lr = _to_python_float(optimizer.param_groups[0]['lr'])
            progress_bar.set_postfix({
                'lr': f'{current_lr:.3e}',
                'loss': mean_unscaled_loss_val,
                'global_step': int(global_step),
            })

    # Keep metric collectives ordered after all ranks finish the epoch loop.
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ppl = perplexity_metric.compute()
    avg_loss = epoch_loss_sum / max(epoch_step_count, 1)
    avg_scaled_loss = epoch_scaled_loss_sum / max(epoch_step_count, 1)
    perplexity_metric.reset()

    log_train_epoch_metrics(
        writer, epoch, avg_scaled_loss, avg_loss, ppl.item(),
        accelerator=accelerator, configs=configs,
        tokens_seen_total=cumulative_tokens_seen
    )

    result = {
        'loss': avg_loss,
        'perplexity': ppl.item(),
        'counter': epoch_step_count,
        'global_step': global_step,
        'tokens_seen_total': cumulative_tokens_seen,
    }

    return result


def evaluation_loop(net, dataloader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    configs = kwargs.pop('configs')
    writer = kwargs.pop('writer')
    _ = kwargs.pop('logging', None)
    mode = kwargs.pop('mode', 'validation')
    global_step = kwargs.pop('global_step')
    _ = kwargs.pop('result_path', None)
    log_steps = kwargs.pop('log_steps', False)
    tokens_seen_total = kwargs.pop('tokens_seen_total', False)

    # Use pad_token_id (0) as ignore_index to exclude padding from perplexity calculation
    perplexity_metric = Perplexity(ignore_index=0).to(accelerator.device)

    total_loss = 0.0
    total_scaled_loss = 0.0
    counter = 0

    # initialize metrics
    perplexity_metric.reset()

    progress_bar = tqdm(
        dataloader,
        leave=False,
        disable=not (configs.tqdm_progress_bar and accelerator.is_main_process),
    )
    progress_bar.set_description(f"{mode}")

    net.eval()
    mean_loss_val = 0.0
    mean_scaled_loss_val = 0.0

    with torch.no_grad():
        for data in progress_bar:
            model_output = net(data)
            logits = model_output['decoder_output']
            targets = data['target_ids']
            perplexity_metric(logits, targets)

            loss_batch = calculate_loss(model_output, configs)
            gathered_unscaled_loss = accelerator.gather_for_metrics(loss_batch['unscaled_loss'].detach())
            mean_loss_val = gathered_unscaled_loss.float().mean().item()
            gathered_scaled_loss = accelerator.gather_for_metrics(loss_batch['scaled_loss'].detach())
            mean_scaled_loss_val = gathered_scaled_loss.float().mean().item()

            counter += 1
            total_loss += mean_loss_val
            total_scaled_loss += mean_scaled_loss_val

            progress_bar.set_description(f"{mode} epoch {epoch} "
                                         + f"[loss: {mean_loss_val:.3f}]")

    # Keep metric collectives ordered after all ranks finish eval iteration.
    accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ppl = perplexity_metric.compute()
    perplexity_metric.reset()
    avg_loss = total_loss / max(counter, 1)
    avg_scaled_loss = total_scaled_loss / max(counter, 1)

    if log_steps:
        log_eval_step_metrics(
            writer, global_step, avg_scaled_loss, avg_loss, ppl.item(),
            accelerator=accelerator, configs=configs,
            tokens_seen_total=tokens_seen_total
        )
    else:
        log_eval_epoch_metrics(
            writer, epoch, avg_scaled_loss, avg_loss, ppl.item(),
            accelerator=accelerator, configs=configs,
            tokens_seen_total=tokens_seen_total
        )

    result = {
        'loss': avg_loss,
        'perplexity': ppl.item(),
        'rec_loss': avg_loss,
        'classification_loss': avg_loss,
        'counter': counter,
    }

    return result


def main(dict_config, config_file_path):
    project_root = os.path.dirname(os.path.abspath(__file__))
    inductor_cache_dir, triton_cache_dir = configure_compile_cache_dirs(project_root)
    autotune_logs_suppressed = suppress_inductor_autotune_logging()

    configs = load_configs(dict_config)
    saved_vocab_path = os.path.join(os.path.dirname(config_file_path), 'tokenizer_vocab.yaml')
    if os.path.isfile(saved_vocab_path):
        configs.model.tokenizer_vocab_path = saved_vocab_path

    transformers.logging.set_verbosity_error()

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.cuda.manual_seed(configs.fix_seed)
        torch.cuda.manual_seed_all(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    kwargs_handlers, fsdp_plugin, parallelism_config = prepare_accelerator_handlers(configs)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        fsdp_plugin=fsdp_plugin,
        parallelism_config=parallelism_config,
        kwargs_handlers=kwargs_handlers,
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

    if fsdp_plugin:
        logging.info(f"Using FSDP plugin: {type(accelerator.state.fsdp_plugin).__name__}")
    else:
        logging.info("Not using FSDP plugin.")

    log_mixed_precision(logging, configs, accelerator)
    log_fp8_ddp_compile_heads_up(logging, configs)
    logging.info(
        "torch.compile cache dirs: TORCHINDUCTOR_CACHE_DIR=%s TRITON_CACHE_DIR=%s",
        inductor_cache_dir,
        triton_cache_dir,
    )
    logging.info("torch.compile autotune verbose logging suppressed: %s", autotune_logs_suppressed)

    train_loader, valid_loader, tokenizer = prepare_dataloaders(
        configs,
        logging,
        accelerator=accelerator,
    )
    logging.info('preparing dataloaders are done')
    
    # Keep references to datasets for epoch-based conditioning control
    # These references remain valid even after accelerator.prepare() wraps the loaders
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset

    # Save tokenizer vocabulary for reproducibility (main process only)
    if accelerator.is_main_process:
        vocab_path = save_tokenizer_vocab(tokenizer, result_path)
        logging.info(f'saved tokenizer vocabulary to {vocab_path}')

    test_gpu_cuda(logging, accelerator=accelerator)

    net = prepare_models(configs, logging, tokenizer_vocab_size=tokenizer.tokenizer_vocab_size, accelerator=accelerator)
    net = compile_model(net, configs, logging)
    logging.info('preparing model is done')

    resume_state = load_resume_checkpoint(net, configs, logging, tokenizer=tokenizer)

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_loader), logging,
                                             accelerator=accelerator)
    logging.info('preparing optimizer is done')

    net, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        net, optimizer, train_loader, valid_loader, scheduler
    )

    if resume_state is not None and resume_state.get('optimizer_state_dict') is not None:
        # When FSDP2 forces a switch from bitsandbytes 8-bit AdamW to native
        # torch.optim.AdamW, the optimizer state formats are incompatible
        # (bnb uses state1/state2 vs. AdamW uses exp_avg/exp_avg_sq).
        # Skip the resume to avoid a KeyError at the first optimizer.step().
        if fsdp_plugin and configs.optimizer.use_8bit_adam:
            logging.info('Skipping optimizer state resume: optimizer type changed '
                         '(bitsandbytes -> torch.optim.AdamW) due to FSDP2; '
                         'state formats are incompatible.')
        else:
            accelerator.wait_for_everyone()
            load_optimizer_state_from_resume(
                optimizer,
                resume_state['optimizer_state_dict'],
                net,
                logging,
            )

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
    if resume_state is not None:
        logging.info("Resume step counters ignored; starting from epoch 1 and global_step 0.")
    tokens_seen_total = 0

    best_metric_key = 'perplexity'  # Updated to use perplexity

    best_valid_metrics = {
        'loss': float('inf'),
        'perplexity': float('inf'),
        'tokens_seen_total': 0.0,
    }

    # Log per-condition start epochs if any are configured
    if accelerator.is_main_process and hasattr(train_dataset, 'condition_start_epochs'):
        for cond_name, start_ep in train_dataset.condition_start_epochs.items():
            if start_ep > 1:
                logging.info(f'Condition "{cond_name}" will be enabled starting from epoch {start_ep}')
    
    logging.info(f'training start')
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        # Update epoch in datasets for epoch-based conditioning control
        # Each process independently sets its own dataset - this is safe in distributed training
        set_datasets_curriculum_epoch(epoch, train_dataset, valid_dataset)
        
        # Log when individual conditions become active
        if accelerator.is_main_process and hasattr(train_dataset, 'condition_start_epochs'):
            for cond_name, start_ep in train_dataset.condition_start_epochs.items():
                if epoch == start_ep and start_ep > 1:
                    logging.info(f'Epoch {epoch} - condition "{cond_name}" is now ENABLED')
        
        start_time = time()
        training_loop_reports = train_loop(
            net, train_loader, epoch, accelerator=accelerator, writer=train_writer,
            optimizer=optimizer, scheduler=scheduler, logging=logging,
            global_step=global_step, configs=configs,
            tokens_seen_total=tokens_seen_total, checkpoint_path=checkpoint_path,
            valid_loader=valid_loader, eval_writer=valid_writer
        )
        end_time = time()
        training_time = end_time - start_time
        metric_segments = []
        log_items = [
            ("loss", training_loop_reports.get("loss"), ".4f"),
            ("perplexity", training_loop_reports.get("perplexity"), ".2f"),
            ("tokens", training_loop_reports.get("tokens_seen_total"), ",.0f"),
        ]
        for label, value, fmt in log_items:
            if value is not None and not np.isnan(value):
                metric_segments.append(f'{label} {value:{fmt}}')
        metric_suffix = ', ' + ', '.join(metric_segments) if metric_segments else ''
        logging.info(
            f'epoch {epoch} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}s, '
            f'global steps {training_loop_reports["global_step"]}{metric_suffix}')

        global_step = training_loop_reports["global_step"]
        tokens_seen_total = training_loop_reports.get("tokens_seen_total", tokens_seen_total)
        accelerator.wait_for_everyone()

        if epoch % configs.checkpoints_every_epoch == 0:
            accelerator.wait_for_everyone()
            # Set the path to save the models checkpoint.
            model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
            save_checkpoint(
                epoch,
                model_path,
                net,
                optimizer,
                accelerator,
                global_step=global_step,
                configs=configs,
                scheduler=scheduler,
            )
            logging.info(f'\tcheckpoint models in {model_path}')
            accelerator.wait_for_everyone()

        if epoch % configs.valid_settings.do_every_epochs == 0:
            accelerator.wait_for_everyone()
            start_time = time()
            evaluation_reports = evaluation_loop(
                net, valid_loader, epoch,
                accelerator=accelerator,
                configs=configs,
                mode='validation',
                logging=logging, global_step=global_step,
                writer=valid_writer, result_path=result_path,
                tokens_seen_total=tokens_seen_total,
            )
            end_time = time()
            evaluation_time = end_time - start_time
            accelerator.wait_for_everyone()
            eval_metric_segments = []
            eval_items = [
                ("loss", evaluation_reports.get("loss"), ".4f"),
                ("perplexity", evaluation_reports.get("perplexity"), ".2f"),
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
                    'perplexity': evaluation_reports['perplexity'],
                    'tokens_seen_total': tokens_seen_total,
                })

                # sync processes before saving best model
                accelerator.wait_for_everyone()
                # Set the path to save the model checkpoint.
                model_path = os.path.join(checkpoint_path, f'best_valid.pth')
                save_checkpoint(
                    epoch,
                    model_path,
                    net,
                    optimizer,
                    accelerator,
                    global_step=global_step,
                    configs=configs,
                    scheduler=scheduler,
                )
                logging.info(f'\tsaving the best models in {model_path}')
                logging.info(
                    f'\tbest validation {best_metric_key}: {best_valid_metrics[best_metric_key]:.4f}, '
                    f'tokens {best_valid_metrics["tokens_seen_total"]:,.0f}'
                )
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        train_writer.close()
        valid_writer.close()

    # best_valid_metrics summary
    logging.info(
        "best validation metrics: "
        f"loss {best_valid_metrics['loss']:.4f}, "
        f"perplexity {best_valid_metrics['perplexity']:.2f}, "
        f"tokens {best_valid_metrics['tokens_seen_total']:,.0f}"
    )

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
    parser = argparse.ArgumentParser(description="Train an autoregressive transformers to learn the language of protein.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./configs/config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
