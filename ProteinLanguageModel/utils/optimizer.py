import numpy as np
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from timm import optim


def _optimizer_requires_tensor_lr(optimizer) -> bool:
    return optimizer.__class__.__module__.startswith("torchao.optim")


def _set_group_lr(param_group, lr_value: float) -> None:
    current_lr = param_group.get("lr")
    if isinstance(current_lr, torch.Tensor):
        current_lr.fill_(float(lr_value))
    else:
        param_group["lr"] = float(lr_value)


def _coerce_optimizer_lrs_to_tensors(optimizer) -> None:
    for param_group in optimizer.param_groups:
        lr = param_group.get("lr")
        if isinstance(lr, torch.Tensor):
            continue
        if lr is None:
            continue
        param_group["lr"] = torch.tensor(float(lr), dtype=torch.float32)


class TensorLRSchedulerAdapter:
    """
    Ensure scheduler-updated learning rates stay Tensor-valued for optimizers
    that require it (e.g. torchao Adam*).
    """

    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(self, epoch=None):
        self.scheduler.step(epoch)
        _coerce_optimizer_lrs_to_tensors(self.optimizer)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
        _coerce_optimizer_lrs_to_tensors(self.optimizer)

    def __getattr__(self, name):
        return getattr(self.scheduler, name)


class CosineThenConstantScheduler:
    """
    A custom scheduler wrapper that performs cosine annealing up to a milestone step
    and then maintains a constant learning rate. This avoids compatibility issues
    between third-party schedulers and PyTorch's SequentialLR.
    """
    def __init__(self, cosine_scheduler, constant_lr, milestone_step):
        self.cosine_scheduler = cosine_scheduler
        self.constant_lr = constant_lr
        self.milestone_step = milestone_step
        self.optimizer = cosine_scheduler.optimizer
        self.current_step = 0

    def step(self, epoch=None):
        self.current_step += 1
        if self.current_step <= self.milestone_step:
            self.cosine_scheduler.step()
        else:
            for param_group in self.optimizer.param_groups:
                _set_group_lr(param_group, self.constant_lr)

    def state_dict(self):
        return {
            'cosine_scheduler': self.cosine_scheduler.state_dict(),
            'current_step': self.current_step,
            'constant_lr': self.constant_lr,
            'milestone_step': self.milestone_step
        }

    def load_state_dict(self, state_dict):
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])
        self.current_step = state_dict.get('current_step', 0)
        self.constant_lr = state_dict.get('constant_lr', self.constant_lr)
        self.milestone_step = state_dict.get('milestone_step', self.milestone_step)


def load_opt(model, config, logging, use_fsdp2=False):
    """Loads an optimizer based on the provided configuration.

    Args:
        model: The model for which the optimizer is to be created.
        config (Box): A configuration object containing optimizer settings.
        logging: A logging object for logging information.
        use_fsdp2: Whether FSDP2 is active. bitsandbytes 8-bit optimizers are
            incompatible with FSDP2's DTensor, so we fall back to
            torch.optim.AdamW(fused=True) when this is True.

    Returns:
        The optimizer.
    """
    use_8bit = bool(getattr(config.optimizer, "use_8bit_adam", False))
    use_4bit = bool(getattr(config.optimizer, "use_4bit_adam", False))
    low_bit_backend = str(getattr(config.optimizer, "low_bit_backend", "")).lower().strip()

    if use_8bit and use_4bit:
        raise ValueError("Only one of use_8bit_adam or use_4bit_adam can be True.")
    if (use_8bit or use_4bit) and low_bit_backend not in {"torchao", "bnb"}:
        raise ValueError(
            f"Invalid optimizer.low_bit_backend='{low_bit_backend}'. "
            "Set it explicitly to 'torchao' or 'bnb' when using low-bit Adam."
        )

    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam':
        if use_4bit and not use_fsdp2 and low_bit_backend == "torchao":
            try:
                import torchao.optim as ao_optim
                logging.info('use 4-bit adamw via torchao (DDP path; experimental)')
                opt = ao_optim.AdamW4bit(
                    model.parameters(), lr=float(config.optimizer.lr),
                    betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                    weight_decay=float(config.optimizer.weight_decay),
                    eps=float(config.optimizer.eps),
                )
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "optimizer.low_bit_backend='torchao' requires torchao low-bit optimizers "
                    "to be available."
                ) from exc
        elif use_4bit and not use_fsdp2 and low_bit_backend == "bnb":
            raise ValueError(
                "4-bit Adam is not available with optimizer.low_bit_backend='bnb'. "
                "Use optimizer.low_bit_backend='torchao' for 4-bit Adam."
            )
        elif (use_8bit or use_4bit) and use_fsdp2 and low_bit_backend == "torchao":
            try:
                import torchao.optim as ao_optim
                if use_4bit:
                    logging.info('use 4-bit adamw via torchao (FSDP2 path; more experimental than 8-bit)')
                    opt = ao_optim.AdamW4bit(
                        model.parameters(), lr=float(config.optimizer.lr),
                        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                        weight_decay=float(config.optimizer.weight_decay),
                        eps=float(config.optimizer.eps),
                    )
                else:
                    logging.info('use 8-bit adamw via torchao (FSDP2 path)')
                    opt = ao_optim.AdamW8bit(
                        model.parameters(), lr=float(config.optimizer.lr),
                        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                        weight_decay=float(config.optimizer.weight_decay),
                        eps=float(config.optimizer.eps),
                    )
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "optimizer.low_bit_backend='torchao' requires torchao low-bit optimizers "
                    "to be available."
                ) from exc
        elif (use_8bit or use_4bit) and use_fsdp2 and low_bit_backend == "bnb":
            raise ValueError(
                "bitsandbytes low-bit Adam is not supported with FSDP2. "
                "Use optimizer.low_bit_backend='torchao' or disable FSDP2."
            )
        elif use_8bit and not use_fsdp2 and low_bit_backend == "torchao":
            try:
                import torchao.optim as ao_optim
                logging.info('use 8-bit adamw via torchao (DDP path)')
                opt = ao_optim.AdamW8bit(
                    model.parameters(), lr=float(config.optimizer.lr),
                    betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                    weight_decay=float(config.optimizer.weight_decay),
                    eps=float(config.optimizer.eps),
                )
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "optimizer.low_bit_backend='torchao' requires torchao low-bit optimizers "
                    "to be available."
                ) from exc
        elif use_8bit and not use_fsdp2 and low_bit_backend == "bnb":
            try:
                import bitsandbytes
                logging.info('use 8-bit adamw via bitsandbytes')
                opt = bitsandbytes.optim.AdamW8bit(
                    model.parameters(), lr=float(config.optimizer.lr),
                    betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                    weight_decay=float(config.optimizer.weight_decay),
                    eps=float(config.optimizer.eps),
                )
            except (ImportError, RuntimeError) as exc:
                raise ImportError(
                    "optimizer.low_bit_backend='bnb' requires bitsandbytes AdamW8bit "
                    "to be available."
                ) from exc
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps)
            )

    elif config.optimizer.name.lower() == 'schedulerfree':
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=float(config.optimizer.lr),
                                             warmup_steps=config.optimizer.decay.warmup,
                                             betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                                             weight_decay=float(config.optimizer.weight_decay),
                                             eps=float(config.optimizer.eps))
    else:
        raise ValueError('wrong optimizer')
    return opt


def prepare_optimizer(net, configs, num_train_samples, logging, **kwargs):
    """Prepares the optimizer and learning rate scheduler.

    Available optimizers:
        - 'adabelief': AdaBelief optimizer.
        - 'adam': AdamW optimizer (can use 8-bit version if specified in config).
        - 'schedulerfree': AdamWScheduleFree optimizer (does not use a learning rate scheduler).

    Args:
        net: The neural network model.
        configs (Box): Configuration object containing training and optimizer settings.
        num_train_samples: The total number of training samples.
        logging: A logging object for logging information.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    from utils.utils import get_fsdp_config

    use_fsdp2 = bool(getattr(get_fsdp_config(configs), "enabled", False))
    optimizer = load_opt(net, configs, logging, use_fsdp2=use_fsdp2)
    scheduler = None
    if configs.optimizer.name.lower() != 'schedulerfree':
        total_epochs = configs.train_settings.num_epochs
        num_restarts = configs.optimizer.decay.num_restarts
        
        # Check if cosine should end early and hold at min_lr
        cosine_end_epoch = getattr(configs.optimizer.decay, 'cosine_end_epoch', None)
        
        if cosine_end_epoch is not None and cosine_end_epoch < total_epochs:
            # Cosine annealing ends at cosine_end_epoch, then constant min_lr
            # Use EXACTLY the same calculation as standard mode would with num_epochs=cosine_end_epoch
            cosine_whole_steps = np.ceil(
                num_train_samples / configs.train_settings.grad_accumulation
            ) * cosine_end_epoch / num_restarts
            cosine_steps = int(np.ceil(cosine_whole_steps / num_restarts))
            
            # Calculate remaining steps for constant phase
            total_whole_steps = np.ceil(
                num_train_samples / configs.train_settings.grad_accumulation
            ) * total_epochs
            constant_steps = int(total_whole_steps) - cosine_steps
            
            # Create the base cosine scheduler
            base_cosine_scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=cosine_steps,
                cycle_mult=1.0,
                max_lr=configs.optimizer.lr,
                min_lr=configs.optimizer.decay.min_lr,
                warmup_steps=configs.optimizer.decay.warmup,
                gamma=configs.optimizer.decay.gamma,
            )
            
            # Use our custom wrapper instead of SequentialLR to avoid compatibility issues
            scheduler = CosineThenConstantScheduler(
                base_cosine_scheduler,
                constant_lr=configs.optimizer.decay.min_lr,
                milestone_step=cosine_steps
            )
            
            if kwargs.get('accelerator') and kwargs['accelerator'].is_main_process:
                logging.info(
                    f'Scheduler: Cosine annealing for {cosine_end_epoch} epochs '
                    f'({cosine_steps} steps), then constant min_lr={configs.optimizer.decay.min_lr} '
                    f'for remaining {total_epochs - cosine_end_epoch} epochs ({constant_steps} steps)'
                )
        else:
            # Standard behavior: cosine spans all epochs
            whole_steps = np.ceil(
                num_train_samples / configs.train_settings.grad_accumulation
            ) * total_epochs / num_restarts
            first_cycle_steps = np.ceil(whole_steps / num_restarts)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=1.0,
                max_lr=configs.optimizer.lr,
                min_lr=configs.optimizer.decay.min_lr,
                warmup_steps=configs.optimizer.decay.warmup,
                gamma=configs.optimizer.decay.gamma,
            )
            
            if kwargs.get('accelerator') and kwargs['accelerator'].is_main_process:
                logging.info(
                    f'Scheduler: Cosine annealing for all {total_epochs} epochs '
                    f'({int(first_cycle_steps)} steps per cycle)'
                )
    else:
        if kwargs['accelerator'].is_main_process:
            logging.info('Using scheduler free optimizer')

    if scheduler is not None and _optimizer_requires_tensor_lr(optimizer):
        _coerce_optimizer_lrs_to_tensors(optimizer)
        scheduler = TensorLRSchedulerAdapter(scheduler, optimizer)

    return optimizer, scheduler
