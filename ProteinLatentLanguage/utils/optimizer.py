import numpy as np
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from timm import optim


def load_opt(model, config, logging):
    """Loads an optimizer based on the provided configuration.

    Args:
        model: The model for which the optimizer is to be created.
        config (Box): A configuration object containing optimizer settings.
        logging: A logging object for logging information.

    Returns:
        The optimizer.
    """
    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam':
        if config.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            opt = bitsandbytes.optim.AdamW8bit(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps),
            )
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
    optimizer = load_opt(net, configs, logging)
    scheduler = None
    if configs.optimizer.name.lower() != 'schedulerfree':
        whole_steps = np.ceil(
            num_train_samples / configs.train_settings.grad_accumulation
        ) * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts
        first_cycle_steps = np.ceil(whole_steps / configs.optimizer.decay.num_restarts)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.decay.min_lr,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)
    else:
        if kwargs['accelerator'].is_main_process:
            logging.info('Using scheduler free optimizer')

    return optimizer, scheduler
