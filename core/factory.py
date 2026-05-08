import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from transformers import get_scheduler as _hf_get_scheduler  # type: ignore

from core.config import Config
from model.model import WaddingtonModel
from trainer.trainer import Trainer
from core.dataset import WaddingtonDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model: WaddingtonModel, config: Config):
    decay_weights = []
    no_decay_weights = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim == 1 or "bias" in name:
                no_decay_weights.append(param)
            else:
                decay_weights.append(param)

    return torch.optim.AdamW(
        [
            {"params": no_decay_weights, "weight_decay": 0.0},
            {"params": decay_weights, "weight_decay": config.training.weight_decay},
        ],
        lr=config.training.lr,
        betas=config.training.betas,
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
    num_training_steps: int,
):
    return _hf_get_scheduler(
        name=config.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=num_training_steps,
    )


def build_trainer(config: Config):
    set_seed(config.project.seed)
    device = get_device()

    dataset = WaddingtonDataset(
        file_path="data/pancreas.h5ad",
        block_size=config.data.block_size,
        n_pcs=config.data.input_cell_dim,
    )
    model = WaddingtonModel(config=config, n_clusters=dataset.n_clusters).to(device)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.project.seed),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )

    optimizer = get_optimizer(model, config)

    steps_per_epoch = len(train_loader) // config.training.grad_accum_steps
    num_training_steps = steps_per_epoch * config.training.epochs

    scheduler = get_scheduler(optimizer, config, num_training_steps)

    trainer = Trainer(
        config=config,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )

    return trainer
