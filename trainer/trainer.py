import torch
import wandb
from core.config import Config
from model.model import WaddingtonModel
from torch.amp import GradScaler, autocast  # type:ignore
from pathlib import Path
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        model: WaddingtonModel,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ):
        self.config = config
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.use_amp = device.type == "cuda"
        self.scaler = GradScaler(device=device.type, enabled=self.use_amp)
        self.grad_accum = config.training.grad_accum_steps
        self.grad_clip = config.training.gradient_clip

        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.global_step = 0

        self.quantile_values = torch.tensor(config.model.quantiles, device=device).view(
            1, 1, 1, -1
        )

    def pinball_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = targets.unsqueeze(-1) - preds
        loss = torch.max(
            self.quantile_values * errors,
            (self.quantile_values - 1) * errors,
        )
        return loss.mean()

    def cluster_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
        )

    def compute_loss(
        self,
        output: dict,
        y_state: torch.Tensor,
        y_cluster: torch.Tensor,
    ):
        loss_flow = self.pinball_loss(output["quantile_preds"], y_state)
        loss_cluster = self.cluster_loss(output["cluster_logits"], y_cluster)
        return loss_flow + loss_cluster, loss_flow, loss_cluster

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_flow = 0.0
        total_cluster = 0.0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(pbar):
            x = batch["x"].to(self.device, non_blocking=True)
            y_state = batch["next_state"].to(self.device, non_blocking=True)
            y_cluster = batch["clusters"].to(self.device, non_blocking=True)
            pseudotime = batch["pseudotime"].to(self.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                output = self.model(x, pseudotime)
                loss, loss_flow, loss_cluster = self.compute_loss(
                    output, y_state, y_cluster
                )
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum == 0 or (i + 1) == len(self.train_dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss_flow.item() + loss_cluster.item()
            total_flow += loss_flow.item()
            total_cluster += loss_cluster.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss_flow.item() + loss_cluster.item():.4f}",
                    "flow": f"{loss_flow.item():.4f}",
                    "cluster": f"{loss_cluster.item():.4f}",
                }
            )

        n = len(self.train_dataloader)
        return total_loss / n, total_flow / n, total_cluster / n

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_flow = 0.0
        total_cluster = 0.0

        for batch in self.val_dataloader:
            x = batch["x"].to(self.device, non_blocking=True)
            y_state = batch["next_state"].to(self.device, non_blocking=True)
            y_cluster = batch["clusters"].to(self.device, non_blocking=True)
            pseudotime = batch["pseudotime"].to(self.device, non_blocking=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                output = self.model(x, pseudotime)
                loss, loss_flow, loss_cluster = self.compute_loss(
                    output, y_state, y_cluster
                )

            total_loss += loss.item()
            total_flow += loss_flow.item()
            total_cluster += loss_cluster.item()

        n = len(self.val_dataloader)
        return total_loss / n, total_flow / n, total_cluster / n

    def save_checkpoint(self, epoch: int, name: str = "latest"):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        return ckpt["epoch"]

    def train(self, resume: str | None = None):
        wandb.init(
            project=self.config.project.name,
            config=self.config.model_dump(),
            name=f"{self.config.project.name}_v{self.config.project.version}",
        )

        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint(resume) + 1

        for epoch in range(start_epoch, self.config.training.epochs):
            train_loss, train_flow, train_cluster = self.train_epoch(epoch)
            val_loss, val_flow, val_cluster = self.validate()

            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/flow_loss": train_flow,
                    "train/cluster_loss": train_cluster,
                    "val/loss": val_loss,
                    "val/flow_loss": val_flow,
                    "val/cluster_loss": val_cluster,
                    "lr": self.scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, "best")

            self.save_checkpoint(epoch, "latest")

        wandb.finish()
