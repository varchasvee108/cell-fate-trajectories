from pathlib import Path
import tomllib
from pydantic import BaseModel, Field, ConfigDict


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    seed: int = Field(ge=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dataset: str
    batch_size: int = Field(gt=0)
    block_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    input_cell_dim: int = Field(gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_embd: int = Field(gt=0)
    dropout: float
    hidden_dim: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    quantiles: tuple[float, float, float]
    n_clusters: int = Field(gt=0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = Field(gt=0)
    betas: tuple[float, float]
    weight_decay: float = Field(ge=0)
    gradient_clip: float = Field(ge=0)
    scheduler: str
    eval_interval: int = Field(gt=0)
    save_interval: int = Field(gt=0)
    grad_accum_steps: int = Field(gt=0)
    max_steps: int = Field(ge=0)
    epochs: int = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    decay_steps: int = Field(ge=0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def load_config(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "rb") as f:
            toml_dict = tomllib.load(f)
        return cls.model_validate(toml_dict)
