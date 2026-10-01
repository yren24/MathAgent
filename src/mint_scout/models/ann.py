from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from mint_scout.invariants.manifest import stable_hash


@dataclass(frozen=True)
class ANNConfig:
    """Fixed, diagnostic MLP configuration derived from the legacy PLBind ANN."""

    config_id: str = "plbind_probe_ann_v1"
    hidden_layers: tuple[int, ...] = (256, 64)
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 5.0e-4
    weight_decay: float = 0.05
    dropout: float = 0.10
    random_seeds: tuple[int, ...] = (42, 1, 2)
    normalize: str = "StandardScaler"
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_layers", tuple(int(value) for value in self.hidden_layers))
        object.__setattr__(self, "random_seeds", tuple(int(value) for value in self.random_seeds))
        if not self.hidden_layers or any(value < 1 for value in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be positive and batch_size must be at least 2")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.random_seeds:
            raise ValueError("random_seeds cannot be empty")
        if self.normalize not in {"StandardScaler", "none"}:
            raise ValueError("normalize must be StandardScaler or none")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    @property
    def parameter_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class ANNOOFPredictionArtifact:
    sample_ids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    fold_ids: Mapping[str, int]
    config: ANNConfig
    input_dimension: int
    parameter_count: int
    device: str
    device_name: str
    torch_version: str
    cuda_version: str | None
    fit_seconds: float


def run_oof_ann(
    features: np.ndarray,
    targets: np.ndarray,
    sample_ids: Sequence[str],
    fold_ids: Mapping[str, int],
    *,
    config: ANNConfig = ANNConfig(),
) -> ANNOOFPredictionArtifact:
    ids = tuple(sample_ids)
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim < 2:
        raise ValueError("features must have a sample axis and at least one feature axis")
    if x.shape[0] != len(ids):
        raise ValueError("features, targets, and sample_ids must be aligned")
    x = x.reshape(len(ids), -1)
    if y.shape != (len(ids),):
        raise ValueError("features, targets, and sample_ids must be aligned")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contains duplicates")
    if set(fold_ids) != set(ids):
        raise ValueError("fold_ids must contain exactly the sample_ids")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and targets must be finite")

    torch = _require_torch()
    device = _resolve_device(torch, config.device)
    y_pred = np.full(len(ids), np.nan, dtype=float)
    prediction_counts = np.zeros(len(ids), dtype=int)
    parameter_count: int | None = None
    started = time.perf_counter()
    for fold in sorted({int(fold_ids[sample_id]) for sample_id in ids}):
        val_idx = np.asarray([index for index, sample_id in enumerate(ids) if fold_ids[sample_id] == fold])
        train_idx = np.asarray([index for index, sample_id in enumerate(ids) if fold_ids[sample_id] != fold])
        if len(val_idx) == 0 or len(train_idx) < 2:
            raise ValueError(f"Fold {fold} has an empty validation set or fewer than two training samples")
        x_train, x_val = _scale_fold(x[train_idx], x[val_idx], normalize=config.normalize)
        run_predictions = []
        for seed in config.random_seeds:
            prediction, run_parameter_count = _fit_predict_one(
                x_train,
                y[train_idx],
                x_val,
                config=config,
                seed=seed,
                device=device,
                torch=torch,
            )
            if parameter_count is None:
                parameter_count = run_parameter_count
            elif parameter_count != run_parameter_count:
                raise AssertionError("ANN parameter count changed between folds or seeds")
            run_predictions.append(prediction)
        y_pred[val_idx] = np.mean(np.asarray(run_predictions), axis=0)
        prediction_counts[val_idx] += 1

    if not np.all(prediction_counts == 1) or not np.all(np.isfinite(y_pred)):
        raise AssertionError("Every sample must receive exactly one finite OOF prediction")
    return ANNOOFPredictionArtifact(
        sample_ids=ids,
        y_true=y,
        y_pred=y_pred,
        fold_ids=dict(fold_ids),
        config=config,
        input_dimension=int(x.shape[1]),
        parameter_count=int(parameter_count or 0),
        device=device,
        device_name=(torch.cuda.get_device_name(0) if device == "cuda" else "CPU"),
        torch_version=str(torch.__version__),
        cuda_version=(str(torch.version.cuda) if torch.version.cuda is not None else None),
        fit_seconds=float(time.perf_counter() - started),
    )


def _scale_fold(
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    normalize: str,
) -> tuple[np.ndarray, np.ndarray]:
    if normalize == "StandardScaler":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        return scaler.fit_transform(x_train), scaler.transform(x_val)
    if normalize == "none":
        return np.asarray(x_train, dtype=float), np.asarray(x_val, dtype=float)
    raise ValueError(f"Unsupported normalize={normalize!r}")


def _fit_predict_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    config: ANNConfig,
    seed: int,
    device: str,
    torch,
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(config.deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    nn = torch.nn

    class FixedMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            modules = []
            width = int(x_train.shape[1])
            for hidden_width in config.hidden_layers:
                linear = nn.Linear(width, hidden_width, bias=False)
                nn.init.xavier_uniform_(linear.weight)
                modules.extend((linear, nn.BatchNorm1d(hidden_width), nn.ReLU()))
                if config.dropout > 0.0:
                    modules.append(nn.Dropout(config.dropout))
                width = hidden_width
            self.hidden = nn.Sequential(*modules)
            self.output = nn.Linear(width, 1, bias=True)
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, values):
            return self.output(self.hidden(values)).reshape(-1)

    model = FixedMLP().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_x = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    train_y = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    drop_last = len(dataset) > config.batch_size and len(dataset) % config.batch_size == 1
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(loader),
        pct_start=0.3,
    )
    criterion = nn.MSELoss()
    for _ in range(config.epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

    model.eval()
    with torch.no_grad():
        values = torch.as_tensor(np.asarray(x_val, dtype=np.float32), device=device)
        prediction = model(values).detach().cpu().numpy().astype(float, copy=False)
    return prediction, int(parameter_count)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the ANN diagnostic; load the Sapelo2 PyTorch module "
            "or install the optional mint-agent[ann] dependencies"
        ) from exc
    return torch


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ANN config requested CUDA, but CUDA is not available")
    return requested
