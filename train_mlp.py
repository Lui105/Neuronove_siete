from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import matplotlib.pyplot as plt


DEFAULT_CONFIG = {
    "experiment_name": "mlp_stage1_sweep",
    "batch_size_train": 64,
    "batch_size_test": 256,
    "epochs": 80,
    "use_early_stopping": False,
    "early_stopping_patience": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "hidden_dims": [128, 64],
    "dropout_rate": 0.0,
    "use_batch_norm": False,
    "use_skip_connection": False,
    "loss_type": "bce",
    "focal_gamma": 2.0,
    "threshold": 0.5,
    "seed": 42,
}

DATA_PATH = Path("dataset_31_credit-g.arff")
PROJECT_NAME = "NSIETE_credit"
ENTITY = "lui-vitarius-slovak-university-of-technology"


def load_env_file(env_path=".env"):
    env_values = {}
    env_file = Path(env_path)
    if not env_file.exists():
        return env_values
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_values[key.strip()] = value.strip().strip('"').strip("'")
    return env_values


def setup_wandb_env():
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    env_values = load_env_file()
    wandb_api_key = (
        os.getenv("WANDB_API_KEY")
        or os.getenv("wandb_api")
        or env_values.get("WANDB_API_KEY")
        or env_values.get("wandb_api")
    )
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        wandb.login(key=wandb_api_key, relogin=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset():
    raw_data, _ = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(raw_data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
    df["target"] = (df["class"] == "bad").astype(int)
    return df


def prepare_data(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["class", "target"]]

    X = df.drop(columns=["class", "target"])
    y = df["target"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test


def make_loader(X_array, y_series, batch_size=64, shuffle=False):
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_series.to_numpy().reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall_bad": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "cost_total": int(fp * 1 + fn * 5),
        "cost_per_sample": (fp * 1 + fn * 5) / len(y_true),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, thresholds=None):
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 1.00, 0.05)
    best_metrics = None
    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        if (
            best_metrics is None
            or metrics["cost_total"] < best_metrics["cost_total"]
            or (
                metrics["cost_total"] == best_metrics["cost_total"]
                and metrics["recall_bad"] > best_metrics["recall_bad"]
            )
        ):
            best_metrics = metrics
    return best_metrics


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float, use_batch_norm: bool, use_skip_connection: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim) if use_batch_norm else None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.use_skip_connection = use_skip_connection
        self.projection = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.linear(x)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.use_skip_connection:
            out = out + self.projection(x)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class CreditMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout_rate: float = 0.0, use_batch_norm: bool = False, use_skip_connection: bool = False):
        super().__init__()
        blocks = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(
                MLPBlock(
                    prev_dim,
                    hidden_dim,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_skip_connection=use_skip_connection,
                )
            )
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output_layer(x)


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - pt).pow(self.gamma)
        loss = focal_factor * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def predict_probabilities(model, loader, device, criterion=None):
    model.eval()
    all_probs, all_targets = [], []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * len(xb)
                total_samples += len(xb)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    probs = np.vstack(all_probs).ravel()
    targets = np.vstack(all_targets).ravel().astype(int)
    avg_loss = total_loss / total_samples if total_samples > 0 else None
    return probs, targets, avg_loss


def build_run_name(run_config):
    name = (
        f"{run_config['experiment_name']}_dropout{run_config['dropout_rate']}"
        f"_hdims{'-'.join(map(str, run_config['hidden_dims']))}"
        f"_bn{int(bool(run_config['use_batch_norm']))}"
        f"_skip{int(bool(run_config['use_skip_connection']))}"
        f"_loss{run_config['loss_type']}"
        f"_es{int(bool(run_config.get('use_early_stopping', False)))}"
        f"_lr{run_config['learning_rate']}"
        f"_wd{run_config['weight_decay']}"
    )
    if run_config.get("use_early_stopping", False):
        name += f"_pat{run_config['early_stopping_patience']}"
    return name


def get_run_config():
    if os.getenv("EXPERIMENT_CONFIG_JSON"):
        overrides = json.loads(os.environ["EXPERIMENT_CONFIG_JSON"])
        return {**DEFAULT_CONFIG, **overrides}
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        config=DEFAULT_CONFIG,
        settings=wandb.Settings(init_timeout=300),
    )
    cfg = dict(run.config)
    wandb.finish()
    return cfg


def main():
    setup_wandb_env()
    run_config = get_run_config()
    set_seed(int(run_config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataset()
    X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test = prepare_data(df)

    train_loader = make_loader(X_train_processed, y_train, batch_size=run_config["batch_size_train"], shuffle=True)
    val_loader = make_loader(X_val_processed, y_val, batch_size=run_config["batch_size_test"], shuffle=False)
    test_loader = make_loader(X_test_processed, y_test, batch_size=run_config["batch_size_test"], shuffle=False)

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    pos_weight_value = num_neg / num_pos
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    model = CreditMLP(
        input_dim=X_train_processed.shape[1],
        hidden_dims=tuple(run_config["hidden_dims"]),
        dropout_rate=float(run_config["dropout_rate"]),
        use_batch_norm=bool(run_config["use_batch_norm"]),
        use_skip_connection=bool(run_config["use_skip_connection"]),
    ).to(device)

    if run_config["loss_type"] == "focal":
        criterion = BinaryFocalLossWithLogits(gamma=float(run_config["focal_gamma"]), pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
    )

    run_name = build_run_name(run_config)
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        name=run_name,
        config={
            **run_config,
            "input_dim": int(X_train_processed.shape[1]),
            "num_train": int(len(y_train)),
            "num_val": int(len(y_val)),
            "num_test": int(len(y_test)),
            "pos_weight": float(pos_weight_value),
            "device": str(device),
        },
        settings=wandb.Settings(init_timeout=300),
        reinit=True,
    )
    wandb.summary["run_name"] = run_name

    history = []
    best_state = None
    best_val_cost = np.inf
    best_val_metrics = None
    best_epoch = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, int(run_config["epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_prob, train_true, train_eval_loss = predict_probabilities(model, train_loader, device, criterion)
        val_prob, val_true, val_eval_loss = predict_probabilities(model, val_loader, device, criterion)
        train_metrics = compute_metrics(train_true, train_prob, threshold=float(run_config["threshold"]))
        val_metrics = compute_metrics(val_true, val_prob, threshold=float(run_config["threshold"]))

        row = {
            "epoch": epoch,
            "train_loss": train_eval_loss,
            "val_loss": val_eval_loss,
            "train_cost_per_sample": train_metrics["cost_per_sample"],
            "train_recall_bad": train_metrics["recall_bad"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_cost_per_sample": val_metrics["cost_per_sample"],
            "val_recall_bad": val_metrics["recall_bad"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
        }
        history.append(row)
        wandb.log(row)

        if val_metrics["cost_total"] < best_val_cost:
            best_val_cost = val_metrics["cost_total"]
            best_val_metrics = val_metrics.copy()
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            wandb.summary["best_epoch"] = int(best_epoch)
            wandb.summary["best_val_cost_total"] = int(best_val_metrics["cost_total"])
            wandb.summary["best_val_cost_per_sample"] = float(best_val_metrics["cost_per_sample"])
            wandb.summary["best_val_recall_bad"] = float(best_val_metrics["recall_bad"])
            wandb.summary["best_val_pr_auc"] = float(best_val_metrics["pr_auc"])
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_eval_loss:.4f} | val_loss={val_eval_loss:.4f} | "
            f"val_cost={val_metrics['cost_total']} | val_recall_bad={val_metrics['recall_bad']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

        if run_config.get("use_early_stopping", False) and epochs_without_improvement >= int(run_config["early_stopping_patience"]):
            stopped_early = True
            wandb.summary["stopped_early"] = True
            wandb.summary["stopped_epoch"] = int(epoch)
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if not stopped_early:
        wandb.summary["stopped_early"] = False
        wandb.summary["stopped_epoch"] = int(best_epoch if best_epoch is not None else run_config["epochs"])

    model.load_state_dict(best_state)

    val_prob, val_true, _ = predict_probabilities(model, val_loader, device, criterion)
    best_threshold_metrics = find_best_threshold(val_true, val_prob)
    best_threshold = best_threshold_metrics["threshold"]
    wandb.summary["best_threshold"] = float(best_threshold)
    wandb.summary["best_val_cost_total_tuned_threshold"] = int(best_threshold_metrics["cost_total"])
    wandb.summary["best_val_cost_per_sample_tuned_threshold"] = float(best_threshold_metrics["cost_per_sample"])
    wandb.summary["best_val_recall_bad_tuned_threshold"] = float(best_threshold_metrics["recall_bad"])
    wandb.summary["best_val_pr_auc_tuned_threshold"] = float(best_threshold_metrics["pr_auc"])

    test_prob, test_true, test_loss = predict_probabilities(model, test_loader, device, criterion)
    test_metrics = compute_metrics(test_true, test_prob, threshold=float(run_config["threshold"]))
    test_metrics_tuned = compute_metrics(test_true, test_prob, threshold=best_threshold)

    wandb.summary.update({
        "test_loss": float(test_loss),
        "test_cost_total": int(test_metrics["cost_total"]),
        "test_cost_per_sample": float(test_metrics["cost_per_sample"]),
        "test_recall_bad": float(test_metrics["recall_bad"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_fp": int(test_metrics["fp"]),
        "test_fn": int(test_metrics["fn"]),
        "test_tp": int(test_metrics["tp"]),
        "test_tn": int(test_metrics["tn"]),
        "test_cost_total_tuned_threshold": int(test_metrics_tuned["cost_total"]),
        "test_cost_per_sample_tuned_threshold": float(test_metrics_tuned["cost_per_sample"]),
        "test_recall_bad_tuned_threshold": float(test_metrics_tuned["recall_bad"]),
        "test_pr_auc_tuned_threshold": float(test_metrics_tuned["pr_auc"]),
        "test_fp_tuned_threshold": int(test_metrics_tuned["fp"]),
        "test_fn_tuned_threshold": int(test_metrics_tuned["fn"]),
        "test_tp_tuned_threshold": int(test_metrics_tuned["tp"]),
        "test_tn_tuned_threshold": int(test_metrics_tuned["tn"]),
        "pos_weight": float(pos_weight_value),
    })

    wandb.log({
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_true,
            preds=(test_prob >= best_threshold).astype(int),
            class_names=["good", "bad"],
        ),
    })

    precision, recall, _ = precision_recall_curve(test_true, test_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall curve")
    wandb.log({"pr_curve": wandb.Image(plt.gcf())})
    plt.close()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    safe_name = run_name.replace("/", "_")
    model_path = artifacts_dir / f"{safe_name}_best.pt"
    history_path = artifacts_dir / f"{safe_name}_training_history.csv"
    metadata_path = artifacts_dir / f"{safe_name}_experiment_metadata.json"

    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    metadata = {
        "run_config": run_config,
        "run_name": run_name,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_threshold": float(best_threshold),
        "best_val_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in best_val_metrics.items()} if best_val_metrics is not None else None,
        "best_val_metrics_tuned_threshold": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in best_threshold_metrics.items()},
        "test_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_metrics.items()},
        "test_metrics_tuned_threshold": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_metrics_tuned.items()},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    artifact = wandb.Artifact("credit-mlp-run", type="experiment")
    artifact.add_file(str(model_path))
    artifact.add_file(str(history_path))
    artifact.add_file(str(metadata_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
