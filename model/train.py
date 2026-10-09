from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30
MODEL_FILES = {
    "CNN1D-Gaurhar": "CNN1D-Gaurhar.py",
    "DeepBiCapsNet-Anitha": "DeepBiCapsNet-Anitha.py",
    "CNNBiLSTM-PrasannaVenkatesh": "CNNBiLSTM-PrasannaVenkatesh.py",
    "CNNAttention-Guhdar": "CNNAttention-Guhdar.py",
    "CycleCMCHA-Kavitha": "CycleCMCHA-Kavitha.py",
    "ASNet-Venkatesh": "ASNet-Venkatesh.py",
    "MSAForm-Tao": "MSAForm-Tao.py",
    "BTFNet-Ao": "BTFNet-Ao.py",
    "PACPVCCNNLSTM-Bashar": "PACPVCCNNLSTM-Bashar.py",
}


def load_module(name: str):
    path = MODEL_DIR / MODEL_FILES[name]
    spec = importlib.util.spec_from_file_location(f"comparison_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data_module(data_dir: Path):
    path = data_dir / "dataset.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset implementation not found: {path}. "
            "The data pipeline is intentionally not distributed with this repository."
        )
    spec = importlib.util.spec_from_file_location("local_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(module, model, loader, criterion, device: torch.device):
    model.eval()
    targets, predictions, losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = module.forward_and_loss(model, criterion, x, y)
            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, predictions, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_FILES))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=63540)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(MODEL_FILES))
        return
    if not args.model:
        parser.error("--model is required unless --list-models is used")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    module = load_module(args.model)
    config = dict(getattr(module, "TRAINING_CONFIG", {}))
    batch_size = args.batch_size or int(config.get("batch_size", 64))
    learning_rate = args.learning_rate or float(config.get("learning_rate", 1e-3))
    data_module = load_data_module(args.data_dir)
    train_loader, val_loader, num_classes, labels = data_module.build_dataloaders(
        batch_size=batch_size, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion = module.create_training_components(num_classes, labels)
    model.to(device)
    if hasattr(criterion, "to"):
        criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, epochs_without_improvement, history = -1.0, -1, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = module.forward_and_loss(model, criterion, x, y)
            loss.backward()
            optimizer.step()
        metrics = evaluate(module, model, val_loader, criterion, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = metrics["f1_macro"], epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "num_classes": num_classes,
                    "config": config,
                },
                args.output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "run.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_f1,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "epochs_completed": len(history),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"best_epoch": best_epoch, "best_val_f1_macro": best_f1, "epochs_completed": len(history), "checkpoint": str(args.output_dir / "best.pt")}, indent=2))


if __name__ == "__main__":
    main()
