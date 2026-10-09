from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train import MODEL_FILES, ROOT, load_data_module, load_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_FILES))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/test"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    data_module = load_data_module(args.data_dir)
    test_loader = data_module.build_test_dataloader(batch_size=args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    module = load_module(args.model)
    model, _ = module.create_training_components(int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    targets, predictions = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = module.forward_logits(model, x.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.cpu().tolist())
    result = {
        "model": args.model,
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, predictions, average="macro", zero_division=0)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
