from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from sklearn.metrics import classification_report
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

import config
import dataset
import utils

logger = utils.setup_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH))
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = config.DEVICE

    try:
        model, model_name, _ = utils.load_model_for_inference(Path(args.checkpoint), device)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded '{model_name}' from {args.checkpoint}")

    data_dir = Path(args.data_dir) if args.data_dir else dataset.locate_dataset()
    train_loader, val_loader, test_loader, class_names = dataset.get_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size, img_size=args.img_size
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    all_preds, all_targets, all_confidences = [], [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"evaluating[{args.split}]", dynamic_ncols=True):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())
            all_confidences.extend(confidences.cpu().tolist())

    metrics = utils.compute_metrics(all_targets, all_preds, class_names)

    logger.info("=" * 70)
    logger.info(f"EVALUATION REPORT — split='{args.split}'  n={len(all_targets)}")
    logger.info("=" * 70)
    logger.info(f"Overall accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"Macro precision:      {metrics['precision_macro']:.4f}")
    logger.info(f"Macro recall:         {metrics['recall_macro']:.4f}")
    logger.info(f"Macro F1:             {metrics['f1_macro']:.4f}")
    logger.info(f"Mean prediction conf: {sum(all_confidences) / len(all_confidences):.4f}")
    logger.info("-" * 70)
    logger.info("Per-class accuracy:")
    for cls, acc in metrics["per_class_accuracy"].items():
        logger.info(f"  {cls:<10} {acc:.4f}")

    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    logger.info("-" * 70)
    logger.info("Classification report:\n" + report)

    cm_path = config.OUTPUT_DIR / f"confusion_matrix_{args.split}.png"
    utils.plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path, normalize=True)
    logger.info(f"Confusion matrix saved to {cm_path}")

    report_path = config.OUTPUT_DIR / f"evaluation_report_{args.split}.json"
    with open(report_path, "w") as f:
        json.dump({
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "per_class_accuracy": metrics["per_class_accuracy"],
            "classification_report": report,
        }, f, indent=2)
    logger.info(f"Full report saved to {report_path}")


if __name__ == "__main__":
    main()
