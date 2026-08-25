from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler, autocast
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

import config
import dataset
import utils

logger = utils.setup_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the FER2013 emotion recognition model.")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--model", type=str, default=config.MODEL_NAME,
                   choices=["efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large", "resnet18"])
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def build_optimizer(model: nn.Module, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.OPTIMIZER == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    return torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)


def build_scheduler(optimizer, epochs: int):
    if config.SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.MIN_LR)
    if config.SCHEDULER == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return None


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(mode=train)
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"

    with context:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for images, targets in pbar:
            images, targets = images.to(config.DEVICE, non_blocking=True), targets.to(config.DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                if train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    optimizer.step()

            preds = outputs.argmax(dim=1)
            batch_acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(batch_acc, images.size(0))
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg, all_preds, all_targets


def main() -> None:
    args = parse_args()
    utils.set_seed(args.seed)

    logger.info("=" * 70)
    logger.info("FER2013 Emotion Recognition — Training")
    logger.info("=" * 70)
    logger.info(f"Device: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        data_dir = Path(args.data_dir) if args.data_dir else dataset.locate_dataset()
        dataset.print_dataset_statistics(data_dir, run_corruption_check=True)
        train_loader, val_loader, _, class_names = dataset.get_dataloaders(
            data_dir=data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.workers
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Classes (dataloader order): {class_names}")
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = utils.build_model(args.model, num_classes=len(class_names), pretrained=config.PRETRAINED)
    model.to(config.DEVICE)

    if config.FREEZE_BACKBONE_EPOCHS > 0:
        utils.set_backbone_trainable(model, args.model, trainable=False)
        logger.info(f"Backbone frozen for the first {config.FREEZE_BACKBONE_EPOCHS} epoch(s) (head-only warmup).")

    class_weights = dataset.get_class_weights_tensor(data_dir).to(config.DEVICE) if config.USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)

    optimizer = build_optimizer(model, args.lr)
    scheduler = build_scheduler(optimizer, args.epochs)
    use_amp = config.DEVICE.type == "cuda" and not args.no_amp
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None
    logger.info(f"Mixed precision (AMP): {use_amp}")

    early_stopper = utils.EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, min_delta=config.EARLY_STOPPING_MIN_DELTA, mode="max"
    )
    writer = SummaryWriter(log_dir=str(config.LOG_DIR / "tensorboard"))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "lr": []}
    start_epoch = 0
    best_metric = -float("inf")

    if args.resume and config.LAST_MODEL_PATH.exists():
        logger.info(f"Resuming from {config.LAST_MODEL_PATH} ...")
        ckpt = utils.load_checkpoint(config.LAST_MODEL_PATH, model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = ckpt.get("best_metric", best_metric)
        history = ckpt.get("history", history)
        logger.info(f"Resumed at epoch {start_epoch}, best {config.MONITOR_METRIC} so far: {best_metric:.4f}")

    interrupted = {"flag": False}

    def _handle_sigint(signum, frame):
        logger.warning("KeyboardInterrupt received — saving last.pt before exiting...")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        for epoch in range(start_epoch, args.epochs):
            if interrupted["flag"]:
                break

            if epoch == config.FREEZE_BACKBONE_EPOCHS:
                utils.set_backbone_trainable(model, args.model, trainable=True)
                logger.info("Backbone unfrozen — fine-tuning full network.")

            t0 = time.time()
            train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, train=True)
            val_loss, val_acc, val_preds, val_targets = run_epoch(model, val_loader, criterion, optimizer, scaler, config.DEVICE, train=False)
            epoch_time = time.time() - t0

            metrics = utils.compute_metrics(val_targets, val_preds, class_names)
            current_lr = optimizer.param_groups[0]["lr"]

            if scheduler is not None:
                if config.SCHEDULER == "plateau":
                    scheduler.step(metrics["f1_macro"])
                else:
                    scheduler.step()

            eta_seconds = epoch_time * (args.epochs - epoch - 1)
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={metrics['f1_macro']:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s | ETA={eta_seconds / 60:.1f}min"
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(metrics["f1_macro"])
            history["lr"].append(current_lr)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("f1/val", metrics["f1_macro"], epoch)
            writer.add_scalar("lr", current_lr, epoch)

            utils.append_history_row(config.HISTORY_CSV_PATH, {
                "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc, "val_f1": metrics["f1_macro"],
                "lr": current_lr, "epoch_time_sec": epoch_time,
            })

            monitor_value = metrics["f1_macro"] if config.MONITOR_METRIC == "val_f1" else val_acc
            is_best = early_stopper.step(monitor_value)
            if is_best:
                best_metric = monitor_value

            ckpt_state = {
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_metric": best_metric,
                "class_names": class_names,
                "history": history,
                "config_img_size": args.img_size,
            }
            utils.save_checkpoint(ckpt_state, config.LAST_MODEL_PATH)
            if is_best:
                utils.save_checkpoint(ckpt_state, config.BEST_MODEL_PATH)
                logger.info(f"  -> New best model saved ({config.MONITOR_METRIC}={best_metric:.4f})")

            if early_stopper.should_stop:
                logger.info(f"Early stopping triggered (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    finally:
        writer.close()
        utils.plot_training_history(history, config.LOG_DIR / "training_curves.png")

        final_metrics = {
            "best_" + config.MONITOR_METRIC: best_metric,
            "final_epoch": len(history["train_loss"]) + start_epoch,
            "class_names": class_names,
        }
        with open(config.METRICS_PATH, "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Training curves saved to {config.LOG_DIR / 'training_curves.png'}")
        logger.info(f"Metrics saved to {config.METRICS_PATH}")
        logger.info(f"Best model: {config.BEST_MODEL_PATH}")
        logger.info(f"Last model: {config.LAST_MODEL_PATH}")

        if interrupted["flag"]:
            logger.warning("Training was interrupted. Re-run with --resume to continue from the last epoch.")
            sys.exit(130)


if __name__ == "__main__":
    main()
