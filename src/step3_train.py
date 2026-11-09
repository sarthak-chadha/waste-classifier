"""
Step 3: Model training pipeline for EfficientNet-B4 on waste classification.

Experimental Configurations:
    1. 'base':
       - Initializes with ImageNet-1K pretrained weights.
       - Freezes all convolutional feature extractor parameters.
       - Trains only the custom 10-class linear classification head for 10 epochs.
    2. 'control':
       - Continues directly from the best 'base' checkpoint.
       - Keeps the convolutional backbone frozen for 15 additional epochs.
       - Serves as the scientific control to decouple extended training time from representation learning.
    3. 'finetune':
       - Continues directly from the best 'base' checkpoint.
       - Unfreezes all convolutional layers with differential learning rates (1e-4 backbone, 1e-3 head).
       - Trains for 15 epochs to adapt low-level and high-level representations to municipal waste imagery.
    4. 'all':
       - Executes base -> control -> finetune sequentially under matched seeds and oversampling draws.

Optimization & Regularization:
    - Optimizer: AdamW with weight decay 0.01.
    - Learning rate schedule: Linear warmup for 1 epoch followed by cosine decay.
    - Loss function: CrossEntropyLoss with 0.1 label smoothing to mitigate overconfidence.
    - Mixed precision: bfloat16 mixed-precision acceleration with zero scale loss.
    - Gradient management: Gradient clipping at norm 1.0; optional gradient accumulation steps.
    - Selection metric: Validation Macro-F1 (test split remains strictly untouched).

Outputs per run (stored in outputs/runs/<mode>/):
    - best.pt: Serialized PyTorch state dict and class name mappings.
    - log.csv: Epoch-by-epoch loss, accuracy, Macro-F1, and learning rate telemetry.
    - curves.png: Training and validation loss/metric curves.
    - summary.json: Hardware utilization, training duration, and peak memory metrics.

Usage:
    python src/step3_train.py --mode all --smoke      # Fast verification dry run
    python src/step3_train.py --mode base
    python src/step3_train.py --mode control
    python src/step3_train.py --mode finetune
    python src/step3_train.py --mode all
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4
from tqdm import tqdm

from step2_dataset import build_dataloaders

ROOT = Path(__file__).resolve().parent.parent


def lr_factor(step, warmup_steps, total_steps, start=0.01, floor=0.01):
    """
    Computes learning rate multiplier with linear warmup followed by cosine annealing decay.
    
    Args:
        step: Current global optimizer step.
        warmup_steps: Number of steps over which learning rate ramps up from `start` to 1.0.
        total_steps: Total optimizer steps across the run.
        start: Starting fraction of peak learning rate during warmup.
        floor: Minimum learning rate multiplier at the conclusion of cosine decay.
    """
    if step < warmup_steps:
        return start + (1 - start) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    # Clamp progress between [0, 1] to prevent cosine oscillation past final step
    progress = min(1.0, max(0.0, progress))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def mode_config(mode, args):
    """Returns training parameters and epoch offset mappings for the specified configuration."""
    if mode == "base":
        return dict(epochs=args.base_epochs, train_backbone=False,
                    init="imagenet", epoch_offset=0)
    # Control and Finetune share identical epoch offsets so their dynamic oversampling draws match
    if mode == "control":
        return dict(epochs=args.extra_epochs, train_backbone=False,
                    init="base", epoch_offset=args.base_epochs)
    if mode == "finetune":
        return dict(epochs=args.extra_epochs, train_backbone=True,
                    init="base", epoch_offset=args.base_epochs)
    raise ValueError(mode)


def build_model(num_classes, pretrained):
    """
    Instantiates EfficientNet-B4 and attaches a custom classification head.
    
    EfficientNet-B4 natively includes Dropout(0.4) preceding the linear projection.
    Only the final linear layer is replaced to match the target class count.
    """
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    """Configures gradient computation flags for the convolutional feature extractor."""
    for p in model.features.parameters():
        p.requires_grad = trainable
    for p in model.classifier.parameters():
        p.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    amp, accum_steps, freeze_backbone, max_batches, desc):
    """
    Executes one training epoch with mixed precision, gradient clipping, and accumulation.
    
    Note: When freeze_backbone is True, model.features is explicitly maintained in eval mode
    so running BatchNorm statistics (mean and variance) remain frozen to ImageNet distributions.
    """
    model.train()
    if freeze_backbone:
        model.features.eval()

    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    steps = n_batches // accum_steps
    n_batches = steps * accum_steps

    params = [p for p in model.parameters() if p.requires_grad]
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    seen = 0

    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, total=n_batches, desc=desc, leave=False,
               disable=not sys.stdout.isatty())
    for i, (x, y) in enumerate(bar):
        if i >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        loss = criterion(out.float(), y)
        (loss / accum_steps).backward()

        if (i + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_sum += loss.detach() * y.size(0)
        correct += (out.argmax(1) == y).sum()
        seen += y.size(0)

    return loss_sum.item() / seen, correct.item() / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, max_batches):
    """
    Evaluates model on validation batches and calculates loss, top-1 accuracy, and Macro-F1.
    """
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        loss_sum += criterion(out.float(), y).item() * y.size(0)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return (loss_sum / len(labels), float((preds == labels).mean()),
            float(f1_score(labels, preds, average="macro", zero_division=0)))


def save_curves(rows, path, title):
    """Plots and saves loss and accuracy/Macro-F1 progression charts across training epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="val")
    axes[0].set_title("Loss")

    # Training accuracy reflects heavy stochastic augmentation; validation reflects clean images
    axes[1].plot(epochs, [r["train_acc"] for r in rows],
                 label="train accuracy", linestyle="--", alpha=0.7)
    axes[1].plot(epochs, [r["val_acc"] for r in rows], label="val accuracy")
    axes[1].plot(epochs, [r["val_macro_f1"] for r in rows], label="val macro-F1")
    axes[1].set_title("Accuracy and macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_mode(mode, args, runs_dir, device):
    """
    Orchestrates end-to-end training execution for a single model configuration.
    
    Handles model instantiation, optimizer differential learning rates, learning rate scheduler,
    training loops, checkpoint persistence on validation Macro-F1 improvement, and metric summaries.
    """
    cfg = mode_config(mode, args)
    out_dir = runs_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = torch.cuda.is_bf16_supported()

    data = build_dataloaders(batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             size=args.size, seed=args.seed)
    class_names = data["class_names"]
    train_loader, val_loader = data["train"], data["val"]
    sampler = data["sampler"]

    if cfg["init"] == "imagenet":
        model = build_model(len(class_names), pretrained=True)
    else:
        base_ckpt = runs_dir / "base" / "best.pt"
        if not base_ckpt.exists():
            sys.exit(f"{base_ckpt} not found. Run --mode base first.")
        model = build_model(len(class_names), pretrained=False)
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded base checkpoint (epoch {ckpt['epoch']}, "
              f"val macro-F1 {ckpt['val_macro_f1']:.4f})")
    model.to(device)

    set_backbone_trainable(model, cfg["train_backbone"])
    groups = [{"params": model.classifier.parameters(), "lr": args.head_lr}]
    if cfg["train_backbone"]:
        groups.insert(0, {"params": model.features.parameters(),
                          "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    n_train_batches = len(train_loader)
    if args.max_train_batches is not None:
        n_train_batches = min(n_train_batches, args.max_train_batches)
    steps_per_epoch = n_train_batches // args.accum_steps
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: lr_factor(s, warmup_steps=steps_per_epoch,
                            total_steps=total_steps))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 64}\nRun: {mode}   epochs: {cfg['epochs']}   "
          f"backbone {'TRAINABLE' if cfg['train_backbone'] else 'frozen'}   "
          f"trainable params: {n_trainable / 1e6:.2f}M\n{'=' * 64}")
    print(f"{'ep':>3}{'tr_loss':>9}{'tr_acc':>8}{'va_loss':>9}{'va_acc':>8}"
          f"{'va_F1':>8}{'lr':>10}{'sec':>7}")

    rows, best_f1, best_epoch = [], -1.0, 0
    torch.cuda.reset_peak_memory_stats()
    run_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        # Synchronize epoch counter so oversampling draws match across comparable runs
        sampler.set_epoch(cfg["epoch_offset"] + epoch - 1)
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, amp,
            args.accum_steps, not cfg["train_backbone"],
            args.max_train_batches, f"{mode} {epoch}/{cfg['epochs']}")
        va_loss, va_acc, va_f1 = evaluate(
            model, val_loader, criterion, device, amp, args.max_val_batches)
        secs = time.time() - t0
        lr_now = optimizer.param_groups[-1]["lr"]

        marker = ""
        if va_f1 > best_f1:
            best_f1, best_epoch, marker = va_f1, epoch, "  *"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": va_f1, "val_acc": va_acc,
                        "class_names": class_names}, out_dir / "best.pt")
        rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                         val_loss=va_loss, val_acc=va_acc, val_macro_f1=va_f1,
                         head_lr=lr_now, seconds=secs))
        print(f"{epoch:>3}{tr_loss:>9.4f}{tr_acc:>8.4f}{va_loss:>9.4f}"
              f"{va_acc:>8.4f}{va_f1:>8.4f}{lr_now:>10.2e}{secs:>7.0f}{marker}")

    total_secs = time.time() - run_start
    with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_curves(rows, out_dir / "curves.png", f"{mode}")

    best_row = rows[best_epoch - 1]
    summary = {
        "mode": mode,
        "epochs": cfg["epochs"],
        "backbone_trainable": cfg["train_backbone"],
        "trainable_params": n_trainable,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_row["val_macro_f1"],
        "best_val_acc": best_row["val_acc"],
        "train_seconds": round(total_secs, 1),
        "mean_epoch_seconds": round(total_secs / cfg["epochs"], 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "smoke_run": args.smoke,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Best epoch {best_epoch}: val macro-F1 {best_f1:.4f}   "
          f"time {total_secs / 60:.1f} min   "
          f"peak GPU memory {summary['peak_gpu_gb']} GB")

    del model, optimizer, train_loader, val_loader, data
    torch.cuda.empty_cache()
    return summary


def main():
    """Parses hyperparameters, seeds runtime environments, and runs model configurations."""
    p = argparse.ArgumentParser(description="Train EfficientNet-B4")
    p.add_argument("--mode", choices=["base", "control", "finetune", "all"],
                    required=True)
    p.add_argument("--base-epochs", type=int, default=10)
    p.add_argument("--extra-epochs", type=int, default=15,
                    help="epochs for control and finetune")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation steps (effective batch = batch-size x accum-steps)")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                    help="quick dry run: 2 epochs, limited batches, separate output folder")
    args = p.parse_args()

    args.max_train_batches = args.max_val_batches = None
    runs_dir = ROOT / "outputs" / "runs"
    if args.smoke:
        args.base_epochs = args.extra_epochs = 2
        args.max_train_batches, args.max_val_batches = 40, 15
        runs_dir = ROOT / "outputs" / "runs_smoke"

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required.")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    modes = ["base", "control", "finetune"] if args.mode == "all" else [args.mode]
    results = []
    try:
        for mode in modes:
            results.append(run_mode(mode, args, runs_dir, device))
    except torch.cuda.OutOfMemoryError:
        sys.exit("Out of GPU memory. Retry with: --batch-size 8 --accum-steps 2")

    if len(results) > 1:
        print(f"\n{'run':<10}{'best ep':>8}{'val F1':>9}{'val acc':>9}{'minutes':>9}")
        for r in results:
            print(f"{r['mode']:<10}{r['best_epoch']:>8}{r['best_val_macro_f1']:>9.4f}"
                  f"{r['best_val_acc']:>9.4f}{r['train_seconds'] / 60:>9.1f}")
        print("(validation numbers only; the test set is evaluated in Step 4)")


if __name__ == "__main__":
    main()
