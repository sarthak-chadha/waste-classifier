"""
Step 4: Held-out test set evaluation and statistical hypothesis testing.

Evaluation Integrity Protocol:
    1. The test split (1,840 unseen photos) must be evaluated strictly once.
       Hyperparameters or checkpoint selections must never be adjusted based on test metrics.
    2. Before evaluating on the test split, each checkpoint is evaluated against the validation split
       to confirm that the reproduced validation Macro-F1 matches the training log within 0.005.
       Any mismatch halts execution immediately to prevent corrupted evaluations.

Statistical Analysis & Outputs:
    - 95% Confidence Intervals: Calculated via paired non-parametric bootstrap resampling (2,000 iterations).
    - Exact McNemar Test: Two-sided paired exact test evaluates whether the discordant prediction
      distribution between any two models is statistically significant (testing H0: error rates are equal).
    - Per-Class Metrics: Precision, Recall, and F1 across all 10 waste categories.
    - Inference Benchmarking: GPU forward latency (median, 95th percentile) and batched throughput.
    - Visual Artifacts:
        * confusion_matrices.png: Side-by-side normalized confusion matrices (Base vs Fine-Tuned).
        * per_class_f1.png: 10-class grouped bar chart comparing Base, Control, and Fine-Tuned F1.
        * misclassifications.png: Visual gallery of model error cases with predicted confidences.

Usage:
    python src/step4_evaluate.py
    python src/step4_evaluate.py --gallery random --force
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from step2_dataset import build_dataloaders
from step3_train import build_model

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["base", "control", "finetune"]


def load_model(mode, class_names, runs_dir, device):
    """Loads model weights from checkpoint and verifies class name alignment."""
    ckpt_path = runs_dir / mode / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"{ckpt_path} not found. Train the {mode} model first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ckpt["class_names"] != class_names:
        sys.exit(f"Class order in {ckpt_path} does not match splits.")
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model, loader, device, amp):
    """Executes non-blocking batched forward passes and returns softmax probabilities and ground truth labels."""
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        probs.append(torch.softmax(out.float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def confusion(labels, preds, k):
    """Constructs a k x k confusion matrix where rows are ground-truth and columns are predicted."""
    return np.bincount(labels * k + preds, minlength=k * k).reshape(k, k)


def metrics_from_cm(cm):
    """
    Computes overall accuracy, Macro-F1, Weighted-F1, per-class precision, recall, and support
    directly from a k x k confusion matrix.
    """
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    zeros = np.zeros_like(tp)
    precision = np.divide(tp, predicted, out=zeros.copy(), where=predicted > 0)
    recall = np.divide(tp, support, out=zeros.copy(), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=zeros.copy(), where=denom > 0)
    n = cm.sum()
    return {
        "accuracy": tp.sum() / n,
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / n,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
    }


def bootstrap(labels, preds, k, n_boot, seed):
    """
    Executes paired non-parametric bootstrap resampling across test samples.
    Using identical resampled index draws across models preserves paired variance.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = rng.integers(0, n, size=(n_boot, n))
    pair = {m: labels * k + p for m, p in preds.items()}
    acc = {m: np.empty(n_boot) for m in preds}
    f1 = {m: np.empty(n_boot) for m in preds}
    for b in range(n_boot):
        for m in preds:
            cm = np.bincount(pair[m][idx[b]], minlength=k * k).reshape(k, k)
            r = metrics_from_cm(cm)
            acc[m][b], f1[m][b] = r["accuracy"], r["macro_f1"]
    return acc, f1


def ci(values):
    """Calculates non-parametric 95% confidence interval using 2.5% and 97.5% percentiles."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(correct_a, correct_b):
    """
    Calculates exact two-sided McNemar test for paired binary classification outcomes.
    
    Returns:
        (a_only, b_only, p_value): Counts of discordant predictions and exact binomial p-value.
    """
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
    return a_only, b_only, min(1.0, 2 * tail / 2 ** n)


@torch.no_grad()
def measure_speed(model, device, amp, size, batch=16):
    """
    Measures GPU forward inference latency (single image) and throughput (batched)
    incorporating CUDA warmup iterations and device synchronization.
    """
    def run(x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            model(x)

    single = torch.randn(1, 3, size, size, device=device)
    batched = torch.randn(batch, 3, size, size, device=device)
    for _ in range(10):
        run(single)
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        t = time.perf_counter()
        run(single)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    for _ in range(3):
        run(batched)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        run(batched)
    torch.cuda.synchronize()
    throughput = 20 * batch / (time.perf_counter() - t)
    return {
        "latency_ms_batch1_median": float(np.median(times)),
        "latency_ms_batch1_p95": float(np.percentile(times, 95)),
        f"throughput_img_per_s_batch{batch}": float(throughput),
    }


def plot_confusions(cms, titles, names, path):
    """Plots and serializes normalized confusion matrix heatmaps with percentage text overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cms), figsize=(8.5 * len(cms), 7.5))
    for ax, cm, title in zip(np.atleast_1d(axes), cms, titles):
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                if norm[i, j] >= 0.005:
                    ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center",
                            va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="% of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_class(f1_by_model, names, path):
    """Generates a grouped bar chart visualizing per-category F1 scores across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, f1) in enumerate(f1_by_model.items()):
        ax.bar(x + (i - 1) * width, f1, width, label=model)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(max(0.0, min(min(v) for v in f1_by_model.values()) - 0.05), 1.0)
    ax.set_title("Per-class F1 on the test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gallery(paths, labels, preds, probs, names, out_path, mode, seed, n=16):
    """Plots a 4x4 visual inspection gallery displaying misclassified images and confidence scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    wrong = np.where(preds != labels)[0]
    if len(wrong) == 0:
        return False
    conf = probs[wrong, preds[wrong]]
    if mode == "confident":
        chosen = wrong[np.argsort(-conf)[:n]]
        heading = "Fine-tuned model: most confident errors"
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
        heading = "Fine-tuned model: random errors"

    fig, axes = plt.subplots(4, 4, figsize=(11, 12))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, chosen):
        with Image.open(paths[i]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"true: {names[labels[i]]}\npred: {names[preds[i]]} "
                     f"({probs[i, preds[i]] * 100:.0f}%)",
                     fontsize=9, color="firebrick")
    fig.suptitle(f"{heading} ({len(wrong)} errors in total)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def main():
    """Main execution orchestrating validation reproduction, test evaluation, and statistical reporting."""
    parser = argparse.ArgumentParser(description="Test-set evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="bootstrap resamples for confidence intervals")
    parser.add_argument("--gallery", choices=["confident", "random"],
                        default="confident",
                        help="which fine-tuned errors to show")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing evaluation results")
    args = parser.parse_args()

    runs_dir = ROOT / "outputs" / "runs"
    out_dir = ROOT / "outputs" / "evaluation"
    if (out_dir / "results.json").exists() and not args.force:
        sys.exit(f"{out_dir / 'results.json'} already exists. The test set "
                 "should be evaluated once; use --force to overwrite.")
    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required.")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    amp = torch.cuda.is_bf16_supported()

    data = build_dataloaders(batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             size=args.size, seed=args.seed)
    names = data["class_names"]
    k = len(names)
    test_records = data["test"].dataset.records
    test_paths = [p for p, _ in test_records]
    print(f"Test images: {len(test_records)}   classes: {k}")

    probs, preds, ckpts, speed = {}, {}, {}, None
    labels = None
    print("\nCheckpoint check (validation macro-F1 vs value logged in training):")
    for mode in MODELS:
        model, ckpt = load_model(mode, names, runs_dir, device)
        ckpts[mode] = {"epoch": int(ckpt["epoch"]),
                       "val_macro_f1": float(ckpt["val_macro_f1"])}
        v_probs, v_labels = predict(model, data["val"], device, amp)
        v_f1 = metrics_from_cm(confusion(v_labels, v_probs.argmax(1), k))["macro_f1"]
        print(f"  {mode:<9}epoch {ckpt['epoch']:>2}   logged "
              f"{ckpt['val_macro_f1']:.4f}   reproduced {v_f1:.4f}")
        # A mismatch indicates corrupted weights or preprocessing differences; halt immediately
        if abs(v_f1 - ckpt["val_macro_f1"]) >= 0.005:
            sys.exit(f"Validation macro-F1 for '{mode}' does not match the "
                     "training log. Stopping before using the test set.")

        probs[mode], test_labels = predict(model, data["test"], device, amp)
        preds[mode] = probs[mode].argmax(1)
        if labels is None:
            labels = test_labels
        if mode == "finetune":
            # Models share identical architecture; benchmark hardware throughput once
            speed = measure_speed(model, device, amp, args.size)
        del model
        torch.cuda.empty_cache()

    n = len(labels)
    cms = {m: confusion(labels, preds[m], k) for m in MODELS}
    mets = {m: metrics_from_cm(cms[m]) for m in MODELS}

    acc_b, f1_b = bootstrap(labels, preds, k, args.n_boot, args.seed)

    print(f"\n{'=' * 78}\nOVERALL TEST RESULTS (n = {n}, 95% bootstrap CI)\n{'=' * 78}")
    print(f"{'model':<10}{'accuracy':>24}{'macro-F1':>24}{'weighted-F1':>14}")
    overall = {}
    for m in MODELS:
        a_lo, a_hi = ci(acc_b[m])
        f_lo, f_hi = ci(f1_b[m])
        r = mets[m]
        overall[m] = {
            "accuracy": float(r["accuracy"]), "accuracy_ci": [a_lo, a_hi],
            "macro_f1": float(r["macro_f1"]), "macro_f1_ci": [f_lo, f_hi],
            "weighted_f1": float(r["weighted_f1"]),
        }
        print(f"{m:<10}{r['accuracy'] * 100:>9.2f}% [{a_lo * 100:5.2f},{a_hi * 100:6.2f}]"
              f"{r['macro_f1']:>11.4f} [{f_lo:.4f},{f_hi:.4f}]"
              f"{r['weighted_f1']:>10.4f}")

    print(f"\n{'=' * 78}\nPAIRED COMPARISONS\n{'=' * 78}")
    print(f"{'comparison':<22}{'d macro-F1':>11}{'95% CI':>20}"
          f"{'only A ok':>11}{'only B ok':>11}{'McNemar p':>11}")
    pairs = [("finetune", "base"), ("finetune", "control"), ("control", "base")]
    comparisons = {}
    for a, b in pairs:
        diff = mets[a]["macro_f1"] - mets[b]["macro_f1"]
        lo, hi = ci(f1_b[a] - f1_b[b])
        a_only, b_only, p = mcnemar_exact(preds[a] == labels, preds[b] == labels)
        comparisons[f"{a}_vs_{b}"] = {
            "macro_f1_diff": float(diff), "macro_f1_diff_ci": [lo, hi],
            "a_only_correct": a_only, "b_only_correct": b_only,
            "mcnemar_p": p,
        }
        print(f"{a + ' vs ' + b:<22}{diff:>+11.4f}  [{lo:+.4f},{hi:+.4f}]"
              f"{a_only:>11}{b_only:>11}{p:>11.4g}")
    print("(A = first model, B = second. 'only A ok' counts images A gets "
          "right and B gets wrong.)")

    print(f"\n{'=' * 78}\nPER-CLASS TEST RESULTS\n{'=' * 78}")
    print(f"{'class':<11}{'n':>5}{'F1 base':>9}{'F1 ctrl':>9}{'F1 fine':>9}"
          f"{'gain':>8}{'  fine P':>9}{'fine R':>8}")
    ft, bs = mets["finetune"], mets["base"]
    for i, name in enumerate(names):
        print(f"{name:<11}{int(ft['support'][i]):>5}"
              f"{bs['f1'][i]:>9.3f}{mets['control']['f1'][i]:>9.3f}"
              f"{ft['f1'][i]:>9.3f}{ft['f1'][i] - bs['f1'][i]:>+8.3f}"
              f"{ft['precision'][i]:>9.3f}{ft['recall'][i]:>8.3f}")
    print("(gain = fine-tuned F1 minus base F1)")

    print(f"\nInference speed (EfficientNet-B4, {args.size}x{args.size}, bf16, "
          "model forward only):")
    for key, value in speed.items():
        print(f"  {key}: {value:.2f}")

    # Persist summary CSV and JSON deliverables
    with open(out_dir / "overall_comparison.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "accuracy", "acc_ci_low", "acc_ci_high",
                    "macro_f1", "f1_ci_low", "f1_ci_high", "weighted_f1"])
        for m in MODELS:
            o = overall[m]
            w.writerow([m, f"{o['accuracy']:.4f}", f"{o['accuracy_ci'][0]:.4f}",
                        f"{o['accuracy_ci'][1]:.4f}", f"{o['macro_f1']:.4f}",
                        f"{o['macro_f1_ci'][0]:.4f}", f"{o['macro_f1_ci'][1]:.4f}",
                        f"{o['weighted_f1']:.4f}"])

    with open(out_dir / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["class", "support"] + [f"{m}_{s}" for m in MODELS
                                           for s in ("precision", "recall", "f1")]
                   + ["finetune_minus_base_f1"])
        for i, name in enumerate(names):
            row = [name, int(ft["support"][i])]
            for m in MODELS:
                row += [f"{mets[m][s][i]:.4f}" for s in ("precision", "recall", "f1")]
            row.append(f"{ft['f1'][i] - bs['f1'][i]:.4f}")
            w.writerow(row)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "label"] + [f"{m}_pred" for m in MODELS]
                   + ["finetune_confidence"])
        for i, path in enumerate(test_paths):
            rel = os.path.relpath(path, ROOT / "data").replace(os.sep, "/")
            w.writerow([rel, names[labels[i]]]
                       + [names[preds[m][i]] for m in MODELS]
                       + [f"{probs['finetune'][i, preds['finetune'][i]]:.4f}"])

    per_class = {
        names[i]: {m: {"precision": float(mets[m]["precision"][i]),
                       "recall": float(mets[m]["recall"][i]),
                       "f1": float(mets[m]["f1"][i])} for m in MODELS}
        for i in range(k)
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"n_test": int(n), "checkpoints": ckpts, "overall": overall,
                   "comparisons": comparisons, "per_class": per_class,
                   "speed": speed, "bootstrap_resamples": args.n_boot}, f, indent=2)

    plot_confusions([cms["base"], cms["finetune"]],
                    [f"Base (accuracy {overall['base']['accuracy'] * 100:.1f}%)",
                     f"Fine-tuned (accuracy {overall['finetune']['accuracy'] * 100:.1f}%)"],
                    names, out_dir / "confusion_matrices.png")
    plot_per_class({m: mets[m]["f1"] for m in MODELS}, names,
                   out_dir / "per_class_f1.png")
    shown = plot_gallery(test_paths, labels, preds["finetune"], probs["finetune"],
                         names, out_dir / "misclassifications.png",
                         args.gallery, args.seed)
    print(f"\nSaved results to {out_dir}"
          + ("" if shown else " (fine-tuned model made no errors; no gallery)"))


if __name__ == "__main__":
    main()
