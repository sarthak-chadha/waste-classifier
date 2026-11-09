"""
Step 1: Stratified dataset partitioning and dynamic oversampling plan.

Generates reproducible splits and balancing policies:
    1. splits.csv:
       Contains relative image paths, class labels, and split assignments (70% train, 15% val, 15% test).
       Enforces stratification per class so each category maintains identical split ratios.
    2. oversampling_plan.json:
       Computes per-epoch oversampling targets for training data to mitigate class imbalance dynamically:
           target = max(n, min(round(average), int(max_repeat * n)))
       Oversampling occurs purely in-memory during training; no redundant files are written to disk.
       Validation and test sets remain strictly natural (unmodified and non-augmented).

Usage:
    python src/step1_split_data.py
    python src/step1_split_data.py --seed 42 --max-repeat 2.0 --force
"""
import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_classes(data_dir):
    """Discovers valid class subdirectories while ignoring hidden or system folders."""
    return sorted(
        e for e in os.listdir(data_dir)
        if not e.startswith(".") and (data_dir / e).is_dir()
    )


def list_images(data_dir, cls):
    """
    Recursively scans class folder for supported image files.
    Returns relative POSIX-style paths (forward slashes) for cross-platform portability.
    """
    paths = []
    for root, dirs, names in os.walk(data_dir / cls):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if name.startswith(".") or Path(name).suffix.lower() not in IMG_EXTS:
                continue
            full = Path(root) / name
            paths.append(full.relative_to(data_dir).as_posix())
    return sorted(paths)


def split_class(paths, train_frac, val_frac, rng):
    """
    Partitions sorted class paths into train, val, and test splits.
    Deterministic shuffling via a seeded RNG ensures 100% reproducible splits across runs.
    """
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    return (
        paths[:n_train],
        paths[n_train:n_train + n_val],
        paths[n_train + n_val:],
    )


def build_plan(train_counts, max_repeat):
    """
    Builds the per-epoch sampling policy for the training set.
    For each class with n samples:
        target = max(n, min(round(average), int(max_repeat * n)))
    Classes above the average remain unmodified. Under-represented classes are capped
    at max_repeat (e.g. 2.0x) to avoid overfitting while improving minority recall.
    """
    average = sum(train_counts.values()) / len(train_counts)
    plan = {}
    for cls, n in train_counts.items():
        target = max(n, min(round(average), int(max_repeat * n)))
        plan[cls] = {
            "train_count": n,
            "epoch_target": target,
            "extra_per_epoch": target - n,
            "avg_repeat": round(target / n, 2),
        }
    return plan, average


def main():
    """Executes stratified splitting, validation checks, and serializes CSV and JSON artifacts."""
    parser = argparse.ArgumentParser(description="Split data and plan oversampling")
    parser.add_argument("--data", default=str(ROOT / "data"),
                        help="dataset root containing one folder per class")
    parser.add_argument("--out", default=str(ROOT / "outputs"),
                        help="folder for splits.csv and oversampling_plan.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for the split (default: 42)")
    parser.add_argument("--train-frac", type=float, default=0.70,
                        help="fraction of each class used for training")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="fraction of each class used for validation; remainder is test")
    parser.add_argument("--max-repeat", type=float, default=2.0,
                        help="maximum per-epoch repeat factor for minority classes")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing splits and plan files")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    splits_path = out_dir / "splits.csv"
    plan_path = out_dir / "oversampling_plan.json"

    if not data_dir.is_dir():
        sys.exit(f"Data folder not found: {data_dir}")

    # Guard against accidental re-splits: maintaining fixed splits is essential for benchmark consistency
    existing = [p for p in (splits_path, plan_path) if p.exists()]
    if existing and not args.force:
        sys.exit(f"{existing[0]} already exists. Use --force to regenerate.")
    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1
            and args.train_frac + args.val_frac < 1):
        sys.exit("train-frac and val-frac must be positive and sum to below 1")
    if args.max_repeat < 1.0:
        sys.exit("max-repeat must be at least 1.0")

    classes = list_classes(data_dir)
    if not classes:
        sys.exit(f"No class folders found in {data_dir}")

    # Seeded pseudo-random generator guarantees cross-platform deterministic splitting
    rng = random.Random(args.seed)
    rows, per_class = [], {}
    for cls in classes:
        paths = list_images(data_dir, cls)
        if not paths:
            sys.exit(f"No images found for class '{cls}'")
        train, val, test = split_class(paths, args.train_frac, args.val_frac, rng)

        # Fail loudly if any class has insufficient images to populate all splits
        if min(len(train), len(val), len(test)) == 0:
            sys.exit(
                f"Class '{cls}' has only {len(paths)} images; the split gives "
                f"train={len(train)}, val={len(val)}, test={len(test)}. "
                "Add images or remove the class."
            )
        per_class[cls] = (len(train), len(val), len(test))
        for split_name, items in (("train", train), ("val", val), ("test", test)):
            rows.extend((p, cls, split_name) for p in items)

    # Verification: ensure zero duplicate path assignments across all splits
    all_paths = [r[0] for r in rows]
    assert len(all_paths) == len(set(all_paths)), "Data leakage detected: an image appears in multiple splits"

    plan, average = build_plan({c: v[0] for c, v in per_class.items()},
                               args.max_repeat)

    # Persist split mappings and oversampling configuration
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "classes": classes,
                "class_to_idx": {c: i for i, c in enumerate(classes)},
                "average_train_count": round(average, 1),
                "max_repeat": args.max_repeat,
                "plan": plan,
            },
            f,
            indent=2,
        )

    # Print summary statistics table
    print(f"\n{'class':<12}{'train':>7}{'val':>6}{'test':>6}"
          f"{'target':>9}{'extra':>7}{'repeat':>8}")
    print("-" * 55)
    for cls in classes:
        tr, va, te = per_class[cls]
        p = plan[cls]
        print(f"{cls:<12}{tr:>7}{va:>6}{te:>6}{p['epoch_target']:>9}"
              f"{p['extra_per_epoch']:>7}{p['avg_repeat']:>7.2f}x")
    totals = [sum(v[i] for v in per_class.values()) for i in range(3)]
    epoch_total = sum(p["epoch_target"] for p in plan.values())
    print("-" * 55)
    print(f"{'TOTAL':<12}{totals[0]:>7}{totals[1]:>6}{totals[2]:>6}"
          f"{epoch_total:>9}{epoch_total - totals[0]:>7}"
          f"{epoch_total / totals[0]:>7.2f}x")
    print(f"\nAverage training images per class: {average:.1f}")
    print(f"Saved: {splits_path}")
    print(f"Saved: {plan_path}")


if __name__ == "__main__":
    main()
