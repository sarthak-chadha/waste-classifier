"""
Step 0: Pre-training environment, hardware, and dataset sanity check.

Verifies the runtime environment before starting training:
    1. Python version and virtual environment activation.
    2. Core dependency packages (PyTorch, Torchvision, NumPy, Pandas, Scikit-Learn,
       Matplotlib, Seaborn, Pillow, Tqdm, Streamlit).
    3. CUDA GPU availability, compute capability, VRAM headroom, cuDNN, and matrix multiplication.
    4. Dataset directory structure, class balance, image decode integrity, exact resolutions, and color modes.
    5. MD5 hash deduplication across classes to catch label leakage.

Usage:
    python src/sanity_check.py
    python src/sanity_check.py --data path/to/data --size 384

Returns exit code 0 if all required checks pass, 1 otherwise.
"""
import argparse
import hashlib
import importlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Project dataset directory defaults to <project_root>/data regardless of current working directory
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"

# Expected image counts per category based on standard benchmark splits (deviations >3% trigger warnings)
EXPECTED = {
    "battery": 756,
    "cardboard": 1411,
    "clothes": 1892,
    "glass": 1736,
    "metal": 930,
    "organic": 699,
    "paper": 1336,
    "plastic": 1597,
    "shoes": 1449,
    "trash": 453,
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Required packages: mapped as (import_name, pip_package_name)
PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("PIL", "pillow"),
    ("tqdm", "tqdm"),
    ("streamlit", "streamlit"),
]

errors, warnings = [], []


def header(title):
    """Prints a structured section header banner."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def ok(msg):
    """Logs a successful check."""
    print(f"  [ OK ] {msg}")


def warn(msg):
    """Logs a non-fatal warning."""
    print(f"  [WARN] {msg}")
    warnings.append(msg)


def fail(msg):
    """Logs a fatal error blocking training execution."""
    print(f"  [FAIL] {msg}")
    errors.append(msg)


def check_packages():
    """Validates Python interpreter, virtual environment, dependencies, and GPU hardware support."""
    header("1. PYTHON AND PACKAGES")
    v = sys.version_info
    ok(f"Python {v.major}.{v.minor}.{v.micro}")

    # Check whether running inside an isolated virtual environment
    if sys.prefix != sys.base_prefix:
        ok(f"Virtual environment active: {os.path.basename(sys.prefix)}")
    else:
        warn("No virtual environment detected")

    loaded = {}
    for module, pip_name in PACKAGES:
        try:
            m = importlib.import_module(module)
            loaded[module] = m
            ok(f"{pip_name:<14} {getattr(m, '__version__', '')}")
        except ImportError:
            fail(f"{pip_name} is not installed (pip install {pip_name})")

    header("2. GPU")
    torch = loaded.get("torch")
    if torch is None:
        fail("Skipped: torch is missing")
        return
    if not torch.cuda.is_available():
        fail("CUDA is not available")
        return

    ok(f"CUDA available, build {torch.version.cuda}")
    ok(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    ok(f"Compute capability {props.major}.{props.minor}")

    # Free memory determines maximum feasible batch size and gradient accumulation
    free_mem, total_mem = torch.cuda.mem_get_info()
    ok(f"VRAM: {free_mem / 1024 ** 3:.1f} GB free / "
       f"{total_mem / 1024 ** 3:.1f} GB total")
    if free_mem / 1024 ** 3 < 4:
        warn("Less than 4 GB of VRAM is free; close other GPU applications")

    # Convolutional backbones rely heavily on cuDNN kernel optimizations
    if torch.backends.cudnn.is_available():
        ok(f"cuDNN available (version {torch.backends.cudnn.version()})")
    else:
        fail("cuDNN is not available")

    # Run real GPU tensor operation to catch binary/driver architectural mismatches
    try:
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        ok("GPU matrix multiplication test passed")
    except Exception as e:
        fail(f"GPU computation failed: {e}")

    if "torchvision" not in loaded:
        return
    try:
        from torchvision.models import efficientnet_b4

        efficientnet_b4(weights=None)
        ok("EfficientNet-B4 available in torchvision")
    except Exception as e:
        fail(f"Could not build EfficientNet-B4: {e}")


def file_md5(path):
    """Computes MD5 hash in 1MB streaming chunks to prevent memory spikes on large files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(data_dir, size):
    """Inspects class subdirectories, image counts, byte corruptions, dimensions, and duplicate hashes."""
    header(f"3. DATASET ({data_dir})")
    try:
        from PIL import Image
    except ImportError:
        fail("pillow is missing, cannot inspect images")
        return
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        fail(f"Data folder not found: {data_dir}")
        return

    def rel(path):
        return os.path.relpath(path, data_dir)

    # Exclude hidden files or OS directory artifacts (.git, .ipynb_checkpoints, .DS_Store)
    entries = sorted(e for e in os.listdir(data_dir) if not e.startswith("."))
    folders = [e for e in entries if (data_dir / e).is_dir()]
    loose = [e for e in entries if e not in folders]
    if loose:
        warn(f"Loose files directly in the data folder (ignored): {loose[:5]}")

    # Confirm all expected category folders exist and no unexpected directories are present
    missing = sorted(set(EXPECTED) - set(folders))
    extra = sorted(set(folders) - set(EXPECTED))
    if missing:
        fail(f"Missing class folders: {missing}")
        if "biological" in extra:
            print("         'biological' should be renamed to 'organic'")
    if extra:
        fail(f"Unexpected folders (would become extra classes): {extra}")
    if not missing and not extra:
        ok(f"All {len(EXPECTED)} expected class folders present")

    counts = {}
    bad_files, wrong_size, non_image = [], [], []
    modes = Counter()
    valid_paths = []

    for cls in folders:
        files = []
        for root, dirs, names in os.walk(data_dir / cls):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files.extend(
                os.path.join(root, n) for n in names if not n.startswith(".")
            )

        n_img = 0
        for path in tqdm(files, desc=f"{cls:<10}", leave=False,
                         disable=not sys.stdout.isatty()):
            if os.path.splitext(path)[1].lower() not in IMG_EXTS:
                non_image.append(path)
                continue
            try:
                # Fully decode raster byte payload to catch truncated or corrupted files
                with Image.open(path) as im:
                    im.load()
                    mode, im_size = im.mode, im.size
            except Exception as e:
                bad_files.append((rel(path), str(e)[:60]))
                continue
            n_img += 1
            modes[mode] += 1
            valid_paths.append(path)
            if im_size != (size, size):
                wrong_size.append((rel(path), im_size))
        counts[cls] = n_img

    print(f"\n  {'class':<12}{'found':>8}{'expected':>10}{'diff':>8}")
    print(f"  {'-' * 38}")
    for cls in sorted(set(EXPECTED) | set(counts)):
        found = counts.get(cls, 0)
        exp = EXPECTED.get(cls)
        diff = "" if exp is None else f"{found - exp:+d}"
        print(f"  {cls:<12}{found:>8}{exp if exp else '-':>10}{diff:>8}")
    total = sum(counts.values())
    print(f"  {'-' * 38}")
    print(f"  {'TOTAL':<12}{total:>8}{sum(EXPECTED.values()):>10}\n")

    for cls, exp in EXPECTED.items():
        found = counts.get(cls, 0)
        if abs(found - exp) > 0.03 * exp:
            warn(f"{cls}: {found} images found, about {exp} expected")
    if total == 0:
        fail("No valid images found")
        return
    ok(f"Total valid images: {total}")

    # Quantify category distribution ratio (highest count vs lowest count)
    populated = {c: n for c, n in counts.items() if n > 0}
    largest = max(populated, key=populated.get)
    smallest = min(populated, key=populated.get)
    ratio = populated[largest] / populated[smallest]
    msg = (f"Imbalance: '{largest}' ({populated[largest]}) is {ratio:.1f}x "
           f"'{smallest}' ({populated[smallest]})")
    ok(msg) if ratio <= 5 else warn(msg)

    if bad_files:
        fail(f"{len(bad_files)} unreadable or corrupt images, "
             f"e.g. {bad_files[:3]}")
    else:
        ok("All images decode without errors")
    if non_image:
        warn(f"{len(non_image)} non-image files in class folders, "
             f"e.g. {[rel(p) for p in non_image[:3]]}")

    if wrong_size:
        sizes = Counter(s for _, s in wrong_size)
        fail(f"{len(wrong_size)} images are not {size}x{size}; "
             f"most common other sizes: {sizes.most_common(3)}")
        print(f"         example: {wrong_size[0][0]}")
    else:
        ok(f"All {total} valid images are exactly {size}x{size}")

    # Ensure color channels conform to standard 3-channel RGB format
    if set(modes) - {"RGB"}:
        warn(f"Color modes found: {dict(modes)}")
    else:
        ok("All images are RGB")

    # Fast two-stage deduplication: group by file byte size first, hash only size collisions
    by_size = defaultdict(list)
    for p in valid_paths:
        by_size[os.path.getsize(p)].append(p)
    hashes = defaultdict(list)
    for paths in by_size.values():
        if len(paths) > 1:
            for p in paths:
                hashes[file_md5(p)].append(p)

    # Cross-class duplicates represent ambiguous labeling; intra-class duplicates risk train/test leakage
    dups = [v for v in hashes.values() if len(v) > 1]
    if dups:
        cross = [
            v for v in dups
            if len({Path(p).relative_to(data_dir).parts[0] for p in v}) > 1
        ]
        msg = (f"{len(dups)} groups of identical files "
               f"({len(cross)} span multiple classes)")
        if cross:
            fail(f"{msg}, e.g. {[rel(p) for p in cross[0][:2]]}")
        else:
            warn(f"{msg}, e.g. {[rel(p) for p in dups[0][:2]]}")
    else:
        ok("No exact duplicate images")


def main():
    """Command-line entrypoint for Step 0 sanity validation."""
    parser = argparse.ArgumentParser(description="Pre-training sanity check")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="dataset root containing one folder per class")
    parser.add_argument("--size", type=int, default=384,
                        help="expected image width and height in pixels")
    args = parser.parse_args()

    check_packages()
    check_dataset(args.data, args.size)

    header("SUMMARY")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    if errors:
        print("\n  NOT READY. Fix the [FAIL] items above and run again.")
        sys.exit(1)
    print("\n  READY. All required checks passed.")


if __name__ == "__main__":
    main()
