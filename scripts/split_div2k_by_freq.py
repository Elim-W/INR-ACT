import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_and_preprocess(path: Path, size: int = 256) -> np.ndarray:
    """Load image as grayscale, resize isotropically, then reflect-pad to square."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L")

    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resample = Image.Resampling.LANCZOS
    img = img.resize((new_w, new_h), resample=resample)

    arr = np.asarray(img, dtype=np.float32) / 255.0

    pad_h = size - new_h
    pad_w = size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # reflect pad; fall back to edge if one dimension is too small
    pad_mode = "reflect"
    if new_h < 2 or new_w < 2:
        pad_mode = "edge"

    arr = np.pad(arr, ((top, bottom), (left, right)), mode=pad_mode)
    return arr


def _aggregate(values: np.ndarray, agg: str) -> float:
    if agg == "mean":
        return float(values.mean())
    if agg == "max":
        return float(values.max())
    if agg.startswith("p"):
        return float(np.percentile(values, float(agg[1:])))
    raise ValueError(f"unknown agg '{agg}' (use mean | pNN | max)")


def patch_intensity_entropy(
    img_gray: np.ndarray,
    patch: int = 32,
    agg: str = "mean",
    n_bins: int = 64,
) -> float:
    """
    Visual-complexity score via per-patch intensity entropy.

    - per patch: Shannon entropy of the quantized pixel-intensity histogram
        * sky / soft background  -> 1-2 bins occupied -> ~0
        * regular pattern (tiles) -> 2-4 dominant bins -> medium
        * market / foliage / crowd -> many bins -> high
    - aggregate across patches with `agg`:
        * "mean": overall information density of the image
        * "p75" / "p90" / "max": sensitive to the busiest region only
    """
    h, w = img_gray.shape
    ph = (h // patch) * patch
    pw = (w // patch) * patch
    if ph == 0 or pw == 0:
        return 0.0

    nh, nw = ph // patch, pw // patch
    patches = (img_gray[:ph, :pw]
               .reshape(nh, patch, nw, patch)
               .transpose(0, 2, 1, 3)
               .reshape(nh * nw, patch * patch))

    bin_idx = np.clip((patches * n_bins).astype(np.int32), 0, n_bins - 1)

    num_patches = bin_idx.shape[0]
    flat_idx = (np.arange(num_patches)[:, None] * n_bins + bin_idx).ravel()
    hist = np.bincount(
        flat_idx,
        minlength=num_patches * n_bins,
    ).reshape(num_patches, n_bins).astype(np.float64)

    totals = hist.sum(axis=1, keepdims=True)
    p = hist / np.maximum(totals, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropies = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)

    return _aggregate(entropies, agg)


def score_image(path: Path, size: int, patch: int, agg: str) -> float:
    arr = load_and_preprocess(path, size=size)
    return patch_intensity_entropy(arr, patch=patch, agg=agg)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(
        description="Split images into low/mid/high-frequency buckets."
    )
    ap.add_argument("--src", type=str, default="data/div2k")
    ap.add_argument("--dst", type=str, default="data/div2k_split")
    ap.add_argument("--size", type=int, default=256,
                    help="square size for frequency analysis")
    ap.add_argument("--patch", type=int, default=32,
                    help="patch size (px) for local entropy map")
    ap.add_argument("--agg", type=str, default="mean",
                    help="aggregation over per-patch entropies: "
                         "mean (overall info density), p75 / p90 / max (busiest region)")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {src}")

    scores = []
    for p in tqdm(paths, desc="Scoring"):
        s = score_image(
            p,
            size=args.size,
            patch=args.patch,
            agg=args.agg,
        )
        scores.append((p, s))

    scores.sort(key=lambda x: x[1])
    n = len(scores)
    k = 30
    mid_start = max(0, n // 2 - k // 2)

    buckets = {
        "low": scores[:k],
        "mid": scores[mid_start:mid_start + k],
        "high": scores[-k:],
    }

    for bucket_name, items in buckets.items():
        out_dir = dst / bucket_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p, _ in items:
            link_or_copy(p, out_dir / p.name, args.mode)

    csv_path = dst / "scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "hf_ratio", "bucket"])
        for bucket_name, items in buckets.items():
            for p, s in items:
                writer.writerow([p.name, f"{s:.8f}", bucket_name])

    print(
        f"done. {n} images -> "
        f"low={len(buckets['low'])}, mid={len(buckets['mid'])}, high={len(buckets['high'])}"
    )
    print(f"output dir: {dst}")
    print(f"scores csv: {csv_path}")

    k = 30
    mid_start = max(0, n // 2 - k // 2)
    previews = [
        ("LOWEST  30 (low-freq)",  scores[:k]),
        ("MIDDLE  30 (median)",    scores[mid_start:mid_start + k]),
        ("HIGHEST 30 (high-freq)", scores[-k:][::-1]),
    ]
    for title, items in previews:
        print(f"\n--- {title} ---")
        print(f"{'rank':>4}  {'filename':<20}  {'hf_ratio':>10}")
        for i, (p, s) in enumerate(items, 1):
            print(f"{i:>4}  {p.name:<20}  {s:>10.6f}")


if __name__ == "__main__":
    main()