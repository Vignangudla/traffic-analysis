#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict
import shutil


def read_yolo_labels(lbl_dir: Path):
    stats = Counter()
    index = {}
    for p in sorted(lbl_dir.glob("*.txt")):
        cls_ids = []
        for line in p.read_text().strip().splitlines():
            if not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            cls_ids.append(cid)
        index[p.stem] = cls_ids
        stats.update(cls_ids)
    return index, stats


def split_balance(yolo_root: Path, out_root: Path, train=0.8, val=0.1, seed=42, oversample_rare=True):
    img_dir = yolo_root / "images"
    lbl_dir = yolo_root / "labels"
    idx, stats = read_yolo_labels(lbl_dir)
    stems = list(idx.keys())
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:]
    }

    if oversample_rare:
        # Simple oversample in training split by duplicating files of rare classes
        cls_freq = Counter()
        for s in splits["train"]:
            cls_freq.update(idx[s])
        if cls_freq:
            max_freq = max(cls_freq.values())
            extra = []
            for s in splits["train"]:
                if not idx[s]:
                    continue
                vote = max(cls_freq[c] for c in idx[s])
                rep = max(1, int(max_freq / max(1, vote)) - 1)
                extra.extend([s] * rep)
            splits["train"].extend(extra)
            random.shuffle(splits["train"])

    # Write out
    for split, lst in splits.items():
        out_img = out_root / split / "images"
        out_lbl = out_root / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for stem in lst:
            # copy image
            src_img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    src_img = p
                    break
            if src_img:
                shutil.copy2(src_img, out_img / src_img.name)
            src_lbl = lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_lbl / src_lbl.name)


def main():
    ap = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test with optional balancing")
    ap.add_argument("yolo_root", type=Path, help="Root that has images/ and labels/")
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_balance", action="store_true")
    args = ap.parse_args()
    split_balance(args.yolo_root, args.out_root, args.train, args.val, args.seed, not args.no_balance)


if __name__ == "__main__":
    main()

