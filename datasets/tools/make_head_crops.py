#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import cv2
import numpy as np


def crop_top(img, box, ratio=0.4):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    th = max(1, int((y2 - y1) * ratio))
    return img[y1:y1 + th, x1:x2]


def yolo_box_to_xyxy(lbl_line, w, h):
    c, xc, yc, bw, bh = map(float, lbl_line.split())
    x = (xc - bw / 2) * w
    y = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return int(x), int(y), int(x2), int(y2)


def main():
    ap = argparse.ArgumentParser(description="Make head crops from YOLO labels (class 0=person)")
    ap.add_argument("images", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--class_person", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.4)
    ap.add_argument("--save_meta", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = []
    for imgp in sorted(args.images.glob("*")):
        if imgp.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        im = cv2.imread(str(imgp))
        if im is None:
            continue
        h, w = im.shape[:2]
        lbl = args.labels / (imgp.stem + ".txt")
        if not lbl.exists():
            continue
        for i, line in enumerate(lbl.read_text().strip().splitlines()):
            if not line:
                continue
            cls = int(line.split()[0])
            if cls != args.class_person:
                continue
            x1, y1, x2, y2 = yolo_box_to_xyxy(line, w, h)
            crop = crop_top(im, (x1, y1, x2, y2), args.ratio)
            if crop.size == 0:
                continue
            outp = args.out / f"{imgp.stem}_p{i}.jpg"
            cv2.imwrite(str(outp), crop)
            if args.save_meta:
                meta.append({"src": str(imgp), "label": str(lbl), "idx": i, "out": str(outp)})
    if args.save_meta:
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

