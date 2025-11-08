#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
from collections import defaultdict


def coco_to_yolo(ann_path: Path, out_dir: Path, categories_map=None):
    ann = json.loads(Path(ann_path).read_text())
    images = {im["id"]: im for im in ann.get("images", [])}
    cats = {c["id"]: c["name"] for c in ann.get("categories", [])}
    if categories_map is None:
        # Default: map to our 4 classes if possible
        canonical = {"person": 0, "motorcycle": 1, "helmet": 2, "plate": 3}
        categories_map = {cid: canonical.get(name, None) for cid, name in cats.items()}
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    by_img = defaultdict(list)
    for a in ann.get("annotations", []):
        img_id = a["image_id"]
        cat_id = a["category_id"]
        if a.get("iscrowd", 0) == 1:
            continue
        yolo_cls = categories_map.get(cat_id, None)
        if yolo_cls is None:
            continue
        bbox = a["bbox"]  # COCO xywh
        by_img[img_id].append((yolo_cls, bbox))

    for img_id, recs in by_img.items():
        im = images[img_id]
        w, h = im["width"], im["height"]
        # copy/mirror image relative to ann file location
        img_path = Path(ann_path).parent / im["file_name"]
        out_img_path = out_img / Path(im["file_name"]).name
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if img_path.exists():
                if img_path.resolve() != out_img_path.resolve():
                    shutil.copy2(img_path, out_img_path)
        except Exception:
            pass
        lines = []
        for ycls, (x, y, bw, bh) in recs:
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{ycls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        (out_lbl / (Path(im["file_name"]).stem + ".txt")).write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    ap.add_argument("annotation", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--map", type=str, default=None,
                    help='Optional JSON dict mapping original category names to our classes, e.g. {"human":"person"}')
    args = ap.parse_args()
    cat_map = None
    if args.map:
        name_map = json.loads(args.map)
        # Build categories_map from name mapping during conversion by reading category list in annotation
        ann = json.loads(args.annotation.read_text())
        cats = {c["id"]: c["name"] for c in ann.get("categories", [])}
        canonical = {"person": 0, "motorcycle": 1, "helmet": 2, "plate": 3}
        cat_map = {cid: canonical.get(name_map.get(n, n), None) for cid, n in cats.items()}
    coco_to_yolo(args.annotation, args.out, cat_map)


if __name__ == "__main__":
    main()

