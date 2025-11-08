#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


CLASS_MAP = {"person": 0, "motorcycle": 1, "helmet": 2, "plate": 3}


def convert_file(xml_path: Path, out_lbl: Path) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return False
    w = float(size.findtext("width", default="0"))
    h = float(size.findtext("height", default="0"))
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="")
        if name not in CLASS_MAP:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))
        bw = xmax - xmin
        bh = ymax - ymin
        xc = xmin + bw / 2
        yc = ymin + bh / 2
        xc, yc, bw, bh = xc / w, yc / h, bw / w, bh / h
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"{CLASS_MAP[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    if lines:
        out_lbl.write_text("".join(lines))
        return True
    return False


def voc_to_yolo(voc_dir: Path, out_dir: Path, img_subdir: str = "JPEGImages", ann_subdir: str = "Annotations"):
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    imgdir = voc_dir / img_subdir
    anndir = voc_dir / ann_subdir
    for xml in sorted(anndir.glob("*.xml")):
        stem = xml.stem
        # copy image if exists
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            imgp = imgdir / f"{stem}{ext}"
            if imgp.exists():
                shutil.copy2(imgp, out_img / imgp.name)
                break
        convert_file(xml, out_lbl / f"{stem}.txt")


def main():
    ap = argparse.ArgumentParser(description="Convert PASCAL VOC to YOLO format")
    ap.add_argument("voc_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--img_subdir", default="JPEGImages")
    ap.add_argument("--ann_subdir", default="Annotations")
    args = ap.parse_args()
    voc_to_yolo(args.voc_dir, args.out, args.img_subdir, args.ann_subdir)


if __name__ == "__main__":
    main()

