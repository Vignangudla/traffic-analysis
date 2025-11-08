#!/usr/bin/env bash
set -euo pipefail

# REQUIRE: ~/.kaggle/kaggle.json must exist (Kaggle API token)
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing ~/.kaggle/kaggle.json (Kaggle API token). Add it and re-run." >&2
  exit 2
fi

mkdir -p samples datasets/_tmp_fetch
: > samples/ATTRIBUTION_DATASET.jsonl

# Prefer small MP4s from these slugs (dataset platforms only)
PREFERRED_DATASETS=(
  "ayushraj2349/sample-videos-for-helmet-detection-on-yolov8"
  "mexwell/motorcycle-accident-video-dataset"   # use normal riding/traffic clips only; skip gore
)

max_files=4
idx=0
total_bytes=0
limit_bytes=$((400 * 1024 * 1024))  # 400MB cap
tmpdir="datasets/_tmp_fetch"

ff_ok () {
  local f="$1"
  # duration (s) and height (px)
  local dur height
  dur=$(ffprobe -v error -select_streams v:0 -show_entries format=duration -of default=nw=1:nk=1 "$f" | awk '{printf("%.0f",$1)}' || echo 0)
  height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "$f" | head -n1 || echo 0)
  [[ "$dur" -le 60 && "$height" -le 720 ]]
}

for ref in "${PREFERRED_DATASETS[@]}"; do
  # list candidate MP4 names from the dataset
  mapfile -t FILES < <(kaggle datasets files "$ref" | tail -n +2 | awk '{$1=$1}1' | awk '{print $1}' | grep -i '\.mp4$' || true)
  for name in "${FILES[@]}"; do
    [[ -n "$name" ]] || continue
    # heuristics to avoid gore/montage; prefer traffic/riding/helmet names
    low=$(echo "$name" | tr '[:upper:]' '[:lower:]')
    if echo "$low" | grep -Eq 'crash|accident|compilation|gore'; then
      continue
    fi
    # download just this file (unzipped)
    kaggle datasets download "$ref" -f "$name" -p "$tmpdir" -q --unzip || continue
    f="$tmpdir/$name"
    [[ -f "$f" ]] || continue
    # per-file size cap (≤95MB)
    bytes=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [[ "$bytes" -gt $((95 * 1024 * 1024)) ]]; then
      rm -f "$f"; continue
    fi
    # duration/height cap
    if ! ff_ok "$f"; then
      rm -f "$f"; continue
    fi
    # total cap
    if [[ $((total_bytes + bytes)) -gt $limit_bytes ]]; then
      rm -f "$f"; continue
    fi
    # accept: move to samples as ds_bike_N.mp4
    idx=$((idx+1))
    out="samples/ds_bike_${idx}.mp4"
    mv -f "$f" "$out"
    total_bytes=$((total_bytes + bytes))
    # attribution line
    url="https://www.kaggle.com/datasets/${ref}"
    jq -nc --arg file "$out" --arg dataset "$ref" --arg title "$name" --arg url "$url" --arg lic "Kaggle dataset terms" \
      '{file:$file, dataset:$dataset, title:$title, url:$url, license:$lic}' >> samples/ATTRIBUTION_DATASET.jsonl
    # stop at 4 clips
    [[ "$idx" -ge "$max_files" ]] && break
  done
  [[ "$idx" -ge "$max_files" ]] && break
done

# print summary and total size
ls -lh samples/ds_bike_*.mp4 2>/dev/null || true
echo "TOTAL:"
du -ch samples/ds_bike_*.mp4 2>/dev/null | tail -n1 || true
