#!/usr/bin/env bash
set -euo pipefail
mkdir -p samples
QUERIES=(
  "motorcycle helmet violation india cctv"
  "motorcycle triple riding india cctv"
  "two wheeler traffic india cctv"
  "motorcycle traffic junction cctv india"
)
FMT='bv*[height<=720][vcodec*=avc1]+ba/best[height<=720]/best'
MAXFILES=4
COUNT=0
for q in "${QUERIES[@]}"; do
  yt-dlp "ytsearch30:${q}" \
    --match-filter 'license~=.*Creative.* & !is_live & duration <= 60' \
    --max-downloads $((MAXFILES-COUNT)) \
    --no-playlist --continue \
    --restrict-filenames \
    --max-filesize 95M \
    -f "$FMT" --merge-output-format mp4 \
    --write-info-json --no-warnings \
    -o "samples/cc_bike_%(id)s.%(ext)s" || true
  COUNT=$(ls -1 samples/cc_bike_*.mp4 2>/dev/null | wc -l || true)
  if [[ "$COUNT" -ge "$MAXFILES" ]]; then break; fi
done
# If we still have none, relax license but keep ≤60s and size cap:
COUNT=$(ls -1 samples/cc_bike_*.mp4 2>/dev/null | wc -l || true)
if [[ "$COUNT" -lt 1 ]]; then
  yt-dlp "ytsearch30:motorcycle traffic india cctv" \
    --match-filter '!is_live & duration <= 60' \
    --max-downloads 1 \
    --no-playlist --continue \
    --restrict-filenames \
    --max-filesize 95M \
    -f "$FMT" --merge-output-format mp4 \
    --write-info-json \
    -o "samples/cc_bike_%(id)s.%(ext)s" || true
fi
# Normalize names to cc_bike_1..4.mp4 (keep .info.json)
idx=0
for f in $(ls -1 samples/cc_bike_*.mp4 2>/dev/null | head -n $MAXFILES); do
  (( idx++ )) || true
  base=$(basename "$f" .mp4)
  mv -f "$f" "samples/cc_bike_${idx}.mp4"
  [[ -f "samples/${base}.info.json" ]] && mv -f "samples/${base}.info.json" "samples/cc_bike_${idx}.info.json" || true
done
# Attribution jsonl
: > samples/ATTRIBUTION.jsonl
for j in samples/cc_bike_*.info.json; do
  [[ -f "$j" ]] || continue
  jq -c '{file: input_filename | sub(".info.json$"; ".mp4"), title, uploader, license, url: .webpage_url}' "$j" >> samples/ATTRIBUTION.jsonl || true
done
# Print summary and total size
ls -lh samples/cc_bike_*.mp4 2>/dev/null || true
echo "TOTAL:"
du -ch samples/cc_bike_*.mp4 2>/dev/null | tail -n1
