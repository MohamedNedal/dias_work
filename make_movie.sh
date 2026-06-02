#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <image_folder> <output_name> [framerate] [crf]
  image_folder : directory containing PNG images (natural sort by filename)
  output_name  : output movie filename (.mp4 appended if missing)
  framerate    : frames per second (default: 10)
  crf          : x264 quality; lower = larger/better, higher = smaller (default: 23)
EOF
    exit 1
}

[ $# -lt 2 ] && usage

IMG_DIR="$1"
OUTPUT_NAME="$2"
FRAMERATE="${3:-10}"
CRF="${4:-23}"

[ ! -d "$IMG_DIR" ] && { echo "Error: directory not found: $IMG_DIR" >&2; exit 1; }
IMG_DIR="$(cd "$IMG_DIR" && pwd)"

shopt -s nullglob
mapfile -t IMAGES < <(cd "$IMG_DIR" && ls -1v *.png 2>/dev/null)
shopt -u nullglob

[ ${#IMAGES[@]} -eq 0 ] && { echo "Error: no PNG images found in $IMG_DIR" >&2; exit 1; }
echo "Found ${#IMAGES[@]} PNG images."

# Read PNG dimensions from file headers in one Python call.
# PNG width/height are big-endian uint32 at bytes 16-23 - no decode needed.
echo "Checking image dimensions..."
read DISTINCT MAX_W MAX_H < <(
    cd "$IMG_DIR"
    python3 - "${IMAGES[@]}" <<'PY'
import sys, struct
sizes = set(); mw = mh = 0
for f in sys.argv[1:]:
    with open(f, 'rb') as fp:
        fp.seek(16)
        w, h = struct.unpack('>II', fp.read(8))
    sizes.add((w, h))
    if w > mw: mw = w
    if h > mh: mh = h
print(len(sizes), mw, mh)
PY
)

if [ "$DISTINCT" -gt 1 ]; then
    PAD_W=$(( (MAX_W + 1) / 2 * 2 ))
    PAD_H=$(( (MAX_H + 1) / 2 * 2 ))
    echo "Mixed dimensions ($DISTINCT distinct sizes). Padding all frames to ${PAD_W}x${PAD_H}."
    VF="pad=${PAD_W}:${PAD_H}:(ow-iw)/2:(oh-ih)/2:color=black"
else
    echo "All images share dimensions ${MAX_W}x${MAX_H}."
    # Match the original script's no-op-when-even pad to reproduce its file size
    VF="pad=ceil(iw/2)*2:ceil(ih/2)*2"
fi

# Non-destructive sequencing: symlink originals into %05d names so ffmpeg
# reads them in natural order without renaming the source files
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
i=1
for img in "${IMAGES[@]}"; do
    ln -s "$IMG_DIR/$img" "$WORK_DIR/$(printf '%05d.png' "$i")"
    i=$((i + 1))
done

OUTPUT="${OUTPUT_NAME%.mp4}.mp4"
echo "Encoding $OUTPUT at ${FRAMERATE} fps (CRF $CRF)..."
ffmpeg -y -framerate "$FRAMERATE" -i "$WORK_DIR/%05d.png" \
       -vf "$VF" -c:v libx264 -preset medium -crf "$CRF" -pix_fmt yuv420p \
       "$IMG_DIR/$OUTPUT"

echo "Movie created: $IMG_DIR/$OUTPUT"
