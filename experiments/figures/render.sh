#!/usr/bin/env bash
# Rasterise the figures at 2x so they can actually be looked at before shipping.
#
# The window is deliberately taller than the SVG and the result is cropped back
# down: headless Chromium clips the screenshot short of the requested height, so
# a window sized to the SVG silently loses the last ~90px -- which is where the
# caption lives. Renders that quietly drop the caption are worse than no render.
#
# Usage: render.sh [output_dir]   -- default is this directory. A relative
# output_dir is resolved before the cd below, so it means what the caller meant.
set -euo pipefail
OUT=$(cd "${1:-$(dirname "$0")}" && pwd)
cd "$(dirname "$0")"
CH=${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}

for svg in fig1_coverage fig3_horizon hook_points; do
  [ -f "$svg.svg" ] || continue
  read -r W H < <(grep -o 'viewBox="0 0 [0-9]* [0-9]*"' "$svg.svg" | head -1 \
                  | awk '{print $3, $4}' | tr -d '"')
  printf '<style>html,body{margin:0;padding:0;background:#fff}</style><img src="%s.svg" width="%s" height="%s">' \
         "$svg" "$W" "$H" > ".render_$svg.html"
  "$CH" --headless --disable-gpu --no-sandbox --hide-scrollbars \
        --force-device-scale-factor=2 --screenshot="$OUT/$svg.png" \
        --window-size="$W,$((H + 400))" "file://$PWD/.render_$svg.html" 2>/dev/null
  python3 - "$OUT/$svg.png" "$W" "$H" <<'PY'
import sys
from PIL import Image
p, w, h = sys.argv[1], int(sys.argv[2]) * 2, int(sys.argv[3]) * 2
im = Image.open(p)
if im.size != (w, h):
    im.crop((0, 0, w, h)).save(p)
print(f"{p}  {w}x{h}")
PY
  rm -f ".render_$svg.html"
done
