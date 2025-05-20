#!/usr/bin/env bash
# stack_videos.sh  ▸  side-by-side comparison with labels
# -------------------------------------------------------
# • Left  input:  normal_video_out.mp4    (label: “LatentSync”)
# • Right input:  video_out.mp4           (label: “LatentSync + Rolling Denoising”)
# • Output:       side_by_side_2160x1920.mp4  (H.264 + AAC)
# -------------------------------------------------------
# Both source clips must already be 1080 × 1920 (portrait).
# Edit FONT, TEXT_L / TEXT_R, or file names below if needed.

set -euo pipefail

LEFT_IN="normal_video_out_19.mp4"
RIGHT_IN="video_out_19.mp4"
OUT="side_by_side_19.mp4"

TEXT_L="LatentSync"
TEXT_R="LatentSync + Rolling Denoising"

# Adjust if your system keeps fonts elsewhere (macOS example shown):
#   FONT="/System/Library/Fonts/SFNSDisplay-Bold.otf"
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ffmpeg -y \
  -i "$LEFT_IN" \
  -i "$RIGHT_IN" \
  -filter_complex "\
    [0:v]drawtext=fontfile=${FONT}:text='${TEXT_L}':x=(w-tw)/2:y=h-80:fontsize=48:fontcolor=white:borderw=3[v0]; \
    [1:v]drawtext=fontfile=${FONT}:text='${TEXT_R}':x=(w-tw)/2:y=h-80:fontsize=48:fontcolor=white:borderw=3[v1]; \
    [v0][v1]hstack=inputs=2[vout]" \
  -map '[vout]' -map 0:a? -map 1:a? \
  -c:v libx264 -crf 18 -preset medium \
  -c:a aac -b:a 192k \
  "$OUT"

echo "✅  Created $OUT

