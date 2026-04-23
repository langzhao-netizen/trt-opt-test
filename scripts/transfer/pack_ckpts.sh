#!/bin/bash
# Pack outputs/ckpts/* into tar.gz for transfer. Each ckpt → one .tar.gz + .sha256.
# Usage: ./scripts/pack_ckpts_for_transfer.sh [CKPT_ROOT]
# Output: ARCHIVE_ROOT/<ckpt_name>.tar.gz and <ckpt_name>.tar.gz.sha256
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CKPT_ROOT="${1:-${PROJECT_ROOT}/outputs/ckpts}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT_ROOT}/outputs/ckpts_archives}"

mkdir -p "$ARCHIVE_ROOT"
cd "$CKPT_ROOT"
echo "Packing ckpts from $CKPT_ROOT -> $ARCHIVE_ROOT"

for d in */; do
  name="${d%/}"
  [ "$name" = "*" ] && continue
  archive="$ARCHIVE_ROOT/$name.tar.gz"
  if [ -f "$archive" ]; then
    echo "SKIP (exists): $name"
    continue
  fi
  echo "Pack: $name"
  tar czf "$archive" -C "$CKPT_ROOT" "$name"
  (cd "$ARCHIVE_ROOT" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
done

echo "Done. Archives in $ARCHIVE_ROOT"
ls -la "$ARCHIVE_ROOT"/*.tar.gz 2>/dev/null | wc -l
