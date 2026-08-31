#!/usr/bin/env bash
# Fetch + unpack the D3IL demonstration archive (~16 GB, Google Drive).
#
# Downloads to SCRATCH, not home: home has a ~50 GB quota that is invisible to
# `quota -s`, and a 7.9 GB partial download was enough to fill it and make the
# filesystem read-only mid-job. scratch has ~700 GB free after the HPTD
# checkpoint prune.
#
# Kept as a file rather than an inline `bash -c`: the SLURM wrapper runs under
# `set -euo pipefail`, so an inline `$?` check aborted the job before it could
# report anything, and the nested quoting swallowed gdown's error output.
#
#   sbatch scripts/slurm/cpu_job.sh bash scripts/maint/fetch_d3il_data.sh
set -uo pipefail          # deliberately NOT -e: we want failures reported

FILE_ID=1SQhbhzV85zf_ltnQ8Cbge2lsSWInxVa8
ROOT=/shared/scratch/SCWF00111/d3il_data
OUT="$ROOT/d3il_data.zip"
DEST="$ROOT/data"

mkdir -p "$ROOT"
echo "=== target: $OUT ==="
lfs quota -h -g SCWF00111 /shared/scratch 2>/dev/null | tail -1

echo "=== attempt 1: gdown ==="
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUT" --continue
echo "gdown exit=$?"

if [ ! -s "$OUT" ]; then
    echo "=== attempt 2: gdown --fuzzy on the sharing URL ==="
    gdown --fuzzy "https://drive.google.com/file/d/${FILE_ID}/view" -O "$OUT" --continue
    echo "gdown fuzzy exit=$?"
fi

if [ ! -s "$OUT" ]; then
    echo "FAILED: no archive downloaded."
    echo "Drive rate-limits large public files; if that is the cause the archive"
    echo "has to be fetched by hand and copied to $OUT."
    exit 1
fi

ls -lh "$OUT"
echo "=== unzip -> $DEST ==="
mkdir -p "$DEST"
unzip -q -o "$OUT" -d "$DEST"
echo "unzip exit=$?"
du -sh "$DEST"
ls "$DEST" | head -20
