#!/usr/bin/env bash
# Retry a push through a transient server-side failure.
# GitHub returned 500 Internal Server Error on a 74-line, single-file commit to a
# 11 MB repository, so the cause is upstream rather than anything local. Backs off
# and keeps trying rather than forcing anything.
set -uo pipefail
cd "$(dirname "$0")/../.."
for i in $(seq 1 60); do
    if [ "$(git log origin/experiments-pipeline..HEAD --oneline 2>/dev/null | wc -l)" = "0" ]; then
        echo "$(date '+%F %T')  nothing to push"; exit 0
    fi
    if git push origin experiments-pipeline 2>&1 | tee -a logs/push_retry.log | grep -q "Internal Server Error"; then
        echo "$(date '+%F %T')  attempt $i: server error, backing off"
    else
        git fetch origin experiments-pipeline -q 2>/dev/null
        if [ "$(git log origin/experiments-pipeline..HEAD --oneline 2>/dev/null | wc -l)" = "0" ]; then
            echo "$(date '+%F %T')  pushed on attempt $i"; exit 0
        fi
    fi
    sleep $(( i < 10 ? 60 : 300 ))
done
echo "$(date '+%F %T')  gave up after 60 attempts"; exit 1
