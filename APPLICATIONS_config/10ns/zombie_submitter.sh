#!/bin/bash
set -euo pipefail

SCRIPT=/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS_config/10ns/short_cut.sh
INTERVAL=$((24*3600))   # 24.2 h = 24 h + 4 min
# only 30s for debug purpose
# INTERVAL=$((60))                        
MAX_RUNS=6

module load micromamba >/dev/null 2>&1 || true  # only if needed
echo "$(date -Iseconds) scheduler started (max ${MAX_RUNS} cycles)"

for ((run=1; run<=MAX_RUNS; run++)); do
    echo "$(date -Iseconds) cycle ${run}/${MAX_RUNS}: launching ${SCRIPT}"
    echo "$(date -Iseconds) launching ${SCRIPT}"
    "${SCRIPT}"
    echo "$(date -Iseconds) cycle ${run}/${MAX_RUNS} done"
    if [ "${run}" -lt "${MAX_RUNS}" ]; then
        echo "$(date -Iseconds) sleeping ${INTERVAL}s before next cycle"
        sleep "${INTERVAL}"
    fi
done

echo "$(date -Iseconds) completed ${MAX_RUNS} cycles; exiting"