#!/bin/bash
# Submit all SLURM job scripts under this directory.
#
# Usage:
#   bash sbatch_submitter.sh           # submit
#   bash sbatch_submitter.sh --dry-run # print what would be submitted

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob globstar
ALL_SH=( "$ROOT_DIR"/**/*.sh )

if (( ${#ALL_SH[@]} == 0 )); then
  echo "No .sh scripts found under: $ROOT_DIR" >&2
  exit 1
fi

is_helper_script() {
  local base
  base="$(basename "$1")"
  [[ "$base" == "sbatch_submitter.sh" ]] && return 0
  [[ "$base" == "local_submitter.sh" ]] && return 0
  [[ "$base" == *"submitter"*".sh" ]] && return 0
  return 1
}

is_slurm_job_script() {
  # Heuristic: contains at least one #SBATCH directive
  grep -qE '^[[:space:]]*#SBATCH[[:space:]]+' "$1"
}

mapfile -t JOBS < <(
  for f in "${ALL_SH[@]}"; do
    [[ -f "$f" ]] || continue
    is_helper_script "$f" && continue
    is_slurm_job_script "$f" || continue
    printf '%s\n' "$f"
  done | LC_ALL=C sort
)

TOTAL=${#JOBS[@]}
if (( TOTAL == 0 )); then
  echo "No SLURM job scripts (with #SBATCH) found under: $ROOT_DIR" >&2
  exit 1
fi

echo "Found $TOTAL SLURM job scripts under:"
echo "  $ROOT_DIR"

if (( DRY_RUN == 1 )); then
  echo
  echo "Dry run. Would submit:"
  for f in "${JOBS[@]}"; do
    echo "  sbatch $f"
  done
  exit 0
fi

echo
for (( i=0; i<TOTAL; i++ )); do
  f="${JOBS[$i]}"
  echo "[$((i+1))/$TOTAL] sbatch $f"
  sbatch "$f"
done
