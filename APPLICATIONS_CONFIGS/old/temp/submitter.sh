#!/usr/bin/env bash

set -euo pipefail

# Root directory that contains the job scripts to submit.
# Edit this path if you move/copy the folder.
ROOT_DIR="/u/yjian1/project/MLFF-distill/APPLICATIONS_CONFIG/temp"

# If set to 1, prints sbatch commands without submitting.
DRY_RUN=0

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi

shopt -s nullglob

scripts=( "$ROOT_DIR"/*.sh )
if (( ${#scripts[@]} == 0 )); then
  echo "No .sh files found under: $ROOT_DIR"
  exit 0
fi

submitter_realpath="$(readlink -f "$0" 2>/dev/null || true)"

count=0
for script in "${scripts[@]}"; do
  script_realpath="$(readlink -f "$script" 2>/dev/null || true)"

  # Skip this submitter script (even if invoked via relative path)
  if [[ -n "$submitter_realpath" && -n "$script_realpath" && "$script_realpath" == "$submitter_realpath" ]]; then
    continue
  fi
  # Also skip by basename as a fallback
  if [[ "$(basename "$script")" == "submitter.sh" ]]; then
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: sbatch \"$script\""
  else
    echo "Submitting: $script"
    sbatch "$script"
  fi
  ((count+=1))
done

echo "Done. Processed $count scripts."
