#!/bin/bash
# Run all MSD jobs sequentially on the local CPU (no sbatch).
# #SBATCH directives in each script are treated as comments by bash.

set -uo pipefail

cd /u/yjian1/project/MLFF-distill
source /u/yjian1/project/Generative_3d_turbulence_flow/yjian1/env/miniconda3/bin/activate fairchemV2

SCRIPT_DIR="APPLICATIONS_CONFIG/ablation_on_factors_after_temperature/eval/outlier_ablation_Feb_2026"

# ── collect jobs ──────────────────────────────────────────────────────────────
mapfile -t JOBS < <(ls "$SCRIPT_DIR"/msd_*.sh | sort)
TOTAL=${#JOBS[@]}

if (( TOTAL == 0 )); then
  echo "No msd_*.sh scripts found in $SCRIPT_DIR" >&2
  exit 1
fi

# ── progress bar helper ───────────────────────────────────────────────────────
BAR_WIDTH=40
progress_bar() {
  local done=$1 total=$2 label=$3
  local filled=$(( done * BAR_WIDTH / total ))
  local empty=$(( BAR_WIDTH - filled ))
  local pct=$(( done * 100 / total ))
  local bar
  bar="$(printf '%0.s#' $(seq 1 $filled))$(printf '%0.s-' $(seq 1 $empty))"
  printf "\r[%s] %d/%d (%3d%%)  %s" "$bar" "$done" "$total" "$pct" "$label"
}

# ── run jobs ──────────────────────────────────────────────────────────────────
PASSED=0
FAILED=0
FAILED_NAMES=()

OVERALL_START=$(date +%s)

echo "Running $TOTAL MSD jobs locally (sequential, CPU)"
echo "──────────────────────────────────────────────────────────"

for (( i=0; i<TOTAL; i++ )); do
  script="${JOBS[$i]}"
  name="$(basename "$script")"
  done_so_far=$i

  progress_bar "$done_so_far" "$TOTAL" "$name"

  JOB_START=$(date +%s)
  if bash "$script" > "${script%.sh}.local.log" 2>&1; then
    status="OK"
    (( PASSED++ )) || true
  else
    status="FAILED"
    (( FAILED++ )) || true
    FAILED_NAMES+=("$name")
  fi
  JOB_END=$(date +%s)
  elapsed=$(( JOB_END - JOB_START ))

  # overwrite progress line with a completed status line
  progress_bar "$(( i+1 ))" "$TOTAL" "$name"
  printf "  [%s] %dm%02ds\n" "$status" "$(( elapsed/60 ))" "$(( elapsed%60 ))"
done

# ── summary ───────────────────────────────────────────────────────────────────
OVERALL_END=$(date +%s)
total_elapsed=$(( OVERALL_END - OVERALL_START ))

echo "──────────────────────────────────────────────────────────"
printf "Done in %dh %02dm %02ds\n" \
  "$(( total_elapsed/3600 ))" \
  "$(( (total_elapsed%3600)/60 ))" \
  "$(( total_elapsed%60 ))"
echo "  Passed : $PASSED / $TOTAL"
echo "  Failed : $FAILED / $TOTAL"
if (( FAILED > 0 )); then
  echo "  Failed jobs:"
  for n in "${FAILED_NAMES[@]}"; do
    echo "    - $n  (log: $SCRIPT_DIR/${n%.sh}.local.log)"
  done
  exit 1
fi
