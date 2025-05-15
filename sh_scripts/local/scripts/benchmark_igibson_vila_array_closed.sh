#!/usr/bin/env bash
set -euo pipefail

# Default flag values
SEED=1
EXPERIMENT_NAME=""
PROMPT_PATH="data/prompts/planning/vila_igibson_json.md"

# Parse additional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --use_cot_prompt)
      PROMPT_PATH="data/prompts/planning/vila_igibson_json_cot.md"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

models=(
  "gpt-4.1"
  "gpt-4.1-nano"
)

splits=("simple" "medium" "hard")
max_steps=(10 20 30)

# Server/job constants
JOB_TYPE_IDX=5
job_id=

mamba activate viplan_env

for MODEL in "${models[@]}"; do
  model_short="${MODEL##*/}"

  for idx in "${!splits[@]}"; do
    PROBLEM_SPLIT="${splits[$idx]}"
    MAX_STEPS="${max_steps[$idx]}"
    echo "=== Running MODEL: ${MODEL}, SPLIT: ${PROBLEM_SPLIT}, MAX_STEPS: ${MAX_STEPS} ==="

    PORT=$((8000 + job_id + 100 * JOB_TYPE_IDX))
    NODE=$(hostname -s)
    BASE_URL="http://${NODE}:${PORT}"

    echo "------------------------------------------------------------"
    echo "Job #$job_id: model=${model_short}, split=${PROBLEM_SPLIT}, max_steps=${MAX_STEPS}"
    echo "Starting server on port ${PORT}…"
    make run PORT=${PORT} &
    sleep 120  # give server time to spin up

    # Paths
    DOMAIN_FILE="data/planning/igibson/domain.pddl"
    PROBLEMS_DIR="data/planning/igibson/${PROBLEM_SPLIT}"

    if [[ -n "${EXPERIMENT_NAME}" ]]; then
      OUTPUT_DIR="results/planning/igibson/${EXPERIMENT_NAME}/vila/${PROBLEM_SPLIT}/${model_short}"
    else
      OUTPUT_DIR="results/planning/igibson/${PROBLEM_SPLIT}/vila/${model_short}"
    fi
    mkdir -p "${OUTPUT_DIR}"

    # Run the benchmark
    python3 -m viplan.experiments.benchmark_igibson_vila \
      --base_url "${BASE_URL}" \
      --model_name "${MODEL}" \
      --domain_file "${DOMAIN_FILE}" \
      --problems_dir "${PROBLEMS_DIR}" \
      --prompt_path "${PROMPT_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --max_steps "${MAX_STEPS}" \
      --seed "${SEED}"

    # clean up the server before next job
    echo "Stopping server on port ${PORT}…"
    pkill -f "make run.*PORT=${PORT}" || true

    job_id=$(( job_id + 1 ))
  done
done

echo "All ${job_id} jobs complete."