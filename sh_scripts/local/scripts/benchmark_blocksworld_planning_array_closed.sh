#!/bin/bash

set -euo pipefail

models=(
  gpt-4.1
  gpt-4.1-nano
)

splits=("simple" "medium" "hard")

# Default values for flags
GPU_RENDERING=true
ENUMERATE_INITIAL_STATE=true
SEED=1
ENUM_BATCH_SIZE=4
FAIL_PROBABILITY=0.0
PROMPT_PATH="data/prompts/benchmark/blocksworld/prompt.md"
EXPERIMENT_NAME=""

# Parse optional command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu_rendering)
      GPU_RENDERING="$2"
      shift 2
      ;;
    --enumerate_initial_state)
      ENUMERATE_INITIAL_STATE="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --enum_batch_size)
      ENUM_BATCH_SIZE="$2"
      shift 2
      ;;
    --fail_probability)
      FAIL_PROBABILITY="$2"
      shift 2
      ;;
    --use_cot_prompt)
      PROMPT_PATH="data/prompts/benchmark/blocksworld/prompt_cot.md"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

mkdir -p ./slurm
source activate viplan_env

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN_FILE="data/planning/blocksworld/domain.pddl"

# Loop through all combinations of model and split
for MODEL in "${models[@]}"; do
  model_short="${MODEL##*/}"

  for PROBLEM_SPLIT in "${splits[@]}"; do
    echo "=== Running MODEL: ${MODEL}, SPLIT: ${PROBLEM_SPLIT}, SEED: ${SEED} ==="

    PROBLEMS_DIR="data/planning/blocksworld/problems/${PROBLEM_SPLIT}"

    if [ -n "$EXPERIMENT_NAME" ]; then
      OUTPUT_DIR="results/planning/blocksworld/${EXPERIMENT_NAME}/predicates/${PROBLEM_SPLIT}/${model_short}"
    else
      OUTPUT_DIR="results/planning/blocksworld/${PROBLEM_SPLIT}/predicates/${model_short}"
    fi

    gpu_flag=""
    if [[ "$GPU_RENDERING" == "true" || "$GPU_RENDERING" == true ]]; then
      gpu_flag="--gpu_rendering"
    fi

    enumerate_flag=""
    if [[ "$ENUMERATE_INITIAL_STATE" == "true" || "$ENUMERATE_INITIAL_STATE" == true ]]; then
      enumerate_flag="--enumerate_initial_state"
    fi

    python3 -m viplan.experiments.benchmark_blocksworld_plan \
      --model_name "$MODEL" \
      --log_level "debug" \
      --root_path "$ROOT_PATH" \
      --prompt_path "$PROMPT_PATH" \
      --domain_file "$DOMAIN_FILE" \
      --problems_dir "$PROBLEMS_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --enum_batch_size "$ENUM_BATCH_SIZE" \
      --fail_probability "$FAIL_PROBABILITY" \
      $gpu_flag $enumerate_flag

  done
done
