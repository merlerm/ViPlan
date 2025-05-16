#!/bin/bash

set -euo pipefail

models=(
  gpt-4.1
  gpt-4.1-nano
)

splits=("simple" "medium" "hard")
max_steps=(10 20 30)

# Default flag values
GPU_RENDERING=true
SEED=1
FAIL_PROBABILITY=0.0
PROMPT_PATH="data/prompts/planning/vila_blocksworld_json.md"
EXPERIMENT_NAME=""

# Parse additional command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu_rendering)
      GPU_RENDERING="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --fail_probability)
      FAIL_PROBABILITY="$2"
      shift 2
      ;;
    --use_cot_prompt)
      PROMPT_PATH="data/prompts/planning/vila_blocksworld_json_cot.md"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

mkdir -p ./slurm

mamba activate viplan_env

# Replace direct conda activation with a more script-friendly approach
# Option 1: Source conda initialization first
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate viplan_env
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate viplan_env
else
  # Option 2: Use the full path to the Python in your environment
  # Change this path to match your actual conda env location
  CONDA_ENV_PYTHON="$HOME/miniconda3/envs/viplan_env/bin/python3"
  # or use conda run as fallback
  echo "Using conda run as fallback"
  PYTHON_CMD="conda run -n viplan_env python3"
fi

# ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH=$PWD
DOMAIN_FILE="data/planning/blocksworld/domain.pddl"

for MODEL in "${models[@]}"; do
  model_short="${MODEL##*/}"
  
  for idx in "${!splits[@]}"; do
    PROBLEM_SPLIT="${splits[$idx]}"
    MAX_STEPS="${max_steps[$idx]}"

    echo "=== Running MODEL: ${MODEL}, SPLIT: ${PROBLEM_SPLIT}, MAX_STEPS: ${MAX_STEPS} ==="

    PROBLEMS_DIR="data/planning/blocksworld/problems/${PROBLEM_SPLIT}"

    if [ -n "$EXPERIMENT_NAME" ]; then
      OUTPUT_DIR="results/planning/blocksworld/${EXPERIMENT_NAME}/vila/${PROBLEM_SPLIT}/${model_short}"
    else
      OUTPUT_DIR="results/planning/blocksworld/${PROBLEM_SPLIT}/vila/${model_short}"
    fi

    # Set GPU flag if needed.
    gpu_flag=""
    if [ "$GPU_RENDERING" = "true" ] || [ "$GPU_RENDERING" = true ]; then
      gpu_flag="--gpu_rendering"
    fi

    python3 -m viplan.experiments.benchmark_blocksworld_vila \
      --model_name "$MODEL" \
      --log_level "debug" \
      --root_path "$ROOT_PATH" \
      --prompt_path "$PROMPT_PATH" \
      --domain_file "$DOMAIN_FILE" \
      --problems_dir "$PROBLEMS_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --max_steps "$MAX_STEPS" \
      --max_new_tokens 3000 \
      --fail_probability "$FAIL_PROBABILITY" \
      $gpu_flag

  done
done
