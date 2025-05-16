#!/bin/bash

set -euo pipefail

# No distinction between big and small models as we assume local resources are the same
models=(
  "OpenGVLab/InternVL3-8B"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "google/gemma-3-12b-it"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "allenai/Molmo-7B-D-0924"
  "microsoft/Phi-4-multimodal-instruct"
  "llava-hf/llava-onevision-qwen2-7b-ov-hf"
  "deepseek-ai/deepseek-vl2"
  "CohereLabs/aya-vision-8b"
  "OpenGVLab/InternVL3-78B"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "google/gemma-3-27b-it"
  "llava-hf/llava-onevision-qwen2-72b-ov-hf"
  "CohereLabs/aya-vision-32b"
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

    # Use the appropriate Python command based on initialization method
    if [[ -z "${PYTHON_CMD:-}" ]]; then
      # If conda was properly activated
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
    else
      # If using conda run
      $PYTHON_CMD -m viplan.experiments.benchmark_blocksworld_plan \
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
    fi

  done
done
