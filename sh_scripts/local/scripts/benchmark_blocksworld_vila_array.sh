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

# mamba activate viplan_env
# conda activate viplan
# conda activate viplan_env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate viplan

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
