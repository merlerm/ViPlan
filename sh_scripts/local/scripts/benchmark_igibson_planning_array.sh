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

# Default values for flags
SEED=1
ENUM_BATCH_SIZE=4
PROMPT_PATH="data/prompts/benchmark/igibson/prompt.md"
EXPERIMENT_NAME=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --use_cot_prompt)
      PROMPT_PATH="data/prompts/benchmark/igibson/prompt_cot.md"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

NODE=$(hostname -s)
echo "Running on node: $NODE"

mamba activate ./viplan_env

JOB_TYPE_IDX=0
port_base=8000
job_id=0

for MODEL in "${models[@]}"; do
  model_short="${MODEL##*/}"
  
  for idx in "${!splits[@]}"; do
    PROBLEM_SPLIT="${splits[$idx]}"
    MAX_STEPS="${max_steps[$idx]}"
    
    PORT=$((port_base + job_id + 100 * JOB_TYPE_IDX))
    BASE_URL="http://${NODE}:${PORT}"
    
    echo "=== Running MODEL: ${MODEL}, SPLIT: ${PROBLEM_SPLIT}, MAX_STEPS: ${MAX_STEPS} ==="

    make run PORT=$PORT & 
    sleep 120

    DOMAIN_FILE="data/planning/igibson/domain.pddl"
    PROBLEMS_DIR="data/planning/igibson/${PROBLEM_SPLIT}"

    if [ -n "$EXPERIMENT_NAME" ]; then
      OUTPUT_DIR="results/planning/igibson/${EXPERIMENT_NAME}/predicates/${PROBLEM_SPLIT}/${model_short}"
    else
      OUTPUT_DIR="results/planning/igibson/predicates/${PROBLEM_SPLIT}/${model_short}"
    fi

    python3 -m viplan.experiments.benchmark_igibson_plan \
      --base_url "${BASE_URL}" \
      --model_name "${MODEL}" \
      --domain_file "$DOMAIN_FILE" \
      --problems_dir "$PROBLEMS_DIR" \
      --prompt_path "$PROMPT_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --max_steps "$MAX_STEPS" \
      --seed "$SEED" \
      --enum_batch_size "$ENUM_BATCH_SIZE"

    job_id=$((job_id + 1))
  done
done
