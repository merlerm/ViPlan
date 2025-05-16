#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=./slurm/%A_%a.out
#SBATCH --array=0-5

# GPU only used for rendering, not for LLM inference

# Define the list of models and problem splits.
models=(
  "gpt-4.1"
  "gpt-4.1-nano"
)
splits=("simple" "medium" "hard")
max_steps=(10 20 30)

# Map the SLURM_ARRAY_TASK_ID to a model and a problem split.
job_id=$SLURM_ARRAY_TASK_ID
num_splits=${#splits[@]}
model_index=$(( job_id / num_splits ))
split_index=$(( job_id % num_splits ))
max_steps_index=$(( job_id % num_splits ))

MODEL="${models[$model_index]}"
PROBLEM_SPLIT="${splits[$split_index]}"
MAX_STEPS="${max_steps[$max_steps_index]}"
model_short="${MODEL##*/}"

# Default flag values.
GPU_RENDERING=true
SEED=1
FAIL_PROBABILITY=0.0
PROMPT_PATH="data/prompts/planning/vila_blocksworld_json.md"

# Parse additional arguments.
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

echo "Running $MODEL on problem split $PROBLEM_SPLIT with seed $SEED and max steps $MAX_STEPS"

# Set file paths.
ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAIN_FILE="data/planning/blocksworld/domain.pddl"
PROBLEMS_DIR="data/planning/blocksworld/problems/${PROBLEM_SPLIT}"

if [ -n "$EXPERIMENT_NAME" ]; then
  OUTPUT_DIR="results/planning/blocksworld/${EXPERIMENT_NAME}/vila/${PROBLEM_SPLIT}/${model_short}"
else
  OUTPUT_DIR="results/planning/blocksworld/${PROBLEM_SPLIT}/vila/${model_short}"
fi

mkdir -p ./slurm

# Load necessary modules and activate the environment.
module load mamba
mamba activate ./viplan_env

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
  --max_steps $MAX_STEPS \
  --max_new_tokens 4096 \
  --fail_probability $FAIL_PROBABILITY \
  $gpu_flag