#!/bin/bash -l
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --output=./slurm/%A_%a.out
#SBATCH --array=0-14

mkdir -p ./slurm

# Define the list of models and problem splits.
models=(
  "OpenGVLab/InternVL3-78B"
  "Qwen/Qwen2.5-VL-72B-Instruct"
  "google/gemma-3-27b-it"
  "llava-hf/llava-onevision-qwen2-72b-ov-hf"
  "CohereLabs/aya-vision-32b"
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
SEED=1
PROMPT_PATH="data/prompts/planning/vila_igibson_json.md"

# Parse additional arguments.
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

# Handle server logic
NODE=$(hostname -s)
echo "Running server+client on node: $NODE"

JOB_TYPE_IDX=4 # 0-2 for plan_array, 3 for vila_array, 4 for vila_array_big, 5 for vila_array_cpu

PORT=$((8000 + SLURM_ARRAY_TASK_ID + 100*JOB_TYPE_IDX)) 

# Start server in background, on this same node
srun --ntasks=1 --gres=gpu:1 --mem=80G --time=18:00:00 -w $NODE \
    make run PORT=$PORT &

echo "Waiting for server to start..."
sleep 120
echo "Server should be running now."

BASE_URL="http://${NODE}:${PORT}"
echo "Testing with BASE_URL=${BASE_URL}"

mamba activate viplan_env

# Set file paths.
DOMAIN_FILE="data/planning/igibson/domain.pddl"
PROBLEMS_DIR="data/planning/igibson/${PROBLEM_SPLIT}"

if [ -n "$EXPERIMENT_NAME" ]; then
  OUTPUT_DIR="results/planning/igibson/${EXPERIMENT_NAME}/vila/${PROBLEM_SPLIT}/${model_short}"
else
  OUTPUT_DIR="results/planning/igibson/${PROBLEM_SPLIT}/vila/${model_short}"
fi

python3 -m viplan.experiments.benchmark_igibson_vila \
    --base_url "${BASE_URL}" \
    --model_name "${MODEL}" \
    --domain_file  "$DOMAIN_FILE"\
    --problems_dir "$PROBLEMS_DIR" \
    --prompt_path "$PROMPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $MAX_STEPS \
    --tensor_parallel_size 2 \
    --seed $SEED
