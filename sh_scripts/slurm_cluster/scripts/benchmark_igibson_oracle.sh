#!/bin/bash -l
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/%A_%a.out
#SBATCH --array=0-14

mkdir -p ./slurm

splits=("simple" "medium" "hard")
max_steps=(10 20 30)

# Map the SLURM_ARRAY_TASK_ID to a model and a problem split.
job_id=$SLURM_ARRAY_TASK_ID
num_splits=${#splits[@]}
split_index=$(( job_id % num_splits ))
max_steps_index=$(( job_id % num_splits ))

PROBLEM_SPLIT="${splits[$split_index]}"
MAX_STEPS="${max_steps[$max_steps_index]}"
PROBLEM_ID=$(( job_id / num_splits ))

# Default values for flags
SEED=1                        # default seed value

# Argument parsing for additional flags (if any)
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
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Handle server logic
NODE=$(hostname -s)
echo "Running server+client on node: $NODE"

JOB_TYPE_IDX=9 # 0 for planning_array, 1 for planning_array_big, 2 for planning_array_cpu, 3-5 for vila_array, 9 for oracle
PORT=$((8000 + SLURM_ARRAY_TASK_ID + 100*JOB_TYPE_IDX)) 

# Start server in background, on this same node
srun --ntasks=1 --gres=gpu:1 --mem=80G --time=18:00:00 -w $NODE \
    make run PORT=$PORT &

echo "Waiting for server to start..."
sleep 120
echo "Server should be running now."

BASE_URL="http://${NODE}:${PORT}"
echo "Testing with BASE_URL=${BASE_URL}"
module load mamba
mamba activate ./viplan_env

# Set file paths.
DOMAIN_FILE="data/planning/igibson/domain.pddl"
PROBLEMS_DIR="data/planning/igibson/${PROBLEM_SPLIT}"

if [ -n "$EXPERIMENT_NAME" ]; then
  OUTPUT_DIR="results/planning/igibson/${EXPERIMENT_NAME}/predicates/${PROBLEM_SPLIT}/oracle"
else
  OUTPUT_DIR="results/planning/igibson/predicates/${PROBLEM_SPLIT}/oracle"
fi

python3 -m viplan.experiments.benchmark_igibson_oracle \
    --base_url "${BASE_URL}" \
    --domain_file  "$DOMAIN_FILE"\
    --problems_dir "$PROBLEMS_DIR" \
    --problem_id "$PROBLEM_ID" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --seed "$SEED" 