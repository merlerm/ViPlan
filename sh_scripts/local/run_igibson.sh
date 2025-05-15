#!/bin/bash -l

# Default flags
RUN_CLOSED_SOURCE=false
RUN_PREDICATES=false
RUN_VILA=true
RUN_ON_SLURM=false

# collect other args
forward_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_closed_source)
            RUN_CLOSED_SOURCE=true
            shift
            ;;
        --run_predicates)
            RUN_PREDICATES="$2"
            shift 2
            ;;
        --run_vila)
            RUN_VILA="$2"
            shift 2
            ;;
        --run_on_slurm)
            RUN_ON_SLURM=true
            shift
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

# If run_locally is set, add it to forward_args
if [[ "$RUN_ON_SLURM" == "true" ]]; then
    forward_args+=("--run_on_slurm")
fi

SCRIPT_DIR=$PWD/"sh_scripts/scripts"
echo "Script directory: $SCRIPT_DIR"

if [ ${#forward_args[@]} -gt 0 ]; then
    echo "Forwarding arguments:"
    for a in "${forward_args[@]}"; do echo "  $a"; done
fi

echo "Run closed source: $RUN_CLOSED_SOURCE"
echo "Run predicates:    $RUN_PREDICATES"
echo "Run vila:          $RUN_VILA"
echo "Run on slurm:      $RUN_ON_SLURM"

# predicates (planning) benchmarks
if [[ "$RUN_PREDICATES" == "true" ]]; then
    if [[ "$RUN_CLOSED_SOURCE" == "true" ]]; then
        bash "$SCRIPT_DIR/benchmark_igibson_planning_array_cpu.sh" "${forward_args[@]}"
    fi
    bash "$SCRIPT_DIR/benchmark_igibson_planning_array_big.sh" "${forward_args[@]}"
    bash "$SCRIPT_DIR/benchmark_igibson_planning_array.sh"     "${forward_args[@]}"
fi

# vila benchmarks
if [[ "$RUN_VILA" == "true" ]]; then
    if [[ "$RUN_CLOSED_SOURCE" == "true" ]]; then
        bash "$SCRIPT_DIR/benchmark_igibson_vila_array_cpu.sh" "${forward_args[@]}"
    fi
    # bash "$SCRIPT_DIR/benchmark_igibson_vila_array_big.sh" "${forward_args[@]}"
    echo "Running Vila array"
    bash "$SCRIPT_DIR/benchmark_igibson_vila_array.sh"     "${forward_args[@]}"
fi