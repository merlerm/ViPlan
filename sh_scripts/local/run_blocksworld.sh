#!/bin/bash -l

# Default flags
RUN_CLOSED_SOURCE=false
RUN_PREDICATES=true
RUN_VILA=true
RUN_BIG=false

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
        --run_big)
            RUN_BIG="$2"
            shift 2
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

SCRIPT_DIR=$PWD/"sh_scripts/scripts"
echo "Script directory: $SCRIPT_DIR"

if [ ${#forward_args[@]} -gt 0 ]; then
    echo "Forwarding arguments:"
    for a in "${forward_args[@]}"; do echo "  $a"; done
fi

echo "Run closed source: $RUN_CLOSED_SOURCE"
echo "Run predicates:    $RUN_PREDICATES"
echo "Run vila:          $RUN_VILA"
echo "Run big:           $RUN_BIG"

# predicates (planning) benchmarks
if [[ "$RUN_PREDICATES" == "true" ]]; then
    if [[ "$RUN_CLOSED_SOURCE" == "true" ]]; then
        bash "$SCRIPT_DIR/benchmark_blocksworld_planning_array_cpu.sh" "${forward_args[@]}"
    fi
    if [[ "$RUN_BIG" == "true" ]]; then
        bash "$SCRIPT_DIR/benchmark_blocksworld_planning_array_big.sh" "${forward_args[@]}"
    fi
    bash "$SCRIPT_DIR/benchmark_blocksworld_planning_array.sh"     "${forward_args[@]}"
fi

# vila benchmarks
if [[ "$RUN_VILA" == "true" ]]; then
    if [[ "$RUN_CLOSED_SOURCE" == "true" ]]; then
        bash "$SCRIPT_DIR/benchmark_blocksworld_vila_array_cpu.sh" "${forward_args[@]}"
    fi
    if [[ "$RUN_BIG" == "true" ]]; then
        bash "$SCRIPT_DIR/benchmark_blocksworld_vila_array_big.sh" "${forward_args[@]}"
    fi
    bash "$SCRIPT_DIR/benchmark_blocksworld_vila_array.sh"     "${forward_args[@]}"
fi