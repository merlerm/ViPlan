# Scripts to run the benchmark

The ViPlan benchmark is designed to be run on SLURM clusters, and the scripts in this directory are tailored for that purpose. If you are using a different cluster manager, you may need to modify the scripts accordingly, or directly run the Python scripts in the `viplan/experiments` directory.

> [!IMPORTANT]
> All sh_scripts are designed to be run from the root directory of the repository. (e.g. `cd ViPlan && ./sh_scripts/scripts/run_blocksworld.sh`)

The "big" scripts are designed to run bigger VLMs that require two GPUs and the "cpu" scripts are designed to run API models that don't require GPUs (although a GPU is still requested for the renderer).

The two main entry points are `run_blocksworld.sh` and `run_igibson.sh` (located at `sh_scripts/slurm_cluster` to run on SLURM clusters; at `sh_scripts/local` to run locally), which are designed to run the Blocksworld and Household environments. The parameters are :
- `--experiment_name` argument can be passed to all scripts and specifies a specific name that will be used to save the results
- `--run_predicates` boolean to determine whether to run experiments on the VLM-as-grounder setting
- `--run_vila` boolean to determine whether to run experiments on the VLM-as-planner setting
- `--run_closed_source` bolean to determine whether to run experiments using close-source models

Check the individual scripts for more details on the arguments.


Back to [Main Documentation](../README.md).