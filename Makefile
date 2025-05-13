.PHONY: shell run

# default port if none given on CLI
PORT ?= 8000

shell:
	srun --gpus=1 --mem=40GB --time 04:00:00 \
	  --pty apptainer exec --nv ./igibson_latest.sif bash

run:
	apptainer exec --nv ./igibson_latest.sif \
	  bash ./iGibson/sim_server/start_server.sh --port $(PORT)

