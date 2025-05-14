# ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models

This codebase contains the implementation of the ViPlan benchmark. 

## Project structure
The project is divided into the following main sections:
- Source code: [viplan](viplan/README.md)
- Notebooks: [notebooks](notebooks/README.md) (mostly to visualize results)
- SLURM scripts: [sh_scripts](sh_scripts/README.md)
- Data: [data](data/README.md)

## Installation

The ViPlan benchmark is made up of several components, including the main experiment code and specific code for the two environments (Blocksworld and Household).

### Experiments

To run the experiments, you need to install the required packages. We recommend using mamba and provide an environment file for easy installation. The virtual environment requirements can be found at `environment.yml`, and it can be created as prefered. Here we report examples using `mamba` and `conda`.
Using `mamba`:
```
mamba env create -p ./viplan_env -f environment.yml
mamba activate ./viplan_env
```
Using `conda`:
```
conda env create -f environment.yml
conda activate viplan_env
```

If you wish to use Flash Attention, it needs to be installed separately with the following command:

```bash
pip install flash-attn --no-build-isolation
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
```

### Environments

The Blocksworld environment is based on the [Photorealistic Blocksworld](https://github.com/IBM/photorealistic-blocksworld) renderer, which is based on Blender. To install the Blender-based renderer, from the root directory of the repository, run the following commands:

```bash
wget https://download.blender.org/release/Blender3.0/blender-3.0.0-linux-x64.tar.xz
tar xf blender-3.0.0-linux-x64.tar.xz
echo $PWD > $(echo blender*/3.*/python/lib/python*/site-packages/)clevr.pth
rm blender-3.0.0-linux-x64.tar.xz
```

#### iGibson

##### Requirements
Here is the list of specific requirements to use iGibson:
- `apptainer` (former Singularity)
- Encription key to be requested at [this link](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform)


The Household environment is instead based on a custom version of [iGibson](https://github.com/StanfordVL/iGibson). 
To install the environment, first clone our fork of iGibson:

```bash
git clone --depth 1 https://github.com/nicoladainese96/iGibson.git ./iGibson --recursive
git clone https://github.com/StanfordVL/behavior.git
```

Since iGibson requires specific packages, we recommend running it inside a container. Our code is designed to work with [Apptainer](https://apptainer.org). To pull the image, run:

```bash
apptainer cache clean
apptainer pull docker://igibson/igibson:latest
```
This will create a file called `igibson_latest.sif`, which is expected to be in the root directory. This file is a Singularity image that contains all the dependencies needed to run iGibson. To open a shell inside the container run:
```bash
apptainer exec --nv igibson_latest.sif bash
```

Then, install the iGibson dependencies from inside the container:

```bash
python -m venv --system-site-packages ./igibson_env
source igibson_env/bin/activate
pip install -e ./iGibson
pip install -e ./behavior
pip install notebook pyquaternion shapely uvicorn fastapi unified_planning
pip install unified_planning[engines]
```

Afterwards, the iGibson custom assets need to be downloaded and decrypted using the encryption key provided by the iGibson team, following the instructions at [this page](https://stanfordvl.github.io/iGibson/dataset.html):

To download the assets, run:

```bash
cd iGibson
wget https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz
tar -xzvf ig_dataset.tar.gz -C ./igibson/data
```

As some of the assets are encrypted, you will need to decrypt them using the key provided by the iGibson team. The key can be requested by filling out the form at [this link](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform) and then needs to be placed inside the `iGibson` folder under `igibson/data/igibson.key`.

After this, the iGibson environment is ready to be used. For the benchmark, we use a client-server architecture, where the server runs inside the container and the client runs in the main execution environment. Scripts are provided in the `sh_scripts` folder to run the server and the client.

## Benchmark

To run the benchmark, we provide bash scripts to run locally as well as SLURM scripts that can be used to run the experiments on a cluster. The scripts are located in the `sh_scripts` folder. If you are using a different cluster manager, you may need to modify the scripts at `sh_scripts/slurm_cluster` accordingly. You could also directly run the Python scripts in the `viplan/experiments` directory.

In order to run some open-source models, you might need to accept their conditions on the huggingface hub. Then, you can include your token in the bash environment by running the following command:

```bash
export HF_TOKEN=<your_token>
```
Similarly, in order to run closed-source models, include your API key in the bash environment by running the following command:

```bash
export OPENAI_API_KEY=<your_key>
export GEMINI_API_KEY=<your_key>
export ANTHROPIC_API_KEY=<your_key>
```