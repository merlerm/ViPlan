# Notebooks

Structure and usage:
- **environments:** for testing RL environments, such as ALFRED, which can require special solutions to run on a cluster because of rendering
- **finetune:** for exploring how to fine-tune VLMs
- **dataset_exploration:** this has the scope of determining what info is available for each dataset, visualising samples of it and understanding further dataset splits based on categories if available. Also it should become clear which ability is tested when benchmarking on that dataset
- **preprocessing:** a clean preprocessing pipeline for each experiment included in the benchmark - each notebook should possibly have its python script equivalent and not store all the functionalities in the notebook
- **large_dataset_classes:** used for development notebooks for HF/torch dataset classes for handling large datasets for fine-tuning
- **vlms:** for checking how to initialise and call every VLM we consider in our experiments

Back to [Main Documentation](../README.md).
