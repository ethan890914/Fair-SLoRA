# Fair S-LoRA


## Introduction
Low-Rank Adaptation of Large Language Models (LoRA) is a popular way to do parameter fine-tuning efficiently by freezing the original weights and inserting lightweight lowrank matrices (adapters) into specific layers. However, serving multiple LoRA adapters on shared LLM backbones has become a challenge. We propose a skew-aware serving framework to reduce repeat adapter computation under extreme-skew regimes. By detecting popularity within a sliding window, merging the dominant adapter into the base weights, and routing batches through a special forward path with lightweight corrections for unpopular adapters, our design preserves ordinary-case efficiency while targeting the extreme regime that stresses typical serving pipelines.

## Requirements
* CUDA 11.8 compatible GPU
  * Recommended: GPUs from the Ampere family, like the A100, which support bfloat16 operations.
  * Note: Older GPUs from the Turing family like the T4, which do not support bfloat16, are not supported.
* 1.13 <= PyTorch <= 2.0.1

## Installation
```bash
# Optional: create conda environment
conda create -n slora python=3.9
conda activate slora 

# Build the project
pip install torch==2.0.1
pip install -e . --no-build-isolation
```

## Trouble Shooting
* If you see `Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1 None of PyTorch,...`, it means the torch version is incorrect. Installing the package and dependency:
```bash
pip install "transformers<4.41" "accelerate<0.24"
pip install "triton==2.1.0"
pip install "numpy<2"
```

* There might be an assertion for triton, to prevent that, look into `/ext3/miniforge3/envs/{your_conda_env}/lib/python3.9/site-packages/triton/common/build.py` and add the following line before assertion:
```python
if "/usr/local/cuda-11.8/compat" not in dirs:
    dirs.append("/usr/local/cuda-11.8/compat")
```

## Example Run
Quick Run Real model weights on S-LoRA
```bash
cd benchmarks
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real
python run_exp.py --debug --model-setting Real
```
Quick Run Real model weights on **Fair S-LoRA**
```bash
cd benchmarks
python launch_server.py --fair_strategy --num-adapter 100 --num-token 10000 --model-setting Real
python run_exp.py --debug --model-setting Real
```
Run Comprehensive Experiments Suite
```bash
cd benchmarks
python launch_server.py --fair_strategy --num-adapter 100 --num-token 10000 --model-setting Real
python run_exp.py --fair --model-setting Real
```

## Acknowledgment
Fair SLoRA is build on top of [S-LoRA](https://github.com/S-LoRA/S-LoRA).






