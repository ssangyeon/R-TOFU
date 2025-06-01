# R-TOFU: Unlearning in Large Reasoning Models
This repository is the official implementation for the paper: **R-TOFU: Unlearning in Large Reasoning Models.**


<p align="center">
  <a href="https://ai-isl.github.io/r-tofu"> 🏠 Homepage</a> |
  <a href="https://arxiv.org/abs/2505.15214"> 📜 Paper</a> | 
  <a href="https://huggingface.co/collections/AI-ISL/r-tofu-unlearning-in-large-reasoning-models-6834177e1869c47bd0c787ce"> 🤗 Dataset</a>
</p>


## Installation

```shell
conda create -n rtofu python=3.11
conda activate rtofu
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.4.1" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


*All experiments are conducted on eight  NVIDIA L40 GPUs (384 GB total VRAM)*

## Fictitious unlearning scenario

**(1) Fine-tuning the Target Model**

```shell
bash scripts/tofu/finetune.sh
```

**(2) Unlearning the Target Model**

```shell
bash scripts/tofu/unlearn.sh
```

## Acknowledgments

This repository builds upon selected components of the codebase from [A Closer Look at Machine Unlearning for Large Language Models](https://github.com/sail-sg/closer-look-LLM-unlearning). We appreciate their outstanding work!
