# Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models

Official implementation of the paper: **Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models**.

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://kaizhao.net/sub-jepa)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/intcomp/Sub-JEPA)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.09241)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-HuggingFace-ffcc4d?logo=huggingface&logoColor=yellow)](https://huggingface.co/intcomp/sub-jepa)


---

## Overview

Joint-Embedding Predictive Architectures (JEPAs) offer an effective framework for learning world models, but without sufficient constraints, their representation spaces can easily collapse. While recent methods like LeWM prevent this by enforcing a strong isotropic Gaussian prior in the high-dimensional ambient space, this can introduce an overly strong structural bias.

**Sub-JEPA** relaxes this global constraint by applying Gaussian regularization across multiple random subspaces instead of the original embedding space. 


<div align="center">
  <p>
    <img src="https://kaizhao.net/images/projects/sub-jepa/two-room.gif" alt="Two-Room Labeled" width="49%" />
    <img src="https://kaizhao.net/images/projects/sub-jepa/reacher.gif" alt="PushT Labeled" width="49%" />
  </p>
  <p>
    <img src="https://kaizhao.net/images/projects/sub-jepa/cube.gif" alt="Cube" width="49%" />
    <img src="https://kaizhao.net/images/projects/sub-jepa/pusht.gif" alt="Reacher" width="49%" />
  </p>
</div>

## Main Result
Sub-JEPA consistently improves over LeWM across four continuous-control environments.

| Method   |         Two-Room |          Reacher |            PushT |         OGB-Cube |
| -------- | ---------------: | ---------------: | ---------------: | ---------------: |
| LeWM     |     84.33 ± 4.23 |     82.67 ± 4.42 |     84.67 ± 6.53 |     67.33 ± 5.01 |
| Sub-JEPA | **95.00 ± 2.76** | **84.00 ± 4.00** | **89.00 ± 5.33** | **76.33 ± 5.99** |


---

## Repository Layout

```
Sub-JEPA/
├── le-wm/                 # Upstream LeWM codebase as a git submodule
├── subjepa.py             # MultiSubspaceSIGReg implementation
├── lewm_subjepa.patch     # Patch that integrates Sub-JEPA into LeWM
└── README.md
```

## Installation

### 1. Clone the repository
```bash
git clone --recursive https://github.com/intcomp/Sub-JEPA.git

cd Sub-JEPA
```

If you forgot `--recursive`, you can run:
```bash
git submodule update --init --recursive
```

### 2. Apply the Sub-JEPA patch to LeWM
```bash
git -C le-wm apply ../lewm_subjepa.patch
```

### 3. Set up the environment + data (same as LeWM)
Follow the upstream LeWM instructions:

- Installation: https://github.com/lucas-maes/le-wm#using-the-code
- Data layout: https://github.com/lucas-maes/le-wm#data

In particular, LeWM (and this repo) expects datasets and checkpoints under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`).

## Pretrained Checkpoints

We provide pretrained Sub-JEPA checkpoints on [Hugging Face](https://huggingface.co/intcomp/sub-jepa).

You can download all released checkpoints with:

```bash
pip install -U huggingface_hub

hf download intcomp/sub-jepa
```

## Train

Training is identical to LeWM and is configured with Hydra. The patch modifies `le-wm/train.py` and `le-wm/config/train/lewm.yaml` to use multi-subspace regularization.

```bash
PYTHONPATH=. python le-wm/train.py data=tworoom
```

Make sure to set your WandB `entity` and `project` in `le-wm/config/train/lewm.yaml`, or disable WandB:

```bash
PYTHONPATH=. python le-wm/train.py data=tworoom wandb.enabled=false
```

### Sub-JEPA hyperparameters

All Sub-JEPA knobs live under `loss.sigreg` in `le-wm/config/train/lewm.yaml`.

| Hyperparameter                     | Description                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| `loss.sigreg.weight`               | Overall regularization weight. Same role as the LeWM SIGReg weight.                    |
| `loss.sigreg.kwargs.num_subspaces` | Number of subspaces, denoted as $K$ in the paper.                                      |
| `loss.sigreg.kwargs.subspace_dim`  | Dimension of each subspace $d_s$. If `null`, uses `embed_dim / K` and requires divisibility. |
| `loss.sigreg.kwargs.init_mode`     | Projection initialization mode.                                                        |
| `loss.sigreg.theta`                | Soft orthogonality penalty weight. Only used by trainable projection variants.         |


## Reproducing Main Results
To easily reproduce the main results presented in our paper across different environments, use the following commands. 
Note the specific num_subspaces configurations tailored for each task.

### Tworoom

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python le-wm/train.py \
  data=tworoom \
  subdir=tworoom/subjepa \
  loss.sigreg.kwargs.init_mode=orthogonal_frozen \
  loss.sigreg.kwargs.num_subspaces=32 \
  trainer.max_epochs=10
```

### Reacher

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python le-wm/train.py \
  data=dmc \
  subdir=dmc/subjepa \
  loss.sigreg.kwargs.init_mode=orthogonal_frozen \
  loss.sigreg.kwargs.num_subspaces=32 \
  trainer.max_epochs=10
```

### PushT

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python le-wm/train.py \
  data=pusht \
  subdir=pusht/subjepa \
  loss.sigreg.kwargs.init_mode=orthogonal_frozen \
  loss.sigreg.kwargs.num_subspaces=16 \
  trainer.max_epochs=10
```

### OGB-Cube

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python le-wm/train.py \
  data=ogb \
  subdir=cube/subjepa \
  loss.sigreg.kwargs.init_mode=orthogonal_frozen \
  loss.sigreg.kwargs.num_subspaces=32 \
  trainer.max_epochs=10
```

## Evaluation

Evaluation configs are located under `le-wm/config/eval/`.

```bash
python le-wm/eval.py --config-name=tworoom.yaml policy=tworoom/subjepa
```

`policy` must be the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix.

Example: Full evaluation with our paper's default seeds:

```bash
python le-wm/eval.py --config-name=tworoom.yaml policy=tworoom/subjepa seed=42,100,2026,3407,1234,4444 --multirun
```

## Acknowledgements

This codebase is built on top of the official [LeWorldModel](https://github.com/lucas-maes/le-wm) implementation.
We thank the authors of LeWM for releasing their codebase.

## Citation

If you find our work useful in your research, please consider citing:
```bibtex
@misc{zhao2026subjepa,
  title        = {Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models},
  author       = {Zhao, Kai and Nie, Dongliang and Lin, Yuchen and Luo, Zhehan and Gu, Yixiao and Fan, Deng-Ping and Zeng, Dan},
  year         = {2026},
  eprint       = {2605.09241},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2605.09241}
}
```
