# ClothPPO: A Proximal Policy Optimization Enhancing Framework for Robotic Cloth Manipulation with Observation-Aligned Action Spaces.

International Joint Conference  on Artificial  Intelligence 2024

[Project Page](vpx-ecnu.github.io/ClothPPO-website/) | [Video](https://www.bilibili.com/video/BV1dmWpekE6s/) | [Arxiv](https://arxiv.org/abs/2405.04549)

This repository contains code for training and evaluating in simulation for **Ubuntu 20.04**. It has been tested on machines with **Nvidia GeForce RTX 4090.**

# Simulation

## Method 1: Compiling the simulator

The installation of simulation can refer to [cloth-funnels](https://github.com/real-stanford/cloth-funnels?tab=readme-ov-file#simulation).

## Method 2: Use the Docker image we provide.

If you are familiar with Docker, we have also open-sourced the Docker image for ClothPPO. You can use ClothPPO through Docker without the need to compile the simulation or configure the conda environment yourself.

```bash
docker pull elcarimqaq/clothppo 
```

# Model

The model checkpoint is shared via [Baidu Netdisk](https://pan.baidu.com/s/147WW6XvGf24xrW8gzq4GCA)   Access code: `nwii`

# Evaluate

```bash
 . ./eval_ppo.sh clothppo
```

# Train

ClothPPO was trained on a single RTX 4090 GPU. You can modify the training script we provide, `train_ppo.sh`, as needed.

```bash
. ./train_ppo.sh 
```

# Acknowledgements

* This codebase is heavily built on on [cloth-funnels](https://github.com/real-stanford/cloth-funnels).
* The cloth simulator is a fork of [PyFlex](https://github.com/YunzhuLi/PyFleX) from [Softgym](https://github.com/Xingyu-Lin/softgym)

If you find this codebase useful, consider citing:

```
@inproceedings{ijcai2024p762,
      title     = {ClothPPO: A Proximal Policy Optimization Enhancing Framework for Robotic Cloth Manipulation with Observation-Aligned Action Spaces},
      author    = {Yang, Libing and Li, Yang and Chen, Long},
      booktitle = {Proceedings of the Thirty-Third International Joint Conference on
                   Artificial Intelligence, {IJCAI-24}},
      publisher = {International Joint Conferences on Artificial Intelligence Organization},
      editor    = {Kate Larson},
      pages     = {6895--6903},
      year      = {2024},
      month     = {8},
      note      = {Main Track},
      doi       = {10.24963/ijcai.2024/762},
      url       = {https://doi.org/10.24963/ijcai.2024/762},
    }
```
