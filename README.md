# Adapting a Language Model While Preserving its General Knowledge

This repository contains the code and pre-trained models for our EMNLP'22 paper [Adapting a Language Model While Preserving its General Knowledge](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.693.pdf) by <a href="https://vincent950129.github.io/"> Zixuan Ke</a>, <a href="https://shaoyijia.github.io/">Yijia Shao</a>, <a href="https://linhaowei1.github.io/">Haowei Lin</a>, <a href="https://howardhsu.github.io/">Hu Xu</a>, <a href="https://leishu02.github.io/">Lei Shu</a>, and <a href="https://www.cs.uic.edu/~liub/">Bing Liu</a>.


## Quick Links

  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Use DGA with Huggingface](#use-cpt-with-huggingface)
  - [Train DGA](#train-cpt)
    - [Data](#data)
    - [Post-Training](#post-training)
    - [End-Task Fine-tuning](#end-task-fine-tuning)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Overview

Domain-adaptive pre-training (or DA-training for short), also known as post-training, aims to train a pre-trained general-purpose language model (LM) using an unlabeled corpus of a particular domain to adapt the LM so that endtasks in the domain can give improved performances. However, existing DA-training methods are in some sense blind as they do not explicitly identify what knowledge in the LM should be preserved and what should be changed by the domain corpus. This paper shows that the existing methods are suboptimal and proposes a novel method to perform a more informed adaptation of the knowledge in the LM by (1) soft-masking the attention heads based on their importance to best preserve the general knowledge in the LM and (2) contrasting the representations of the general and the full (both general and domain knowledge) to learn an integrated representation with both general and domain-specific knowledge. Experimental results demonstrate the effectiveness of the proposed approach.

<p align="center">
    <br>
    <a href="https://github.com/UIC-Liu-Lab/DGA">
        <img src="https://github.com/UIC-Liu-Lab/DGA/blob/main/figures/model.png" width="500"/>
    </a>    
    <br>
<p>

## Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.0` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.0` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.0
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

**Attention**: Our model is based on `transformers==4.11.3` and `adapter-transformers==2.2.0`. Using them from other versions may cause some unexpected bugs.

## Use DGA with Huggingface

You can easily import our continually post-trained model with HuggingFace's `transformers`:

**[TODO]**

## Train DGA

In the following section, we describe how to train a DGA model by using our code.

### Data

Before training and evaluation, please download the dataset from this [Google Drive link](https://drive.google.com/file/d/1_fAu9dPHUpFyAbAN1aBByib3tEVRlZpS/view?usp=sharing) and save them in the `./data` directory. 

### Post-Training

**Training scripts**

We provide an example training script to run DGA. We explain the arguments in the following:

* `--pt_task`: The id for the post-train task. e.g. `--pt_task 3` means post-train the model on the fourth dataset. 
* `--idrandom`: choose the task sequence. See `./sequence` for more details.
  * You can post-train DGA using other task sequences by modifying this argument.
* `--baseline`: The name of the model. Our codebase only supports `dga`.
  * Actually, our codebase is very flexible for adding more baselines. We will add more baselines in the future.


All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--max_seq_length`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to train and evaluate the model on the `cpt_datasets_pt` and `cpt_datasets_ft` sequence files. See `./sequence` for details.

For the results in the paper, we use Nvidia GeForce RTX2080 GPUs with CUDA 10. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance.

**Hyperparameters**

**[TODO]**


### End-Task Fine-tuning

Once you finished post-train, come back to the root directory and simply run

```bash
CUDA_VISIBLE_DEVICES=${your_cuda_device_id} bash scripts/finetune_dga.sh
```

Arguments for the end-task fine-tuning script are as follows,

* `--pt_task`: The id for the post-train task. e.g. `--pt_task 3` means using the model after continually post-trained on the four datasets. 
* `ft_task`: The id for the fine-tuning task. e.g. `--ft_task 0` means doing fine-tuning on the first dataset.
* `--idrandom`: choose the task sequence. See `sequence_10` for more details.
  * You can post-train DGA using other task sequences by modifying this argument.
* `--pt_seed`: the seed used for post-training, used to find the right checkpoint dir of post-trained models.
* `--unfreeze_lm`: whether to unfreeze the backbone (Roberta) when fine-tuning.

## Bugs or questions?
If you have any questions related to the code or the paper, feel free to email [Zixuan](`zke4@uic.edu`), [Yijia](shaoyj.pku.edu.cn), and [Haowei](`linhaowei@pku.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Acknowledgement
This codebase is adapted from [CPT](https://github.com/UIC-Liu-Lab/CPT) and [PyContinual](https://github.com/ZixuanKe/PyContinual).

## Citation

Please cite our paper if you use DGA in your work:

```bibtex
@inproceedings{ke2022dga,
   title={Adapting a Language Model While Preserving its General Knowledge},
   author={Ke, Zixuan and Shao, Yijia and Lin, Haowei and Xu, Hu and Shu, Lei, and Liu, Bing},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2022}
}
```
