#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the entropy of the head attentions
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GlueDataset,
    default_data_collator,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils import utils
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)

logger = logging.getLogger(__name__)
import torch.nn.utils.prune as prune



def gather_importance(head_importance):
    head_importance_list = [torch.zeros_like(head_importance) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_importance_list, tensor=head_importance.contiguous()) # everyone need to do this
    head_importance_list = torch.stack(head_importance_list)
    # head_importance = torch.mean(head_importance_list,dim=0)
    head_importance, _ = head_importance_list.max(0)  # take a, TODO: Used in submision and subset
    return head_importance

def initial_importance(config,args):
    n_layer, n_heads = config.num_hidden_layers, config.num_attention_heads
    intermediate_importance = torch.zeros(n_layer, config.intermediate_size).to(args.device)
    output_importance = torch.zeros(n_layer, config.hidden_size).to(args.device)
    intermediate_mask = torch.ones(n_layer, config.intermediate_size).to(args.device)
    output_mask = torch.ones(n_layer, config.hidden_size).to(args.device)
    intermediate_mask.requires_grad_(requires_grad=True)
    output_mask.requires_grad_(requires_grad=True)
    head_importance = torch.zeros(n_layer, n_heads).to(args.device)
    head_mask = torch.ones(n_layer, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    embedding_importance = torch.zeros(1, config.hidden_size).to(args.device)
    embedding_mask = torch.ones(1, config.hidden_size).to(args.device)
    embedding_mask.requires_grad_(requires_grad=True)


    tot_tokens = 0.0

    return head_importance, intermediate_importance, output_importance,embedding_importance,  head_mask, intermediate_mask, output_mask, embedding_mask, tot_tokens


def random_importance(config,args):
    n_layer, n_heads = config.num_hidden_layers, config.num_attention_heads
    intermediate_importance = torch.rand(size=(n_layer, config.intermediate_size)).to(args.device)
    output_importance = torch.rand(size=(n_layer, config.hidden_size)).to(args.device)
    head_importance = torch.rand(size=(n_layer, n_heads)).to(args.device)
    embedding_importance = torch.rand(size=(1, config.hidden_size)).to(args.device)

    return head_importance, intermediate_importance, output_importance,embedding_importance


# TODO: NAN is highly possible when use all data, mean, one, zero-grad all gives me nan

def compute_heads_importance(args,config, model,eval_dataloader,accelerator,prune_loss=None):



    if 'random' in args.softmask_compute:
        head_importance, intermediate_importance, output_importance, embedding_importance = random_importance(config, args)

    else:

        # model.train() # train results in NAN

        # MLM/Distill loss *****************************
        head_importance, intermediate_importance, output_importance, embedding_importance, head_mask, intermediate_mask, output_mask, embedding_mask, tot_tokens = initial_importance(config, args)

        progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

        for step, inputs in enumerate(eval_dataloader):

            outputs = model(inputs,
                            head_mask=head_mask,intermediate_mask=intermediate_mask,output_mask=output_mask,embedding_mask=embedding_mask,
                            prune_mdoel=True,prune_loss=prune_loss)

            loss = outputs.loss

            accelerator.backward(loss)

            # torch.nn.utils.clip_grad_norm([head_mask,intermediate_mask,output_mask,embedding_mask], 2) # deal with NAN

            head_importance += head_mask.grad.detach()
            intermediate_importance += intermediate_mask.grad.detach()
            output_importance += output_mask.grad.detach()
            embedding_importance += embedding_mask.grad.detach()
            tot_tokens += inputs["attention_mask"].float().detach().sum().data

            progress_bar.update(1)
            progress_bar.set_description(
                "Iteration " + prune_loss + ' Importance Computation Iter (loss=%5.3f)' % loss.item())  # show the loss, mean while

        # Normalize
        head_importance /= tot_tokens
        intermediate_importance /= tot_tokens
        output_importance /= tot_tokens
        embedding_importance /= tot_tokens

    # Print/save matrices
    accelerator.wait_for_everyone()

    # TODO: consider use only 1 node to do that
    head_importance = gather_importance(head_importance)
    intermediate_importance = gather_importance(intermediate_importance)
    output_importance = gather_importance(output_importance)
    embedding_importance = gather_importance(embedding_importance)



    if accelerator.is_main_process:

        if 'distill' in prune_loss:
            np.save(os.path.join(args.output_dir, prune_loss + str(args.task)+"/head_importance.npy"), head_importance.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, prune_loss +  str(args.task)+"/intermediate_importance.npy"),intermediate_importance.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, prune_loss +  str(args.task)+"/output_importance.npy"), output_importance.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, prune_loss +  str(args.task)+"/embedding_importance.npy"), embedding_importance.detach().cpu().numpy())

        else:
            np.save(os.path.join(args.output_dir + prune_loss+'/', "head_importance.npy"), head_importance.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir  + prune_loss+'/', "intermediate_importance.npy"),intermediate_importance.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir  + prune_loss+'/', "output_importance.npy"), output_importance.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir  + prune_loss+'/', "embedding_importance.npy"), embedding_importance.detach().cpu().numpy())

        print(head_importance, intermediate_importance, output_importance, embedding_importance)
    return head_importance, intermediate_importance, output_importance, embedding_importance


