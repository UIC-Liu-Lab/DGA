import copy
import shutil
import argparse
import logging
import math
import os
import random
import sys
import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from tqdm.auto import tqdm
from networks import prompt
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
import numpy as np
from transformers import RobertaForMaskedLM, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from networks import prune_model
from networks.roberta_model import MyRoberta, MyRobertaForMaskedLM, MyRobertaModel
from approaches.my_optimizer import MyAdamW
# sys.path.append("..")
from utils import utils
import torch.distributed as dist

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return



    def gather_loss(self, head_importance):
        head_importance_list = [torch.zeros_like(head_importance) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=head_importance_list,
                        tensor=head_importance.contiguous())  # everyone need to do this
        head_importance_list = torch.stack(head_importance_list)
        head_importance = torch.mean(head_importance_list, dim=0)
        return head_importance

    def importance_norm(self, importance, tanh_temp):
        for layer in range(importance.size(0)):
            importance[layer] = (importance[layer] - importance[layer].mean()) / importance[
                layer].std()  # 2D, we need to deal with this for each layer
        importance = self.tanh(tanh_temp * importance).abs()

        return importance

    def use_heads_importance(self, tanh_temp=1):
        head_importance_list = []
        intermediate_importance_list = []
        output_importance_list = []
        embedding_importance_list = []

        for importance_dir_id, importance_dir in enumerate(self.args.saved_output_dir):
            # head
            head_importance = torch.Tensor(np.load(os.path.join(importance_dir, "head_importance.npy")))
            head_importance = self.importance_norm(head_importance, tanh_temp)
            head_importance_list.append(head_importance)

            # intermediate
            intermediate_importance = torch.Tensor(
                np.load(os.path.join(importance_dir, "intermediate_importance.npy")))
            intermediate_importance = self.importance_norm(intermediate_importance, tanh_temp)
            intermediate_importance_list.append(intermediate_importance)

            output_importance = torch.Tensor(np.load(os.path.join(importance_dir, "output_importance.npy")))
            output_importance = self.importance_norm(output_importance, tanh_temp)
            output_importance_list.append(output_importance)


            embedding_importance = torch.Tensor(np.load(os.path.join(importance_dir, "embedding_importance.npy")))
            embedding_importance = self.importance_norm(embedding_importance, tanh_temp)
            embedding_importance_list.append(embedding_importance)



        if len(head_importance_list) > 0:

            # print('head_importance_list: ',head_importance_list)

            head_importances = torch.stack(head_importance_list)
            head_importance, _ = head_importances.max(
                0)  # take a max, so that block all importance nurons for all previous tasks;
            # if you stack have to use this for element-wise version
            # you cannot input list to torch.max, unless you specify torch.max(a,b)

            intermediate_importances = torch.stack(intermediate_importance_list)
            intermediate_importance, _ = intermediate_importances.max(
                0)  # take a max, so that block all importance nurons for all previous tasks;

            output_importances = torch.stack(output_importance_list)
            output_importance, _ = output_importances.max(
                0)  # take a max, so that block all importance nurons for all previous tasks;


            embedding_importances = torch.stack(embedding_importance_list)
            embedding_importance, _ = embedding_importances.max(
                0)  # take a max, so that block all importance nurons for all previous tasks;
            # if you stack have to use this for element-wise version
            # you cannot input list to torch.max, unless you specify torch.max(a,b)

            head_importance, intermediate_importance, output_importance, embedding_importance = head_importance.cuda(), intermediate_importance.cuda(), output_importance.cuda(), embedding_importance.cuda()

        else:
            head_importance, intermediate_importance, output_importance, embedding_importance = None,None,None,None


        return head_importance, intermediate_importance, output_importance, embedding_importance


    def train(self, model, accelerator, train_dataset,train_loader, tokenizer, train_loader_prune,train_dataloader_prune_dataset):
        # ********************************* before tranining *********************************
        train_loader_prune = accelerator.prepare(train_loader_prune)
        config = accelerator.unwrap_model(model).model.config

        self.args.prune_model = True
        if self.args.softmask_compute is not None:

            if not self.args.eval_only and 'before_distill' in self.args.softmask_compute and (self.args.task == 0 or 'one' in self.args.baseline):

                student = MyRobertaModel.from_pretrained(self.args.model_name_or_path)
                teacher = MyRobertaModel.from_pretrained(self.args.model_name_or_path)

                for param in teacher.parameters():  # nothing is trainable in teacher
                    param.requires_grad = False

                model_prune = MyRoberta(student, teacher, args=self.args)

                model_prune = accelerator.prepare(model_prune)

                prune_model.compute_heads_importance(args=self.args, config=config, model=model_prune,
                                                     eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                     prune_loss='before_distill')


            if not self.args.eval_only and 'before_mlm' in self.args.softmask_compute and self.args.task == 0: # only for wiki in task 0

                model = accelerator.prepare(model)
                prune_model.compute_heads_importance(args=self.args, config=config, model=model,
                                                     eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                     prune_loss='before_mlm')


            accelerator.wait_for_everyone()
            pre_head_importance, pre_intermediate_importance, pre_output_importance,pre_embedding_importance = self.use_heads_importance()


            if accelerator.is_main_process:

                print('pre_head_importance: ', pre_head_importance)
                print('pre_intermediate_importance: ', pre_intermediate_importance)
                print('pre_output_importance: ', pre_output_importance)
                print('pre_embedding_importance: ', pre_embedding_importance)

        #TODO: need to double check whether everything is ok

        self.args.prune_model = False

        if 'pre_as_general' in self.args.baseline:

            general_head_importance = None
            general_intermediate_importance = None
            general_output_importance = None
            general_embedding_importance = None

            if 'head_mask' in self.args.baseline:
                general_head_importance = pre_head_importance
            if 'intermediate_mask' in self.args.baseline:
                general_intermediate_importance = pre_intermediate_importance
            if 'output_mask' in self.args.baseline:
                general_output_importance = pre_output_importance
            if 'embedding_mask' in self.args.baseline:
                general_embedding_importance = pre_embedding_importance


        if 'wiki' in self.args.dataset_name: exit() # no eed to run wiki
        # ********************************* before tranining *********************************

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                'name': [n for n, p in model.named_parameters()
                         if p.requires_grad and not any(nd in n for nd in no_decay)],
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                'name': [n for n, p in model.named_parameters()
                         if p.requires_grad and any(nd in n for nd in no_decay)],
                "params": [p for n, p in model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate

            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_loader, train_loader_prune = accelerator.prepare(
            model, optimizer, train_loader, train_loader_prune
        )

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if accelerator.distributed_type == DistributedType.TPU:
            model.tie_weights()

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)

        if self.args.max_samples is not None:
            self.args.max_train_steps = self.args.max_samples // (
                    self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps)

        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # TODO: Warm up can be important
        # warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
        self.args.num_warmup_steps = int(float(self.args.warmup_proportion) * float(self.args.max_train_steps))  # 0.1

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Train!
        total_batch_size = self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

        if accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
            logger.info(f"  Total samples = {self.args.max_train_steps * total_batch_size}")
            logger.info(
                f"  Learning Rate = {self.args.learning_rate}, Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f"  Seq ID = {self.args.idrandom}, Task id = {self.args.task}, dataset name = {self.args.dataset_name}")
            logger.info(f"  Baseline = {self.args.baseline}, Smax = {self.args.smax}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        global_step = 0  # This will be used by CLMOE if we choose 'auto_encoder' as the route type.

        if accelerator.is_main_process:
            tensorboard_file = os.path.join(self.args.output_dir, str(self.args.dataset_name) + '_log')
            print('tensorboard_file: ', tensorboard_file)
            if os.path.isdir(tensorboard_file):
                shutil.rmtree(tensorboard_file)
            writer = utils.setup_writer(tensorboard_file)

        try:
            if not self.args.eval_only:
                model_ori = accelerator.unwrap_model(model)
                if accelerator.is_main_process and (self.args.task > 0 or 'proxy' in self.args.baseline) and pre_head_importance is not None:
                    if pre_head_importance is not None:
                        if 'head_mask' in self.args.baseline:
                            print('head usage: ', (pre_head_importance.sum() / pre_head_importance.numel()).item())
                        if 'intermediate_mask' in self.args.baseline:
                            print('intermediate usage: ', (pre_intermediate_importance.sum() / pre_intermediate_importance.numel()).item())
                        if 'output_mask' in self.args.baseline:
                            print('output usage: ', (pre_output_importance.sum() / pre_output_importance.numel()).item())
                        if 'embedding_mask' in self.args.baseline:
                            print('embedding usage: ', (pre_embedding_importance.sum() / pre_embedding_importance.numel()).item())

                for epoch in range(self.args.num_train_epochs):
                    # break
                    model.train()
                    for step, inputs in enumerate(train_loader):

                        if 'contrast' in self.args.baseline:
                            # outputs = model(inputs,general_head_mask=general_head_mask)
                            outputs = model(inputs,
                                            general_head_mask=general_head_importance,
                                            general_intermediate_mask=general_intermediate_importance,
                                            general_output_mask=general_output_importance,
                                            general_embedding_mask=general_embedding_importance)


                        else:
                            outputs = model(inputs)

                        loss = outputs.loss  # loss 1
                        if 'contrast' in self.args.baseline:
                            contrast_loss = outputs.contrast_loss  # loss 1
                            loss = loss + contrast_loss

                        loss = loss / self.args.gradient_accumulation_steps
                        # add model needs to be careful! make sure it is in parameters and please double check its gradient
                        accelerator.backward(loss)  # sync

                        # if accelerator.is_main_process:
                        #     for n,p in accelerator.unwrap_model(model).named_parameters():
                        #         if p.grad is not None:
                        #             print('n,p： ',n,p.size())


                        # we need this even for the first task
                        if self.args.task > 0 or 'proxy' in self.args.baseline and pre_head_importance is not None:
                            n_layers, n_heads = model_ori.model.config.num_hidden_layers, model_ori.model.config.num_attention_heads
                            head_size = int(model_ori.model.config.hidden_size / model_ori.model.config.num_attention_heads)



                            for layer in range(n_layers):

                                if 'head_mask' in self.args.baseline:
                                    head_importance = pre_head_importance[layer].unsqueeze(-1).repeat((1, head_size))
                                    head_importance = head_importance.flatten()
                                    head_mask = 1 - head_importance

                                    model_ori.model.roberta.encoder.layer[layer].attention.self.query.weight.grad *= head_mask
                                    model_ori.model.roberta.encoder.layer[layer].attention.self.query.bias.grad *= head_mask

                                    model_ori.model.roberta.encoder.layer[layer].attention.self.key.weight.grad *= head_mask
                                    model_ori.model.roberta.encoder.layer[layer].attention.self.key.bias.grad *= head_mask

                                    model_ori.model.roberta.encoder.layer[layer].attention.self.value.weight.grad *= head_mask
                                    model_ori.model.roberta.encoder.layer[layer].attention.self.value.bias.grad *= head_mask

                                    model_ori.model.roberta.encoder.layer[layer].attention.output.dense.weight.grad *= head_mask
                                    model_ori.model.roberta.encoder.layer[layer].attention.output.dense.bias.grad *= head_mask

                                if 'intermediate_mask' in self.args.baseline:

                                    intermediate_mask = (1 - pre_intermediate_importance[layer])
                                    model_ori.model.roberta.encoder.layer[
                                        layer].intermediate.dense.weight.grad *= intermediate_mask.unsqueeze(1)
                                    model_ori.model.roberta.encoder.layer[
                                        layer].intermediate.dense.bias.grad *= intermediate_mask
                                    # compute_mask(model_ori.model.roberta.encoder.layer[layer].intermediate.dense.bias.grad,intermediate_importance)

                                if 'output_mask' in self.args.baseline:

                                    output_mask = (1 - pre_output_importance[layer])
                                    model_ori.model.roberta.encoder.layer[
                                        layer].output.dense.weight.grad *= output_mask.unsqueeze(1)
                                    model_ori.model.roberta.encoder.layer[layer].output.dense.bias.grad *= output_mask

                            if 'embedding_mask' in self.args.baseline:

                                embedding_mask = (1 - pre_embedding_importance)

                                model_ori.model.roberta.embeddings.word_embeddings.weight.grad *= embedding_mask
                                model_ori.model.roberta.embeddings.position_embeddings.weight.grad *= embedding_mask
                                model_ori.model.roberta.embeddings.token_type_embeddings.weight.grad *= embedding_mask
                                model_ori.model.roberta.embeddings.LayerNorm.weight.grad *= embedding_mask.squeeze(0)
                                model_ori.model.roberta.embeddings.LayerNorm.bias.grad *= embedding_mask.squeeze(0)

                        # n, p：  model.roberta.embeddings.word_embeddings.weight
                        # torch.Size([50265, 768])
                        # n, p：  model.roberta.embeddings.position_embeddings.weight
                        # torch.Size([514, 768])
                        # n, p：  model.roberta.embeddings.token_type_embeddings.weight
                        # torch.Size([1, 768])
                        # n, p：  model.roberta.embeddings.LayerNorm.weight
                        # torch.Size([768])
                        # n, p：  model.roberta.embeddings.LayerNorm.bias
                        # torch.Size([768])


                        global_step += 1

                        if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:

                            optimizer.step()
                            lr_scheduler.step()

                            optimizer.zero_grad()
                            progress_bar.update(1)
                            completed_steps += 1
                            progress_bar.set_description(
                                'Train Iter (loss=%5.3f)' % loss.item())  # show the loss, mean while

                            if accelerator.is_main_process:
                                utils.log_loss(writer, scalar_value=loss.item(), global_step=global_step)
                                utils.log_loss(writer, loss_name=' MLM loss', scalar_value=outputs.loss.item(),
                                               global_step=global_step)
                                if 'contrast' in self.args.baseline:
                                    utils.log_loss(writer, loss_name=' contrastive loss',
                                                   scalar_value=outputs.contrast_loss.item(), global_step=global_step)
                        break
                        if completed_steps >= self.args.max_train_steps:
                            break

        except KeyboardInterrupt:  # even if contro-C, I still want to save model
            return


        # after training ***********************************************************************************************

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:  # onlyh discriminator is saved. I don't need anything about geenrator
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.model.save_pretrained(self.args.output_dir)
            tokenizer.save_pretrained(self.args.output_dir)


        if self.args.softmask_compute is not None:

            if 'after_mlm' in self.args.softmask_compute and 'wiki' not in self.args.dataset_name:
                # mlm_model = MyRobertaForMaskedLM.from_pretrained(self.args.model_name_or_path, args=self.args)
                # mlm_model = accelerator.prepare(mlm_model)

                prune_model.compute_heads_importance(args=self.args, config=config, model=model,
                                                     eval_dataloader=train_loader_prune, accelerator=accelerator,
                                                     prune_loss='mlm')


        # after training ***********************************************************************************************
