
import logging
import math
import os
import torch
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

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from utils import utils

class Appr(object):

    def __init__(self,args):
        super().__init__()
        self.args=args
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        return




    def importance_norm(self, importance, exponent=2):
        # Layerwise importance normalization
        if 'default' in self.args.pipline_norm:
            norm_by_layer = torch.pow(torch.pow(importance, exponent).sum(-1), 1 / exponent)
            importance /= norm_by_layer.unsqueeze(-1) + 1e-20
            importance = (importance - importance.min()) / (importance.max() - importance.min())

        elif 'standard_norm' in self.args.pipline_norm:
            for layer in range(importance.size(0)):
                importance[layer] = (importance[layer] - importance[layer].mean()) / importance[
                    layer].std()  # 2D, we need to deal with this for each layer
        # importance = self.tanh(importance).abs()

        return importance

    def load_heads_importance(self):
        importance_dir = self.args.saved_output_dir[self.args.task]
        print('importance_dir: ',importance_dir)

        # head
        head_importance = torch.Tensor(np.load(os.path.join(importance_dir, "head_importance.npy"))).cuda()
        head_importance = self.importance_norm(head_importance)
        head_importance = self.tanh(head_importance).abs()


        # intermediate
        intermediate_importance = torch.Tensor(
            np.load(os.path.join(importance_dir, "intermediate_importance.npy"))).cuda()
        intermediate_importance = self.importance_norm(intermediate_importance)
        intermediate_importance = self.tanh(intermediate_importance).abs()

        # output
        output_importance = torch.Tensor(np.load(os.path.join(importance_dir, "output_importance.npy"))).cuda()
        output_importance = self.importance_norm(output_importance)
        output_importance = self.tanh(output_importance).abs()

        return head_importance, intermediate_importance, output_importance


    # TODO: for now, it only supports single GPU

    def train(self,model,accelerator,tokenizer,train_loader, test_loader):

        # Set the optimizer
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                          weight_decay=self.args.weight_decay)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epoch * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # TODO: Warm up can be important

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

        # Set tensorboard
        if self.args.tensorboard_dir:
            writer = SummaryWriter(log_dir=self.args.tensorboard_dir)

        # Prepare everything with the accelerator
        model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

        # Train!
        logger.info("***** Running training *****")
        logger.info( f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset_name}, seed = {self.args.seed}")

        summary_path = os.path.join(self.args.output_dir , '../'+ str(self.args.dataset_name) + '_finetune_summary')
        print('summary_path: ', summary_path)

        for epoch in range(self.args.epoch):
            print("Epoch {} started".format(epoch))
            train_acc, training_loss = self.train_epoch(model, optimizer, train_loader, accelerator, lr_scheduler)
            print("train acc = {:.4f}, training loss = {:.4f}".format(train_acc, training_loss))

        micro_f1, macro_f1, acc, test_loss = self.eval(model, test_loader, accelerator)

        if self.args.dataset_name in ['chemprot_sup', 'rct_sample_sup']:
            macro_f1 = micro_f1  # we report micro instead

        logger.info(
            "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(self.args.model_name_or_path,
                                                                                    self.args.dataset_name, macro_f1,
                                                                                    acc, self.args.seed))

        if 'one' in self.args.baseline:
            if accelerator.is_main_process:
                # with open(detail_path,'w') as ppl_f:
                #     ppl_f.writelines(str(macro_f1) + '\t' + str(acc))

                with open(summary_path, 'a') as ppl_f:
                    ppl_f.writelines(str(macro_f1) + '\t' + str(acc) + '\n')
        else:
            if accelerator.is_main_process:
                progressive_f1_path = os.path.join(self.args.output_dir + '/../', 'progressive_f1_' + str(self.args.seed))
                progressive_acc_path = os.path.join(self.args.output_dir + '/../', 'progressive_acc_' + str(self.args.seed))
                print('progressive_f1_path: ', progressive_f1_path)
                print('progressive_acc_path: ', progressive_acc_path)

                if os.path.exists(progressive_f1_path):
                    f1s = np.loadtxt(progressive_f1_path)
                    accs = np.loadtxt(progressive_acc_path)

                else:
                    f1s = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)
                    accs = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)

                f1s[self.args.pt_task][self.args.ft_task] = macro_f1
                np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

                accs[self.args.pt_task][self.args.ft_task] = acc
                np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

                if self.args.ft_task == self.args.ntasks - 1:  # last ft task, we need a final one
                    final_f1 = os.path.join(self.args.output_dir + '/../', 'f1_' + str(self.args.seed))
                    final_acc = os.path.join(self.args.output_dir + '/../', 'acc_' + str(self.args.seed))

                    forward_f1 = os.path.join(self.args.output_dir + '/../', 'forward_f1_' + str(self.args.seed))
                    forward_acc = os.path.join(self.args.output_dir + '/../', 'forward_acc_' + str(self.args.seed))

                    print('final_f1: ', final_f1)
                    print('final_acc: ', final_acc)

                    if self.args.baseline == 'one':
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

                    else:
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[-1][j]) + '\n')
                                f1_file.writelines(str(f1s[-1][j]) + '\n')

                        with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')


    def train_epoch(self,model, optimizer, dataloader, accelerator, lr_scheduler):
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        model.train()
        train_acc = 0.0
        training_loss = 0.0
        total_num = 0.0
        for batch, inputs in enumerate(dataloader):


            if 'use_all' in self.args.baseline:
                res = model(**inputs, args=self.args, return_dict=True)

            elif 'transformer_hat' in self.args.baseline:
                model_ori = accelerator.unwrap_model(model)
                head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                res = model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance, args=self.args, return_dict=True)

            elif 'use_importance' in self.args.baseline: # TODO: rememeber change both train and eval, this is using the importance to guide end-task
                head_importance, intermediate_importance, output_importance = self.load_heads_importance()
                res = model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance, args=self.args, return_dict=True)

            else:
                res = model(**inputs, args=self.args, return_dict=True)

            outp = res.logits
            loss = res.loss
            optimizer.zero_grad()
            accelerator.backward(loss)

            # for n,p in accelerator.unwrap_model(model).named_parameters():
            #     if p.grad is not None:
            #         print('n,p： ',n)

            optimizer.step()
            lr_scheduler.step()

            pred = outp.max(1)[1]

            predictions = accelerator.gather(pred)
            references = accelerator.gather(inputs['labels'])


            train_acc += (references == predictions).sum().item()
            training_loss += loss.item()
            total_num += references.size(0)

            progress_bar.update(1)
            # break
        return train_acc / total_num, training_loss / total_num

    def eval(self,model, dataloader, accelerator):
        model.eval()
        label_list = []
        prediction_list = []
        total_loss=0
        total_num=0
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']
                labels = inputs['labels']
                attention_mask = inputs['attention_mask']

                if 'use_all' in self.args.baseline:
                    print('use_all: ')
                    res = model(**inputs, args=self.args, return_dict=True)

                elif 'transformer_hat' in self.args.baseline:
                    model_ori = accelerator.unwrap_model(model)
                    head_importance, intermediate_importance, output_importance = model_ori.transformer_mask()
                    res = model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance, args=self.args, return_dict=True)
                elif 'use_importance' in self.args.baseline:
                    head_importance, intermediate_importance, output_importance = self.load_heads_importance()
                    res = model(**inputs, head_mask=head_importance, intermediate_mask=intermediate_importance,
                                output_mask=output_importance, args=self.args, return_dict=True)

                else:
                    res = model(**inputs, args=self.args, return_dict=True)

                real_b=input_ids.size(0)
                loss = res.loss
                outp = res.logits
                if self.args.problem_type != 'multi_label_classification':
                    pred = outp.max(1)[1]
                else:
                    pred = outp.sigmoid() > 0.5

                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b

                predictions = accelerator.gather(pred)
                references = accelerator.gather(inputs['labels'])

                label_list += references.cpu().numpy().tolist() # we may use multi-node
                prediction_list += predictions.cpu().numpy().tolist()
                progress_bar.update(1)
                # break

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        accuracy = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))]) * 1.0 / len(prediction_list)

        # print('micro_f1: ', micro_f1)
        # print('macro_f1: ', macro_f1)
        # print('accuracy: ', accuracy)
        # print('loss: ', total_loss/total_num)

        return micro_f1, macro_f1, accuracy,total_loss/total_num

