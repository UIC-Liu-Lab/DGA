import os,sys
import numpy as np
from copy import deepcopy
import torch
import time
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from networks.roberta_model import MyRoberta,MyRobertaForMaskedLM,MyRobertaForSequenceClassification,MyRobertaModel,RobertaClassificationHead
from transformers import RobertaForMaskedLM,RobertaModel,RobertaConfig
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
from networks import prune_model
import copy

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def save_discriminator(model,save_dir,saved_name='plugin_ckpt.pth'):
    torch.save(model, os.path.join(save_dir, saved_name))

def load_discriminator(model,state_dict):
    try:
        print('loading ')
        model.load_state_dict(state_dict)
    except Exception as e:  # work on python 3.x
        print('Missing key: ' + str(e))
        model.load_state_dict(state_dict,strict=False)

    return model

def save_prompt(model,save_dir,saved_name='embed_pool.pth'):
    torch.save(model.prompt_embed_pool, os.path.join(save_dir, saved_name))

def moe_experts_recover(model,saved_model,t):
    for pre_t in t: #only pre_t
        for (name, sub_module),(saved_name, saved_sub_module) in zip(model.named_modules(),saved_model.named_modules):
            if isinstance(sub_module, MOEAdapter):
                set_model_(sub_module.moe_adapter[pre_t],saved_sub_module.moe_adapter[pre_t])

def save_plugin(model, save_dir,MOEAdapter,saved_name='plugin_ckpt.pth'):
    """
    Save prompt embedding pool and plugins at different locations.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plugin = {}
    # Save MOE adapter
    idx = 0
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, MOEAdapter):
            plugin[f'adapter_{idx}'] = sub_module.moe_adapter.state_dict()
            idx += 1

    torch.save(plugin, os.path.join(save_dir, saved_name))
    print('Save plugins successfully!')


def load_plugin(model, plugin_ckpt,MOEAdapter):
    """
    Load weights for plugins at different locations.
    plugin_ckpt should be a dict which corresponds to `plugin` in save_plugin()
    """
    idx = 0
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, MOEAdapter):
            try:
                print('load state ',name)
                sub_module.moe_adapter.load_state_dict(plugin_ckpt[f'adapter_{idx}'])
            except Exception as e:  # work on python 3.x
                print('Missing key: ' + str(e))
                sub_module.moe_adapter.load_state_dict(plugin_ckpt[f'adapter_{idx}'], strict=False)
            idx += 1

    print('Load plugins successfully!')




########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

# for ACL
def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

def report_val(res):
    # Validation performance
    print(' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
        res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

########################################################################################################################



def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################

#we need to analysis the results, tensorboard

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
def setup_writer(name):
    writer = SummaryWriter(name)
    return writer
    
def project_layer(writer,features,class_labels):
    writer.add_embedding(features,metadata=class_labels)

def log_loss(writer,loss_name='training loss',scalar_value=None,global_step =None):
    # ...log the running loss
    writer.add_scalar(loss_name,scalar_value,global_step=global_step)

def log_gate(writer, loss_name='log gate',gate_sum_dict=None,global_step =None):
    # ...log the running loss
    writer.add_scalars(loss_name,
                       gate_sum_dict,
                       global_step=global_step)

########################################################################################################################
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    # get_scheduler,
    set_seed,
)
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
InputDataClass = NewType("InputDataClass", Any)



def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class PTDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        if "labels" in batch:
            batch["labels_ori"] = batch["labels"]

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None) # excloude special token
        if self.mlm:
            batch["input_ids"], batch["inputs_ori_ids"],batch["labels"], batch["masked_indices"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        inputs_ori =  inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, inputs_ori, labels, masked_indices


# distillation ########################

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


# contrastive ########################


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())] #TODO: what is a world size?
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out



class MyContrastive(nn.Module):
    # https://github.com/facebookresearch/moco/blob/3631be074a0a14ab85c206631729fe035e54b525/moco/builder.py#L155
    def __init__(self):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MyContrastive, self).__init__()
        self.n_gpu = torch.cuda.device_count()
        self.T = 1
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.sup_con = SupConLoss()

    def forward(self, x, order_x=None, labels=None, con_type=None):
        """
        labels indicate sample label
        x include all samples to be contrast
        """
        if self.n_gpu > 1:
            x = torch.cat(GatherLayer.apply(x), dim=0)
            if order_x is not None:
                order_x = torch.cat(GatherLayer.apply(order_x), dim=0)
            elif labels is not None:
                labels = torch.cat(GatherLayer.apply(labels), dim=0)

        if con_type == 'supervised':
            loss = self.sup_con(x.unsqueeze(1),labels)
        elif con_type == 'unsupervised':
            loss = self.sup_con(x)
        elif con_type == 'soft_contrast':
            loss = self.unsupervised_loss(x,order_x,labels)

        return loss




    # Let's use forward to enable distributed training
    def unsupervised_loss(self, aug_x, order_x, output_label):
        logits = torch.einsum('nci,nkc->nki', [aug_x.unsqueeze(-1), order_x.permute(0, 2, 1)]).squeeze(-1)  # -dim=1 as number of order_x

        # apply temperature
        logits /= self.T

        contrast_loss = self.bce(logits, output_label)

        return contrast_loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1, contrast_mode='all',
                 base_temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def tacl_loss(z1, z2, contrastive_labels, eps=0.0):
    # https://github.com/yxuansu/TaCL/blob/main/pretraining/bert_contrastive.py
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 1., otherwise 0. I only want masked
    '''
    z1 = torch.cat(GatherLayer.apply(z1), dim=0)
    z2 = torch.cat(GatherLayer.apply(z2), dim=0)
    contrastive_labels = torch.cat(GatherLayer.apply(contrastive_labels), dim=0)

    # if self.sim == 'dot_product':
    contrastive_scores = torch.matmul(z1, z2.transpose(1, 2))

    # elif self.sim == 'cosine':  # 'cosine'
    #     masked_rep = masked_rep / masked_rep.norm(dim=2, keepdim=True)
    #     truth_rep = truth_rep / truth_rep.norm(dim=2, keepdim=True)
    #     contrastive_scores = torch.matmul(masked_rep,
    #                                       truth_rep.transpose(1, 2)) / self.temperature  # bsz x seqlen x seqlen
    #
    #
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss) / contrastive_labels.sum()
    return loss


def taco_loss(z1, z2):
    # https://github.com/yxuansu/TaCL/blob/main/pretraining/bert_contrastive.py
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 1., otherwise 0. I only want masked
    '''
    z1 = torch.cat(GatherLayer.apply(z1), dim=0)
    z2 = torch.cat(GatherLayer.apply(z2), dim=0)

    z1 = z1 / z1.norm(dim=2, keepdim=True)
    z2 = z2 / z2.norm(dim=2, keepdim=True)
    contrastive_scores = torch.matmul(z1,z2.transpose(1, 2)) # bsz x seqlen x seqlen

    #
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen)
    loss = torch.sum(loss)
    return loss



class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        # return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))

def prepare_sequence_posttrain(args):

    if 'dga' in args.baseline:
        args.baseline = 'one_contrast_pre_as_general_proxy_head_mask'
        args.softmask_compute = 'before_distill'

    # different baseline has its special setting
    # if 'dgi' in args.baseline:
    #     args.addition_loss += 'contrast'
    #     args.prune_technique = 'proxy'

    with open('sequence_10','r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()


    # print('data: ',data)
    output = args.base_dir+"/seq"+str(args.idrandom)+"/"+str(args.max_samples) + "samples/"+str(args.baseline)+'/'+str(data[args.task])+"_roberta/"
    ckpt = args.base_dir+"/seq"+str(args.idrandom)+"/"+str(args.max_samples) + "samples/"+str(args.baseline)+'/'+str(data[args.task-1])+"_roberta/"


    if 'proxy' in args.baseline: #TODO: add some condition for better testing
        # os.makedirs(output + 'distill/', exist_ok=True)
        # os.makedirs(output + 'contrast/', exist_ok=True)

        output_dir = args.base_dir + "/seq" + str(args.idrandom) + "/"+str(args.max_samples) + "samples/"+str(args.baseline)
        args.saved_output_dir = []
        # args.saved_output_dir = [output_dir +'/'+str(data[t])+"_roberta/distill/" for t in range(args.task+1)] # my base need to be read
        # args.saved_output_dir += [output_dir+'/'+str(data[t])+"_roberta/contrast/" for t in range(args.task)]

        if args.softmask_compute is not None:
            if 'before_distill' in args.softmask_compute and 'one' not in args.baseline:
                for pre_t in range(args.task + 1):
                    os.makedirs(output + 'before_distill' + str(pre_t) + '/', exist_ok=True)
                args.saved_output_dir += [output_dir + '/' + str(data[0]) + "_roberta/before_distill" + str(0) + '/' ] # only use the first one

            if 'before_distill' in args.softmask_compute and 'one' in args.baseline:
                for pre_t in range(args.task + 1):
                    os.makedirs(output + 'before_distill' + str(pre_t) + '/', exist_ok=True)
                args.saved_output_dir += [output_dir + '/' + str(data[args.task]) + "_roberta/before_distill" + str(args.task) + '/' ]

            if 'before_mlm' in args.softmask_compute and 'one' not in args.baseline:
                os.makedirs(output + 'before_mlm/', exist_ok=True)
                args.saved_output_dir += [output_dir + '/' + str(data[0]) + "_roberta/before_mlm/" ] # only use the first one

            if 'after_mlm' in args.softmask_compute:
                os.makedirs(output + 'mlm/', exist_ok=True)
                args.saved_output_dir += [output_dir+'/'+str(data[t])+"_roberta/mlm/" for t in range(1,args.task)]

    else:
        args.saved_output_dir = [args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
            args.baseline) + '/' + str(args.dataset_name) + '/' + str(data[t]) + "_roberta/" for t in
                                 range(args.task + 1)]

    print('saved_output_dir: ',args.saved_output_dir)


    args.output_dir = output

    args.data = data
    args.base_model_name_or_path = "roberta-base"

    if 'comb' in args.baseline:
        args.dataset_name = '_unsup'
    else:
        args.dataset_name = data[args.task]

    if args.task == 0 or 'one' in args.baseline or ('wiki' in args.baseline and args.task==1): # no pre-trained for the first
        args.model_name_or_path = "roberta-base"
    else:
        args.model_name_or_path = ckpt

    if args.eval_only: # no pre-trained for the first
        args.model_name_or_path = args.base_dir+"/seq"+str(args.idrandom)+"/"+str(args.max_samples) + "samples/"+str(args.baseline)+'/'+str(data[args.task])+"_roberta/"
        # args.model_name_or_path = "roberta-base"


    print('output_dir: ',args.output_dir)
    print('args.dataset_name: ',args.dataset_name)
    print('args.model_name_or_path: ',args.model_name_or_path)

    if 'ewc' in args.baseline:
        args.lamb = 5000  # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000 for ewc
    if 'adapter_hat' in args.baseline \
            or 'transformer_hat' in args.baseline \
            or 'adapter_bcl' in args.baseline \
            or 'adapter_classic' in args.baseline:
        args.lamb=0.75

    return args


def prepare_sequence_finetune(args):

    if 'dga' in args.baseline:
        args.baseline = 'one_contrast_pre_as_general_proxy_head_mask'
        args.softmask_compute = 'before_distill'

    posttrain2endtask = {"pubmed_unsup":"chemprot_sup", "phone_unsup":"phone_sup", "ai_unsup":"scierc_sup", "camera_unsup":"camera_sup", "acl_unsup":"aclarc_sup", "restaurant_unsup":"restaurant_sup"}

    with open('sequence_10','r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    output = args.base_dir+"/seq"+str(args.idrandom)+"/"+str(args.max_samples) + "samples/"+str(args.baseline)+'/'+str(data[args.pt_task])+"_roberta/"
    ckpt = args.base_dir+"/seq"+str(args.idrandom)+"/"+str(args.max_samples) + "samples/"+str(args.baseline)+'/'+str(data[args.pt_task])+"_roberta/"


    args.output_dir = output
    args.saved_output_dir = [args.base_dir+"/seq"+str(args.idrandom)+"/"+str(args.max_samples) + "samples/"+str(args.baseline)+'/'+str(data[t])+"_roberta/"for t in range(args.pt_task+1)]
    args.dataset_name = posttrain2endtask[data[args.ft_task]]
    args.model_name_or_path = ckpt

    args.task = args.ft_task


    print('output_dir: ',args.output_dir)
    print('args.dataset_name: ',args.dataset_name)
    print('args.model_name_or_path: ',args.model_name_or_path)

    if args.dataset_name in ['aclarc_sup']:
        args.epoch = 10
    elif args.dataset_name in ["hoc_multi","scierc_sup", "covidintent_sup",'restaurant_sup',"laptop_sup"]:
        args.epoch = 5
    elif args.dataset_name in ['phone_sup', "camera_sup"]:
        args.epoch = 15
    elif args.dataset_name in ['chemprot_sup','rct_sample_sup','electric_sup','hyperpartisan_sup']:
        args.epoch = 10

    args.s = args.smax

    return args





def lookfor_model_posttrain(args):

    if args.model_name_or_path:  # only do electra
        if 'adapter' in args.baseline:
            model = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args)
            model = MyRoberta(model,args=args)
            for name,param in model.named_parameters():
                if 'moe' not in name: #only MOE is trainale
                    param.requires_grad = False

        elif 'prompt' in args.baseline:
            model = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args)
            model = MyRoberta(model,args=args)
            for name,param in model.named_parameters():
                if 'prompt_embed_pool' not in name: #only prompt_embed_pool is trainale
                    param.requires_grad = False


        elif 'distill' in args.baseline:
            model = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args)
            teacher = MyRobertaModel.from_pretrained(args.model_name_or_path) # Careful! last one, not always pre-trained. This is CL setting
            for param in teacher.parameters():  # nothing is trainable in teacher
                param.requires_grad = False
            model = MyRoberta(model,teacher,args=args)

        elif 'ewc' in args.baseline:

            model = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args)
            teacher = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args) # Careful! last one, not always pre-trained. This is CL setting
            for param in teacher.parameters():  # nothing is trainable in teacher
                param.requires_grad = False
            model = MyRoberta(model,teacher,args=args)

        elif 'tacl' in args.baseline:
            model = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args)
            teacher = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args) # Careful! last one, not always pre-trained. This is CL setting
            for param in teacher.parameters():  # nothing is trainable in teacher
                param.requires_grad = False
            model = MyRoberta(model,teacher,args=args)

        else:

            if 'comb' in args.baseline:
                model = MyRobertaForMaskedLM.from_pretrained(args.base_model_name_or_path,args=args)
            else:
                model = MyRobertaForMaskedLM.from_pretrained(args.model_name_or_path,args=args)
            model = MyRoberta(model,args=args)

            if 'fix_lmhead' in args.baseline:
                for n, param in model.model.lm_head.named_parameters():
                    param.requires_grad = False
            if 'fix_embedding' in args.baseline:
                for n, param in model.model.roberta.embeddings.named_parameters():
                    param.requires_grad = False

    else:
        raise ValueError('You must provide the model name or path.')

    return model



def lookfor_model_finetune(args):

    if args.model_name_or_path: #only do electra

        if 'adapter' in args.baseline:
            model = MyRobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                            num_labels=args.class_num,
                                                                            problem_type=args.problem_type,
                                                                            args=args)
        elif 'prompt' in args.baseline:
            model = MyRobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                            num_labels=args.class_num,
                                                                            problem_type=args.problem_type,
                                                                            args=args)

        else:
            model = MyRobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                            num_labels=args.class_num,
                                                                            problem_type=args.problem_type,
                                                                            args=args)

    else:
        raise ValueError('You must provide the model name or path.')


    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,attn_size):
        super(Self_Attn,self).__init__()

        self.query_conv = nn.Linear(attn_size,attn_size)
        self.key_conv = nn.Linear(attn_size , attn_size)
        self.value_conv = nn.Linear(attn_size ,attn_size)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B,max_length,hidden_size)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        # print('x: ',x.size())
        m_batchsize,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,width,height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,width,height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # print('energy: ',energy.size())

        attention = self.softmax(energy) # BX (N) X (N)

        # attention =  F.gumbel_softmax(energy,hard=True,dim=-1)
        # print('attention: ',attention)
        proj_value = self.value_conv(x).view(m_batchsize,width,height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,width,height)

        out = self.gamma*out + x


        return out