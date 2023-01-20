"""
    Modified RobertaForSequenceClassification, RobertaForMaskedLM to accept **kwargs in forward.
"""
import pdb
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.adapters.model_mixin import ModelWithHeadsAdaptersMixin
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput, ModelOutput
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaLMHead, RobertaSelfAttention, \
    RobertaSelfOutput, RobertaAttention,RobertaPooler,RobertaForMaskedLM,RobertaForSequenceClassification
import sys
from networks import prompt,simcse
from networks.my_transformer import MyRobertaModel
from networks.plugin import add_roberta_adapters
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
# sys.path.append("..")
from utils import utils
from networks import prune_model
import numpy as np

class MyRobertaOutput(ModelOutput):
    all_attention: torch.FloatTensor = None
    loss: torch.FloatTensor = None


class MyRobertaForMaskedLM(RobertaForMaskedLM):
    # own model, need to remove some original stuff, word embedding is fixe in this forMaskedLM (tied weight with the output)

    def __init__(self, config,args):
        super().__init__(config)
        self.roberta = MyRobertaModel(config, add_pooling_layer=False)
        self.args = args
        self.config = config
        if 'adapter' in args.baseline:
            self.roberta = add_roberta_adapters(self.roberta, args)

        if 'prompt' in args.baseline:
            self.n_tokens = 100

            self.prompt_embed_pool = nn.ModuleList()

            for i in range(args.ntasks):
                init_prompt_value = torch.FloatTensor(self.n_tokens, self.config.hidden_size).uniform_(-0.5, 0.5)
                embedding = nn.Embedding(self.n_tokens, self.config.hidden_size)
                embedding.weight = nn.parameter.Parameter(init_prompt_value)
                self.prompt_embed_pool.append(embedding)

        if 'adapter_classic' in args.baseline:
            self.self_attns = nn.ModuleList()
            for t in range(args.ntasks):
                self.self_attns.append(utils.Self_Attn(t + 1))

        if 'transformer_hat' in args.baseline:

            self.ehead = torch.nn.ModuleList()
            self.eintermediate = torch.nn.ModuleList()
            self.eoutput = torch.nn.ModuleList()
            self.gate = torch.nn.Sigmoid()
            n_layers, n_heads = self.config.num_hidden_layers, self.config.num_attention_heads

            for i in range(args.ntasks):
                self.ehead.append(torch.nn.Embedding(n_layers, n_heads).cuda())
                self.eintermediate.append(torch.nn.Embedding(n_layers, self.config.intermediate_size).cuda())
                self.eoutput.append(torch.nn.Embedding(n_layers, self.config.hidden_size).cuda())


        self.init_weights()

    def transformer_mask(self):

        n_layers, n_heads = self.config.num_hidden_layers, self.config.num_attention_heads

        head_importances = []
        output_importances = []
        intermediate_importances = []


        for i in range(n_layers):
            head_importances.append(self.gate(self.args.s*self.ehead[self.args.task](torch.LongTensor([i]).cuda())))
            intermediate_importances.append(self.gate(self.args.s*self.eintermediate[self.args.task](torch.LongTensor([i]).cuda())))
            output_importances.append(self.gate(self.args.s*self.eoutput[self.args.task](torch.LongTensor([i]).cuda())))


        head_importance = torch.stack(head_importances).squeeze()
        output_importance = torch.stack(output_importances).squeeze()
        intermediate_importance = torch.stack(intermediate_importances).squeeze()

        return head_importance,intermediate_importance,output_importance


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        adapter_names=None,
        output_mask=None,
        intermediate_mask=None,
        embedding_mask=None,
        only_return_output=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
            output_mask=output_mask,
            intermediate_mask=intermediate_mask,
            embedding_mask=embedding_mask

        )

        if only_return_output: return outputs

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(
            sequence_output,
            inv_lang_adapter=self.roberta.get_invertible_adapter(),
        )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class MyRoberta(nn.Module):

    def __init__(self, model,teacher=None,args=None):
        super().__init__()
        #please make sure there is no chanable layers in this class, other than "model"
        self.model = model
        self.contrast = utils.MyContrastive()
        self.teacher = teacher
        self.kd_loss =  utils.DistillKL(1)
        self.config = model.config
        self.args = args
        self.mse = torch.nn.MSELoss()
        self.dropout = nn.Dropout(0.1)


    def forward(self,inputs, head_mask=None,
                output_mask=None,
                intermediate_mask=None,
                embedding_mask=None,
                general_head_mask=None,
                general_intermediate_mask=None,
                general_output_mask=None,
                general_embedding_mask=None,
                prune_mdoel=False,
                prune_loss=None,
                self_fisher=None,
                masks=None,
                mask_pre=None,
                buffer=None):

        # we probably always want to use general

        input_ids = inputs['input_ids']
        inputs_ori_ids = inputs['inputs_ori_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']
        contrast_loss = None
        distill_loss = None
        simcse_loss = None
        tacl_loss = None
        taco_loss = None
        infoword_loss = None
        hidden_states = None

        if prune_loss is not None and 'distill' in prune_loss:
            #  use original ids
            outputs = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     output_mask=output_mask,
                                     intermediate_mask=intermediate_mask,
                                     embedding_mask=embedding_mask,
                                     output_hidden_states=True, output_attentions=True)
            teacher_outputs = self.teacher(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                           head_mask=head_mask,
                                           output_mask=output_mask,
                                           intermediate_mask=intermediate_mask,
                                           embedding_mask=embedding_mask,
                                           output_hidden_states=True, output_attentions=True)


            loss = self.kd_loss(teacher_outputs.hidden_states[-1], outputs.hidden_states[-1])  # no need for mean

        elif prune_loss is not None and 'mlm' in prune_loss:
            outputs = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     output_mask=output_mask,
                                     intermediate_mask=intermediate_mask,
                                     embedding_mask=embedding_mask,
                                     output_hidden_states=True, output_attentions=True)
            loss = outputs.loss  # no need for mean

        else:

            if 'prompt' in self.args.baseline:
                inputs_embeds = prompt.cat_learned_embedding_to_input(self.model, input_ids, self.args.task).cuda()
                labels = prompt.extend_labels(self.model, labels).cuda()
                attention_mask = prompt.extend_attention_mask(self.model, attention_mask).cuda()

                outputs = self.model(inputs_embeds=inputs_embeds,labels=labels,attention_mask=attention_mask,output_hidden_states=True)

            else:
                if 'distill' in self.args.baseline:

                    student_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,output_hidden_states=True)

                    teacher_ori = self.teacher(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask, output_hidden_states=True)

                    distill_loss = self.kd_loss(teacher_ori.hidden_states[-1], student_ori.hidden_states[-1])  # no need for mean. The simplist way to do distillation


                outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     output_mask=output_mask,
                                     intermediate_mask=intermediate_mask,
                                     embedding_mask=embedding_mask,
                                     output_hidden_states=True)

            loss = outputs.loss

            if 'ewc' in self.args.baseline:
                loss_reg = 0
                if self.args.task > 0:

                    for (name, param), (_, param_old) in zip(self.model.named_parameters(),self.teacher.named_parameters()):
                        loss_reg += torch.sum(self_fisher['module.model.'+name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                loss += self.args.lamb * loss_reg


            elif 'derpp' in self.args.baseline:
                hidden_states = outputs.hidden_states
                if not (buffer is None or buffer.is_empty()):
                    buf_inputs,buf_labels, buf_logits, buf_attention_mask = buffer.get_data(50*2) # 50*self.args.task. OOM if more
                    #TODO: consider data loader if needed for efficient

                    buf_inputs = buf_inputs.long().cuda()
                    buf_labels = buf_labels.long().cuda()
                    buf_logits = buf_logits.float().cuda()
                    buf_attention_mask = buf_attention_mask.long().cuda()

                    outputs = self.model(input_ids=buf_inputs, labels=buf_labels, attention_mask=buf_attention_mask, output_hidden_states=True)

                    loss += self.args.beta * outputs.loss
                    loss += self.args.alpha * self.mse(outputs.hidden_states[-1], buf_logits)

            elif 'adapter_hat' in self.args.baseline \
                    or 'transformer_hat' in self.args.baseline \
                    or 'adapter_bcl' in self.args.baseline\
                    or 'adapter_classic' in self.args.baseline: # Transformer hat also need to deal with loss
                reg = 0
                count = 0

                if mask_pre is not None:
                    # for m,mp in zip(masks,self.mask_pre):
                    for key in set(masks.keys()) & set(mask_pre.keys()):
                        m = masks[key]
                        mp = mask_pre[key]
                        aux = 1 - mp
                        reg += (m * aux).sum()
                        count += aux.sum()
                else:
                    for m_key, m_value in masks.items():
                        reg += m_value.sum()
                        count += np.prod(m_value.size()).item()

                reg /= count

                loss += self.args.lamb * reg

                if self.args.task > 0 and 'adapter_classic' in self.args.baseline:
                    pre_pooled_outputs = []
                    cur_task = self.args.task
                    cur_s = self.args.s
                    for pre_t in [x for x in range(cur_task)]:
                        self.args.s = self.args.smax
                        self.args.task = pre_t

                        with torch.no_grad():
                            pre_outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                                 head_mask=head_mask,
                                                 output_mask=output_mask,
                                                 intermediate_mask=intermediate_mask,
                                                 embedding_mask=embedding_mask,
                                                 output_hidden_states=True)

                        pre_pooled_output = pre_outputs.hidden_states[-1]
                        mean_pre_pooled_output = torch.mean(pre_pooled_output, dim=1)

                        pre_pooled_outputs.append(mean_pre_pooled_output.unsqueeze(-1).clone())


                    self.args.task = cur_task
                    self.args.s = cur_s

                    pre_pooled_outputs = torch.cat(pre_pooled_outputs, -1)

                    cur_pooled_outputs = outputs.hidden_states[-1]
                    mean_cur_pooled_output = torch.mean(cur_pooled_outputs, dim=1)

                    pre_pooled_outputs = torch.cat([pre_pooled_outputs, mean_cur_pooled_output.unsqueeze(-1).clone()],-1)  # include itselves

                    pooled_output = self.model.self_attns[self.args.task](pre_pooled_outputs)  # softmax on task
                    pooled_output = pooled_output.sum(-1)  # softmax on task
                    pooled_output = self.dropout(pooled_output)
                    pooled_output = F.normalize(pooled_output, dim=1)

                    mix_pooled_reps = [mean_cur_pooled_output.clone().unsqueeze(1)]
                    mix_pooled_reps.append(pooled_output.unsqueeze(1).clone())
                    cur_mix_outputs = torch.cat(mix_pooled_reps, dim=1)

                    loss += self.contrast(cur_mix_outputs,con_type='unsupervised')  # train attention and contrastive learning at the same time


            elif 'simcse' in self.args.baseline:
                inputs_ori_ids_dup = inputs_ori_ids.repeat(2, 1)
                labels_dup = labels.repeat(2, 1)
                attention_mask_dup = attention_mask.repeat(2, 1)

                outputs_ori = self.model(input_ids=inputs_ori_ids_dup, labels=labels_dup,
                                         attention_mask=attention_mask_dup,
                                         output_hidden_states=True)

                outputs_ori_hidden_state = outputs_ori.hidden_states[-1].view(-1, 2, 164, 768)

                z1 = outputs_ori_hidden_state[:, 0]
                z2 = outputs_ori_hidden_state[:, 1]
                mean_z1 = torch.mean(z1, dim=1)
                mean_z2 = torch.mean(z2, dim=1)
                simcse_loss = simcse.sequence_level_contrast(mean_z1, mean_z2)



            elif 'contrast' in self.args.baseline and not prune_mdoel:
                inputs_ori_ids_dup = inputs_ori_ids.repeat(2,1)
                labels_dup = labels.repeat(2,1)
                attention_mask_dup = attention_mask.repeat(2,1)

                if 'domain_specific' in self.args.baseline: # what if we choose not to use all, but domain-specific

                    if 'head_mask' not in self.args.baseline:
                        general_head_mask = 0
                    if 'intermediate_mask' not in self.args.baseline:
                        general_intermediate_mask = 0
                    if 'output_mask' not in self.args.baseline:
                        general_output_mask = 0
                    if 'embedding_mask' not in self.args.baseline:
                        general_embedding_mask = 0 # so 1-0=1, keep everything


                    outputs_ori = self.model(input_ids=inputs_ori_ids_dup,labels=labels_dup,attention_mask=attention_mask_dup,
                                             head_mask=(1-general_head_mask),
                                             intermediate_mask=(1-general_intermediate_mask),
                                             output_mask=(1-general_output_mask),
                                             embedding_mask=(1-general_embedding_mask),
                                             only_return_output=True,
                                             output_hidden_states=True)
                    # print('domain_specific')

                else:
                    outputs_ori = self.model(input_ids=inputs_ori_ids_dup,labels=labels_dup,attention_mask=attention_mask_dup,
                                             only_return_output=True,
                                             output_hidden_states=True)

                outputs_pre = self.model(input_ids=inputs_ori_ids,labels=labels,attention_mask=attention_mask,
                                         head_mask=general_head_mask,
                                         intermediate_mask=general_intermediate_mask,
                                         output_mask=general_output_mask,
                                         embedding_mask=general_embedding_mask,
                                         only_return_output=True,
                                         output_hidden_states=True)

                outputs_ori_hidden_state = outputs_ori.hidden_states[-1].view(-1,2,164,768)

                z1 = outputs_ori_hidden_state[:,0]
                z2 = outputs_ori_hidden_state[:,1]
                z3 = outputs_pre.hidden_states[-1]


                mean_z1 = torch.mean(z1, dim=1)
                mean_z2 = torch.mean(z2, dim=1)
                mean_z3 = torch.mean(z3, dim=1)
                contrast_loss = simcse.sequence_level_contrast(mean_z1, mean_z2, mean_z3)


                # for analysis ====================
                # y1 = torch.Tensor([1 for _ in range(mean_z2.size(0))]).long()
                # y2= torch.Tensor([0 for _ in range(mean_z2.size(0))]).long()

                # z = torch.cat([mean_z1,mean_z3])
                # y = torch.cat([y1,y2])


                # tsne.compute_tsne(z,y,'fig')
                # exit()

            elif 'tacl' in self.args.baseline and not prune_mdoel:
                #TODO: we need a teacher for TACL

                outputs_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,output_hidden_states=True)
                outputs_teacher = self.teacher(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,output_hidden_states=True)

                z1 = outputs_teacher.hidden_states[-1]  # anchor: masks
                z2 = outputs_ori.hidden_states[-1]  # positive samples: original
                tacl_loss = utils.tacl_loss(z1, z2, (labels == -100).long(), eps=0.0) # contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.

            elif 'taco' in self.args.baseline and not prune_mdoel:
                #TODO: not done, need check


                outputs_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,output_hidden_states=True)

                inputs_embeds = getattr(self.model, 'roberta').embeddings(inputs_ori_ids)
                z1 = outputs_ori.hidden_states[-1]  # anchor: masks

                global_z1 = z1 - inputs_embeds

                bsz, ntoken, nfeature = z1.size()
                #ws: window_size
                ws = 5
                offset = torch.randint(-ws,0,(bsz,ntoken))
                ids = torch.arange(0,ntoken).expand(bsz,ntoken)

                new_ids = ids+offset
                gloabl_z2 = global_z1[torch.arange(global_z1.shape[0]).unsqueeze(-1), new_ids]# trick to index a 3D tensor using 2D tensor https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor

                global_z1 = F.normalize(global_z1, dim=1)
                gloabl_z2 = F.normalize(gloabl_z2, dim=1)

                taco_loss = utils.taco_loss(global_z1, gloabl_z2) * 1e-5

            elif 'infoword' in self.args.baseline and not prune_mdoel:

                outputs_ori = self.model(input_ids=inputs_ori_ids, labels=labels, attention_mask=attention_mask,output_hidden_states=True)
                outputs_mask = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,output_hidden_states=True)

                #TODO: bad choice
                # print(labels != -100)
                ngram_z1 = outputs_ori.hidden_states[-1]
                # z1 = ngram_z1[new_ids]# trick to index a 3D tensor using 2D tensor https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor
                # z1 = z1.view(ngram_z1.size(0),-1,ngram_z1.size(-1)) # cannot do this, becuase each sequence has a different span

                mean_z1 = []
                for z_id,z in enumerate(ngram_z1):

                    z1 = ngram_z1[z_id][(labels[z_id] != -100).unsqueeze(-1).expand_as(ngram_z1[z_id])] # Only the span for masked
                    z1 = z1.view(1,-1,ngram_z1.size(-1)) # cannot do this, becuase each sequence has a different span
                    mean_z1.append(torch.mean(z1, dim=1))

                mean_z1 = torch.stack(mean_z1).squeeze(1)

                z2 = outputs_mask.hidden_states[-1]
                mean_z2 = torch.mean(z2, dim=1)

                # print('mean_z1: ',mean_z1.size())
                # print('mean_z2: ',mean_z2.size())


                infoword_loss = simcse.sequence_level_contrast(mean_z1, mean_z2)



        return MyRobertaOutput(
            loss = loss,
            contrast_loss = contrast_loss,
            distill_loss = distill_loss,
            simcse_loss=simcse_loss,
            tacl_loss=tacl_loss,
            taco_loss=taco_loss,
            infoword_loss=infoword_loss,
            hidden_states=hidden_states,

        )


class loss(ModelOutput):
    all_attention: torch.FloatTensor = None
    contrast_loss: torch.FloatTensor = None
    distill_loss: torch.FloatTensor = None
    tacl_loss: torch.FloatTensor = None
    taco_loss: torch.FloatTensor = None
    infoword_loss: torch.FloatTensor = None
    simcse_loss: torch.FloatTensor = None
    hidden_state = None







class MyRobertaForSequenceClassification(RobertaForSequenceClassification):

    def __init__(self, config,args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.args = args

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        if 'adapter' in args.baseline:
            self.roberta = add_roberta_adapters(self.roberta, args)

        if 'prompt' in args.baseline:
            self.n_tokens = 100

            self.prompt_embed_pool = nn.ModuleList()

            for i in range(args.ntasks):
                init_prompt_value = torch.FloatTensor(self.n_tokens, self.config.hidden_size).uniform_(-0.5, 0.5)
                embedding = nn.Embedding(self.n_tokens, self.config.hidden_size)
                embedding.weight = nn.parameter.Parameter(init_prompt_value)
                self.prompt_embed_pool.append(embedding)

        elif 'transformer_hat' in args.baseline:

            self.ehead = torch.nn.ModuleList()
            self.eintermediate = torch.nn.ModuleList()
            self.eoutput = torch.nn.ModuleList()
            self.gate = torch.nn.Sigmoid()
            n_layers, n_heads = self.config.num_hidden_layers, self.config.num_attention_heads

            for i in range(args.ntasks):
                self.ehead.append(torch.nn.Embedding(n_layers, n_heads).cuda())
                self.eintermediate.append(torch.nn.Embedding(n_layers, self.config.intermediate_size).cuda())
                self.eoutput.append(torch.nn.Embedding(n_layers, self.config.hidden_size).cuda())


        self.init_weights()


    def transformer_mask(self):

        n_layers, n_heads = self.config.num_hidden_layers, self.config.num_attention_heads

        head_importances = []
        output_importances = []
        intermediate_importances = []


        for i in range(n_layers):
            head_importances.append(self.gate(self.args.s*self.ehead[self.args.task](torch.LongTensor([i]).cuda())))
            intermediate_importances.append(self.gate(self.args.s*self.eintermediate[self.args.task](torch.LongTensor([i]).cuda())))
            output_importances.append(self.gate(self.args.s*self.eoutput[self.args.task](torch.LongTensor([i]).cuda())))


        head_importance = torch.stack(head_importances).squeeze()
        output_importance = torch.stack(output_importances).squeeze()
        intermediate_importance = torch.stack(intermediate_importances).squeeze()

        return head_importance,intermediate_importance,output_importance



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            args=None,
            inputs_ori_ids=None,
            labels_ori=None,
            masked_indices=None,
            output_mask=None,
            intermediate_mask=None,
            embedding_mask=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if 'prompt' in args.baseline:
            inputs_embeds = prompt.cat_learned_embedding_to_input(self, input_ids, args.task).cuda()
            attention_mask = prompt.extend_attention_mask(self, attention_mask).cuda()

            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                output_mask=output_mask,
                intermediate_mask=intermediate_mask,
                embedding_mask=embedding_mask,
                **kwargs
            )

        else:

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                output_mask=output_mask,
                intermediate_mask=intermediate_mask,
                embedding_mask=embedding_mask,
                **kwargs
            )

        sequence_output = outputs[0]

        mean_sequence_output = torch.mean(sequence_output, dim=1).unsqueeze(1)
        logits = self.classifier(mean_sequence_output)

            # logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
