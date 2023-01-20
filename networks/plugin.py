
import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel
from transformers.models.bert.modeling_bert import BertSelfOutput
from transformers.models.roberta.modeling_roberta import RobertaSelfOutput, RobertaSelfAttention
import torch.nn.functional as F


class CapsNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.semantic_capsules = CapsuleLayer(args,'semantic')
        self.tsv_capsules = CapsuleLayer(args,'tsv')
        self.args = args

    def forward(self, x):
        semantic_output = self.semantic_capsules(self.args.task,x,'semantic')
        tsv_output = self.tsv_capsules(self.args.task,semantic_output,'tsv')
        return tsv_output



class CapsuleLayer(nn.Module): #it has its own number of capsule for output
    def __init__(self, args,layer_type):
        super().__init__()


        if layer_type=='tsv':
            self.num_routes = args.ntasks
            self.num_capsules = args.semantic_cap_size
            self.class_dim = args.max_seq_length
            self.in_channel = args.max_seq_length*args.semantic_cap_size
            # self.in_channel = 100
            self.elarger=torch.nn.Embedding(args.ntasks,768)
            self.larger=torch.nn.Linear(args.semantic_cap_size,768) #each task has its own larger way
            self.gate=torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
            self.num_iterations = 3
            self.route_weights = \
                nn.Parameter(torch.randn(self.num_capsules, self.num_routes, self.in_channel, self.class_dim))

            self.tsv = torch.tril(torch.ones(args.ntasks,args.ntasks)).data.cuda()# for backward

        elif layer_type=='semantic':
            self.fc1 = nn.ModuleList([torch.nn.Linear(768, args.semantic_cap_size) for _ in range(args.ntasks)])


        self.args= args

    def forward(self, t,x,layer_type=None):
        if layer_type=='tsv':
            batch_size = x.size(0)
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size()).cuda()
            mask=torch.zeros(self.args.ntasks).data.cuda()
            for x_id in range(self.args.ntasks):
                if self.tsv[t][x_id] == 0: mask[x_id].fill_(-10000) # block future, all previous are the same

            for i in range(self.num_iterations):
                logits = logits*self.tsv[t].data.view(1,1,-1,1,1) #multiply 0 to future task
                logits = logits + mask.data.view(1,1,-1,1,1) #add a very small negative number
                probs = self.my_softmax(logits, dim=2)
                vote_outputs = (probs * priors).sum(dim=2, keepdim=True) #voted
                outputs = self.squash(vote_outputs)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits

            h_output = vote_outputs.view(batch_size,self.args.max_seq_length,-1)

            h_output= self.larger(h_output)

            return h_output

        elif layer_type=='semantic':

            outputs = [fc1(x).view(x.size(0), -1, 1) for fc1 in self.fc1]
            outputs = torch.cat(outputs, dim=-1)

            outputs = self.squash(outputs)
            return outputs.transpose(2,1)

    def my_softmax(self,input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)



class Adapter(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.fc1 = torch.nn.Linear(768, 128) #bottle net size
        self.fc2 = torch.nn.Linear(128, 768)
        self.activation = torch.nn.GELU()
        self.args = args

        if 'adapter_hat' in self.args.baseline\
                or 'adapter_classic' in self.args.baseline:
            self.efc1=torch.nn.Embedding(args.ntasks,128)
            self.efc2=torch.nn.Embedding(args.ntasks,768)
            self.gate=torch.nn.Sigmoid()

        elif 'adapter_bcl' in self.args.baseline:
            self.capsule_net = CapsNet(args)
            self.efc1=torch.nn.Embedding(args.ntasks,128)
            self.efc2=torch.nn.Embedding(args.ntasks,768)
            self.gate=torch.nn.Sigmoid()


    def transfer_weight(self, model_for_transfer):
        self.fc1.weight = nn.Parameter(model_for_transfer.fc1.weight)
        self.fc1.bias = nn.Parameter(model_for_transfer.fc1.bias)
        self.fc2.weight = nn.Parameter(model_for_transfer.fc2.weight)
        self.fc2.bias = nn.Parameter(model_for_transfer.fc2.bias)

    def forward(self, x):

        if 'adapter_hat' in self.args.baseline\
                or 'adapter_classic' in self.args.baseline:
            gfc1, gfc2 = self.mask()
            h=self.activation(self.fc1(x))
            h=h*gfc1.expand_as(h)

            h=self.activation(self.fc2(h))
            h=h*gfc2.expand_as(h)

        elif 'adapter_bcl' in self.args.baseline:

            capsule_output = self.capsule_net(x)

            h = x + capsule_output  # skip-connection
            # h = capsule_output #skip-connection

            # task specifc
            gfc1, gfc2 = self.mask()

            h = self.activation(self.fc1(h))
            h = h * gfc1.expand_as(h)

            h = self.activation(self.fc2(h))
            h = h * gfc2.expand_as(h)

        else:
            h = self.activation(self.fc1(x))
            h = self.activation(self.fc2(h))

        return x + h



    def mask(self):

       efc1 = self.efc1(torch.LongTensor([self.args.task]).cuda())
       efc2 = self.efc2(torch.LongTensor([self.args.task]).cuda())

       gfc1=self.gate(self.args.s*efc1)
       gfc2=self.gate(self.args.s*efc2)

       return [gfc1,gfc2]


class MOE(nn.Module):
    def __init__(self, args):
        """
        Args are seperated into different sub-args here
        """
        super().__init__()
        self.args = args

        if 'adapter_moe' in args.baseline or 'adapter_demix' in args.baseline:
            self.adapters = torch.nn.ModuleList()
            for i in range(args.ntasks):
                self.adapters.append(Adapter(args))
        elif 'adapter_share' in args.baseline \
                or 'adapter_hat' in args.baseline \
                or 'adapter_bcl' in args.baseline \
                or 'adapter_classic' in args.baseline:
            self.adapters = Adapter(args)


    def forward(self, x):

        if 'adapter_moe' in self.args.baseline or 'adapter_demix' in self.args.baseline:
            return self.adapters[self.args.task](x)
        elif 'adapter_share' in self.args.baseline \
                or 'adapter_hat' in self.args.baseline\
                or 'adapter_bcl' in self.args.baseline\
                or 'adapter_classic' in self.args.baseline:
            return self.adapters(x)





# class RobertaSelfOutput(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states


class RobertaAdaptedSelfOutput(nn.Module):
    """
    Modify RobertaSelfOutput to insert adapter into Roberta model.
    """
    # the original code only valid if you separately load weight and add (unless write inside from_pretrained)
    # if you add first and load, problem will happen and I don't want to write inside from_pretrianed

    def __init__(self,
                 self_output: RobertaSelfOutput,
                 args):
        super(RobertaAdaptedSelfOutput, self).__init__()
        self.moe = MOE(args) # a real adapter

        self.dense = self_output.dense
        self.LayerNorm = self_output.LayerNorm
        self.dropout = self_output.dropout


    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, output_mask=None, **kwargs):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


def adapt_roberta_self_output(args):
    return lambda self_output: RobertaAdaptedSelfOutput(self_output,args)


def add_roberta_adapters(roberta_model: RobertaModel, args) -> RobertaModel:
    """
    Add MOEAdapter into each Roberta layer.
    """
    for layer in roberta_model.encoder.layer:
        layer.attention.output = adapt_roberta_self_output(args)(
            layer.attention.output)
        layer.output = adapt_roberta_self_output(args)(layer.output)

    return roberta_model
