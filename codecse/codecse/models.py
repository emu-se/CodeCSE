"""
This file is created by modifying SimCSE model
https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
"""

import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel 
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch.distributed as dist
from .configurations import CLRobertaConfig

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            sum_state = (last_hidden * attention_mask.unsqueeze(-1)).sum(1)
            cnt_state = attention_mask.sum(-1).unsqueeze(-1)
            return ( sum_state / cnt_state )

        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def cl_init(cls, sent_emb, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = config.cl_pooler_type
    cls.pooler = Pooler(cls.pooler_type)
    cls.temp = config.cl_temp # temperature for similarity
    cls.mlp_layer = config.cl_mlp_layer
    cls.loss = config.cl_loss
    assert cls.loss in ["asymmetric", "symmetric", "asymmetric_opp"], "unrecognized loss type %s" % cls.loss
        
    if cls.mlp_layer > 0:
        cls.mlp = MLPLayer(config)
        if cls.mlp_layer == 2:
            cls.mlp_2 = MLPLayer(config)
    
    # if the model is loaded only for evaluation (using the sent_emb forward)
    # we don't need the similarity function and the temp arg is not needed
    if not sent_emb:
        cls.sim = Similarity(temp=cls.temp)

    # if the model is GraphCodeBERTForCL, we need three more settings
    if isinstance(cls, GraphCodeBERTForCL):
        cls.code_length = config.gc_code_length
        cls.data_flow_length = config.gc_data_flow_length
        cls.nl_length = config.gc_nl_length
        assert cls.code_length >= 0, f"config.gc_code_length needs to be set >= 0 for GraphCodeBERTForCL"
        assert cls.data_flow_length >= 0, f"config.gc_data_flow_length needs to be set >= 0 for GraphCodeBERTForCL"
        assert cls.nl_length >= 0, f"config.gc_nl_length needs to be set >= 0 for GraphCodeBERTForCL"

    cls.init_weights()

class GraphCodeBERTForCL(RobertaPreTrainedModel):
    config_class = CLRobertaConfig
    def __init__(self, config, sent_emb=False, *model_args, **model_kargs):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Not supporting mlm and no lm_head
        cl_init(self, sent_emb, config)
        self.code_inputs_length = self.code_length + self.data_flow_length
        return
    
    # def from_pretrained():
        

    def sentemb_forward(cls, outputs, attention_mask, return_dict):
        pooler_output = cls.pooler(attention_mask, outputs)
        if cls.mlp_layer > 0:
            pooler_output = cls.mlp(pooler_output)
            if cls.mlp_layer == 2:
                pooler_output = cls.mlp_2(pooler_output)

        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def cl_forward(cls, code_outputs, code_attn_mask, nl_outputs, nl_attn_mask, return_dict):
        # Pooling
        code_pooler_output = cls.pooler(code_attn_mask, code_outputs)
        nl_pooler_output = cls.pooler(nl_attn_mask, nl_outputs)

        pooler_output = torch.stack((code_pooler_output, nl_pooler_output), 1)
        if cls.mlp_layer > 0:
            pooler_output = cls.mlp(pooler_output)
            if cls.mlp_layer == 2:
                pooler_output = cls.mlp_2(pooler_output)

        # Separate representation
        if cls.loss == "asymmetric_opp":
            z2, z1 = pooler_output[:,0], pooler_output[:,1]
        else:
            z1, z2 = pooler_output[:,0], pooler_output[:,1]

        # orig_shape = z1.shape

        if dist.is_initialized() and cls.training:
            print(f"distributed training")
             # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
        
        # print(f"z1 shape: {orig_shape} -> {z1.shape}")
        # No hard negatives
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        output = (cos_sim,)
        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()
        loss_1 = loss_fct(cos_sim, labels)
        if cls.loss == "symmetric":
            opposite_cos_sim = torch.transpose(cos_sim, 0, 1)
            output = output + (opposite_cos_sim,)
            loss_2 = loss_fct(opposite_cos_sim, labels)
            loss = (loss_1 + loss_2)/2
        else: # "asymmetric" or "asymmetric_opp"
            loss = loss_1

        # the following line works under multi-gpu setup without distributed training
        # convert the scalar to a one-dim tensor (later to be gathered in multi-gpu training)
        # loss = loss.view(1)
        
        if not return_dict:
            output = output + code_outputs[2:] + nl_outputs[2:]
            result = ((loss,) + output) if loss is not None else output
            return result
        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=(code_outputs.hidden_states, nl_outputs.hidden_states),
            attentions=(code_outputs.attentions, nl_outputs.attentions),
        )

    def get_code_outputs(self, code_inputs, nodes_to_token_mask, nodes_mask, attn_mask, position_idx):
        inputs_embeddings=self.roberta.embeddings.word_embeddings(code_inputs)
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(nodes_mask.eq(0))[:,:,None]+avg_embeddings*nodes_mask[:,:,None]

        code_outputs = self.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        return code_outputs

    def get_nl_outputs(self, nl_inputs):
        nl_outputs = self.roberta(
            nl_inputs,
            attention_mask=nl_inputs.ne(self.config.pad_token_id),
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        return nl_outputs

    def forward(self, 
        attn_mask=None,
        position_idx=None, 
        input_ids=None,
        nodes_mask=None,
        nodes_to_token_mask=None,
        return_dict=None,
        sent_emb=None,): 
        if sent_emb == "code":
            # do sent_emb for code
            code_inputs = input_ids
            code_outputs = self.get_code_outputs(code_inputs, nodes_to_token_mask, nodes_mask, attn_mask, position_idx)
            return self.sentemb_forward(code_outputs, code_inputs.ne(self.config.pad_token_id), return_dict)
        elif sent_emb == "nl":
            # do sent_emb for nl
            nl_inputs = input_ids
            nl_outputs = self.get_nl_outputs(nl_inputs)
            return self.sentemb_forward(nl_outputs, nl_inputs.ne(self.config.pad_token_id), return_dict)
        else:
            # when setn_emb is None (or not code/nl), we split the input_ids into code/nl inputs for cl training
            code_inputs, nl_inputs = torch.hsplit(input_ids, [self.code_inputs_length])
            code_outputs = self.get_code_outputs(code_inputs, nodes_to_token_mask, nodes_mask, attn_mask, position_idx)
            nl_outputs = self.get_nl_outputs(nl_inputs)
            return self.cl_forward(code_outputs, code_inputs.ne(self.config.pad_token_id), nl_outputs, nl_inputs.ne(self.config.pad_token_id), return_dict)
        
    