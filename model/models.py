import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    # BertModel,
    # BertTokenizer,
    # BertConfig
)
import torch.nn.functional as F

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        # assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0] # [batch, len, dim] -> # [batch, 0, dim] -> # [batch, dim] using the first [cls] as the final output

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

## FirstP
class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # last linear transfer layer
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights) # initialize all layers' parameters weight

    def query_emb(self, input_ids, attention_mask):
        # roberta accepts input_ids, and attention_mask for each sequence, i.e., [token_id1, token_id2, ...], [1,1,1, ..., 0,0,0]
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb)) # linear layer, following layerNorm
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
# (query_data[0], query_data[1], # content, mask
#  pos_data[0], pos_data[1],
#  neg_data[0], neg_data[1],) 
class NLL(EmbeddingMixin):
    def forward(self, query_ids, attention_mask_q, 
                      input_ids_a=None, attention_mask_a=None, # positive passage
                      input_ids_b=None, attention_mask_b=None, # negative passage
                      is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        # get the dense representation of query, postive passage, negtive passage
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        # nll loss
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1), (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1) # apply in the dim=1 
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)
    
## MaxP
class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512 # base length of input

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(batchS, chunk_factor, full_length //chunk_factor).reshape(batchS *chunk_factor, full_length //chunk_factor)
        attention_mask_seq = attention_mask.reshape(batchS, chunk_factor, full_length // chunk_factor).reshape(batchS * chunk_factor, full_length // chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq, attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(outputs_k[0])  # [batch, len, dim] -> [batch, len, 768]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :]) # [batch, 0, 768] -> [batch, 768] using the [cls] as the final output

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]

class NLL_MultiChunk(EmbeddingMixin):
    def forward(self, query_ids, attention_mask_q, 
                      input_ids_a=None, attention_mask_a=None, 
                      input_ids_b=None, attention_mask_b=None, 
                      is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q) # [batchS, embeddingS]
        a_embs = self.body_emb(input_ids_a, attention_mask_a) # [batchS, chunk_factor, embeddingS]
        b_embs = self.body_emb(input_ids_b, attention_mask_b) # [batchS, chunk_factor, embeddingS]

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len # the number of chunk

        # special handle of attention mask -----[batchS, chunk_factor, base_len] -> [:, :, 0] - > [batchS, chunk_factor]
        attention_mask_body = attention_mask_a.reshape(batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float() # [batchS, chunk_factor]
        a12 = torch.matmul(q_embs.unsqueeze(1), a_embs.transpose(1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float() # [batchS, chunk_factor]
        a12 = torch.matmul(q_embs.unsqueeze(1), b_embs.transpose(1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat([logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1) # apply in the dim=1
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)
        
# --------------------------------------------------
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (RobertaConfig,) if hasattr(conf, 'pretrained_config_archive_map')),(),)

class MSMarcoConfig:
    def __init__(self, name, model, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.use_mean = use_mean
        # model config and tokenizer
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class
        self.model_class = model

configs = [MSMarcoConfig(name="rdot_nll", model=RobertaDot_NLL_LN, use_mean=False,),
           MSMarcoConfig(name="rdot_nll_multi_chunk", model=RobertaDot_CLF_ANN_NLL_MultiChunk, use_mean=False,),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}