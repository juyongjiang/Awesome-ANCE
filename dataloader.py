import sys
sys.path += ['./']
import os
import argparse
import json
import numpy as np
import torch
from models import MSMarcoConfigDict, ALL_MODELS
from torch.utils.data import TensorDataset, IterableDataset

class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type']) # "int32"
            self.total_number = meta['total_number']
            # the size of single record: passage_len, passage, stored by bytes
            self.record_size = int(meta['embedding_size']) * self.dtype.itemsize + 4 # dtype.itemsize = 4
        
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(self.total_number) # generate random list shuffle([i for i in range(total_number)])
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size) # read record_size bytes
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError("Index {} is out of bound for cached embeddings of size {}".format(key, self.total_number))
        self.f.seek(key * self.record_size) # offset
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number

def GetProcessingFn(args, query=False):
    def fn(vals, i): # i: id
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length
        """
        Args:
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
            token_type_ids: Segment token indices to indicate first and second portions of the inputs.
            label: Label corresponding to the input
        """
        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len
        # id, passage_each_token_id, [1,1,1, ..., 0,0,0], [0,0,0, ..., 0,0,0]/[1,1,1, ..., 0,0,0]
        passage_collection = [(i, passage, attention_mask, token_type_ids)] 

        # change input into torch.tensor format
        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long) # [id]
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int) # [passage_each_token_id]
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool) # [1,1,1, ..., 0,0,0]
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8) # [0,0,0, ..., 0,0,0]/[1,1,1, ..., 0,0,0]
        # passage_each_token_id, [1,1,1, ..., 0,0,0], [0,0,0, ..., 0,0,0]/[1,1,1, ..., 0,0,0], id
        # zip a, b, c, d, https://blog.csdn.net/qq_40211493/article/details/107529148
        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor) # like zip

        return [ts for ts in dataset] # [[a,b,c,d]]

    return fn

# read from ann file to get query id, pos_id, neg_id which is index in the dataset
def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        # all_input_ids_a = []
        # all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn

def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    # query_cache: [(len, [id1, id2, id3, ....., 1, 1, 1, 1]), ...]
    # passage_cache: [(len, [id1, id2, id3, ....., 1, 1, 1, 1]), ...]
    def fn(line, i): # ann data: for i, line in enumerate(ann_data.readlines())
        line_arr = line.split('\t')
        # qid, pos_pid, neg_pids (token index in the dataset)
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        # all_input_ids_a = []
        # all_attention_mask_a = []
        
        # qid, pos_pid, neg_pids are the index from preprocessed dataset
        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0] # [a,b,c,d]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0] # [a,b,c,d]
        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
                   neg_data[0], neg_data[1], neg_data[2]) # qid, pos_pid, and neg_pid are not needed. 

    return fn

class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn, distributed=True):
        super().__init__()
        self.elements = elements  # elements: ann_data_file -> f.readlines()
        self.fn = fn
        self.num_replicas=-1 
        self.distributed = distributed
    
    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("Not running in distributed mode")
            
        for i, element in enumerate(self.elements):
            if self.distributed and self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            # (query: all_input_ids_a, all_attention_mask_a, all_token_type_ids_a
            # positive: all_input_ids_a, all_attention_mask_a, all_token_type_ids_a
            # negtive: all_input_ids_a, all_attention_mask_a, all_token_type_ids_a)
            records = self.fn(element, i)
            for rec in records:
                yield rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/MSMARCO", type=str, help="The input data dir",)
    parser.add_argument("--out_data_dir", default="./data/MSMARCO/preprocessed", type=str, help="The output data dir",)
    parser.add_argument("--model_type", default="rdot_nll", type=str, help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="Path to pre-trained model or shortcut name selected in the list: " +", ".join(ALL_MODELS),)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--max_seq_length", default=2048, type=int, help="The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_doc_character", default=10000, type=int, help="used before tokenizer to save tokenizer latency",)
    parser.add_argument("--data_type", default=1, type=int, help="0 for doc, 1 for passage",)
    args = parser.parse_args()

    embedding_cache = EmbeddingCache(out_passage_path)
    with embedding_cache as emb:
        print("Passage embedding cache first line", emb[0])

    # embedding cache
    embedding_cache = EmbeddingCache(out_query_path)
    with embedding_cache as emb:
        print("Query embedding cache first line", emb[0])


if __name__ == '__main__':
    main()
