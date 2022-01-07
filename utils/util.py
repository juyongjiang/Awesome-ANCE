import sys
sys.path += ['../']
import pandas as pd
from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import isfile, join
import json
import logging
import random
import pytrec_eval
import pickle
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import re
from model.models import MSMarcoConfigDict, ALL_MODELS
from typing import List, Set, Dict, Tuple, Callable, Iterable, Any
logger = logging.getLogger(__name__)

def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer): # judge whether obj exist attr (layer)
            obj = getattr(obj, layer) # get the value of layer
        else:
            return None
    return obj

def barrier_array_merge(args, data_array, merge_axis=0, prefix="", load_cache=False, only_load_in_master=False):
    # data array: [B, any dimension]
    # merge alone one axis
    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(rank), prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg

def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids

def pad_ids(input_ids, attention_mask, token_type_ids, max_length,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length
    padding_type = [pad_token_segment_id] * padding_length
    padding_attention = [0 if mask_padding_with_zero else 1] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
            attention_mask = padding_attention + attention_mask
            token_type_ids = padding_type + token_type_ids
        else:
            input_ids = input_ids + padding_id
            attention_mask = attention_mask + padding_attention
            token_type_ids = token_type_ids + padding_type

    return input_ids, attention_mask, token_type_ids

# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}
    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)

def get_checkpoint_no(checkpoint_path):
    nums = re.findall(r'\d+', checkpoint_path)
    return int(nums[-1]) if len(nums) > 0 else 0

def get_latest_ann_data(ann_data_path): # ann_dir
    ANN_PREFIX = "ann_ndcg_"
    if not os.path.exists(ann_data_path):
        print("%s is not existed" % ann_data_path)
        return -1, None, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len(ANN_PREFIX)
    # sequence of ann data with ann_ndcg_[data_no], [data_no] represents which time it generates.
    data_no_list = [int(s[num_start_pos:]) for s in files if s[:num_start_pos] == ANN_PREFIX]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(ann_data_path, ANN_PREFIX + str(data_no)), 'r') as f:
            ndcg_json = json.load(f)
        return data_no, os.path.join(ann_data_path, "ann_training_data_" + str(data_no)), ndcg_json
    return -1, None, None

# out_passage_path, 32, 8 + 4 + args.max_seq_length * 4
def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b

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

class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn, distributed=True):
        super().__init__()
        self.elements = elements
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
            records = self.fn(element, i)
            for rec in records:
                yield rec
# input the doc then use tokenizer to split each word id
def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):
    configObj = MSMarcoConfigDict[args.model_type] # rdot_nll
    tokenizer = configObj.tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=None,)

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f: # i is the index of processing
        for idx, line in enumerate(in_f):
            if idx % num_process != i: # distribute file to correspoinding processing
                continue
            # transfer each line to "p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + 
            # content=np.array(input_id_b, np.int32).tobytes(): max length"
            out_f.write(line_fn(args, line, tokenizer))

# multiple processing operation 
def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file, args=(args, i, num_process, in_path, out_path, line_fn,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list