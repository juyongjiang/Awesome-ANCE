import sys
sys.path += ['../']
import os
import json
import random
import pickle
import numpy as np
import re
import torch
from torch import nn
import torch.distributed as dist

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