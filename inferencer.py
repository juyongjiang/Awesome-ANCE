import sys
import torch
import os
import faiss ## generate new ann index [qid, pos_id, neg_id]
import numpy as np
import argparse
import json
import logging
import random
import time
import pytrec_eval ##
import csv
import copy
import transformers
import torch.distributed as dist
##
from dataloader import GetProcessingFn, EmbeddingCache, StreamingDataset
from models import MSMarcoConfigDict, ALL_MODELS
from utils.util import barrier_array_merge, convert_to_string_id, is_first_worker, get_checkpoint_no, get_latest_ann_data
##
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import isfile, join
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

def GenerateNegativePassaageID(args, query_embedding2id, passage_embedding2id, training_query_positive_id, I_nearest_neighbor, effective_q_id):
    # training_query_positive_id: {query_id: passage_id, ...}
    # I_nearest_neighbor: _, I = cpu_index.search(query_embedding, top_k) # I: [query_embedding.shape[0], 100 (passage id)]
    query_negative_passage = {}
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0  # only meaningful if it is SelectTopK = True
    num_queries = 0

    for query_idx in range(I_nearest_neighbor.shape[0]):
        query_id = query_embedding2id[query_idx]
        if query_id not in effective_q_id:
            continue
        num_queries += 1
        pos_pid = training_query_positive_id[query_id]
        top_ann_pid = I_nearest_neighbor[query_idx, :].copy() # [100 (passage id)]

        if SelectTopK: # select negative from topk-topk nearest_neighbor, which are informative negative samples
            selected_ann_idx = top_ann_pid[:args.negative_sample + 1] # e.g. negative_sample=5, num+1 can make sure get negative samples.
        else:
            negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
            random.shuffle(negative_sample_I_idx) # random select from topk negative samples not topk-topk
            selected_ann_idx = top_ann_pid[negative_sample_I_idx]

        query_negative_passage[query_id] = []
        neg_cnt = 0
        rank = 0
        for idx in selected_ann_idx:
            neg_pid = passage_embedding2id[idx]
            rank += 1
            if neg_pid == pos_pid:
                if rank <= 10:
                    mrr += 1 / rank
                continue
            if neg_pid in query_negative_passage[query_id]:
                continue
            if neg_cnt >= args.negative_sample:
                break
            query_negative_passage[query_id].append(neg_pid)
            neg_cnt += 1

    if SelectTopK:
        print("Rank:" + str(args.rank) + " --- ANN MRR:" + str(mrr / num_queries))

    return query_negative_passage # {query_id: negative_passage_id, ...}

# query id [all_data_num, 1], passage id, positive id ({query_id: {passage_id:rel, ...}, ...}), retrieval topk id
def EvalDevQuery(args, query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor):
    # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    prediction = {} # {query_id: {passage_id:rel, ...}, ...}
    for query_idx in range(I_nearest_neighbor.shape[0]): # all number data
        query_id = query_embedding2id[query_idx] # 
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy() # 100
        selected_ann_idx = top_ann_pid[:50] # topk=50 results passage index
        rank = 0
        seen_pid = set()
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]
            if pred_pid not in seen_pid:
                # this check handles multiple vector per document
                rank += 1
                prediction[query_id][pred_pid] = -rank # unique passage id to avoid repeating
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    # {query_id: {passage_id:rel, ...}, ...}
    evaluator = pytrec_eval.RelevanceEvaluator(convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut'})

    result = evaluator.evaluate(convert_to_string_id(prediction))
    eval_query_cnt = 0
    ndcg = 0
    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
    final_ndcg = ndcg / eval_query_cnt
    print("Rank:" + str(args.rank) + " --- ANN NDCG@10:" + str(final_ndcg))

    return final_ndcg, eval_query_cnt

def InferenceEmbeddingFromStreamDataLoader(args, model, train_dataloader, is_query_inference=True, prefix=""):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for batch in tqdm(train_dataloader, desc="Inferencing", disable=args.local_rank not in [-1, 0], position=0, leave=True):
        # batch: all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor [index of dataset]
        idxs = batch[3].detach().numpy()  # [#B]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs) # query1 = self.norm(self.embeddingHead(full_emb)) # linear layer, following layerNorm
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy() # detach: avoid gradient backward anymore

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3: # [batchS, chunk_factor, embeddingS]
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0) # [all_data_num, embeddingS]
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id # [all_data_num, embeddingS], # [all_data_num, 1]

# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference=True): # f: input data
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn) # fn: passage_each_token_id, [1,1,1, ..., 0,0,0], [0,0,0, ..., 0,0,0]/[1,1,1, ..., 0,0,0], id
    inference_dataloader = DataLoader(inference_dataset, batch_size=inference_batch_size) # single input, not Triplet

    if args.local_rank != -1:
        dist.barrier()  # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(args, 
                                                                       model, 
                                                                       inference_dataloader, 
                                                                       is_query_inference=is_query_inference, 
                                                                       prefix=prefix)

    logger.info("merging embeddings from multiple processings")
    # preserve to memory, prefix="dev_query__emb_p_" or "passage__emb_p_" or "query__emb_p_"
    full_embedding = barrier_array_merge(args, _embedding, prefix=prefix + "_emb_p_", load_cache=False, only_load_in_master=True)
    full_embedding2id = barrier_array_merge(args, _embedding2id, prefix=prefix + "_embid_p_", load_cache=False, only_load_in_master=True)

    return full_embedding, full_embedding2id

def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower() 
    configObj = MSMarcoConfigDict[args.model_type]

    args.model_name_or_path = checkpoint_path # saved/checkpint-[step_no]
    config = configObj.config_class.from_pretrained(args.model_name_or_path,
                                                    num_labels=num_labels,
                                                    finetuning_task="MSMarco",
                                                    cache_dir=None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(args.model_name_or_path,
                                                          do_lower_case=True,
                                                          cache_dir=None,
    )
    model = configObj.model_class.from_pretrained(args.model_name_or_path,
                                                  from_tf=bool(".ckpt" in args.model_name_or_path),
                                                  config=config,
                                                  cache_dir=None,
    )
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True,
        )
    return config, tokenizer, model

def generate_new_ann(args, output_num, checkpoint_path, training_query_positive_id, dev_query_positive_id, latest_step_num):
    config, tokenizer, model = load_model(args, checkpoint_path) # saved/checkpint-[step_no]

    #==============================================================================================
    logger.info("***** inference of dev query *****")
    dev_query_collection_path = os.path.join(args.data_dir, "dev-query") # preprocessed data path
    dev_query_cache = EmbeddingCache(dev_query_collection_path) # passage_len, passage
    with dev_query_cache as emb:
        # [all_data_num, embeddingS]
        dev_query_embedding, dev_query_embedding2id = StreamInferenceDoc(args, model, 
                                                                         GetProcessingFn(args, query=True), # test query
                                                                         "dev_query_" + str(latest_step_num) + "_", # latest_step_num: model training step
                                                                         emb, # dev_query_cache
                                                                         is_query_inference=True)
    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        passage_embedding, passage_embedding2id = StreamInferenceDoc(args, model, 
                                                                     GetProcessingFn(args, query=False), # passage
                                                                     "passage_" + str(latest_step_num) + "_", 
                                                                     emb, 
                                                                     is_query_inference=False)
    logger.info("***** Done passage inference (test) *****") 
    if args.inference:
        return
    logger.info("***** inference of train query *****")
    train_query_collection_path = os.path.join(args.data_dir, "train-query")
    train_query_cache = EmbeddingCache(train_query_collection_path)
    with train_query_cache as emb:
        query_embedding, query_embedding2id = StreamInferenceDoc(args, model, 
                                                                 GetProcessingFn(args, query=True), # train query
                                                                 "query_" + str(latest_step_num) + "_", 
                                                                 emb, 
                                                                 is_query_inference=True)
    #==============================================================================================
    if is_first_worker():
        dim = passage_embedding.shape[1] # [all_data_num, embeddingS]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        top_k = args.topk_training # top k from which negative samples are collected
        # ---------------------------------------------------------------------------------
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(passage_embedding) # as passage index
        logger.info("***** Done ANN Index *****")
        # ---------------------------------------------------------------------------------

        # measure ANN mrr
        # I: [number of queries, topk]
        _, dev_I = cpu_index.search(dev_query_embedding, 100) # topk=100 -> [all_data_num, 100 (passage id)]
        # query id, passage id, positive id ({query_id: {passage_id:rel, ...}, ...}), retrieval topk id
        dev_ndcg, num_queries_dev = EvalDevQuery(args, dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I)

        # Construct new traing set ==================================
        chunk_factor = args.ann_chunk_factor
        effective_idx = output_num % chunk_factor # devide training queries into chunks effective_idx = 0 ~ chunk_factor-1
        if chunk_factor <= 0:
            chunk_factor = 1

        num_queries = len(query_embedding) # training dataset length
        queries_per_chunk = num_queries // chunk_factor

        q_start_idx = queries_per_chunk * effective_idx
        q_end_idx = num_queries if (effective_idx == (chunk_factor - 1)) else (q_start_idx + queries_per_chunk) # if the last chunk

        #
        query_embedding = query_embedding[q_start_idx:q_end_idx]
        query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]
        logger.info("Chunked {} query from {}".format(len(query_embedding), num_queries))
        #---------------------------------------------------negative samples generation--------------------------------------------------------------- 
        # I: [number of queries, topk]
        _, I = cpu_index.search(query_embedding, top_k) # I: [query_embedding.shape[0], 100 (passage id)]
        effective_q_id = set(query_embedding2id.flatten())
        # {query_id: negative_passage_id, ...}
        query_negative_passage = GenerateNegativePassaageID(args,
                                                            query_embedding2id,
                                                            passage_embedding2id,
                                                            training_query_positive_id, # training_query_positive_id, {query_id: passage_id, ...}
                                                            I,
                                                            effective_q_id)

        #-------------------------------------------------------new ann data generation--------------------------------------------------------
        logger.info("***** Construct ANN Triplet *****")
        # ann_dir/ann_training_data_[output_num]-(query_id, pos_pid, neg_pid)
        # ann_dir/ann_ndcg_[output_num]-({ndcg: dev_ndcg (ndcg results from dev dataset), checkpoint: checkpoint_path (current checkpoint used for generation)})
        train_data_output_path = os.path.join(args.ann_dir, "ann_training_data_" + str(output_num))
        with open(train_data_output_path, 'w') as f:
            query_range = list(range(I.shape[0])) # [0, 1, 2, ..., queries_per_chunk-1]
            random.shuffle(query_range)
            for query_idx in query_range:
                query_id = query_embedding2id[query_idx]
                # training_query_positive_id: {query_id: passage_id, ...}, query_negative_passage: {query_id: negative_passage_id, ...}
                if query_id not in effective_q_id or query_id not in training_query_positive_id:
                    continue
                pos_pid = training_query_positive_id[query_id]
                f.write("{}\t{}\t{}\n".format(query_id, pos_pid, ','.join(str(neg_pid) for neg_pid in query_negative_passage[query_id])))

        ndcg_output_path = os.path.join(args.ann_dir, "ann_ndcg_" + str(output_num))
        with open(ndcg_output_path, 'w') as f:
            json.dump({'ndcg': dev_ndcg, 'checkpoint': checkpoint_path}, f) # checkpoint_path, saved/checkpint-[step_no]
        #---------------------------------------------------------------------------------------------------------------------------------------
        return dev_ndcg, num_queries_dev

def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir): # training_dir means "saved": which store training model checkpoint
        return args.init_model_dir, 0
    subdirectories = list(next(os.walk(args.training_dir))[1])

    def valid_checkpoint(checkpoint):
        chk_path = os.path.join(args.training_dir, checkpoint)
        scheduler_path = os.path.join(chk_path, "scheduler.pt")
        return os.path.exists(scheduler_path)

    checkpoint_nums = [get_checkpoint_no(s) for s in subdirectories if valid_checkpoint(s)]

    if len(checkpoint_nums) > 0:
        # saved/checkpint-[step_no], [step_no]
        return os.path.join(args.training_dir, "checkpoint-" + str(max(checkpoint_nums))), max(checkpoint_nums)
    
    return args.init_model_dir, 0

def load_positive_ids(args):
    logger.info("Loading query_2_pos_docid")
    training_query_positive_id = {}
    # each line: str(qid2offset[topicid]) + "\t" + str(pid2offset[docid]) + "\t" + rel + "\n"
    query_positive_id_path_train = os.path.join(args.data_dir, "train-qrel.tsv")
    with open(query_positive_id_path_train, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            assert rel == "1"
            topicid = int(topicid) # query id index
            docid = int(docid) # passage id index
            training_query_positive_id[topicid] = docid # {query_id: passage_id, ...}

    logger.info("Loading dev query_2_pos_docid")
    dev_query_positive_id = {}
    query_positive_id_path_dev = os.path.join(args.data_dir, "dev-qrel.tsv")
    with open(query_positive_id_path_dev, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            topicid = int(topicid) # query id index
            docid = int(docid) # passage id index
            if topicid not in dev_query_positive_id:
                dev_query_positive_id[topicid] = {}
            dev_query_positive_id[topicid][docid] = int(rel) # {query_id: {passage_id:rel, ...}, ...}

    return training_query_positive_id, dev_query_positive_id

def ann_data_gen(args):
    # get latest ann data
    # ann_no: latest generated ann data index; 
    # ann_path: the path of training ann file [qid, pos_id, neg_id]; 
    # ndcg_json: a dict of checkpoint path info
    ann_no, ann_path, ndcg_json = get_latest_ann_data(args.ann_dir) # we only need ann_no for record the next generation no. 
    output_num = ann_no + 1 # for this time generation
    if is_first_worker():
        if not os.path.exists(args.ann_dir):
            os.makedirs(args.ann_dir)
    # positive id for train and dev dataset
    # {query_id: passage_id, ...}, {query_id: {passage_id:rel, ...}, ...}
    training_positive_id, dev_positive_id = load_positive_ids(args)
    last_checkpoint = args.init_model_dir #
    while args.end_output_num == -1 or output_num <= args.end_output_num:
        # get latest DR model checkpoint path
        # saved/checkpint-[step_no], [step_no]
        next_checkpoint, latest_step_num = get_latest_checkpoint(args) # if not, args.init_model_dir, 0

        if next_checkpoint == last_checkpoint:
            time.sleep(60) # avoid repeatly generation
        else:
            logger.info("start generate ann data number %d", output_num)
            logger.info("next checkpoint (latest) at " + next_checkpoint)
            # generate new ann data (Important!)
            generate_new_ann(args, output_num, next_checkpoint, training_positive_id, dev_positive_id, latest_step_num)
            #############
            if args.inference: # only generate new ann data at one time
                break
            logger.info("finished generating ann data number %d", output_num)
            output_num += 1 # for the next time generation
            last_checkpoint = next_checkpoint
        
        if args.local_rank != -1:
            dist.barrier()

def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    logging.basicConfig(filename=os.path.join(args.log_dir, "inferencer_ance.log"),
                        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN, # only in rank 0 to print logging.INFO
    )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank,
                   device,
                   args.n_gpu,
                   bool(args.local_rank != -1),
    )

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir", default="./data/MSMARCO/preprocessed", type=str, help="The preprocessed data dir",)
    parser.add_argument("--training_dir", default="./saved", type=str, help="Training dir for latest checkpoint dir in here",)
    parser.add_argument("--init_model_dir", default="", type=str, help="Initial model dir, will use this if no checkpoint is found in training_dir",)

    parser.add_argument("--model_type", default="rdot_nll", type=str, help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--ann_dir", default="./data/MSMARCO/ann_data", type=str, help="The output directory where the ANN data will be written",)
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. \
                                                        Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum total input sequence length after tokenization. \
                                              Sequences longer than this will be truncated, sequences shorter will be padded.",)
    
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, help="The starting output file number",)
    parser.add_argument("--ann_chunk_factor", default=-1, type=int, help="devide training queries into chunks",)
    parser.add_argument("--topk_training", default=500, type=int, help="top k from which negative samples are collected",)
    parser.add_argument("--negative_sample", default=1, type=int, help="at each resample, how many negative samples per query do I use",)
    parser.add_argument("--ann_measure_topk_mrr", default=False, action="store_true", help="load scheduler from checkpoint or not",)
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank",)
    parser.add_argument("--inference", default=False, action="store_true", help="only do inference if specify",)
    parser.add_argument("--end_output_num", default=-1, type=int, help="Stop after this number of data versions has been generated, default run forever",)
    
    args = parser.parse_args()
    
    # ----------------------------------
    set_env(args)
    ann_data_gen(args)

if __name__ == "__main__":
    main()
