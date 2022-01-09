import sys
import csv
import collections
import gzip
import pickle
import argparse
import numpy as np
import faiss
import os
import pytrec_eval
import json
from tqdm import tqdm 
from utils.msmarco_eval import compute_metrics
from utils.util import convert_to_string_id

# query id [all_data_num, 1], passage id, positive id ({query_id: {passage_id:rel, ...}, ...}), retrieval topk id
def EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor, topN):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    total = 0
    labeled = 0 # 
    Atotal = 0
    Alabeled = 0 # how many results are incorrect
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): # [len(dev_query_embedding2id), topk], dev_query_embedding2id: [all_query, 1]
        seen_pid = set()
        query_id = dev_query_embedding2id[query_idx]
        prediction[query_id] = {}
        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN] # topN ann pid
        rank = 0
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp       
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx] # get passage index
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]: # not in positive id, it is incorrect
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1  # 
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)
    #-------------------------------------------------------------------------------------------------------------------
    # use out of the box evaluation script
    # to reuse pytrec_eval, id (keys) must be string
    evaluator = pytrec_eval.RelevanceEvaluator(convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})
    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction)) # positive id ({query_id: {passage_id:rel, ...}, ...})
    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0
    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall += result[k]["recall_"+str(topN)]
    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    #-------------------------------------------------------------------------------------------------------------------
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                if pid > 0:
                    qids_to_relevant_passageids[qid].append(pid) # 
            
    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
    #-------------------------------------------------------------------------------------------------------------------
    hole_rate = labeled/total
    Ahole_rate = Alabeled/Atotal # hole incorrect pid results (not be retrieved)
    #-------------------------------------------------------------------------------------------------------------------

    return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, result, prediction

def barrier_array_merge(args)
    dev_query_embedding = []
    dev_query_embedding2id = []
    passage_embedding = []
    passage_embedding2id = []
    for i in range(args.gpu_num): # it should change the num according to the number of gpus in the training stage. 
        try: # prefix="rank_data_obj_dev_query__emb_p_" or "rank_data_obj_passage__emb_p_" or "rank_data_obj_query__emb_p_"
            with open(args.emb_dir + "dev_query_"+str(args.step_num)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                dev_query_embedding.append(pickle.load(handle))
            with open(args.emb_dir + "dev_query_"+str(args.step_num)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                dev_query_embedding2id.append(pickle.load(handle))
            with open(args.emb_dir + "passage_"+str(args.step_num)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                passage_embedding.append(pickle.load(handle))
            with open(args.emb_dir + "passage_"+str(args.step_num)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                passage_embedding2id.append(pickle.load(handle))
        except:
            break
    if (not dev_query_embedding) or (not dev_query_embedding2id) or (not passage_embedding) or not (passage_embedding2id):
        print("No data found for checkpoint: ", args.step_num)

    dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
    dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)
    passage_embedding = np.concatenate(passage_embedding, axis=0)
    passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
    
    return dev_query_embedding, dev_query_embedding2id, passage_embedding, passage_embedding2id

def prepare_rerank_data(args, raw_data_dir)
    qidmap_path = args.processed_data_dir + "/qid2offset.pickle" # {real_query_id: query_id, ...}
    pidmap_path = args.processed_data_dir + "/pid2offset.pickle" # {real_passage_id: passage_id, ...}
    # qidmap and pidmap: {raw_id: dataset_id, ...}
    with open(qidmap_path, 'rb') as handle:
        qidmap = pickle.load(handle)
    with open(pidmap_path, 'rb') as handle:
        pidmap = pickle.load(handle)

    # query data and passage data from raw data
    if args.data_type == 0:
        if args.test_set == 1:
            query_path = os.path.join(raw_data_dir, "doc/msmarco-test2019-queries.tsv") # 2019qrels-docs.txt
            passage_path = os.path.join(raw_data_dir, "doc/msmarco-doctest2019-top100") # passage
        else:
            query_path = os.path.join(raw_data_dir, "doc/msmarco-docdev-queries.tsv")
            passage_path = os.path.join(raw_data_dir, "doc/msmarco-docdev-top100")
    else:
        if args.test_set == 1:
            query_path = os.path.join(raw_data_dir, "doc/msmarco-test2019-queries.tsv") # 2019qrels-docs.txt
            passage_path = os.path.join(raw_data_dir, "doc/msmarco-passagetest2019-top1000.tsv")
        else:
            query_path = os.path.join(raw_data_dir, "passage/queries.dev.small.tsv") # qrels.dev.small.tsv
            passage_path = os.path.join(raw_data_dir, "passage/top1000.dev")

    bm25 = collections.defaultdict(set) # [(key, value[set]), ...]
    # load query data and get query id set: practical id 
    qset = set()
    with gzip.open(query_path, 'rt', encoding='utf-8') if query_path[-2:] == "gz" else open(query_path, 'rt', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [qid, query] in tsvreader: # each query data
            qset.add(qid) # qset = {qid1, qid2, ...}
    # passage data (including qrel data)
    with gzip.open(passage_path, 'rt', encoding='utf-8') if passage_path[-2:] == "gz" else open(passage_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            if args.data_type == 0:
                [qid, Q0, pid, rank, score, runstring] = line.split(' ')
                pid = pid[1:] # passage id in practical
            else:
                [qid, pid, query, passage] = line.split("\t")
            
            if qid in qset and int(qid) in qidmap: # int(qid) in qidmap means in qidmap.keys()
                bm25[qidmap[int(qid)]].add(pidmap[int(pid)]) # [(qid_index, {passage_index}), ...]
    print("number of queries with " + str(topN) + " BM25 passages:", len(bm25))
    return bm25

def get_dev_query_pos_id(args, processed_data_dir):
    dev_query_positive_id = {}
    # str(qid2offset[topicid]) + "\t" + str(pid2offset[docid]) + "\t" + rel + "\n"
    query_positive_id_path = os.path.join(processed_data_dir, "dev-qrel.tsv")
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader: # [topicid, docid, rel] is processing unique index not practical id
            topicid = int(topicid) # query id
            docid = int(docid) # passage id
            if topicid not in dev_query_positive_id:
                dev_query_positive_id[topicid] = {} 
            dev_query_positive_id[topicid][docid] = int(rel) # {topici_id:{docid:rel, ...}, ...} # rel=1
    return dev_query_positive_id

def main():
    parser = argparse.ArgumentParser()
    ## required arguments
    parser.add_argument("--raw_data_dir", default="./data/MSMARCO/", type=str, help="The path of raw data dir",)
    parser.add_argument("--processed_data_dir", default="./data/MSMARCO/preprocessed", type=str, help="The path of preprocessed data dir",)
    parser.add_argument("--emb_dir", default="./data/MSMARCO/emb_data", type=str, help="Path of dumpped query and passage/document embeddings",)
    parser.add_argument("--step_num", default=1000000, type=int, help="Embedding from which checkpoint(ie: 200000)",)
    parser.add_argument("--gpu_num", default=4, type=int, help="The number of gpus in training stage.",)
    parser.add_argument("--data_type", default=1, type=int, help="0 for document, 1 for passage",)
    parser.add_argument("--test_set", default=0, type=int, help="0 for dev_set, 1 for eval_set",)
    args = parser.parse_args()

    topN = 100 if args.data_type == 0 else 1000

    dev_query_positive_id = get_dev_query_pos_id(args, args.processed_data_dir) # {topici_id:{docid:rel, ...}, ...} # rel=1
    bm25 = prepare_rerank_data(args, args.raw_data) # bm25:  [(qid_index, {passage_index}), ...]
    dev_query_embedding, dev_query_embedding2id, passage_embedding, passage_embedding2id = barrier_array_merge(args) # [all_data_num, embeddingS], # [all_data_num, 1]

    '''
        reranking metrics
    '''
    # dataset index -> rele pos (val)
    pidmap = collections.defaultdict(list)
    for i in range(len(passage_embedding2id)):
        pidmap[passage_embedding2id[i]].append(i)  # abs pos(key) to rele pos(val)

    if len(bm25) == 0:
        print("Rerank data set is empty. Check if your data prepration is done on the same data set. Rerank metrics is skipped.")
    else:
        rerank_data = {}
        all_dev_I = []
        for i, qid in enumerate(dev_query_embedding2id):
            p_set = []
            p_set_map = {} # {}
            if qid not in bm25: # qid is processing index of dataset, bm25:  [(qid_index, {passage_index}), ...]
                print(qid, "not in bm25")
            else:
                count = 0
                for k, pid in enumerate(bm25[qid]): # the qid with many relevant qid, both are index of dataset
                    if pid in pidmap: # pid in passage_embedding
                        for val in pidmap[pid]: # val is rel index
                            p_set.append(passage_embedding[val])
                            p_set_map[count] = val # new rele pos(key) to old rele pos(val)
                            count += 1
                    else:
                        print(pid, "not in passages")
            #-------------------------------------------------------------------------------------------------
            dim = passage_embedding.shape[1] # hidden dim
            faiss.omp_set_num_threads(16)
            cpu_index = faiss.IndexFlatIP(dim) # get the index for dense embedding
            p_set =  np.asarray(p_set)
            cpu_index.add(p_set)    
            _, dev_I = cpu_index.search(dev_query_embedding[i:i+1], len(p_set)) # topk=len(p_set)
            for j in range(len(dev_I[0])): # [tokp] or len(p_set)
                dev_I[0][j] = p_set_map[dev_I[0][j]]
            all_dev_I.append(dev_I[0])
        # pytrec_eval
        result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, all_dev_I, topN) # topN = 100 if args.data_type == 0 else 1000
        final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
        print("Reranking Results for checkpoint " + str(args.step_num))
        print("Reranking NDCG@10:" + str(final_ndcg))
        print("Reranking map@10:" + str(final_Map))
        print("Reranking pytrec_mrr:" + str(final_mrr))
        print("Reranking recall@"+str(topN)+":" + str(final_recall))
        print("Reranking hole rate@10:" + str(hole_rate))
        print("Reranking hole rate:" + str(Ahole_rate))
        print("Reranking ms_mrr:" + str(ms_mrr))

    '''
        full ranking metrics
    '''
    dim = passage_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding) 
    _, dev_I = cpu_index.search(dev_query_embedding, topN) # topN = 100 if args.data_type == 0 else 1000
    # pytrec_eval
    result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I, topN)
    final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
    print("Results for checkpoint " + str(args.step_num))
    print("NDCG@10:" + str(final_ndcg))
    print("map@10:" + str(final_Map))
    print("pytrec_mrr:" + str(final_mrr))
    print("recall@"+str(topN)+":" + str(final_recall))
    print("hole rate@10:" + str(hole_rate))
    print("hole rate:" + str(Ahole_rate))
    print("ms_mrr:" + str(ms_mrr))

if __name__ == "__main__":
    main()