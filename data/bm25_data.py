import sys
import os
import csv
import numpy as np
import argparse
import json
import random

def load_positive_ids(args):
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

    return training_query_positive_id

def GenerateNegativePassaageID(pos_passage_idx, all_passage_idx):
    # TODO BM25
    neg_passage_idx = random.choice(all_passage_idx) # random selection
    while neg_passage_idx == pos_passage_idx:
        neg_passage_idx = random.choice(all_passage_idx)

    return neg_passage_idx

# ann_dir/ann_training_data_[output_num]-(query_id, pos_pid, neg_pid)
# ann_dir/ann_ndcg_[output_num]-({ndcg: dev_ndcg, checkpoint: checkpoint_path})
# training_query_positive_id: {query_id: passage_id, ...}
def generate_bm25_ann(args):
    training_query_positive_id = load_positive_ids(args)
    output_num, dev_ndcg = 0, 0
    checkpoint_path = ""
    if not os.path.exists(args.ann_dir):
        os.makedirs(args.ann_dir)
    train_data_output_path = os.path.join(args.ann_dir, "ann_training_data_" + str(output_num))
    with open(train_data_output_path, 'w') as f:
        all_passage_idx = list(set(training_query_positive_id.values()))
        random.shuffle(all_passage_idx)
        for query_idx, passage_idx in training_query_positive_id.items():
            neg_passage_idx = GenerateNegativePassaageID(passage_idx, all_passage_idx)
            f.write("{}\t{}\t{}\n".format(query_idx, passage_idx, ','.join(str(neg_passage_idx))))
    # meta info
    ndcg_output_path = os.path.join(args.ann_dir, "ann_ndcg_" + str(output_num))
    with open(ndcg_output_path, 'w') as f:
        json.dump({'ndcg': dev_ndcg, 'checkpoint': checkpoint_path}, f) # checkpoint_path, saved/checkpint-[step_no]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default="./data/MSMARCO/", type=str, help="The raw data dir",)
    parser.add_argument("--data_dir", default="./data/MSMARCO/preprocessed", type=str, help="The preprocessed data dir",)
    parser.add_argument("--ann_dir", default="./data/MSMARCO/ann_data", type=str, help="The output directory where the ANN data will be written",)
    parser.add_argument("--topk_training", default=500, type=int, help="top k from which negative samples are collected",)
    parser.add_argument("--negative_sample", default=1, type=int, help="at each resample, how many negative samples per query do I use",)
    
    args = parser.parse_args()

    generate_bm25_ann(args)

if __name__ == "__main__":
    main()