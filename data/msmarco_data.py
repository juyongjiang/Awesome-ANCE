import sys
sys.path += ['./']
import os
import gzip
import csv
import pickle
import argparse
import json
import glob
import numpy as np
from model.models import MSMarcoConfigDict, ALL_MODELS
from multiprocessing import Process

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

# process each line from file
# transfer each line to "p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + content=np.array(input_id_b, np.int32).tobytes()"
def PassagePreprocessingFn(args, line, tokenizer):
    if args.data_type == 0:
        line_arr = line.split('\t')
        # doc
        p_id = int(line_arr[0][1:])  # remove "D"
        url = line_arr[1].rstrip()
        title = line_arr[2].rstrip()
        p_text = line_arr[3].rstrip()

        #full_text = url + "<sep>" + title + "<sep>" + p_text
        full_text = url + " "+tokenizer.sep_token+" " + title + " "+tokenizer.sep_token+" " + p_text
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = full_text[:args.max_doc_character]
    else:
        # passage
        line = line.strip()
        line_arr = line.split('\t')
        p_id = int(line_arr[0])

        p_text = line_arr[1].rstrip()

        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = p_text[:args.max_doc_character]
    # tokenizer.encode: using vocab.txt from BERT change token to dict_id, and add 101=[cls] and 102=[sep] in the before and after passage
    passage = tokenizer.encode(full_text, add_special_tokens=True, max_length=args.max_seq_length,) # return token id
    passage_len = min(len(passage), args.max_seq_length)
    # expand passage with max length by using tokenizer.pad_token_id
    input_id_b = pad_input_ids(passage, args.max_seq_length, pad_token=tokenizer.pad_token_id) # keep the same seq length by padding

    return p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()

# process each line from file
def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    # query
    q_id = int(line_arr[0])
    q_text = line_arr[1].rstrip()

    passage = tokenizer.encode(q_text, add_special_tokens=True, max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    # expand passage with max length by using tokenizer.pad_token_id
    input_id_b = pad_input_ids(passage, args.max_query_length, pad_token=tokenizer.pad_token_id)

    return q_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()

def write_passage_doc(args, in_passage_path, out_passage_path, PassagePreprocessingFn)
    '''
        passage: out_passage_path
        passage_len.to_bytes(4, 'big') + content=np.array(input_id_b, np.int32).tobytes()
    '''
    print('start passage file split processing') 
    multi_file_process(args, 32, in_passage_path, out_passage_path, PassagePreprocessingFn) # 32 is the number of processing

    print('start merging splits')
    # read each record by bytes then use int.from_bytes to recover integer number
    pid2offset = {}
    out_line_count = 0
    with open(out_passage_path, 'wb') as f:
        # transfer each line to "p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + content=np.array(input_id_b, np.int32).tobytes()"
        for idx, record in enumerate(numbered_byte_file_generator(out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
            p_id = int.from_bytes(record[:8], 'big') # p_id: 8 bytes encoder
            f.write(record[8:]) # saved by bytes
            pid2offset[p_id] = idx
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1
    print("Total lines written: " + str(out_line_count))
    
    # data proprecessig meta info
    meta = {'type': 'int32', 'total_number': out_line_count, 'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)

    # data pid2offset info
    pid2offset_path = os.path.join(args.out_data_dir, "pid2offset.pickle",)
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4) # save dictionary in pickle, {p_id:idx} p_id is the id of document, idx is the index
    print("done saving pid2offset")

    return pid2offset

def write_query_rel(args, pid2offset, query_file, positive_id_file, out_query_file, out_id_file, QueryPreprocessingFn):
    print("Writing query files " + str(out_query_file) + " and " + str(out_id_file))
    query_positive_id = set()
    query_positive_id_path = os.path.join(args.data_dir, positive_id_file,)
    print("Loading query_to_positive_doc_id")
    with gzip.open(query_positive_id_path, 'rt', encoding='utf8') if positive_id_file[-2:] == "gz" else open(query_positive_id_path, 'r', encoding='utf8') as f:
        if args.data_type == 0:
            tsvreader = csv.reader(f, delimiter=" ")
        else:
            tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            query_positive_id.add(int(topicid))
    '''
        query: out_query_file
        passage_len.to_bytes(4, 'big') + np.array(input_id_b, np.int32).tobytes()
    '''
    query_collection_path = os.path.join(args.data_dir, query_file,) 
    out_query_path = os.path.join(args.out_data_dir, out_query_file,)

    qid2offset = {}
    print('start query file split processing')
    multi_file_process(args, 32, query_collection_path, out_query_path, QueryPreprocessingFn)
    
    print('start merging splits')
    idx = 0
    with open(out_query_path, 'wb') as f:
        # transfer each line to "p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + content=np.array(input_id_b, np.int32).tobytes()"
        for record in numbered_byte_file_generator(out_query_path, 32, 8 + 4 + args.max_query_length * 4):
            q_id = int.from_bytes(record[:8], 'big')
            ####
            if q_id not in query_positive_id: # query_positive_id is a set 
                # exclude the query as it is not in label set
                continue
            ####
            # ******************exclude q_id, only save q_len, q_content************
            f.write(record[8:]) 
            qid2offset[q_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(q_id))
    
    # qid2offset info
    qid2offset_path = os.path.join(args.out_data_dir, "qid2offset.pickle",)
    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)
    print("done saving qid2offset")
    
    # query info
    print("Total lines written: " + str(idx))
    meta = {'type': 'int32', 'total_number': idx, 'embedding_size': args.max_query_length}
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)
    
    '''
        positive id (relevant): out_id_file
        str(qid2offset[topicid]) + "\t" + str(pid2offset[docid]) + "\t" + rel + "\n"
    '''
    out_id_path = os.path.join(args.out_data_dir, out_id_file,)
    print("Writing qrels")
    # write down: str(qid2offset[topicid]) + "\t" + str(pid2offset[docid]) + "\t" + rel + "\n"
    with gzip.open(query_positive_id_path, 'rt', encoding='utf8') if positive_id_file[-2:] == "gz" else open(query_positive_id_path, 'r', encoding='utf8') as f, \
            open(out_id_path, "w", encoding='utf-8') as out_id:
        if args.data_type == 0:
            tsvreader = csv.reader(f, delimiter=" ")
        else:
            tsvreader = csv.reader(f, delimiter="\t")
        out_line_count = 0

        for [topicid, _, docid, rel] in tsvreader:
            topicid = int(topicid)
            if args.data_type == 0:
                docid = int(docid[1:])
            else:
                docid = int(docid)
            # topicid == query id, docid == passage id in raw data
            out_id.write(str(qid2offset[topicid]) + "\t" + str(pid2offset[docid]) + "\t" + rel + "\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))

def preprocess(args):
    args.data_dir = os.path.join(args.data_dir, "doc") if args.data_type == 0 else os.path.join(args.data_dir, "passage")
    args.out_data_dir = args.out_data_dir + "_{}_{}_{}".format(args.model_name_or_path, args.max_seq_length, args.data_type)
    
    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    '''
        passage
    '''
    # input dataset path
    if args.data_type == 0:
        in_passage_path = os.path.join(args.data_dir, "msmarco-docs.tsv",) # MSMARCO/doc
    else: 
        in_passage_path = os.path.join(args.data_dir, "collection.tsv",) # MSMARCO/passage
    # output dataset path
    out_passage_path = os.path.join(args.out_data_dir, "passages",) # raw_data/ann_data_tokenizer_seqlen/passages
    if os.path.exists(out_passage_path): # out_passage_path is file not dir
        print("preprocessed data already exist, exit preprocessing")
        return

    pid2offset = write_passage_doc(args, in_passage_path, out_passage_path, PassagePreprocessingFn)
    
    '''
        query
    '''
    # start processing
    # pid2offset, query_file, positive_id_file, out_query_file, out_id_file
    if args.data_type == 0:
        write_query_rel(args, pid2offset, "msmarco-doctrain-queries.tsv", "msmarco-doctrain-qrels.tsv", "train-query", "train-qrel.tsv", QueryPreprocessingFn)
        write_query_rel(args, pid2offset, "msmarco-test2019-queries.tsv", "2019qrels-docs.txt", "dev-query", "dev-qrel.tsv", QueryPreprocessingFn)
    else:
        # train-qrel.tsv saves "query index and relevant passage index"
        write_query_rel(args, pid2offset, "queries.train.tsv", "qrels.train.tsv", "train-query", "train-qrel.tsv", QueryPreprocessingFn)
        write_query_rel(args, pid2offset, "queries.dev.small.tsv", "qrels.dev.small.tsv", "dev-query", "dev-qrel.tsv", QueryPreprocessingFn)

    '''
        remove *_split* files
    '''
    for split_file in glob.glob(os.path.join(args.out_data_dir, '*_split*')):
        print("remove %s" % split_file)
        os.remove(split_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/MSMARCO", type=str, help="The input data dir",)
    parser.add_argument("--out_data_dir", default="./data/MSMARCO/preprocessed", type=str, help="The output data dir",)
    parser.add_argument("--model_type", default="rdot_nll", type=str, help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="Path to pre-trained model or shortcut name selected in the list: " +", ".join(ALL_MODELS),)
    parser.add_argument("--do_lower_case", default=False, help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--max_seq_length", default=2048, type=int, help="The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum total input sequence length after tokenization. Sequences longer ""than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_doc_character", default=10000, type=int, help="used before tokenizer to save tokenizer latency",)
    parser.add_argument("--data_type", default=1, type=int, help="0 for doc, 1 for passage",)
    args = parser.parse_args()

    preprocess(args)


if __name__ == '__main__':
    main()