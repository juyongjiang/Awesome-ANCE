import numpy as np
import json
import os

if __name__ == '__main__':
    # base_path = os.path.join("./data/MSMARCO/preprocessed", "train-query")
    # with open(base_path + '_meta', 'r') as f:
    #     meta = json.load(f)
    #     dtype = np.dtype(meta['type']) # "int32"
    #     total_number = meta['total_number']
    #     # the size of single record: passage_len, passage, stored by bytes
    #     record_size = int(meta['embedding_size']) * 4 + 4 # dtype.itemsize = 4

    # with open(base_path, 'rb') as f:
    #     record_bytes = f.read(record_size) # read record_size bytes
    #     passage_len = int.from_bytes(record_bytes[:4], 'big')
    #     passage = np.frombuffer(record_bytes[4:], dtype=dtype)
    #     print(passage_len)
    #     print(list(passage))
    #     print(len(list(passage)))

    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # a = tokenizer("I have a new GPU!")
    # print(a)
    class Coordinate:
        x = 10
        y = -5
        z = 0
    
    point1 = Coordinate() 
    print(getattr(point1, 'x'))