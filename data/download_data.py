import sys
sys.path += ['./']
import argparse
import gzip
import tarfile
import logging
import os
import pathlib
import wget

from typing import Tuple

logger = logging.getLogger(__name__)

# TODO: move to hydra config group

NQ_LICENSE_FILES = [
    "https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE",
    "https://dl.fbaipublicfiles.com/dpr/nq_license/README",
]

# dict_map = {name: dict, ...}
MSMARCO_MAP = {
    # MSMARCO passage data
    "passage.collectionandqueries": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
        "original_ext": "",
        "compressed": True,
        "zip_format": ".tar.gz",
        "desc": "MSMARCO passage.collectionandqueries",
    },
    "passage.top1000.dev": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
        "original_ext": "",
        "compressed": True,
        "zip_format": ".tar.gz",
        "desc": "MSMARCO passage.top1000.dev",
    },
    "passage.triples.train.small": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz",
        "original_ext": "",
        "compressed": True,
        "zip_format": ".tar.gz",
        "desc": "MSMARCO passage.triples.train.small",
    },
    "passage.msmarco-passagetest2019-top1000": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "MSMARCO passage.msmarco-passagetest2019-top1000",
    },
    # MSMARCO doc data
    "doc.msmarco-docs": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "MSMARCO doc.msmarco-docs",
    },
    "doc.msmarco-doctrain-queries": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "MSMARCO doc.msmarco-doctrain-queries",
    },
    "doc.msmarco-doctrain-qrels": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "2019qrels_docs": "https://trec.nist.gov/data/deep/2019qrels-docs.txt",
        "desc": "MSMARCO doc.msmarco-doctrain-qrels"
    },
    "doc.msmarco-test2019-queries": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "MSMARCO doc.msmarco-test2019-queries",
    },
    "doc.msmarco-docdev-queries": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "MSMARCO doc.msmarco-docdev-queries",
    },
    "doc.msmarco-doctest2019-top100": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "MSMARCO doc.msmarco-doctest2019-top100",
    },
    "doc.msmarco-docdev-top100": {
        "s3_url": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz",
        "original_ext": "",
        "compressed": True,
        "desc": "MSMARCO docmsmarco-docdev-top100",
    },
}


NQ_TQA_MAP = {
    "data.wikipedia_split.psgs_w100": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
    "data.retriever.nq-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "NQ dev subset with passages pools for the Retriever train time validation",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.nq-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "NQ train subset with passages pools for the Retriever training",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.nq-adv-hn-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-adv-hn-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "NQ train subset with hard negative passages mined using the baseline DPR NQ encoders & wikipedia index",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.trivia-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.trivia-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 train subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.webq-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "WebQuestions dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.webq-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "WebQuestions dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.qas.nq-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "NQ dev subset for Retriever validation and IR results generation",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.qas.nq-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "NQ test subset for Retriever validation and IR results generation",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.qas.nq-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "NQ train subset for Retriever validation and IR results generation",
        "license_files": NQ_LICENSE_FILES,
    },
    #
    "data.retriever.qas.trivia-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz",
        "original_ext": ".csv",
        "compressed": True,
        "desc": "Trivia dev subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.trivia-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz",
        "original_ext": ".csv",
        "compressed": True,
        "desc": "Trivia test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.trivia-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz",
        "original_ext": ".csv",
        "compressed": True,
        "desc": "Trivia train subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.squad1-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/squad1-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "Trivia test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.webq-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "WebQuestions test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.curatedtrec-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "CuratedTrec test subset for Retriever validation and IR results generation",
    },
    "data.gold_passages_info.nq_train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our train subset) gold positive passages and alternative question tokenization",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.gold_passages_info.nq_dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our dev subset) gold positive passages and alternative question tokenization",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.gold_passages_info.nq_test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our test, original dev subset) gold positive passages and alternative question "
        "tokenization",
        "license_files": NQ_LICENSE_FILES,
    },
    "pretrained.fairseq.roberta-base.dict": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/dict.txt",
        "original_ext": ".txt",
        "compressed": False,
        "desc": "Dictionary for pretrained fairseq roberta model",
    },
    "pretrained.fairseq.roberta-base.model": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/model.pt",
        "original_ext": ".pt",
        "compressed": False,
        "desc": "Weights for pretrained fairseq roberta base model",
    },
    "pretrained.pytext.bert-base.model": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/pretrained/pytext/bert/bert-base-uncased.pt",
        "original_ext": ".pt",
        "compressed": False,
        "desc": "Weights for pretrained pytext bert base model",
    },
    "data.retriever_results.nq.single.wikipedia_passages": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq/wiki_passages_{}".format(i)
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Encoded wikipedia files using a biencoder checkpoint("
        "checkpoint.retriever.single.nq.bert-base-encoder) trained on NQ dataset ",
    },
    "data.retriever_results.nq.single-adv-hn.wikipedia_passages": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single-adv-hn/nq/wiki_passages_{}".format(i)
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Encoded wikipedia files using a biencoder checkpoint("
        "checkpoint.retriever.single-adv-hn.nq.bert-base-encoder) trained on NQ dataset + adversarial hard negatives",
    },
    "data.retriever_results.nq.single.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ test dataset for the encoder trained on NQ",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single.dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ dev dataset for the encoder trained on NQ",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single.train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ train dataset for the encoder trained on NQ",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single-adv-hn.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single-adv-hn/nq-test.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ test dataset for the encoder trained on NQ + adversarial hard negatives",
        "license_files": NQ_LICENSE_FILES,
    },
    "checkpoint.retriever.single.nq.bert-base-encoder": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on NQ data and HF bert-base-uncased model",
    },
    "checkpoint.retriever.multiset.bert-base-encoder": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on multi set data and HF bert-base-uncased model",
    },
    "checkpoint.retriever.single-adv-hn.nq.bert-base-encoder": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/single-adv-hn/nq/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on the original DPR NQ data combined with adversarial hard negatives (See data.retriever.nq-adv-hn-train resource). "
        "The model is HF bert-base-uncased",
    },
    "data.reader.nq.single.train": {
        "s3_url": ["https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/train.{}.pkl".format(i) for i in range(8)],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ train dataset input data preprocessed from retriever results (also trained on NQ)",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.reader.nq.single.dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/dev.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ dev dataset input data preprocessed from retriever results (also trained on NQ)",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.reader.nq.single.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/test.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ test dataset input data preprocessed from retriever results (also trained on NQ)",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.reader.trivia.multi-hybrid.train": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/train.{}.pkl".format(i)
            for i in range(8)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia train dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "data.reader.trivia.multi-hybrid.dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/dev.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia dev dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "data.reader.trivia.multi-hybrid.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/test.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia test dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "checkpoint.reader.nq-single.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on NQ-single retriever results and HF bert-base-uncased model",
    },
    "checkpoint.reader.nq-trivia-hybrid.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-trivia-hybrid/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on Trivia multi hybrid retriever results and HF bert-base-uncased model",
    },
    # extra checkpoints for EfficientQA competition
    "checkpoint.reader.nq-single-subset.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single-seen_only/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on NQ-single retriever results and HF bert-base-uncased model, when only Wikipedia pages seen during training are considered",
    },
    "checkpoint.reader.nq-tfidf.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-drqa/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on TFIDF results and HF bert-base-uncased model",
    },
    "checkpoint.reader.nq-tfidf-subset.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-drqa-seen_only/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on TFIDF results and HF bert-base-uncased model, when only Wikipedia pages seen during training are considered",
    },
    # retrieval indexes
    "indexes.single.nq.full.index": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever",
    },
    "indexes.single.nq.full.index_meta": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index_meta.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever (metadata)",
    },
    "indexes.single.nq.subset.index": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/seen_only.index.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered",
    },
    "indexes.single.nq.subset.index_meta": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/seen_only.index_meta.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered (metadata)",
    },
    "indexes.tfidf.nq.full": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/drqa/nq/full-tfidf.npz",
        "original_ext": ".npz",
        "compressed": False,
        "desc": "TFIDF index",
    },
    "indexes.tfidf.nq.subset": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/drqa/nq/seen_only-tfidf.npz",
        "original_ext": ".npz",
        "compressed": False,
        "desc": "TFIDF index when only Wikipedia pages seen during training are considered",
    },
}

def unpack(gzip_file: str, out_file: str, zip_format: str = ".gz"):
    logger.info("Uncompressing %s", gzip_file)
    if zip_format == ".gz":
        input = gzip.GzipFile(gzip_file, "rb")
        s = input.read()
        input.close()
        output = open(out_file, "wb")
        output.write(s)
        output.close()
    elif zip_format == ".tar.gz":
        t = tarfile.open(gzip_file)
        out_file = os.path.dirname(out_file)
        t.extractall(path = out_file)   
    logger.info(" Saved to %s", out_file)


def download_resource(
    data_name: str, s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str, zip_format: str
    ) -> Tuple[str, str]:
    logger.info("Requested resource from %s", s3_url)
    
    if zip_format=='.gz':
        path_names = resource_key.split(".")  
    else:
        doc_pass = resource_key.split(".")[0] # doc or passage
        file_name = ".".join(resource_key.split(".")[1:])
        path_names = [doc_pass, file_name]

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    logger.info("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, data_name, *path_names[:-1])  # last segment is for file name
    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] + (".tmp" if compressed else original_ext)))

    wget.download(s3_url, out=local_file)
    logger.info("Downloaded to %s", local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file, zip_format)
        os.remove(local_file)
        local_file = uncompressed_file
    
    return save_root, local_file


def download_file(s3_url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(s3_url, out=local_file)
    logger.info("Downloaded to %s", local_file)


def download(data_name: str, resource_map: dict, resource_key: str, out_dir: str = None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) 
    if resource_key not in resource_map:
        # match by prefix
        resources = [k for k in resource_map.keys() if k.startswith(resource_key)]
        print("Match by prefix resources: ", resources)
        if resources:
            for key in resources:
                download(data_name, resource_map, key, out_dir)
        else:
            logger.info("no resources found for specified key")
            raise ValueError("Error: No resources found for specified key!")
    else:
        download_info = resource_map[resource_key]
        s3_url = download_info["s3_url"]
        zip_format = ".gz" if not download_info.get("zip_format", None) else ".tar.gz"

        # download datasets
        save_root_dir = None
        data_files = []
        if isinstance(s3_url, list):
            for i, url in enumerate(s3_url):
                save_root_dir, local_file = download_resource(
                    data_name,
                    url, 
                    download_info["original_ext"],
                    download_info["compressed"],
                    "{}_{}".format(resource_key, i),
                    out_dir,
                    zip_format,
                )
                data_files.append(local_file)
        else:
            save_root_dir, local_file = download_resource(
                data_name, 
                s3_url,
                download_info["original_ext"],
                download_info["compressed"],
                resource_key,
                out_dir,
                zip_format,
            )
            data_files.append(local_file)

        # download LICENSE and README files
        license_files = download_info.get("license_files", None)
        if license_files:
            download_file(license_files[0], save_root_dir, "LICENSE")
            download_file(license_files[1], save_root_dir, "README")
        
        qrels_docs = download_info.get("2019qrels_docs", None)
        if qrels_docs:
            download_file(qrels_docs, save_root_dir, "2019qrels-docs.txt")

        return data_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default=["MSMARCO"], type=list, help="The list of dataset name")
    parser.add_argument("--output_dir", default="./data", type=str, help="The output directory to download file")
    args = parser.parse_args()
    
    for data_name in args.data_name:
        if data_name == "MSMARCO":
            RESOURCE_MAP = MSMARCO_MAP
            logger.info("data name %s", data_name)
            resource = list(MSMARCO_MAP.keys())
            logger.info("resource name %s", *resource)
            print("Download resources: ", resource)
            for resource_key in resource:
                download(data_name, RESOURCE_MAP, resource_key, args.output_dir)
        elif data_name == "NQ_TQA":
            RESOURCE_MAP = NQ_TQA_MAP
            logger.info("data name %s", data_name)
            resource = ["data.wikipedia_split.psgs_w100", "data.retriever.nq", "data.retriever.trivia", \
                        "data.retriever.qas.nq", "data.retriever.qas.trivia", \
                        "checkpoint.retriever.multiset.bert-base-encoder"]
            logger.info("resource name %s", *resource)
            print("Download resources: ", resource)
            for resource_key in resource:
                download(data_name, RESOURCE_MAP, resource_key, args.output_dir)
        else:
            logger.info("no dataset support for %s", data_name)
            raise NotImplementedError("Error: No dataset support for data name!")

if __name__ == "__main__":
    main()
