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
        else:
            logger.info("no dataset support for %s", data_name)
            raise NotImplementedError("Error: No dataset support for data name!")

if __name__ == "__main__":
    main()
