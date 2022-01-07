import sys
import os
import torch
from data.msmarco_data import GetTrainingDataProcessingFn, GetTripletTrainingDataProcessingFn
from utils.util import (
    getattr_recursive,
    set_seed,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data,
    is_first_worker
)
import pandas as pd
from transformers import glue_processors as processors
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
import transformers
from utils.lamb import Lamb
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import glob
import json
import logging
import random
torch.multiprocessing.set_sharing_strategy('file_system')
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)


def train(args, model, tokenizer, query_cache, passage_cache):
    """ Train the model """
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    # get the whole batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # get optimizer parameters and set optimizer
    optimizer_grouped_parameters = []
    layer_optim_params = set()
    for layer_name in ["roberta.embeddings", "score_out", "downsample1", "downsample2", "downsample3"]:
        layer = getattr_recursive(model, layer_name)
        if layer is not None:
            optimizer_grouped_parameters.append({"params": layer.parameters()}) # [{"params": layer.parameters()}, ...]
            for p in layer.parameters():
                layer_optim_params.add(p)
    if getattr_recursive(model, "roberta.encoder.layer") is not None:
        for layer in model.roberta.encoder.layer:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    optimizer_grouped_parameters.append({"params": [p for p in model.parameters() if p not in layer_optim_params]})

    if args.optimizer.lower() == "lamb":
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and args.load_optimizer_scheduler:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))

    # Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit to speed up model training and memory cost
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level) # O0: fp32; O1: mix; O2: fp16, ex bn; O3: fp16

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True,
        )

    # initialize model training
    global_step = 0

    tr_loss = 0.0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility

    last_ann_no = -1
    train_dataloader = None
    train_dataloader_iter = None
    dev_ndcg = 0
    step = 0
    
    #======================================================================================================================================
    if args.single_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    '''
        start model training with max steps
    '''
    while global_step < args.max_steps:
        if step % args.gradient_accumulation_steps == 0 and global_step % args.logging_steps == 0:
            # check if new ann training data is availabe
            ann_no, ann_path, ndcg_json = get_latest_ann_data(args.ann_dir)
            if ann_path is not None and ann_no != last_ann_no:
                logger.info("Training on new add data at %s", ann_path)
                with open(ann_path, 'r') as f:
                    ann_training_data = f.readlines()
                dev_ndcg = ndcg_json['ndcg']
                ann_checkpoint_path = ndcg_json['checkpoint']
                ann_checkpoint_no = get_checkpoint_no(ann_checkpoint_path)

                aligned_size = (len(ann_training_data) // args.world_size) * args.world_size
                ann_training_data = ann_training_data[:aligned_size]

                logger.info("Total ann queries: %d", len(ann_training_data))
                if args.triplet:
                    train_dataset = StreamingDataset(ann_training_data, GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache))
                else:
                    train_dataset = StreamingDataset(ann_training_data, GetTrainingDataProcessingFn(args, query_cache, passage_cache))
                train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
                train_dataloader_iter = iter(train_dataloader)

                # re-warmup
                if not args.single_warmup:
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(ann_training_data))

                if args.local_rank != -1:
                    dist.barrier()

                if is_first_worker():
                    # add ndcg at checkpoint step used instead of current step
                    tb_writer.add_scalar("dev_ndcg", dev_ndcg, ann_checkpoint_no)
                    if last_ann_no != -1:
                        tb_writer.add_scalar("epoch", last_ann_no, global_step - 1)
                    tb_writer.add_scalar("epoch", ann_no, global_step)
                last_ann_no = ann_no

        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)

        batch = tuple(t.to(args.device) for t in batch)
        step += 1

        # 
        if args.triplet:
            inputs = {
                "query_ids": batch[0].long(),
                "attention_mask_q": batch[1].long(),

                "input_ids_a": batch[3].long(),
                "attention_mask_a": batch[4].long(),

                "input_ids_b": batch[6].long(),
                "attention_mask_b": batch[7].long()}
        else:
            inputs = {
                "input_ids_a": batch[0].long(),
                "attention_mask_a": batch[1].long(),

                "input_ids_b": batch[3].long(),
                "attention_mask_b": batch[4].long(),

                "labels": batch[6]}

        # sync gradients only at gradient accumulation step
        if step % args.gradient_accumulation_steps == 0:
            outputs = model(**inputs)
        else:
            with model.no_sync():
                outputs = model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = outputs[0]

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if step % args.gradient_accumulation_steps == 0:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()

        tr_loss += loss.item()
        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                loss_scalar = tr_loss / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                tr_loss = 0

                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

            if is_first_worker() and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
            
            dist.barrier()
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                if args.evaluate_during_training and global_step % (args.logging_steps_per_eval*args.logging_steps) == 0:
                    model.eval()
                    reranking_mrr, full_ranking_mrr = passage_dist_eval(args, model, tokenizer)
                    if is_first_worker():
                        print("Reranking/Full ranking mrr: {0}/{1}".format(str(reranking_mrr), str(full_ranking_mrr)))
                        mrr_dict = {"reranking": float(reranking_mrr), "full_raking": float(full_ranking_mrr)}
                        tb_writer.add_scalars("mrr", mrr_dict, global_step)
                        print(args.output_dir)

                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                if is_first_worker():
                    for key, value in logs.items():
                        print(key, type(value))
                        tb_writer.add_scalar(key, value, global_step)
                    tb_writer.add_scalar("epoch", m_epoch, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))
                dist.barrier()
    #======================================================================================================================================
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step


def save_checkpoint(args, model, tokenizer):
    # Saving best-practices: if you use defaults names for the model, you can
    # reload it using from_pretrained()
    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained
        # model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.local_rank != -1:
        dist.barrier()

def load_model(args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower() # MSMarco
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    # model configuration followed by huggingface to load pretrained model and tokenizer
    args.model_type = args.model_type.lower() # rdot_nll (FirstP) or rdot_nll_multi_chunk (MaxP)
    configObj = MSMarcoConfigDict[args.model_type]
    config = configObj.config_class.from_pretrained(args.model_name_or_path,
                                                    num_labels=num_labels,
                                                    finetuning_task=args.task_name,
                                                    cache_dir=None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(args.model_name_or_path,
                                                          do_lower_case=args.do_lower_case, # whether transfer words into lower case
                                                          cache_dir=None,
    )
    model = configObj.model_class.from_pretrained(args.model_name_or_path,
                                                  from_tf=bool(".ckpt" in args.model_name_or_path), 
                                                  config=config,
                                                  cache_dir=None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return tokenizer, model

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

    # Set seed
    set_seed(args)

def evaluation(args, model, tokenizer):
    # Evaluation
    results = {}
    if args.do_eval:
        model_dir = args.model_name_or_path if args.model_name_or_path else args.output_dir
        checkpoints = [model_dir]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model.eval()
            reranking_mrr, full_ranking_mrr = passage_dist_eval(args, model, tokenizer)
            if is_first_worker():
                print("Reranking/Full ranking mrr: {0}/{1}".format(str(reranking_mrr), str(full_ranking_mrr)))
            dist.barrier()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization",)
    parser.add_argument("--task_name", default="MSMarco", type=str, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),)
    # Required parameters
    parser.add_argument("--data_dir", default="./data/MSMARCO/preprocessed", type=str, help="The input preprocessed data dir. Should contain the cached passage and query files",)
    parser.add_argument("--ann_dir", default="./data/ann_data", type=str, help="The ann training data dir. Should contain the output of ann data generation job",)
    parser.add_argument("--model_type", default="rdot_nll", type=str, help="rdot_nll (FirstP) or rdot_nll_multi_chunk (MaxP)",)
    parser.add_argument("--output_dir", default="saved", type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    # pretrained model
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    # training setting
    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.",)
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. \
                                            Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum total input sequence length after tokenization. \
                                            Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.",)
    parser.add_argument("--max_steps", default=1000000, type=int, help="If > 0: set total number of training steps to perform",)
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.",)
    # optimizer
    parser.add_argument("--optimizer", default="lamb", type=str, help="Optimizer - lamb or adamW",)
    parser.add_argument("--load_optimizer_scheduler", default=False, action="store_true", help="load scheduler from checkpoint or not",)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",)
    parser.add_argument("--single_warmup", default=False, action="store_true", help="use single or re-warmup",)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.",)
    # log setting
    parser.add_argument("--log_dir", default=None, type=str, help="Tensorboard log dir",)
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.",)
    # mixed precision
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', \
                                                                          and 'O3']. See details at https://nvidia.github.io/apex/amp.html",)
    # dis setting
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank",)

    args = parser.parse_args()
    
    # -----------------------------------------------------
    set_env(args) # set run env and reproducible seed
    tokenizer, model = load_model(args)
    # query 
    query_collection_path = os.path.join(args.data_dir, "train-query")
    query_cache = EmbeddingCache(query_collection_path)
    # passages
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)

    with query_cache, passage_cache:
        global_step = train(args, model, tokenizer, query_cache, passage_cache)

    save_checkpoint(args, model, tokenizer)
    results = evaluation(args, model, tokenizer)
    print("Model Evaluation: ", results)

if __name__ == "__main__":
    main()