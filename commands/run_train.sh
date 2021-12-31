#!/bin/bash

# # Passage ANCE(FirstP) 
# gpu_no=4
# seq_length=512
# model_type=rdot_nll
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSPass512"
# pretrained_checkpoint_dir="warmup or trained checkpoint path"
# data_type=1
# warmup_steps=5000
# per_gpu_train_batch_size=8
# gradient_accumulation_steps=2
# learning_rate=1e-6

# # Document ANCE(FirstP) 
# gpu_no=4
# seq_length=512
# tokenizer_type="roberta-base"
# model_type=rdot_nll
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc512"
# pretrained_checkpoint_dir="warmup or trained checkpoint path"
# data_type=0
# warmup_steps=3000
# per_gpu_train_batch_size=8
# gradient_accumulation_steps=2
# learning_rate=5e-6

# # Document ANCE(MaxP) 
gpu_no=8
seq_length=2048
tokenizer_type="roberta-base"
model_type=rdot_nll_multi_chunk
base_data_dir="../data/raw_data/"
preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
job_name="OSDoc2048"
pretrained_checkpoint_dir="warmup or trained checkpoint path"
data_type=0
warmup_steps=500
per_gpu_train_batch_size=2
gradient_accumulation_steps=8
learning_rate=1e-5

##################################### Data Preprocessing ################################
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"

preprocess_cmd="\
python ../data/msmarco_data.py --data_dir $base_data_dir --out_data_dir $preprocessed_data_dir --model_type $model_type \
--model_name_or_path roberta-base --max_seq_length $seq_length --data_type $data_type\
"

echo $preprocess_cmd
eval $preprocess_cmd

if [[ $? = 0 ]]; then
    echo "successfully created preprocessed data"
else
	echo "preprocessing failed"
    echo "failure: $?"
    exit 1
fi

##################################### Inital ANN Data generation ################################
initial_data_gen_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --end_output_num 0 \
"

echo $initial_data_gen_cmd
eval $initial_data_gen_cmd

if [[ $? = 0 ]]; then
    echo "successfully created initial ann training data"
else
	echo "initial data generation failed"
    echo "failure: $?"
    exit 1
fi

############################################# Training ########################################
train_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann.py --model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco --triplet --data_dir $preprocessed_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$per_gpu_train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
--warmup_steps $warmup_steps --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup \
"

echo $train_cmd
eval $train_cmd