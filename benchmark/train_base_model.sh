#!/bin/bash
dataset="student"
model="mlp"

dir="./out/train_base_model/tabular/$dataset/$model"
if [ ! -d "$dir" ];then
mkdir -p $dir
fi
nohup python -u train_base_model.py >$dir/split.output    --gpu 0 --dataset $dataset --model $model  --model_ids -1  2>&1
nohup python -u train_base_model.py >$dir/train0.output   --gpu 0 --dataset $dataset --model $model  --model_ids 0   2>&1 &
nohup python -u train_base_model.py >$dir/train1.output   --gpu 0 --dataset $dataset --model $model  --model_ids 1   2>&1 &
nohup python -u train_base_model.py >$dir/train2.output   --gpu 0 --dataset $dataset --model $model  --model_ids 2   2>&1 &
nohup python -u train_base_model.py >$dir/train3.output   --gpu 0 --dataset $dataset --model $model  --model_ids 3   2>&1 &
nohup python -u train_base_model.py >$dir/train4.output   --gpu 1 --dataset $dataset --model $model  --model_ids 4   2>&1 &
nohup python -u train_base_model.py >$dir/train5.output   --gpu 1 --dataset $dataset --model $model  --model_ids 5   2>&1 &
nohup python -u train_base_model.py >$dir/train6.output   --gpu 1 --dataset $dataset --model $model  --model_ids 6   2>&1 &
nohup python -u train_base_model.py >$dir/train7.output   --gpu 1 --dataset $dataset --model $model  --model_ids 7   2>&1 &
nohup python -u train_base_model.py >$dir/train8.output   --gpu 2 --dataset $dataset --model $model  --model_ids 8   2>&1 &
nohup python -u train_base_model.py >$dir/train9.output   --gpu 2 --dataset $dataset --model $model  --model_ids 9   2>&1 &
nohup python -u train_base_model.py >$dir/train10.output  --gpu 2 --dataset $dataset --model $model  --model_ids 10  2>&1 &
nohup python -u train_base_model.py >$dir/train11.output  --gpu 2 --dataset $dataset --model $model  --model_ids 11  2>&1 &
nohup python -u train_base_model.py >$dir/train12.output  --gpu 3 --dataset $dataset --model $model  --model_ids 12  2>&1 &
nohup python -u train_base_model.py >$dir/train13.output  --gpu 3 --dataset $dataset --model $model  --model_ids 13  2>&1 &
nohup python -u train_base_model.py >$dir/train14.output  --gpu 3 --dataset $dataset --model $model  --model_ids 14  2>&1 &
nohup python -u train_base_model.py >$dir/train15.output  --gpu 3 --dataset $dataset --model $model  --model_ids 15  2>&1 &
