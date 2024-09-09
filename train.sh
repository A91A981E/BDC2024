#/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

model_name=iTransformer_iFlashFormer

seeds=(3042 3407)   # 3042 3407
tasks=("both" "temp" "wind")   # temp wind both

for seed in ${seeds[@]};
do
  for task in ${tasks[@]};
  do
    python -u my_code/train.py \
      --root_path /home/mw/input/bdc_train9198/global/global \
      --seed $seed \
      --model $model_name \
      --seq_len 168 \
      --pred_len 72 \
      --e_layers 1 \
      --enc_in 56 \
      --d_model 128 \
      --d_ff 64 \
      --n_heads 8 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --batch_size 10240 \
      --train_epochs 3 \
      --k_fold 3 \
      --n_experts 2 \
      --print_freq 50 \
      --num_workers 9 \
      --use_multi_gpu \
      --task $task \
      --lradj type1
  done
done

# for seed in ${seeds[@]};
# do
#   for task in ${tasks[@]};
#   do
#     python -u my_code/train.py \
#       --root_path /BDC_data \
#       --seed $seed \
#       --model $model_name \
#       --seq_len 168 \
#       --pred_len 72 \
#       --e_layers 1 \
#       --enc_in 56 \
#       --d_model 128 \
#       --d_ff 64 \
#       --n_heads 8 \
#       --dropout 0.1 \
#       --learning_rate 0.001 \
#       --batch_size 10240 \
#       --train_epochs 3 \
#       --k_fold 3 \
#       --n_experts 2 \
#       --print_freq 500 \
#       --num_workers 18 \
#       --use_multi_gpu \
#       --task $task \
#       --clean_data \
#       --lradj type1
#   done
# done