#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

forget_losses=(
################ GA
    GA1             ##CoT+Answer
    GA2             ##Answer-only
    GA3             ##CoT-only
################ GD
    GA1+GD          ##CoT+Answer                 
    GA2+GD          ##Answer-only
    GA3+GD          ##CoT-only
################ KL
    GA1+KL          ##CoT+Answer    
    GA2+KL          ##Answer-only
    GA3+KL          ##CoT-only
################ PO
    IDK1+GD         ##Direct IDK
    IDK2+GD         ##Answer IDK
    IDK3+GD         ##Reasoned IDK
)


task_list=(1)

learning_rates=(
    1e-5
    2e-6
    5e-6
)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=sangyon/LRM-target
mask=true

use_LoRA=false
save_root=results/rtofu

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false

save_steps=last
eval_steps=(last)

num_epochss=(1 2 3 4 5)
split=forget01
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval2.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
        done
    done
done

split=forget05
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval2.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
        done
    done
done

split=forget10
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval2.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
        done
    done
done
