#!/bin/bash
export ROOT_DIR=/mnt/Data2/nerf_datasets/tanks_and_temples

task_number=10
rep=$1

data=tat_intermediate_M60
#for ((i=0; i<$task_number; i++))
for ((i=0; i<3; i++))
do
    python train_CLNerfv2.py \
        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
        --exp_name $data'_'$task_number'task_'$rep'_v4' \
        --num_epochs 20 --scale 4.0 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number \
        --ray_sampling_strategy uncert_images
done
#
#data=tat_intermediate_Playground
#for ((i=0; i<$task_number; i++))
#do
#    python train_CLNerf.py \
#        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
#        --exp_name $data'_'$task_number'task_'$rep'_v3' \
#        --num_epochs 20 --scale 4.0 --rep_size $rep --eval_lpips \
#        --task_curr $i --task_number $task_number \
#        --ray_sampling_strategy prior_images
#done
#
#
#data=tat_intermediate_Train
#for ((i=0; i<$task_number; i++))
#do
#    python train_CLNerf.py \
#        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
#        --exp_name $data'_'$task_number'task_'$rep'_v3' \
#        --num_epochs 20 --scale 16.0 --batch_size 4096 --rep_size $rep --eval_lpips \
#        --task_curr $i --task_number $task_number \
#        --ray_sampling_strategy prior_images
#done

#data=tat_training_Truck
#for ((i=0; i<$task_number; i++))
#do
#    python train_CLNerf.py \
#        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
#        --exp_name $data'_'$task_number'task_'$rep'_v3' \
#        --num_epochs 20 --scale 16.0 --batch_size 4096 --rep_size $rep --eval_lpips \
#        --task_curr $i --task_number $task_number \
#        --ray_sampling_strategy prior_images
#done

