#!/bin/bash
export ROOT_DIR=/mnt/Data2/nerf_datasets/tanks_and_temples

task_number=10
rep=$1

#data=tat_intermediate_M60
#for ((i=9; i<$task_number; i++))
#do
#    if [ $i -gt 0 ]
#    then
#    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19-v'$i'.ckpt'
#    else
#    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19.ckpt'
#    fi
#    python train_CLNerf.py \
#        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
#        --exp_name $data'_'$task_number'task_'$rep \
#        --num_epochs 0 --scale 4.0 --rep_size $rep --eval_lpips \
#        --task_curr $i --task_number $task_number \
#        --val_only \
#        --weight_path $ckpt_file \
#        --save_depth_pcd --depth_clip 0.2
#done

#data=tat_intermediate_Playground
#for ((i=9; i<$task_number; i++))
#do
#  if [ $i -gt 0 ]
#    then
#    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19-v'$i'.ckpt'
#    else
#    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19.ckpt'
#    fi
#    python train_CLNerf.py \
#        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
#        --exp_name $data'_'$task_number'task_'$rep \
#        --num_epochs 0 --scale 4.0 --rep_size $rep --eval_lpips \
#        --task_curr $i --task_number $task_number \
#        --val_only \
#        --weight_path $ckpt_file \
#        --save_depth_pcd --depth_clip 0.8
#done
#
#
#data=tat_intermediate_Train
#for ((i=9; i<$task_number; i++))
#do
#  if [ $i -gt 0 ]
#    then
#    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19-v'$i'.ckpt'
#    else
#    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19.ckpt'
#    fi
#    python train_CLNerf.py \
#        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
#        --exp_name $data'_'$task_number'task_'$rep \
#        --num_epochs 0 --scale 16.0 --batch_size 4096 --rep_size $rep --eval_lpips \
#        --task_curr $i --task_number $task_number \
#        --val_only \
#        --weight_path $ckpt_file \
#        --save_depth_pcd --depth_clip 0.2
#done
data=tat_training_Truck
for ((i=0; i<$task_number; i++))
do
  if [ $i -gt 0 ]
    then
    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19-v'$i'.ckpt'
    else
    ckpt_file='ckpts/lb/nerfpp_CLNerf/'$data'_'$task_number'task_'$rep'/epoch=19.ckpt'
    fi
    python train_CLNerf.py \
        --root_dir $ROOT_DIR'/'$data --dataset_name nerfpp_CLNerf \
        --exp_name $data'_'$task_number'task_'$rep \
        --num_epochs 0 --scale 16.0 --batch_size 4096 --rep_size $rep --eval_lpips \
        --task_curr $i --task_number $task_number \
        --val_only \
        --weight_path $ckpt_file \
        --save_depth_pcd --depth_clip 0.5
done

