#!/bin/bash

# LB on WAT dataset
rep=0
export ROOT_DIR=/mnt/Data2/nerf_datasets/WAT/
sh scripts/NT/WAT/breville.sh 0
sh scripts/NT/WAT/nerf_baseline.sh community ${rep} 10 dataset/WAT
sh scripts/NT/WAT/nerf_baseline.sh kitchen ${rep} 5 dataset/WAT
sh scripts/NT/WAT/nerf_baseline.sh living_room ${rep} 5 dataset/WAT
sh scripts/NT/WAT/nerf_baseline.sh spa ${rep} 5 dataset/WAT
sh scripts/NT/WAT/nerf_baseline.sh street ${rep} 5 dataset/WAT

# # # # Synth-NeRF dataset
export ROOT_DIR=/mnt/Data2/nerf_datasets/Synthetic_NeRF
bash scripts/NT/SynthNeRF/benchmark_synth_nerf.sh


# # # #  NeRF++ dataset
export ROOT_DIR=/mnt/Data2/nerf_datasets/tanks_and_temples
bash scripts/NT/nerfpp/benchmark_nerfpp.sh
