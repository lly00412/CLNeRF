#!/bin/bash

# EWC on WAT dataset
rep=0
export ROOT_DIR=/mnt/Data2/nerf_datasets/WAT/
bash scripts/EWC/WAT/nerf_baseline.sh breville ${rep} 5 ${ROOT_DIR}
bash scripts/EWC/WAT/nerf_baseline.sh community ${rep} 5 ${ROOT_DIR}
bash scripts/EWC/WAT/nerf_baseline.sh kitchen ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/WAT/nerf_baseline.sh living_room ${rep} 5 ${ROOT_DIR}
bash scripts/EWC/WAT/nerf_baseline.sh spa ${rep} 5 ${ROOT_DIR}
bash scripts/EWC/WAT/nerf_baseline.sh street ${rep} 5 ${ROOT_DIR}

# # # # EWC on Synth-NeRF dataset
export ROOT_DIR=/mnt/Data2/nerf_datasets/nerf_synthetic
bash scripts/EWC/SynthNeRF/nerf_baseline.sh chair ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh drums ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh ficus ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh hotdog ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh lego ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh materials ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh mic ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/SynthNeRF/nerf_baseline.sh ship ${rep} 10 ${ROOT_DIR}

# # # # EWC on NeRF++ dataset
export ROOT_DIR=/mnt/Data2/nerf_datasets/tanks_and_temples
bash scripts/EWC/nerfpp/nerf_baseline.sh tat_intermediate_M60 ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/nerfpp/nerf_baseline.sh tat_intermediate_Playground ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/nerfpp/nerf_baseline.sh tat_intermediate_Train ${rep} 10 ${ROOT_DIR}
bash scripts/EWC/nerfpp/nerf_baseline.sh tat_training_Truck ${rep} 10 ${ROOT_DIR}
