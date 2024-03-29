#!/bin/bash

# CLNeRF on WAT dataset
rep=10
export ROOT_DIR=/mnt/Data2/nerf_datasets/WAT/
sh scripts/CLNeRF/WAT/breville.sh ${rep}
sh scripts/CLNeRF/WAT/community.sh ${rep}
sh scripts/CLNeRF/WAT/kitchen.sh ${rep}
sh scripts/CLNeRF/WAT/living_room.sh ${rep}
sh scripts/CLNeRF/WAT/spa.sh ${rep}
sh scripts/CLNeRF/WAT/street.sh ${rep}
sh scripts/CLNeRF/WAT/car.sh ${rep}
sh scripts/CLNeRF/WAT/grill.sh ${rep}
sh scripts/CLNeRF/WAT/mac.sh ${rep}
sh scripts/CLNeRF/WAT/ninja.sh ${rep}


# CLNeRF on Synth-NeRF dataset
rep=10
export ROOT_DIR=/mnt/Data2/nerf_datasets/Synthetic_NeRF
sh scripts/CLNeRF/SynthNeRF/benchmark_synth_nerf.sh ${rep}

# CLNeRF on NeRF++ dataset
rep=10
export ROOT_DIR=/mnt/Data2/nerf_datasets/tanks_and_temples
sh scripts/CLNeRF/nerfpp/benchmark_nerfpp.sh ${rep}

