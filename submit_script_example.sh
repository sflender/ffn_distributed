#!/bin/bash
#COBALT -t 1440
#COBALT -n 1024
#COBALT --attrs mcdram=cache:numa=quad
#COBALT -A datascience
#COBALT -q default
source activate /home/flender/envs/py3.6_tf1.4

# Number of MPI ranks per node
PPN=1
# Pete's recommendation: 62 threads and 3 tensorflow "hyperthreads" (num_inter_threads)
export OMP_NUM_THREADS=64

train_dir=${PWD}/train_dir

cd /home/flender/projects/ffn_distributed/

aprun -n $((${COBALT_PARTSIZE} * ${PPN})) -N ${PPN} -d 64 -j 1 -cc depth -b python train.py \
    --train_coords /projects/datascience/flender/data/ac4/ac4_partitions_lom41-41-8_minsize0_coords.tfr \
    --data_volumes ac4:/projects/datascience/flender/data/ac4/ac4_den15_clahe2-32.h5:raw \
    --label_volumes ac4:/projects/datascience/flender/data/ac4/ac4_den15_clahe2-32.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [55, 55, 11], \"deltas\": [14, 14, 3]}" \
    --image_mean 141 \
    --image_stddev 59 \
    --max_steps 30000000 \
    --optimizer "adam" \
    --learning_rate 0.001 \
    --batch_size 1 \
    --summary_every_steps 30 \
    --save_model_secs 1800 \
    --train_dir ${train_dir} \
    --num_intra_threads 64 \
    --num_inter_threads 1 \