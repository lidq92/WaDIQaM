#!/bin/bash
# nohup ./jobWaDIQaM.sh > WaDIQaM-FR_TID2013-0-10.log 2>&1 &

source activate ~/anaconda3/envs/tensorflow/
cd /home/ldq/WaDIQaM/
for ((i=0; i<10; i++)); do
    CUDA_VISIBLE_DEVICES=0 python WaDIQaM.py $i config.yaml TID2013 WaDIQaM-FR
done;
source deactivate
