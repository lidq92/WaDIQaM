#!/bin/bash
# nohup ./job.sh > WaDIQaM-FR-LIVE-0-2.log 2>&1 &
source activate research
cd /home/ldq/LDQ/Research/WaDIQaM-pytorch/

for ((i=0; i<2; i++)); do
    for ((k=1; k<=5; k++)); do
        CUDA_VISIBLE_DEVICES=1 python main.py --exp_id=$i --k_test=$k --database=LIVE --model=WaDIQaM-FR
    done;
done;
source deactivate
