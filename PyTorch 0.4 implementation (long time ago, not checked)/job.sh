#!/bin/bash
# nohup ./job.sh > WaDIQaM-FR-LIVE-0-10.log 2>&1 &
source activate research
cd /home/ldq/LDQ/Research/WaDIQaM-pytorch/
for ((i=0; i<10; i++)); do
    CUDA_VISIBLE_DEVICES=1 python main.py --exp_id=$i --database=LIVE --model=WaDIQaM-FR
done;
source deactivate
