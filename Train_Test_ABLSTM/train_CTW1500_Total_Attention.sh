#!/bin/bash
LOG=log/train_CTW1500_Total_Attention-`date +%Y-%m-%d-%H-%M-%S`.log

cd /data4/liuzd/ABLSTM/
MPLBACKEND=Agg python ./train_CTW1500_Total_Attention.py --gpu 0 2>&1 | tee $LOG