#!/bin/bash
LOG=log/train_Total_Text_WSR-`date +%Y-%m-%d-%H-%M-%S`.log

cd /data4/liuzd/ABLSTM/
MPLBACKEND=Agg python ./train_Total_Text_WSR.py --gpu 0 2>&1 | tee $LOG