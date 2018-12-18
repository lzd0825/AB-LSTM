#!/bin/bash
LOG=log/train_ICDAR2013_TCB-`date +%Y-%m-%d-%H-%M-%S`.log

cd /data4/liuzd/ABLSTM/
MPLBACKEND=Agg python ./train_ICDAR2013_TCB.py --gpu 0 2>&1 | tee $LOG