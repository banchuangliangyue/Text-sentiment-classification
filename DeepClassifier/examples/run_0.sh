#!/bin/bash


/home/jiaweili/anaconda3/envs/pytorch1.6/bin/python example_berttextcnn.py --epochs=10 --bs=64 --lr=0.00005 \
    --wd=0 --run_id=0 --gpu=0  --lr_decay=false --nb_encoder=12

/home/jiaweili/anaconda3/envs/pytorch1.6/bin/python example_berttextcnn.py --epochs=10 --bs=16 --lr=0.00005 \
    --wd=0 --run_id=0 --gpu=0  --lr_decay=false --nb_encoder=6