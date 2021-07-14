#!/bin/bash


/home/jiaweili/anaconda3/envs/pytorch1.6/bin/python example_berttextcnn.py --epochs=10 --bs=32 --lr=0.0001 \
    --wd=0 --run_id=1 --gpu=1  --lr_decay=false --nb_encoder=6

/home/jiaweili/anaconda3/envs/pytorch1.6/bin/python example_berttextcnn.py --epochs=10 --bs=32 --lr=0.0001 \
    --wd=0 --run_id=1 --gpu=1  --lr_decay=false --nb_encoder=9

/home/jiaweili/anaconda3/envs/pytorch1.6/bin/python example_berttextcnn.py --epochs=10 --bs=32 --lr=0.0001 \
    --wd=0 --run_id=1 --gpu=1  --lr_decay=false --nb_encoder=12