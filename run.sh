#!/bin/bash

#python main.py --model meta-decomposition --data sp500 --gpu 0 --rank 6 --epochs 30 --missing-ratio 0.1
#python main.py --model meta-decomposition --data nasdaq --gpu 0 --rank 5 --epochs 30 --missing-ratio 0.1
#python main.py --model meta-decomposition --data kor-stock --gpu 0 --rank 6 --epochs 30 --missing-ratio 0.1
#python main.py --model meta-decomposition --data fingermovement --gpu 0 --rank 28 --epochs 30 --missing-ratio 0.1
#python main.py --model meta-decomposition --data cricket --gpu 0 --rank 6 --epochs 30 --missing-ratio 0.1
#python main.py --model meta-decomposition --data natops --gpu 0 --rank 24 --epochs 30 --missing-ratio 0.1
python main.py --model meta-decomposition --data natops2 --gpu 0 --rank 24 --epochs 30 --missing-ratio 0.1