#!/usr/bin/env sh

MODEL=dmv
# MODEL=loglineardmv
CONFIG=./config/experiment_26.ini
NAME=trial1

python main.py \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype train

python main.py \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype evaluation

python main.py \
    --model ${MODEL} \
    --config ${CONFIG} \
    --name ${NAME} \
    --actiontype dump_outputs

