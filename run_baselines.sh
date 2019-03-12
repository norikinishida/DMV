#!/usr/bin/env sh

CONFIG=./config/baseline.ini

for MODEL in random
do
    for NAME in trial1 trial2 trial3 trial4 trial5
    do
     python main.py \
         --model ${MODEL} \
         --config ${CONFIG} \
         --name ${NAME} \
         --actiontype baseline
    done
done

for MODEL in right_headed left_headed
do
    python main.py \
        --model ${MODEL} \
        --config ${CONFIG} \
        --name trial1 \
        --actiontype baseline
done

