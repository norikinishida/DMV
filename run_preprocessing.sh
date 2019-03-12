#!/usr/bin/env sh

PATH_PTBWSJ=/mnt/hdd/dataset/Penn-Treebank/np_bracketing/PTB_NP_Bracketing_Data_1.0/TreeBank3_np_v1.0
PENNCONVERTER=/home/nishida/main/software/pennconverter
PATH_DEP=/mnt/hdd/projects/DMV/data

./preprocessing/prepare_ptbwsj.sh ${PATH_PTBWSJ} ${PENNCONVERTER} ${PATH_DEP}
python preprocessing/remove_punctuations_ptbwsj.py
python preprocessing/convert_conllx_ptbwsj.py
python preprocessing/preprocess_ptbwsj.py
python preprocessing/build_vocabulary_ptbwsj.py
