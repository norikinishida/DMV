#!/usr/bin/env sh

PATH_PTBWSJ=$1
PENNCONVERTER=$2
PATH_DEP=$3

######################
CONLLX_DIR=${PATH_DEP}/"ptbwsj-conllx"
if [ ! -d ${CONLLX_DIR} ]; then
    mkdir ${CONLLX_DIR}
fi
for dir in ${PATH_PTBWSJ}/*
do
    CONLLX_SUBDIR="${CONLLX_DIR}/`basename ${dir}`"
    if [ ! -d ${CONLLX_SUBDIR} ]; then
        mkdir ${CONLLX_SUBDIR}
    fi
    for f in ${dir}/*.mrg
    do
        CONLLX_FILE="${CONLLX_SUBDIR}/`basename "${f}" .mrg`.conllx"
        # java -mx2g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile "${f}" > "${CONLLX_FILE}"
        java -jar ${PENNCONVERTER}/pennconverter.jar < ${f} > ${CONLLX_FILE}
        echo "Processed ${f}, output in ${CONLLX_FILE}"
    done
done

######################
CONCAT_DIR=${PATH_DEP}/"ptbwsj-conllx.concat"
if [ ! -d ${CONCAT_DIR} ]; then
    mkdir ${CONCAT_DIR}
fi
for dir in ${CONLLX_DIR}/*
do
    if [ ! -d "${CONCAT_DIR}/`basename ${dir}`" ]; then
        mkdir "${CONCAT_DIR}/`basename ${dir}`"
    fi
    cat "${CONLLX_DIR}/`basename ${dir}`"/*.conllx > "${CONCAT_DIR}/`basename ${dir}`"/concat.conllx
    echo "Processed ${CONLLX_DIR}/`basename ${dir}`, output in ${CONCAT_DIR}/`basename ${dir}`/concat.conllx"
done

######################
SPLIT_DIR=${PATH_DEP}/"ptbwsj-conllx.concat.split"
if [ ! -d ${SPLIT_DIR} ]; then
    mkdir ${SPLIT_DIR}
fi
cat ${CONCAT_DIR}/02/concat.conllx \
    ${CONCAT_DIR}/03/concat.conllx \
    ${CONCAT_DIR}/04/concat.conllx \
    ${CONCAT_DIR}/05/concat.conllx \
    ${CONCAT_DIR}/06/concat.conllx \
    ${CONCAT_DIR}/07/concat.conllx \
    ${CONCAT_DIR}/08/concat.conllx \
    ${CONCAT_DIR}/09/concat.conllx \
    ${CONCAT_DIR}/10/concat.conllx \
    ${CONCAT_DIR}/11/concat.conllx \
    ${CONCAT_DIR}/12/concat.conllx \
    ${CONCAT_DIR}/13/concat.conllx \
    ${CONCAT_DIR}/14/concat.conllx \
    ${CONCAT_DIR}/15/concat.conllx \
    ${CONCAT_DIR}/16/concat.conllx \
    ${CONCAT_DIR}/17/concat.conllx \
    ${CONCAT_DIR}/18/concat.conllx \
    ${CONCAT_DIR}/19/concat.conllx \
    ${CONCAT_DIR}/20/concat.conllx \
    ${CONCAT_DIR}/21/concat.conllx \
    > ${SPLIT_DIR}/train.conllx
cp ${CONCAT_DIR}/22/concat.conllx ${SPLIT_DIR}/dev.conllx
cp ${CONCAT_DIR}/23/concat.conllx ${SPLIT_DIR}/test.conllx
echo "Processed splitting."


