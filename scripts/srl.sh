#!/bin/bash

source ~/environments/python3env/bin/activate

EVAL_STEPS=3200
DECAY_EVALS=5
DECAY_TIMES=2
DECAY_RATIO=0.1

BATCH_SIZE=8

LEARNING_RATE=1e-5
BETA1=0.9
BETA2=0.999
EPSILON=1e-8
WEIGHT_DECAY=0
WARMUP=320

CLIP=5.0

GPU=True

CUTOFF=1
WORD_SMOOTH=0.3

WDIMS=100
EDIMS=0
CDIMS=0
PDIMS=16
WORD_DROPOUT=0.

BILSTM_DIMS=800
BILSTM_LAYERS=3
BILSTM_DROPOUT=0.

CHAR_HIDDEN=128
CHAR_DROPOUT=0.

UTAGGER_DIMS=256
UTAGGER_LAYERS=1
UTAGGER_DROPOUT=0.

HSEL_DIMS=400
HSEL_DROPOUT=0.

REL_DIMS=100
REL_DROPOUT=0.

HSEL_WEIGHT=1.0
REL_WEIGHT=1.0

TRANS_POS_DIM=256
TRANS_FFN_DIM=512
TRANS_EMB_DROPOUT=0.
TRANS_NUM_LAYERS=4
TRANS_NUM_HEADS=4
TRANS_ATTN_DROPOUT=0.
TRANS_ACTN_DROPOUT=0.
TRANS_RES_DROPOUT=0.

BASELINE=True
BIOCRF=False

TRANSFORMER=False
BERT=False

EMBEDDING_FILE=./embeddings/glove.6b.100

TRAIN_FILE=./path/to/trainfile
DEV_FILE=./path/to/devfile

LOG_FOLDER=./models/localtest/

mkdir -p $LOG_FOLDER

RUN=testrun

SAVE_PREFIX=${LOG_FOLDER}/${RUN}

mkdir -p $SAVE_PREFIX

OMP_NUM_THREADS=3 \
python3 -m srl.parser --baseline $BASELINE \
    - build-vocab $TRAIN_FILE --cutoff ${CUTOFF} \
    - create-parser --batch-size $BATCH_SIZE --word-smooth $WORD_SMOOTH \
        --learning-rate $LEARNING_RATE --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --clip $CLIP \
        --wdims $WDIMS --cdims $CDIMS --edims $EDIMS --pdims $PDIMS \
        --word-dropout $WORD_DROPOUT \
        --bilstm-dims $BILSTM_DIMS --bilstm-layers $BILSTM_LAYERS --bilstm-dropout $BILSTM_DROPOUT \
        --char-hidden $CHAR_HIDDEN --char-dropout $CHAR_DROPOUT \
        --utagger-dims $UTAGGER_DIMS --utagger-dropout $UTAGGER_DROPOUT \
        --utagger-layers $UTAGGER_LAYERS \
        --hsel-dims $HSEL_DIMS --hsel-dropout $HSEL_DROPOUT \
        --rel-dims $REL_DIMS --rel-dropout $REL_DROPOUT \
        --rel-weight $REL_WEIGHT --hsel-weight $HSEL_WEIGHT \
        --weight-decay $WEIGHT_DECAY \
        --warmup $WARMUP \
        --biocrf $BIOCRF \
        --bert $BERT \
        --gpu $GPU \
        --transformer $TRANSFORMER \
        --trans-pos-dim $TRANS_POS_DIM --trans-ffn-dim $TRANS_FFN_DIM --trans-emb-dropout $TRANS_EMB_DROPOUT \
        --trans-num-layers $TRANS_NUM_LAYERS --trans-num-heads $TRANS_NUM_HEADS \
        --trans-attn-dropout $TRANS_ATTN_DROPOUT --trans-actn-dropout $TRANS_ACTN_DROPOUT \
        --trans-res-dropout $TRANS_RES_DROPOUT \
    - train $TRAIN_FILE --dev $DEV_FILE \
        --eval-steps $EVAL_STEPS --decay-evals $DECAY_EVALS --decay-times $DECAY_TIMES --decay-ratio $DECAY_RATIO \
        --save-prefix $SAVE_PREFIX/ \
    - finish
