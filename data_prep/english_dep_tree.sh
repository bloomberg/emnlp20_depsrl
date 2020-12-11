#!/bin/bash

for split in train dev test; do
    FILE=./english.${split}.conll
    PENNFILE=${FILE}.penn
    DEPFILE=${FILE}.dep

    CORENLP="./stanford-corenlp-full-2018-10-05/*"

    java -mx8g -cp "${CORENLP}" edu.stanford.nlp.trees.EnglishGrammaticalStructure \
        -treeFile $PENNFILE \
        -basic -conllx -keepPunct -makeCopulaHead \
        > $DEPFILE
done
