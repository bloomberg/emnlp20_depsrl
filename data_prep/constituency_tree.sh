#!/bin/bash

for lan in english chinese; do
    for split in train dev test; do
        FILE=./${lan}.${split}.conll
        OUTFILE=${FILE}.penn

        cut -f4,5,6 <(sed -e 's/^#.*$//g' $FILE.tab) | \
            sed 's/\(.*\)\t\(.*\)\t\(.*\)\*\(.*\)/\3(\2 \1)\4/' | \
            sed 's/(VV ｌａｕｇｈ)/(INTJ ｌａｕｇｈ)/g' \
            > $OUTFILE
    done
done
