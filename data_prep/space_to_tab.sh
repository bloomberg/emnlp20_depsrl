#!/bin/bash

for lan in english chinese; do
    for split in train dev test; do
        FILE=./${lan}.${split}.conll

        cat $FILE | tr -s " " | \
            sed "/^#begin document/d" |
            sed "/^#end document/d" |
            sed "s/\s/\t/g" | sed "s/\t$//g" | sed "s/\t$//g" > $FILE.tab
    done
done
