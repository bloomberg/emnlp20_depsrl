#!/bin/bash

rm -f english.train.conll
cat ./conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/*/*/*/*.gold_conll >> english.train.conll

rm -f english.dev.conll
cat ./conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/development/data/english/annotations/*/*/*/*.gold_conll >> english.dev.conll

rm -f english.test.conll
cat ./conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/conll-2012-test/data/english/annotations/*/*/*/*.gold_conll >> english.test.conll

rm -f chinese.train.conll
cat ./conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/train/data/chinese/annotations/*/*/*/*gold_conll >> chinese.train.conll

rm -f chinese.dev.conll
cat ./conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/development/data/chinese/annotations/*/*/*/*gold_conll >> chinese.dev.conll

rm -f chinese.test.conll
cat ./conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/test/data/chinese/annotations/*/*/*/*gold_conll >> chinese.test.conll
