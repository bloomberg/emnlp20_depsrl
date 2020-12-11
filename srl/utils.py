import re
import random
from collections import Counter

import torch

if torch.cuda.is_available():
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).cuda()
else:
    from torch import from_numpy


BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
}


def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def buildVocab(sents, graphs, cutoff=1, labelCutoff=1):
    wordsCount = Counter()
    charsCount = Counter()
    xposCount = Counter()
    relsCount = Counter()
    srelsCount = Counter()
    srels_rCount = Counter()
    srels_mCount = Counter()
    srels_cCount = Counter()
    spanlabelCount = Counter()

    for sent in sents:
        if len(sent.srl) == 0:
            continue

        wordsCount.update([node.norm for node in sent.nodes])
        for node in sent.nodes[:]:
            charsCount.update(list(node.word))
        xposCount.update([node.xpos for node in sent.nodes])
        relsCount.update(sent.rels[1:])
        srelsCount.update(sent.srels[1:])
        srels_rCount.update(sent.srels_r[1:])
        srels_mCount.update(sent.srels_m[1:])
        srels_cCount.update(sent.srels_c[1:])

    for graph in graphs:
        spanlabelCount.update(graph.labels)

    print("Number of tokens in training corpora: {}".format(sum(wordsCount.values())))
    print("Vocab containing {} types before cutting off".format(len(wordsCount)))
    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})

    print(
        "Vocab containing {} types, covering {} words".format(
            len(wordsCount), sum(wordsCount.values())
        )
    )
    print("Charset containing {} chars".format(len(charsCount)))
    print("XPOS containing {} tags".format(len(xposCount)), xposCount)
    print("Rel set containing {} tags".format(len(relsCount)), relsCount)
    print("SRel set containing {} tags".format(len(srelsCount)), srelsCount)
    print("SRel_R set containing {} tags".format(len(srels_rCount)), srels_rCount)
    print("SRel_M set containing {} tags".format(len(srels_mCount)), srels_mCount)
    print("SRel_C set containing {} tags".format(len(srels_cCount)), srels_cCount)
    print("Label set containing {} labels".format(len(spanlabelCount)), spanlabelCount)
    relsCount = Counter({w: i for w, i in relsCount.items() if i >= labelCutoff})
    srelsCount = Counter({w: i for w, i in srelsCount.items() if i >= labelCutoff})
    srels_rCount = Counter({w: i for w, i in srels_rCount.items() if i >= labelCutoff})
    srels_mCount = Counter({w: i for w, i in srels_mCount.items() if i >= labelCutoff})
    srels_cCount = Counter({w: i for w, i in srels_cCount.items() if i >= labelCutoff})
    print("After Cutoff:")
    print("Rel set containing {} tags".format(len(relsCount)))
    print("SRel set containing {} tags".format(len(srelsCount)))
    print("SRel_R set containing {} tags".format(len(srels_rCount)))
    print("SRel_M set containing {} tags".format(len(srels_mCount)))
    print("SRel_C set containing {} tags".format(len(srels_cCount)))

    labset = set(spanlabelCount.keys()) - {"O"}
    labset = ["O"] + list(labset)

    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "xpos": list(xposCount.keys()),
        "spanlab": labset,
        "rels": list(relsCount.keys()),
        "srels": list(srelsCount.keys()),
        "srels_r": list(srels_rCount.keys()),
        "srels_m": list(srels_mCount.keys()),
        "srels_c": list(srels_cCount.keys()),
    }

    return ret


def shuffled_stream(data, batch_size):
    len_data = len(data)
    ret = []
    while True:
        for d in random.sample(data, len_data):
            ret.append(d)
            if len(ret) >= batch_size:
                yield ret
                ret = []


def get_span_from_bio(labels):
    ret = []

    curstart = None
    curlabel = None
    for i, l in enumerate(labels + ["O"]):
        if l[0] == "B":
            if curstart is not None:
                ret.append((curstart, i - 1, curlabel))
            curstart = i
            curlabel = l[2:]
        elif l[0] == "I":
            if curstart is None:
                curstart = i
                curlabel = l[2:]
        elif l[0] == "O":
            if curstart is not None:
                ret.append((curstart, i - 1, curlabel))
            curstart = None
            curlabel = None

    return ret
