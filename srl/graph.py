import numpy as np
from .utils import normalize


class Word:
    def __init__(self, word, xpos=None):
        self.word = word
        self.norm = normalize(word)
        self.xpos = xpos if xpos else "_"

    def clone(self):
        return Word(self.word, self.xpos)

    def __repr__(self):
        return "{}_{}".format(self.word, self.xpos)


class Sentence:
    def __init__(self, words, heads, rels):
        self.nodes = np.array([Word("*ROOT*")] + list(words))
        self.heads = np.array([-1] + list(heads))
        self.rels = np.array(["_"] + list(rels))
        self.srels = np.array([None] * len(self.heads))
        self.srels_r = np.array([None] * len(self.heads))
        self.srels_m = np.array([None] * len(self.heads))
        self.srels_c = np.array([None] * len(self.heads))
        self.srl = []
        self.pset = set()


class DataInstance(object):
    def __init__(self, sent, pred_id, labels):
        self.sent = sent
        self.pred_id = pred_id

        # construct spans from label sequences
        spans = []
        spanbeg = None
        spanlab = None
        for j, lab in enumerate(labels):
            if lab == "*":
                continue
            # last token
            elif lab[0] == "*":
                spans.append((spanbeg, j, spanlab))
                spanbeg = None
                spanlab = None
            # beginning token
            elif lab[-1] == "*":
                spanlab = lab[1:-1]
                spanbeg = j
            # single-token span
            elif lab[-2] == "*":
                spans.append((j, j, lab[1:-2]))

        self.spans = spans
        self.span_left = np.array([-1] * len(sent.nodes))
        self.span_right = np.array([-1] * len(sent.nodes))
        self.bio_labels = ["O" for i in range(len(sent.nodes))]
        self.labels = ["O" for i in range(len(sent.nodes))]

        for l, r, lab in spans:
            self.bio_labels[l] = "B-" + lab
            for i in range(l + 1, r + 1):
                self.bio_labels[i] = "I-" + lab

            for i in range(l, r + 1):
                self.span_left[i] = l
                self.span_right[i] = r
                self.labels[i] = lab
