from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .crf import CRF
from .attention import BilinearMatrixAttention
from .conversion import srl_backward_v2

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from .calgorithm import (
    parse_proj,
    constrained_bio,
)


class ParserModule(ABC):
    @property
    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @staticmethod
    @abstractmethod
    def load_data(parser, graph, baseline=False):
        pass

    @staticmethod
    @abstractmethod
    def batch_label(batch):
        pass

    @abstractmethod
    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        pass

    @abstractmethod
    def metrics(self, results):
        pass


class SequenceLabeler(nn.Module, ParserModule):
    def __init__(
        self, parser, layer_size, hidden_size, label_size, dropout=0.0, crf=False
    ):
        super(SequenceLabeler, self).__init__()
        print("build sequence labeling network...", self.__class__.name, "crf:", crf)

        self.use_crf = crf
        ## add two more label for downlayer lstm, use original label size for CRF
        self.label_size = label_size + 2
        self.parser = parser

        lst = []
        for i in range(layer_size):
            if i == 0:
                lst.append(nn.Linear(parser._bilstm_dims, hidden_size))
            else:
                lst.append(nn.Linear(hidden_size, hidden_size))

            lst.append(nn.PReLU())
            lst.append(nn.Dropout(dropout))

        if layer_size > 0:
            lst.append(nn.Linear(hidden_size, self.label_size))
        else:
            lst.append(nn.Linear(parser._bilstm_dims, self.label_size))

        self.transform = nn.Sequential(*lst)

        if self.use_crf:
            self.crf = CRF(label_size, gpu=parser._gpu)
        else:
            self.loss = nn.NLLLoss(ignore_index=0, reduction="sum")

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask_h, batch_label)
            total_loss = total_loss / float(batch_size)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask_h)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = self.loss(score, batch_label.view(batch_size * seq_len))
            total_loss = total_loss / float(batch_size)
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        return total_loss, tag_seq

    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask_h)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq

        batch["pred_" + self.name] = tag_seq

        return tag_seq

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        overlaped = pred == gold
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]


class XPOSTagger(SequenceLabeler):

    name = "XPOS"

    @staticmethod
    def load_data(parser, graph, baseline=False):
        labels = [0] + [parser._xpos.get(n.xpos, 1) for n in graph.nodes[1:]]
        return {"xpos": labels}

    @staticmethod
    def batch_label(batch):
        return batch["xpos"]


class UPOSTagger(SequenceLabeler):

    name = "UPOS"

    @staticmethod
    def load_data(parser, graph, baseline=False):
        labels = [0] + [parser._upos.get(n.upos, 1) for n in graph.nodes[1:]]
        return {"upos": labels}

    @staticmethod
    def batch_label(batch):
        return batch["upos"]


class PointerSelector(nn.Module, ParserModule):
    def __init__(self, parser, hidden_size, dropout=0.0):
        super(PointerSelector, self).__init__()
        print("build pointer selector ...", self.__class__.name)
        self.head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.attention = BilinearMatrixAttention(hidden_size, hidden_size, True)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction="sum")

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = self.batch_cand_mask(batch)

        heads = self.head_mlp(lstm_features)
        deps = self.dep_mlp(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = (
            self.attention(deps, heads)
            .masked_fill((1 - mask_att).bool(), float("-inf"))
            .view(batch_size * seq_len, -1)
        )
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        total_loss = self.loss(scores, (batch_label - 1).view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)

        return total_loss, tag_seq

    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]

        mask_h = self.batch_cand_mask(batch)
        heads = self.head_mlp(lstm_features)
        deps = self.dep_mlp(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = (
            self.attention(deps, heads)
            .masked_fill((1 - mask_att).bool(), float("-inf"))
            .view(batch_size * seq_len, -1)
        )
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        scores = (
            scores.view(batch_size, seq_len, -1).cpu().data.numpy().astype("float64")
        )
        word_length = batch["word_length"].cpu().data.numpy()
        for i in range(batch_size):
            l = int(word_length[i])
            s = scores[i, :l, :l].T
            heads = parse_proj(s)
            tag_seq[i, :l] = torch.Tensor(heads + 1)

        ## filter padded position with zero
        tag_seq = mask.long() * tag_seq

        batch["pred_head"] = tag_seq

        return tag_seq

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        overlaped = pred == gold
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    @staticmethod
    @abstractmethod
    def batch_cand_mask(batch):
        pass


class HSelParser(PointerSelector):

    name = "HSel"

    @staticmethod
    def load_data(parser, graph, baseline=False):
        return {"head": graph.heads + 1}

    @staticmethod
    def batch_label(batch):
        return batch["head"]

    @staticmethod
    def batch_cand_mask(batch):
        return batch["mask_h"]


class RelLabeler(nn.Module, ParserModule):

    name = "Rel"

    def __init__(self, parser, hidden_size, dropout=0.0):
        super(RelLabeler, self).__init__()
        print("build rel labeler...", self.__class__.name)
        self.head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.attention = nn.Bilinear(
            hidden_size, hidden_size, len(parser._rels) + 1, True
        )
        self.bias_x = nn.Linear(hidden_size, len(parser._rels) + 1, False)
        self.bias_y = nn.Linear(hidden_size, len(parser._rels) + 1, False)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction="sum")

    def calculate_loss(self, lstm_features, batch):
        batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask = batch["mask"]
        head = torch.abs(batch["head"] - 1)

        ran = torch.arange(batch_size, device=head.get_device()).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        heads = self.head_mlp(lstm_features).view(batch_size * seq_len, -1)[idx]
        deps = self.dep_mlp(lstm_features).view(batch_size * seq_len, -1)

        scores = self.attention(deps, heads) + self.bias_x(heads) + self.bias_y(deps)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)

        total_loss = self.loss(scores, batch_label.view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)

        return total_loss, tag_seq

    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask = batch["mask"]
        if "pred_head" in batch:
            head = torch.abs(batch["pred_head"] - 1)
        else:
            head = torch.abs(batch["head"] - 1)

        ran = torch.arange(batch_size, device=head.get_device()).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        heads = self.head_mlp(lstm_features).view(batch_size * seq_len, -1)[idx]
        deps = self.dep_mlp(lstm_features).view(batch_size * seq_len, -1)

        scores = self.attention(deps, heads) + self.bias_x(heads) + self.bias_y(deps)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        tag_seq = mask.long() * tag_seq

        batch["pred_rel"] = tag_seq

        return tag_seq

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        overlaped = pred == gold
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    @staticmethod
    def load_data(parser, graph, baseline=False):
        labels = [0] + [parser._rels.get(r, 1) for r in graph.rels[1:]]
        return {"rel": labels}

    @staticmethod
    def batch_label(batch):
        return batch["rel"]


class SRLBIOTagger(SequenceLabeler):

    name = "SRLBIO"

    @staticmethod
    def load_data(parser, graph, baseline=False):
        labels = [parser._label[lab] for lab in graph.bio_labels]
        return {"srlbio": labels}

    @staticmethod
    def batch_label(batch):
        return batch["srlbio"]

    def forward(self, parser, lstm_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        outs = self.transform(lstm_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask_h)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            outs = (
                F.log_softmax(outs, dim=1)
                .view(batch_size, seq_len, -1)
                .cpu()
                .data.numpy()
                .astype("float64")
            )
            word_lengths = mask.sum(dim=1).cpu().data.numpy().astype("int")

            tag_seq = constrained_bio(word_lengths, outs)
            tag_seq = torch.LongTensor(tag_seq).to(mask.device)

        batch["pred_" + self.name] = tag_seq

        return tag_seq

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        batch_size, length = mask.shape
        pred = pred.cpu().data.numpy()
        gold = gold.cpu().data.numpy()
        mask = mask.cpu().data.numpy()
        pred_set = set()
        gold_set = set()
        for i in range(batch_size):
            curstart = None
            curlabel = None
            for j in range(1, length):
                if mask[i, j] == 0:
                    if curstart is not None and curlabel != "V":
                        pred_set.add((i, curstart, j, curlabel))
                    break
                if self.parser._ilabel[pred[i, j]][0] == "B":
                    if curstart is not None and curlabel != "V":
                        pred_set.add((i, curstart, j, curlabel))
                    curstart = j
                    curlabel = self.parser._ilabel[pred[i, j]][2:]
                elif self.parser._ilabel[pred[i, j]][0] == "I":
                    if curstart is None:
                        curstart = j
                        curlabel = self.parser._ilabel[pred[i, j]][2:]
                elif self.parser._ilabel[pred[i, j]][0] == "O":
                    if curstart is not None and curlabel != "V":
                        pred_set.add((i, curstart, j, curlabel))
                    curstart = None

        for i in range(batch_size):
            curstart = None
            curlabel = None
            for j in range(1, length):
                if mask[i, j] == 0:
                    if curstart is not None and curlabel != "V":
                        gold_set.add((i, curstart, j, curlabel))
                    break
                if self.parser._ilabel[gold[i, j]][0] == "B":
                    if curstart is not None and curlabel != "V":
                        gold_set.add((i, curstart, j, curlabel))
                    curstart = j
                    curlabel = self.parser._ilabel[gold[i, j]][2:]
                elif self.parser._ilabel[gold[i, j]][0] == "I":
                    if curstart is None:
                        curstart = j
                        curlabel = self.parser._ilabel[gold[i, j]][2:]
                elif self.parser._ilabel[gold[i, j]][0] == "O":
                    if curstart is not None and curlabel != "V":
                        gold_set.add((i, curstart, j, curlabel))
                    curstart = None

        results["{}-p".format(self.__class__.name)] += len(pred_set)
        results["{}-r".format(self.__class__.name)] += len(gold_set)
        results["{}-c".format(self.__class__.name)] += len(
            pred_set.intersection(gold_set)
        )

    def metrics(self, results):
        p = results["{}-p".format(self.__class__.name)]
        r = results["{}-r".format(self.__class__.name)]
        c = results["{}-c".format(self.__class__.name)]
        precision = c / (p + 1e-6)
        recall = c / (r + 1e-6)
        results["metrics/{}-p".format(self.__class__.name)] = precision * 100.0
        results["metrics/{}-r".format(self.__class__.name)] = recall * 100.0
        results["metrics/{}-f1".format(self.__class__.name)] = (
            2.0 / (1.0 / (precision + 1e-6) + 1.0 / (recall + 1e-6)) * 100.0
        )
        del results["{}-p".format(self.__class__.name)]
        del results["{}-r".format(self.__class__.name)]
        del results["{}-c".format(self.__class__.name)]


class SRLDepParser(nn.Module, ParserModule):

    name = "SRLDep"

    def __init__(self, parser, hsel_dims, rel_dims, dropout=0.0):
        super(SRLDepParser, self).__init__()

        print("build srl dep parser ...", self.__class__.name)

        # HSEL
        self.hsel_head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hsel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.hsel_dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, hsel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.hsel_attention = BilinearMatrixAttention(hsel_dims, hsel_dims, True)

        # REL
        self.rel_head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.rel_dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.rel_attention = nn.Bilinear(
            rel_dims, rel_dims, len(parser._rels), True
        )
        self.rel_bias_x = nn.Linear(rel_dims, len(parser._rels), False)
        self.rel_bias_y = nn.Linear(rel_dims, len(parser._rels), False)

        # SREL
        self.srel_head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_attention = nn.Bilinear(
            rel_dims, rel_dims, len(parser._srels), True
        )
        self.srel_bias_x = nn.Linear(rel_dims, len(parser._srels), False)
        self.srel_bias_y = nn.Linear(rel_dims, len(parser._srels), False)

        # SREL_R
        self.srel_r_head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_r_dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_r_attention = nn.Bilinear(
            rel_dims, rel_dims, len(parser._srels_r), True
        )
        self.srel_r_bias_x = nn.Linear(rel_dims, len(parser._srels_r), False)
        self.srel_r_bias_y = nn.Linear(rel_dims, len(parser._srels_r), False)

        # SREL_M
        self.srel_m_head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_m_dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_m_attention = nn.Bilinear(
            rel_dims, rel_dims, len(parser._srels_m), True
        )
        self.srel_m_bias_x = nn.Linear(rel_dims, len(parser._srels_m), False)
        self.srel_m_bias_y = nn.Linear(rel_dims, len(parser._srels_m), False)

        # SREL_C
        self.srel_c_head_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_c_dep_mlp = nn.Sequential(
            nn.Linear(parser._bilstm_dims, rel_dims),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )

        self.srel_c_attention = nn.Bilinear(
            rel_dims, rel_dims, len(parser._srels_c), True
        )
        self.srel_c_bias_x = nn.Linear(rel_dims, len(parser._srels_c), False)
        self.srel_c_bias_y = nn.Linear(rel_dims, len(parser._srels_c), False)

        self.hsel_loss = nn.NLLLoss(ignore_index=-1, reduction="sum")
        self.rel_loss = nn.NLLLoss(ignore_index=0, reduction="sum")

    @staticmethod
    def load_data(parser, graph, baseline=False):
        heads = graph.heads + 1
        rels = [parser._rels.get(r, 0) for r in graph.rels]
        srels = [parser._srels.get(r, 0) for r in graph.srels]
        srels_r = [parser._srels_r.get(r, 0) for r in graph.srels_r]
        srels_m = [parser._srels_m.get(r, 0) for r in graph.srels_m]
        srels_c = [parser._srels_c.get(r, 0) for r in graph.srels_c]

        return {
            "head": heads,
            "rel": rels,
            "srel": srels,
            "srel_r": srels_r,
            "srel_m": srels_m,
            "srel_c": srels_c,
        }

    @staticmethod
    def batch_label(batch):
        return (
            batch["head"],
            batch["rel"],
            batch["srel"],
            batch["srel_r"],
            batch["srel_m"],
            batch["srel_c"],
        )

    def calculate_loss(self, lstm_features, batch):
        batch_head, batch_rel, batch_srel, batch_srel_r, batch_srel_m, batch_srel_c = self.batch_label(
            batch
        )
        mask = batch["mask"]
        mask_h = batch["mask_h"]

        batch_size = mask.size(0)
        seq_len = mask.size(1)

        # HSEL
        heads = self.hsel_head_mlp(lstm_features)
        deps = self.hsel_dep_mlp(lstm_features)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = (
            self.hsel_attention(deps, heads)
            .masked_fill((1 - mask_att).bool(), float("-inf"))
            .view(batch_size * seq_len, -1)
        )
        scores = F.log_softmax(scores, dim=1)
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1
        total_loss = self.hsel_loss(scores, (batch_head - 1).view(batch_size * seq_len))

        # COMMON PRE
        head = torch.abs(batch_head - 1)
        ran = torch.arange(batch_size, device=mask.device).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        # REL
        heads = self.rel_head_mlp(lstm_features).view(batch_size * seq_len, -1)[idx]
        deps = self.rel_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.rel_attention(deps, heads)
            + self.rel_bias_x(heads)
            + self.rel_bias_y(deps)
        )
        scores = F.log_softmax(scores, dim=1)
        total_loss = total_loss + self.rel_loss(
            scores, batch_rel.view(batch_size * seq_len)
        )

        # SREL
        heads = self.srel_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_attention(deps, heads)
            + self.srel_bias_x(heads)
            + self.srel_bias_y(deps)
        )
        scores = F.log_softmax(scores, dim=1)
        total_loss = total_loss + self.rel_loss(
            scores, batch_srel.view(batch_size * seq_len)
        )

        # SREL_R
        heads = self.srel_r_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_r_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_r_attention(deps, heads)
            + self.srel_r_bias_x(heads)
            + self.srel_r_bias_y(deps)
        )
        scores = F.log_softmax(scores, dim=1)
        total_loss = total_loss + self.rel_loss(
            scores, batch_srel_r.view(batch_size * seq_len)
        )

        # SREL_M
        heads = self.srel_m_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_m_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_m_attention(deps, heads)
            + self.srel_m_bias_x(heads)
            + self.srel_m_bias_y(deps)
        )
        scores = F.log_softmax(scores, dim=1)
        total_loss = total_loss + self.rel_loss(
            scores, batch_srel_m.view(batch_size * seq_len)
        )

        # SREL_C
        heads = self.srel_c_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_c_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_c_attention(deps, heads)
            + self.srel_c_bias_x(heads)
            + self.srel_c_bias_y(deps)
        )
        scores = F.log_softmax(scores, dim=1)
        total_loss = total_loss + self.rel_loss(
            scores, batch_srel_c.view(batch_size * seq_len)
        )

        total_loss = total_loss / float(batch_size)

        return total_loss, tag_seq

    def forward(self, parser, lstm_features, batch):
        mask = batch["mask"]
        mask_h = batch["mask_h"]
        word_length = batch["word_length"].cpu().data.numpy()
        batch_size = mask.size(0)
        seq_len = mask.size(1)

        # HSEL
        heads = self.hsel_head_mlp(lstm_features)
        deps = self.hsel_dep_mlp(lstm_features)
        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = (
            self.hsel_attention(deps, heads)
            .masked_fill((1 - mask_att).bool(), float("-inf"))
            .view(batch_size * seq_len, -1)
        )
        scores = F.log_softmax(scores, dim=1)
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1
        scores = (
            scores.view(batch_size, seq_len, seq_len)
            .cpu()
            .data.numpy()
            .astype("float64")
        )
        for i in range(batch_size):
            l = int(word_length[i])
            s = scores[i, :l, :l].T
            heads = parse_proj(s)
            tag_seq[i, :l] = torch.Tensor(heads + 1)
        pred_head = mask.long() * tag_seq

        # REL - PREP
        head = torch.abs(pred_head - 1)
        ran = torch.arange(batch_size, device=mask.device).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        # REL
        heads = self.rel_head_mlp(lstm_features).view(batch_size * seq_len, -1)[idx]
        deps = self.rel_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.rel_attention(deps, heads)
            + self.rel_bias_x(heads)
            + self.rel_bias_y(deps)
        )
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        pred_rel = mask.long() * tag_seq

        # SREL
        heads = self.srel_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_attention(deps, heads)
            + self.srel_bias_x(heads)
            + self.srel_bias_y(deps)
        )
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        pred_srel = mask.long() * tag_seq

        # SREL_R
        heads = self.srel_r_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_r_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_r_attention(deps, heads)
            + self.srel_r_bias_x(heads)
            + self.srel_r_bias_y(deps)
        )
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        pred_srel_r = mask.long() * tag_seq

        # SREL_M
        heads = self.srel_m_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_m_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_m_attention(deps, heads)
            + self.srel_m_bias_x(heads)
            + self.srel_m_bias_y(deps)
        )
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        pred_srel_m = mask.long() * tag_seq

        # SREL_C
        heads = self.srel_c_head_mlp(lstm_features).view(batch_size * seq_len, -1)[
            idx
        ]
        deps = self.srel_c_dep_mlp(lstm_features).view(batch_size * seq_len, -1)
        scores = (
            self.srel_c_attention(deps, heads)
            + self.srel_c_bias_x(heads)
            + self.srel_c_bias_y(deps)
        )
        _, tag_seq = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        pred_srel_c = mask.long() * tag_seq

        return pred_head, pred_rel, pred_srel, pred_srel_r, pred_srel_m, pred_srel_c

    def evaluate(self, results, parser, graphs, pred, gold, mask, train=False):
        if train:
            return
        else:
            total = float(mask.sum())
            head, rel, srel, srel_r, srel_m, srel_c = gold
            pred_head, pred_rel, pred_srel, pred_srel_r, pred_srel_m, pred_srel_c = pred
            batch_len = mask.size(0)

            results["{}-t".format(self.__class__.name)] += total
            results["{}-hsel-c".format(self.__class__.name)] += float(
                ((pred_head == head) * mask).sum()
            )
            results["{}-rel-c".format(self.__class__.name)] += float(
                ((pred_rel == rel) * mask).sum()
            )
            results["{}-srel-c".format(self.__class__.name)] += float(
                ((pred_srel == srel) * mask).sum()
            )
            results["{}-srel-r-c".format(self.__class__.name)] += float(
                ((pred_srel_r == srel_r) * mask).sum()
            )
            results["{}-srel-m-c".format(self.__class__.name)] += float(
                ((pred_srel_m == srel_m) * mask).sum()
            )
            results["{}-srel-c-c".format(self.__class__.name)] += float(
                ((pred_srel_c == srel_c) * mask).sum()
            )

            pred_head = pred_head.cpu().data.numpy()
            pred_rel = pred_rel.cpu().data.numpy()
            pred_srel = pred_srel.cpu().data.numpy()
            pred_srel_r = pred_srel_r.cpu().data.numpy()
            pred_srel_m = pred_srel_m.cpu().data.numpy()
            pred_srel_c = pred_srel_c.cpu().data.numpy()
            mask = mask.cpu().data.numpy()
            for batch_i in range(batch_len):
                graph = graphs[batch_i]
                length = int(sum(mask[batch_i])) + 1
                h = pred_head[batch_i, :length] - 1
                r = [parser._irels[x] for x in pred_rel[batch_i, :length]]
                sr = [parser._isrels[x] for x in pred_srel[batch_i, :length]]
                sr_r = [parser._isrels_r[x] for x in pred_srel_r[batch_i, :length]]
                sr_m = [parser._isrels_m[x] for x in pred_srel_m[batch_i, :length]]
                sr_c = [parser._isrels_c[x] for x in pred_srel_c[batch_i, :length]]
                pred_spans = set(
                    srl_backward_v2(h, r, sr, sr_r, sr_m, sr_c, graph.pset)
                )
                gold_spans = set()
                for srl in graph.srl:
                    for l, r, lab in srl.spans:
                        if lab != "V":
                            gold_spans.add((srl.pred_id, l, r, lab))
                for srl in graph.srl:
                    srl.pred_spans = []
                    for pid, l, r, lab in pred_spans:
                        if pid == srl.pred_id:
                            srl.pred_spans.append((l, r, lab))
                    srl.pred_spans.append((srl.pred_id, srl.pred_id, "V"))
                results["{}-p".format(self.__class__.name)] += len(pred_spans)
                results["{}-r".format(self.__class__.name)] += len(gold_spans)
                results["{}-c".format(self.__class__.name)] += len(
                    pred_spans.intersection(gold_spans)
                )
            return

    def metrics(self, results):
        total = results["{}-t".format(self.__class__.name)]

        correct = results["{}-hsel-c".format(self.__class__.name)]
        results["metrics/{}-hsel-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-hsel-c".format(self.__class__.name)]
        correct = results["{}-rel-c".format(self.__class__.name)]
        results["metrics/{}-rel-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-rel-c".format(self.__class__.name)]
        correct = results["{}-srel-c".format(self.__class__.name)]
        results["metrics/{}-srel-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-srel-c".format(self.__class__.name)]
        correct = results["{}-srel-r-c".format(self.__class__.name)]
        results["metrics/{}-srel-r-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-srel-r-c".format(self.__class__.name)]
        correct = results["{}-srel-m-c".format(self.__class__.name)]
        results["metrics/{}-srel-m-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-srel-m-c".format(self.__class__.name)]
        correct = results["{}-srel-c-c".format(self.__class__.name)]
        results["metrics/{}-srel-c-acc".format(self.__class__.name)] = (
            correct / (total + 1e-10) * 100.0
        )
        del results["{}-srel-c-c".format(self.__class__.name)]

        del results["{}-t".format(self.__class__.name)]

        p = results["{}-p".format(self.__class__.name)]
        r = results["{}-r".format(self.__class__.name)]
        c = results["{}-c".format(self.__class__.name)]
        precision = c / (p + 1e-10)
        recall = c / (r + 1e-10)
        results["metrics/{}-p".format(self.__class__.name)] = precision * 100.0
        results["metrics/{}-r".format(self.__class__.name)] = recall * 100.0
        results["metrics/{}-f1".format(self.__class__.name)] = (
            2.0 / (1.0 / (precision + 1e-10) + 1.0 / (recall + 1e-10)) * 100.0
        )
        del results["{}-p".format(self.__class__.name)]
        del results["{}-r".format(self.__class__.name)]
        del results["{}-c".format(self.__class__.name)]
