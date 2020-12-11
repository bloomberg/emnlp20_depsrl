import json
import sys
import time
from collections import defaultdict

import fire
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .modules import SRLBIOTagger, SRLDepParser
from .features import WordSequence
from .io import read_conll, write_conll
from .utils import buildVocab, get_span_from_bio
from .data import DataProcessor, DataCollate, InfiniteDataLoader
from .adamw import AdamW


class SRLParser:
    def __init__(self, baseline=False):
        self._baseline = baseline

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
            sys.stdout.flush()

        self._args = kwargs

        self._gpu = kwargs.get("gpu", True)

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-8)
        self._weight_decay = kwargs.get("weight_decay", 0.0)
        self._warmup = kwargs.get("warmup", 1)

        self._clip = kwargs.get("clip", 5.0)

        self._batch_size = kwargs.get("batch_size", 16)

        self._word_smooth = kwargs.get("word_smooth", 0.25)
        self._char_smooth = kwargs.get("char_smooth", 0.25)

        self._wdims = kwargs.get("wdims", 128)
        self._edims = kwargs.get("edims", 0)
        self._cdims = kwargs.get("cdims", 32)
        self._pdims = kwargs.get("pdims", 0)

        self._word_dropout = kwargs.get("word_dropout", 0.0)

        self._char_hidden = kwargs.get("char_hidden", 128)
        self._char_dropout = kwargs.get("char_dropout", 0.0)
        self._bilstm_dims = kwargs.get("bilstm_dims", 256)
        self._bilstm_layers = kwargs.get("bilstm_layers", 2)
        self._bilstm_dropout = kwargs.get("bilstm_dropout", 0.0)

        self._utagger_dims = kwargs.get("utagger_dims", 256)
        self._utagger_layers = kwargs.get("utagger_layers", 1)
        self._utagger_dropout = kwargs.get("utagger_dropout", 0.0)

        self._hsel_dims = kwargs.get("hsel_dims", 200)
        self._hsel_dropout = kwargs.get("hsel_dropout", 0.0)
        self._hsel_weight = kwargs.get("hsel_weight", 1.0)

        self._rel_dims = kwargs.get("rel_dims", 50)
        self._rel_dropout = kwargs.get("rel_dropout", 0.0)
        self._rel_weight = kwargs.get("rel_weight", 1.0)

        self._biocrf = kwargs.get("biocrf", False)
        self._spancrf = kwargs.get("spancrf", False)

        self._bert = kwargs.get("bert", False)
        self._transformer = kwargs.get("transformer", False)
        self._trans_pos_dim = kwargs.get("trans_pos_dim", 128)
        self._trans_ffn_dim = kwargs.get("trans_ffn_dim", 256)
        self._trans_emb_dropout = kwargs.get("trans_emb_dropout", 0.0)
        self._trans_num_layers = kwargs.get("trans_num_layers", 8)
        self._trans_num_heads = kwargs.get("trans_num_heads", 8)
        self._trans_attn_dropout = kwargs.get("trans_attn_dropout", 0.0)
        self._trans_actn_dropout = kwargs.get("trans_actn_dropout", 0.0)
        self._trans_res_dropout = kwargs.get("trans_res_dropout", 0.0)

        self.init_model()
        return self

    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._xpos = {p: i + 2 for i, p in enumerate(vocab["xpos"])}
        self._vocab = {w: i + 2 for i, w in enumerate(vocab["vocab"])}
        self._charset = {c: i + 2 for i, c in enumerate(vocab["charset"])}
        self._wordfreq = vocab["wordfreq"]
        self._charfreq = vocab["charfreq"]
        self._spanlab = {l: i + 1 for i, l in enumerate(vocab["spanlab"])}
        self._ispanlab = ["O"] + vocab["spanlab"] + ["O", "O"]

        self._label = {"O": 1}
        self._ilabel = ["O", "O"]

        for i, l in enumerate(self._ispanlab[2:-2]):
            self._ilabel.append("B-" + l)
            self._ilabel.append("I-" + l)
            self._label["B-" + l] = i * 2 + 2
            self._label["I-" + l] = i * 2 + 3
        self._ilabel.extend(["O", "O"])

        self._irels = ["unk"] + vocab["rels"]
        self._rels = {w: i for i, w in enumerate(self._irels)}
        self._isrels = ["unk"] + vocab["srels"]
        self._srels = {w: i for i, w in enumerate(self._isrels)}
        self._isrels[0] = None
        self._isrels_r = ["unk"] + vocab["srels_r"]
        self._srels_r = {w: i for i, w in enumerate(self._isrels_r)}
        self._isrels_r[0] = None
        self._isrels_m = ["unk"] + vocab["srels_m"]
        self._srels_m = {w: i for i, w in enumerate(self._isrels_m)}
        self._isrels_m[0] = None
        self._isrels_c = ["unk"] + vocab["srels_c"]
        self._srels_c = {
            (tuple(w) if w else w): i for i, w in enumerate(self._isrels_c)
        }
        self._isrels_c[0] = None

    def load_vocab(self, filename):
        with open(filename, "rb") as f:
            vocab = json.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "wb") as f:
            f.write(json.dumps(self._fullvocab).encode("utf-8"))
        return self

    def build_vocab(self, filename, cutoff=1):
        sents = read_conll(filename)
        graphs = [x for g in sents for x in g.srl]

        self._fullvocab = buildVocab(sents, graphs, cutoff)
        self._load_vocab(self._fullvocab)

        return self

    def load_embeddings(self, filename):
        with open(filename + ".vocab", "rb") as f:
            _external_mappings = json.load(f)
        with open(filename + ".npy", "rb") as f:
            _external_embeddings = np.load(f)

        count = 0
        for w in self._vocab:
            if w in _external_mappings:
                count += 1
        print(
            "Loaded embeddings from", filename, count, "hits out of", len(self._vocab)
        )
        self._external_mappings = _external_mappings
        self._external_embeddings = _external_embeddings

        return self

    def save_model(self, filename):
        print("Saving model to", filename)
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "wb") as f:
            f.write(json.dumps(self._args).encode("utf-8"))
        with open(filename + ".model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_model(self, filename, **kwargs):
        print("Loading model from", filename)
        self.load_vocab(filename + ".vocab")
        with open(filename + ".params", "rb") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        with open(filename + ".model", "rb") as f:
            if kwargs.get("gpu", False):
                self._model.load_state_dict(torch.load(f))
            else:
                self._model.load_state_dict(torch.load(f, map_location="cpu"))
        return self

    def init_model(self):
        self._seqrep = WordSequence(self)

        if self._baseline:
            self._srl_tagger = SRLBIOTagger(
                self, self._utagger_layers, self._utagger_dims, len(self._label) + 1,
                self._utagger_dropout, crf=self._biocrf
            )
        else:
            self._srl_tagger = SRLDepParser(
                self, self._hsel_dims, self._rel_dims, self._utagger_dropout
            )

        self._srl_tagger.l_weight = 1.0

        self._modules = [self._srl_tagger]

        modules = [self._seqrep, self._srl_tagger]

        self._model = nn.ModuleList(modules)

        if self._gpu:
            print("Detected", torch.cuda.device_count(), "GPUs")
            self._device_ids = [i for i in range(torch.cuda.device_count())]
            self._model.cuda()

        return self

    def train(
        self,
        filename,
        eval_steps=100,
        decay_evals=5,
        decay_times=0,
        decay_ratio=0.5,
        dev=None,
        save_prefix=None,
        **kwargs
    ):
        train_graphs = DataProcessor(filename, self, self._model, baseline=self._baseline)
        train_loader = InfiniteDataLoader(
            train_graphs,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=DataCollate(self, train=True),
        )
        dev_graphs = DataProcessor(dev, self, self._model, baseline=self._baseline)

        optimizer = AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
            betas=(self._beta1, self._beta2),
            eps=self._epsilon,
            weight_decay=self._weight_decay,
            amsgrad=False,
            warmup=self._warmup,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "max",
            factor=decay_ratio,
            patience=decay_evals,
            verbose=True,
            cooldown=1,
        )

        print("Model")
        for param_tensor in self._model.state_dict():
            print(param_tensor, "\t", self._model.state_dict()[param_tensor].size())
        print("Opt")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        t0 = time.time()
        results, eloss = defaultdict(float), 0.0
        max_dev = 0.0

        for batch_i, batch in enumerate(train_loader):
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw":
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            self._model.train()
            self._model.zero_grad()

            loss = []

            if self._gpu and len(self._device_ids) > 1:
                replicas = nn.parallel.replicate(self._seqrep, self._device_ids)
                inputs = nn.parallel.scatter(batch, self._device_ids)
                replicas = replicas[: len(inputs)]
                seq_features = nn.parallel.parallel_apply(replicas, inputs)
            else:
                seq_features = self._seqrep(batch)

            for module in self._modules:

                if self._gpu and len(self._device_ids) > 1:
                    replicas = nn.parallel.replicate(module, self._device_ids)
                    replicas = [r.calculate_loss for r in replicas]

                    outputs = nn.parallel.parallel_apply(
                        replicas, list(zip(seq_features, inputs))
                    )
                    l, pred = nn.parallel.gather(outputs, 0)
                    l = torch.sum(l)
                else:
                    l, pred = module.calculate_loss(seq_features, batch)

                batch_label = module.batch_label(batch)

                if l is not None:
                    loss.append(l * module.l_weight)
                    module.evaluate(
                        results, self, None, pred, batch_label, mask, train=True
                    )

            loss = sum(loss)
            eloss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
            optimizer.step()

            if batch_i and batch_i % 100 == 0:
                for module in self._modules:
                    module.metrics(results)
                results["loss/loss"] = eloss
                print(batch_i // 100, "{:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                results, eloss = defaultdict(float), 0.0
                t0 = time.time()

            if batch_i and (batch_i % eval_steps == 0):
                results = self.evaluate(dev_graphs)
                if "metrics/SRLDep-f1" in results:
                    performance = results["metrics/SRLDep-f1"]
                else:
                    performance = results["metrics/SRLBIO-f1"]

                results = defaultdict(float)
                scheduler.step(performance)
                if scheduler.in_cooldown:
                    optimizer.state = defaultdict(dict)
                    if decay_times <= 0:
                        break
                    else:
                        decay_times -= 1

                print()
                print(performance)
                print()
                if performance >= max_dev:
                    max_dev = performance
                    if save_prefix:
                        self.save_model("{}model".format(save_prefix))

        return self

    def evaluate(self, data, output_file=None):
        results = defaultdict(float)
        self._model.eval()
        start_time = time.time()

        dev_loader = DataLoader(
            data,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=DataCollate(self, train=False),
        )

        for batch in dev_loader:
            graphs = [data.graphs[idx] for idx in batch["graphidx"]]
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw":
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            if self._gpu and len(self._device_ids) > 1:
                replicas = nn.parallel.replicate(self._seqrep, self._device_ids)
                inputs = nn.parallel.scatter(batch, self._device_ids)
                replicas = replicas[: len(inputs)]
                seq_features = nn.parallel.parallel_apply(replicas, inputs)
            else:
                seq_features = self._seqrep(batch)

            for module in self._modules:
                batch_label = module.batch_label(batch)
                if self._gpu and len(self._device_ids) > 1:
                    replicas = nn.parallel.replicate(module, self._device_ids)
                    outputs = nn.parallel.parallel_apply(
                        replicas,
                        list(zip([self] * len(self._device_ids), seq_features, inputs)),
                    )
                    pred = nn.parallel.gather(outputs, 0)
                else:
                    pred = module(self, seq_features, batch)
                module.evaluate(
                    results, self, graphs, pred, batch_label, mask, train=False
                )

            if output_file and "pred_SRLBIO" in batch:
                for idx, t in zip(
                    batch["graphidx"], batch["pred_SRLBIO"].cpu().data.numpy()
                ):
                    g = data.graphs[idx]
                    labels = ["O"] + [self._ilabel[t[i]] for i in range(1, len(g.sent.nodes))]
                    spans = get_span_from_bio(labels)
                    g.pred_spans = spans

        decode_time = time.time() - start_time
        results["speed/speed"] = len(data) / decode_time

        for module in self._modules:
            module.metrics(results)

        print(results)

        if output_file:
            write_conll(output_file, data.sents)

        return results

    def finish(self, **kwargs):
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    fire.Fire(SRLParser)
