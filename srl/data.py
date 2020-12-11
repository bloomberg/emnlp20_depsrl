import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .io import read_conll


class DataProcessor(Dataset):
    def __init__(self, filename, parser, modules, baseline=False):
        data = read_conll(filename)
        self.sents = data
        if baseline:
            data = [x for g in data for x in g.srl]
            print(
                "Read",
                filename,
                len(self.sents),
                "sentences",
                len(data),
                "predicates",
                sum([len(x.sent.nodes) for x in data]),
                "words",
            )
        else:
            print(
                "Read",
                filename,
                len(self.sents),
                "sentences",
                len(data),
                "predicates",
                sum([len(x.nodes) for x in data]),
                "words",
            )
        self.data = [{"graphidx": i} for i, d in enumerate(data)]
        self.graphs = data
        for m in modules:
            for d, d_ in zip(self.data, data):
                d.update(m.load_data(parser, d_, baseline=baseline))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataCollate:
    def __init__(self, parser, train=True):
        self.parser = parser
        self.train = train

    def __call__(self, data):
        ret = {}
        batch_size = len(data)
        keywords = set(data[0].keys()) - {
            "word",
            "char",
            "norm",
            "xpos",
            "pred",
            "pred_id",
            "graphidx",
            "raw",
        }
        graphidx = [d["graphidx"] for d in data]
        raw = [d["raw"] for d in data]
        words = [d["word"] for d in data]
        chars = [d["char"] for d in data]
        xposs = [d["xpos"] for d in data]
        preds = [d["pred"] for d in data]
        pred_id = torch.LongTensor([d["pred_id"] for d in data])

        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        max_seq_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
        xpos_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
        pred_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
        mask = torch.zeros((batch_size, max_seq_len)).byte()
        mask_h = torch.zeros((batch_size, max_seq_len)).byte()
        for idx, (seq, xpos, pred, seqlen) in enumerate(
            zip(words, xposs, preds, word_seq_lengths)
        ):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            xpos_seq_tensor[idx, :seqlen] = torch.LongTensor(xpos)
            pred_seq_tensor[idx, :seqlen] = torch.LongTensor(pred)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
            mask_h[idx, :seqlen] = torch.Tensor([1] * seqlen)
        mask[:, 0] = 0

        for keyword in keywords:
            label_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
            labels = [d[keyword] for d in data]
            for idx, (label, seqlen) in enumerate(zip(labels, word_seq_lengths)):
                label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            ret[keyword] = label_seq_tensor

        word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        word_seq_tensor = word_seq_tensor[word_perm_idx]
        xpos_seq_tensor = xpos_seq_tensor[word_perm_idx]
        pred_seq_tensor = pred_seq_tensor[word_perm_idx]
        pred_id_seq_tensor = pred_id[word_perm_idx]
        graphidx = [graphidx[i] for i in word_perm_idx]
        raw = [raw[i] for i in word_perm_idx]

        for keyword in keywords:
            ret[keyword] = ret[keyword][word_perm_idx]

        mask = mask[word_perm_idx]
        mask_h = mask_h[word_perm_idx]
        ### deal with char
        pad_chars = [
            chars[idx] + [[0]] * (max_seq_len - len(chars[idx]))
            for idx in range(len(chars))
        ]
        length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
        max_word_len = max(map(max, length_list))
        char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
        char_seq_lengths = torch.LongTensor(length_list)
        for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
            for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
                char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

        char_seq_tensor = char_seq_tensor[word_perm_idx].view(
            batch_size * max_seq_len, -1
        )
        char_seq_lengths = char_seq_lengths[word_perm_idx].view(
            batch_size * max_seq_len
        )
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = word_perm_idx.sort(0, descending=False)

        if self.parser._edims > 0:
            norms = [d["norm"] for d in data]
            emb_tensor = np.zeros(
                (batch_size, max_seq_len, self.parser._edims), dtype=np.float32
            )
            for i in range(batch_size):
                for j, n in enumerate(norms[i]):
                    if n in self.parser._external_mappings:
                        emb_tensor[i, j] = self.parser._external_embeddings[
                            self.parser._external_mappings[n]
                        ]
            emb_tensor = torch.from_numpy(emb_tensor)[word_perm_idx]
            ret["emb"] = emb_tensor

        ret.update(
            {
                "graphidx": graphidx,
                "raw": raw,
                "word": word_seq_tensor,
                "word_length": word_seq_lengths,
                "word_recover": word_seq_recover,
                "char": char_seq_tensor,
                "char_length": char_seq_lengths,
                "char_recover": char_seq_recover,
                "xpos": xpos_seq_tensor,
                "pred": pred_seq_tensor,
                "pred_id": pred_id_seq_tensor,
                "mask": mask,
                "mask_h": mask_h,
            }
        )

        return ret


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
