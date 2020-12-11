from .const import (
    FORM,
    PLEMMA,
    PID,
    DEPHEAD,
    DEPREL,
    PSTART,
)
from .graph import DataInstance, Word, Sentence
from .conversion import srl_forward_v2, get_deptree_spans


def read_conll(filename):
    def get_instance(graph, graphs):
        words = [Word(row[FORM]) for row in graph]
        plemmas = [row[PLEMMA] for row in graph]
        pids = [row[PID] for row in graph]
        heads = [int(row[DEPHEAD]) for row in graph]
        rels = [row[DEPREL] for row in graph]
        lemmaidx = []
        for i, (p, pid) in enumerate(zip(plemmas, pids)):
            if p != "-" and pid != "-":
                lemmaidx.append(i)

        sentence = Sentence(words, heads, rels)

        heads = sentence.heads
        rels = sentence.rels
        words = [x.word for x in sentence.nodes]

        for i, pid in zip(range(PSTART, len(graph[0]) - 1), lemmaidx):
            labels = ["*"] + [line[i] for line in graph]
            sentence.srl.append(DataInstance(sentence, pid + 1, labels))

        depspans, lmost, rmost = get_deptree_spans(heads)
        srlspans = [
            (srl.pred_id, l, r, lab) for srl in sentence.srl for l, r, lab in srl.spans
        ]
        argspans = {(l, r) for v, l, r, lab in srlspans}
        unlabspans = {(v, l, r) for v, l, r, lab in srlspans}

        # Heuristics to fix the tree
        for i in range(1, len(heads)):
            if (
                heads[i] > 0
                and rels[heads[i]] in {"nsubj", "nsubjpass"}
                and rels[i] == "advmod"
                and i < heads[i]
                and words[i].lower() in {"then", "now", "first", "surely", "currently"}
            ):
                heads[i] = heads[heads[i]]

        for i in range(1, len(heads)):
            if (
                heads[i] > 0
                and (heads[heads[i]], lmost[i], rmost[i]) in unlabspans
                and (heads[i], lmost[i], rmost[i]) not in unlabspans
                and (lmost[heads[i]], rmost[heads[i]]) not in argspans
            ):
                if rels[i] in {"prep", "dobj", "advmod", "dep", "vmod"}:
                    heads[i] = heads[heads[i]]

        # fix non-proj
        def non_proj_edges(heads):
            ret = set()
            mins = [min(i, heads[i]) for i in range(len(heads))]
            maxs = [max(i, heads[i]) for i in range(len(heads))]
            for i in range(1, len(heads)):
                for j in range(1, len(heads)):
                    if (
                        mins[i] > mins[j] and mins[i] < maxs[j] and maxs[i] > maxs[j]
                    ) or (
                        mins[j] > mins[i] and mins[j] < maxs[i] and maxs[j] > maxs[i]
                    ):
                        ret.add(i)
                        ret.add(j)
            return ret

        edges = non_proj_edges(heads)
        if len(edges) > 0:
            for i in edges:
                if words[i].lower() in {
                    "what",
                    "when",
                    "where",
                    "how",
                    "which",
                    "who",
                    "that",
                    "whom",
                    "whatever",
                    "whose",
                }:
                    for j in range(i + 1, len(heads)):
                        if heads[j] < i:
                            heads[i] = j
                            break

        srl_forward_v2(sentence)
        graphs.append(sentence)

    file = open(filename, "rb")

    graphs = []
    graph = []

    sent_count = 0
    for line in file:
        line = line.decode("utf-8").strip()

        if len(line):
            graph.append(line.split("\t"))
        else:
            sent_count += 1
            get_instance(graph, graphs)
            graph = []

    if len(graph):
        get_instance(graph, graphs)

    print(
        "Read", sent_count, "sents", sum([len(sent.srl) for sent in graphs]), "predicates"
    )

    file.close()

    return graphs


def write_conll(filename, sents):
    file = open(filename, "w")

    for sent in sents:
        length = len(sent.nodes)
        rows = [["-"] for i in range(length)]
        for graph in sent.srl:
            rows[graph.pred_id][0] = sent.nodes[graph.pred_id].word
            labels = ["*" for i in range(length)]

            for l, r, lab in graph.pred_spans:
                labels[l] = "(" + lab + labels[l]
                labels[r] = labels[r] + ")"

            for i, l in enumerate(labels):
                rows[i].append(l)
        for r in rows[1:]:
            file.write("\t".join(r))
            file.write("\n")
        file.write("\n")

    file.close()
