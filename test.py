# -*- coding: utf-8 -*-

from srl.parser import SRLParser
from srl.data import DataProcessor

parser = SRLParser(baseline=False)

parser.load_model("./path/to/model", batch_size=16, gpu=False)
parser.load_embeddings("./embeddings/glove.6b.100")

parser._model.eval()

graphs = DataProcessor("./path/to/data", parser, parser._model)

result = parser.evaluate(graphs, "./path/to/output")

print(result)
