from typing import List

import logging
import json

from stanfordcorenlp import StanfordCoreNLP
from stan.dataset_readers import Instance
from stan.annotators import Annotator, AnnotatedInstance


class CoreNlpAnnotator(Annotator):
    def __init__(self, path_or_host: str, n_jobs: int = 1, verbose: bool = False):
        super().__init__(n_jobs)
        corenlp_log_level = logging.DEBUG if verbose else logging.WARNING
        self.corenlp = StanfordCoreNLP(path_or_host, logging_level=corenlp_log_level)
        self.n_jobs = n_jobs

    def annotate(self, instance: Instance) -> AnnotatedInstance:
        props = dict(
            annotators="tokenize,pos,ner,depparse",
            pipelineLanguage="en",
            outputFormat="json",
        )
        corenlp_result = json.loads(
            self.corenlp.annotate(instance.text, properties=props))

        sents = corenlp_result["sentences"]
        tokens = [token["word"] for sentence in sents for token in sentence["tokens"]]
        ner = [token["ner"] for sentence in sents for token in sentence["tokens"]]
        pos = [token["pos"] for sentence in sents for token in sentence["tokens"]]

        dep_parse = sorted(
            [
                (dep["dep"], dep["governor"], dep["dependent"])
                for sentence in sents
                for dep in sentence["basicDependencies"]
            ],
            key=lambda x: x[2],
        )

        dep_head = [p[1] for p in dep_parse]
        dep_rel = [p[0] for p in dep_parse]

        lengths = map(len, [tokens, ner, pos, dep_head, dep_rel])
        assert len(set(lengths)) == 1

        annotations = dict(ner=ner, pos=pos, dep_head=dep_head, dep_rel=dep_rel)

        return AnnotatedInstance(tokens, annotations, instance.metadata)

    def cleanup(self) -> None:
        self.corenlp.close()
