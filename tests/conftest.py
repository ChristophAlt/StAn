import pytest
import os
import logging
import json
import stanfordcorenlp

from tests import FIXTURES_ROOT

from stan.annotators.corenlp import CoreNlpAnnotator
from stan.dataset_annotators.semeval2010_task8 import Semeval2010Task8Annotator
from stan.dataset_readers.semeval2010_task8 import SemEval2010Task8DatasetReader


@pytest.fixture
def annotated_instances(monkeypatch):
    dataset_path = os.path.join(FIXTURES_ROOT, "semeval2010_task8.txt")
    reader = SemEval2010Task8DatasetReader()
    instances = reader.read(dataset_path)

    corenlp_results_path = os.path.join(
        FIXTURES_ROOT, "semeval2010_task8_corenlp_results.json")
    with open(corenlp_results_path, "r") as f:
        corenlp_results = list(reversed(json.load(f)))

    def init(_, path_or_host, logging_level):
        assert path_or_host == "path_or_host"
        assert logging_level == logging.DEBUG

    def annotate(_, text, properties):
        assert properties == dict(
            annotators="tokenize,pos,ner,depparse",
            pipelineLanguage="en",
            outputFormat="json",
        )
        return json.dumps(corenlp_results.pop())

    monkeypatch.setattr(stanfordcorenlp.StanfordCoreNLP, "__init__", init)
    monkeypatch.setattr(stanfordcorenlp.StanfordCoreNLP, "annotate", annotate)

    corenlp_annotator = CoreNlpAnnotator(
        path_or_host="path_or_host", n_jobs=1, verbose=True)

    annotator = Semeval2010Task8Annotator(corenlp_annotator)

    annotated_instances = annotator.annotate_instances(instances)

    assert len(annotated_instances) == 2

    return annotated_instances
