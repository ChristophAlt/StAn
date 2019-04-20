import os
import logging
import json
import pytest
import stanfordcorenlp

from tests import FIXTURES_ROOT

from stan.dataset_readers.semeval2010_task8 import SemEval2010Task8DatasetReader
from stan.annotators.corenlp import CoreNlpAnnotator


def test_annotate(monkeypatch):
    dataset_path = os.path.join(FIXTURES_ROOT, "semeval2010_task8.txt")
    reader = SemEval2010Task8DatasetReader()
    instances = reader.read(dataset_path)

    corenlp_results_path = os.path.join(
        FIXTURES_ROOT, "semeval2010_task8_corenlp_results.json")
    with open(corenlp_results_path, "r") as f:
        corenlp_results = list(reversed(json.load(f)))

    annotation_results_path = os.path.join(
        FIXTURES_ROOT, "semeval2010_task8_results.json")
    with open(annotation_results_path, "r") as f:
        annotation_results = json.load(f)

    def init(_, path_or_host, logging_level):
        assert path_or_host == "path_or_host"
        assert logging_level == logging.DEBUG

    def annotate(_, text, properties):
        # assert text == instances[instance_idx].text
        assert properties == dict(
            annotators="tokenize,pos,ner,depparse",
            pipelineLanguage="en",
            outputFormat="json",
        )
        return json.dumps(corenlp_results.pop())

    monkeypatch.setattr(stanfordcorenlp.StanfordCoreNLP, "__init__", init)
    monkeypatch.setattr(stanfordcorenlp.StanfordCoreNLP, "annotate", annotate)

    corenlp = CoreNlpAnnotator(path_or_host="path_or_host", n_jobs=1, verbose=True)

    annotated_instances = corenlp.annotate_instances(instances)

    assert len(annotated_instances) == 2

    assert not corenlp_results

    instance = annotated_instances[0]
    annotations = instance.annotations
    metadata = instance.metadata

    annotation_result = annotation_results[0]

    assert annotation_result["tokens"] == instance.tokens
    assert annotation_result["ner"] == annotations["ner"]
    assert annotation_result["pos"] == annotations["pos"]
    assert annotation_result["dep_head"] == annotations["dep_head"]
    assert annotation_result["dep_rel"] == annotations["dep_rel"]

    assert metadata == instances[0].metadata
