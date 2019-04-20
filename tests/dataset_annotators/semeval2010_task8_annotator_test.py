import os
import logging
import json
import stanfordcorenlp

from tests import FIXTURES_ROOT

from stan.annotators.corenlp import CoreNlpAnnotator
from stan.dataset_annotators.semeval2010_task8 import Semeval2010Task8Annotator
from stan.dataset_readers.semeval2010_task8 import SemEval2010Task8DatasetReader


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
        expected_annotations = json.load(f)

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

    corenlp_annotator = CoreNlpAnnotator(
        path_or_host="path_or_host", n_jobs=1, verbose=True)

    annotator = Semeval2010Task8Annotator(corenlp_annotator)

    annotated_instances = annotator.annotate_instances(instances)

    assert len(annotated_instances) == 2

    assert not corenlp_results

    instance = annotated_instances[0]
    annotations = instance.annotations
    metadata = instance.metadata

    expected_annotation = expected_annotations[0]

    assert metadata["label"] == "Other"

    assert expected_annotation["subject_type"] == annotations["subject_type"]
    assert expected_annotation["object_type"] == annotations["object_type"]
    assert expected_annotation["subject_start"] == annotations["subject_start"]
    assert expected_annotation["subject_end"] == annotations["subject_end"]
    assert expected_annotation["object_start"] == annotations["object_start"]
    assert expected_annotation["object_end"] == annotations["object_end"]
