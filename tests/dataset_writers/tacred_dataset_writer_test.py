import os
import logging
import json
import stanfordcorenlp

from tests import FIXTURES_ROOT

from stan.annotators.corenlp import CoreNlpAnnotator
from stan.dataset_annotators.semeval2010_task8 import Semeval2010Task8Annotator
from stan.dataset_readers.semeval2010_task8 import SemEval2010Task8DatasetReader
from stan.dataset_writers.tacred import TacredDatasetWriter


def test_write(monkeypatch, tmpdir):
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
        return json.dumps(corenlp_results.pop())

    monkeypatch.setattr(stanfordcorenlp.StanfordCoreNLP, "__init__", init)
    monkeypatch.setattr(stanfordcorenlp.StanfordCoreNLP, "annotate", annotate)

    corenlp_annotator = CoreNlpAnnotator(
        path_or_host="path_or_host", n_jobs=1, verbose=True)

    annotator = Semeval2010Task8Annotator(corenlp_annotator)

    annotated_instances = annotator.annotate_instances(instances)

    assert len(annotated_instances) == 2

    file = tmpdir.join("output.json")

    writer = TacredDatasetWriter(fmt="json")
    writer.write(file.strpath, annotated_instances)

    written_annotations = json.loads(file.read())

    assert len(written_annotations) == 2

    expected_annotation = expected_annotations[0]
    written_annotation = written_annotations[0]

    assert expected_annotation["id"] == written_annotation["id"]
    assert expected_annotation["doc_id"] == written_annotation["docid"]
    assert expected_annotation["label"] == written_annotation["relation"]
    assert expected_annotation["subject_type"] == written_annotation["subj_type"]
    assert expected_annotation["object_type"] == written_annotation["obj_type"]
    assert expected_annotation["subject_start"] == written_annotation["subj_start"]
    assert expected_annotation["subject_end"] == written_annotation["subj_end"]
    assert expected_annotation["object_start"] == written_annotation["obj_start"]
    assert expected_annotation["object_end"] == written_annotation["obj_end"]

    assert expected_annotation["ner"] == written_annotation["stanford_ner"]
    assert expected_annotation["pos"] == written_annotation["stanford_pos"]
    assert expected_annotation["dep_head"] == written_annotation["stanford_head"]
    assert expected_annotation["dep_rel"] == written_annotation["stanford_deprel"]
