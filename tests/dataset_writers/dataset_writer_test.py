import os
import json

from tests import FIXTURES_ROOT

from stan.dataset_writers.dataset_writer import DatasetWriter


def test_write_json(annotated_instances, tmpdir):
    annotation_results_path = os.path.join(
        FIXTURES_ROOT, "semeval2010_task8_results.json")
    with open(annotation_results_path, "r") as f:
        expected_annotations = json.load(f)

    file = tmpdir.join("output.json")

    writer = DatasetWriter(fmt="json")
    writer.write(file.strpath, annotated_instances)

    written_annotations = json.loads(file.read())

    assert len(written_annotations) == 2
    assert expected_annotations == written_annotations


def test_write_jsonl(annotated_instances, tmpdir):
    annotation_results_path = os.path.join(
        FIXTURES_ROOT, "semeval2010_task8_results.json")
    with open(annotation_results_path, "r") as f:
        expected_annotations = json.load(f)

    file = tmpdir.join("output.jsonl")

    writer = DatasetWriter(fmt="jsonl")
    writer.write(file.strpath, annotated_instances)

    written_annotations = [json.loads(line) for line in file.readlines()]

    assert len(written_annotations) == 2
    assert expected_annotations == written_annotations
