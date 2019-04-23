import os
import json

from tests import FIXTURES_ROOT

from stan.dataset_writers.tacred import TacredDatasetWriter


def test_write(annotated_instances, tmpdir):
    annotation_results_path = os.path.join(
        FIXTURES_ROOT, "semeval2010_task8_results.json")
    with open(annotation_results_path, "r") as f:
        expected_annotations = json.load(f)

    assert len(annotated_instances) == 2

    file = tmpdir.join("output.json")

    writer = TacredDatasetWriter(fmt="json")
    writer.write(file.strpath, annotated_instances)

    written_annotations = json.loads(file.read())

    assert len(written_annotations) == 2

    expected_annotation = expected_annotations[0]
    written_annotation = written_annotations[0]

    print(expected_annotation)
    print(written_annotation)

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
