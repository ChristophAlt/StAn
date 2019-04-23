import os

from tests import FIXTURES_ROOT

from stan.dataset_readers.jsonl import JsonlDatasetReader


def test_read():
    path = os.path.join(FIXTURES_ROOT, "test_data.jsonl")
    reader = JsonlDatasetReader()

    instances = reader.read(path)
    assert len(instances) == 2

    instance = instances[0]
    metadata = instance.metadata

    assert instance.text == "This is the first test text."

    assert list(metadata.keys()) == ["label", "another_field"]

    assert metadata["label"] == "label1"
