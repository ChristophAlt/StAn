import os

from tests import FIXTURES_ROOT

from stan.dataset_readers.json import JsonDatasetReader


def test_read():
    path = os.path.join(FIXTURES_ROOT, "test_data.json")
    reader = JsonDatasetReader()

    instances = reader.read(path)
    assert len(instances) == 2

    instance = instances[0]
    metadata = instance.metadata

    assert instance.text == "This is the first test text."

    assert list(metadata.keys()) == ["label", "another_field"]

    assert metadata["label"] == "label1"
