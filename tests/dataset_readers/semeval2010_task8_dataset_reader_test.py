import os

from tests import FIXTURES_ROOT

from stan.dataset_readers.semeval2010_task8 import SemEval2010Task8DatasetReader


def test_read():
    path = os.path.join(FIXTURES_ROOT, "semeval2010_task8.txt")
    reader = SemEval2010Task8DatasetReader()

    instances = reader.read(path)
    assert len(instances) == 2

    instance = instances[0]
    metadata = instance.metadata

    assert instance.text == (
        "The staff in the shop are all left-handed "
        "themselves and are happy to demonstrate products, explain why they are "
        "left-handed and give helpful advice to left-handers of all ages."
        )

    assert metadata["raw_text"] == (
        "The <e1>staff</e1> in the <e2>shop</e2> are all left-handed "
        "themselves and are happy to demonstrate products, explain why they are "
        "left-handed and give helpful advice to left-handers of all ages."
        )

    assert metadata["id"] == 2740
    assert metadata["doc_id"] == 2740
    assert metadata["label"] == "Other"
