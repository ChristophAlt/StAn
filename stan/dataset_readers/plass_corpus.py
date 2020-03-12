from typing import List

import json

from stan.dataset_readers import DatasetReader, Instance


class PlassCorpusDatasetReader(DatasetReader):

    def read(self, path: str) -> List[Instance]:
        instances = []

        with open(path, "r") as input_file:
            for line in input_file:
                example = json.loads(line)

                id_ = example["id"]
                label = example["label"]
                tokens = example["tokens"]
                subject_type = example["type"][0]
                object_type = example["type"][1]
                subject_start, subject_end = example["entities"][0]
                object_start, object_end = example["entities"][1]

                metadata = dict(label=label,
                                id=id_,
                                doc_id=id_,
                                subject_type=subject_type,
                                object_type=object_type,
                                subject_start=subject_start,
                                subject_end=subject_end,
                                object_start=object_start,
                                object_end=object_end)
                instances.append(Instance(" ".join(tokens), metadata))

        return instances
