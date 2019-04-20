from typing import List

import json

from stan.dataset_readers import DatasetReader, Instance


class JsonlDatasetReader(DatasetReader):

    def read(self, path: str) -> List[Instance]:
        instances = []
        with open(path, "r") as jsonl_file:
            for line in jsonl_file:
                example = json.loads(line)
                text = example.pop("text")
                instances.append(Instance(text, example))

        return instances
