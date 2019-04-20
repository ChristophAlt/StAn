from typing import List

import json

from stan.dataset_readers import DatasetReader, Instance


class JsonDatasetReader(DatasetReader):

    def read(self, path: str) -> List[Instance]:
        instances = []
        with open(path, "r") as json_file:
            data = json.load(json_file)
            for example in data:
                text = example.pop("text")
                instances.append(Instance(text, example))

        return instances
