from typing import List, Dict, Any

import json
from stan.dataset_readers import Instance


class DatasetWriter:
    SUPPORTED_FORMATS = ["json", "jsonl"]

    def __init__(self, format: str) -> None:
        if format.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Format '{format}' not supported.")
        self.format = format.lower()

    def _convert_instances(self, instances: List[Instance]) -> List[Dict[str, Any]]:
        converted_instances = []
        for instance in instances:
            converted_instance = dict(tokens=instance.tokens)
            converted_instance.update(instance.annotations)
            converted_instance.update(instance.metadata)
            converted_instances.append(converted_instance)
        return converted_instances

    def write(self, path: str, instances: List[Instance]) -> None:
        converted_instances = self._convert_instances(instances)

        with open(path, "w") as out_f:
            if self.format == "json":
                json.dump(converted_instances, out_f)
            elif self.format == "jsonl":
                for instance in converted_instances:
                    out_f.write(json.dumps(instance) + "\n")
