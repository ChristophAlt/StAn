from typing import List, Dict, Any

import json

from stan.annotators import AnnotatedInstance


class DatasetWriter:
    SUPPORTED_FORMATS = ["json", "jsonl"]

    def __init__(self, fmt: str) -> None:
        if fmt.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Format '{fmt}' not supported.")
        self.format = fmt.lower()

    def _convert_instances(
            self, instances: List[AnnotatedInstance]) -> List[Dict[str, Any]]:
        converted_instances = []
        for instance in instances:
            converted_instance = dict(tokens=instance.tokens)
            converted_instance.update(instance.annotations)
            if instance.metadata:
                converted_instance.update(instance.metadata)
            converted_instances.append(converted_instance)
        return converted_instances

    def write(self, path: str, instances: List[AnnotatedInstance]) -> None:
        converted_instances = self._convert_instances(instances)

        with open(path, "w") as out_f:
            if self.format == "json":
                json.dump(converted_instances, out_f)
            elif self.format == "jsonl":
                for instance in converted_instances:
                    out_f.write(json.dumps(instance) + "\n")
