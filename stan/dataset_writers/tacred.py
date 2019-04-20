from typing import List, Dict, Any

from stan.annotators import AnnotatedInstance
from stan.dataset_writers.dataset_writer import DatasetWriter


class TacredDatasetWriter(DatasetWriter):
    FIELD_MAP = {
        "tokens": "token",
        "label": "relation",
        "id": "id",
        "doc_id": "docid",
        "ner": "stanford_ner",
        "pos": "stanford_pos",
        "dep_head": "stanford_head",
        "dep_rel": "stanford_deprel",
        "subject_type": "subj_type",
        "object_type": "obj_type",
        "subject_start": "subj_start",
        "subject_end": "subj_end",
        "object_start": "obj_start",
        "object_end": "obj_end",
    }

    def _convert_instances(
            self, instances: List[AnnotatedInstance]) -> List[Dict[str, Any]]:
        expected_fields = set(self.FIELD_MAP.values())

        mapped_instances = []
        for instance in super()._convert_instances(instances):
            mapped_instance = {
                self.FIELD_MAP[name]: val for name, val in instance.items()}

            assert set(mapped_instance.keys()) == expected_fields

            mapped_instances.append(mapped_instance)

        return mapped_instances
