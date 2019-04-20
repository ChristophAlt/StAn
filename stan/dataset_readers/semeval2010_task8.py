from typing import List

import re

from stan.dataset_readers import DatasetReader, Instance
from stan.dataset_readers.utils import remove_argument_markers


class SemEval2010Task8DatasetReader(DatasetReader):

    def _to_instance(self, raw: List[str]) -> Instance:
        assert len(raw) == 3

        raw_id, raw_text = raw[0].split("\t")
        label = raw[1]
        id_ = int(raw_id)
        raw_text = raw_text.strip('"')

        # Some special cases (e.g. missing spaces before entity marker)
        if id_ in [213, 4612, 6373, 8411, 9867]:
            raw_text = raw_text.replace("<e2>", " <e2>")
        if id_ in [2740, 4219, 4784]:
            raw_text = raw_text.replace("<e1>", " <e1>")
        if id_ == 9256:
            raw_text = raw_text.replace("log- jam", "log-jam")
        # necessary if returned tokens must be whitespace tokenizeable
        if id_ in [2609, 7589]:
            raw_text = raw_text.replace("1 1/2", "1-1/2")
        if id_ == 10591:
            raw_text = raw_text.replace("1 1/4", "1-1/4")
        if id_ == 10665:
            raw_text = raw_text.replace("6 1/2", "6-1/2")

        raw_text = re.sub(r"[?!.](?!$)", "", raw_text)
        text = remove_argument_markers(raw_text)

        metadata = dict(label=label, id=id_, doc_id=id_, raw_text=raw_text)

        return Instance(text, metadata)

    def read(self, path: str) -> List[Instance]:
        instances = []

        with open(path, "r") as input_file:
            raw = []  # type: List[str]
            for line in input_file:
                line = line.strip()

                if not line:
                    instances.append(self._to_instance(raw))
                    raw = []
                    continue

                raw.append(line)

        return instances
