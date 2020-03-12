from typing import List

import logging

from stan.dataset_readers import Instance
from stan.annotators import Annotator, AnnotatedInstance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PlassCorpusAnnotator(Annotator):

    def __init__(self, annotator: Annotator, n_jobs: int = 1) -> None:
        super().__init__(n_jobs)
        self.annotator = annotator

    @staticmethod
    def _add_annotations(instance: AnnotatedInstance) -> AnnotatedInstance:
        assert instance.metadata is not None
        instance.annotations.update(instance.metadata)
        return instance

    def annotate(self, instance: Instance) -> AnnotatedInstance:
        return self._add_annotations(self.annotator.annotate(instance))
