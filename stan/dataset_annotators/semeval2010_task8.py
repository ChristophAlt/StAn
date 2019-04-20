from typing import List

import logging

from stan.dataset_readers import Instance
from stan.annotators import Annotator, AnnotatedInstance
from stan.dataset_readers.utils import (
    arguments_from_text, argument_spans_from_tokens, use_closest_span,
    argument_types_from_label,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Semeval2010Task8Annotator(Annotator):

    def __init__(self, annotator: Annotator, n_jobs: int = 1) -> None:
        super().__init__(n_jobs)
        self.annotator = annotator

    @staticmethod
    def _add_annotations(instance: AnnotatedInstance) -> AnnotatedInstance:
        assert instance.metadata is not None

        raw_text = instance.metadata.pop("raw_text")
        label = instance.metadata["label"]

        subj_type, obj_type = argument_types_from_label(label)
        subj_arg, obj_arg, subj_loc, obj_loc = arguments_from_text(raw_text)

        subj_spans = argument_spans_from_tokens(subj_arg, instance.tokens)
        obj_spans = argument_spans_from_tokens(obj_arg, instance.tokens)

        subj_span = use_closest_span(subj_loc, subj_spans)
        obj_span = use_closest_span(obj_loc, obj_spans)

        assert subj_span is not None
        assert obj_span is not None

        subj_start, subj_end = subj_span
        obj_start, obj_end = obj_span

        if len(subj_spans) > 1:
            id_ = instance.metadata["id"]
            logger.debug(
                f"[{id_}] Multiple spans '{subj_spans}' for subject '{subj_arg}'."
                + f"Using '({subj_start}, {subj_end})'"
            )
            logger.debug("Raw text: %s", raw_text)

        if len(obj_spans) > 1:
            id_ = instance.metadata["id"]
            logger.debug(
                f"[{id_}] Multiple spans '{obj_spans}' for object '{obj_arg}'."
                + f"Using '({obj_start}, {obj_end})'"
            )
            logger.debug("Raw text: %s", raw_text)

        annotations = instance.annotations
        annotations["subject_type"] = subj_type
        annotations["object_type"] = obj_type
        annotations["subject_start"] = subj_start
        annotations["subject_end"] = subj_end
        annotations["object_start"] = obj_start
        annotations["object_end"] = obj_end

        return instance

    def annotate(self, instance: Instance) -> AnnotatedInstance:
        return self._add_annotations(self.annotator.annotate(instance))
