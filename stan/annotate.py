from typing import Tuple, Optional

import os
import random
import logging

from stan.dataset_readers.semeval2010_task8 import SemEval2010Task8DatasetReader
from stan.dataset_writers import DatasetWriter, TacredDatasetWriter
from stan.annotators.corenlp import CoreNlpAnnotator
from stan.dataset_annotators.semeval2010_task8 import Semeval2010Task8Annotator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def split_files_for_dataset(dataset: str, path: str) -> Tuple[str, Optional[str], str]:
    if dataset == "semeval2010task8":
        train_file = os.path.join(
            path, "SemEval2010_task8_training", "TRAIN_FILE.TXT")
        val_file = None
        test_file = os.path.join(
            path, "SemEval2010_task8_testing_keys", "TEST_FILE_FULL.TXT")

    return train_file, val_file, test_file


def annotate(
        input_dir: str,
        output_dir: str,
        dataset: str,
        corenlp: str,
        output_format: str,
        shuffle: bool,
        validation_size: float,
        seed: int,
        n_jobs: int,
        debug: bool,
        ) -> None:

    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    train_file, val_file, test_file = split_files_for_dataset(dataset, input_dir)

    dataset_reader = {
        "semeval2010task8": SemEval2010Task8DatasetReader,
    }[dataset]()

    dataset_writer = {
        "tacred": TacredDatasetWriter(format="json"),
        "json": DatasetWriter(format="json"),
        "jsonl": DatasetWriter(format="jsonl"),
    }[output_format]

    train_instances = dataset_reader.read(train_file)
    test_instances = dataset_reader.read(test_file)

    if shuffle:
        random.shuffle(train_instances)

    if val_file is not None:
        val_instances = dataset_reader.read(val_file)
    elif validation_size > 0:
        train_val_instances = train_instances
        split_idx = int(len(train_val_instances) * validation_size)
        train_instances = train_val_instances[split_idx:]
        val_instances = train_val_instances[:split_idx]
    else:
        val_instances = None

    base_annotator = CoreNlpAnnotator(
        path_or_host=corenlp, n_jobs=n_jobs, verbose=debug)

    annotator = {
        "semeval2010task8": Semeval2010Task8Annotator,
    }[dataset](base_annotator)

    try:
        logger.info("Annotate train instances:")
        annotated_train_instances = annotator.annotate_instances(train_instances)

        if val_instances:
            logger.info("Annotate validation instances:")
            annotated_val_instances = annotator.annotate_instances(val_instances)

        logger.info("Annotate test instances:")
        annotated_test_instances = annotator.annotate_instances(test_instances)
    finally:
        base_annotator.cleanup()

    for filename, split_instances in zip(
        ["train.json", "val.json", "test.json"],
        [annotated_train_instances, annotated_val_instances, annotated_test_instances],
    ):
        if split_instances is not None:
            output_file = os.path.join(output_dir, filename)
            dataset_writer.write(output_file, split_instances)
