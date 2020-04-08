from typing import List, Tuple, Optional

import os
import random
import logging

from stan.dataset_readers import (
    Instance, SemEval2010Task8DatasetReader, JsonDatasetReader, JsonlDatasetReader, PlassCorpusDatasetReader)
from stan.dataset_writers import DatasetWriter, TacredDatasetWriter
from stan.annotators.corenlp import CoreNlpAnnotator
from stan.dataset_annotators import Semeval2010Task8Annotator, PlassCorpusAnnotator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _split_files_for_format(
        fmt: str, path: str,
        train_filename: Optional[str] = None,
        val_filename: Optional[str] = None,
        test_filename: Optional[str] = None) -> Tuple[str, Optional[str], str]:

    if fmt == "semeval2010task8":
        train_filename = "SemEval2010_task8_training/TRAIN_FILE.TXT"
        test_filename = "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
        val_filename = None
    elif fmt == "plass_corpus":
        train_filename = train_filename or "train.jsonl"
        val_filename = val_filename or "dev.jsonl"
        test_filename = test_filename or "test.jsonl"
    else:
        train_filename = train_filename or f"train.{fmt}"
        val_filename = val_filename or f"val.{fmt}"
        test_filename = test_filename or f"test.{fmt}"

    train_file = os.path.join(path, train_filename)
    test_file = os.path.join(path, test_filename)
    val_file = None
    if val_filename:
        val_file_path = os.path.join(path, val_filename)
        # in case no validation file exists
        val_file = val_file_path if os.path.isfile(val_file_path) else None

    return train_file, val_file, test_file


def annotate(
        input_dir: str,
        output_dir: str,
        input_format: str,
        corenlp: str,
        output_format: str,
        shuffle: bool,
        validation_size: float,
        seed: int,
        n_jobs: int,
        debug: bool) -> None:

    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    train_file, val_file, test_file = _split_files_for_format(input_format, input_dir)

    dataset_reader = {
        "semeval2010task8": SemEval2010Task8DatasetReader,
        "plass_corpus": PlassCorpusDatasetReader,
        "json": JsonDatasetReader,
        "jsonl": JsonlDatasetReader,
    }[input_format]()

    dataset_writer = {
        "tacred": TacredDatasetWriter(fmt="json"),
        "json": DatasetWriter(fmt="json"),
        "jsonl": DatasetWriter(fmt="jsonl"),
    }[output_format]

    train_instances = dataset_reader.read(train_file)
    test_instances = dataset_reader.read(test_file)

    if shuffle:
        random.shuffle(train_instances)

    val_instances = None  # type: Optional[List[Instance]]
    if val_file is not None:
        val_instances = dataset_reader.read(val_file)
    elif validation_size > 0:
        train_val_instances = train_instances
        split_idx = int(len(train_val_instances) * validation_size)
        train_instances = train_val_instances[split_idx:]
        val_instances = train_val_instances[:split_idx]

    base_annotator = CoreNlpAnnotator(
        path_or_host=corenlp, n_jobs=n_jobs, verbose=debug)

    annotator = {
        "semeval2010task8": Semeval2010Task8Annotator,
        "plass_corpus": PlassCorpusAnnotator,
    }[input_format](base_annotator)

    try:
        logger.info("Annotate train instances:")
        annot_train_instances = annotator.annotate_instances(train_instances)

        if val_instances:
            logger.info("Annotate validation instances:")
            annot_val_instances = annotator.annotate_instances(val_instances)

        logger.info("Annotate test instances:")
        annot_test_instances = annotator.annotate_instances(test_instances)
    finally:
        base_annotator.cleanup()

    file_extension = {
        "tacred": "json",
        "json": "json",
        "jsonl": "jsonl"
    }[output_format]

    for filename, split_instances in zip(
            ["train", "val", "test"],
            [annot_train_instances, annot_val_instances, annot_test_instances]):
        if split_instances is not None:
            output_file = os.path.join(output_dir, f"{filename}.{file_extension}")
            dataset_writer.write(output_file, split_instances)
