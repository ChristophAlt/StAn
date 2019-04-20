from typing import List, Dict, Tuple, Any

import logging
import os
import re
import json
import random
import copy
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
from stanfordcorenlp import StanfordCoreNLP

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def argument_spans_from_tokens(
    argument: str, tokens: List[str], nlp: StanfordCoreNLP
) -> List[Tuple[int, int]]:
    # logger.info(f"argument: {argument}")
    # logger.info(f"tokens: {tokens}")

    def find_sub_list(sl, l):
        results = []
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind : ind + sll] == sl:
                results.append((ind, ind + sll - 1))

        return results

    # argument_tokens = argument.split(" ")
    at = tokenize(argument, nlp)
    argument_tokens = []
    for t in at:
        if t == "Gimme":
            argument_tokens.extend(["Gim", "me"])
        elif t == "Wanna":
            argument_tokens.extend(["Wan", "na"])
        else:
            argument_tokens.append(t)

    matching_spans = find_sub_list(argument_tokens, tokens)

    # logger.info(f"matching_spans: {matching_spans}")

    assert (
        len(matching_spans) >= 1
    ), f"argument: {argument} | arg-tokens: {argument_tokens} | tokens: {tokens} | spans: {matching_spans}"

    return matching_spans


def use_closest_span(location: int, spans: List[Tuple[int, int]]) -> Tuple[int, int]:
    assert len(spans) > 0

    if len(spans) == 1:
        return spans[0]

    else:
        closest_span = None
        lowest_dist = float("inf")
        for span in spans:
            start, end = span
            dist = min(abs(location - start), abs(location - end))
            if dist < lowest_dist:
                closest_span = span
                lowest_dist = dist

        return closest_span


def annotate_with_nlp(text: str, nlp: StanfordCoreNLP) -> Dict[str, Any]:
    props = dict(
        annotators="tokenize,pos,ner,depparse",
        pipelineLanguage="en",
        outputFormat="json",
    )
    return json.loads(nlp.annotate(text, properties=props))


def tokenize(text: str, nlp: StanfordCoreNLP) -> List[str]:
    props = dict(annotators="tokenize", pipelineLanguage="en", outputFormat="json")
    r = json.loads(nlp.annotate(text, properties=props))
    return [t["word"] for t in r["tokens"]]


# def is_before(idx, indices):
#     return idx < indices[0]


# def clean_tokens(example):
#     tokens = example["tokens"]
#     subj_indices = example["h"][2][0]
#     obj_indices = example["t"][2][0]

#     subj_offset = 0
#     obj_offset = 0

#     cleaned_tokens = []
#     for idx in reversed(range(len(tokens))):
#         token = tokens[idx]
#         if (
#             (token in ["?", "!", "."] and idx < len(tokens) - 1)
#             or token in ["\xa0", " ", ".-"]
#             or "\n" in token
#         ):
#             if is_before(idx, subj_indices):
#                 subj_offset += 1

#             if is_before(idx, obj_indices):
#                 obj_offset += 1

#             continue

#         cleaned_tokens.append(token)

#     new_example = copy.deepcopy(example)
#     new_example["tokens"] = list(reversed(cleaned_tokens))
#     new_example["h"][2][0] = [i - subj_offset for i in new_example["h"][2][0]]
#     new_example["t"][2][0] = [i - obj_offset for i in new_example["t"][2][0]]

#     return new_example


def cleanest_tokens(tokens: List[str]) -> List[str]:
    cleaned_tokens = []
    for token in tokens:
        if token == "–":
            cleaned_tokens.append("--")
        else:
            cleaned_tokens.append(token)

    return cleaned_tokens


def raw_to_example(
    raw_example: Tuple[str, int, Dict[str, Any]], nlp: StanfordCoreNLP
) -> Dict[str, Any]:
    relation, id_, example = raw_example

    # example = clean_tokens(example)

    raw_tokens = example["tokens"]

    raw_text = " ".join(raw_tokens)
    # raw_text = ftfy.fix_text(raw_text)
    text = re.sub(r"[?!.](?!$)", "", raw_text)

    subj_indices = example["h"][2][0]
    obj_indices = example["t"][2][0]

    subj_arg = " ".join(cleanest_tokens([example["tokens"][i] for i in subj_indices]))
    obj_arg = " ".join(cleanest_tokens([example["tokens"][i] for i in obj_indices]))
    subj_arg = re.sub(r"[?!.]", "", subj_arg)
    obj_arg = re.sub(r"[?!.]", "", obj_arg)

    subj_loc = subj_indices[0]
    obj_loc = obj_indices[0]

    # subj_arg, obj_arg, subj_loc, obj_loc = arguments_from_text(raw_text)

    # text = remove_argument_markers(raw_text)

    corenlp_result = annotate_with_nlp(text, nlp)

    sentences = corenlp_result["sentences"]

    tokens = [token["word"] for sentence in sentences for token in sentence["tokens"]]

    # tokens = cleanest_tokens(tokens)

    assert len(sentences) == 1, f"Raw: {raw_tokens}, New: {tokens}"
    # assert len(raw_tokens) == len(tokens), f"Raw: {raw_tokens}, New: {tokens}"

    # subj_type, obj_type = argument_types_from_label(label)
    subj_type, obj_type = "NONE", "NONE"

    subj_spans = argument_spans_from_tokens(subj_arg, tokens, nlp)
    obj_spans = argument_spans_from_tokens(obj_arg, tokens, nlp)

    subj_start, subj_end = use_closest_span(subj_loc, subj_spans)
    obj_start, obj_end = use_closest_span(obj_loc, obj_spans)

    # subj_start, subj_end = subj_indices[0], subj_indices[-1]
    # obj_start, obj_end = obj_indices[0], obj_indices[-1]

    assert (
        len(subj_spans) > 0
    ), f"[{id_}] Raw: {raw_tokens}, New: {tokens}, [{subj_indices}, {subj_start} - {subj_end}]"
    assert (
        len(obj_spans) > 0
    ), f"[{id_}] Raw: {raw_tokens}, New: {tokens}, [{obj_indices}, {obj_start} - {obj_end}]"

    # assert subj_end < len(
    #     tokens
    # ), f"[{id_}] Raw: {raw_tokens}, New: {tokens}, [{subj_indices}, {subj_start} - {subj_end}]"
    # assert obj_end < len(
    #     tokens
    # ), f"[{id_}] Raw: {raw_tokens}, New: {tokens}, [{subj_indices}, {subj_start} - {subj_end}]"

    if len(subj_spans) > 1:
        logger.debug(
            f"[{id_}] Multiple spans '{subj_spans}' for subject '{subj_arg}'."
            + f"Using '({subj_start}, {subj_end})'"
        )
        logger.debug(f"Raw text: {raw_text}")

    if len(obj_spans) > 1:
        logger.debug(
            f"[{id_}] Multiple spans '{obj_spans}' for object '{obj_arg}'."
            + f"Using '({obj_start}, {obj_end})'"
        )
        logger.debug(f"Raw text: {raw_text}")

    ner = [token["ner"] for sentence in sentences for token in sentence["tokens"]]
    pos = [token["pos"] for sentence in sentences for token in sentence["tokens"]]

    dep_parse = sorted(
        [
            (dep["dep"], dep["governor"], dep["dependent"])
            for s in sentences
            for dep in s["basicDependencies"]
        ],
        key=lambda x: x[2],
    )

    dep_head = [p[1] for p in dep_parse]
    dep_rel = [p[0] for p in dep_parse]

    lengths = map(len, [tokens, ner, pos, dep_head, dep_rel])
    assert len(set(lengths)) == 1

    return (
        relation,
        dict(
            id=id_,
            docid=id_,
            token=tokens,
            subj_type=subj_type,
            obj_type=obj_type,
            subj_start=subj_start,
            subj_end=subj_end,
            obj_start=obj_start,
            obj_end=obj_end,
            stanford_ner=ner,
            stanford_pos=pos,
            stanford_head=dep_head,
            stanford_deprel=dep_rel,
            relation=relation,
        ),
    )


def convert_with_corenlp_annotation(
    path: str, nlp: StanfordCoreNLP, n_jobs: int
) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        dataset = json.load(f)

    id_ = 1
    raw_examples = []
    for relation, examples in dataset.items():
        for example in examples:
            raw_examples.append((relation, id_, example))
            id_ += 1

    examples = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(raw_to_example)(raw_example, nlp) for raw_example in tqdm(raw_examples)
    )

    grouped_examples = defaultdict(list)
    for relation, example in examples:
        grouped_examples[relation].append(example)

    return grouped_examples


def annotate_fewrel(
    input_path: str,
    output_path: str,
    corenlp: str,
    shuffle: bool,
    validation_size: float,
    seed: int,
    n_jobs: int,
    debug: bool,
) -> None:
    random.seed(seed)

    train_file = os.path.join(input_path, "train.json")
    validation_file = os.path.join(input_path, "val.json")

    with StanfordCoreNLP(
        corenlp, logging_level=(logging.DEBUG if debug else logging.INFO)
    ) as nlp:
        train_validation_examples = convert_with_corenlp_annotation(
            train_file, nlp, n_jobs
        )

        validation_examples = convert_with_corenlp_annotation(
            validation_file, nlp, n_jobs
        )

    # assert len(train_validation_examples) == 44800
    # assert len(validation_examples) == 11200

    if shuffle:
        random.shuffle(train_validation_examples)

    if validation_size > 0:
        split_idx = int(len(train_validation_examples) * validation_size)
        train_examples = train_validation_examples[split_idx:]
        validation_examples = train_validation_examples[:split_idx]
        test_examples = validation_examples
    else:
        train_examples = train_validation_examples
        test_examples = None

    if train_examples:
        logger.info(f"Number of examples in train: {len(train_examples)}")
    if validation_examples:
        logger.info(f"Number of examples in validation: {len(validation_examples)}")
    if test_examples:
        logger.info(f"Number of examples in test: {len(test_examples)}")

    for filename, split in zip(
        ["train.json", "val.json", "test.json"],
        [train_examples, validation_examples, test_examples],
    ):
        if split is not None:
            with open(os.path.join(output_path, filename), "w") as out_f:
                json.dump(split, out_f)
