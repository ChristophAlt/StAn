from typing import List, Tuple, Optional

import re


def arguments_from_text(text: str) -> Tuple[str, str, int, int]:
    arg1 = re.findall(r"e1>(.*)</e1", text)
    arg2 = re.findall(r"e2>(.*)</e2", text)

    arg1_location = [i for i, token in enumerate(text.split(" ")) if "<e1>" in token]
    arg2_location = [i for i, token in enumerate(text.split(" ")) if "<e2>" in token]

    assert len(arg1) == 1
    assert len(arg2) == 1
    assert len(arg1_location) == 1
    assert len(arg2_location) == 1

    return arg1[0], arg2[0], arg1_location[0], arg2_location[0]


def argument_types_from_label(label: str) -> Tuple[str, str]:
    subj_type, obj_type = "Other", "Other"

    if label != "Other":
        relation, order = label.split("(")
        if order.endswith(",e2)"):
            subj_type = relation.split("-")[0]
            obj_type = relation.split("-")[1]
        else:
            subj_type = relation.split("-")[1]
            obj_type = relation.split("-")[0]

    return subj_type, obj_type


def argument_spans_from_tokens(
        argument: str,
        tokens: List[str]) -> List[Tuple[int, int]]:

    def find_sub_list(sublst, lst):
        results = []
        sll = len(sublst)
        for ind in (i for i, e in enumerate(lst) if e == sublst[0]):
            if lst[ind: ind + sll] == sublst:
                results.append((ind, ind + sll - 1))

        return results

    argument_tokens = argument.split(" ")
    matching_spans = find_sub_list(argument_tokens, tokens)

    assert len(matching_spans) >= 1

    return matching_spans


def use_closest_span(
        location: int, spans: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    assert spans

    if len(spans) == 1:
        return spans[0]

    closest_span = None
    lowest_dist = float("inf")
    for span in spans:
        start, end = span
        dist = min(abs(location - start), abs(location - end))
        if dist < lowest_dist:
            closest_span = span
            lowest_dist = dist

    return closest_span


def remove_argument_markers(text: str) -> str:
    all_tags = ["<e1>", "</e1>", "<e2>", "</e2>"]

    for tag in all_tags:
        text = text.replace(tag, "")

    assert not [tag for tag in all_tags if tag in text]

    return text
