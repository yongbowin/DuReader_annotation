"""
This module finds the most related paragraph of each document according to recall.

1.Filter samples
    If the `metric value` < `threshold`, filter it.
2.Expend samples
    If the `fake_answer` in the same doc, save the best. ==> a sample.
    If the `fake_answer` in the different doc, save them all. ==> several samples.
"""

import sys
import json
from collections import Counter


def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        """
        such as:
        ground_truth_tokens:  ['微信']
        common:  Counter({'微信': 1})
        ground_truth_tokens:  ['分享']
        common:  Counter()
        """
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())  # calculate common words between question and paragraph
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    params: (f1_score, para_tokens, question)
    """

    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        metric_fn: metric function pointer which calculates scores according to corresponding logic.
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    best_score = 0
    best_idx = 0
    scores_for_ground_truths = []
    for idx, ground_truth in enumerate(ground_truths):  # i.e., for a single question token in question_tokens_list
        # for each token in question tokens list, to calculate f1 with para_tokens (para list)
        score = metric_fn(prediction, ground_truth)  # f1_score()
        scores_for_ground_truths.append(score)
        # save the max score `fake_answer` and its `idx`
        if score > best_score:
            best_idx = idx
            best_score = score

    return best_score, best_idx


def find_fake_answer(sample):
    """
    For each document, finds the most related paragraph based on recall,
    then finds a span that maximize the f1_score compared with the gold answers
    and uses this span as a fake answer span
    Args:
        sample: a sample in the dataset
    Returns:
        None
    Raises:
        None
    """
    for doc in sample['documents']:
        most_related_para = -1
        most_related_para_len = 999999
        max_related_score = 0
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(sample['segmented_answers']) > 0:
                related_score = metric_max_over_ground_truths(recall,
                                                              para_tokens,
                                                              sample['segmented_answers'])
            else:
                continue
            if related_score > max_related_score \
                    or (related_score == max_related_score
                        and len(para_tokens) < most_related_para_len):
                most_related_para = p_idx
                most_related_para_len = len(para_tokens)
                max_related_score = related_score
        doc['most_related_para'] = most_related_para

    sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []

    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    for d_idx, doc in enumerate(sample['documents']):
        if not doc['is_selected']:  # TODO: remove or remain?
            continue
        if doc['most_related_para'] == -1:
            doc['most_related_para'] = 0
        most_related_para_tokens = doc['segmented_paragraphs'][doc['most_related_para']][:1000]
        for start_tidx in range(len(most_related_para_tokens)):
            if most_related_para_tokens[start_tidx] not in answer_tokens:
                continue
            for end_tidx in range(len(most_related_para_tokens) - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
                if len(sample['segmented_answers']) > 0:
                    match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                                sample['segmented_answers'])
                else:
                    match_score = 0
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_d_idx = d_idx
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_fake_answer = ''.join(span_tokens)
        """
        each doc save a `fake_answer`
        """
        if best_match_score > 0:
            sample['answer_docs'].append(best_match_d_idx)
            sample['answer_spans'].append(best_match_span)
            sample['fake_answers'].append(best_fake_answer)
            sample['match_scores'].append(best_match_score)


if __name__ == '__main__':
    for line in sys.stdin:
        sample = json.loads(line)
        find_fake_answer(sample)
        print(json.dumps(sample, encoding='utf8', ensure_ascii=False))
