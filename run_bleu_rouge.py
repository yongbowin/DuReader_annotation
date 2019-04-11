from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import normalize
from utils import compute_bleu_rouge
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# /DATA/disk1/wangyongbo/lic2019/DuReader/official_data/extracted/results_dev
BASE_PATH_PRED = "/DATA/disk1/wangyongbo/lic2019/DuReader/official_data/extracted/"
BASE_PATH_REF = "/DATA/disk1/wangyongbo/lic2019/DuReader/data/extracted/"


def run_eval():
    """
    Run eval.
    """
    with open(BASE_PATH_PRED + "results_dev/test_result_merge_best_rm_dev.json", "r") as f1:
        pred_lines = f1.readlines()

    # search dev
    with open(BASE_PATH_REF + "devset/search.dev.json", "r") as f2:
        ref_lines_search = f2.readlines()

    # zhidao dev
    with open(BASE_PATH_REF + "devset/zhidao.dev.json", "r") as f3:
        ref_lines_zhidao = f3.readlines()

    pred_answers, ref_answers = [], []
    for line in pred_lines:
        sample = json.loads(line)  # type is <class 'dict'>
        pred_answers.append(sample)

    for line in ref_lines_search:
        sample = json.loads(line)  # type is <class 'dict'>
        ref_answers.append(sample)

    for line in ref_lines_zhidao:
        sample = json.loads(line)  # type is <class 'dict'>
        ref_answers.append(sample)

    # compute the bleu and rouge scores if reference answers is provided
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None

    return bleu_rouge


bleu_rouge = run_eval()

logger.info('Dev eval result: {}'.format(bleu_rouge))
