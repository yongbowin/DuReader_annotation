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


def prepare_data():
    """
    On DuReader 2.0 `devset`.

    Prepare data for eval,
        1.overall eval.
        2.eval `search` and `zhidao`, respectively.
    """
    # -------------------------- for pred answers (start) --------------------------
    with open(BASE_PATH_PRED + "results_dev/test_result_merge_best_rm.json", "r") as f1:
        pred_lines = f1.readlines()

    with open(BASE_PATH_PRED + "results_dev/test_result_search_best.json", "r") as f1_1:
        pred_lines_search = f1_1.readlines()

    with open(BASE_PATH_PRED + "results_dev/test_result_zhidao_best.json", "r") as f1_2:
        pred_lines_zhidao = f1_2.readlines()
    # -------------------------- for pred answers (end) --------------------------

    # -------------------------- for ref answers (start) --------------------------
    # search dev
    with open(BASE_PATH_REF + "devset/search.dev.json", "r") as f2:
        ref_lines_search = f2.readlines()

    # zhidao dev
    with open(BASE_PATH_REF + "devset/zhidao.dev.json", "r") as f3:
        ref_lines_zhidao = f3.readlines()
    # -------------------------- for ref answers (end) --------------------------

    pred_answers, ref_answers = [], []
    pred_answers_search, pred_answers_zhidao, ref_answers_search, ref_answers_zhidao = [], [], [], []
    for line in pred_lines:
        sample = json.loads(line)  # type is <class 'dict'>
        pred_answers.append(sample)

    for line in pred_lines_search:
        sample = json.loads(line)  # type is <class 'dict'>
        pred_answers_search.append(sample)

    for line in pred_lines_zhidao:
        sample = json.loads(line)  # type is <class 'dict'>
        pred_answers_zhidao.append(sample)

    for line in ref_lines_search:
        sample = json.loads(line)  # type is <class 'dict'>
        ref_answers.append(sample)
        ref_answers_search.append(sample)

    for line in ref_lines_zhidao:
        sample = json.loads(line)  # type is <class 'dict'>
        ref_answers.append(sample)
        ref_answers_zhidao.append(sample)

    return pred_answers, ref_answers, pred_answers_search, pred_answers_zhidao, ref_answers_search, ref_answers_zhidao


# acquire eval data
pred_answers, ref_answers, pred_answers_search, pred_answers_zhidao, ref_answers_search, ref_answers_zhidao = prepare_data()


def run_eval(pred_answers_list, ref_answers_list):
    """
    Run eval.
    """
    pred_answers = pred_answers_list
    ref_answers = ref_answers_list

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


# run overall
bleu_rouge = run_eval(pred_answers, ref_answers)

# run respectively
bleu_rouge_search = run_eval(pred_answers_search, ref_answers_search)
bleu_rouge_zhidao = run_eval(pred_answers_zhidao, ref_answers_zhidao)

"""On dureader2.0 `dev`
{
    'Bleu-1': 0.5837681460393821, 
    'Bleu-2': 0.5226142125975742, 
    'Bleu-3': 0.48347404434526725, 
    'Bleu-4': 0.45561738017705655, 
    'Rouge-L': 0.5022681316295654
}
"""
logger.info('(Overall) Dev eval result: \n {}'.format(bleu_rouge))
logger.info('(search) Dev eval result: \n {}'.format(bleu_rouge_search))
logger.info('(zhidao) Dev eval result: \n {}'.format(bleu_rouge_zhidao))
