#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import json
import copy
from preprocess import metric_max_over_ground_truths, f1_score


def compute_paragraph_score(sample):
    """
    For each paragraph, compute the f1 score compared with the question
    Args:
        sample: a sample in the dataset.
    Returns:
        None
    Raises:
        None
    """
    question = sample["segmented_question"]
    for doc in sample['documents']:
        doc['segmented_paragraphs_scores'] = []
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score,
                        para_tokens,
                        question)
            else:
                related_score = 0.0
            doc['segmented_paragraphs_scores'].append(related_score)


def dup_remove(doc):
    """
    For each document, remove the duplicated paragraphs
    Args:
        doc: a doc in the sample
    Returns:
        bool
    Raises:
        None
    """
    paragraphs_his = {}
    del_ids = []
    para_id = None

    # Others modified in Github.
    # # ----------------- modify start -----------------
    # if 'most_related_para' in doc:  # for trainset and devset
    #     para_id = doc['most_related_para']
    # else:  # for testset
    #     para_id = find_best_question_match(doc, question)
    # # ----------------- modify end -----------------

    if 'most_related_para' in doc:
        para_id = doc['most_related_para']
    doc['paragraphs_length'] = []
    for p_idx, (segmented_paragraph, paragraph_score) in \
        enumerate(zip(doc["segmented_paragraphs"], doc["segmented_paragraphs_scores"])):
        doc['paragraphs_length'].append(len(segmented_paragraph))
        paragraph = ''.join(segmented_paragraph)
        if paragraph in paragraphs_his:
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx
    # delete
    prev_del_num = 0
    del_num = 0
    # should judge "del_ids" and "para_id" have values
    if del_ids and para_id:
        for p_idx in del_ids:
            if p_idx < para_id:
                prev_del_num += 1
            del doc["segmented_paragraphs"][p_idx - del_num]
            del doc["segmented_paragraphs_scores"][p_idx - del_num]
            del doc['paragraphs_length'][p_idx - del_num]
            del_num += 1
        if 'most_related_para' in doc:
            doc['most_related_para'] = para_id - prev_del_num
        doc['paragraphs'] = []
        for segmented_para in doc["segmented_paragraphs"]:
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def paragraph_selection(sample, mode):
    """
    For each document, select paragraphs that includes as much information as possible
    Args:
        sample: a sample in the dataset.
        mode: string of ("train", "dev", "test"), indicate the type of dataset to process.
    Returns:
        None
    Raises:
        None
    """
    # predefined maximum length of (each) paragraph.
    MAX_P_LEN = 500
    # predefined splitter
    splitter = u'<splitter>'
    # topN of related paragraph to choose
    # topN = 3
    topN = 1000000
    doc_id = None
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]
        if doc_id >= len(sample['documents']):
            # Data error, answer doc ID > number of documents, this sample
            # will be filtered by dataset.py
            return
    for d_idx, doc in enumerate(sample['documents']):
        if 'segmented_paragraphs_scores' not in doc:
            continue
        """
        To remove the duplicated paragraphs of documents
        """
        status = dup_remove(doc)
        segmented_title = doc["segmented_title"]
        title_len = len(segmented_title)
        para_id = None
        if doc_id is not None:
            para_id = sample['documents'][doc_id]['most_related_para']
        """
        doc['paragraphs_length'] is each paragraph length in document.
        In order to calculate whether the total length larger than threshold "MAX_P_LEN". 
        """
        total_len = title_len + sum(doc['paragraphs_length'])
        # add splitter
        para_num = len(doc["segmented_paragraphs"])  # the nums of all paragraphs in document = the nums of "splitter"
        total_len += para_num
        """
        total_len = len(segmented_title) + sum(doc['paragraphs_length']) + len(doc["segmented_paragraphs"])
        """
        if total_len <= MAX_P_LEN:
            incre_len = title_len
            total_segmented_content = copy.deepcopy(segmented_title)
            for p_idx, segmented_para in enumerate(doc["segmented_paragraphs"]):
                """
                doc_id: answer document ID
                para_id: most related paragraph ID in this documents
                
                The following operation is to calculate previous total length of title and paragraph,
                in order to find the position of answer. 
                """
                if doc_id == d_idx and para_id > p_idx:  # para_id in behind
                    incre_len += len([splitter] + segmented_para)
                if doc_id == d_idx and para_id == p_idx:
                    incre_len += 1
                """
                total_segmented_content = segmented_title + [splitter] + segmented_para
                """
                total_segmented_content += [splitter] + segmented_para
            if doc_id == d_idx:  # i.e., current document is the answer document
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]  # the whole document segmented with '<splitter>'
            """
            If the document's "total_len" <= "MAX_P_LEN", we select the whole document.
            
            (why set to 1.0)??? beacuse we select the whole document, the whole score is certain 1.0
            """
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0

            continue  # break this foreach, start next question

        # ============= otherwise =============
        # find topN paragraph id
        para_infos = []
        for p_idx, (para_tokens, para_scores) in \
                enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens, para_scores, len(para_tokens), p_idx))
        para_infos.sort(key=lambda x: (-x[1], x[2]))  # sort by `para_scores` (question vs para)

        # select top N paragraph.
        topN_idx = []
        for para_info in para_infos[:topN]:
            topN_idx.append(para_info[-1])

        """
        doc_id: answer document ID
        para_id: most related paragraph ID in this documents
        d_idx: current document
        """
        # -------------------------- start --------------------------
        """
        The following codes is to add `para id` from `topN_idx` to `final_idx`, 
        we know that `testset` has no `doc_id` (None), so should ensure the `final_idx` has the most related para id when `train`.
        """
        final_idx = []
        total_len = title_len
        if doc_id == d_idx:  # i.e., current document is the answer document
            if mode == "train":
                final_idx.append(para_id)  # para_id: most related paragraph ID in this documents
                total_len = title_len + 1 + doc['paragraphs_length'][para_id]
        for id in topN_idx:
            if total_len > MAX_P_LEN:  # so `topN` could set to 1000000
                break
            # doc_id: answer document ID, d_idx: current document
            if doc_id == d_idx and id == para_id and mode == "train":  # for `trainset`
                continue  # because line 198-201 has added this para to `final_idx`.
            total_len += 1 + doc['paragraphs_length'][id]
            final_idx.append(id)
        # ---------------------------- end ---------------------------
        total_segmented_content = copy.deepcopy(segmented_title)
        final_idx.sort()
        incre_len = title_len
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += 1 + doc['paragraphs_length'][id]
            if doc_id == d_idx and id == para_id:
                incre_len += 1
            total_segmented_content += [splitter] + doc['segmented_paragraphs'][id]
        if doc_id == d_idx:
            answer_start = incre_len + sample['answer_spans'][0][0]
            answer_end = incre_len + sample['answer_spans'][0][1]
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end
        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len]
        doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0


if __name__ == "__main__":
    # mode="train"/"dev"/"test"
    mode = sys.argv[1]
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue
        try:
            sample = json.loads(line, encoding='utf8')
        except:
            print(sys.stderr, "Invalid input json format - '{}' will be ignored".format(line))
            continue
        compute_paragraph_score(sample)
        paragraph_selection(sample, mode)
        # print(json.dumps(sample, encoding='utf8', ensure_ascii=False))  # for python2
        print(json.dumps(sample, ensure_ascii=False))  # for python3

