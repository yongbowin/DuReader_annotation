import json
import csv


BASE_PATH = "/home/wangyongbo/2019rc/DuReader_test/data/"
OUTPUT_PATH = '/home/wangyongbo/2019rc/DuReader_test/data/yesno/'


def all_ques_ids():
    """
    acquire all questions ids.

    Extract answers from `test_result_rm.json`(cleaned results)

    Return: (all yes_no type ...)
        {11253: "This is a answer of question xxx.", 8907: "This is the answer of question xxxx", ......}
    """
    with open(BASE_PATH + "results/" + "test_result_rm.json", "r", encoding="utf-8") as f:
        res = f.readlines()  # list, len=120000

    ques_ids_list = {}
    for i in res:
        data = json.loads(i)
        if data["question_type"] == "YES_NO":
            answers = ""
            for a in data["answers"]:
                answers += a
            ques_ids_list[data["question_id"]] = answers

    return ques_ids_list


"""
test_result.json:
    {
        "question_id": 191572, 
        "question_type": "DESCRIPTION", 
        "answers": ["手账，指用于记事的本子。"], 
        "entity_answers": [[]], 
        "yesno_answers": []
    }
    ......

Train format:
    question,answer,label
    ..., ..., ...

Test format:
    question,answer,ids
    ..., ..., ...
"""

ques_ids_list = all_ques_ids()
# 1.extract question from testset
sources = ["search", "zhidao"]
for source in sources:
    yes_no_rows = [["question", "answer", "ids"]]  # append title line
    with open(BASE_PATH + "extracted/testset/" + source + ".test.json", "r", encoding="utf-8") as f1:
        items = f1.readlines()  # list, len=120000

    for item in items:
        line = json.loads(item)
        if line["question_id"] in ques_ids_list:
            new_line = []  # question,answer,ids
            new_line.append(line["question"])
            new_line.append(ques_ids_list[line["question_id"]])  # answers from test_result_rm.json
            new_line.append(line["question_id"])
            yes_no_rows.append(new_line)

    """
    1.Write to `zhidao.test.yesno_op.csv` and `search.test.yesno_op.csv`. These two files are used as input of Bert classifier.
    
    2.In order to predict `Yes` or `No` or `Depends` by using Bert classifier, 
    then write to `predict_results_zhidao.txt` and `predict_results_search.txt`.
    """
    output_file = OUTPUT_PATH + source + ".test.yesno_op.csv"
    with open(output_file, 'w') as fout:
        f_csv = csv.writer(fout)
        f_csv.writerows(yes_no_rows)
        print(output_file.split("/")[-1] + " finished!")
