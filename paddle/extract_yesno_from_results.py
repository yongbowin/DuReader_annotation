import json
import csv


BASE_PATH = "/home/wangyongbo/2019rc/DuReader_test/data/"
OUTPUT_PATH = '/home/wangyongbo/2019rc/DuReader_test/data/yesno/'


def all_ques_ids():
    """
    acquire all questions ids.
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
            new_line = []
            new_line.append(line["question"])
            new_line.append(ques_ids_list[line["question_id"]])
            new_line.append(line["question_id"])
            yes_no_rows.append(new_line)

    # write csv file
    output_file = OUTPUT_PATH + source + ".test.yesno.csv"
    with open(output_file, 'w') as fout:
        f_csv = csv.writer(fout)
        f_csv.writerows(yes_no_rows)
        print(output_file.split("/")[-1] + " finished!")
