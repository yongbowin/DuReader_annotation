import json
import csv


# BASE_PATH = "/home/wangyongbo/2019rc/pytorch-pretrained-BERT_classifier/data/yesno/"
BASE_PATH = "/home/wyb/PycharmProjects/DuReader_annotation/paddle/lic2019_results-master/"
# OUTPUT_PATH = '/home/wangyongbo/2019rc/DuReader_test/data/yesno/'

# with open(BASE_PATH + "zhidao.test.yesno_op.csv", "r", encoding="utf-8") as f2:
ids_list = []
pred_list = []
for source in ["zhidao", "search"]:
    cou = 0
    with open(BASE_PATH + source + ".test1.yesno_op.csv", "r", encoding="utf-8") as f2:
        reader = csv.reader(f2, delimiter=",")
        for line in reader:
            if cou != 0:
                ids_list.append(line[-1])
            cou += 1

    with open(BASE_PATH + "predict_results_" + source + ".txt", "r", encoding="utf-8") as f3:
        res = f3.readlines()
        for line in res:
            pred_list.append(line.strip())

if len(pred_list) != len(ids_list):
    print("========================= Error! =========================")
else:
    print("======> The length is equal.")

new_dict = {}
for id,pred in zip(ids_list, pred_list):
    new_dict[int(id)] = pred


with open(BASE_PATH + "test_result_merge_best_rm_new_split_add_logit.json", "r", encoding="utf-8") as f1:
    res_rm = f1.readlines()

# ---------------------------
"""
{
"question_id": 403770,
"question_type": "YES_NO",
"answers": ["我都是免费几分钟测试可以玩而已。"],
"entity_answers": [[]],
"yesno_answers": []
}
"""
json_list = []
for i in res_rm:
    item_dict = {}

    data = json.loads(i)
    text = data["answers"][0]

    item_dict["question_id"] = data["question_id"]
    item_dict["question_type"] = data["question_type"]
    item_dict["answers"] = [text]
    item_dict["entity_answers"] = data["entity_answers"]

    if int(data["question_id"]) in new_dict:
        item_dict["yesno_answers"] = [new_dict[int(data["question_id"])]]
    else:
        item_dict["yesno_answers"] = data["yesno_answers"]

    json_list.append(item_dict)

# ================ write to file ================
with open(BASE_PATH + "test_result_merge_best_rm_new_split_add_logit_yesno.json", 'w') as fout:
    for pred_answer in json_list:  # [{}, {}, ...]
        fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
