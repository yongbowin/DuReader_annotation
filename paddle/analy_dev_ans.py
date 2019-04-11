import json

BASE_PATH = ""

with open("predictions.json", "r") as f:
    pred_data = json.loads(f)

pred_ans_dict = {}
for k, v in pred_data.items():
    id = k.split("_")[0]

    pred_ans_dict[id] = v


