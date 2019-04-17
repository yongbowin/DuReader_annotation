import json
from tqdm import tqdm

"""
Split trainset to many subfiles, each file has 1000 samples.

search: 137
zhidao: 136
"""

datasets = ["search", "zhidao"]
for dataset in datasets:
    with open(dataset + ".train.json", "r") as f:
        lines = f.readlines()

    count = 0
    num = 0
    temp_list = []
    for line in tqdm(lines):
        count += 1
        sample = json.loads(line)

        if count % 1000 != 0:
            temp_list.append(sample)
        else:
            num += 1
            # ================ write to file ================
            with open("split_data/split_" + dataset + "/" + dataset + ".train_" + str(num) + ".json", 'w') as fout:
                for item in temp_list:  # [{}, {}, ...]
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

            temp_list = []

    # ================ write to file ================
    with open("split_data/split_" + dataset + "/" + dataset + ".train_" + str(num+1) + ".json", 'w') as fout:
        for item in temp_list:  # [{}, {}, ...]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

