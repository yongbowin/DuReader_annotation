import json
from tqdm import tqdm

"""
Merge all subfile to a large file.

search: 137
zhidao: 136
"""

search_nums = 137
zhidao_nums = 136
subfile_nums = 0
datasets = ["search", "zhidao"]
for dataset in datasets:
    # search.train_ot_5.json
    # zhidao.train_ot_6.json
    if dataset == "search":
        subfile_nums = search_nums
    elif dataset == "zhidao":
        subfile_nums = zhidao_nums

    merge_list = []
    for i in tqdm(range(subfile_nums)):
        with open("split_data/split_" + dataset + "_output/" + dataset + ".train_ot_" + str(i+1) + ".json", "r") as f:
            lines = f.readlines()

        for line in lines:
            sample = json.loads(line)
            merge_list.append(sample)

    # ================ write to file ================
    with open(dataset + ".train.merge_after_multi_candidate.json", 'w') as fout:
        for item in merge_list:  # [{}, {}, ...]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')


