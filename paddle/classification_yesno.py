import json

"""
Train:
    search, length=136208, yes_no_depends=11476
    zhidao, length=135366, yes_no_depends=11273
"""

PATH = '/home/wangyongbo/2019rc/DuReader_test/data/preprocessed/trainset'
# search.train.json   zhidao.train.json


def count_yesno():
    """
    To count the nums of tes/no/depends in dataset
    """
    with open(PATH + "/zhidao.train.json", "r", encoding="utf-8") as f:
        res_list = f.readlines()
        print("Total nums: ", len(res_list))

    cou_t = 0
    cou = 0
    yes_no = []
    for i in res_list:
        cou_t += 1
        ii = json.loads(i)
        # print(ii['question_type'])
        if 'yesno_answers' in ii:
            if ii['yesno_answers']:
                cou += 1
                yes_no.append(ii['yesno_answers'])

    # print(yes_no)
    print(len(yes_no))
    print("num of yes_no: ", cou)
    print("num of train samples: ", cou_t)



