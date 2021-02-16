import pickle
from tqdm import tqdm
from typing import List

from sklearn.preprocessing import MultiLabelBinarizer

import config


def make_train_test_file():
    """
    tag_ranks
    dataset: 136022
    train dataset: 100000

    tag_ranks1
    dataset: 65857
    train dataset: 40000


    :return:
    """
    with open(config.setting['ranks_tags_path'], 'rt', encoding='utf-8') as file:
        lines = file.readlines()
        for idx, line in enumerate(tqdm(lines)):
            if idx < 40000:
                with open(config.setting['train_file_path'], 'at', encoding='utf-8') as train_file:
                    train_file.writelines(line)
            else:
                with open(config.setting['test_file_path'], 'at', encoding='utf-8') as test_file:
                    test_file.writelines(line)


# make_train_test_file()

mlb = MultiLabelBinarizer()


def multi_hot_encoding():
    # global mlb
    with open(config.setting['ranks_tags_path'], 'rt', encoding='utf-8') as file:
        tag_lst = list()
        lines = file.readlines()
        for line in lines:
            tag_lst.append(line.strip().split()[1:])
            # tag_set.update(line.strip().split()[1:])
        # tag_set = sorted(list(tag_lst))
        # print(len(tag_set))

        tag_encoding = mlb.fit_transform(tag_lst)
        # print(tag_encoding.shape)
        return tag_encoding
        # tag_decode = mlb.inverse_transform(tag_encoding)
        # return tag_encoding


# multi_hot_encoding()


def multi_hot_decoding(pred):
    # global mlb
    # tag_lst: List = multi_hot_encoding()
    tag_set = set()
    with open(config.setting['ranks_tags_path'], 'rt', encoding='utf-8') as file:
        # tag_lst = list()
        lines = file.readlines()
        for line in lines:
            # tag_lst.extend(line.strip().split()[1:])
            tag_set.update(line.strip().split()[1:])
        tag_lst = sorted(list(tag_set))
        # print(len(tag_lst))
    # print(len(tag_lst))
    pred_tag_lst: List = []
    pred_tag_idx_lst: List = []
    # print(pred.shape)
    for idx, value in enumerate(pred[0]):
        # print(idx)
        if pred[0][idx] == 1:
            pred_tag_lst.append(tag_lst[idx])
            pred_tag_idx_lst.append(idx)
            # print(tag_lst[idx])
    return pred_tag_lst, pred_tag_idx_lst

    # return mlb.inverse_transform(tag_list)


def find_num_data():
    with open(config.setting['tags_path'], 'rt', encoding='utf-8') as file:
        lines = file.readlines()
        print(len(lines))
