import os
from pickle import load
from typing import List, Dict, Set

import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from conceptnet_preprocess import PreProcess, Vocabulary, Relation


class ConceptNetDataSet(Dataset):
    def __init__(self, vocab, rel):
        preprocess = PreProcess(vocab)
        self.vocab = vocab
        self.rel = rel

        self.symmetric: List[str] = preprocess.symmetric
        self.asymmetric: List[str] = preprocess.asymmetric
        self.relationships: List[str] = preprocess.relationships
        self.tag_list: List[str] = preprocess.get_tag_list
        self.spo_list: List[str] = []
        self.image_tag_list: List[str] = []

        self.read_spo_data()
        self.read_image_tag_data()
        self.tag_representation: np.array = self.get_tag_representation()

        self.n_tags = len(self.tag_list)
        self.n_edges = len(self.spo_list)

    def __len__(self) -> int:
        return len(self.image_tag_list)
        # return self.n_edges

    def __getitem__(self, idx):
        """
        :param idx
        :return: wi(tensor), p(int), wj(tensor)
        wi, wj is tag_representation vector(1, 300)
        relation is relationship np.array(1, 14)
        """

        spo = self.spo_list[idx].split()
        s: str = spo[0]
        o: str = spo[1]
        relation: np.array = np.array(list(map(int, spo[2:])))

        wi: Tensor = Tensor(self.tag_representation[self.vocab.word2idx[s]])
        wj: Tensor = Tensor(self.tag_representation[self.vocab.word2idx[o]])

        return wi, relation, wj

    def get_tag_representation(self) -> np.array:
        tag_representation: np.array = np.zeros((len(self.tag_list), 300))
        with open('C:\dataset\ADD\KIPS_train\\tag_representation.txt', 'rt', encoding='UTF-8') as tag_representation_file:
        # with open('E:\ADD\ADD\\tag_representation.txt', 'rt', encoding='UTF-8') as tag_representation_file:
            for line in tqdm(tag_representation_file.readlines()):
                line = line.split()
                tag = line[0]
                tag_idx = self.vocab.word2idx[tag]

                tag_representation[tag_idx] = list(map(float, line[1:]))
        return tag_representation

    def read_spo_data(self):
        with open('C:\dataset\ADD\KIPS_train\\SPO1.txt', 'rt', encoding='UTF-8') as spo_txt_file:
        # with open('E:\ADD\ADD\\SPO.txt', 'rt', encoding='UTF-8') as spo_txt_file:
            for idx, line in enumerate(tqdm(spo_txt_file)):
                line = line.strip()
                line = line.split()
                tag1 = ''.join(line[0])
                tag2 = ''.join(line[1])
                relation = line[2:]
                self.spo_list.append([tag1, tag2, relation])

    def read_image_tag_data(self):
        with open('E:\ADD\ADD\\NUS_WID_Tags\dataset\\rank_tags.txt', 'rt', encoding='UTF-8') as image_tag_data:
            for line in tqdm(image_tag_data.readlines()):
                data = line.split()
                image_id = data[0]
                tags = data[1:]
                self.image_tag_list.append(tags)
