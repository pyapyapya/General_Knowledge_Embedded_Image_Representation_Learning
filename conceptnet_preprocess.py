import os
import pickle
from typing import List, Set, Dict, Tuple

import numpy as np
from tqdm import tqdm

# from KIPS.build_voca import Vocabulary

"""
논문에서 Conceptent 데이터 셋을 사용하여 Knowledge Graph Embedding을 수행합니다.
이 코드는 데이터 셋을 만들기 위해 데이터를 전처리 하는 코드입니다.
필요한 파일은 1. 단어와 단어와의 관계인 assertions.csv 파일
            2. 단어가 dense vector로 pretrained된 numberbatch.txt 가 필요합니다.
            이는 Conceptnet5 Github wiki https://github.com/commonsense/conceptnet5/wiki 에서 구할 수 있습니다.
            
하지만, 5.7.0 ver 부터는 Relation중 하나인 'InstanceOf' 이 IsA로 merged 되었기 때문에,
    논문에서 제안한 15개 -> 14개로 줄여서 진행합니다.

"""


class PreProcess:
    def __init__(self, vocab):
        self.symmetric = ['RelatedTo', 'LocatedNear', 'Synonym']
        self.asymmetric = ['IsA', 'HasA', 'UsedFor', 'AtLocation', 'DefinedAs', 'PartOf', 'HasProperty',
                           'CapableOf', 'SymbolOf', 'ReceivesAction', 'MadeOf']
        self.relationships = self.symmetric + self.asymmetric

        self.tag_list: List = self._make_tag_list(vocab)
        self.spo_list: np.array = np.array([])
        self.spo: Dict = {}
        self.cnt_subject_object: Dict = {}

    """
    def read_conceptnet_assertions_csv(self):
        with open('E:\ADD\ADD\\assertions.csv', 'r', encoding='UTF-8') as csv_file:
            with open('E:\ADD\ADD\\SPO.txt', 'wt', encoding='UTF-8') as spo_txt_file:
                spo_set: Set = set()
                for line in tqdm(csv_file.readlines()):
                    try:
                        relation = line.split()[1].split('/')[2]
                        start_node = line.split()[2].split('/')[3]
                        end_node = line.split()[3].split('/')[3]

                        if start_node == end_node:
                            continue
                        elif (relation in self.relationships) and (start_node in self.tag_list) and \
                                (end_node in self.tag_list):
                            spo = start_node + ' ' + relation + ' ' + end_node
                            if spo in spo_set:
                                continue
                            spo_set.add(spo)
                            spo_txt_file.write(spo + '\n')
                    except IndexError:
                        print(line)
    """

    def read_conceptnet(self):
        with open('E:\ADD\ADD\\assertions.csv', 'r', encoding='UTF-8') as csv_file:
            cnt = 0
            idx = 0

            for line in tqdm(csv_file.readlines()):
                try:
                    relation_list: List = [0 for i in range(len(self.relationships))]
                    relation = line.split()[1].split('/')[2]
                    start_node = line.split()[2].split('/')[3]
                    end_node = line.split()[3].split('/')[3]

                except IndexError as error_message:
                    print('IndexError Except: ', line)
                    print('error message:', error_message)

                else:
                    if start_node == end_node:
                        continue
                    elif (relation in self.relationships) and (start_node in self.tag_list) \
                            and (end_node in self.tag_list):
                        relation_idx = self.relationships.index(relation)
                        relation_list[relation_idx] = 1

                        if start_node not in self.spo:
                            self.spo[start_node] = {}
                            self.cnt_subject_object[start_node] = {}

                            self.spo[start_node][end_node] = relation_list
                            self.cnt_subject_object[start_node][end_node] = idx

                            idx += 1
                            cnt += 1

                        elif end_node not in self.spo[start_node]:
                            self.spo[start_node][end_node] = relation_list
                            self.cnt_subject_object[start_node][end_node] = idx

                            cnt += 1
                            idx += 1

                        elif end_node in self.spo[start_node]:
                            self.spo[start_node][end_node][relation_idx] = 1
                            cnt += 1

        print('Total Edges: ', cnt)

    def write_conceptnet(self):
        with open('C:\dataset\ADD\KIPS_train\\SPO1.txt', 'wt', encoding='UTF-8') as spo_test:
            for subjects in tqdm(self.spo):
                for objects, relations in self.spo[subjects].items():
                    spo_test.write(subjects + ' ' + objects + ' ' + ' '.join(map(str, relations)) + '\n')

    def read_conceptent_numberbatch(self):
        with open('E:\ADD\ADD\\numberbatch-en.txt', 'rt', encoding='UTF-8-sig') as numberbatch_file:
            with open('C:\dataset\ADD\KIPS_train\\tag_representation.txt', 'wt', encoding='UTF-8') as tag_representation_file:
                for line in tqdm(numberbatch_file.readlines()):
                    tag = line.split()[0]
                    if tag in self.tag_list:
                        tag_representation_file.write(line)

    def _make_two_tag_pkl(self):
        two_tags = TwoTags(self.cnt_subject_object)
        print(two_tags.cnt_object_subject)
        with open('C:\dataset\ADD\KIPS_train\\two_tag1.pkl', 'wb') as pkl:
            pickle.dump(two_tags, pkl)

    def make_two_tags(self):
        with open('C:\dataset\ADD\KIPS_train\\SPO1.txt', 'rt', encoding='UTF-8') as csv_file:
            cnt = 0
            idx = 0

            for line in tqdm(csv_file.readlines()):
                try:
                    start_node = line.split()[0]
                    end_node = line.split()[1]

                except IndexError as error_message:
                    print('IndexError Except: ', line)
                    print('error message:', error_message)

                else:
                    if start_node == end_node:
                        continue
                    if start_node not in self.cnt_subject_object:
                        self.cnt_subject_object[start_node] = {}

                        self.cnt_subject_object[start_node][end_node] = idx

                        idx += 1
                        cnt += 1

                    elif end_node not in self.cnt_subject_object[start_node]:
                        self.cnt_subject_object[start_node][end_node] = idx

                        cnt += 1
                        idx += 1

                    elif end_node in self.cnt_subject_object[start_node]:
                        cnt += 1

    def file_preprocess(self):
        self.read_conceptnet()
        self.write_conceptnet()

        # self.make_two_tags()
        # self._make_two_tag_pkl()
        # self.read_conceptent_numberbatch()

    @staticmethod
    def _make_tag_list(vocab) -> List:
        # tag_list: List = list(vocab.word2idx)
        tag_list = []
        with open('C:\dataset\ADD\KIPS_train\\tag_representation.txt', 'rt') as txt_file:
            for line in txt_file:
                tag = line.split()[0].lower()
                tag_list.append(tag)

        return tag_list

    @property
    def get_tag_list(self):
        return self.tag_list

    @property
    def get_cnt_object_subject(self):
        return self.cnt_subject_object


class Vocabulary:
    def __init__(self, preprocess):
        self.tag_list = preprocess.get_tag_list

        self.word2idx: Dict = {}
        self.idx2word: Dict = {}
        self.idx = 0

    def __len__(self) -> int:
        return len(self.word2idx)

    def __call__(self, word) -> int:
        return self.word2idx[word]

    def add_word(self):
        for word in self.tag_list:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1


class Relation:
    def __init__(self, preprocess):
        self.symmetric: List[str] = preprocess.symmetric
        self.asymmetric: List[str] = preprocess.asymmetric
        self.relationships: List[str] = preprocess.relationships

        self.rel2idx: Dict = {}
        self.idx2rel: Dict = {}
        self.idx = 0

    def __len__(self) -> int:
        return len(self.rel2idx)

    def __call__(self, rel):
        return self.rel2idx[rel]

    def add_rel(self):
        for rel in self.relationships:
            if rel not in self.rel2idx:
                self.rel2idx[rel] = self.idx
                self.idx2rel[self.idx] = rel
                self.idx += 1


class TwoTags:
    def __init__(self, cnt_object_subject):
        self.cnt_object_subject = cnt_object_subject

    def __len__(self):
        len(self.cnt_object_subject)

    def __call__(self, tag1: str, tag2: str):
        if tag1 not in self.cnt_object_subject:
            return None
        elif tag2 not in self.cnt_object_subject[tag1]:
            return None
        else:
            return self.cnt_object_subject[tag1][tag2]


def make_pkl(preprocess):
    path: str = 'E:\ADD\ADD'
    vocab_file_name: str = 'tag_vocab.pkl'
    rel_file_name: str = 'conceptnet_rel.pkl'

    vocab = Vocabulary(preprocess)
    rel = Relation(preprocess)
    vocab.add_word()
    rel.add_rel()
    print(vocab)
    print(rel)
    with open(os.path.join(path, vocab_file_name), 'wb') as f:
        pickle.dump(vocab, f)   

    with open(os.path.join(path, rel_file_name), 'wb') as f:
        pickle.dump(rel, f)


def make_preprocess_data():
    with open('C:\dataset\ADD\KIPS_train\\cnn_vocab.pk1', 'rb') as f:
        vocab = pickle.load(f)
    preprocess = PreProcess(vocab)
    ######################################################
    # If you want to make preprocess file. use this method
    # preprocess.file_preprocess()
    # make_pkl(preprocess)
    ######################################################


# make_preprocess_data()
