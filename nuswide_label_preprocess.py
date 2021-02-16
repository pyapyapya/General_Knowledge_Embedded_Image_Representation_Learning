import os
import requests
import logging
from typing import List, Set, Dict

from nltk.corpus import words, wordnet
from tqdm import tqdm

from nuswide_tags_ranking import english_tag_ranking


def image_in_dir(image_id: str) -> bool:
    return not os.path.isfile('E:\dataset\ADD\ADD\\' + image_id + '.jpg')


def word_preprocessor(tag: str, line: List) -> str:
    """
    actor, actors, actress 의 빈도가 많기 때문에 feature extract 하는 과정에서 man-actor의 관계로 유도되어질 가능성이 크게 존재함.
    따라서 actor를 man으로, actress를 woman으로 수정하는 작업이 필요함.
    UPDATE: man과 people이 서로 다른 class로 존재함 person으로 통일
    :param tag: str
    :param line: List
    :return: str
    """

    """    
    if ('actor' == tag) or ('actors' == tag) or ('men' == tag):
        if "man" not in line:
            line.append('man')
        tag = tag.replace(tag, '')

    elif ('actress' == tag) or ('women' == tag):
        if "woman" not in line:
            line.append('woman')
        tag = tag.replace(tag, '')

    return tag
    """

    if tag not in line:
        tag = tag.replace('people', 'person').replace('woman', 'person').replace('man', 'person')
    return tag


class MakeFinalTags:
    """
    이 class는 'All_Tags'를 parsing하여 Final_Tag_List에 write하는 class이다.
    'All_Tags'는 photo-id / tags 로 구성되어 있다.
    Final_Tag_Class는 All_Tags들에 있는 valid한 정보들만을 남겨놓은 tag들이다. (make_tag_set function)
    따라서 All_Tags들에 Final_Tag_Class와 맞는 태그들로만 추출하여 Tag가 3개 이상 존재하는 이미지만 남겨 놓은
    txt file을 만드는 class를 선언하였다.

    function process
    make_file -> make_tag_set -> check_tag_in_tags

    """

    def __init__(self):
        self.tag_set: Set = set()

    def make_tag_set(self):
        with open('E:\ADD\ADD\\NUS_WID_Tags\\Final_Tag_List.txt', 'r', encoding='utf-8') as file:
            final_tags_list = file.readlines()
            for tag in final_tags_list:
                tag = tag.replace('\n', '')
                self.tag_set.add(tag)

    def check_tag_in_tags(self, tags: List) -> List:
        usable_tag = []
        for tag in tags:
            if tag in self.tag_set:
                tag = word_preprocessor(tag, tags)
                usable_tag.append(tag)
        return ' '.join(usable_tag).split()

    def make_file(self, lines):
        with open('E:\ADD\ADD\\NUS_WID_Tags\dataset\\final_tags.txt', 'wt', encoding='utf-8-sig') as final_file_tags:
            self.make_tag_set()
            for line in tqdm(lines):
                label_lst: List = line.split()
                image_id: List = label_lst[:1]
                image_tags: List = label_lst[1:]
                image_tags: List = self.check_tag_in_tags(tags=image_tags)

                if image_in_dir(image_id=''.join(image_id)) or (len(image_tags) < 3):
                    continue

                label: List = image_id + image_tags
                for usable_tag in label:
                    final_file_tags.write(usable_tag + ' ')
                final_file_tags.write('\n')


class MakeEnglishTags:
    """
    이 class는 english 사전에 있는 단어만을 추출하였다.
    final_tags에서는 actor, actors 등 복수명사들도 허용하기 때문에 영어 사전에 있는 경우만을 허용하여
    데이터 셋을 두 개로 만들었다.
    기본 흐름은 MakeFinalTags와 동일하다.
    """

    def __init__(self):
        # self.english_words: Set = set(words.words())
        self.english_words: Set = set(words.words())

    def check_english_word(self, tags: List) -> List:
        usable_tag: List = []
        for tag in tags:
            if tag in self.english_words:
                tag = word_preprocessor(tag, tags)
                usable_tag.append(tag)
        return ' '.join(usable_tag).split()

    def make_file(self, lines):
        with open('E:\ADD\ADD\\NUS_WID_Tags\dataset\english_tags2.txt', 'wt', encoding='utf-8-sig') \
                as english_file_tags:
            for line in tqdm(lines):
                label_lst: List = line.split()
                image_id: List = label_lst[:1]
                image_tags: List = label_lst[1:]
                image_tags: List = self.check_english_word(tags=image_tags)

                if image_in_dir(image_id=''.join(image_id)) or (len(image_tags) < 3):
                    continue

                label: List = image_id + image_tags
                for usable_tag in label:
                    english_file_tags.write(usable_tag + ' ')
                english_file_tags.write('\n')


class MakeRankTag:
    """
    만들어진 1000개의 rank_tag들을 다시 txt화 시키는 클래스이다.
    23545개의 class가 생성되어졌기 때문에 학습시키는데 힘들 수 있다.
    1000개의 class로 줄였고, 1등 tag에는 약 14000번, 1000등 tag에는 약 800번이 누적되었다.
    """
    def __init__(self):
        self._100_ranks_tag: Dict = english_tag_ranking()

    def check_rank_tag(self, tags: List) -> List:
        usable_tag = []
        for tag in tags:
            if tag in self._100_ranks_tag:
                tag = word_preprocessor(tag, tags)
                usable_tag.append(tag)
        return usable_tag

    def make_ranks_tag(self):
        with open('C:\dataset\\add\\rank_tags.txt', 'rt', encoding='utf-8-sig') as file:
            with open('C:\dataset\\add\\rank_tags1.txt', 'wt', encoding='utf-8-sig') as rank_file:
                lines = file.readlines()
                for line in tqdm(lines):
                    label_lst: List = line.split()
                    image_id: List = label_lst[:1]
                    image_tags: List = label_lst[1:]
                    image_tags: List = self.check_rank_tag(image_tags)

                    if image_in_dir(image_id=''.join(image_id)) or (len(image_tags) < 3):
                        continue

                    label: List = image_id + image_tags
                    for usable_tag in label:
                        rank_file.write(usable_tag + ' ')
                    rank_file.write('\n')


def file_process():
    """
    file 처리를 하는 함수로써, MakeFinalTags 혹은 MakeEnglishTags 등 파일을 생성하고 다룰 때 사용하는 함수이다.
    :return: None
    """
    with open('E:\ADD\ADD\\NUS_WID_Tags\All_Tags.txt', 'r', encoding='utf-8-sig') as file_all_tags:
        lines: List = file_all_tags.readlines()

        # final_tags = MakeFinalTags()
        # final_tags.make_file(lines)

        # english_tags = MakeEnglishTags()
        # english_tags.make_file(lines)

        rank_tags = MakeRankTag()
        rank_tags.make_ranks_tag()


def main():
    file_process()


main()
