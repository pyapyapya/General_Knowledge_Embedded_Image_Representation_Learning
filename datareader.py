import os
# from pickle import loads

import pandas as pd

import config


class DataReader:
    """
    Data Read(txt, dat, csv) Using os, pickle, pandas
    """
    def file_read(self, path: str, file_name: str):
        print(os.path.join(path, file_name))
        with open(os.path.join(path, file_name), 'rb') as file:
            lines = file.readlines()
            for line in lines:
                print(line)

    def cvs_read(self, path: str, file_name: str) ->pd.DataFrame:
        with open(os.path.join(path, file_name), 'rb', encoding='UTF8') as file_name:
            concept_data = pd.read_csv(file_name,
                                       engine='python', header=None, sep='delimiter', error_bad_lines=False)
            print(concept_data)
            return concept_data


class DataLoader:
    """
    NUS-WIDE Data (image_tags, image_list, Low_Level_Features...)
    """

    def __init__(self):
        self.reader = DataReader()

    def load_groundtruth(self):
        groundtruth_path = config.setting['Groundtruth_TrainTestLabels_path']
        self.reader.file_read(groundtruth_path, 'Labels_airport_Train.txt')

    def load_tags(self):
        nuswide_tags_path = config.setting['NusWide_Tags_path']
        self.reader.file_read(nuswide_tags_path, 'AllTags1k.txt')

    def load_concepts(self) -> pd.DataFrame:
        concepts_path = config.setting['data_path']
        concepts_data = self.reader.cvs_read(concepts_path, 'assertions.csv')
        return concepts_data

    def load_low_level_feature(self):
        lowlevelfeature_path = config.setting['LowLevelFeature_path']
        self.reader.file_read(lowlevelfeature_path, 'Normalized_CH.dat')


def main():
    data = DataLoader()
    data.load_groundtruth()
    # data.load_tags()
    # data.load_concepts()
    # data.load_low_level_feature()

main()
