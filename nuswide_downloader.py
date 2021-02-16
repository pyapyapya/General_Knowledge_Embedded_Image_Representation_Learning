import logging
from multiprocessing import Pool, freeze_support
import os
from logging import Logger
from typing import TextIO
from urllib import request

from dataclasses import dataclass
from tqdm import tqdm

import config


def make_log(log_path: str) -> logging:
    logger: Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_path, 'asdf.txt'))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


@dataclass
class Photo:
    type: str
    id: str
    url: str
    name: str


@dataclass
class DataCount:
    data_cnt: int
    value_error_cnt: int
    http_error_cnt: int
    exception_cnt: int


class ImageDownloader:
    """
    nus-wide Image Dataset Downloader

    pipeline
    data url read-> web scrapping -> image download -> save in path
    Maybe It gets 250,000 images.
    """

    photo_data: TextIO
    lines: str

    def __init__(self, path, logger):
        self.logger = logger
        self.path = path
        self.photo = Photo('', '', '', '')
        self.count = DataCount(0, 0, 0, 0)
        self.photo_data = open(os.path.join(path, "NUS-WIDE-urls.txt"), 'r')
        self.lines: str = self.photo_data.readlines()

    def image_load(self):
        for line in tqdm(self.lines):
            photo_data = self.data_preprocess(line)
            self.photo.type = photo_data[0].split('\\')[3]
            self.photo.name = photo_data[0].split('\\')[4].split('_')[1]
            self.photo.id = photo_data[1]
            self.photo.url = photo_data[3]
            # self.make_class_dir()
            self.image_download()
            self.count.data_cnt += 1
            logging.info(f'{line}')
        self.photo_data.close()

    def data_preprocess(self, line: str) -> list:
        photo_data = line.split()
        return photo_data

    def image_download(self):
        """
        parse image in your disk
        :return: None
        """
        try:
            request.urlretrieve(self.photo.url, os.path.join(self.path, self.photo.name))
            # request.urlretrieve(self.photo.url, self.photo.name)
            self.logger.debug(f'photo.url: {self.photo.url}')
        except ValueError:
            self.count.value_error_cnt += 1
            self.logger.debug(f'ValueError: {self.photo.type}')
        except request.HTTPError:
            self.count.http_error_cnt += 1
            self.logger.debug(f'HTTPError: {self.photo.url}')
        except Exception as etc_error:
            self.count.exception_cnt += 1
            self.logger.debug(f'Exception: {etc_error}')

    def make_class_dir(self):
        """"
        I made class directory. But It is option
        I made code in this program, but I don't recommend.
        Because Data label is given file name. not dir.
        """
        try:
            if not os.path.exists(os.path.join(self.path, self.photo.type)):
                os.mkdir(os.path.join(self.path, self.photo.type))
                self.logger.info("mkdir" + self.photo.type)
        except OSError:
            self.logger.info("failed make_class_dir")


if __name__ == '__main__':
    freeze_support()
    # path = config.setting['image_path']
    pool = Pool(4)
    path = os.path.abspath('E:\dataset\ADD\ADD')
    logger = make_log(path)
    img = ImageDownloader(path, logger)
    pool.map(img.image_load())
