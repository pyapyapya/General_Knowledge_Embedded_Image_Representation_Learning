import os
from multiprocessing import freeze_support
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import nn, save, Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, average_precision_score, precision_score

import config
from conceptnet_preprocess import Vocabulary


class GetImageElements:
    """
    DataSet에 필요한 image의 id를 구하는 class이다.
    """
    def __init__(self, txt_file):
        self.txt_file = txt_file

    def get_img_id(self) -> List:
        with open(self.txt_file, 'rt', encoding='utf-8-sig') as english_file:
            lines = english_file.readlines()
            img_id = [line.split(' ')[:1] for line in lines]
            return img_id


class NusWideTrainDataSet(Dataset):

    def __init__(self, num_train_data: int, img_dir: str, label: np.ndarray, img_id: List):
        """
        :param num_train_data: __len__에 사용하기 위한 Dataset의 크기
        :param img_dir: img를 포함하고 있는 dir_path
        :param label: tag들을 multi-hot-encoding한 label들 1000
        :param img_id: img를 찾기위해 필요한 img_id
        """
        self.num_train_data = num_train_data
        self.img_dir = img_dir
        self.img_id = img_id
        self.label = label

        self.transform = \
            transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

    def __len__(self) -> int:
        """
        :return: int Dataset의 크기
        """
        return self.num_train_data

    def __getitem__(self, idx):
        """
        label -> multi-hot-encoding // vector size: 1000
        :param idx: DataSet의 idx
        :return: img의 tensor, label -> tags를 나타내는 list를 묶어놓은 list를 return
        """
        img = Image.open(os.path.join(self.img_dir, ' '.join(self.img_id[idx]) + '.jpg'))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]


class NewNusWideTrainDataset(Dataset):
    def __init__(self, vocab):
        self.vocab = vocab

        self.image_id: List = []
        self.label: List = []

        self.image_id, self.label = self._load_dataset()

        self.image_path = config.setting['image_path']
        self.transform = \
            transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_path, self.image_id[idx] + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target: np.array = np.zeros((len(self.vocab)))

        for tag in self.label[idx]:
            tag_idx = self.vocab.word2idx[tag]
            target[tag_idx] = 1

        target = Tensor(target)
        return img, target

    @staticmethod
    def _load_dataset():
        image_id_list: List = []
        label_list: List = []

        data_path = config.setting['ranks_tags_path']
        with open(data_path, 'rt', encoding='UTF-8') as nuswide_txt_file:
            for line in nuswide_txt_file.readlines()[:100000]:
                line = line.split()
                image_id = line[0]
                label = line[1:]

                image_id_list.append(image_id)
                label_list.append(label)
        return image_id_list, label_list


class NusWideTestDataSet(Dataset):
    def __init__(self, num_test_data: int, img_dir: str, label: np.ndarray, img_id: List):
        """
       Args:
           test_file (string): txt 파일의 경로
           img_dir (string): 모든 이미지가 존재하는 디렉토리 경로
       """

        self.num_test_data = num_test_data
        self.img_dir = img_dir
        self.img_id = img_id
        self.label = label

        self.transform = \
            transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])

    def __len__(self) -> int:
        """
        :return: Dataset의 크기
        """
        return num_test_data

    def __getitem__(self, idx)-> torch.tensor:
        """
        :param idx: DataSet의 idx
        :return: img의 tensor, label -> tags를 나타내는 list를 묶어놓은 List를 return
        """
        img = Image.open(os.path.join(self.img_dir, ' '.join(self.img_id[idx]) + '.jpg'))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]


def train_model(train_dataloader, model, batch_size):
    """
    :param dataset: nus-wide DataSet
    :param model: Resnet-50

    epoch:
    batch:
    activiation function: Softmax -> BCEWithLogitsLoss(Sigmoid base, Multi-label classifier)
    optimizer: Adam(lr=0.01, Beta=0.9, 0.99)
    :return: None
    """
    model = model.train()
    model = model.cuda()
    num_train_data = len(train_dataloader)

    epochs = 10
    loss_function = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-08)
    lr_scheduler: nn.Module = StepLR(optimizer, step_size=60, gamma=0.1)

    for epoch in range(epochs):
        train_loss = 0.0

        idx = 0
        for img, label in train_dataloader:
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()

            output: torch.Tensor = model(img)
            print('output shape', output.shape)
            print('label shape', label.shape)
            loss = loss_function(output, label.float())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f'Training Epoch; {epoch + 1} / {epochs} '
                  f'[{(idx+1) * batch_size}/{num_train_data} '
                  f'Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.7f}')
            idx += 1
        lr_scheduler.step()
        print(f'average loss: {train_loss / num_train_data}')


def test_model(test_loader, model):
    model = model.eval()
    model = model.cuda()
    test_loss = 0.0
    test_acc = 0.0
    num_test_dataset = test_loader.dataset.__len__()

    loss_function = BCEWithLogitsLoss()

    pred_list: List = []
    label_list: List = []
    precision_list: List = []
    for img, label in test_loader:
        img = img.cuda()
        label = label.clone().detach()
        label = label.cuda()
        output = model(img)

        pred = torch.sigmoid(output).data > 0.5
        loss = loss_function(output.float(), label.float()).cuda()

        pred = pred.cpu().numpy().astype(int)
        label = label.cpu().numpy()
        print('label', np.where(label == 1))
        print('pred', np.where(pred == 1))
        test_loss += loss.item()
        precision = precision_score(y_true=label, y_pred=pred, average='samples')
        # test_acc += precision

        precision_list.append(precision)
        label_list.append(label)
        pred_list.append(pred)
    print(label_list)
    print(pred_list)
    # mAP = average_precision_score(y_true=label_list, y_score=pred_list, average='samples')
    top_precision = sorted(precision_list, reverse=True)
    precision_at_5 = sum(top_precision[:5]) / 5
    precision_at_10 = sum(top_precision[:10]) / 10
    precision_at_50 = sum(top_precision[:50]) / 50

    # average_loss = test_loss / num_test_dataset
    # average_acc = test_acc / num_test_dataset

    # print(f'Test Average Loss : {average_loss}')
    # print(f'Test Acc: : {average_acc}')
    # print(f'mAP: {mAP}')
    print(f'P@5: {precision_at_5}')
    print(f'P@10: {precision_at_10}')
    print(f'P@50: {precision_at_50}')


def fine_tuning_resnet():
    """
    model: Alexnet -> resnet50
    last fc channels = 1000 that is num of total tags
    :return: nn.Models
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet50(pretrained=True)
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, 1000)
    model = model.to(device)
    return model


def cal_num_dataset(path_file):
    with open(path_file, 'rt', encoding='UTF-8') as file:
        lines = file.readlines()
        return len(lines)


if __name__ == '__main__':
    freeze_support()
    batch_size = 16

    model = fine_tuning_resnet()
    # model.load_state_dict(torch.load('E:\\untitled3\\test_model.pth'))
    """
    img_dir = config.setting['image_path']
    label = multi_hot_encoding()
    get_image_elements = GetImageElements(txt_file=config.setting['ranks_tags_path'])
    img_id: List = get_image_elements.get_img_id()
    path_train_file = config.setting['train_file_path']
    num_train_data = cal_num_dataset(path_train_file)

    nuswide_train_dataset = NusWideTrainDataSet(num_train_data, img_dir, label, img_id)
    nuswide_train_dataloader = DataLoader(
        dataset=nuswide_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    path_test_file = config.setting['test_file_path']
    num_test_data = cal_num_dataset(path_test_file)

    nuswide_test_dataset = NusWideTestDataSet(num_test_data, img_dir, label, img_id)
    nuswide_test_dataloader = DataLoader(
        dataset=nuswide_test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    train_model(nuswide_train_dataloader, model, batch_size=batch_size)
    save(model.state_dict(), config.setting['python_path'])
    model = torch.load(config.setting['model_path'])
    test_model(nuswide_test_dataloader, model)
    """