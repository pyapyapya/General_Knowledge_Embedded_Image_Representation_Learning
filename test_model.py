"""
epoch 60, batch 32, resnet-50, lr = 0.001 StepLR -> 20 -> gamma 0.1
loss는 0.04로 시작해서 0.025로 fit되었음 test_rdf loss의 경우 0.007정도

epoch: 60 BEST!!
model3.pth => average loss: 0.05269172199258581
Test set: Average Loss : 0.001223999480107499

epoch: 15
model4.pth => average loss: 0.014893659411296249
Test set: Average Loss : 0.0009589633496365372

epoch: 5
average loss: 0.005270841861627996
Test set: Average Loss : 0.0010323114876730154


"""

import requests
import torch
import config
from PIL import Image
from typing import List, Dict
from torch import nn
from torchvision import transforms
from numpy import random
from torchvision.models import resnet50

from util import multi_hot_decoding


def img_load():
    img = Image.open('E:\\untitled3\\test_rdf.jpg')
    img = img.convert('RGB')
    transform = transforms.\
        Compose([transforms.RandomResizedCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = transform(img)
    return img


def test_model(model):
    model.eval()

    img = img_load()
    img = img.cuda()
    img = img.unsqueeze(0)
    output = model(img)
    label = torch.sigmoid(output)
    pred = label.data > 0.5
    pred = pred.to(torch.int)
    return pred


def relation_dict() -> List:
    relation: List = ['IsA', 'HasA', 'RelatedTo', 'UsedFor', 'AtLocation', 'DefinedAs', 'InstanceOf', 'PartOf'
                      'HasProperty', 'CapableOf', 'SymbolOf', 'LocatedNear', 'ReceivesAction', 'MadeOf', 'Synonym']
    return relation


def get_relation_from_conceptnet(tag_list: List):
    for tag_idx, node1 in enumerate(tag_list):
        for node2 in tag_list[tag_idx+1:]:
            response = requests.get('http://api.conceptnet.io/query?=/c/en/'+node1+'&other=/c/en/'+node2)
            obj = response.json()
            for value in obj['edges']:
                start_node = value["start"]["label"]
                end_node = value["end"]["label"]
                relate_node = value["rel"]["label"]
                weight = value["weight"]
                if node1 in start_node and node2 in end_node and value["weight"] > 1:
                    print(f'start: {start_node} end: {end_node} '
                          f'relate: {relate_node} weight: {weight}')
                    break


class KnowledgeGraphEmbedding(nn.Module):
    def __init__(self, tag_idx_list, weight_matrix):
        super().__init__()

        vocab_size = 300
        hidden_size = 100

        self.Lp = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.Rp = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.weight_matrix = weight_matrix

        self.tag_idx_list = tag_idx_list
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = torch.mean(self.Lp(x), dim=0).view(1, -1)
        out = self.linear(embed)
        return out

    def distance(self):
        for tag_idx, node1_idx in enumerate(self.tag_idx_list):
            for node2_idx in self.tag_idx_list[tag_idx+1:]:
                wi: torch = self.weight_matrix[node1_idx]
                wj: torch = self.weight_matrix[node2_idx]


# def _loss_function():


# def train_knowledge_embedding(model):
#     lr = 0.001
#     epochs = 60
#     loss = _loss_function()
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     for epoch in range(epochs):


def fine_tuning_resnet(model):
    """
    model: Alexnet -> resnet50
    last fc channels = 1000 that is num of total tags
    :return: nn.Models
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, 1000)
    model.fc = nn.Linear(1000, 100)
    model = model.to(device)
    torch.save(model, config.setting['python_path'])
    return model


def main():
    model = torch.load('E:\\untitled3\\test_model3.pth')

    # fine_tuning_resnet(model)
    # fine_tuning_model = torch.load('E:\\untitled3\\fine_tuning.pth')
    # print(fine_tuning_model.fc.weight.data.shape)

    weight = model.fc.weight.data
    pred = test_model(model)
    pred_tag_list, pred_tag_idx_list = multi_hot_decoding(pred)
    # get_relation_from_conceptnet(pred_tag_list)

    kge = KnowledgeGraphEmbedding(pred_tag_idx_list, weight)

    # train_knowledge_embedding(model)

main()
