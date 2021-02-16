import argparse
import os
from pickle import load
from typing import List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, save, Tensor, FloatTensor, nonzero, norm, matmul, eye, zeros, tensor
from torch.optim import SGD, Adam, Adagrad, RMSprop
from torch.utils.data import DataLoader, random_split
from sklearn.manifold import TSNE

from conceptnet_preprocess import PreProcess, TwoTags
from feature_extract_model import NewNusWideTrainDataset
from knowledge_graph_dataloader import ConceptNetDataSet
from knowledge_graph_model import KnowledgeGraphEmbedding
# from test_rdf.data_loader import scene_graph_dataloader
from KIPS.build_voca import Vocabulary, Relation
from test_rdf.util import make_transform


def read_pkl(path):
    with open(path, 'rb') as f:
        pkl = load(f)

    return pkl


def get_tag_representation(spo_list, tag_representation, tag1, tag2, idx):
    if idx is None:
        relation: np.array = np.zeros(14)
    else:
        relation: np.array = np.array(list(map(int, spo_list[idx][2])))
    wi: Tensor = Tensor(tag_representation[tag1])
    wj: Tensor = Tensor(tag_representation[tag2])

    """
    :param
    :return: wi(tensor), p(int), wj(tensor)
    wi, wj is tag_representation vector(1, 300)
    relation is relationship np.array(1, 14)
    """
    return wi, relation, wj


def train(args, dataloader, vocab, rel, spo_list, two_tag, tag_representation):
    kge = KnowledgeGraphEmbedding(args, vocab, rel, tag_representation).cuda()
    kge = kge.train()

    # optim = Adam(params=kge.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # optim = RMSprop(params=kge.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.1)
    # optim = Adagrad(params=kge.parameters(), lr=args.learning_rate, weight_decay=0.1)
    optim = SGD(params=kge.parameters(), lr=args.learning_rate, momentum=0.9)
    I = eye(args.hidden_size).requires_grad_(True).cuda()
    term2_reg = tensor(1).cuda()
    gamma = tensor(args.gamma).cuda()
    parameter_lambda = tensor(args.parameter_lambda).cuda()
    delta = 1e-6
    loss_value = 0
    n_tag = 1000
    n_edge = 17664
    rel_size = len(rel)

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch} / {args.epochs}')
        loss = 0
        total_term1 = FloatTensor([0]).cuda()
        total_term2 = FloatTensor([0]).cuda()
        total_term3 = FloatTensor([0]).cuda()
        # for (wi, relation, wj) in dataloader:
        #     wi = wi.cuda()
        #     wj = wj.cuda()
        #     optim.zero_grad()
        #     loss = kge(wi, relation, wj).cuda()
        #     loss.backward()
        #     optim.step()
        #     loss_value = loss.item()
        #     print(f'loss: {loss_value}')

        for iter_idx, (image, label) in enumerate(dataloader):
            _, tag_list = nonzero(label, as_tuple=True)
            tag_list = sorted(set(tag_list.tolist()))
            n_tag = tensor(len(tag_list)).cuda()
            n_edge = torch.zeros(len(rel)).cuda()
            not_edges = torch.zeros(len(rel)).cuda()

            total_term1 = zeros(rel_size).cuda()
            total_term2 = zeros(rel_size).cuda()
            total_term3 = FloatTensor([0]).cuda()

            optim.zero_grad()
            for tag1_idx, tag1 in enumerate(tag_list):
                for tag2_idx, tag2 in enumerate(tag_list):
                    tag1_name = vocab.idx2word[tag1]
                    tag2_name = vocab.idx2word[tag2]
                    two_tag_idx = two_tag(tag1_name, tag2_name)
                    if tag1_idx == tag2_idx or two_tag_idx is None:
                        continue

                    wi, relation, wj = get_tag_representation(spo_list, tag_representation, tag1, tag2, two_tag_idx)

                    term1, term2, edge, not_edge = kge(tag1_idx, relation, tag2_idx)
                    total_term1 += term1
                    total_term2 += term2
                    n_edge += edge
                    not_edges += not_edge

            total_edge = n_edge.sum()
            print('n_tag', n_tag.item())
            print('total_edge', total_edge.item())
            if total_edge == 0 or (n_tag**2-total_edge) == 0:
                continue
            for parameter in list(kge.parameters()):
                orthogonal_constraints = matmul(parameter, parameter.transpose(0, 1)) - I
                total_term3 += norm(orthogonal_constraints, p=2) ** 2

            print('total_term1[0]', total_term1[0])
            for rel_idx in range(rel_size):
                if n_edge[rel_idx] != 0:
                    total_term1[rel_idx] /= n_edge[rel_idx]
                if not_edges[rel_idx] != 0:
                    total_term2[rel_idx] /= (not_edges[rel_idx])

            print('total_term1', total_term1)
            print('total_term2', total_term2)
            # print('not_edges', not_edges)

            total_term1 = total_term1.sum()
            # total_term2 = (1 / (n_tag ** 2 - n_edge)) * total_term2
            total_term2 = (gamma * total_term2.sum())
            total_term3 *= parameter_lambda
            print('term1: ', total_term1.item(), 'term2:', total_term2.item(), 'term3:', total_term3.item())

            loss = (total_term1-total_term2+total_term3) * tensor(0.02)
            loss_value = loss.item()
            # print('idx', iter_idx)
            print('loss', loss_value)
            loss.backward()
            optim.step()

            if loss_value < delta:
                break
        if loss_value < delta:
            break

    save(kge.state_dict(), args.model_path)


def test(args, vocab, rel, spo_list, two_tag, tag_representation):
    symmetric = rel.symmetric
    asymmetric = rel.asymmetric

    kge = KnowledgeGraphEmbedding(args, vocab, rel, tag_representation).cuda()
    model = torch.load(args.model_path)

    kge.load_state_dict(model)
    kge = kge.eval()
    # tags: List = list(vocab.idx2word)

    tags = []
    tags.append(vocab.word2idx['dog'])
    tags.append(vocab.word2idx['dogs'])
    tags.append(vocab.word2idx['cute'])
    tags.append(vocab.word2idx['animal'])

    print(tags)

    delta = args.delta
    eps = args.eps

    for tag1_idx, tag1 in enumerate(tags):
        for tag2_idx, tag2 in enumerate(tags):
            tag1_name = vocab.idx2word[tag1]
            tag2_name = vocab.idx2word[tag2]
            two_tag_idx = two_tag(tag1_name, tag2_name)

            if tag1_idx == tag2_idx:
                continue

            wi, _, wj = get_tag_representation(spo_list, tag_representation, tag1, tag2, two_tag_idx)
            wi = wi.cuda()
            wj = wj.cuda()
            for relation_p in range(len(rel)):

                distance = kge.distance_i_to_j(wi, relation_p, wj)
                if distance < 0.15:
                    relation = rel.idx2rel[relation_p]
                    if relation in symmetric:
                        print(f'{tag1_name}<->{rel.idx2rel[relation_p]}<->{tag2_name}, distance: {distance}')
                    elif relation in asymmetric:
                        print(f'{tag1_name}->{rel.idx2rel[relation_p]}->{tag2_name}, distance: {distance}')


def main(args):
    transform = make_transform(args)

    vocab = read_pkl(args.vocab_path)
    rel = read_pkl(args.rel_path)
    print(len(vocab))
    two_tag = read_pkl(args.tag_path)
    conceptnet_dataset = ConceptNetDataSet(vocab=vocab, rel=rel)
    # conceptnet_train_dataset, conceptnet_test_dataset = random_split(conceptnet_dataset,
    #                                                                  [8000, len(conceptnet_dataset)-8000])
    # conceptnet_dataloader = DataLoader(dataset=conceptnet_dataset, batch_size=args.batch_size, shuffle=True,
    #                                    num_workers=args.num_workers)

    spo_list: np.array = conceptnet_dataset.spo_list
    tag_representation: np.array = conceptnet_dataset.get_tag_representation()

    new_nuswide_dataset = NewNusWideTrainDataset(vocab)
    new_nuswide_train_dataset, _ = random_split(new_nuswide_dataset, [8000, len(new_nuswide_dataset)-8000])

    new_nuswide_dataloader = DataLoader(dataset=new_nuswide_train_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=8, pin_memory=True)

    # train_dataloader, test_dataloader = scene_graph_dataloader(path=args.image_dir, scene_graph=scene_graph,
    #                                                            transform=transform, batch_size=args.batch_size,
    #                                                            num_workers=args.num_workers)

    print('vocab_size: ', len(vocab))
    print('rel_size: ', len(rel))
    # print('len_dataloader', len(new_nuswide_dataloader))
    # train(args, new_nuswide_dataloader, vocab, rel, spo_list, two_tag, tag_representation)
    test(args, vocab, rel, spo_list, two_tag, tag_representation)


def visuallize_embedding(vocab):
    n_sne = len(vocab)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_result = tsne.fit_transform()


if __name__ == '__main__':
    n_train = str(0)
    parser = argparse.ArgumentParser()

    # Load PATH
    parser.add_argument('--vocab_path', type=str, default='E:\ADD\ADD\\tag_vocab.pkl')
    parser.add_argument('--rel_path', type=str, default='E:\ADD\ADD\\conceptnet_rel.pkl')
    parser.add_argument('--tag_path', type=str, default='E:\ADD\ADD\\two_tag.pkl')
    parser.add_argument('--image_dir', type=str, default='C:\ADD\\ADD\\train_image',
                        help='path for image dir')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--scene_graph_path', type=str, default='C:\dataset\ADD\KIPS_train\\scene_graph.pk1')

    # Model Hyper-Parameters
    parser.add_argument('--embedding_size', type=int, default=300, help='dimension of word embedding space')
    parser.add_argument('--hidden_size', type=int, default=50, help='dimension of hidden space')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--parameter_lambda', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.08)
    parser.add_argument('--eps', type=float, default=0.03)

    # Training Hyper-Parameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--num_workers', type=int, default=8)

    # Save Models
    parser.add_argument('--model_path', type=str, default='E:\\untitled3\knowledge_embedding3.pth')

    args = parser.parse_args()
    main(args)

