from typing import List

import numpy as np
from torch import nn, Tensor, eye, norm, FloatTensor, LongTensor, sub, matmul, zeros

from conceptnet_preprocess import Vocabulary, Relation


class KnowledgeGraphEmbedding(nn.Module):
    def __init__(self, args, vocab: Vocabulary, rel: Relation, tag_representation: np.array):
        super().__init__()
        self.args = args
        self.relation_size = len(rel)
        self.embedding_size = self.args.embedding_size
        self.hidden_size = self.args.hidden_size
        self.vocab = vocab
        self.rel = rel
        self.tag_representation: Tensor = FloatTensor(tag_representation)
        # self.tag_vector = nn.ModuleList([nn.Embedding.from_pretrained(tag_representation) for i in range(len(rel))])
        self.Lp = nn.ModuleList([nn.Linear(in_features=self.args.embedding_size, out_features=self.args.hidden_size,
                                           bias=False)
                                 for p in range(self.relation_size)])
        self.Rp = nn.ModuleList([nn.Linear(in_features=self.args.embedding_size, out_features=self.args.hidden_size,
                                           bias=False)
                                 for p in range(self.relation_size)])

    def forward(self, wi, relation_p, wj):
        loss_2 = self.loss_2_function(wi, relation_p, wj)
        return loss_2

    def distance_i_to_j(self, wi: Tensor, rel_idx: int, wj: Tensor):
        """
        :param wi: Tensor[batch_size, 300]
        :param rel_idx: int
        :param wj: Tensor[batch_size, 300]
        :return: [batch_size, 1(distance i to j)]

        [batch_size, hidden_size] batch-wise dot product [hidden_size, batch_size] -> shape[batch_size, 1]
        """
        # print('befor Lp', norm(self.Lp[rel_idx].weight.data, p=2, keepdim=True))
        # print('After Lp', norm(self.Lp[rel_idx](wi), p=2, keepdim=True))
        concept_mapping = self.Lp[rel_idx](wi) - self.Rp[rel_idx](wj)
        d_i_j = norm(concept_mapping, p=2, keepdim=True)
        return d_i_j ** 2

    def loss_2_function(self, tag1_idx: int, relation: np.array, tag2_idx: int):
        """
        relation_value = 0 -> wi is not related to wj
        relation_value = 1 -> wi is related to wj

        :param wi: Tensor(1, 300)
        :param relation: np.array[batch_size, relation_size]
        :param wj: Tensor(1, 300)
        :return: loss value(batch_size, 1)
        """
        # I = eye(self.hidden_size).requires_grad_(False).cuda()

        term1 = zeros(self.relation_size, requires_grad=True).cuda()
        term2 = zeros(self.relation_size, requires_grad=True).cuda()
        n_edge = zeros(self.relation_size, requires_grad=True).cuda()
        not_edge = zeros(self.relation_size, requires_grad=True).cuda()
        # tag1_idx = LongTensor([tag1_idx]).cuda()
        # tag2_idx = LongTensor([tag2_idx]).cuda()
        wi = self.tag_representation[tag1_idx].requires_grad_(True).cuda()
        wj = self.tag_representation[tag2_idx].requires_grad_(True).cuda()

        for rel_idx in range(self.relation_size):
            # wi = self.tag_vector[rel_idx](tag1_idx)
            # wj = self.tag_vector[rel_idx](tag2_idx)

            distance = self.distance_i_to_j(wi, rel_idx, wj).cuda()
            relation_value = relation[rel_idx]
            if relation_value == 1:
                term1[rel_idx] += distance
                n_edge[rel_idx] += 1

            elif relation_value == 0:
                term2[rel_idx] += distance
                not_edge[rel_idx] += 1


        # for rel_idx in range(self.relation_size):
        #     concept_Lp = self.Lp[rel_idx].requires_grad_(False).weight.data
        #     concept_Rp = self.Rp[rel_idx].requires_grad_(False).weight.data
        #     concept_Lp = self.Lp[rel_idx].weight.data
        #     concept_Rp = self.Rp[rel_idx].weight.data
        #
        #     orthogonal_constraints_Lp = matmul(concept_Lp, concept_Lp.transpose(0, 1)) - I
        #     orthogonal_constraints_Rp = matmul(concept_Rp, concept_Rp.transpose(0, 1)) - I
        #     print('norm2', (norm(orthogonal_constraints_Lp, p=2) ** 2 + norm(orthogonal_constraints_Rp, p=2) ** 2))
        #     term3 += (norm(orthogonal_constraints_Lp, p='fro') ** 2 + norm(orthogonal_constraints_Rp, p='fro') ** 2)

        return term1, term2, n_edge, not_edge
