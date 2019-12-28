import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
from tqdm import tqdm


class IntracranialDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), torch.Tensor([self.y[idx]])


class TripletDataset(Dataset):

    def __init__(self, pairs_dataset, emb, is_triplet=True, stack_samples = True, return_words=False):
        self.emb = emb
        self.stack_samples = stack_samples
        self.dataset = pairs_dataset
        self.is_triplet = is_triplet
        self.relation_encoder = {'S': 0, 'A': 1}
        self._len = len(self.dataset)
        self.return_words = return_words
        if self.is_triplet:
            self.word_statistic = pd.DataFrame(self.pair_statistic()).T
            self.words = self.get_triplet_words(self.word_statistic)
            self._len = len(self.words)

    def __getitem__(self, index):
        if self.is_triplet:
            word = self.words[index]
            anchor = self.emb[word]
            ant, syn = self._get_word_pairs(index)
            ant = self._rebuild_pairs(word, ant)
            syn = self._rebuild_pairs(word, syn)
            positive = self.emb[random.choice(syn)]
            negative = self.emb[random.choice(ant)]
            return (anchor, positive, negative), []

        if not self.is_triplet:
            word = self.dataset[index][0]
            anchor = self.emb[word]
            related_word = self.dataset[index][1]
            related_word_emb = self.emb[related_word]
            relation = self.relation_encoder[self.dataset[index][2]]
            if self.stack_samples:
                x = np.hstack((anchor, related_word_emb))
                return x, relation
            else:
                if self.return_words:
                    return (anchor, related_word_emb), relation, (word, related_word)
                else:
                    return (anchor, related_word_emb), relation
            #return (anchor, related_word_emb), relation

    def _get_word_pairs(self, idx):
        d = np.where(self.dataset[:, [0, 1]] == self.words[idx])[0]
        split = self.dataset[d]
        ant = split[split[:, 2] == 'A']
        syn = split[split[:, 2] == 'S']
        return ant, syn

    def _rebuild_pairs(self, word, array):
        return (list(set([item for sublist in array[:, [0, 1]] for item in sublist]) - {word}))

    def _append_word_to_dict(self, d, word):
        d[word] = {'S': 0, 'A': 0}

    def _check_word(self, d, w):
        if w not in d.keys():
            self._append_word_to_dict(d, w)

    def pair_statistic(self):
        result_dict = dict()
        for w0, w1, r in tqdm(self.dataset[:, [0, 1, 2]]):
            self._check_word(result_dict, w0)
            self._check_word(result_dict, w1)

            result_dict[w0][r] += 1
            result_dict[w1][r] += 1
        return result_dict

    def get_triplet_words(self, df):
        return df[((df['A'] > 0) & (df['S'] > 0))].index.to_list()

    def __len__(self):
        return self._len


# class TripletDataset(Dataset):
#
#     def __init__(self, pairs_dataset, emb, is_triplet=True):
#         self.emb = emb
#         self.dataset = pairs_dataset
#         self.is_triplet = is_triplet
#         if self.is_triplet:
#             self.word_statistic = pd.DataFrame(self.pair_statistic()).T
#             self.words = self.get_triplet_words(self.word_statistic)
#         else:
#             self.words = np.unique(list(self.dataset[:, 0]) + list(self.dataset[:, 1]))
#
#     def __getitem__(self, index):
#         word = self.words[index]
#         ant, syn = self._get_word_pairs(index)
#         # return word, ant, syn
#         anchor = self.emb[word]
#         ant = self._rebuild_pairs(word, ant)
#         syn = self._rebuild_pairs(word, syn)
#
#         if self.is_triplet:
#             positive = self.emb[random.choice(syn)]
#             negative = self.emb[random.choice(ant)]
#             # return (word,random.choice(syn),random.choice(ant))
#             return (anchor, positive, negative), []
#
#         if not self.is_triplet:
#             if len(ant) < 1:
#                 related_word_emb = self.emb[random.choice(syn)]
#                 relation = 0
#                 return (anchor, related_word_emb), relation
#             elif len(syn) < 1:
#                 related_word_emb = self.emb[random.choice(ant)]
#                 relation = 1
#                 return (anchor, related_word_emb), relation
#             else:
#                 if random.randint(0, 1):
#                     related_word_emb = self.emb[random.choice(syn)]
#                     relation = 0
#                 else:
#                     related_word_emb = self.emb[random.choice(ant)]
#                     relation = 1
#                 return (anchor, related_word_emb), relation
#
#     def _get_word_pairs(self, idx):
#         d = np.where(self.dataset[:, [0, 1]] == self.words[idx])[0]
#         split = self.dataset[d]
#         ant = split[split[:, 2] == 'A']
#         syn = split[split[:, 2] == 'S']
#         # print(f'ant: {len(ant)}, syn: {len(syn)}')
#         return ant, syn
#
#     def _rebuild_pairs(self, word, array):
#         return (list(set([item for sublist in array[:, [0, 1]] for item in sublist]) - {word}))
#
#
#     def _append_word_to_dict(self, d, word):
#         d[word] = {'S': 0, 'A': 0}
#
#     def _check_word(self, d, w):
#         if w not in d.keys():
#             self._append_word_to_dict(d, w)
#
#     def pair_statistic(self):
#         result_dict = dict()
#         for w0, w1, r in tqdm(self.dataset[:, [0, 1, 2]]):
#             self._check_word(result_dict, w0)
#             self._check_word(result_dict, w1)
#
#             result_dict[w0][r] += 1
#             result_dict[w1][r] += 1
#         return result_dict
#
#     def get_triplet_words(self, df):
#         return df[((df['A'] > 0) & (df['S'] > 0))].index.to_list()
#
#     def __len__(self):
#         return len(self.words)