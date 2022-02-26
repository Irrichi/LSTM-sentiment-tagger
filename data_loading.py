import numpy as np
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torchtext
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple
import typing


PADDING_VALUE = 0

def load_data(name = 'imdb_dataset.csv'):
    data = pd.read_csv(name)
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english', language='en')
    data['tokenized'] = [tokenizer(review) for review in data['review']]
    data.loc[data['sentiment']=='positive','sentiment'] = 1
    data.loc[data['sentiment']=='negative','sentiment'] = 0
    data.loc[0:25000, 'split'] = 'train'
    data.loc[25000:40000, 'split'] = 'validation'
    data.loc[40000:50000, 'split'] = 'test'
    return data


class Vectorizer:
    def __init__(self, tokenized_data):
        # Prepares lookup dict self.word_vector that maps tokens to ints.
        self.word_count = {}
        k = 1
        for review in tokenized_data:
            for word in review:
                if word not in self.word_count.keys():
                    self.word_count[word]=1
                else:
                    self.word_count[word]+=1
        self.word_vector={}
        for iter, word in enumerate(sorted(self.word_count.items(), key=lambda item: -item[1])):
            self.word_vector[word[0]]=iter+1


    def vectorize(self, tokenized_seq: List[int]) -> torch.Tensor:
        # Converts sequence of tokens into sequence of indices
        seq = []
        for i in tokenized_seq:
            if i in self.word_vector.keys():
                seq.append(self.word_vector[i])
        return torch.tensor(seq, dtype = torch.long)


class ImdbDataset(Dataset):
    SPLIT_TYPES = ["train", "test", "validation"]

    def __init__(self, data, preprocess_fn, split="train"):
        super().__init__()
        if split not in self.SPLIT_TYPES:
            raise AttributeError(f"No such split type: {split}")
        self.split = split
        self.label = [i for i, c in enumerate(data.columns) if c == "sentiment"][0]
        self.data_col = [i for i, c in enumerate(data.columns) if c == "tokenized"][0]
        self.data = data[data["split"] == self.split]
        self.preprocess_fn = preprocess_fn


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        seq = self.preprocess_fn(self.data.iloc[idx, self.data_col])
        label = self.data.iloc[idx, self.label]
        return (seq, label)


def get_datasets_and_vectorizer() -> typing.Tuple[ImdbDataset, ImdbDataset]:
    data = load_data()
    vectorizer = Vectorizer(data.loc[data["split"] == "train", "tokenized"])
    train_dataset = ImdbDataset(data, vectorizer.vectorize)
    validation_dataset = ImdbDataset(data, vectorizer.vectorize, split="validation")
    test_dataset = ImdbDataset(data, vectorizer.vectorize, split="test")
    return train_dataset, validation_dataset, test_dataset, vectorizer


def custom_collate_fn(pairs: List[Tuple]) -> typing.Tuple[torch.Tensor, typing.Tuple[List[int], torch.Tensor]]:
    # Used by dataloader to prepare batches
    seqcs = []
    lengths = []
    labels = []
    for i in pairs:
        seqcs.append(i[0])
        lengths.append(len(i[0]))
        labels.append(torch.tensor(i[1]))
    seqcs = torch.nn.utils.rnn.pad_sequence(seqcs, batch_first=True, padding_value=PADDING_VALUE)
    labels = torch.stack(labels)
    return seqcs, lengths, labels
