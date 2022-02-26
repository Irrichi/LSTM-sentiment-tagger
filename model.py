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
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
from torchtext.vocab import GloVe
from typing import List, Tuple
import typing


class LSTMSentimentTagger(LightningModule):
    def __init__(self, word_vector: dict, hidden_dim: int = 256,
                 embedding_dim: int = 50, num_layers: int = 1, pretrained_embedding: bool = True):
        super(LSTMSentimentTagger, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim

        if pretrained_embedding:
            embedding_glove = GloVe(name='6B', dim=embedding_dim)
            embedding_matrix = torch.zeros((len(word_vector)+1, embedding_dim))
            for word, i in word_vector.items():
                embedding_vector = embedding_glove[word]
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            self.word_embeddings = nn.Embedding.from_pretrained(embedding_matrix)
            self.word_embeddings.weight.requires_grad = False
        else:
            self.word_embeddings = nn.Embedding(len(word_vector)+1, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.hidden2hidden2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.hidden2tag = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, sentence: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, h = self.lstm(packed_embeds)
        output_padded, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        x = output_padded[torch.arange(sentence.size(0)), output_lengths - 1]
        x = self.bn1(x)
        x = self.relu(self.hidden2hidden(x))
        x = self.bn2(x)
        x = self.relu(self.hidden2hidden2(x))
        x = self.bn3(x)
        x = self.sigmoid(self.hidden2tag(x))
        return x


    def training_step(self, batch: torch.Tensor, batch_nb: int) -> torch.Tensor:
        x, lenghts, y = batch
        result = self(x, lenghts).flatten()
        loss = F.binary_cross_entropy(result, y.to(torch.float))
        self.log('train_loss', loss, on_epoch=True)
        k = 0
        for i in range(0, len(y)):
            if (y[i] == 1 and result[i] > 0.5) or (y[i] == 0 and result[i] < 0.5):
                k = k + 1
        self.log('train_accuracy', k / len(y), on_epoch=True)
        return loss


    def validation_step(self, batch: torch.Tensor, batch_nb: int) -> None:
        x, lenghts, y = batch
        result = self(x, lenghts).flatten()
        loss = F.binary_cross_entropy(result, y.to(torch.float))
        k = 0
        for i in range(0, len(y)):
            if (y[i] == 1 and result[i] > 0.5) or (y[i] == 0 and result[i] < 0.5):
                k = k + 1
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', k / len(y), on_epoch=True, prog_bar=True)


    def test_step(self, batch: torch.tensor, batch_nb: int) -> None:
        x, lenghts, y = batch
        result = self(x, lenghts).flatten()
        loss = F.binary_cross_entropy(result, y.to(torch.float))
        k = 0
        for i in range(0, len(y)):
            if (y[i] == 1 and result[i] > 0.5) or (y[i] == 0 and result[i] < 0.5):
                k = k + 1
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', k / len(y), on_epoch=True, prog_bar=True)


    def configure_optimizers(self) -> torch.optim:
        return torch.optim.Adam(self.parameters(), lr=0.001)


    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)


    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        