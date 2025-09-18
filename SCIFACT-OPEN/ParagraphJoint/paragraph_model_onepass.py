import torch
import jsonlines
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import math
import numpy as np

from tqdm import tqdm
from util import read_passages, clean_words, test_f1, to_BIO, from_BIO


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class TimeDistributedDense(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(TimeDistributedDense, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.timedistributedlayer = TimeDistributed(self.linear)
    def forward(self, x):
        # x: (BATCH_SIZE, ARRAY_LEN, INPUT_SIZE)

        return self.timedistributedlayer(x)

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob = 0.1):
        super().__init__()
        self.dense = TimeDistributedDense(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = TimeDistributedDense(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class OnePassParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2):
        super(OnePassParagraphClassifier, self).__init__()
        self.stance_label_size = 3
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)
        self.stance_criterion = nn.CrossEntropyLoss()
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index = 2)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.bert_dim = bert_dim
        # Twice the bert_dim since we're going to use the <s> token and the relevant </s> token.
        self.rationale_linear = ClassificationHead(2 * bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.stance_linear = ClassificationHead(bert_dim, self.stance_label_size, hidden_dropout_prob = dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.stance_linear,
            self.stance_criterion,
            self.rationale_criterion
        ]

    def reinitialize(self):
        # TODO(dwadden) Does this actually help?
        self.extra_modules = []
        self.rationale_linear = ClassificationHead(self.bert_dim, self.rationale_label_size, hidden_dropout_prob = self.dropout)
        self.stance_linear = ClassificationHead(self.bert_dim, self.stance_label_size, hidden_dropout_prob = self.dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.stance_linear,
            self.stance_criterion,
            self.rationale_criterion,
        ]

    def forward(self, encoded_dict, transformation_indices, stance_label = None, rationale_label = None, sample_p=1, rationale_score = False):
        # NOTE(dwadden) In this model, the `transformation_indices` are just the
        # indices of the sentence separators.
        encoded = self.bert(**encoded_dict)
        bert_out = encoded.last_hidden_state  # (BATCH_SIZE, sequence_len, BERT_DIM)

        pooled_output = self.dropout_layer(encoded.pooler_output)
        stance_out = self.stance_linear(pooled_output)

        # This only works for a batch size of 1. Will deal with scaling up later.
        sent_indices = transformation_indices[0]
        sentence_reps = bert_out[0][sent_indices[0]]
        pooled_rep = pooled_output.repeat([sentence_reps.size(0), 1])
        sentence_cat = torch.cat([pooled_rep, sentence_reps], dim=1)
        rationale_out = self.rationale_linear(sentence_cat).unsqueeze(0)

        # Since we've got a batch of 1, the mask is always on.
        sentence_mask = torch.ones_like(sent_indices)

        if stance_label is not None:
            stance_loss = self.stance_criterion(stance_out, stance_label)
        else:
            stance_loss = None
        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size),
                                                      rationale_label.view(-1)) # ignore index 2
        else:
            rationale_loss = None

        stance_out = torch.argmax(stance_out.cpu(), dim=-1).detach().numpy().tolist()
        if rationale_score:
            rationale_pred = rationale_out.cpu()[:,:,1] # (Batch_size, N_sep)
        else:
            rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1) # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in zip(rationale_pred, sentence_mask.bool())]
        return rationale_out, stance_out, rationale_loss, stance_loss
