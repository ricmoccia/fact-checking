import sys
import argparse

import torch
import jsonlines
import os

import functools
print = functools.partial(print, flush=True)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import math
import random
import numpy as np

from tqdm import tqdm
from util import arg2param, flatten, stance2json, rationale2json
from paragraph_model_dynamic import JointParagraphClassifier
from paragraph_model_onepass import OnePassParagraphClassifier
from dataset import FEVERParagraphBatchDataset

import logging

def schedule_sample_p(epoch, total):
    return np.sin(0.5* np.pi* epoch / (total-1))

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def batch_rationale_label(labels, padding_idx = 2):
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        for j, evid in enumerate(label):
            label_matrix[i,j] = int(evid)
        label_list.append([int(evid) for evid in label])
    return label_matrix.long(), label_list

def evaluation(model, dataset):
    model.eval()
    rationale_predictions = []
    rationale_labels = []
    stance_preds = []
    stance_labels = []

    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)

            # Like in the training loop, need different types of transformation
            # indices based on model.
            if isinstance(model, OnePassParagraphClassifier):
                transformation_indices = get_sep_tokens(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
            else:
                # Otherwise, get the indices for each sentence in the evidence.
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]

            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            stance_label = batch["stance"].to(device)
            padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
            if padded_rationale_label.size(1) == transformation_indices[-1].size(1):
                rationale_out, stance_out, rationale_loss, stance_loss = \
                    model(encoded_dict, transformation_indices, stance_label = stance_label,
                          rationale_label = padded_rationale_label.to(device))
                stance_preds.extend(stance_out)
                stance_labels.extend(stance_label.cpu().numpy().tolist())

                rationale_predictions.extend(remove_dummy(rationale_out))
                rationale_labels.extend(remove_dummy(rationale_label))

    stance_f1 = f1_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_precision = precision_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_recall = recall_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    rationale_f1 = f1_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_precision = precision_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_recall = recall_score(flatten(rationale_labels),flatten(rationale_predictions))
    return stance_f1, stance_precision, stance_recall, rationale_f1, rationale_precision, rationale_recall

def encode(tokenizer, batch, max_sent_len = 512, model_type="dynamic"):
    def truncate(input_ids, max_length, sep_token_id, pad_token_id):
        def longest_first_truncation(sentences, objective):
            sent_lens = [len(sent) for sent in sentences]
            while np.sum(sent_lens) > objective:
                max_position = np.argmax(sent_lens)
                sent_lens[max_position] -= 1
            return [sentence[:length] for sentence, length in zip(sentences, sent_lens)]

        all_paragraphs = []
        for paragraph in input_ids:
            valid_paragraph = paragraph[paragraph != pad_token_id]
            if valid_paragraph.size(0) <= max_length:
                all_paragraphs.append(paragraph[:max_length].unsqueeze(0))
            else:
                sep_token_idx = np.arange(valid_paragraph.size(0))[(valid_paragraph == sep_token_id).numpy()]
                idx_by_sentence = []
                prev_idx = 0
                for idx in sep_token_idx:
                    idx_by_sentence.append(paragraph[prev_idx:idx])
                    prev_idx = idx
                objective = max_length - 1 - len(idx_by_sentence[0]) # The last sep_token left out
                truncated_sentences = longest_first_truncation(idx_by_sentence[1:], objective)
                truncated_paragraph = torch.cat([idx_by_sentence[0]] + truncated_sentences + [torch.tensor([sep_token_id])],0)
                all_paragraphs.append(truncated_paragraph.unsqueeze(0))

        return torch.cat(all_paragraphs, 0)

    def make_global_attention_mask(input_ids, sep_token_id):
        # For Longformer, assign global attention to everything before the evidence.
        global_attention_mask = torch.zeros_like(input_ids)

        for batch_ix in range(input_ids.size(0)):
            entry = input_ids[batch_ix]
            first_sep = torch.where(entry == sep_token_id)[0][0].item()
            global_attention = torch.zeros_like(entry)
            global_attention[:first_sep] = 1
            # If we're using the one-pass model, add global attention on the
            # </s> tokens.
            if model_type == "onepass":
                sep_ix = entry == sep_token_id
                global_attention[sep_ix] = 1

            # Set the
            global_attention_mask[batch_ix] = global_attention

        return global_attention_mask

    inputs = zip(batch["claim"], batch["paragraph"])
    # Transformers no long accepts zips; needs to be converted to list.
    inputs = [x for x in inputs]
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        padding="longest", add_special_tokens=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        if 'token_type_ids' in encoded_dict:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len,
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'token_type_ids': encoded_dict['token_type_ids'][:,:max_sent_len],
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }
        else:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len,
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }

    # For Longformer, need to do the global attention mask.
    if "longformer" in tokenizer.name_or_path:
        encoded_dict["global_attention_mask"] = make_global_attention_mask(
            encoded_dict["input_ids"], tokenizer.sep_token_id)

    return encoded_dict

def token_idx_by_sentence(input_ids, sep_token_id, model_name):
    """
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    padding_idx = -1
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        # Since `longformer` has the same tokenizer as RoBERTa, this should work.
        # NOTE(dwadden) We take paragraph [1:] because the RoBERTa tokenizer
        # puts two sep tokens between the claim and the evidence doc.
        if "roberta" in model_name or "longformer" in model_name:
            paragraph = paragraph[1:]
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-1)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch>=0)

    return batch_indices.long(), indices_by_batch.long(), mask.long()

def get_sep_tokens(input_ids, sep_token_id, model_name):
    sep_tokens = torch.where((input_ids == sep_token_id).bool())[1]
    # Take the end of sentence token as the representation for each sentence.
    sep_indices = sep_tokens[2:]
    # Return a list to be consistent with `token_idx_by_sentence`.
    return [sep_indices.unsqueeze(0)]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="fever/fever_train_retrieved_5.jsonl")
    argparser.add_argument('--pre_trained_model', type=str)
    #argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str, default="fever/fever_dev_retrieved_5.jsonl")
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=10, help="Training epoch")
    argparser.add_argument('--max_sent_len', type=int, default=512)
    argparser.add_argument('--loss_ratio', type=float, default=5)
    argparser.add_argument('--checkpoint', type=str, default = "fever_roberta_joint_paragraph_dynamic")
    argparser.add_argument('--log_file', type=str, default = "fever_joint_paragraph_performances.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--k', type=int, default=0)
    argparser.add_argument('--evaluation_step', type=int, default=50000)
    argparser.add_argument("--device", default=0)
    argparser.add_argument("--model_type", type=str, default="dynamic")
    argparser.add_argument("--to_console", action="store_true")
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    reset_random_seed(12345)

    args = argparser.parse_args()

    # The code I added only works for a batch size of 1.
    assert args.batch_size == 1

    with open(args.checkpoint+".log", 'w') as f:
        # If we're debugging, don't redirect.
        if not args.to_console:
            sys.stdout = f

        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.repfile)

        if args.train_file:
            train = True
            #assert args.repfile is not None, "Word embedding file required for training."
        else:
            train = False
        if args.test_file:
            test = True
        else:
            test = False

        params = vars(args)

        for k,v in params.items():
            print(k,v)

        if train:
            train_set = FEVERParagraphBatchDataset(args.train_file,
                                                     sep_token = tokenizer.sep_token, k=args.k)
        dev_set = FEVERParagraphBatchDataset(args.test_file,
                                               sep_token = tokenizer.sep_token, k=args.k)

        print("Loaded dataset!")

        assert args.model_type in ["onepass", "dynamic"]
        if args.model_type == "onepass":
            model = OnePassParagraphClassifier(args.repfile, args.bert_dim, args.dropout).to(device)
        else:
            model = JointParagraphClassifier(args.repfile, args.bert_dim,
                                            args.dropout).to(device)

        # NOTE(dwadden) For some reason, the `position_ids` are missing. Deal
        # with this by just using the ones from the original model; they
        # shouldn't change.
        if args.pre_trained_model is not None:
            loaded = torch.load(args.pre_trained_model)
            if "bert.embeddings.position_ids" not in loaded:
                loaded["bert.embeddings.position_ids"] = model.state_dict()["bert.embeddings.position_ids"]

            model.load_state_dict(loaded)
            print("Loaded pre-trained model.")

        if train:
            settings = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
            for module in model.extra_modules:
                settings.append({'params': module.parameters(), 'lr': args.lr})
            optimizer = torch.optim.Adam(settings)
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epoch)

            #if torch.cuda.device_count() > 1:
            #    print("Let's use", torch.cuda.device_count(), "GPUs!")
            #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #    model = nn.DataParallel(model)

            model.train()

            for epoch in range(args.epoch):
                error_count = 0
                sample_p = schedule_sample_p(epoch, args.epoch)
                tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
                for i, batch in enumerate(tq):
                    encoded_dict = encode(tokenizer, batch, args.max_sent_len, args.model_type)
                    # NOTE(dwadden) If we're doing the `onepass` model, just get
                    # the indices of the sep tokens.
                    if isinstance(model, OnePassParagraphClassifier):
                        transformation_indices = get_sep_tokens(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                    else:
                        # Otherwise, get the indices for each sentence in the evidence.
                        transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                    transformation_indices = [tensor.to(device) for tensor in transformation_indices]

                    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                    stance_label = batch["stance"].to(device)
                    padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
                    if padded_rationale_label.size(1) == transformation_indices[-1].size(1): # Skip some rare cases with inconsistent input size.
                        rationale_out, stance_out, rationale_loss, stance_loss = \
                            model(encoded_dict, transformation_indices, stance_label = stance_label,
                                  rationale_label = padded_rationale_label.to(device), sample_p = sample_p)
                        rationale_loss *= args.loss_ratio
                        loss = rationale_loss + stance_loss
                        loss.sum().backward()
                    else:
                        error_count += 1

                    if i % args.update_step == args.update_step - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}, stance loss: {round(stance_loss.item(), 4)}, rationale loss: {round(rationale_loss.item(), 4)}')


                    if i % args.evaluation_step == args.evaluation_step-1:
                        torch.save(model.state_dict(), args.checkpoint+"_"+str(epoch)+"_"+str(i)+".model")

                        # Evaluation
                        subset_train = Subset(train_set, range(0, 1000))
                        train_score = evaluation(model, subset_train)
                        print(f'Epoch {epoch}, step {i}, train stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % train_score)

                        subset_dev = Subset(dev_set, range(0, 1000))
                        dev_score = evaluation(model, subset_dev)
                        print(f'Epoch {epoch}, step {i}, dev stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)
                scheduler.step()
                torch.save(model.state_dict(), args.checkpoint+"_"+str(epoch)+".model")
                print(error_count, "mismatch occurred.")

                # Evaluation
                subset_train = Subset(train_set, range(0, 10000))
                train_score = evaluation(model, subset_train)
                print(f'Epoch {epoch}, train stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % train_score)

                subset_dev = Subset(dev_set, range(0, 10000))
                dev_score = evaluation(model, subset_dev)
                print(f'Epoch {epoch}, dev stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)



        if test:
            # NOTE(dwadden) Don't need this.
            # model = JointParagraphClassifier(args.repfile, args.bert_dim,
            #                                   args.dropout).to(device)
            # model.load_state_dict(torch.load(args.pre_trained_model))


            # Evaluation
            print("Evaluating.")
            subset_dev = Subset(dev_set, range(0, 10000))
            dev_score = evaluation(model, subset_dev)
            print(f'Test stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)


            if train:
                params["stance_f1"] = dev_score[0]
                params["stance_precision"] = dev_score[1]
                params["stance_recall"] = dev_score[2]
                params["rationale_f1"] = dev_score[3]
                params["rationale_precision"] = dev_score[4]
                params["rationale_recalls"] = dev_score[5]

                with jsonlines.open(args.log_file, mode='a') as writer:
                    writer.write(params)
