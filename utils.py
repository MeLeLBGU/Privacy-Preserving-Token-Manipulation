import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from sklearn.utils.class_weight import compute_class_weight
# import parser

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier

def preprocess_text_for_bert(tokenizer, data, max_len):
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # sent = sent["sentence"]
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text = sent,  # Preprocess sentence
            add_special_tokens = True,  # Add `[CLS]` and `[SEP]`
            max_length = max_len,  # Max length to truncate/pad
            pad_to_max_length = True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask = True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



# create dataloader for training
def create_dataloader(train_input_ids, train_attention_mask, train_labels, batch_size):
    train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

    return train_data, train_sampler, train_dataloader

# get the maximum length of a word
def get_max_len(dataset, tokenizer):
    sentence_train = [sent["sentence"] for sent in dataset["train"]]
    sentence_validation = [sent["sentence"] for sent in dataset["validation"]]
    sentence_test = [sent["sentence"] for sent in dataset["test"]]

    data = np.concatenate([sentence_train, sentence_validation, sentence_test])

    # Encode our concatenated data
    encoded_data = [tokenizer.encode(sent, add_special_tokens = True) for sent in data]

    # Find the maximum length
    max_len = max([len(sent) for sent in encoded_data])
    print('Max length: ', max_len)
    return max_len

# create dictonairy with text and label from sst2-like datasets
def create_data(dataset, type):
    sentence_train = [sent["sentence"] for sent in dataset[type]]
    labels_train = [sent["label"] for sent in dataset[type]]
    df = {}
    df["text"] = sentence_train
    df["label"] = labels_train
    return df
