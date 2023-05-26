import argparse
import sys

# import pandas as pd
import torch.nn as nn
import train as training
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier
import train_new as train2
import test as predict
from utils import *
import text_manipulation

MAX_LEN = 66

device = torch.device("cuda")


def initialize_model(epochs, dataloader):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert = False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr = 1e-5,  # Default learning rate
                      eps = 1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,  # Default value
                                                num_training_steps = total_steps)
    loss_fn = nn.CrossEntropyLoss()

    return bert_classifier, optimizer, scheduler, loss_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', default = False, action = "store_true",
                        dest = 'predict', help = 'Predict phase.')
    parser.add_argument('--shuffle', default = False, action = "store_true",
                        dest = 'shuffle', help = 'shuffle the voac phase.')
    parser.add_argument('--vocab_type', default = "token", type = str,
                        dest = 'vocab_type', help = 'The vocab type, by token? by word?')
    parser.add_argument('--remap', default = 2, type = int,
                        dest = 'remap', help = 'How many tokens are we going to remap')
    parser.add_argument('--save', default = "result.pt", type = str,
                        dest = 'save', help = 'Predict phase.')

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    data = load_dataset('sst2')

    # prep input
    batch_size = 32

    # training mode
    if not args.predict:
        print("Training Model")
        train = create_data(data, "train")  # READ DATA SST2
        validation = create_data(data, "validation")
        train_text = train["text"]
        val_text = validation["text"]

        if args.vocab_type == "token":
            vocab = text_manipulation.create_vocabulary(tokenizer = tokenizer)  # bert vocab
        if args.shuffle:
            vocab = text_manipulation.shuffle_vocab(vocab)  # shuffle the vocabulary
        vocab = text_manipulation.remap_vocab(vocab, args.remap)  # remap the vocabulary in some form of ratio

        train_input_ids, train_attention_mask = preprocess_text_for_bert(tokenizer, train_text, MAX_LEN)
        text_manipulation.remap_input_ids(train_input_ids, vocab)
        exit(1)
        val_input_ids, val_attention_mask = preprocess_text_for_bert(tokenizer, val_text, MAX_LEN)

        train_labels = torch.tensor(train["label"])
        val_labels = torch.tensor(validation["label"])
        # Create the DataLoader for our training set
        train_data, train_sampler, train_dataloader = create_dataloader(train_input_ids, train_attention_mask,
                                                                        train_labels, batch_size)
        val_data, val_sampler, val_dataloader = create_dataloader(val_input_ids, val_attention_mask, val_labels,
                                                                  batch_size)
        bert_classifier, optimizer, scheduler, loss = initialize_model(2, train_dataloader)

        train2.train(bert_classifier, epochs = 3, train_dataloader = train_dataloader, device = device,
                     cross_entropy = loss, optimizer = optimizer
                     , val_dataloader = val_dataloader, save = args.save)
    else:
        print("Predicting model")
        test = create_data(data, "validation")
        test_input_ids, test_attention_mask = preprocess_text_for_bert(tokenizer, test["text"], MAX_LEN)
        test_labels = torch.tensor(test["label"])
        test_data, test_sampler, test_dataloader = create_dataloader(test_input_ids, test_attention_mask, test_labels,
                                                                     batch_size)
        bert_classifier, optimizer, scheduler, loss = initialize_model(2, test_dataloader)
        predict.predict(bert_classifier, device, test_dataloader, test_labels)
    training.set_seed(42)  # Set seed for reproducibility
