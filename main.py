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
import naive_attacker
MAX_LEN = 66




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
    parser.add_argument('--remap_count', default = 2, type = int,
                        dest = 'remap_count', help = 'How many tokens are we going to remap')
    parser.add_argument('--save', default = "result.pt", type = str,
                        dest = 'save', help = 'Predict phase.')
    parser.add_argument('--remap', default = "validation", type = str,
                        dest = 'remap', help = 'Predict phase.')
    parser.add_argument('--remap_type', default = "random", type = str,
                        dest = 'remap_type', help = 'what type of remap.')
    parser.add_argument('--cpu', default = False, action = "store_true",
                        dest = 'cpu', help = 'Use cpu instead of a device')
    parser.add_argument('--attacker', default = False, action = "store_true",
                        dest = 'attacker', help = 'Initiate attack mode :), get the sentence *')

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    data = load_dataset('sst2')

    # prep input
    batch_size = 32
    if not args.cpu:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")


    # training mode
    if not args.predict:
        print("Training Model")
        train = create_data(data, "train")  # READ DATA SST2
        validation = create_data(data, "validation")
        train_text = train["text"]
        val_text = validation["text"]
        train_labels = torch.tensor(train["label"])
        val_labels = torch.tensor(validation["label"])

        # preprocess the text to create the input ids
        train_input_ids, train_attention_mask = preprocess_text_for_bert(tokenizer, train_text, MAX_LEN)
        val_input_ids, val_attention_mask = preprocess_text_for_bert(tokenizer, val_text, MAX_LEN)

        if args.vocab_type == "token":
            # create vocabulary from the tokenizer -- note that this is a list that each index is mapped
            # to an input id, which we will later remap. Given input id which is equal to the index
            # We remap it to the value in the vocab[input_id]
            vocab = text_manipulation.create_vocabulary(tokenizer = tokenizer)  # bert vocab
        if args.remap_type == "random":
            vocab, reverse_vocab = text_manipulation.remap_vocab(vocab, args.remap_count, args.shuffle)  # remap the vocabulary in some form of ratio
        else:
            vocab, reverse_vocab = text_manipulation.remap_vocab_by_frequency(vocab, train_input_ids, args.remap_type)  # remap the vocabulary in some form of ratio


        # Now remap the tokens to the new tokens
        if args.remap == "validation" or args.remap == "all":
            val_input_ids = text_manipulation.remap_input_ids(val_input_ids, vocab)
        if args.remap == "all":
            train_input_ids = text_manipulation.remap_input_ids(train_input_ids, vocab)

        if args.attacker:
            naive_attacker.main(tokenizer, device, train_input_ids, val_input_ids, reverse_vocab, args.remap_count)
            exit(0)
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
