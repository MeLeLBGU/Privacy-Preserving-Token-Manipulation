import argparse
import sys

# import pandas as pd
import torch.nn as nn
import train as training
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier
import train_new as train2
import test as predict
from utils import *
import naive_attacker
import logging as log
import pickle
from remap_base import *
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
                        dest = 'predict', help = 'Predict phase. (default: False)')
    parser.add_argument('--vocab_type', default = "token", type = str,
                        dest = 'vocab_type', help = 'The vocab type, by token? by word? Dont need to change this, it is set to token (input if)', choices=["token", "word"])
    parser.add_argument('--remap_count', default = 2, type = int,
                        dest = 'remap_count', help = 'How many tokens are we going to remap (default: 2)')
    parser.add_argument('--save', default = "result.pt", type = str,
                        dest = 'save', help = 'Save mode path. (default: result.pt)')
    parser.add_argument('--remap', default = "validation", type = str,
                        dest = 'remap', help = 'Remap the validation/text set or all the sets. (default: validation)', choices=["validation", "all"])
    parser.add_argument('--remap_type', default = "random", type = str,
                        dest = 'remap_type', help = 'what type of remap. freq-high is mapping low to high (default: random)', choices=["random", "freq-high","freq-low", "none"])
    parser.add_argument('--cpu', default = False, action = "store_true",
                        dest = 'cpu', help = 'Use cpu instead of a device')
    parser.add_argument('--attacker', default = False, action = "store_true",
                        dest = 'attacker', help = 'Initiate attack mode :), get the sentence *')
    parser.add_argument('--dataset', default = "sst2", type = str,
                        dest = 'dataset', help = 'What database to use', choices=["sst2", "imdb"])
    parser.add_argument('--frequency_path', default = "wiki_freq.pkl", type = str,
                        dest = 'frequency_path', help = 'Path to input ids frequency. (default: "wiki_freq.pkl" - no path).')
    parser.add_argument('--frequency_window', default = "all", type = str,
                        dest = 'frequency_window', help = 'Path to input ids frequency. (default: "all")')
    parser.add_argument('--model', default = "bert-base-uncased", type = str,
                        dest = 'model', help = 'What base model to use')
    parser.add_argument('--finetune', default = False, action="store_true",
                        dest = 'finetune', help = 'What base model to use')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(args.model)
    #  data = load_dataset(args.dataset)
    fname = args.save.split(".pt")[0] + ".log"
    print(fname)
    log.basicConfig(filename=fname, level=log.INFO)
    # prep input
    batch_size = 32
    if not args.cpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Get the vocabulary of the model, it is simply an
    if args.vocab_type == "token":
        vocab = create_vocabulary(tokenizer=tokenizer)  # bert vocab

    if args.remap_type == "random":
        remapper = RemapRandom(vocab)
    elif "freq" in args.remap_type:
        remapper = RemapFrequency(vocab, args.frequency_path, args.remap_type, args.frequency_window)

    # training mode
    if not args.predict:
        print("Training Model")
        # data = load_dataset(args.dataset)
        # this is just to create the data from downstream task
        train, validation = create_data(args.dataset)  # READ DATA SST2
        # validation = create_data(data, "validation")
        train_text = train["text"]
        val_text = validation["text"]
        train_labels = torch.tensor(train["label"])
        val_labels = torch.tensor(validation["label"])

        # preprocess the text to create the input ids
        train_input_ids, train_attention_mask = encode_text(tokenizer, train_text, MAX_LEN)
        val_input_ids, val_attention_mask = encode_text(tokenizer, val_text, MAX_LEN)

        # Now remap the tokens to the new tokens
        if not args.finetune:
            if args.remap == "validation" or args.remap == "all":
                val_input_ids = remapper.remap_input_ids(val_input_ids)
            if args.remap == "all":
                train_input_ids = remapper.remap_input_ids(train_input_ids)

        if args.attacker:
            naive_attacker.main(tokenizer, device, train_input_ids, val_input_ids, remapper, args.remap_count)
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
