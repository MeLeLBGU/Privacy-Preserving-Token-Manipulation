import argparse
import sys

# import pandas as pd
import torch.nn as nn
import train as training
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoModel
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier
import train_new as train2
import test as predict
from utils import *
import naive_attacker as naive_attacker
import logging as log
import pickle
import copy
import nn_attacker as nn_attacker
from remap_base import *
# MAX_LEN = 512


def initialize_model(model, epochs, dataloader):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(model, freeze_bert = False)

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
    parser.add_argument('--remap', default = "all", type = str,
                        dest = 'remap', help = 'Remap the validation/text set or all the sets. (default: all)', choices=["validation", "all", "train"])
    parser.add_argument('--remap_type', default = "random", type = str,
                        dest = 'remap_type', help = 'what type of remap. freq-high is mapping low to high (default: random)', choices=["random", "freq-high","freq-low", "conv", "none"])
    parser.add_argument('--cpu', default = False, action = "store_true",
                        dest = 'cpu', help = 'Use cpu instead of a device')
    parser.add_argument('--attacker', default = "", type = str,
                        dest = 'attacker', help = 'Initiate attacker. (default: naive)', choices=["naive", "knn"])
    parser.add_argument('--dataset', default = "sst2", type = str,
                        dest = 'dataset', help = 'What database to use. (default: sst2)', choices=["sst2", "imdb"])
    parser.add_argument('--frequency_path', default = "wiki_freq.pkl", type = str,
                        dest = 'frequency_path', help = 'Path to input ids frequency. (default: "wiki_freq.pkl" - no path).')
    parser.add_argument('--frequency_window', default = "all", type = str,
                        dest = 'frequency_window', help = 'What window. (default: "all")')
    parser.add_argument('--model', default = "roberta-base", type = str,
                        dest = 'model', help = 'What base model to use')
    parser.add_argument('--segment', default = "", type = str,
                        dest = 'segment', help = 'What segment of data')
    parser.add_argument('--finetune', default = False, action="store_true",
                        dest = 'finetune', help = 'What base model to use')
    parser.add_argument('--stencil_size', default = 3, type = int,
                        dest = 'stencil_size', help = 'How many values to consider in the stencil')
    parser.add_argument('--stencil_stride', default = 1, type = int,
                        dest = 'stencil_stride', help = 'How many values to consider in the stencil')

    args = parser.parse_args()
    if args.attacker == "naive":
        attacker_file = "attacker"
    else:
        attacker_file = "attacker" + "_" + args.attacker
    is_naive_attacker = True if args.attacker == "naive" else False
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = AutoModel.from_pretrained(args.model)
    #  data = load_dataset(args.dataset)
    if args.dataset == "sst2":
        MAX_LEN = 66
    else:
        MAX_LEN = 512
    fname = args.save.split(".pt")[0] + ".log"
    f = open(fname, "w")
    f.close()
    print(fname)
    log.basicConfig(filename=fname, level=log.INFO)
    # prep input
    batch_size = 8
    if not args.cpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Get the vocabulary of the model, it is simply an
    if args.vocab_type == "token":
        vocab = create_vocabulary(tokenizer=tokenizer)  # bert vocab

    if args.remap_type == "random":
        remapper = RemapRandom(vocab, forbidden_tokens=is_naive_attacker, remap_count=args.remap_count)
    elif "freq" in args.remap_type:
        remapper = RemapFrequency(vocab, args.frequency_path, args.remap_type, forbid=is_naive_attacker, window=args.frequency_window)
    elif "conv" in args.remap_type:
        remapper = RemapConv(args.model, args.dataset, args.stencil_size, args.stencil_stride)
    attacker_file = attacker_file + "_" + str(args.remap_count) + args.remap_type
    attacker_file = attacker_file + "_" + args.dataset
    
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
        # note that if we are in attacking mode, we want to truncate all the special tokens, and the padding, so we sent as an argument "not args.attacker" which tells the
        # encode_text to remove these tokens
        train_input_ids, train_attention_mask = encode_text(tokenizer, train_text, MAX_LEN, not is_naive_attacker)
        val_input_ids, val_attention_mask = encode_text(tokenizer, val_text, MAX_LEN, not is_naive_attacker)
        if args.attacker != "":
            original_train_input_ids = copy.deepcopy(train_input_ids) # save the original tokens!
        # Now remap the tokens to the new tokens
        if not args.finetune:
            if args.remap == "validation" or args.remap == "all":
                val_input_ids = remapper.remap_input_ids(val_input_ids, val_attention_mask, "val")
            if args.remap == "all" or args.remap == "train":
                train_input_ids = remapper.remap_input_ids(train_input_ids, train_attention_mask, "train")

        if args.attacker != "":
            print("Attack mode!")
            if args.segment != "":
                segment = int(args.segment)
                attacker_file = attacker_file + "_segment" + args.segment
                seg_len = int(len(train_input_ids) * (segment / 100))
                len10 = int(len(train_input_ids) * (10 / 100))
                train_input_ids = train_input_ids[seg_len: seg_len + len10 + 1]
                original_train_input_ids = original_train_input_ids[seg_len: seg_len + len10 + 1]
            attacker_file = attacker_file + ".json"
            
            if args.attacker == "naive":
                naive_attacker.main(tokenizer, device, train_input_ids, original_train_input_ids, remapper, args.remap_count, attacker_file)
            elif args.attacker == "knn":
                attacker_file = remapper.get_file_name().split('.pkl')[0]
                attacker_file = "attacker_" + attacker_file + ".json"
                nn_attacker.main(args.model, tokenizer, train_input_ids, original_train_input_ids, attacker_file)

            exit(0)
        # Create the DataLoader for our training set
        train_data, train_sampler, train_dataloader = create_dataloader(train_input_ids, train_attention_mask,
                                                                        train_labels, batch_size)
        val_data, val_sampler, val_dataloader = create_dataloader(val_input_ids, val_attention_mask, val_labels,
                                                                  batch_size)
        bert_classifier, optimizer, scheduler, loss = initialize_model(args.model, 3, train_dataloader)
        print("Starting to train!")
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
