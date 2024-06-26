import argparse
import torch.nn as nn
# from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier
import train_new as train2
#import test as predict
from utils import *
import naive_attacker as naive_attacker
import logging
# log = logging.getLogger(__name__)

import copy
import nn_attacker as nn_attacker
from remap_base import *
import MyDataset
# MAX_LEN = 512
import new_t5_model

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



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
                        dest = 'remap_type', help = 'what type of remap. freq-high is mapping low to high (default: random)', choices=["random", "freq-high","freq-low", "conv", "noise", "none"])
    parser.add_argument('--cpu', default = False, action = "store_true",
                        dest = 'cpu', help = 'Use cpu instead of a device')
    parser.add_argument('--attacker', default = "", type = str,
                        dest = 'attacker', help = 'Initiate attacker. (default: naive)', choices=["naive", "knn", "stencil"])
    parser.add_argument('--dataset', default = "sst2", type = str,
                        dest = 'dataset', help = 'What database to use. (default: sst2)', choices=["sst2", "imdb", "qnli"])
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
    parser.add_argument('--stencil_sigma', default = 1, type = float,
                        dest = 'stencil_sigma', help = 'How many values to consider in the stencil')
    parser.add_argument('--eta', default = 1, type = int,
                        dest = 'eta', help = 'The eta parameter')
    parser.add_argument('--print', default=False, action="store_true", dest="print")
    parser.add_argument('--skip_stop_words', default=False, action="store_true", dest="skip_stop_words")
    parser.add_argument('--punctuated', default=False, action="store_true", dest="punctuated")
    args = parser.parse_args()
    # log.basicConfig(
    #     format="%(asctime)s -  %(message)s",
    #     datefmt="%d/%m/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    print("You choose to run with:")
    print("Model:", args.model)
    print("Attacking?:", args.attacker)
    
    if args.remap == "all":
        print("The privatization will be applied on: Train+Test")
    else:
        print("The privatization will be applied on: Test")
    print("Dataset:", args.dataset)
    print("The privatization technique:", args.remap_type)
    if args.remap_type == "noise":
        print("Eta:", args.eta)
    if args.remap_type == "conv":
        print("Sigma:", args.stencil_sigma)
    
    print("Saving results:", args.save)
    if args.attacker == "naive":
        attacker_file = "attacker"
    else:
        attacker_file = "attacker" + "_" + args.attacker
    is_naive_attacker = True if args.attacker == "naive" else False
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.dataset == "sst2":
        MAX_LEN = 66
    else:
        MAX_LEN = 512

    # fname = args.save.split(".pt")[0] + ".log"
    # f = open(fname, "w")
    # f.close()
    # print(fname)
    # log.basicConfig(filename=fname, level=print)
    # prep input
    batch_size = 8
    if not args.cpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Get the vocabulary of the model, it is simply an
    if args.vocab_type == "token":
        vocab = create_vocabulary(tokenizer=tokenizer)  # bert vocab
    remapper = RemapBase()
    if args.remap_type == "random":
        remapper = RemapRandom(vocab, forbidden_tokens=is_naive_attacker, remap_count=args.remap_count)
    elif "freq" in args.remap_type:
        remapper = RemapFrequency(vocab, args.frequency_path, args.remap_type, forbid=is_naive_attacker, window=args.frequency_window)
    elif "conv" in args.remap_type:
        remapper = RemapConv(args.model, args.dataset, args.stencil_size, args.stencil_stride, args.stencil_sigma, punctuated=args.punctuated, stopwords=args.skip_stop_words)
    elif "noise" in args.remap_type:
        remapper = RemapNoise(args.model, args.dataset, eta=args.eta, stop_words=args.skip_stop_words)

    attacker_file = attacker_file + "_" + str(args.remap_count) + args.remap_type
    attacker_file = attacker_file + "_" + args.dataset
    attacker_file = attacker_file + "_" + args.model
    
    # training mode
    if not args.predict:
        print("Training Model")
        # data = load_dataset(args.dataset)
        # this is just to create the data from downstream task
        if "t5" not in args.model:
            train, validation = create_data(args.dataset)  # READ DATA SST2
            # validation = create_data(data, "validation")
            train_text = train["text"]
            val_text = validation["text"]
            train_labels = torch.tensor(train["label"])
            val_labels = torch.tensor(validation["label"])
        if args.print:
            remapper.print_noise_text(tokenizer, train_text)
            exit(1)
        # preprocess the text to create the input ids
        # note that if we are in attacking mode, we want to truncate all the special tokens, and the padding, so we sent as an argument "not args.attacker" which tells the
        # encode_text to remove these tokens
        # if "t5" in args.model:
            # dataset, tokenizer, type, remapper, max_len=66):
        if (args.remap == "validation" or args.remap == "all") and not args.finetune:
            val_dataset = MyDataset.MyDataset(args.dataset, tokenizer, "validation", remapper, skip=args.skip_stop_words)    
        else:
            val_dataset = MyDataset.MyDataset(args.dataset, tokenizer, "validation", RemapBase(), skip=args.skip_stop_words)
        if (args.remap == "all" or args.remap == "train") and not args.finetune:
            train_dataset = MyDataset.MyDataset(args.dataset, tokenizer, "train", remapper, skip=args.skip_stop_words)
        elif args.attacker == "":
            train_dataset = MyDataset.MyDataset(args.dataset, tokenizer, "train", RemapBase(), skip=args.skip_stop_words)
        
        if args.attacker != "": # if we have an attacker we want the original data
            if args.remap == "all":
                original_train_dataset = MyDataset.MyDataset(args.dataset, tokenizer, "train", RemapBase(), skip=args.skip_stop_words)
            else: 
                original_train_dataset = MyDataset.MyDataset(args.dataset, tokenizer, "validation", RemapBase(), skip=args.skip_stop_words)
        # else:
        #     train_input_ids, train_attention_mask = encode_text(tokenizer, train_text, MAX_LEN, not is_naive_attacker)
        #     val_input_ids, val_attention_mask = encode_text(tokenizer, val_text, MAX_LEN, not is_naive_attacker)
        #     if args.attacker != "":
        #         original_train_input_ids = copy.deepcopy(train_input_ids) # save the original tokens!
        # # Now remap the tokens to the new tokens
        #     if not args.finetune:
        #         if args.remap == "validation" or args.remap == "all":
        #             val_input_ids, val_attention_mask = remapper.remap_input_ids(val_input_ids, val_attention_mask, "val")
        #         if args.remap == "all" or args.remap == "train":
        #             train_input_ids, train_attention_mask = remapper.remap_input_ids(train_input_ids, train_attention_mask, "train")

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
            
            else: # args.attacker == "knn":
                attacker_file = remapper.get_file_name().split('.pkl')[0]
                attacker_file = "attacker_" + attacker_file + ".json"
                if True:#"t5" in args.model:
                    train_attention_mask = []
                    train_input_ids = []
                    original_train_input_ids = []
                    print("Getting the train ids")
                    if args.remap == "all":
                        inputss = train_dataset
                    else:
                        inputss = val_dataset
                    for inp in inputss.inputs:
                        train_attention_mask.append(inp["attention_mask"].numpy()[0])
                        train_input_ids.append(inp["input_ids"].numpy()[0])
                    for inp in original_train_dataset.inputs:
                        original_train_input_ids.append(inp["input_ids"].numpy()[0])
                nn_attacker.main(args.model, tokenizer, train_attention_mask, train_input_ids, original_train_input_ids, attacker_file, remapper)

            exit(0)
        # Create the DataLoader for our training set
        if "t5" not in args.model:
            # train_data, train_sampler, train_dataloader = create_dataloader(train_input_ids, train_attention_mask,
            #                                                                 train_labels, batch_size)
            # val_data, val_sampler, val_dataloader = create_dataloader(val_input_ids, val_attention_mask, val_labels,
            #                                                         batch_size)
            train_dataloader, val_dataloader = new_t5_model.create_dataloader_t5(train_dataset, val_dataset, batch_size)

            bert_classifier, optimizer, scheduler, loss = initialize_model(args.model, epochs=10, dataloader=train_dataloader)
            print("Starting to train!")
            train2.train(bert_classifier, epochs = 10, train_dataloader = train_dataloader, device = device,
                        cross_entropy = loss, optimizer = optimizer
                        , val_dataloader = val_dataloader, save = args.save)
        else:
            # checkpoint_callback = ModelCheckpoint(monitor="val_loss",save_weights_only=True)
            # train_params = dict(
            # accumulate_grad_batches=8,
            # max_epochs=8, num_sanity_val_steps=0
            #                         #callbacks=checkpoint_callback
            # )
            if args.dataset == "sst2":
                max_len = 66
            else:
                max_len = 512
            train_dataloader, val_dataloader = new_t5_model.create_dataloader_t5(train_dataset, val_dataset, batch_size)
            print(len(train_dataloader), len(val_dataloader))
            model, optimizer, scheduler = new_t5_model.initialize_model_t5(device, train_dataloader, epochs=5,modell=args.model)
            train2.train(model, epochs = 5, train_dataloader = train_dataloader, device = device,
                        cross_entropy = None, optimizer = optimizer
                        , val_dataloader = val_dataloader, save = args.save, tokenizer=tokenizer)
            
            # model = t5_model.T5FineTuner(model_name='t5-base', fname=fname.split('.log')[0] + '.txt'
            #         , MAX_LEN = max_len, batch_size = 4, epochs=5, train_dataset=train_dataset, val_dataset=val_dataset)
            # trainer = pl.Trainer(**train_params)
            # trainer.fit(model)   
