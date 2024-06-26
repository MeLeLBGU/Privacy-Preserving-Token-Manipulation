import os
import re
from torch.utils.data import Dataset
from datasets import load_dataset
import remap_base
import pickle
from utils import find_matching_positions
import time
import logging
log = logging.getLogger(__name__)
import lzma

class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer, type, remapper, skip = False):
        self.dataset = dataset
        self.remapper = remapper
        self.tokenizer = tokenizer

        if "sst2" in dataset:
            text_key = "sentence"
            self.max_len = 66
            if type == "test":
                type = "validation"
        if "imdb" in dataset:
            text_key = "text"
            if type == "validation":
                type = "test"
            self.max_len = 512
        if "sst2" in dataset or "imdb" in dataset:
            data = load_dataset(dataset) 
            texts = [sent[text_key] for sent in data[type]]
            labels = [sent["label"] for sent in data[type]]
            self.label_1 = "positive"
            self.label_0 = "negative"
        if "qnli" in dataset:
            if type == "test" or type == "validation":
                type = "validation"
            max_len = 512
            self.max_len = max_len

            data = load_dataset("glue", dataset) 
            data = data[type]
            self.data = data
            texts = []
            labels = []
            for q, s, l in zip(data["question"], data["sentence"], data["label"]):
                line = "qnli question: " + q + " sentence: " + s
                texts.append(line)
                labels.append(l)

            self.tokenized_q = tokenizer.batch_encode_plus(
                ["qnli question: "],add_special_tokens=False, pad_to_max_length=False, return_tensors="pt")

            self.tokenized_s = tokenizer.batch_encode_plus(
                ["sentence: "],add_special_tokens=False, pad_to_max_length=False, return_tensors="pt")
            self.lenq = len(self.tokenized_q["input_ids"][0])
            self.lens = len(self.tokenized_s["input_ids"][0])
            self.label_1 = "yes" #"entailment"
            self.label_0 = "no"#"not_entailment"
        self.texts = texts
        self.labels = labels
        
        self.inputs = []
        self.targets = []
        if not isinstance(remapper, remap_base.RemapConv) and not isinstance(remapper,remap_base.RemapNoise):
            self.should_save = False
        else:
            if skip:
                self.fname = remapper.fname + "_" + type + "_skipword" ".pkl.xz"
            else:
                self.fname = remapper.fname + "_" + type + ".pkl.xz"
            print("Saving to pickle: ", self.fname)
            self.should_save = True
        self.t1 = 0
        if self.should_save:
            self.load() # try..
        if "sst2" in dataset or "imdb" in dataset:
            self._build_roberta()
        else:
            self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
    
    def _build(self):
    
        REPLACE_NO_SPACE = re.compile("[.;!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        already_remapped = len(self.inputs)
        self.t1 = time.time()
        mappings = 0
        for i, text in enumerate(self.texts):
            if i < already_remapped:
                mappings = mappings + 1
                continue
            line = text.strip()
            line = REPLACE_NO_SPACE.sub("", line) 
            line = REPLACE_WITH_SPACE.sub("", line)

            line = line + ' </s> ' #need to append

            if self.labels[i] == 0:
                label = self.label_0
            else:
                label = self.label_1
            target = label + " </s> "
            
            
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [line], max_length = self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            attention_mask = tokenized_inputs["attention_mask"][0]
            # QNLI adds some tokens that we do not want to change as it is part of the algorithm
            # : the qnli question and sentence, so we want to add them to the attention mask momentaraly for the
            # apply noise, because apply noise doesn't add noise to the attention mask 0 and doesn't consider it
            if "qnli" in self.dataset:
                attention_mask = self.hide_tokens(tokenized_inputs)
            tokenized_inputs["input_ids"][0] = self.remapper.apply_noise_to_input_ids(line, tokenized_inputs["input_ids"][0], attention_mask)

            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            if time.time() - self.t1 > 20600 and self.should_save:
                self.save()
                time.sleep(500)
                exit(1)
        if self.should_save and mappings+1 <= len(self.texts):
            self.save()

    def _build_roberta(self):
        already_remapped = len(self.inputs)
        self.t1 = time.time()
        mappings = 0
        for i, text in enumerate(self.texts):
            if i < already_remapped:
                mappings = mappings + 1
                continue
            line = text
            # line = text.strip()
            # line = REPLACE_NO_SPACE.sub("", line) 
            # line = REPLACE_WITH_SPACE.sub("", line)

            # line = line + ' </s> ' #need to append

            
            target = str(self.labels[i])
            
            
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [line], max_length = self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            attention_mask = tokenized_inputs["attention_mask"][0]
            # QNLI adds some tokens that we do not want to change as it is part of the algorithm
            # : the qnli question and sentence, so we want to add them to the attention mask momentaraly for the
            # apply noise, because apply noise doesn't add noise to the attention mask 0 and doesn't consider it
            tokenized_inputs["input_ids"][0] = self.remapper.apply_noise_to_input_ids(line, tokenized_inputs["input_ids"][0], attention_mask)
            # sent_data = {"attention_mask": tokenized_inputs["attention_mask"][0], "input_ids": tokenized_inputs["input_ids"][0]}

            # # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=1, pad_to_max_length=True, return_tensors="pt"
            )

            tokenized_targets["attention_mask"][0] = 1
            tokenized_targets["input_ids"][0] = self.labels[i]

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            if time.time() - self.t1 > 20600 and self.should_save:
                self.save()
                time.sleep(500)
                exit(1)
        if self.should_save and mappings+1 <= len(self.texts):
            self.save()

    def save(self):
        # with h5py.File(self.fname, 'a') as hdf5_file:
        #     print(self.inputs.shape)
        #     current_size1 = hdf5_file['inputs'].shape[0]
        #     current_size2 = hdf5_file['targets'].shape[0]
        #     hdf5_file['inputs'].resize(len(self.inputs), axis = 0)
        #     hdf5_file['targets'].resize(len(self.targets), axis = 0)
        #     print(hdf5_file['inputs'])
        #     hdf5_file['inputs'][current_size1:] = self.inputs[current_size1:]
        #     hdf5_file['targets'][current_size2:] = self.targets[current_size2:]
            
        with lzma.open(self.fname, "wb") as f:
            pickle.dump([self.inputs, self.targets], f)
            print("SAVING")


    def load(self):
        if os.path.exists(self.fname) and os.stat(self.fname).st_size > 1:
            with lzma.open(self.fname, "rb") as f:
                print("Already remapped this:", self.fname)
                x = pickle.load(f)
                self.inputs, self.targets = x[0], x[1]

    def hide_tokens(self, tokenized_line):
        idxq = find_matching_positions(tokenized_line["input_ids"][0].numpy(), self.tokenized_q["input_ids"][0].numpy())
        idxs = find_matching_positions(tokenized_line["input_ids"][0].numpy(), self.tokenized_s["input_ids"][0].numpy())
        attention_mask = tokenized_line["attention_mask"][0]
        attention_mask[idxq:idxq+self.lenq] = 0
        attention_mask[idxs:idxs+self.lens] = 0
        return attention_mask
