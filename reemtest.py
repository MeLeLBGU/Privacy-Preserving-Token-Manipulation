import sys
sys.path.append("lm-scorer/")
from transformers import AutoModel, AutoTokenizer
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer
import time
import utils
import MyDataset
import t5_model 
from scipy.stats import gamma
import pytorch_lightning as pl

import numpy as np
tok = AutoTokenizer.from_pretrained("t5-base")
MAX_LEN = 66
batch_size = 16
epochs = 2

train_dataset = MyDataset.MyDataset("sst2", tok, "train")
val_dataset = MyDataset.MyDataset("sst2", tok, "train")
model = t5_model.T5FineTuner(hparams=hparams, train_dataset=train_dataset, val_dataset=val_dataset)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filename='reemtest_t5', monitor="val_loss", mode="min", save_top_k=1
)
train_params = dict(
    accumulate_grad_batches=16,
    max_epochs=2,
    precision= 32
    #callbacks=checkpoint_callback
    )

trainer = pl.Trainer(**train_params)
trainer.fit(model)
