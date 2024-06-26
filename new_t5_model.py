import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

def initialize_model_t5(device, dataloader, epochs, modell):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    model = T5ForConditionalGeneration.from_pretrained(modell)
    # Tell PyTorch to run the model on GPU
    model.to(device)
    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr = 3e-4,  # Default learning rate
                      )

    # Total number of training steps
    total_steps = len(dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,  # Default value
                                                num_training_steps = total_steps)

    return model, optimizer, scheduler

def create_dataloader_t5(train_dataset, val_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last=True, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    return train_dataloader, val_dataloader

# def create_dataloader_roberta(train_dataset, val_dataset, batch_size):
#     train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last=True, shuffle=True, num_workers=0)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
#     return train_dataloader, val_dataloader
