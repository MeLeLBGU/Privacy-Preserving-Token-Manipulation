import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score
import os

def train(model, epochs, train_dataloader, device, cross_entropy, optimizer, val_dataloader, save, tokenizer=None):
    train_losses = []
    valid_losses = []
    best_train = 0
    best_acc = 0
    # set initial loss to infinite
    best_valid_loss = float('inf')
    
    result_name = save.split(".pt")[0] + ".txt"
    if os.path.exists(result_name):
        with open(result_name, "r") as f:
            if len(f.readlines()) >= 5:
                print("Already did this!")
                exit(0)
    f = open(result_name, "w")
    f.writelines("Train | Valid | Accuracy\n")
    f.close()
    if "t5" in save or "qnli" in save:
        t5 = True
        pad_token_id = tokenizer.pad_token_id
    else:
        t5 = False
    # for each epoch
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        # train model
        if t5:
            train_loss = train_epoch_t5(model, train_dataloader, device, optimizer, pad_token_id)
            valid_loss, acc_score = evaluate_t5(model, val_dataloader, device, tokenizer)
        else:
            train_loss = train_epoch(model, train_dataloader, device, cross_entropy, optimizer)
            # evaluate model
            valid_loss, acc_score = evaluate(model, val_dataloader, device, cross_entropy)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'Accuracy: {acc_score:.3f}')
        with open(result_name, "a") as f:
            f.writelines(f'{train_loss:.3f} | {valid_loss:.3f} | {acc_score:.3f}')
            f.writelines("\n")


def train_epoch(model, train_dataloader, device, cross_entropy, optimizer):
    model.train()
    total_loss = 0
    # empty list to save model predictions
    # iterate over batches
    for step, batch in enumerate(train_dataloader):
        # progress update after every 50 batches.

        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        # push the batch to gpu
        #batch = [r.to(device) for r in batch]
        sent_id = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long) #batch["source_ids"].to(device)
        labels = batch['target_ids'].to(device, dtype = torch.long)#batch["target_ids"]
        # sent_id, mask, labels = batch["input_ids"], batch["attention_mask"], batch[""]
        # clear previously calculated gradients
        model.zero_grad()
        #if labels[:, 0] != labels[:, 1]  or labels[:,1] != lables[:,2]:
        #if not (labels[:, 0] == labels).all(dim=1):

         #   print("bad labels")
         #   exit(1)
        labels = labels[:, 0]#labels.squeeze(1)
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values
        #print(preds, labels)

        loss = cross_entropy(preds, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)
    
    # returns the loss and predictions
    return avg_loss


# function for evaluating the model
def evaluate(model, val_dataloader, device, cross_entropy):
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    acc_score = 0
    # empty list to save the model predictions
    total_preds = []
    t0 = time.time()
    # iterate over batches
    for step, batch in enumerate(val_dataloader):
        # Progress update every 50 batches.
        
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        #batch = [t.to(device) for t in batch]
        sent_id = batch['source_ids'].to(device, dtype = torch.long)
        mask = batch['source_mask'].to(device, dtype = torch.long) #batch["source_ids"].to(device)
        labels = batch['target_ids'].to(device, dtype = torch.long)#
        labels = labels[:, 0]
#sent_id = batch["source_ids"]
        #mask = batch["source_mask"]
        #labels = batch["target_ids"]

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()
            labels_acc = labels.detach().cpu().numpy()

            # print("Accuracy:", accuracy_score(labels_acc, np.argmax(preds, axis = 1)))
            acc_score = acc_score + accuracy_score(labels_acc, np.argmax(preds, axis = 1))
            #total_preds.append(preds)
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)
    acc_score = acc_score / len(val_dataloader)
    # reshape the predictions in form of (number of samples, no. of classes)
   # total_preds = np.concatenate(total_preds, axis = 0)

    return avg_loss, acc_score

def train_epoch_t5(model, train_dataloader, device, optimizer, pad_token_id):
    model.train()
    total_loss = 0
    # iterate over batches
    for step, data in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step > 500000:
            break
        if step % 500 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        loss = step_t5(model, device, pad_token_id, data)

        total_loss = total_loss + loss #.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(train_dataloader)
    print(avg_loss)
    # returns the loss and predictions
    return avg_loss

def evaluate_t5(model, val_dataloader, device, tokenizer):
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    acc_score = 0
    # empty list to save the model predictions
    total_preds = []
    pad_token_id = tokenizer.pad_token_id
    # iterate over batches
    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            # Progress update every 50 batches.
            if step % 100 == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
            loss = step_t5(model, device, pad_token_id, data)
            total_loss = total_loss + loss
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length = 2, 
                early_stopping=True
            )
            output = [tokenizer.decode(o) for o in generated_ids]
            target = [tokenizer.decode(o) for o in y]
            preds = []
            for o in output:
                preds.append(o.split('<pad> ')[1])
            targets = []
            for t in target:
                targets.append(t.split('</s>')[0])
            acc_score = acc_score + sum(x == y for x, y in zip(targets, preds)) 
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)
    acc_score = acc_score / len(val_dataloader) / 8
    # reshape the predictions in form of (number of samples, no. of classes)

    return avg_loss, acc_score

def step_t5(model, device, pad_token_id, data):
    y = data["target_ids"].to(device, dtype=torch.long)
    lm_labels = y[:, :].clone().detach()
    lm_labels[y[:, :] == pad_token_id] = -100
    
    ids = data["source_ids"].to(device, dtype=torch.long)
    mask = data["source_mask"].to(device, dtype=torch.long)
    outputs = model(
        input_ids=ids,
        attention_mask=mask,
        decoder_attention_mask=data['target_mask'].to(device, dtype=torch.long),
        labels=lm_labels
    )
    loss = outputs[0]
    return loss
