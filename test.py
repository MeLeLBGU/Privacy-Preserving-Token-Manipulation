import sys
sys.path.append("..")
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def predict(model, device, test_dataloader, test_y):
    model.eval()
    total_preds = []
    total_ids = []
    real_label = []
    real_ids = []
    # get predictions for test data
    for step, batch in enumerate(test_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            # preds = model(test_seq.to(device), test_mask.to(device))was
            preds = model(sent_id, mask)
            preds = preds.detach().cpu().numpy()
            print(preds)
            labels_acc = labels.detach().cpu().numpy()
            preds = np.argmax(preds, axis = 1)

            total_preds.extend(preds)
            # print(ids, labels)
            real_label.extend(labels.detach().cpu().numpy())

    cls_rpt = classification_report(test_y, total_preds)
    acc_score = accuracy_score(test_y, total_preds)

    print(cls_rpt)
    print(acc_score)

