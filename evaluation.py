import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn import metrics

# ------------------------------------------------------------
# Performance function
# ------------------------------------------------------------
def performance(ground_truth, prediction):
    gt_np = ground_truth.detach().cpu().numpy()
    pred_np = prediction.detach().cpu().numpy()
    pred_binary = torch.round(prediction).detach().cpu().numpy()

    fpr, tpr, _ = metrics.roc_curve(gt_np, pred_np)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(gt_np, pred_binary)
    precision = metrics.precision_score(gt_np, pred_binary)

    print(f"auc: {auc:.2f} | f1: {f1:.4f} | precision: {precision:.2f}")
    return round(auc, 2), round(precision, 2), round(f1, 2)

# ------------------------------------------------------------
# Loss function
# ------------------------------------------------------------
class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device, compressed=True):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device
        self.compressed = compressed

    def forward(self, pred, batch, true_labels=None):
        loss = 0
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        for student in range(pred.shape[0]):
            if not self.compressed:
                delta = batch[student][:, 0:self.num_of_questions] + batch[student][:, self.num_of_questions:]
                temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
                index = torch.arange(self.max_step - 1, device=self.device).unsqueeze(0)
                p = temp.gather(0, index)[0]
                a = (((batch[student][:, 0:self.num_of_questions] -
                       batch[student][:, self.num_of_questions:]).sum(1) + 1) // 2)[1:]
            else:
                p = pred[student][:self.max_step - 1][:, 0]
                a = true_labels[student]

            for i in reversed(range(len(p))):
                if p[i] > 0:
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break

            loss += self.crossEntropy(p.float(), a.float())
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        return loss, prediction, ground_truth

# ------------------------------------------------------------
# One ephoc train
# ------------------------------------------------------------
def train_epoch(model, trainLoader, optimizer, loss_func, device, model_type='RNN'):
    model.to(device).train()

    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        optimizer.zero_grad()

        if model_type == 'KQN':
            in_data, seq_len, next_skills, correctness, mask = [x.to(device) for x in batch]
            pred = model(in_data, seq_len, next_skills)
            loss = model.loss(pred, correctness, mask)

        elif model_type == 'LITE':
            batch = batch.to(device)
            pred = model(batch)
            loss, _, _ = loss_func(pred, batch)

        else:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, labels = [x.to(device) for x in batch]
                pred = model(data)
                if getattr(loss_func, "compressed", False):
                    loss, _, _ = loss_func(pred, labels, true_labels=labels)
                else:
                    loss, _, _ = loss_func(pred, labels)
            else:
                data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                pred = model(data)
                loss, _, _ = loss_func(pred, data)

        loss.backward()
        optimizer.step()

    return model, optimizer

# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------
def test_epoch(model, testLoader, loss_func, device, model_type='RNN'):
    model.to(device).eval()
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)

    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        with torch.no_grad():
            if model_type == 'KQN':
                in_data, seq_len, next_skills, correctness, mask = [x.to(device) for x in batch]
                pred = torch.sigmoid(model(in_data, seq_len, next_skills))

                if pred.dim() == 3:
                    pred = pred.squeeze(-1)

                for tensor_pair in [(mask, pred), (correctness, pred)]:
                    tensor, ref = tensor_pair
                    pad_len = ref.shape[1] - tensor.shape[1]
                    if pad_len > 0:
                        tensor = F.pad(tensor, (0, pad_len), value=0)
                    else:
                        tensor = tensor[:, :ref.shape[1]]

                preds = pred.masked_select(mask)
                labels = correctness.masked_select(mask)

            elif model_type == 'LITE':
                batch = batch.to(device)
                pred = model(batch)
                _, preds, labels = loss_func(pred, batch)

            else:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, labels = [x.to(device) for x in batch]
                    pred = model(data)
                    if getattr(loss_func, "compressed", False):
                        _, preds, labels = loss_func(pred, labels, true_labels=labels)
                    else:
                        _, preds, labels = loss_func(pred, labels)
                else:
                    data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    pred = model(data)
                    _, preds, labels = loss_func(pred, data)

        prediction = torch.cat([prediction, preds])
        ground_truth = torch.cat([ground_truth, labels])

    return performance(ground_truth, prediction)
