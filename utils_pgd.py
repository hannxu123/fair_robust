import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import os
import argparse
from torchvision import datasets, transforms
import numpy as np



def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd


def in_class(predict, label):

    probs = []
    for i in range(10):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs.append(acc)

    return probs

def evaluate(model, test_loader, configs1, device):

    print('Doing test')
    model.eval()

    correct = 0
    correct_adv = 0

    all_label = []
    all_pred = []
    all_pred_adv = []

    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        all_label.append(target)

        ## clean test
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add = pred.eq(target.view_as(pred)).sum().item()
        correct = correct + add
        all_pred.append(pred)
        model.zero_grad()

        ## adv test
        adv_samples = pgd_attack(model, X = data, y = target, **configs1)
        output1 = model(adv_samples)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv = correct_adv + add1
        all_pred_adv.append(pred1)
        model.zero_grad()

    print('clean accuracy  = ' + str(correct / len(test_loader.dataset)), flush= True)
    print('adv accuracy  = ' + str(correct_adv / len(test_loader.dataset)), flush=True)

    clean_acc = correct / len(test_loader.dataset)
    adv_acc = correct_adv / len(test_loader.dataset)

    ## collect each class performance
    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()
    class_acc = in_class(all_pred, all_label)
    class_adv_acc = in_class(all_pred_adv, all_label)

    return ([clean_acc, adv_acc] + class_acc + class_adv_acc)

