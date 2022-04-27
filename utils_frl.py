import torch
import torch.nn as nn
import torch.nn.functional as F
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


def pgd_attack_frl(model,
                  X,
                  y,
                  weight,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)

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
        #eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        eta = torch.min(torch.max(X_pgd - X.data, -1.0 * new_eps), new_eps)
        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd

def in_class(predict, label):

    probs = torch.zeros(10)
    for i in range(10):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs

def match_weight(label, diff0, diff1, diff2):

    weight0 = torch.zeros(label.shape[0], device='cuda')
    weight1 = torch.zeros(label.shape[0], device='cuda')
    weight2 = torch.zeros(label.shape[0], device='cuda')

    for i in range(10):
        weight0 += diff0[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight1 += diff1[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight2 += diff2[i] * torch.tensor(label == i, dtype= torch.float).cuda()

    return weight0, weight1, weight2



def cost_sensitive(lam0, lam1, lam2):

    ll0 = torch.clone(lam0)
    ll1 = torch.clone(lam1)

    diff0 = torch.ones(10) * 1 / 10
    for i in range(10):
        for j in range(10):
            if j == i:
                diff0[i] = diff0[i] + 9 / 10 * ll0[i]
            else:
                diff0[i] = diff0[i] - 1 / 10 * ll0[j]

    diff1 = torch.ones(10) * 1/ 10
    for i in range(10):
        for j in range(10):
            if j == i:
                diff1[i] = diff1[i] + 9 / 10 * ll1[i]
            else:
                diff1[i] = diff1[i] - 1 / 10 * ll1[j]

    diff2 = torch.clamp(torch.exp(2 * lam2), min = 0.98, max = 2.5)

    return diff0, diff1, diff2


def evaluate(model, test_loader, configs, device, mode = 'Test'):

    print('Doing evaluation mode ' + mode)
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
        correct += add
        model.zero_grad()
        all_pred.append(pred)

        ## adv test
        x_adv = pgd_attack(model, X = data, y = target, **configs)
        output1 = model(x_adv)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv += add1
        all_pred_adv.append(pred1)

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)

    total_clean_error = 1- correct / len(test_loader.dataset)
    total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error





def trades_adv(model,
               x_natural,
               weight,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    # define KL-loss
    new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)

    criterion_kl = nn.KLDivLoss(size_average = False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - new_eps), x_natural + new_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv




def train_model(model, train_loader, optimizer, diff0, diff1, diff2, epoch, beta, configs, device):

    criterion_kl = nn.KLDivLoss(reduction='none')
    criterion_nat = nn.CrossEntropyLoss(reduction='none')

    print('Doing Training on epoch:  ' + str(epoch))

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)

        weight0, weight1, weight2 = match_weight(target, diff0, diff1, diff2)
        ## generate adv examples
        x_adv = trades_adv(model, x_natural = data, weight = weight2, **configs)

        model.train()
        ## clear grads
        optimizer.zero_grad()

        ## get loss
        loss_natural = criterion_nat(model(data), target)
        loss_bndy_vec = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))
        loss_bndy = torch.sum(loss_bndy_vec, 1)

        ## merge loss
        loss = torch.sum(loss_natural * weight0)/ torch.sum(weight0)\
               + beta * torch.sum(loss_bndy * weight1) / torch.sum(weight1)        ## back propagates
        loss.backward()
        optimizer.step()

        ## clear grads
        optimizer.zero_grad()



def frl_train(h_net, ds_train, ds_valid, optimizer, now_epoch, configs, configs1, device, delta0, delta1, rate1, rate2, lmbda, beta, lim):
    print('train epoch ' + str(now_epoch), flush=True)

    ## given model, get the validation performance and gamma
    class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
        evaluate(h_net, ds_valid, configs1, device, mode='Validation')

    ## get gamma on validation set
    gamma0 = class_clean_error - total_clean_error - delta0
    gamma1 = class_bndy_error - total_bndy_error - delta1

    ## print inequality results
    print('total clean error ' + str(total_clean_error))
    print('total boundary error ' + str(total_bndy_error))

    print('.............')
    print('each class inequality constraints')
    print(gamma0)
    print(gamma1)

    #################################################### do training on now epoch
    ## constraints coefficients
    lmbda0 = lmbda[0:10] + rate1 * torch.clamp(gamma0, min = -1000)      ## update langeragian multiplier
    lmbda1 = lmbda[10:20] + rate1 * 2 * torch.clamp(gamma1, min = -1000)      ## update langeragian multiplier
    lmbda2 = lmbda[20:30] #+ rate2 * gamma1

    lmbda0 = normalize_lambda(lmbda0, lim)
    lmbda1 = normalize_lambda(lmbda1, lim)   ## normalize back to the simplex

    ## given langerangian multipliers, get each class's weight
    lmbda = torch.cat([lmbda0, lmbda1, lmbda2])
    diff0, diff1, diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2)

    print('..............................')
    print('current lambda after update')
    print(lmbda0)
    print(lmbda1)
    print(lmbda2)

    print('..............................')
    print('current weight')
    print(diff0)
    print(diff1)
    print(diff2)
    print('..............................')
    ## do the model parameter update based on gamma
    _ = train_model(h_net, ds_train, optimizer, diff0, diff1, diff2, now_epoch,
                    beta, configs, device)

    return lmbda

def normalize_lambda(lmb, lim = 0.8):

    lmb = torch.clamp(lmb, min=0)
    if torch.sum(lmb) > lim:
        lmb = lim * lmb / torch.sum(lmb)
    else:
        lmb = lmb
    return lmb
