from __future__ import print_function

from utils_frl import *
from pre_resnet import PreActResNet18
from wide_resnet import WideResNet
from data_loader import get_cifar10_loader

import os
import argparse
import torch
import torch.optim as optim


def main(args):

    if args.model == 'PreResNet18':
        h_net = PreActResNet18().cuda()
    elif args.model == 'WRN28':
        h_net = WideResNet().cuda()

    h_net.load_state_dict(torch.load('pgd/hot_start.pt'))

    ds_train, ds_valid, ds_test = get_cifar10_loader(batch_size=args.batch_size)

    ## other layer optimizer
    optimizer = optim.SGD(h_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)

    ## attack parameters during test
    configs =  {
    'epsilon': 8/255,
    'num_steps': 10,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 20,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    ### main training loop
    maxepoch = args.epoch
    device = 'cuda'
    beta = args.beta
    rate1 = args.rate1
    rate2 = args.rate2
    lim = args.lim
    delta0 = args.bound0 * torch.ones(10)   ## fair constraints
    delta1 = args.bound1 * torch.ones(10)   ## fair constraints
    lmbda = torch.zeros(30)

    REPORT = []
    REPORT1 = []

    for now_epoch in range(1, maxepoch + 1):

        if now_epoch % 5 == 0:
            print('test on current epoch')
            class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
                evaluate(h_net, ds_test, configs1, device, mode='Test')

            results = np.concatenate((np.array([total_clean_error, total_bndy_error]),
                                      class_clean_error.numpy().flatten(), class_bndy_error.numpy().flatten()))
            REPORT.append(results)
            report = np.array(REPORT)
            np.savetxt('Report_frlrw_test_' + args.model + '_' + str(args.seed) + '_' + str(args.rate1) + str(args.rate2) + '_' + str(
                args.bound0) + '_' + str(args.bound1) + '_' + str(args.lim) + '.txt', report)

            print('test on current epoch on training set')
            class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
                evaluate(h_net, ds_train, configs1, device, mode='Test')

            results = np.concatenate((np.array([total_clean_error, total_bndy_error]),
                                      class_clean_error.numpy().flatten(), class_bndy_error.numpy().flatten()))
            REPORT1.append(results)
            report = np.array(REPORT1)
            np.savetxt('Report_frlrw_train_' + args.model + '_' + str(args.seed) + '_' + str(args.rate1) + str(args.rate2) + '_' + str(
                args.bound0) + '_' + str(args.bound1) + '_' + str(args.lim) + '.txt', report)

        if now_epoch % 40 == 0:
            rate1 = rate1 / 2

        lmbda = frl_train(h_net, ds_train, ds_valid, optimizer, now_epoch, configs,
                              configs1, device, delta0, delta1, rate1, rate2, lmbda, beta, lim)
        lr_scheduler.step(now_epoch)
        print('................................................................................')



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--beta', help='trade off parameter', type = float, default=1.5)
    argparser.add_argument('--model', help='model structure', default='PreResNet18')
    argparser.add_argument('--bound0', type=float, help='fair constraints for clean error', default=0.1)
    argparser.add_argument('--bound1', type=float, help='fair constraints for bndy error', default=0.1)
    argparser.add_argument('--rate1', type=float, help='hyper-par update rate', default=0.05)
    argparser.add_argument('--rate2', type=float, help='hyper-par update rate', default=0.1)
    argparser.add_argument('--lim', type=float, default=0.5)
    args = argparser.parse_args()

    main(args)
