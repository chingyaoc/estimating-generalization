import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from utils import *
from model import Model
from test import test


def train(model, dataloader_source, dataloader_target, epoch, nepoch):
    model.train()

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:
        # progressive
        p = float(i + epoch * len_dataloader) / nepoch / len_dataloader
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * 0.1

        # source
        s_img, s_label, domain_label = sample_batch(data_source_iter, True)

        class_output, _, domain_output = model(s_img, alpha=alpha)
        loss_s_domain = loss_domain(domain_output, domain_label)
        loss_s_label = loss_class(class_output, s_label)

        # target
        t_img, _, domain_label = sample_batch(data_target_iter, False)
        _, _, domain_output = model(t_img, alpha=alpha)
        loss_t_domain = loss_domain(domain_output, domain_label)

        # domain-invariant loss
        loss = loss_t_domain + loss_s_domain + loss_s_label
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain domain-invariant check model')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    batch_size, nepoch = args.batch_size, args.nepoch

    # MNIST
    dataset_source = datasets.MNIST(root='dataset/mnist', train=True, transform=img_transform_source, download=True)
    dataloader_source = DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=2)
    dataset_source_test = datasets.MNIST(root='dataset/mnist', train=False, transform=img_transform_source, download=True)
    dataloader_source_test = DataLoader(dataset=dataset_source_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # MNIST-M
    train_list = os.path.join('dataset/mnist_m/mnist_m_train_labels.txt')
    test_list = os.path.join('dataset/mnist_m/mnist_m_test_labels.txt')

    dataset_target = GetLoader(data_root='dataset/mnist_m/mnist_m_train', data_list=train_list, transform=img_transform_target)
    dataloader_target = DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=2)
    dataset_target_test =  GetLoader(data_root='dataset/mnist_m/mnist_m_test', data_list=test_list, transform=img_transform_target)
    dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle= False, num_workers=2)

    # Model Setup
    model = Model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_class = torch.nn.NLLLoss().cuda()
    loss_domain = torch.nn.NLLLoss().cuda()

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    for epoch in range(nepoch):
        train(model, dataloader_source, dataloader_target, epoch, nepoch)
        acc_s = test(model, dataloader_source_test)
        acc_t = test(model, dataloader_target_test)
        print('EPOCH {} Acc: MNIST {:.2f}% MNIST-M {:.2f}%'.format(epoch, acc_s*100, acc_t*100))

    torch.save(model, 'checkpoints/model_check.pth')
