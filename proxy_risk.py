import argparse
import random
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from utils import *
from model import Model
from test import test, test_disagreement, test_divergence


def sample_batch(data_iter, source):
    img, label = data_iter.next()
    # domain labels
    batch_size = len(label)
    if source:
        domain_label = torch.zeros(batch_size).long()
    else:
        domain_label = torch.ones(batch_size).long()
    return img.cuda(), label.cuda(), domain_label.cuda()

def train(model, model2, dataloader_source, dataloader_target, max_epoch, lam):
    loss_class = torch.nn.NLLLoss().cuda()
    loss_domain = torch.nn.NLLLoss().cuda()
    optimizer = optim.Adam(model2.parameters(), lr=1e-4)

    model.eval()
    model2.eval()

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < (len_dataloader):
        # source
        s_img, s_label, domain_label = sample_batch(data_source_iter, source=True)
        class_output, _, domain_output = model2(s_img, alpha=0.1)
        loss_s_domain = loss_domain(domain_output, domain_label)
        loss_s_label = loss_class(class_output, s_label)

        # target
        t_img, _, domain_label = sample_batch(data_target_iter, source=False)
        _, output, domain_output = model2(t_img, alpha=0.1)
        loss_t_domain = loss_domain(domain_output, domain_label)

        # maximize the disagreement
        _, output_, _ = model(t_img)
        disagree = torch.norm(output - output_, p=2, dim=1).mean()

        # Lagrangian relaxation
        loss = - disagree + lam * (loss_s_label + loss_s_domain + loss_t_domain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate proxy risk with domain-invariant check model')
    parser.add_argument('--model_path', default=None, type=str, help='Path to the candidate model')
    parser.add_argument('--check_model_path', default=None, type=str, help='Path to the pretrained check model')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--nepoch', default=20, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--eps', default=0.14, type=int, help='Parameter for domain-invariant constrain')
    parser.add_argument('--lam', default=50, type=int, help='Tradeoff parameter for maximizing disgreement')

    # args parse
    args = parser.parse_args()
    model_path, check_model_path = args.model_path, args.check_model_path
    batch_size, nepoch, eps, lam = args.batch_size, args.nepoch, args.eps, args.lam

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

    # any candidate model 
    model = torch.load(model_path).cuda()
    acc_t = test(model, dataloader_target_test)
    print('True target risk: %4f' % (1 - acc_t))

    # load pretrained check model
    model2 = torch.load('checkpoints/model_check.pth').cuda()

    # estimate proxy risk
    max_proxy_risk = 0
    for epoch in range(nepoch):
        # maximize disagreement
        train(model, model2, dataloader_source, dataloader_target, nepoch, lam)

        # check domain-invariant constrain
        acc_s = test(model2, dataloader_source_test)
        loss_s_domain = test_divergence(model2, dataloader_source_test, source=True)
        loss_t_domain = test_divergence(model2, dataloader_target_test, source=False)

        disagree = test_disagreement(model, model2, dataloader_target_test)
        dir_loss = (1. - acc_s) + 0.1*(loss_s_domain + loss_t_domain)
        if dir_loss <= eps and max_proxy_risk < disagree:
           max_proxy_risk = disagree

        print('EPOCH {} Proxy risk: {:.4f} DIR loss: {:.4f}'.format(epoch, disagree, dir_loss))
    print('Estimated proxy risk: %4f' % (max_proxy_risk))
    print('Estimation Error: %4f' % (abs(max_proxy_risk - (1 - acc_t))))
