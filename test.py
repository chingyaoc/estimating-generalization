import os
import numpy as np
import torch

def test(model, dataloader):
    model.eval()
    n_correct, n_total = 0, 0
    for img, label in iter(dataloader):
        batch_size = len(label)
        img, label = img.cuda(), label.cuda()

        class_output, _, _ = model(img)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    acc = n_correct.double() / n_total
    return acc


def test_disagreement(model, model2, dataloader):
    model.eval()
    model2.eval()
    n_correct, n_total = 0, 0
    for img, label in iter(dataloader):
        batch_size = len(label)
        img, label = img.cuda(), label.cuda()

        _, output_1, _ = model(img)
        _, output_2, _ = model2(img)

        pred_1 = output_1.data.max(1, keepdim=True)[1]
        pred_2 = output_2.data.max(1, keepdim=True)[1]
        n_correct += pred_1.eq(pred_2.data.view_as(pred_1)).cpu().sum()
        n_total += batch_size

    agree = n_correct.double() * 1.0 / n_total
    return 1 - agree


def test_divergence(model, dataloader, source):
    model.eval()
    loss_domain = torch.nn.NLLLoss().cuda()
    loss_list = []
    for img, label in iter(dataloader):
        batch_size = len(label)
        if source:
            domain_label = torch.zeros(batch_size).long()
        else:
            domain_label = torch.ones(batch_size).long()
        img, domain_label = img.cuda(), domain_label.cuda()

        _, _, domain_output = model(img)
        loss = loss_domain(domain_output, domain_label)        
        loss_list.append(loss.detach().cpu().mean().numpy())

    return np.mean(loss_list)
