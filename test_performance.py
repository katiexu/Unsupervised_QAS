import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
# from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders, MOSIDataLoaders
from FusionModel import QNet
# from FusionModel import translator
import os
import csv
import json

from Arguments import Arguments
import random


def display(metrics):
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

    print(YELLOW + "\nTest Accuracy: {}".format(metrics) + RESET)


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor()
    output_all = torch.Tensor()
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)
            output = model(images)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
            output_all = torch.cat((output_all, output), dim=0)
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return total_loss, accuracy


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}

    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)
            output = model(images)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    metrics = accuracy
    return metrics


def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    args = Arguments(task)
    if epochs == None:
        epochs = args.epochs

    if task == 'MOSI':
        dataloader = MOSIDataLoaders(args)
    else:
        dataloader = MNISTDataLoaders(args, task)

    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    if weight != 'init':
        if weight != 'base':
            model.load_state_dict(weight, strict= False)
        else:
            model.load_state_dict(torch.load('weights/base_fashion'))
            # model.load_state_dict(torch.load('weights/mnist_best_3'))
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 0

    start = time.time()
    for epoch in range(epochs):
        try:
            train(model, train_loader, optimizer, criterion, args)
        except Exception as e:
            print('No parameter gate exists')
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = evaluate(model, val_loader, args)
        val_loss_list.append(val_loss)
        metrics = evaluate(model, test_loader, args)
        val_loss = 0.5 *(val_loss + train_loss[-1])
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics, 'saving model')
            best_model = copy.deepcopy(model)
        else:
            if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics)
        if not os.path.isfile('results.csv'):
            with open('results.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['design', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'test_acc'])
        new_row = [design, epoch, train_loss[0], train_loss[1], val_loss_list[-1], metrics]
        with open('results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

    end = time.time()
    # best_model = model
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    print("Running time: %s seconds" % (end - start))
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'mae': metrics}

    if save:
        torch.save(best_model.state_dict(), 'weights/init_weight')
    return best_model, report


if __name__ == '__main__':
    with open('models/circuits_with_noises.json', 'r') as file:
        circuits_with_noise = json.load(file)

    for i in range(len(circuits_with_noise)):
        design = circuits_with_noise[i]
        best_model, report = Scheme(design, 'MNIST', 'init', 15)

