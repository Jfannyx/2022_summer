#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 11:04
# @File     : train.py

"""
import argparse

import torch
import torchvision

from .lenet_model.lenet import LeNet, LeNet3
from .utils import pre_process

# from app01.lossplt import loss_plot


def get_data_loader(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    return train_loader, test_loader


def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))
    return 'Test Accuracy of the model is: {} %'.format(100 * correct / total)


def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def lenet_train(epochs, batch_size, learning_rate, num_classes, task_name, structure, optimizer_name):

    logger=open('log/'+task_name+'.txt',mode='a',encoding="utf8")
    logger.close()
    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if structure == "LeNet":
        model = LeNet(num_classes).to(device)
    elif structure == "LeNet with an added layer":
        model = LeNet3(num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_name == "Adam": 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # start train
    total_step = len(train_loader)
    loss_list = [] #
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):

            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                step_info=('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                logger=open('log/'+task_name+'.txt',mode='a',encoding="utf8")
                logger.write(step_info+"\n")
                logger.close()
            loss_list.append(loss.item()) # store loss of each step
        # evaluate after epoch train
        step_info=(evaluate(model, test_loader, device))
        logger=open('log/'+task_name+'.txt',mode='w',encoding="utf8")    
        logger.write(step_info+"\n")
        logger.close()
    loss0 = torch.tensor(loss_list)
    torch.save(loss0, "static/result/"+task_name) # save loss
    # save the trained model
    save_model(model, save_path='lenet.pth')
    return model
