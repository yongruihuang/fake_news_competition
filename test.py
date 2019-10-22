from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from senet.se_resnet import se_resnet50
import logging
import sys
import pandas as pd
from tqdm import tqdm
from dataset.data import RumorNewsDataset
from torch.utils.data import DataLoader


def get_logger(name, file_name='./', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not os.path.exists(file_name):
        os.makedirs(file_name)

    file_handler = logging.FileHandler(file_name+'model.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_brats_train_loaders(dataset, test_path):
    """
    Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset.

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """

    # get train and validation files
    train_path = dataset
    val_path = dataset
    test_path = test_path

    print(f'Loading training set from: {train_path}...')
    train_datasets = RumorNewsDataset(train_path)

    print(f'Loading validation set from: {val_path}...')
    val_datasets = RumorNewsDataset(val_path, mode='validation')

    print(f'Loading test set from: {test_path}...')
    test_datasets = RumorNewsDataset(test_path, mode='test', hasMasks=False)

    num_workers = 1
    print(f'Number of workers for train/val datasets: {num_workers}')

    return {
        'train': DataLoader(train_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
        'test': DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
    }


def make_csv(args, model1, model2, dataset_sizes):
    val_list = []
    model1.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        for i, t in enumerate(tqdm(dataloders['val'])):
            inputs, pid, labels, pixels, Hash = t
            inputs = inputs.to(device=0, dtype=torch.float)
            labels = labels.to(device=0, dtype=torch.int64)
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs, feature = model1(inputs)
            # _, category = torch.max(category.data, 1)
            _, preds = torch.max(outputs.data, 1)
            loss = F.cross_entropy(outputs, labels)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == labels.data).float()

            list = [pid[0], outputs[0, 0].item(), outputs[0, 1].item(), pixels.item(), int(Hash[0])]

            val_list.append(list)

        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects / dataset_sizes['val']

        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'val', epoch_loss, epoch_acc))

    model2.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        for i, t in enumerate(tqdm(dataloders['train'])):
            inputs, pid, labels, pixels, Hash = t
            inputs = inputs.to(device=0, dtype=torch.float)
            labels = labels.to(device=0, dtype=torch.int64)
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs, feature = model2(inputs)
            # _, category = torch.max(category.data, 1)
            _, preds = torch.max(outputs.data, 1)
            loss = F.cross_entropy(outputs, labels)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == labels.data).float()

            list = [pid[0], outputs[0, 0].item(), outputs[0, 1].item(), pixels.item(), int(Hash[0])]

            val_list.append(list)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']

        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'val', epoch_loss, epoch_acc))

    for i, t in enumerate(tqdm(dataloders['test'])):
        inputs, pid, pixels, Hash = t
        inputs = inputs.to(device=0, dtype=torch.float)

        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        outputs1, feature1 = model1(inputs)
        outputs2, feature2 = model2(inputs)

        outputs = (outputs1+outputs2)/2
        _, preds = torch.max(outputs.data, 1)

        list = [pid[0], outputs[0, 0].item(), outputs[0, 1].item(), pixels.item(), int(Hash[0])]
        val_list.append(list)

    name = ['pid', 'true', 'rumor', 'pixels', 'Hash']
    logger.info(f'make .csv file {args.save_path}/all_pred.csv')
    test = pd.DataFrame(columns=name, data=val_list)
    test.to_csv(args.save_path + '/all_pred.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--resume1', type=str,
                        default="/home/server/github/senet.pytorch/output/best_checkpoint.pytorch",
                        help="For training from one checkpoint")
    parser.add_argument('--resume2', type=str,
                        default="/home/server/github/senet.pytorch/output/best_checkpoint1.pytorch",
                        help="For training from one checkpoint")
    parser.add_argument('--data-path', type=str, default='/home/server/data/task3/data_train_val_2_fold_size_224_224_3.hdf5')
    parser.add_argument('--test-path', type=str, default='/home/server/data/task3/data_test2_size_224_224_3.hdf5')
    parser.add_argument('--save-path', type=str, default="/home/server/github/senet.pytorch/output")
    parser.add_argument('--num-class', type=int, default=2)
    args = parser.parse_args()

    # creative logger
    logger = get_logger('SENetTesting', file_name='output')

    # read data
    path = args.data_path
    test_path = args.test_path
    dataloders = get_brats_train_loaders(path, test_path)
    dataset_sizes = {'train': 16736, 'val': 16736, 'test': 3791}

    # define loss function
    criterion = nn.CrossEntropyLoss()

    use_gpu = torch.cuda.is_available()
    logger.info("use_gpu:{}".format(use_gpu))

    model1 = se_resnet50(num_classes=args.num_class)
    model2 = se_resnet50(num_classes=args.num_class)
    if os.path.isfile(args.resume1):
        logger.info(("=> loading checkpoint '{}'".format(args.resume1)))
        state = torch.load(args.resume1)
        try:
            model1.load_state_dict(state['model_state_dict'])
        except BaseException as e:
            print('Failed to do something: ' + str(e))
    else:
        logger.info(("=> no checkpoint found at '{}'".format(args.resume1)))

    if os.path.isfile(args.resume2):
        logger.info(("=> loading checkpoint '{}'".format(args.resume2)))
        state = torch.load(args.resume2)
        try:
            model2.load_state_dict(state['model_state_dict'])
        except BaseException as e:
            print('Failed to do something: ' + str(e))
    else:
        logger.info(("=> no checkpoint found at '{}'".format(args.resume2)))

    if use_gpu:
        model1 = model1.cuda()
        model2 = model2.cuda()

    make_csv(args, model1, model2, dataset_sizes=dataset_sizes)