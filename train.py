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
from senet.baseline import resnet20
from senet.se_resnet import se_resnet50
from dataset.data import get_brats_train_loaders
import shutil
import logging
import sys
import pandas as pd
from tqdm import tqdm
from torchvision import models


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path)
    try:
        model.load_state_dict(state['model_state_dict'])
    except BaseException as e:
        print('Failed to do something: ' + str(e))

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_logger(name, file_name='./', level=logging.INFO):
    # logging.basicConfig(filename=file_name+'model.log', level=level)
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


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, f'epoch{state["epoch"]}_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def _split_training_batch(t):
    def _move_to_device(input):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([_move_to_device(input[0]), input[1], _move_to_device(input[2])])
        else:
            return input.to(0, dtype=torch.float)

    t = _move_to_device(t)
    if len(t) == 2:
        input, target = t
        return input, target
    elif len(t) == 3:
        input, pid, target = t
        return input, pid, target
    elif len(t) == 4:
        input, pid, target, graph_brain = t
        return input, pid, target, graph_brain

def validate_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    model.eval()
    val_list = []

    # imagenet_model = models.resnet101(pretrained=True)
    # imagenet_model = imagenet_model.cuda()

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
            outputs, feature = model(inputs)
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

        for i, t in enumerate(tqdm(dataloders['test'])):
            inputs, pid, pixels, Hash = t
            inputs = inputs.to(device=0, dtype=torch.float)

            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            # forward
            outputs, feature = model(inputs)
            # category = imagenet_model(inputs)
            # _, category = torch.max(category.data, 1)
            _, preds = torch.max(outputs.data, 1)

            list = [pid[0], outputs[0, 0].item(), outputs[0, 1].item(), pixels.item(), int(Hash[0])]
            val_list.append(list)

        name = ['pid', 'true', 'rumor', 'pixels', 'Hash']
        logger.info(f'make .csv file {args.save_path}/val_test2_pred.csv')
        test = pd.DataFrame(columns=name, data=val_list)
        test.to_csv(args.save_path + '/val_test2_pred.csv')


def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1, num_epochs):
        # Each epoch has a training and validation phase

        validate_model(args, model, criterion, optimizer_ft, scheduler, num_epochs, dataset_sizes)

        phase = 'train'
        if phase == 'train':
            if args.start_epoch > 0 and (not resumed):
                scheduler.step(args.start_epoch+1)
                resumed = True
            else:
                scheduler.step(epoch)
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        tic_batch = time.time()
        # Iterate over data.
        for i, t in enumerate(dataloders['train']):
            # wrap them in Variable
            inputs, pid, labels, pixels, Hash = t
            inputs = inputs.to(device=0, dtype=torch.float)
            labels = labels.to(device=0, dtype=torch.int64)
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs, feature = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = F.cross_entropy(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == labels.data).float()

            batch_loss = running_loss / ((i+1)*args.batch_size)
            batch_acc = running_corrects / ((i+1)*args.batch_size)

            if phase == 'train' and i % args.print_freq == 0:
                logger.info('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                      epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_lr()[0], phase, batch_loss, batch_acc, \
                    args.print_freq/(time.time()-tic_batch)))
                tic_batch = time.time()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        validate_model(args, model, criterion, optimizer_ft, scheduler, num_epochs, dataset_sizes)

        if (epoch+1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            # torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'device': 0,
                'max_num_epochs': num_epochs,
            }, True, checkpoint_dir=os.path.join(args.save_path),
                logger=logger)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="/home/server/github/senet.pytorch/output")
    parser.add_argument('--resume', type=str, default="/home/server/github/senet.pytorch/output/best_checkpoint.pytorch", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet50", help="")
    parser.add_argument("--reduction", type=int, default=16)
    args = parser.parse_args()

    # creative logger
    logger = get_logger('SENetTrainer', file_name='output')

    # read data
    path = '/home/server/data/task3/data_train_val_2_fold_size_224_224_3.hdf5'
    test_path = '/home/server/data/task3/data_test2_size_224_224_3.hdf5'
    dataloders = get_brats_train_loaders(path, test_path)
    # dataset_sizes = {'train': 26778, 'val': 6695}
    dataset_sizes = {'train': 16736, 'val': 16736, 'test': 4247}
    # dataset_sizes = {'train': 26777, 'val': 6695, 'test': 3791}
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    logger.info("use_gpu:{}".format(use_gpu))

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])

    if script_name == "se_resnet50":
        model = se_resnet50(num_classes=args.num_class)
    else:
        raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            state = torch.load(args.resume)
            try:
                model.load_state_dict(state['model_state_dict'])
            except BaseException as e:
                print('Failed to do something: ' + str(e))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    torch.multiprocessing.set_sharing_strategy('file_system')

    model = train_model(args=args,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=args.num_epochs,
                           dataset_sizes=dataset_sizes)