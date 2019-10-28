import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import h5py


class RumorNewsDataset(torch.utils.data.Dataset):
    #mode must be trian, test or val
    def __init__(self, filePath, mode="train", hasMasks=True):
        super(RumorNewsDataset, self).__init__()
        self.filePath = filePath
        self.mode = mode
        self.file = None
        self.hasMasks = hasMasks

    def __getitem__(self, index):
        #lazily open file
        self.openFileIfNotOpen()

        #load from hdf5 file
        image = self.file["images_" + self.mode][index, ...]
        if self.hasMasks:
            labels = self.file["labels_" + self.mode][index, ...]
            labels_one_hot = np.zeros(2)
            labels_one_hot[labels] = 1

        # to tensor
        image = torch.from_numpy(image)
        image = np.transpose(image, (2, 0, 1))

        #get pid
        pid = self.file["pids_" + self.mode][index]

        # get pixel number
        pixels = self.file['pixels_' + self.mode][index]

        # get Hash
        Hash = self.file['Hash_' + self.mode][index]

        if self.hasMasks:
            return image, str(pid), labels, pixels, Hash
        else:
            return image, pid, pixels, Hash

    def __len__(self):
        #lazily open file
        self.openFileIfNotOpen()

        return self.file["images_" + self.mode].shape[0]

    def openFileIfNotOpen(self):
        if self.file == None:
            self.file = h5py.File(self.filePath, "r")

    def _toEvaluationOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 3], dtype=np.float32)
        out[:, :, :, 0] = (labels != 0)
        out[:, :, :, 1] = (labels != 0) * (labels != 2)
        out[:, :, :, 2] = (labels == 4)
        return out

    def _toOrignalCategoryOneHot(self, labels):
        shape = labels.shape
        out = np.zeros([shape[0], shape[1], shape[2], 5], dtype=np.float32)
        for i in range(5):
            out[:, :, :, i] = (labels == i)
        return out

    def _toOrdinal(self, labels):
        return np.argmax(labels, axis=3)


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
    # loss_file_num = 0
    # for i in train_ids:
    #     name = i + ".tfrecord.gzip"
    #     answer = search('/home/server/data/TFRecord/val', name)
    #     if answer == -1:
    #         print("查无此文件", name)
    #         loss_file_num += 1
    # print(f'loss file num is {loss_file_num}')

    print(f'Loading training set from: {train_path}...')
    # train_datasets = BraTSDataset(brats, train_ids, phase='train',
    #                               transformer_config=loaders_config['transformer'],
    #                               is_mixup=loaders_config['mixup'])
    train_datasets = RumorNewsDataset(train_path)

    print(f'Loading validation set from: {val_path}...')
    # brats = BraTS.DataSet(brats_root=data_paths[0], year=2019).train
    # val_datasets = BraTSDataset(brats, validation_ids, phase='val',
    #                             transformer_config=loaders_config['transformer'],
    #                             is_mixup=False)
    val_datasets = RumorNewsDataset(val_path, mode='validation')

    print(f'Loading test set from: {test_path}...')
    # brats = BraTS.DataSet(brats_root=data_paths[0], year=2019).train
    # val_datasets = BraTSDataset(brats, validation_ids, phase='val',
    #                             transformer_config=loaders_config['transformer'],
    #                             is_mixup=False)
    test_datasets = RumorNewsDataset(test_path, mode='test', hasMasks=False)

    num_workers = 1
    print(f'Number of workers for train/val datasets: {num_workers}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints

    return {
        'train': DataLoader(train_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
        'test': DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=num_workers),
    }

if __name__ == '__main__':
    path = '/home/server/data/task3/train/data_3D_size_480_480_3.hdf5'
    dataset = get_brats_train_loaders(path)
    # trainset = dataset['train']
    # for i, t in enumerate(trainset):
    #     image, pid, label = t
    #     b=1
    # a = 1