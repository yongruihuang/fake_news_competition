import h5py
import os
import numpy as np
from random import shuffle
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import string
import cv2
import gc
import dhash
import argparse

MAX_WRITE_BUFFER = 5
alpha_dic = {ch: n for n, ch in enumerate(string.ascii_uppercase)}

def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, pids_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    print('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr
    hdf5_data['pids_%s' % train_test][counter_from:counter_to, ...] = pids_list[train_test]


def _release_tmp_memory(img_list, mask_list, pids_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    pids_list[train_test].clear()
    gc.collect()


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def test_train_val_split(patient_id):

    if patient_id % 10 >= 8:
        return 'train'
    elif patient_id % 10 >= 6:
        return 'validation'
    else:
        return 'train'


def prepare_test_data(input_folder, output_file):
    hdf5_file = h5py.File(output_file, "w")

    print('Counting files and parsing meta data...')

    dir = input_folder
    for root, dirs, files in os.walk(dir):
        pids = files
        images = [*map(lambda x: dir + '/' + x, files)]

    train_shape = (len(images), 224, 224, 3)

    hdf5_file.create_dataset("images_test", train_shape, np.int8)

    hdf5_file.create_dataset("pids_test", [len(pids)], dtype=h5py.special_dtype(vlen=str))
    hdf5_file['pids_test'][...] = pids

    hdf5_file.create_dataset("pixels_test", [len(pids)], dtype=np.int64)
    hdf5_file.create_dataset('Hash_test', [len(pids)], dtype=h5py.special_dtype(vlen=str))

    # loop over train addresses
    for i, addr in enumerate(images):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(images)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(addr)
        Hash_image = Image.open(addr)
        Hash_image = Hash_image.convert('L').resize((9, 9), Image.ANTIALIAS)
        Hash_valu = dhash.dhash_int(Hash_image)
        hdf5_file['Hash_test'][i, ...] = Hash_valu
        try:
            image_size = img.size
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hdf5_file["images_test"][i, ...] = img
            hdf5_file['pixels_test'][i, ...] = image_size

        except:
            print(addr)
            os.remove(addr)

    hdf5_file.close()
    print('finished')


def prepare_data(input_folder, output_file, size):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    hdf5_file = h5py.File(output_file, "w")

    print('Counting files and parsing meta data...')

    dir = input_folder + '/' + 'truth_pic'
    for root, dirs, files in os.walk(dir):
        truth_pids = files
        truth = [*map(lambda x:dir+'/'+x, files)]
        truth_label = [0 for _ in range(len(truth_pids))]

    dir = input_folder + '/' + 'rumor_pic'
    for root, dirs, files in os.walk(dir):
        rumor_pids = files
        rumor = [*map(lambda x: dir + '/' + x, files)]
        rumor_label = [1 for _ in range(len(rumor_pids))]

    addrs = truth + rumor
    labels = truth_label + rumor_label
    pids = truth_pids + rumor_pids

    # to shuffle data
    shuffle_data = True
    if shuffle_data:
        c = list(zip(addrs, labels, pids))
        shuffle(c)
        addrs, labels, pids = zip(*c)

    # Divide the hata into 50% train, 50% validation, and 20% test
    train_addrs = addrs[0:int(0.5 * len(addrs))]
    train_labels = labels[0:int(0.5 * len(labels))]
    train_pids = pids[0:int(0.5 * len(labels))]

    val_addrs = addrs[int(0.5 * len(addrs)):]
    val_labels = labels[int(0.5 * len(addrs)):]
    val_pids = pids[int(0.5 * len(addrs)):]

    # train_addrs2 = addrs[0:int(0.6 * len(addrs))] + addrs[int(0.8 * len(addrs)):]
    # train_labels2 = labels[0:int(0.6 * len(labels))] + labels[int(0.8 * len(addrs)):]
    # train_pids2 = pids[0:int(0.6 * len(labels))] + pids[int(0.8 * len(addrs)):]
    #
    # val_addrs2 = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
    # val_labels2 = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
    # val_pids2 = pids[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
    #
    # train_addrs3 = addrs[0:int(0.4 * len(addrs))] + addrs[int(0.6 * len(addrs)):]
    # train_labels3 = labels[0:int(0.4 * len(labels))] + labels[int(0.6 * len(addrs)):]
    # train_pids3 = pids[0:int(0.4 * len(labels))] + pids[int(0.6 * len(addrs)):]
    #
    # val_addrs3 = addrs[int(0.4 * len(addrs)):int(0.6 * len(addrs))]
    # val_labels3 = labels[int(0.4 * len(addrs)):int(0.6 * len(addrs))]
    # val_pids3 = pids[int(0.4 * len(addrs)):int(0.6 * len(addrs))]
    #
    # train_addrs4 = addrs[0:int(0.2 * len(addrs))] + addrs[int(0.4 * len(addrs)):]
    # train_labels4 = labels[0:int(0.2 * len(labels))] + labels[int(0.4 * len(addrs)):]
    # train_pids4 = pids[0:int(0.2 * len(labels))] + pids[int(0.4 * len(addrs)):]
    #
    # val_addrs4 = addrs[int(0.2 * len(addrs)):int(0.4 * len(addrs))]
    # val_labels4 = labels[int(0.2 * len(addrs)):int(0.4 * len(addrs))]
    # val_pids4 = pids[int(0.2 * len(addrs)):int(0.4 * len(addrs))]
    #
    # train_addrs5 = addrs[0:int(0.2 * len(addrs))] + addrs[int(0.4 * len(addrs)):]
    # train_labels5 = labels[0:int(0.2 * len(labels))] + labels[int(0.4 * len(addrs)):]
    # train_pids5 = pids[0:int(0.2 * len(labels))] + pids[int(0.4 * len(addrs)):]
    #
    # val_addrs5 = addrs[int(0.2 * len(addrs)):int(0.4 * len(addrs))]
    # val_labels5 = labels[int(0.2 * len(addrs)):int(0.4 * len(addrs))]
    # val_pids5 = pids[int(0.2 * len(addrs)):int(0.4 * len(addrs))]

    train_shape = (len(train_addrs), 224, 224, 3)
    val_shape = (len(val_addrs), 224, 224, 3)

    print('Debug: Check if sets add up to correct value:')

    hdf5_file.create_dataset("images_train", train_shape, np.int8)
    hdf5_file.create_dataset("images_validation", val_shape, np.int8)

    hdf5_file.create_dataset("labels_train", (len(train_addrs),), np.int8)
    hdf5_file["labels_train"][...] = train_labels
    hdf5_file.create_dataset("labels_validation", (len(val_addrs),), np.int8)
    hdf5_file["labels_validation"][...] = val_labels

    hdf5_file.create_dataset("pids_train", [len(train_pids)], dtype=h5py.special_dtype(vlen=str))
    hdf5_file['pids_train'][...] = train_pids
    hdf5_file.create_dataset("pids_validation", [len(val_pids)], dtype=h5py.special_dtype(vlen=str))
    hdf5_file['pids_validation'][...] = val_pids

    hdf5_file.create_dataset('pixels_train', [len(train_pids)], dtype=np.int64)
    hdf5_file.create_dataset('pixels_validation', [len(val_pids)], dtype=np.int64)

    hdf5_file.create_dataset('Hash_train', [len(train_pids)], dtype=h5py.special_dtype(vlen=str))
    hdf5_file.create_dataset('Hash_validation', [len(val_pids)], dtype=h5py.special_dtype(vlen=str))

    # loop over train addresses
    for i, addr in enumerate(train_addrs):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(train_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB

        img = cv2.imread(addr)
        Hash_image = Image.open(addr)
        Hash_image = Hash_image.convert('L').resize((9, 9), Image.ANTIALIAS)
        Hash_valu = dhash.dhash_int(Hash_image)
        hdf5_file['Hash_train'][i, ...] = Hash_valu
        try:
            img_size = img.size
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hdf5_file["images_train"][i, ...] = img
            hdf5_file['pixels_train'][i, ...] = img_size
        except:
            print(addr)
            os.remove(addr)

        # add any image pre-processing here

        # save the image and calculate the mean so far
        # img.shape := (224, 224, 3)

    # loop over validation addresses
    for i, addr in enumerate(val_addrs):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print('Validation data: {}/{}'.format(i, len(val_addrs)))

        # read an image and resize to (224, 224)
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(addr)
        Hash_image = Image.open(addr)
        Hash_image = Hash_image.convert('L').resize((9, 9), Image.ANTIALIAS)
        Hash_valu = dhash.dhash_int(Hash_image)
        hdf5_file['Hash_validation'][i, ...] = Hash_valu
        try:
            img_size = img.size
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hdf5_file["images_validation"][i, ...] = img
            hdf5_file['pixels_validation'][i, ...] = img_size

        except:
            print(addr)
            os.remove(addr)

    # save the mean and close the hdf5 file

    hdf5_file.close()
    print('finished')

def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size, mode,
                                force_overwrite=False):

    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])

    data_file_name = 'data_train_val_2_fold_size_%s.hdf5' % (size_str)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)
    makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        print('This configuration of mode, size and target resolution has not yet been preprocessed')
        print('Preprocessing now!')
        if mode == 'train':
            prepare_data(input_folder, data_file_path, size)
        else:
            data_file_name = 'data_test2_size_%s.hdf5' % (size_str)
            data_file_path = os.path.join(preprocessing_folder, data_file_name)
            prepare_test_data(input_folder, data_file_path)
    else:
        print('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--train_data_folder', type=str, default="/home/server/data/task3/train")
    parser.add_argument('--test_data_folder', type=str, default="/home/server/data/task3/task3/task3_new_stage2_pic")
    parser.add_argument('--preprocessing_folder', type=str, default="/home/server/data/task3/")

    args = parser.parse_args()
    d = load_and_maybe_process_data(args.train_data_folder, args.preprocessing_folder, (224, 224, 3), mode='train', force_overwrite=True)
    a = load_and_maybe_process_data(args.test_data_folder, args.preprocessing_folder,  (224, 224, 3), mode='test', force_overwrite=True)
