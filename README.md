# fake_new_competition
假新闻预测竞赛

文本fine-tining方法参考：https://github.com/PaddlePaddle/ERNIE/blob/develop/README.zh.md#%E5%8D%95%E5%8F%A5%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1
构造数据集后，直接运行脚本



首先对数据进行预处理：

```sh
python hdf5.py --train_data_folder='/path_to/task3/train' --test_data_floder='/path_to/task3/task3_new_stage2_pic' --preprocessing_folder='/path_to/task3'
```

- --train_data_folder：原始训练集图片路径
- --test_data_floder：原始测试集图片路径
- --preprocessing_floder：预处理后数据存储路径



test.sh:

```sh
python test.py --resume1='/path_to/checkpoint1.pytorch' --resume2='/path_to/checkpoint2.pytorch' --data-path='/path_to/data_train_val_2_fold_size_224_224_3.hdf5' --test-path='/path_to/data_test2_size_224_224_3.hdf5' --save-path='./output'
```

- --resume1/resume2：把训练集平分成两部分，并分别训练的两个checkpoints的存放路径
- --save_path：CSV文件的保存路径
- --data_path：预处理后训练集路径
- --test-path：预处理后测试集路径

生成训练集和测试集每张图片的2-fold真新闻和假新闻的置信度，以及图像的分辨率、Hash值作为图片的统计特征，保存在`all_pred.csv`文件中。