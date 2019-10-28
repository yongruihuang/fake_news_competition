# fake_new_competition
假新闻预测竞赛

## 说明
- 所依赖的深度学习框架：pytorch1.0
- CUDA Version: 9.0.252
- CUDNN Version: 7.0

## 图像部分

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

## 总体
- 我们的主体代码是main_submit.ipynb这个文件
- 请使用anoconda的jupyter notebook运行主代码，jupyter notebook使用参考：https://jupyter.org/documentation
- 我们主代码使用Jupyter notebook实现，可通过main_submit. Ipynb查看运行过程，并且可以使用Jupyter notebook直接查看和复现结果，这部分无需提供test.sh和train.sh代码生成训练结果和测试结果。
- 图像的模型提供出一张图片属于假新闻图片的分数，作为特征。
- 文本没有单独的模型，提取特征后跟其他特征一起训练，其中图像特征为3所述图像模型的结果
- 至于总的模型，我们的模型首先是由5 * 20个GDBT（梯度提升决策树）取得平均，再用这5个输出集成一个GDBT得到的。因此有101个模型。

