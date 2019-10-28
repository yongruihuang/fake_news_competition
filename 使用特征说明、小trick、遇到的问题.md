# 背景
使用新闻文本、新闻图片和用户画像来预测是否是假新闻

# 文件存放位置
- 数据：submit/data/task3
- 主代码：submit/code/main_submit.ipynb

# 特征
## 图像
- 使用Snet end2end训练出来的真、假新闻分数
- 图像的宽、高和像素个数（多个取平均）
- feature_len_piclist：图像数量

## 新闻文本
- 新闻包含地点名称：包括jieba默认提取出来的、省粒度
- TF-IDF特征SVD降维成100
- LDA特征20

## 用户画像
- userFollowCount：关注数量
- userFansCount：粉丝数量
- feature_follow_fan_rate：关注数量/粉丝数量
- feature_userLocation：用户所在地点，根据所有userLocation编码
- feature_userProvince：用户所在地点，省粒度
- feature_len_userDescription：用户个性签名长度
- 个性签名TF-IDF特征SVD降维成100
- 个性签名LDA特征10

## 混合特征
- feature_pic_text_rate：图像长度/新闻长度

# 一些trick
- 相同的新闻（不同用户转发）同时出现在测试集和训练集中，先把这些新闻找出来（去除标点符号后前5个字符相同认为是一个新闻），从训练集中挑走，保存结果，最后测试的时候遇到这些新闻直接查表
- 模型融合，不同的参数lgb模型stacking在一起

# 遇到问题
- 训练集挑出测试集中出现的新闻后发现训练集和测试集样本分布不一样了，真的新闻太多了，解决办法：阈值调低一点，从0.5变0.2
- bert特征没有用，复赛一开始，用Bert之类预训练模型搞出来的特征只有80分左右。结合用户画像一起的话直接过拟合训练集。因为训练集中太多重复新闻了。调掉数量又太少了。
- 其他分类器，除了Lgb之外，居然效果不好，而且差的很多？？这个要分析一下原因。